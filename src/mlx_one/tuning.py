"""Native LoRA layers, injection, and adapter loading for causal language models."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class LoRALinear(nn.Module):
    """Low-rank update over a frozen dense or quantized MLX linear layer."""

    def __init__(
        self,
        linear: Any,
        *,
        input_dims: int,
        output_dims: int,
        rank: int,
        scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.linear.freeze()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
        bound = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(-bound, bound, shape=(input_dims, rank))
        self.lora_b = mx.zeros((rank, output_dims))

    @classmethod
    def from_base(
        cls, linear: Any, *, rank: int, scale: float, dropout: float
    ) -> LoRALinear:
        output_dims, stored_input_dims = linear.weight.shape
        input_dims = (
            stored_input_dims * 32 // linear.bits
            if isinstance(linear, nn.QuantizedLinear)
            else stored_input_dims
        )
        return cls(
            linear,
            input_dims=input_dims,
            output_dims=output_dims,
            rank=rank,
            scale=scale,
            dropout=dropout,
        )

    def __call__(self, value: Any) -> Any:
        update = (self.dropout(value) @ self.lora_a) @ self.lora_b
        return self.linear(value) + (self.scale * update).astype(value.dtype)


def apply_lora(
    model: Any,
    *,
    num_layers: int,
    rank: int = 8,
    scale: float = 2.0,
    dropout: float = 0.0,
    target_modules: tuple[str, ...] | list[str] = (),
) -> tuple[str, ...]:
    """Freeze a model and inject LoRA into the selected final decoder layers."""
    if num_layers < 1 or rank < 1 or scale <= 0 or not 0 <= dropout < 1:
        raise ValueError("invalid LoRA layer count, rank, scale, or dropout")
    layers = _decoder_layers(model)
    if num_layers > len(layers):
        raise ValueError("LoRA layer count exceeds the decoder layer count")
    requested = tuple(target_modules)
    if any(not isinstance(name, str) or not name.strip() for name in requested):
        raise ValueError("LoRA target modules must be non-empty strings")
    defaults = {"q_proj", "k_proj", "v_proj", "o_proj"}
    selected = set(requested) if requested else defaults
    model.freeze()
    replaced: list[str] = []
    start = len(layers) - num_layers
    for layer_index, layer in enumerate(layers[start:], start=start):
        updates = []
        for path, module in layer.named_modules():
            leaf = path.rsplit(".", 1)[-1]
            if path not in selected and leaf not in selected:
                continue
            if not isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                raise ValueError(f"LoRA target {path!r} is not a linear layer")
            updates.append(
                (
                    path,
                    LoRALinear.from_base(
                        module,
                        rank=rank,
                        scale=scale,
                        dropout=dropout,
                    ),
                )
            )
            replaced.append(f"layers.{layer_index}.{path}")
        if updates:
            layer.update_modules(tree_unflatten(updates))
    if not replaced:
        choices = ", ".join(sorted(selected))
        raise ValueError(f"no LoRA target modules matched: {choices}")
    if requested:
        matched = {name for name in requested if any(path.endswith(name) for path in replaced)}
        missing = sorted(set(requested) - matched)
        if missing:
            raise ValueError(f"LoRA target modules were not found: {', '.join(missing)}")
    return tuple(replaced)


def save_adapter(model: Any, directory: str | Path, config: dict[str, Any]) -> Path:
    """Save only trainable LoRA tensors and strict adapter metadata."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    if not weights or any(not name.endswith(("lora_a", "lora_b")) for name in weights):
        raise ValueError("adapter save expected only trainable LoRA tensors")
    target = root / "adapters.safetensors"
    mx.save_safetensors(str(target), weights)
    config = dict(config)
    config["weights_sha256"] = _sha256_file(target)
    (root / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def load_adapter(
    model: Any,
    adapter_path: str | Path,
    *,
    base_model_id: str,
    base_revision: str | None,
) -> dict[str, Any]:
    """Validate adapter ancestry and tensor shapes before applying weights."""
    root = Path(adapter_path).expanduser()
    if root.is_file():
        root = root.parent
    config_path = root / "adapter_config.json"
    weights_path = root / "adapters.safetensors"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read adapter_config.json: {exc}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("adapter configuration requires schema_version 1")
    checksum = config.get("weights_sha256")
    if not isinstance(checksum, str) or checksum != _sha256_file(weights_path):
        raise ValueError("adapter weights checksum is missing or does not match")
    if config.get("base_model_id") != base_model_id:
        raise ValueError("adapter base model does not match the loaded model")
    if base_revision is not None and config.get("base_revision") != base_revision:
        raise ValueError("adapter base revision does not match the loaded revision")
    apply_lora(
        model,
        num_layers=_positive_int(config, "num_layers"),
        rank=_positive_int(config, "rank"),
        scale=_positive_number(config, "scale"),
        dropout=_dropout(config.get("dropout", 0.0)),
        target_modules=tuple(config.get("target_modules", ())),
    )
    try:
        weights = mx.load(str(weights_path))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"cannot read adapter safetensors: {exc}") from exc
    if not isinstance(weights, dict):
        raise ValueError("adapter safetensors must contain named tensors")
    expected = dict(tree_flatten(model.trainable_parameters()))
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(weights)
        if tuple(expected[name].shape) != tuple(weights[name].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "adapter tensor contract failed: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, shape_mismatch={mismatched[:5]}"
        )
    model.load_weights(list(weights.items()), strict=False)
    model.eval()
    return config


def _decoder_layers(model: Any) -> list[Any]:
    for candidate in (
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
    ):
        layers = getattr(candidate, "layers", None)
        if isinstance(layers, list) and layers:
            return layers
    raise ValueError("model does not expose supported decoder layers")


def _positive_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"adapter {name} must be a positive integer")
    return value


def _positive_number(config: dict[str, Any], name: str) -> float:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"adapter {name} must be positive")
    return float(value)


def _dropout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ValueError("adapter dropout must be in [0, 1)")
    return float(value)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read adapter weights: {exc}") from exc
    return digest.hexdigest()
