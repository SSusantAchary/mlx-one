"""Safe registry-driven native causal-language model loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

from mlx_one.core.registry import get_registration
from mlx_one.text.chat import ChatTemplate
from mlx_one.text.tokenizer import GPT2Tokenizer
from mlx_one.text.tokenizers import HFTokenizerAdapter, TextTokenizer


class TextModelLoadError(RuntimeError):
    """Raised when a native text bundle is unsafe, incomplete, or unsupported."""


@dataclass(frozen=True)
class LoadedTextModel:
    model: Any
    tokenizer: TextTokenizer
    path: Path
    model_id: str
    revision: str | None
    architecture: str = "gpt2"
    context_length: int = 0
    quantization: dict[str, Any] = field(default_factory=dict)
    chat_template: ChatTemplate | None = None
    parameter_count: int | None = None


_TEXT_TYPES = {"gpt2", "lfm2", "lfm2_moe", "openelm", "qwen2", "qwen2_moe", "qwen3"}


def resolve_text_model_type(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> str:
    local = Path(model).expanduser()
    if not local.exists() and (
        isinstance(model, Path) or str(model).startswith((".", "/", "~"))
    ):
        raise TextModelLoadError(f"local text model directory does not exist: {local}")
    try:
        config_path = (
            local / "config.json"
            if local.is_dir()
            else Path(
                hf_hub_download(
                    str(model),
                    "config.json",
                    revision=revision,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                    local_files_only=offline,
                )
            )
        )
        return str(_read_json(config_path).get("model_type", ""))
    except Exception as exc:
        if isinstance(exc, TextModelLoadError):
            raise
        raise TextModelLoadError(f"cannot resolve text model type: {exc}") from exc


def load_text_model(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    tokenizer_source: str | Path | None = None,
) -> LoadedTextModel:
    root = _resolve(model, revision=revision, offline=offline, cache_dir=cache_dir)
    config_data = _read_json(root / "config.json")
    model_type = str(config_data.get("model_type", ""))
    if model_type not in _TEXT_TYPES | {"qwen3_5"}:
        raise TextModelLoadError(f"unsupported native text model type: {model_type!r}")
    try:
        registration = get_registration(model_type)
        if registration.modality != "text" and model_type != "qwen3_5":
            raise ValueError(f"model type {model_type!r} is not text-generative")
        config = registration.config_class().from_dict(config_data)
        tokenizer_root = (
            _resolve_tokenizer(
                tokenizer_source, revision=revision, offline=offline, cache_dir=cache_dir
            )
            if tokenizer_source is not None
            else root
        )
        tokenizer = _load_tokenizer(tokenizer_root, model_type)
        chat_template = ChatTemplate.from_directory(tokenizer_root, model_type, tokenizer)
        tensors = registration.sanitizer()(_read_safetensors(root), config)
        native = registration.model_class()(config)
        quantization = _quantization_config(config_data)
        if quantization:
            _prepare_quantized_model(native, tensors, quantization)
            _validate_quantized_tensors(native, tensors)
        else:
            registration.weight_contract()(config).validate(tensors)
        native.load_weights(list(tensors.items()), strict=True)
        native.eval()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        if isinstance(exc, TextModelLoadError):
            raise
        raise TextModelLoadError(f"cannot load native {model_type or 'text'} model: {exc}") from exc
    resolved_revision = revision
    if resolved_revision is None and re.fullmatch(r"[0-9a-f]{40,64}", root.name):
        resolved_revision = root.name
    return LoadedTextModel(
        native,
        tokenizer,
        root,
        str(model),
        resolved_revision,
        architecture=model_type,
        context_length=_context_length(config),
        quantization=quantization,
        chat_template=chat_template,
        parameter_count=_parameter_count(config_data),
    )


def _load_tokenizer(root: Path, model_type: str) -> TextTokenizer:
    try:
        if (
            model_type == "gpt2"
            and (root / "vocab.json").is_file()
            and (root / "merges.txt").is_file()
        ):
            return GPT2Tokenizer.from_directory(root)
        return HFTokenizerAdapter.from_directory(root)
    except ValueError as exc:
        raise TextModelLoadError(f"invalid tokenizer assets: {exc}") from exc


def _resolve(
    model: str | Path,
    *,
    revision: str | None,
    offline: bool,
    cache_dir: str | Path | None,
) -> Path:
    local = Path(model).expanduser()
    if local.exists():
        if not local.is_dir():
            raise TextModelLoadError("local text model must be a directory")
        return local.resolve()
    if isinstance(model, Path) or str(model).startswith((".", "/", "~")):
        raise TextModelLoadError(f"local text model directory does not exist: {local}")
    try:
        return Path(
            snapshot_download(
                str(model),
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=offline,
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                    "vocab.json",
                    "merges.txt",
                    "*.safetensors",
                    "*.safetensors.index.json",
                ],
            )
        )
    except Exception as exc:
        raise TextModelLoadError(f"cannot resolve native text model: {exc}") from exc


def _resolve_tokenizer(
    source: str | Path,
    *,
    revision: str | None,
    offline: bool,
    cache_dir: str | Path | None,
) -> Path:
    local = Path(source).expanduser()
    if local.exists():
        if not local.is_dir():
            raise TextModelLoadError("tokenizer source must be a directory")
        return local.resolve()
    if isinstance(source, Path) or str(source).startswith((".", "/", "~")):
        raise TextModelLoadError(f"local tokenizer directory does not exist: {local}")
    try:
        return Path(
            snapshot_download(
                str(source),
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
                local_files_only=offline,
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                    "vocab.json",
                    "merges.txt",
                ],
            )
        )
    except Exception as exc:
        raise TextModelLoadError(f"cannot resolve tokenizer: {exc}") from exc


def _quantization_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("quantization_config", config.get("quantization"))
    if value is None and isinstance(config.get("text_config"), dict):
        text = config["text_config"]
        value = text.get("quantization_config", text.get("quantization"))
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError("quantization metadata must be an object")
    bits = value.get("bits")
    group_size = value.get("group_size", 64)
    mode = value.get("mode", "affine")
    if bits != 4:
        raise ValueError("only MLX 4-bit checkpoints are supported")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("quantization group_size must be a positive integer")
    if mode not in {"affine", "mxfp4", "nvfp4"}:
        raise ValueError(f"unsupported MLX quantization mode: {mode!r}")
    method = value.get("quant_method", value.get("method"))
    if method is not None and method not in {"mlx", "affine", "mxfp4", "nvfp4"}:
        raise ValueError(f"unsupported quantization method: {method!r}")
    return {"bits": 4, "group_size": group_size, "mode": mode}


def _prepare_quantized_model(
    model: Any, tensors: dict[str, Any], quantization: dict[str, Any]
) -> None:
    import mlx.nn as nn

    prefixes = {name.removesuffix(".scales") for name in tensors if name.endswith(".scales")}
    if not prefixes:
        raise ValueError("4-bit checkpoint has no quantization scale tensors")
    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=4,
        mode=quantization["mode"],
        class_predicate=lambda path, module: path in prefixes and hasattr(module, "to_quantized"),
    )


def _validate_quantized_tensors(model: Any, tensors: dict[str, Any]) -> None:
    """Validate mixed quantized parameters against the transformed model tree."""
    from mlx.utils import tree_flatten

    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(tensors)
        if tuple(expected[name].shape) != tuple(tensors[name].shape)
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        if mismatched:
            details.append(f"shape_mismatch={mismatched[:5]}")
        raise ValueError("invalid mixed 4-bit checkpoint: " + ", ".join(details))


def _context_length(config: Any) -> int:
    if hasattr(config, "text_config"):
        config = config.text_config
    for name in ("max_position_embeddings", "max_context_length", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


def _parameter_count(config: dict[str, Any]) -> int | None:
    for name in ("num_parameters", "parameter_count", "n_params"):
        value = config.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TextModelLoadError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise TextModelLoadError(f"{path.name} must contain a JSON object")
    return value


def _read_safetensors(root: Path) -> dict[str, Any]:
    files, expected_map = _checkpoint_files(root)

    result: dict[str, Any] = {}
    try:
        for path in files:
            tensors = _load_safetensor_file(path)
            for name, tensor in tensors.items():
                if name in result:
                    raise TextModelLoadError(f"duplicate text tensor: {name}")
                if expected_map is not None and expected_map.get(name) != path.name:
                    raise TextModelLoadError(f"safetensors index mismatch for tensor: {name}")
                result[name] = tensor
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, TextModelLoadError):
            raise
        raise TextModelLoadError(f"cannot read text safetensors: {exc}") from exc
    if expected_map is not None and set(result) != set(expected_map):
        raise TextModelLoadError("safetensors index and shard tensors do not match")
    return result


def _load_safetensor_file(path: Path) -> dict[str, Any]:
    """Load safetensors directly with MLX so BF16 never passes through NumPy."""

    import mlx.core as mx

    tensors = mx.load(str(path))
    if not isinstance(tensors, dict):
        raise ValueError(f"{path.name} did not contain a tensor mapping")
    return tensors


def _checkpoint_files(root: Path) -> tuple[tuple[Path, ...], dict[str, str] | None]:
    index_path = root / "model.safetensors.index.json"
    files = tuple(sorted(root.glob("*.safetensors")))
    if not files:
        unsafe = tuple(root.glob("*.bin")) + tuple(root.glob("*.pt"))
        detail = "; pickle checkpoints are not accepted" if unsafe else ""
        raise TextModelLoadError(f"native text safetensors checkpoint is missing{detail}")
    expected_map: dict[str, str] | None = None
    if index_path.exists():
        index = _read_json(index_path)
        value = index.get("weight_map")
        if not isinstance(value, dict) or not value:
            raise TextModelLoadError("safetensors index requires a non-empty weight_map")
        if any(
            not isinstance(name, str) or not isinstance(file, str) for name, file in value.items()
        ):
            raise TextModelLoadError("safetensors weight_map must map tensor names to files")
        expected_map = value
        referenced = {root / file for file in value.values()}
        if any(path.parent != root or not path.is_file() for path in referenced):
            raise TextModelLoadError("safetensors index references a missing or unsafe shard")
        files = tuple(sorted(referenced))
    elif len(files) != 1:
        raise TextModelLoadError("multiple safetensors files require an index")
    return files, expected_map
