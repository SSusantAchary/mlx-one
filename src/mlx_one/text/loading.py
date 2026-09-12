"""Safe native text-model loading with GPT-2 as the first vertical slice."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import SafetensorError, safe_open

from mlx_one.models.language.gpt2.config import GPT2Config
from mlx_one.models.language.gpt2.weights import sanitize_weights, weight_contract
from mlx_one.text.tokenizer import GPT2Tokenizer


class TextModelLoadError(RuntimeError):
    """Raised when a native text bundle is unsafe, incomplete, or unsupported."""


@dataclass(frozen=True)
class LoadedTextModel:
    model: Any
    tokenizer: GPT2Tokenizer
    path: Path
    model_id: str
    revision: str | None


def resolve_text_model_type(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> str:
    local = Path(model).expanduser()
    if not local.exists() and (isinstance(model, Path) or str(model).startswith((".", "/", "~"))):
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
) -> LoadedTextModel:
    root = _resolve(model, revision=revision, offline=offline, cache_dir=cache_dir)
    config_data = _read_json(root / "config.json")
    if config_data.get("model_type") != "gpt2":
        raise TextModelLoadError(
            f"unsupported native text model type: {config_data.get('model_type')!r}"
        )
    config = GPT2Config.from_dict(config_data)
    tokenizer = GPT2Tokenizer.from_directory(root)
    weights = sanitize_weights(_read_safetensors(root), config)
    weight_contract(config).validate(weights)
    from mlx_one.models.language.gpt2.model import GPT2LMHeadModel

    native = GPT2LMHeadModel(config)
    native.load_weights(list(weights.items()), strict=True)
    native.eval()
    resolved_revision = revision
    if resolved_revision is None and re.fullmatch(r"[0-9a-f]{40,64}", root.name):
        resolved_revision = root.name
    return LoadedTextModel(native, tokenizer, root, str(model), resolved_revision)


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
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.json",
                    "merges.txt",
                    "*.safetensors",
                    "*.safetensors.index.json",
                ],
            )
        )
    except Exception as exc:
        raise TextModelLoadError(f"cannot resolve native text model: {exc}") from exc


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
    import mlx.core as mx

    result: dict[str, Any] = {}
    try:
        for path in files:
            with safe_open(path, framework="numpy") as handle:
                for name in handle.keys():
                    if name in result:
                        raise TextModelLoadError(f"duplicate text tensor: {name}")
                    if expected_map is not None and expected_map.get(name) != path.name:
                        raise TextModelLoadError(f"safetensors index mismatch for tensor: {name}")
                    result[name] = mx.array(handle.get_tensor(name))
    except (OSError, SafetensorError) as exc:
        raise TextModelLoadError(f"cannot read text safetensors: {exc}") from exc
    if expected_map is not None and set(result) != set(expected_map):
        raise TextModelLoadError("safetensors index and shard tensors do not match")
    return result


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
