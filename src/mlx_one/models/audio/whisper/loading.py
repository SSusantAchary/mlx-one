"""Safe local and Hugging Face loading for native Whisper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from safetensors import SafetensorError, safe_open

from mlx_one.models.audio.whisper.config import WhisperConfig
from mlx_one.models.audio.whisper.processing import WhisperProcessorConfig
from mlx_one.models.audio.whisper.tokenizer import WhisperTokenizer
from mlx_one.models.audio.whisper.weights import sanitize_weights, weight_contract


class WhisperLoadError(RuntimeError):
    """Raised when a Whisper bundle is unsafe, incomplete, or incompatible."""


@dataclass(frozen=True)
class LoadedWhisper:
    model: Any
    tokenizer: WhisperTokenizer
    processor: WhisperProcessorConfig
    generation_config: dict[str, Any]
    path: Path
    revision: str | None


def load_whisper(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> LoadedWhisper:
    root = _resolve(model, revision=revision, offline=offline, cache_dir=cache_dir)
    config = WhisperConfig.from_dict(_read_json(root / "config.json"))
    tokenizer = WhisperTokenizer.from_directory(root)
    processor = WhisperProcessorConfig.from_directory(root)
    generation = _read_json(root / "generation_config.json", required=False)
    if processor.feature_size != config.num_mel_bins:
        raise WhisperLoadError("preprocessor feature_size does not match Whisper config")
    weights = _read_safetensors(root)
    cleaned = sanitize_weights(weights, config)
    weight_contract(config).validate(cleaned)
    from mlx_one.models.audio.whisper.model import WhisperForConditionalGeneration

    native = WhisperForConditionalGeneration(config)
    native.load_weights(list(cleaned.items()), strict=True)
    return LoadedWhisper(native, tokenizer, processor, generation, root, revision)


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
            raise WhisperLoadError("local Whisper model must be a directory")
        return local.resolve()
    if isinstance(model, Path) or str(model).startswith((".", "/", "~")):
        raise WhisperLoadError(f"local Whisper model directory does not exist: {local}")
    try:
        path = snapshot_download(
            str(model),
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=offline,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
                "*.safetensors",
                "*.safetensors.index.json",
            ],
        )
    except Exception as exc:
        raise WhisperLoadError(f"cannot resolve Whisper model: {exc}") from exc
    return Path(path)


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists() and not required:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WhisperLoadError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise WhisperLoadError(f"{path.name} must contain a JSON object")
    return value


def _read_safetensors(root: Path) -> dict[str, Any]:
    files = tuple(sorted(root.glob("*.safetensors")))
    if not files:
        unsafe = tuple(root.glob("*.bin")) + tuple(root.glob("*.pt"))
        detail = "; pickle checkpoints are not accepted" if unsafe else ""
        raise WhisperLoadError(f"Whisper safetensors checkpoint is missing{detail}")
    import mlx.core as mx

    result: dict[str, Any] = {}
    try:
        for path in files:
            with safe_open(path, framework="numpy") as handle:
                for name in handle.keys():
                    if name in result:
                        raise WhisperLoadError(f"duplicate Whisper tensor: {name}")
                    result[name] = mx.array(handle.get_tensor(name))
    except (OSError, SafetensorError) as exc:
        raise WhisperLoadError(f"cannot read Whisper safetensors: {exc}") from exc
    return result
