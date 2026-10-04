"""Safe local and Hugging Face loading for native Whisper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from safetensors import SafetensorError, safe_open

from mlx_one.audio.weights import AudioWeightsError, select_safetensors, validate_tensor_index
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
    _validate_generation_config(generation, config)
    if processor.feature_size != config.num_mel_bins:
        raise WhisperLoadError("preprocessor feature_size does not match Whisper config")
    weights = _read_safetensors(root)
    _validate_tied_projection(weights)
    cleaned = sanitize_weights(weights, config)
    weight_contract(config).validate(cleaned)
    from mlx_one.models.audio.whisper.model import WhisperForConditionalGeneration

    native = WhisperForConditionalGeneration(config)
    native.load_weights(list(cleaned.items()), strict=True)
    native.eval()
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
        metadata_patterns = [
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors.index.json",
        ]
        path = snapshot_download(
            str(model),
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=offline,
            allow_patterns=metadata_patterns,
        )
        # The index pins one layout; alternate fp32/consolidated exports are not
        # downloaded or combined. Resolve weights at the same snapshot commit.
        root = Path(path)
        index = root / "model.safetensors.index.json"
        filenames = (
            [item.name for item in select_safetensors(root)[0]]
            if index.is_file()
            else ["model.safetensors"]
        )
        path = snapshot_download(
            str(model),
            revision=root.name,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=offline,
            allow_patterns=filenames,
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
    try:
        files, mapping = select_safetensors(root)
        names = {}
        for path in files:
            with safe_open(path, framework="numpy") as handle:
                names[path.name] = tuple(handle.keys())
        validate_tensor_index(files, mapping, names)
        import mlx.core as mx

        result: dict[str, Any] = {}
        for path in files:
            # MLX reads stored dtypes directly, including bfloat16, without a
            # NumPy conversion or an implicit precision change.
            result.update(mx.load(str(path)))
    except (OSError, SafetensorError, AudioWeightsError, ValueError) as exc:
        raise WhisperLoadError(f"cannot read Whisper safetensors: {exc}") from exc
    return result


def _validate_tied_projection(weights: dict[str, Any]) -> None:
    output = weights.get("proj_out.weight")
    if output is None:
        return
    embedding = weights.get("model.decoder.embed_tokens.weight")
    if embedding is None or output.shape != embedding.shape:
        raise WhisperLoadError("tied output projection does not match decoder embeddings")
    import mlx.core as mx

    if not bool(mx.all(output == embedding).item()):
        raise WhisperLoadError("untied output projection is unsupported")


def _validate_generation_config(generation: dict[str, Any], config: WhisperConfig) -> None:
    for field in ("suppress_tokens", "begin_suppress_tokens"):
        values = generation.get(field)
        if values is None:
            continue
        if not isinstance(values, list) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < config.vocab_size
            for value in values
        ):
            raise WhisperLoadError(f"generation {field} must contain valid vocabulary IDs")
    heads = generation.get("alignment_heads")
    if heads is not None:
        if not isinstance(heads, list) or not heads:
            raise WhisperLoadError("alignment_heads must be a nonempty list")
        for pair in heads:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in pair)
                or not 0 <= pair[0] < config.decoder_layers
                or not 0 <= pair[1] < config.decoder_attention_heads
            ):
                raise WhisperLoadError("alignment_heads must address existing decoder heads")
