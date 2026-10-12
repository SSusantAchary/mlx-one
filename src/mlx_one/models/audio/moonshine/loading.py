"""Strict native loading of original Moonshine safetensors and processor assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_one.audio.weights import read_safetensors, select_safetensors
from mlx_one.models.audio.moonshine.config import MoonshineConfig
from mlx_one.models.audio.moonshine.weights import sanitize_weights, weight_contract


class MoonshineLoadError(ValueError):
    """The checkpoint is incomplete or not the supported Moonshine contract."""


@dataclass(frozen=True)
class LoadedMoonshine:
    model: Any
    tokenizer: Any
    generation_config: dict[str, Any]
    path: Path
    revision: str | None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MoonshineLoadError(f"{path.name} must be a JSON object")
    return value


def validate_processor(data: dict[str, Any]) -> None:
    expected = {
        "feature_extractor_type": "Wav2Vec2FeatureExtractor",
        "sampling_rate": 16000,
        "feature_size": 1,
        "padding_side": "right",
        "padding_value": 0.0,
        "do_normalize": False,
        "return_attention_mask": True,
    }
    if any(data.get(name) != value for name, value in expected.items()):
        raise MoonshineLoadError("unsupported Moonshine waveform preprocessing configuration")


def validate_generation(data: dict[str, Any], config: MoonshineConfig) -> None:
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id"):
        value = data.get(name, getattr(config, name))
        if isinstance(value, bool) or value != getattr(config, name):
            raise MoonshineLoadError(f"generation {name} disagrees with model configuration")
    length = data.get("max_length", 194)
    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise MoonshineLoadError("generation max_length must be an integer greater than one")
    # Do not silently ignore generation processors not implemented here.
    for name, default in (
        ("num_beams", 1),
        ("do_sample", False),
        ("repetition_penalty", 1.0),
        ("length_penalty", 1.0),
        ("min_length", 0),
        ("no_repeat_ngram_size", 0),
    ):
        if data.get(name, default) != default:
            raise MoonshineLoadError(f"unsupported Moonshine generation setting: {name}")
    for name in (
        "forced_decoder_ids",
        "suppress_tokens",
        "begin_suppress_tokens",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "max_new_tokens",
    ):
        if data.get(name) is not None:
            raise MoonshineLoadError(f"unsupported Moonshine generation setting: {name}")


def load_moonshine(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> LoadedMoonshine:
    local = Path(model).expanduser()
    if local.is_dir():
        root = local.resolve()
    elif local.exists() or isinstance(model, Path) or str(model).startswith(("/", ".", "~")):
        raise MoonshineLoadError("local Moonshine model must be an existing directory")
    else:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                str(model),
                revision=revision,
                local_files_only=offline,
                cache_dir=str(cache_dir) if cache_dir else None,
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "preprocessor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "model.safetensors.index.json",
                ],
            )
        )
        revision = root.name
        filenames = (
            [path.name for path in select_safetensors(root)[0]]
            if (root / "model.safetensors.index.json").exists()
            else ["model.safetensors"]
        )
        root = Path(
            snapshot_download(
                str(model),
                revision=revision,
                local_files_only=offline,
                cache_dir=str(cache_dir) if cache_dir else None,
                allow_patterns=filenames,
            )
        )
    config = MoonshineConfig.from_dict(read_json(root / "config.json"))
    validate_processor(read_json(root / "preprocessor_config.json"))
    generation = read_json(root / "generation_config.json")
    validate_generation(generation, config)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise MoonshineLoadError("tokenizer vocabulary size does not match Moonshine config")
    if set(tokenizer.get_vocab(with_added_tokens=True).values()) != set(range(config.vocab_size)):
        raise MoonshineLoadError("tokenizer IDs must cover the contiguous model vocabulary")
    for text, token_id in (("<s>", config.bos_token_id), ("</s>", config.eos_token_id)):
        if tokenizer.token_to_id(text) != token_id:
            raise MoonshineLoadError("tokenizer special IDs disagree with Moonshine config")
    weights = read_safetensors(root)
    if "proj_out.weight" in weights:
        import mlx.core as mx

        embedding, output = (
            weights.get("model.decoder.embed_tokens.weight"),
            weights["proj_out.weight"],
        )
        if (
            embedding is None
            or embedding.shape != output.shape
            or not mx.all(embedding == output).item()
        ):
            raise MoonshineLoadError("untied Moonshine output projection is unsupported")
    cleaned = sanitize_weights(weights, config)
    weight_contract(config).validate(cleaned)
    from mlx_one.models.audio.moonshine.model import MoonshineForConditionalGeneration

    native = MoonshineForConditionalGeneration(config)
    native.load_weights(list(cleaned.items()), strict=True)
    native.eval()
    return LoadedMoonshine(native, tokenizer, generation, root, revision)
