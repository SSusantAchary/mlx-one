"""Strict, lazy loading of original Qwen audio publisher safetensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_one.audio.weights import read_safetensors, select_safetensors
from mlx_one.models.audio.qwen3_asr.config import QwenASRConfig
from mlx_one.models.audio.qwen3_asr.processing import (
    read_json,
    tokenizer_from_directory,
    validate_preprocessor,
)
from mlx_one.models.audio.qwen3_asr.weights import sanitize_weights


@dataclass(frozen=True)
class LoadedQwenAudio:
    model: Any
    tokenizer: Any
    generation_config: dict[str, Any]
    chat_template: str
    path: Path
    revision: str | None


def validate_generation(data: dict[str, Any], tokenizer: Any) -> None:
    if data.get("eos_token_id") != [151643, 151645] or data.get("pad_token_id") != 151643:
        raise ValueError("unsupported Qwen generation token contract")
    if (
        tokenizer.token_to_id("<|endoftext|>") != 151643
        or tokenizer.token_to_id("<|im_end|>") != 151645
    ):
        raise ValueError("Qwen tokenizer EOS IDs disagree with generation config")
    for name, default in (
        ("do_sample", False),
        ("num_beams", 1),
        ("repetition_penalty", 1.0),
        ("min_length", 0),
        ("no_repeat_ngram_size", 0),
    ):
        if data.get(name, default) != default:
            raise ValueError(f"unsupported Qwen generation setting: {name}")
    for name in (
        "forced_decoder_ids",
        "suppress_tokens",
        "forced_bos_token_id",
        "forced_eos_token_id",
    ):
        if data.get(name) is not None:
            raise ValueError(f"unsupported Qwen generation setting: {name}")


def load_qwen_audio(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> LoadedQwenAudio:
    path = Path(model).expanduser()
    if path.is_dir():
        root = path.resolve()
    elif path.exists() or isinstance(model, Path) or str(model).startswith(("/", ".", "~")):
        raise ValueError("local Qwen audio checkpoint must be an existing directory")
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
                    "preprocessor_config.json",
                    "generation_config.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "chat_template.json",
                    "model.safetensors.index.json",
                ],
            )
        )
        revision = root.name
        shards = (
            [p.name for p in select_safetensors(root)[0]]
            if (root / "model.safetensors.index.json").exists()
            else ["model.safetensors"]
        )
        root = Path(
            snapshot_download(
                str(model),
                revision=revision,
                local_files_only=offline,
                cache_dir=str(cache_dir) if cache_dir else None,
                allow_patterns=shards,
            )
        )
    payload = read_json(root / "config.json")
    config = QwenASRConfig.from_dict(payload)
    validate_preprocessor(read_json(root / "preprocessor_config.json"))
    tokenizer = tokenizer_from_directory(root, config)
    generation = read_json(root / "generation_config.json")
    validate_generation(generation, tokenizer)
    template = read_json(root / "chat_template.json").get("chat_template")
    if not isinstance(template, str):
        raise ValueError("Qwen checkpoint requires its chat template asset")
    from mlx_one.text.loading import (
        _materialize_loaded_model,
        _prepare_quantized_model,
        _quantization_config,
        _validate_quantized_tensors,
    )

    quantization = _quantization_config(payload)
    raw = read_safetensors(root)
    if quantization:
        # MLX exports are already in native NHWC conv layout, with no thinker prefix.
        if any(k.startswith("thinker.") for k in raw):
            raise ValueError("quantized Qwen requires the MLX native tensor layout")
        weights = raw
    else:
        weights = sanitize_weights(raw, config)
    from mlx_one.models.audio.qwen3_asr.model import QwenASRModel

    native = QwenASRModel(config)
    if quantization:
        _prepare_quantized_model(native, weights, quantization)
        _validate_quantized_tensors(native, weights)
    native.load_weights(list(weights.items()), strict=True)
    _materialize_loaded_model(native)
    native.eval()
    return LoadedQwenAudio(native, tokenizer, generation, template, root, revision)
