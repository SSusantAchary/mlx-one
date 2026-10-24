"""Strict native loading of canonical Parakeet TDT safetensors and tokenizer assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mlx_one.audio.weights import read_safetensors, select_safetensors
from mlx_one.models.audio.parakeet.config import ParakeetConfig
from mlx_one.models.audio.parakeet.processing import ParakeetProcessor
from mlx_one.models.audio.parakeet.weights import sanitize_weights, weight_contract


@dataclass(frozen=True)
class LoadedParakeet:
    model: Any
    tokenizer: Any
    processor: ParakeetProcessor
    generation_config: dict
    path: Path
    revision: str | None
    variant: str


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def validate_generation(data: dict, config: ParakeetConfig) -> None:
    if data.get("decoder_start_token_id") != config.blank_token_id:
        raise ValueError("TDT decoder must start with the checkpoint blank ID")
    if data.get("pad_token_id") != config.pad_token_id:
        raise ValueError("TDT generation padding ID disagrees with checkpoint")
    expected = list(range(config.vocab_size, config.vocab_size + len(config.durations)))
    if data.get("suppress_tokens") != expected:
        raise ValueError("TDT generation must suppress exactly the duration-head IDs")
    eos = data.get("eos_token_id")
    if eos is not None and (
        isinstance(eos, bool) or not isinstance(eos, int) or not 0 <= eos < config.vocab_size
    ):
        raise ValueError("TDT EOS must be an in-vocabulary ID")
    for key, default in (("do_sample", False), ("num_beams", 1), ("repetition_penalty", 1.0)):
        if data.get(key, default) != default:
            raise ValueError(f"unsupported TDT generation option: {key}")
    for key in (
        "forced_decoder_ids",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "begin_suppress_tokens",
        "max_new_tokens",
    ):
        if data.get(key) is not None:
            raise ValueError(f"unsupported TDT generation processor: {key}")


def validate_headers(root: Path, config: ParakeetConfig) -> None:
    from safetensors import safe_open

    files, _ = select_safetensors(root, layout="canonical")
    with safe_open(files[0], framework="numpy") as handle:
        shapes = {}
        for name in handle.keys():
            tensor = handle.get_slice(name)
            expected_dtype = "I64" if name.endswith("num_batches_tracked") else "F32"
            if tensor.get_dtype() != expected_dtype:
                raise ValueError(f"unsupported Parakeet tensor dtype: {name}={tensor.get_dtype()}")
            shapes[name] = SimpleNamespace(shape=tuple(tensor.get_shape()))
        weight_contract(config, native=False).validate(shapes)


def validate_conversion(root: Path) -> dict:
    data = read_json(root / "conversion-provenance.json")
    if (
        data.get("model") != "nvidia/parakeet-tdt-0.6b-v2"
        or data.get("revision") != "ae9ad07059c7c739ffaf932226a8fe64ae2620b0"
        or data.get("archive_sha256")
        != "d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c"
    ):
        raise ValueError("converted Parakeet v2 provenance differs from pinned source")
    assets = data.get("assets")
    if not isinstance(assets, dict) or not {
        "model.safetensors",
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    }.issubset(assets):
        raise ValueError("incomplete Parakeet v2 conversion manifest")
    for name, checksum in assets.items():
        if Path(name).name != name or "\\" in name:
            raise ValueError("unsafe conversion asset name")
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != checksum:
            raise ValueError(f"converted Parakeet v2 asset checksum differs: {name}")
    return data


def load_parakeet(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> LoadedParakeet:
    local = Path(model).expanduser()
    if local.is_dir():
        root = local.resolve()
        if revision is None and len(root.name) == 40:
            revision = root.name
    elif local.exists() or isinstance(model, Path) or str(model).startswith(("/", ".", "~")):
        raise ValueError("local Parakeet checkpoint must be an existing directory")
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
                    "processor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors",
                ],
            )
        )
        revision = root.name
    payload = read_json(root / "config.json")
    if payload.get("model_type") is None and "joint" in payload:
        return _load_mlx_export(root, payload, revision)
    config = ParakeetConfig.from_dict(payload)
    variant = config.extra.get("mlx_one_checkpoint_variant", "v3")
    if variant not in {"v2", "v3"}:
        raise ValueError("unsupported Parakeet checkpoint variant")
    if variant == "v2":
        revision = validate_conversion(root)["revision"]
    processor = ParakeetProcessor.from_dict(read_json(root / "processor_config.json"))
    if processor.feature_size != config.num_mel_bins:
        raise ValueError("Parakeet processor Mel dimensions disagree with encoder")
    generation = read_json(root / "generation_config.json")
    validate_generation(generation, config)
    validate_headers(root, config)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    token_config = read_json(root / "tokenizer_config.json")
    if (
        tokenizer.token_to_id("<blank>") != config.blank_token_id
        or tokenizer.token_to_id(token_config.get("pad_token", "")) != config.pad_token_id
    ):
        raise ValueError("Parakeet tokenizer padding/blank IDs disagree with checkpoint")
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Parakeet tokenizer vocabulary disagrees with checkpoint")
    import mlx.core as mx

    from mlx_one.models.audio.parakeet.model import ParakeetTDT

    weights = sanitize_weights(read_safetensors(root, layout="canonical"), config)
    weight_contract(config).validate(weights)
    model_object = ParakeetTDT(config)
    model_object.load_weights(list(weights.items()), strict=True)
    model_object.eval()
    mx.eval(model_object.parameters())
    return LoadedParakeet(model_object, tokenizer, processor, generation, root, revision, variant)


def _load_mlx_export(root, payload, revision):
    from mlx_one.models.audio.parakeet.mlx_export import (
        VocabularyTokenizer,
        config_from_mlx,
        map_weights,
    )
    from mlx_one.models.audio.parakeet.model import ParakeetTDT

    config = config_from_mlx(payload)
    if config.vocab_size not in {1025, 8193}:
        raise ValueError('unknown MLX Parakeet checkpoint vocabulary/variant')
    frontend = payload['preprocessor']
    if any(frontend.get(k) != v for k, v in {'sample_rate': 16000, 'normalize': 'per_feature',
            'features': 128, 'n_fft': 512, 'window_size': 0.025, 'window_stride': 0.01,
            'window': 'hann', 'frame_splicing': 1, 'log': True}.items()):
        raise ValueError('unsupported MLX Parakeet frontend')
    weights = map_weights(read_safetensors(root, layout='canonical'), config)
    if any(str(v.dtype).rsplit('.', 1)[-1] not in {'float32', 'int64'}
           for v in weights.values()):
        raise ValueError('MLX Parakeet export requires its published float32 weights')
    native = ParakeetTDT(config)
    native.load_weights(list(weights.items()), strict=True)
    native.eval()
    import mlx.core as mx

    mx.eval(native.parameters())
    generation = {'decoder_start_token_id': config.blank_token_id,
                  'pad_token_id': config.pad_token_id,
                  'eos_token_id': payload['joint']['vocabulary'].index('<|endoftext|>')
                  if '<|endoftext|>' in payload['joint']['vocabulary'] else None}
    return LoadedParakeet(native, VocabularyTokenizer(payload['joint']['vocabulary']),
        ParakeetProcessor(), generation, root, revision,
        'v2' if config.vocab_size == 1025 else 'v3')
