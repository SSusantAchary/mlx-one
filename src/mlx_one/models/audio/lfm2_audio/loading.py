"""Pinned native LFM Audio loading; separate model and waveform-codec assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_one.audio.weights import read_safetensors
from mlx_one.models.audio.lfm2_audio.checkpoint_config import LFM2AudioConfig


@dataclass(frozen=True)
class LoadedLFMAudio:
    model: Any
    tokenizer: Any
    codec: Any
    path: Path
    revision: str | None
    quantization: dict
    architecture: str = "lfm2_audio"


def load_lfm_audio(model, *, revision=None, offline=False, cache_dir=None):
    root = Path(model).expanduser()
    if not root.is_dir():
        if isinstance(model, Path) or str(model).startswith(("/", ".", "~")):
            raise ValueError("local LFM audio checkpoint does not exist")
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                str(model),
                revision=revision,
                local_files_only=offline,
                cache_dir=str(cache_dir) if cache_dir else None,
                allow_patterns=[
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors.index.json",
                ],
            )
        )
        revision = root.name
        index_path = root / "model.safetensors.index.json"
        shards = ["model.safetensors"]
        if index_path.exists():
            shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
            if any(Path(x).name != x or not x.endswith(".safetensors") for x in shards):
                raise ValueError("unsafe LFM checkpoint shard path")
        root = Path(
            snapshot_download(
                str(model),
                revision=revision,
                local_files_only=offline,
                cache_dir=str(cache_dir) if cache_dir else None,
                allow_patterns=shards
                + ["audio_detokenizer/config.json", "audio_detokenizer/model.safetensors"],
            )
        )
    payload = json.loads((root / "config.json").read_text())
    if payload.get("model_type") not in {"lfm_audio", "lfm2_audio"}:
        raise ValueError("not a publisher LFM Audio configuration")
    config = LFM2AudioConfig.from_dict(payload)
    validate_config(config)
    from tokenizers import Tokenizer

    from mlx_one.models.audio.lfm2_audio.detokenizer import (
        DetokenizerConfig,
        LFM2AudioDetokenizer,
    )
    from mlx_one.models.audio.lfm2_audio.native import NativeLFMAudio
    from mlx_one.text.loading import (
        _check_checkpoint_memory,
        _materialize_loaded_model,
        _prepare_quantized_model,
        _quantization_config,
        _validate_quantized_tensors,
    )

    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    for text, token in [
        ("<|startoftext|>", 1),
        ("<|im_start|>", 6),
        ("<|im_end|>", 7),
        ("<|audio_start|>", 128),
        ("<|text_end|>", 130),
    ]:
        if tokenizer.token_to_id(text) != token:
            raise ValueError(f"LFM tokenizer special token mismatch: {text}")
    _check_checkpoint_memory(root)
    quant = _quantization_config(payload)
    if quant and (quant["mode"] != "affine" or quant["bits"] not in {4, 8}):
        raise ValueError("LFM audio supports affine Q4/Q8 only")
    native = NativeLFMAudio(config)
    weights = read_safetensors(root)
    # Current MLX export uses the shared LFM2 names except depthwise filter storage.
    weights = {
        k.removesuffix(".weight")
        if k.startswith("lfm.") and k.endswith(".conv.conv.weight")
        else k: v[..., 0]
        if k.startswith("lfm.") and k.endswith(".conv.conv.weight") and v.ndim == 3
        else v
        for k, v in weights.items()
    }
    if quant:
        _prepare_quantized_model(native, weights, quant)
        _validate_quantized_tensors(native, weights)
    native.load_weights(list(weights.items()), strict=True)
    native.eval()
    _materialize_loaded_model(native)
    codec_root = root / "audio_detokenizer"
    codec_payload = json.loads((codec_root / "config.json").read_text())
    codec_config = DetokenizerConfig.from_dict(codec_payload)
    codec_weights = read_safetensors(codec_root)
    codec_config.intermediate_size = codec_weights["lfm.layers.0.feed_forward.w1.weight"].shape[0]
    window = codec_weights.pop("istft.window", None)
    if window is None or tuple(window.shape) != (codec_config.n_fft,):
        raise ValueError("LFM waveform codec requires its published ISTFT window")
    codec = LFM2AudioDetokenizer(codec_config)
    # Separate codec export is BF16, with native convolution layout.
    codec_weights = {
        k: v.transpose(0, 2, 1)
        if k.endswith(".conv.conv.weight") and v.ndim == 3 and v.shape[1] == 1
        else v
        for k, v in codec_weights.items()
    }
    codec.load_weights(list(codec_weights.items()), strict=True)
    codec._window = window
    codec.eval()
    _materialize_loaded_model(codec)
    return LoadedLFMAudio(native, tokenizer, codec, root.resolve(), revision, quant)


def validate_config(config):
    frontend = config.preprocessor
    if (
        frontend.sample_rate != 16000
        or frontend.n_fft != 512
        or frontend.features != 128
        or frontend.window_size != 0.025
        or frontend.window_stride != 0.01
        or frontend.preemph != 0.97
        or frontend.normalize != "per_feature"
        or frontend.window != "hann"
        or frontend.frame_splicing != 1
        or not frontend.log
        or frontend.pad_to != 0
    ):
        raise ValueError("unsupported publisher LFM audio frontend configuration")
    if (
        config.tie_audio_embeddings
        or config.encoder.xscaling
        or config.encoder.att_context_size != [-1, -1]
        or config.codebooks != 8
        or config.audio_vocab_size != 2049
        or config.sample_rate != 24000
        or config.encoder.subsampling_factor != 8
        or config.encoder.self_attention_model != "rel_pos"
        or config.encoder.conv_norm_type != "batch_norm"
        or config.encoder.causal_downsampling
        or config.encoder.reduction is not None
        or config.depthformer.dim % config.depthformer.num_heads
        or config.depthformer.num_heads % config.depthformer.num_kv_heads
    ):
        raise ValueError("unsupported LFM audio architecture/codec configuration")
