"""Strict Hugging Face safetensors contract for Whisper."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.audio.whisper.config import WhisperConfig


def sanitize_weights(weights: dict[str, Any], config: WhisperConfig) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for name, value in weights.items():
        if name == "proj_out.weight":
            continue
        if name in {"model.encoder.embed_positions.weight", "model.decoder.embed_positions.weight"}:
            cleaned[name] = value
            continue
        if name in {"model.encoder.conv1.weight", "model.encoder.conv2.weight"}:
            value = value.transpose(0, 2, 1)
        cleaned[name] = value
    return cleaned


def weight_contract(config: WhisperConfig) -> WeightContract:
    dim = config.d_model
    expected: dict[str, tuple[int, ...]] = {
        "model.encoder.conv1.weight": (dim, 3, config.num_mel_bins),
        "model.encoder.conv1.bias": (dim,),
        "model.encoder.conv2.weight": (dim, 3, dim),
        "model.encoder.conv2.bias": (dim,),
        "model.encoder.embed_positions.weight": (config.max_source_positions, dim),
        "model.encoder.layer_norm.weight": (dim,),
        "model.encoder.layer_norm.bias": (dim,),
        "model.decoder.embed_tokens.weight": (config.vocab_size, dim),
        "model.decoder.embed_positions.weight": (config.max_target_positions, dim),
        "model.decoder.layer_norm.weight": (dim,),
        "model.decoder.layer_norm.bias": (dim,),
    }
    _layers(expected, "model.encoder.layers", config.encoder_layers, dim, config.encoder_ffn_dim)
    _layers(
        expected,
        "model.decoder.layers",
        config.decoder_layers,
        dim,
        config.decoder_ffn_dim,
        cross_attention=True,
    )
    return WeightContract(expected)


def _layers(
    expected: dict[str, tuple[int, ...]],
    stem: str,
    count: int,
    dim: int,
    ffn: int,
    *,
    cross_attention: bool = False,
) -> None:
    for index in range(count):
        prefix = f"{stem}.{index}"
        attentions = ("self_attn", "encoder_attn") if cross_attention else ("self_attn",)
        for attention in attentions:
            for projection in ("q_proj", "v_proj", "out_proj"):
                expected[f"{prefix}.{attention}.{projection}.weight"] = (dim, dim)
                expected[f"{prefix}.{attention}.{projection}.bias"] = (dim,)
            expected[f"{prefix}.{attention}.k_proj.weight"] = (dim, dim)
        norms = ["self_attn_layer_norm", "final_layer_norm"]
        if cross_attention:
            norms.append("encoder_attn_layer_norm")
        for norm in norms:
            expected[f"{prefix}.{norm}.weight"] = (dim,)
            expected[f"{prefix}.{norm}.bias"] = (dim,)
        expected[f"{prefix}.fc1.weight"] = (ffn, dim)
        expected[f"{prefix}.fc1.bias"] = (ffn,)
        expected[f"{prefix}.fc2.weight"] = (dim, ffn)
        expected[f"{prefix}.fc2.bias"] = (dim,)
