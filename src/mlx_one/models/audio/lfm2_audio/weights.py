"""Strict top-level weight contract for LFM2.5-Audio."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.audio.lfm2_audio.config import Lfm2AudioConfig
from mlx_one.models.language.lfm2.weights import weight_contract as text_contract


def sanitize_weights(weights: dict[str, Any], config: Lfm2AudioConfig) -> dict[str, Any]:
    del config
    return {
        name: value
        for name, value in weights.items()
        if not name.endswith(("position_ids", "rotary_emb.inv_freq", "causal_mask"))
    }


def weight_contract(config: Lfm2AudioConfig) -> WeightContract:
    expected = {
        f"language_model.{name.removeprefix('model.')}": shape
        for name, shape in text_contract(config.text_config).expected.items()
    }
    audio = config.audio_config
    expected.update(
        {
            "audio_encoder.input_projection.weight": (audio.hidden_size, audio.input_dim),
            "audio_encoder.input_projection.bias": (audio.hidden_size,),
            "audio_projection.weight": (config.text_config.hidden_size, audio.hidden_size),
            "audio_projection.bias": (config.text_config.hidden_size,),
        }
    )
    for index in range(audio.num_hidden_layers):
        prefix = f"audio_encoder.blocks.{index}"
        for norm in (
            "ffn1_norm",
            "attn_norm",
            "conv_norm",
            "ffn2_norm",
            "final_norm",
        ):
            expected[f"{prefix}.{norm}.weight"] = (audio.hidden_size,)
            expected[f"{prefix}.{norm}.bias"] = (audio.hidden_size,)
        for stem in ("ffn1", "ffn2"):
            expected[f"{prefix}.{stem}.weight"] = (
                audio.intermediate_size,
                audio.hidden_size,
            )
            expected[f"{prefix}.{stem}.bias"] = (audio.intermediate_size,)
            expected[f"{prefix}.{stem}_out.weight"] = (
                audio.hidden_size,
                audio.intermediate_size,
            )
            expected[f"{prefix}.{stem}_out.bias"] = (audio.hidden_size,)
        expected[f"{prefix}.self_attn.qkv.weight"] = (
            3 * audio.hidden_size,
            audio.hidden_size,
        )
        expected[f"{prefix}.self_attn.qkv.bias"] = (3 * audio.hidden_size,)
        expected[f"{prefix}.self_attn.out_proj.weight"] = (
            audio.hidden_size,
            audio.hidden_size,
        )
        expected[f"{prefix}.self_attn.out_proj.bias"] = (audio.hidden_size,)
        expected[f"{prefix}.conv_in.weight"] = (2 * audio.hidden_size, audio.hidden_size)
        expected[f"{prefix}.conv_in.bias"] = (2 * audio.hidden_size,)
        expected[f"{prefix}.depthwise_kernel"] = (audio.hidden_size, audio.conv_kernel_size)
        expected[f"{prefix}.conv_out.weight"] = (audio.hidden_size, audio.hidden_size)
        expected[f"{prefix}.conv_out.bias"] = (audio.hidden_size,)
    depth = config.depthformer_config
    for codebook in range(config.num_codebooks):
        expected[f"depthformer.embeddings.{codebook}.weight"] = (
            config.audio_vocab_size,
            depth.hidden_size,
        )
        if not depth.tie_word_embeddings:
            expected[f"depthformer.heads.{codebook}.weight"] = (
                config.audio_vocab_size,
                depth.hidden_size,
            )
    _add_depth_blocks(expected, "depthformer", depth)
    expected["depthformer.norm.weight"] = (depth.hidden_size,)
    detok = config.detokenizer_config
    for codebook in range(detok.num_codebooks):
        expected[f"detokenizer.embeddings.{codebook}.weight"] = (
            detok.codebook_size,
            detok.hidden_size,
        )
    _add_depth_blocks(expected, "detokenizer", detok)
    expected["detokenizer.norm.weight"] = (detok.hidden_size,)
    expected["detokenizer.output.weight"] = (detok.n_fft + 2, detok.hidden_size)
    expected["detokenizer.output.bias"] = (detok.n_fft + 2,)
    return WeightContract(expected)


def _add_depth_blocks(expected: dict[str, tuple[int, ...]], root: str, config: Any) -> None:
    for index in range(config.num_hidden_layers):
        prefix = f"{root}.blocks.{index}"
        expected[f"{prefix}.attn_norm.weight"] = (config.hidden_size,)
        expected[f"{prefix}.attn.qkv.weight"] = (
            3 * config.hidden_size,
            config.hidden_size,
        )
        expected[f"{prefix}.attn.qkv.bias"] = (3 * config.hidden_size,)
        expected[f"{prefix}.attn.out_proj.weight"] = (
            config.hidden_size,
            config.hidden_size,
        )
        expected[f"{prefix}.attn.out_proj.bias"] = (config.hidden_size,)
        expected[f"{prefix}.ffn_norm.weight"] = (config.hidden_size,)
        expected[f"{prefix}.fc1.weight"] = (
            2 * config.intermediate_size,
            config.hidden_size,
        )
        expected[f"{prefix}.fc2.weight"] = (
            config.hidden_size,
            config.intermediate_size,
        )
