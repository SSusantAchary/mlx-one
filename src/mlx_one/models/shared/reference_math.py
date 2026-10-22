# Copyright © 2023-2026 Apple Inc.
# SPDX-License-Identifier: MIT
"""Dense mathematical helpers adapted from the pinned local MLX-LM snapshot.

Only model mathematics lives here. MLX-One retains cache ownership and execution
policy. See models/UPSTREAM_LICENSE and the reference-first source provenance.
"""

import mlx.core as mx
import mlx.nn as nn


def swiglu(gate, x):
    from mlx_one.models.shared.reference_activations import swiglu as compiled_swiglu

    return compiled_swiglu(gate, x)


def precise_swiglu(h, gate, x):
    from mlx_one.models.shared.reference_activations import precise_swiglu as compiled_swiglu

    return compiled_swiglu(h, gate, x)


class Llama3RoPE(nn.Module):
    """MLX-LM rope_utils.Llama3RoPE, including its frequency operation order."""

    def __init__(self, dims, base, traditional, scaling_config):
        super().__init__()
        self.dims, self.traditional = dims, traditional
        factor = scaling_config["factor"]
        low = scaling_config.get("low_freq_factor", 1.0)
        high = scaling_config.get("high_freq_factor", 4.0)
        original = scaling_config.get("original_max_position_embeddings", 8192)
        low_wavelength, high_wavelength = original / low, original / high
        freqs = base ** (mx.arange(0, dims, 2) / dims)
        wavelengths = 2 * mx.pi * freqs
        freqs = mx.where(wavelengths > low_wavelength, freqs * factor, freqs)
        medium = (wavelengths > high_wavelength) & (wavelengths < low_wavelength)
        smooth = (original / wavelengths - low) / (high - low)
        smooth_freqs = freqs / ((1 - smooth) / factor + smooth)
        self._freqs = mx.where(medium, smooth_freqs, freqs)

    def __call__(self, x, offset=0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def initialize_rope(
    dims, base, traditional=False, scaling_config=None, max_position_embeddings=None
):
    kind = (scaling_config or {}).get("type") or (scaling_config or {}).get("rope_type", "default")
    if kind in {"default", "linear", "mrope"}:
        scale = 1 / scaling_config["factor"] if kind == "linear" else 1.0
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)
    if kind == "llama3":
        return Llama3RoPE(dims, base, traditional, scaling_config)
    raise ValueError(f"unsupported reference-port RoPE type: {kind}")


def rotary(x, rope, args, offset, position_ids):
    if position_ids is None:
        return rope(x, offset=offset)
    # Padding, explicit positions and multimodal positions are existing MLX-One
    # extensions. The contiguous, unpadded path above is exactly the source path.
    from mlx_one.models.shared.transformer import apply_rope

    scaling = getattr(args, "rope_scaling", None) or {}
    return apply_rope(
        x,
        offset=offset,
        theta=args.rope_theta,
        position_ids=position_ids,
        scaling_factor=scaling.get("factor", 1.0)
        if scaling.get("rope_type", scaling.get("type")) == "linear"
        else 1.0,
        rope_scaling=scaling,
        mrope_section=getattr(args, "mrope_section", None),
    )


def attention_mask(length, offset=0, padding=None, window=None):
    if padding is not None and (padding.ndim != 2 or padding.shape[1] != offset + length):
        raise ValueError("attention_mask must cover the complete cached sequence")
    if padding is None and window is None:
        return None if length == 1 else "causal"
    query = mx.arange(offset, offset + length)[:, None]
    key = mx.arange(offset + length)[None, :]
    mask = query >= key
    if window is not None:
        mask = mask & (query < key + window)
    if padding is not None:
        mask = mask & (padding[:, None, None, :] != 0)
    return mask


def scaled_dot_product_attention(queries, keys, values, *, cache, scale, mask):
    # Native cache owners may store FP16 while a source model runs in BF16.
    # SDPA requires matching operand types; this is an explicit cache boundary.
    keys, values = keys.astype(queries.dtype), values.astype(queries.dtype)
    if mask is not None and not isinstance(mask, str) and mask.dtype != mx.bool_:
        mask = mask.astype(queries.dtype)
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask, sinks=None
    )
