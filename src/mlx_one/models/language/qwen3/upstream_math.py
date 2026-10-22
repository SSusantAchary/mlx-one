# Copyright © 2023-2024 Apple Inc.
# Copyright © 2023-2026 Apple Inc. (swiglu)
# SPDX-License-Identifier: MIT
"""Required dense Qwen3 helpers adapted from mlx-lm v0.31.3.

Upstream commit: ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd.
See UPSTREAM_LICENSE in this directory. This module has no mlx-lm dependency.
Only the default/linear RoPE configurations admitted by Qwen3Config are exposed.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    return nn.silu(gate) * x


def initialize_rope(dims, base, traditional, scaling_config=None):
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")
    else:
        rope_type = "default"
    if rope_type in ["default", "linear"]:
        scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)
    raise ValueError(f"Unsupported candidate Qwen3 RoPE type {rope_type}")


def create_attention_mask(h, cache=None):
    length = h.shape[1]
    if cache is not None:
        return cache.make_mask(length)
    if length == 1:
        return None
    return "causal"


def scaled_dot_product_attention(queries, keys, values, *, cache, scale, mask):
    # The adapter admits only dense, unquantized KV storage. Q8 *weights* still
    # use nn.QuantizedLinear directly; they are independent of KV quantization.
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask, sinks=None
    )
