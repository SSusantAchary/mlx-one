# Copyright © 2026 Neywa Labs Inc, Abdur Rahim
# SPDX-License-Identifier: MIT
"""Adapt Neywa's grouped expert QMV to MLX-One's FP32 NVFP4 operands.

Source: mlx-vlm, mlx_vlm/models/inkling2/grouped_experts.py at
8e80d434fe59f4d4e00bc91a418e0a16ad951326. See NEYWA_LICENSE.txt.
MLX-One keeps the interleaved gate/up matrix and applies the outer scales
after the raw product, at the same precision as its native gather path.
"""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from mlx_one.metal_source import metal_source

from .long_attention import _supported_gpu

_TILE = metal_source("inkling_expert_grouped_helpers.metal")


@lru_cache(maxsize=1)
def _kernel():
    headers = Path(mx.__file__).resolve().parent / "include/mlx/backend/metal/kernels"
    # Use the installed MLX decoders for its exact NVFP4 and E4M3 representation.
    fp_headers = "\n".join((headers / name).read_text().replace("#pragma once", "")
                           for name in ("fp4.h", "fp8.h"))
    cases = "\n".join(
        f"case {width}: expert_tile<{width}>(x, w, s, out, src, dst, expert, row, lane, K, N);"
        " break;"
        for width in range(1, 5)
    )
    return mx.fast.metal_kernel(
        name="mlx_one_inkling_grouped_nvfp4",
        input_names=["x", "w", "s", "experts", "source", "destination"],
        output_names=["out"],
        header=fp_headers + "\n" + _TILE,
        compile_options={"math_mode": "safe"},
        source=metal_source("inkling_expert_grouped_projection.metal") + cases + "}}",
    )


def raw_product(inputs, codes, scales, experts, source, destination):
    """Read sorted expert runs and write FP32 products in the original slot order."""
    if (inputs.ndim != 3 or inputs.shape[1] != 1 or inputs.dtype != mx.float32
            or codes.ndim != 3 or codes.dtype != mx.uint32 or scales.dtype != mx.uint8
            or experts.ndim != 1 or not 16 <= experts.size <= 32
            or any(x.shape != experts.shape or x.dtype != mx.uint32
                   for x in (experts, source, destination))
            or mx.default_device() != mx.gpu or not _supported_gpu()):
        return None
    width = inputs.shape[-1]
    banks, outputs, words = codes.shape
    if (width not in (1152, 2304) or (banks, outputs) != (128, 2304)
            or words != width // 8 or scales.shape != (banks, outputs, width // 16)):
        return None
    try:
        kernel = _kernel()
    except OSError:
        return None
    return kernel(
        inputs=[inputs.reshape(-1, width), codes, scales, experts, source, destination],
        grid=(32, outputs // 4, experts.size),
        threadgroup=(32, 2, 1),
        output_shapes=[(experts.size, 1, outputs)],
        output_dtypes=[mx.float32],
    )[0]
