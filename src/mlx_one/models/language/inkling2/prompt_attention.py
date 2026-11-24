"""Opt-in SIMD prompt attention using the installed MLX Metal headers."""

import re
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from . import _prompt_attention_metal as metal
from .long_attention import _SCALE, _supported_gpu

_METAL = "mlx/backend/metal/kernels/"
_ready = {}


@lru_cache(maxsize=1)
def _kernel_header():
    root = Path(mx.__file__).resolve().parent / "include"
    seen = set()
    lines = []

    def walk(name, emit):
        if name in seen:
            return
        seen.add(name)
        for line in (root / name).read_text().splitlines():
            include = re.fullmatch(r'\s*#\s*include\s+"([^"]+)"\s*', line)
            if include:
                walk(include[1], emit)
            elif emit and line.strip() != "#pragma once":
                lines.append(line)

    try:
        # Custom Metal kernels already include utils.h.
        walk(_METAL + "utils.h", False)
        walk(_METAL + "steel/attn/mma.h", True)
    except (OSError, UnicodeError):
        return None
    headers = "\n".join(lines)
    return headers if "struct BaseMMAFrag" in headers else None


def available():
    return mx.default_device() == mx.gpu and _supported_gpu() and _kernel_header() is not None


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mlx_one_inkling_prompt_simd",
        input_names=["q", "k", "v", "prof", "sink", "params"],
        output_names=["out"],
        header=_kernel_header() + "\n" + metal.HEADER,
        source=metal.SOURCE,
        ensure_row_contiguous=False,
    )


def prompt_attention(query, keys, values, profiles, sink, window):
    """Use explicit causal positions and preserve Inkling's per-query sink."""
    batch, heads, rows, dim = query.shape
    count = keys.shape[2]
    if (not 32 < rows <= 512 or (batch, heads, dim) != (1, 24, 128)
            or keys.shape != (1, 8, count, 128) or values.shape != keys.shape
            or profiles.shape != (1, rows, 24, 1024) or sink.shape != (1, rows, 24)
            or any(x.dtype != mx.bfloat16 for x in (query, keys, values, profiles, sink))
            or not available()):
        return None
    if _ready.get(window) is False:
        return None
    try:
        result, = _kernel()(
            inputs=[query, keys, values, profiles, sink, mx.array([rows, count], mx.int32)],
            template=[("T", mx.bfloat16), ("G", 3), ("TG_M", 3), ("HQ", 24),
                      ("WINDOW", window), ("EXTENT", 1024), ("SCALE", _SCALE), ("TGBAR", True)],
            grid=(((rows + 15) // 16) * 96, 8, 1), threadgroup=(96, 1, 1),
            output_shapes=[(1, rows, 24, 128)], output_dtypes=[mx.bfloat16],
        )
        if window not in _ready:
            # Force compilation here so an unsupported kernel can use stock attention.
            _ready[window] = bool(mx.all(mx.isfinite(result)).item())
            if not _ready[window]:
                return None
        return result.transpose(0, 2, 1, 3)
    except (RuntimeError, ValueError):
        _ready[window] = False
        return None
