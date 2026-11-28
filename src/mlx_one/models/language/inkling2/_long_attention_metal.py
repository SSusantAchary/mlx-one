# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX v2.13.0, commit bebe21b5ba36af3688335ab66d365509fd042fad,
# models/inkling/fast/{attend,prefill_attend}.py. See MTPLX_NOTICE.txt.
"""Metal sources for the exact two-pass BF16 global attention on M4."""

from mlx_one.metal_source import metal_source

HEADER = metal_source("inkling_long_chains.metal")

PARTIALS = metal_source("inkling_long_partials.metal")

MERGE = metal_source("inkling_long_merge.metal")

def partials_source(separate_rows):
    if not separate_rows:
        return PARTIALS
    return PARTIALS.replace("// APPENDED_ROWS", """
        if (p >= pos0) {
            kp = new_k + size_t(g) * new_k_strides[1] + size_t(p - pos0) * new_k_strides[2];
            vp = new_v + size_t(g) * new_v_strides[1] + size_t(p - pos0) * new_v_strides[2];
        }
    """)
