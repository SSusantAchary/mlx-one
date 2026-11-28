# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX v2.13.0, commit bebe21b5ba36af3688335ab66d365509fd042fad.
# See MTPLX_NOTICE.txt and MTPLX_LICENSE.txt.
# Source: mtplx/models/inkling/fast/prompt_flash.py.
# Reduction order follows MLX v0.32.2 Steel SDPA. See mlxboost/native/NOTICE.txt.
"""Simdgroup attention for prompt chunks on Apple M4."""

from mlx_one.metal_source import metal_source

HEADER = metal_source("inkling_prompt_header.metal")

SOURCE = metal_source("inkling_prompt_attention.metal")
