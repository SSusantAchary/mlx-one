# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX v2.13.0, commit bebe21b5ba36af3688335ab66d365509fd042fad.
# See MTPLX_NOTICE.txt and MTPLX_LICENSE.txt.
# Sources: mtplx/models/inkling/fast/{affine_lane,qproduct}.py.
"""One-row affine arithmetic shared across a draft verification panel."""

from mlx_one.metal_source import metal_source

HEADER = metal_source("inkling_verify_affine_helpers.metal")
