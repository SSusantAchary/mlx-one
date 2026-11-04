#!/usr/bin/env python3
"""Isolated Torch interpolation probes for the native depth resize implementation."""

import argparse
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
rng = np.random.default_rng(20261103)
cases = {}
for shape, size, mode, align in [
    ((1, 3, 5, 7), (8, 11), "bilinear", True),
    ((1, 3, 5, 7), (3, 4), "bilinear", False),
    ((1, 3, 5, 7), (8, 11), "bicubic", False),
    ((1, 3, 37, 37), (25, 37), "bicubic", False),
]:
    key = f"{shape[-2]}x{shape[-1]}-{size[0]}x{size[1]}-{mode}-{align}"
    value = rng.normal(size=shape).astype(np.float32)
    out = torch.nn.functional.interpolate(
        torch.from_numpy(value), size=size, mode=mode, align_corners=align
    )
    cases[key + "_input"], cases[key + "_output"] = value, out.numpy()
with args.output.open("xb") as stream:
    np.savez(stream, **cases)
