#!/usr/bin/env python3
"""Run only in the pinned reference environment; creates tiny random fixtures."""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from turboquant_mlx.core.polar_quantize import polar_quantize_weight
from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for inputs in (64, 2880, 4096):
        rng = np.random.default_rng(82)
        weight = mx.array(rng.normal(0, 0.1, (3, inputs)).astype(np.float32))
        x = mx.array(rng.normal(0, 1, (2, inputs)).astype(np.float16))
        data = polar_quantize_weight(weight, bits=3, group_size=64, seed=42)
        layer = PolarQuantizedLinear(inputs, 3)
        layer.weight = data["packed_weight"]
        layer.scales = data["scales"]
        layer.signs = data["signs"]
        layer.codebook = data["codebook"]
        y = layer(x)
        mx.eval(y)
        np.savez(
            args.output / f"reference-generated-{inputs}.npz",
            weight=np.array(layer.weight),
            scales=np.array(layer.scales),
            codebook=np.array(layer.codebook),
            signs=np.array(layer.signs),
            x=np.array(x),
            expected=np.array(y),
            original_weight=np.array(weight),
        )


if __name__ == "__main__":
    main()
