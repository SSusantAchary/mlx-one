#!/usr/bin/env python3
"""Compare all loaded native ViT parameters against source safetensors."""

import argparse
import json
from pathlib import Path

import numpy as np
from mlx.utils import tree_flatten
from safetensors import safe_open

from mlx_one.vision.vit_classification import load_vit_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = load_vit_classifier(args.checkpoint).model
    native = dict(tree_flatten(model.parameters()))
    rows = {}
    with safe_open(args.checkpoint / "model.safetensors", framework="numpy") as source:
        source_keys = set(source.keys())
        for name in sorted(source_keys & set(native)):
            original = source.get_tensor(name)
            actual = np.asarray(native[name])
            if name == "vit.embeddings.patch_embeddings.projection.weight":
                original = original.transpose(0, 2, 3, 1)
            rows[name] = {
                "shape": list(actual.shape),
                "dtype": str(actual.dtype),
                "exact_equal": bool(np.array_equal(original, actual)),
            }
    result = {
        "missing": sorted(source_keys - set(native)),
        "unexpected": sorted(set(native) - source_keys),
        "tensors": rows,
        "all_exact": source_keys == set(native) and all(x["exact_equal"] for x in rows.values()),
    }
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("tensor_count", len(rows), "all_exact", result["all_exact"])


if __name__ == "__main__":
    main()
