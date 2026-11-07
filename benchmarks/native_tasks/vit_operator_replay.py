#!/usr/bin/env python3
"""Replay each selected operation with the exact reference input, independently."""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_one.vision.vit_classification import load_vit_classifier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("reference", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    model = load_vit_classifier(args.checkpoint).model
    trace = {}
    with np.load(args.reference, allow_pickle=False) as ref:
        for index in (4, 5):
            block = model.vit.encoder.layer[index - 1]
            prefix = f"block_{index:02}_"

            def value(name, prefix=prefix):
                return mx.array(ref[prefix + name])

            def put(name, x, prefix=prefix):
                trace[prefix + name] = x

            put("ln1", block.layernorm_before(value("input")))
            for name in ("query", "key", "value"):
                put(name, block.attention.attention[name](value("ln1")))

            def split(x):
                return x.reshape(x.shape[0], x.shape[1], 12, 64).transpose(0, 2, 1, 3)

            q, k = split(value("query")), split(value("key"))
            put("scores", (q @ k.transpose(0, 1, 3, 2)) * (64**-0.5))
            put("probabilities", mx.softmax(value("scores"), axis=-1, precise=True))
            put("attention_output", block.attention.output.dense(value("attention_context")))
            put("residual1", value("attention_output") + value("input"))
            put("ln2", block.layernorm_after(value("residual1")))
            put("mlp1", block.intermediate.dense(value("ln2")))
            put("gelu", nn.gelu(value("mlp1")))
            put("mlp2", block.output.dense(value("gelu")))
            put("residual2", value("mlp2") + value("residual1"))
    mx.eval(*trace.values())
    with args.output.open("xb") as f:
        np.savez(f, **{k: np.asarray(v) for k, v in trace.items()})


if __name__ == "__main__":
    main()
