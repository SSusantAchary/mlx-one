#!/usr/bin/env python3
"""Export matching native ViT layer traces, outside the production forward path."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_one.vision.vit_classification import load_vit_classifier, preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--processor-only", action="store_true")
    args = parser.parse_args()
    if args.processor_only:
        pixels, _ = preprocess(args.image)
        with args.output.open("xb") as stream:
            np.savez(stream, pixels=np.transpose(pixels[None], (0, 3, 1, 2)))
        return
    model = load_vit_classifier(args.checkpoint).model
    pixels, _ = preprocess(args.image)
    x = mx.array(pixels[None])
    embeddings = model.vit.embeddings
    x = embeddings.patch_embeddings.projection(x)
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    x = mx.concatenate([mx.broadcast_to(embeddings.cls_token, (x.shape[0], 1, x.shape[-1])), x], 1)
    x = x + embeddings.position_embeddings
    trace = {"pixels": np.transpose(pixels[None], (0, 3, 1, 2)), "layer_00": x}
    for index, layer in enumerate(model.vit.encoder.layer, 1):
        x = layer(x)
        trace[f"layer_{index:02}"] = x
    x = model.vit.layernorm(x)
    logits = model.classifier(x[:, 0])
    trace.update(
        final_hidden=x, logits=logits, probabilities=mx.softmax(logits, axis=-1, precise=True)
    )
    mx.eval(*[v for v in trace.values() if isinstance(v, mx.array)])
    with args.output.open("xb") as stream:
        np.savez(stream, **{name: np.asarray(value) for name, value in trace.items()})


if __name__ == "__main__":
    main()
