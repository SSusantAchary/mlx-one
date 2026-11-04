#!/usr/bin/env python3
"""Trace the unchanged ViT model; optional single-operation diagnostic ablations."""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_one.vision.vit_classification import load_vit_classifier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("pixels", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument(
        "--variant",
        choices=[
            "baseline",
            "eager",
            "approx_gelu",
            "explicit_ln",
            "transposed_linear",
            "residual_reverse",
            "materialize",
        ],
        default="baseline",
    )
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    model = load_vit_classifier(args.checkpoint).model
    pixels = np.load(args.pixels, allow_pickle=False)
    trace = {"pixels": mx.array(pixels)}
    emb = model.vit.embeddings
    x = emb.patch_embeddings.projection(mx.array(pixels).transpose(0, 2, 3, 1))
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    x = (
        mx.concatenate([mx.broadcast_to(emb.cls_token, (x.shape[0], 1, x.shape[-1])), x], axis=1)
        + emb.position_embeddings
    )
    trace["layer_00"] = x

    def norm(layer, x):
        if args.variant != "explicit_ln":
            return layer(x)
        mean = mx.mean(x, axis=-1, keepdims=True)
        variance = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(variance + 1e-12) * layer.weight + layer.bias

    def linear(layer, x):
        if args.variant != "transposed_linear":
            return layer(x)
        out = x @ mx.contiguous(layer.weight.T)
        return out + layer.bias if "bias" in layer else out

    def block_run(block, x, prefix=None):
        out = {}
        out["input"] = x
        z = norm(block.layernorm_before, x)
        out["ln1"] = z
        if args.variant == "materialize":
            mx.eval(z)
        attn = block.attention
        b, n, d = z.shape
        heads = attn.heads

        def split(t):
            return t.reshape(b, n, heads, d // heads).transpose(0, 2, 1, 3)

        for name in ("query", "key", "value"):
            out[name] = linear(attn.attention[name], z)
        if args.variant == "materialize":
            mx.eval(out["query"], out["key"], out["value"])
        q, k, v = (split(out[name]) for name in ("query", "key", "value"))
        scores = (q @ k.transpose(0, 1, 3, 2)) * ((d // heads) ** -0.5)
        probs = mx.softmax(scores, axis=-1, precise=True)
        out["scores"] = scores
        out["probabilities"] = probs
        if args.variant == "eager":
            context = probs @ v
        else:
            context = mx.fast.scaled_dot_product_attention(q, k, v, scale=(d // heads) ** -0.5)
        context = context.transpose(0, 2, 1, 3).reshape(b, n, d)
        out["attention_output"] = linear(attn.output.dense, context)
        out["residual1"] = (
            out["attention_output"] + x
            if args.variant == "residual_reverse"
            else x + out["attention_output"]
        )
        if args.variant == "materialize":
            mx.eval(out["residual1"])
        out["ln2"] = norm(block.layernorm_after, out["residual1"])
        if args.variant == "materialize":
            mx.eval(out["ln2"])
        out["mlp1"] = linear(block.intermediate.dense, out["ln2"])
        out["gelu"] = (
            nn.gelu_approx(out["mlp1"]) if args.variant == "approx_gelu" else nn.gelu(out["mlp1"])
        )
        if args.variant == "materialize":
            mx.eval(out["mlp1"], out["gelu"])
        out["mlp2"] = linear(block.output.dense, out["gelu"])
        out["residual2"] = (
            out["mlp2"] + out["residual1"]
            if args.variant == "residual_reverse"
            else out["residual1"] + out["mlp2"]
        )
        if prefix:
            trace.update({prefix + k: v for k, v in out.items()})
        return out["residual2"]

    started = time.perf_counter()
    for index, layer in enumerate(model.vit.encoder.layer, 1):
        x = block_run(layer, x, f"block_{index:02}_" if index in (4, 5) else None)
        trace[f"layer_{index:02}"] = x
    x = model.vit.layernorm(x)
    logits = model.classifier(x[:, 0])
    trace.update(
        final_hidden=x, logits=logits, probabilities=mx.softmax(logits, axis=-1, precise=True)
    )
    mx.eval(*trace.values())
    print(f"trace_compute_seconds={time.perf_counter() - started:.6f}")
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as ref:
            for index in (4, 5):
                input_key = f"block_{index:02}_input"
                block_run(
                    model.vit.encoder.layer[index - 1],
                    mx.array(ref[input_key]),
                    f"isolated_block_{index:02}_",
                )
        mx.eval(*trace.values())
    with args.output.open("xb") as f:
        np.savez(f, **{k: np.asarray(v) for k, v in trace.items()})


if __name__ == "__main__":
    main()
