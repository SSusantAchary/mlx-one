"""Exact Qwen3.5 source operands: convolution, decay and recurrent-state controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from four_path_oracle import compare, load_reference, snapshot


def run(path, precision="stored"):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from mlx_one.models.vision_language.qwen3_5.model import (
        _gated_delta_decay,
    )
    from mlx_one.models.vision_language.qwen3_5.reference_delta import normalize_qk
    from mlx_one.server.model_manager import ModelManager

    manager = ModelManager()
    bundle = manager.load(path, offline=True)

    def compute():
        nonlocal bundle
        bundle, _, _ = load_reference(bundle, precision)
        def projection(module, value):
            return module(value)

        layer = bundle.model.language_model.layers[0]
        linear = layer.linear_attn
        seed = bundle.tokenizer.encode("The capital of France is")
        hidden = layer.input_layernorm(
            bundle.model.language_model.embed_tokens(mx.array([(seed * 16)[:16]]))
        )
        projected = projection(linear.in_proj_qkv, hidden)
        conv = linear.conv1d
        padded = mx.pad(projected, ((0, 0), (conv.kernel_size - 1, 0), (0, 0)))
        rounded = mx.stack(
            [
                mx.sum(padded[:, i : i + conv.kernel_size] * conv.weight.T[None], axis=1)
                for i in range(16)
            ],
            axis=1,
        )
        corrected = conv(projected, None)
        primitive = nn.Conv1d(
            conv.weight.shape[0],
            conv.weight.shape[0],
            conv.kernel_size,
            groups=conv.weight.shape[0],
            bias=False,
        )
        primitive.weight = conv.weight[..., None]
        stock = primitive(padded)
        cx, cw = snapshot(padded).astype(np.float64), snapshot(conv.weight).astype(np.float64)
        cpu_conv = np.stack(
            [(cx[:, i : i + conv.kernel_size] * cw.T[None]).sum(1) for i in range(16)], axis=1
        )
        a = projection(linear.in_proj_a, hidden)
        shifted = a.astype(mx.float32) + linear.dt_bias
        old_decay = mx.exp(-mx.exp(linear.A_log) * mx.logaddexp(shifted, mx.zeros_like(shifted)))
        decay = _gated_delta_decay(a, linear.A_log, linear.dt_bias)
        cpu_decay = np.exp(
            -np.exp(snapshot(linear.A_log).astype(np.float64))
            * np.logaddexp(
                snapshot(a).astype(np.float64) + snapshot(linear.dt_bias).astype(np.float64), 0
            )
        )
        mixed = nn.silu(corrected)
        q, k, v = mx.split(mixed, [linear.key_dim, 2 * linear.key_dim], axis=-1)
        shape = (1, 16, linear.num_k_heads, linear.head_k_dim)
        repeats = linear.num_v_heads // linear.num_k_heads
        q, k = normalize_qk(
            q.reshape(shape), k.reshape(shape), inv_scale=linear.head_k_dim**-0.5, eps=1e-6
        )
        q, k = mx.repeat(q, repeats, axis=2), mx.repeat(k, repeats, axis=2)
        v = v.reshape(1, 16, linear.num_v_heads, linear.head_v_dim)
        beta = mx.sigmoid(projection(linear.in_proj_b, hidden))
        cq, ck, cv, cb, cd = (snapshot(x).astype(np.float64) for x in (q, k, v, beta, decay))
        state = np.zeros((1, linear.num_v_heads, linear.head_k_dim, linear.head_v_dim))
        for i in range(16):
            state *= cd[:, i, :, None, None]
            memory = (state * ck[:, i, :, :, None]).sum(-2)
            delta = (cv[:, i] - memory) * cb[:, i, :, None]
            state += ck[:, i, :, :, None] * delta[:, :, None, :]
        cache = bundle.model.make_cache()[0]
        mx.eval(linear(hidden, cache=cache, attention_mask=None))
        dtype = str(projected.dtype).rsplit(".", 1)[-1]
        return {
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "precision": precision,
            "layer": 0,
            "input_dtype": str(hidden.dtype),
            "conv_weight_dtype": str(conv.weight.dtype),
            "a_log_dtype": str(linear.A_log.dtype),
            "decay_dtype": str(decay.dtype),
            "recurrent_state_dtype": str(cache.recurrent_state.dtype),
            "conv_old_vs_cpu": compare(cpu_conv, snapshot(rounded), dtype),
            "conv_corrected_vs_cpu": compare(cpu_conv, snapshot(corrected), dtype),
            "conv_corrected_vs_stock": compare(snapshot(stock), snapshot(corrected), dtype),
            "decay_old_vs_cpu": compare(cpu_decay, snapshot(old_decay), "float32"),
            "decay_corrected_vs_cpu": compare(cpu_decay, snapshot(decay), "float32"),
            "recurrent_state_vs_cpu": compare(state, snapshot(cache.recurrent_state), "float32"),
            "cache_offset": cache.offset,
            "scope": "First source-weight GDN layer, identical evaluated projection/gate operands; "
            "CPU float64 independently computes convolution, decay and state transition.",
            "qualification": "COMPONENT_DIAGNOSTIC_ONLY",
        }

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--precision", default="stored")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    report = run(args.checkpoint, args.precision)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["checkpoint"], report["recurrent_state_vs_cpu"]["status"], flush=True)


if __name__ == "__main__":
    main()
