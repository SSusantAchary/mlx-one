#!/usr/bin/env python3
"""Synchronized synthetic operator benchmark; run each backend in its own process."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from oracle import fixture


def measure(fn, repetitions, flops):
    mx.synchronize()
    start = time.perf_counter()
    first = fn()
    mx.eval(first)
    mx.synchronize()
    cold = time.perf_counter() - start
    for _ in range(3):
        mx.eval(fn())
    mx.synchronize()
    del first
    mx.clear_cache()
    active = mx.get_active_memory()
    mx.reset_peak_memory()
    elapsed, graphs = [], []
    for _ in range(repetitions):
        start = time.perf_counter()
        out = fn()
        built = time.perf_counter()
        mx.eval(out)
        mx.synchronize()
        elapsed.append(time.perf_counter() - start)
        graphs.append(built - start)
        del out
    median = statistics.median(elapsed)
    return dict(
        median_ms=median * 1000,
        p95_ms=float(np.percentile(elapsed, 95)) * 1000,
        graph_build_median_ms=statistics.median(graphs) * 1000,
        first_call_including_compile_ms=cold * 1000,
        effective_gflops=flops / median / 1e9 if flops else None,
        baseline_active_bytes=active,
        peak_increment_bytes=max(0, mx.get_peak_memory() - active),
        process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        samples_seconds=elapsed,
    )


def layers(backend):
    if backend == "native":
        from mlx_one.quantization.turboquant.experts import TQ3ExpertLinear
        from mlx_one.quantization.turboquant.linear import TQ3Linear, rotate_input

        return TQ3Linear, TQ3ExpertLinear, rotate_input
    from turboquant_mlx.core.rotation import rotate_input
    from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear
    from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear

    def dense(inputs, outputs, params):
        layer = PolarQuantizedLinear(inputs, outputs, bias="bias" in params)
        for key, value in params.items():
            setattr(layer, key, value)
        return layer

    def expert(inputs, outputs, params):
        layer = PolarQuantizedSwitchLinear(
            inputs, outputs, params["weight"].shape[0], bias="bias" in params
        )
        for key, value in params.items():
            setattr(layer, key, value)
        return layer

    return dense, expert, rotate_input


def block(backend, profile):
    """Actual native/reference GPT layer with identical fabricated packed parameters."""
    if profile == "gpt":
        d, f, h, kv, hd, e = 2880, 2880, 64, 8, 64, 32
    else:
        d, f, h, kv, hd, e = 128, 128, 4, 2, 32, 8
    config = dict(
        model_type="gpt_oss",
        hidden_size=d,
        intermediate_size=f,
        head_dim=hd,
        num_attention_heads=h,
        num_key_value_heads=kv,
        num_hidden_layers=1,
        num_local_experts=e,
        num_experts_per_tok=4,
        vocab_size=128,
        sliding_window=128,
        layer_types=["full_attention"],
        rope_scaling=dict(
            rope_type="yarn", factor=32, original_max_position_embeddings=4096, truncate=False
        ),
    )
    if backend == "native":
        from mlx_one.models.language.gpt_oss.config import GptOssConfig
        from mlx_one.models.language.gpt_oss.model import DecoderLayer

        actual_layer = DecoderLayer(GptOssConfig.from_dict(config), 0)
    else:
        from mlx_lm.models.gpt_oss import ModelArgs, TransformerBlock

        actual_layer = TransformerBlock(ModelArgs.from_dict(config))
    import mlx.nn as nn

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = [actual_layer]
    dense, expert, _ = layers(backend)
    rng = np.random.default_rng(21)
    updates = []
    for path, module in list(model.named_modules()):
        if not path.startswith("model.layers.0."):
            continue
        if "weight" in module and module.weight.ndim in (2, 3) and not path.endswith("router"):
            shape = module.weight.shape
            params = {
                k: mx.array(v)
                for k, v in fixture(
                    shape[-2], shape[-1], experts=shape[0] if len(shape) == 3 else None
                ).items()
            }
            if "bias" in module:
                params["bias"] = mx.array(rng.normal(0, 0.01, shape[:-1]).astype(np.float16))
            updates.append(
                (path, (expert if len(shape) == 3 else dense)(shape[-1], shape[-2], params))
            )
    from mlx.utils import tree_unflatten

    model.update_modules(tree_unflatten(updates))
    # Identical nonpacked layer parameters in both implementations.
    for path, module in model.named_modules():
        if path.startswith("model.layers.0.") and not hasattr(module, "codebook"):
            for name in ("weight", "bias", "sinks"):
                if name in module and isinstance(module[name], mx.array):
                    shape = module[name].shape
                    if len(shape) == 1 and name == "weight":
                        module[name] = mx.ones(shape, mx.float16)
                    else:
                        module[name] = mx.array(rng.normal(0, 0.01, shape).astype(np.float16))
    mx.eval(model.parameters())
    layer = model.model.layers[0]
    fn = (lambda x: layer(x)) if backend == "native" else (lambda x: layer(x, mask="causal"))
    return layer, fn, d, e


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "reference"], required=True)
    parser.add_argument("--profile", choices=["small", "gpt"], default="small")
    parser.add_argument("--rows", default="1,2,8,16,32,64,128,256")
    parser.add_argument("--operations", default="dense,expert,router_expert,moe,decoder")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("at least two repetitions required")
    rows = [int(x) for x in args.rows.split(",")]
    if min(rows) < 1:
        parser.error("positive row counts required")
    dense, expert, _ = layers(args.backend)
    operations = set(args.operations.split(","))
    shapes = (
        [(128, 128), (67, 256)]
        if args.profile == "small"
        else [(4096, 2880), (512, 2880), (2880, 4096), (2880, 2880)]
    )
    results = []
    for o, k in shapes:
        if "dense" not in operations:
            break
        data = fixture(o, k)
        params = {key: mx.array(value) for key, value in data.items()}
        layer = dense(k, o, params)
        mx.eval(layer.parameters())
        for n in rows:
            x = mx.array(np.random.default_rng(19).normal(size=(n, k)).astype(np.float16))
            mx.eval(x)
            metrics = measure(lambda layer=layer, x=x: layer(x), args.repetitions, 2 * n * k * o)
            results.append(
                dict(
                    operation="dense",
                    shape=[n, o, k],
                    fixture_sha256=hashlib.sha256(data["weight"].tobytes()).hexdigest(),
                    **metrics,
                )
            )
        del layer, params, data
        mx.clear_cache()
    if "expert" in operations:
        o, k, e = (128, 128, 8) if args.profile == "small" else (2880, 2880, 32)
        params = {key: mx.array(value) for key, value in fixture(o, k, experts=e).items()}
        layer = expert(k, o, params)
        mx.eval(layer.parameters())
        for n in rows:
            indices = mx.array(np.random.default_rng(14).integers(0, e, (1, n, 4), dtype=np.uint32))
            for shared in (True, False):
                shape = (1, n, 1, 1, k) if shared else (1, n, 4, 1, k)
                x = mx.array(np.random.default_rng(15).normal(size=shape).astype(np.float16))
                mx.eval(x, indices)
                metrics = measure(
                    lambda layer=layer, x=x, indices=indices: layer(x, indices),
                    args.repetitions,
                    2 * n * 4 * k * o,
                )
                results.append(
                    dict(
                        operation="expert_shared" if shared else "expert_multi",
                        shape=[n, o, k, e, 4],
                        **metrics,
                    )
                )
        del layer, params
        mx.clear_cache()
    if operations & {"router_expert", "moe", "decoder"}:
        layer, decoder, d, e = block(args.backend, args.profile)
        for n in rows:
            x = mx.array(np.random.default_rng(18).normal(size=(1, n, d)).astype(np.float16))
            mx.eval(x)
            for name, fn in [
                ("router_expert", lambda x=x: router_projection(layer, x)),
                ("moe", lambda x=x: layer.mlp(x)),
                ("decoder", lambda x=x: decoder(x)),
            ]:
                if name in operations:
                    results.append(
                        dict(operation=name, shape=[n, d, e, 4], **measure(fn, args.repetitions, 0))
                    )
    report = dict(
        schema_version="tq3-operators-v1",
        backend=args.backend,
        profile=args.profile,
        platform=platform.platform(),
        mlx=importlib.metadata.version("mlx"),
        device=mx.device_info(),
        rows=rows,
        repetitions=args.repetitions,
        reference_commit="54443c8b83424adb5dac7e6ffc7ac2d91fe88af1",
        measurement=(
            "fresh process; FP16 synthetic; 3 warmups; "
            "synchronized wall clock incl graph dispatch; no KV state"
        ),
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), cases=len(results), backend=args.backend)))


def router_projection(layer, x):
    logits = layer.mlp.router(x)
    idx = mx.argpartition(logits, kth=-4, axis=-1)[..., -4:]
    return layer.mlp.experts.gate_proj(mx.expand_dims(x, (-2, -3)), idx)


if __name__ == "__main__":
    main()
