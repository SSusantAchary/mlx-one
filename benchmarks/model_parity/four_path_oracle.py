"""Offline full-prefix/cache oracle with materialized, independent boundaries.

This is development tooling. Diagnostic primitive substitutions are process-local,
explicitly labelled, and never qualify the production execution profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

PAIRS = (("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C"))
ROOT = Path(__file__).resolve().parents[2]


def cache_offsets(cache):
    """Reference recurrent arrays carry state without an absolute offset field."""
    return [int(c.offset) if hasattr(c, "offset") else None for c in cache]


def partitions(length):
    """Distinct valid partitions, including the empty-cache whole-prefill control."""
    result = [(length,)]
    for first in (length - 1, length - 2, 1):
        if 0 < first < length:
            result.append((first, length - first))
    for width in (4, 8):
        result.append(
            tuple([width] * (length // width) + ([length % width] if length % width else []))
        )
    return list(dict.fromkeys(result))


def snapshot(array):
    import mlx.core as mx
    import numpy as np

    mx.eval(array)
    return np.array(array.astype(mx.float32), copy=True)


def compare(left, right, dtype):
    import numpy as np

    from mlx_one.ops.numerical_trace import compare_tensors

    result = compare_tensors(left, right, dtype=dtype)
    result["rms_error"] = float(np.sqrt(np.mean((left.astype(np.float64) - right) ** 2)))
    errors = np.abs(left.astype(np.float64) - right)
    index = np.unravel_index(np.argmax(errors), errors.shape)
    result["max_error_witness"] = {
        "index": [int(i) for i in index],
        "reference": float(left[index]),
        "candidate": float(right[index]),
    }
    return result


def load_reference(bundle, precision):
    import mlx.core as mx
    import mlx.nn as nn
    from local_reference import reference_module

    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.text.loading import _read_safetensors

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("differential model qualification requires real Metal GPU execution")
    module = reference_module(bundle.architecture)
    reference_path = Path(module.__file__).resolve()
    if not reference_path.is_relative_to(ROOT / "references_libs/mlx-lm-main"):
        raise ValueError("differential worker imported an unpinned reference module")
    config = json.loads((bundle.path / "config.json").read_text())
    reference = module.Model(module.ModelArgs.from_dict(config))
    weights = _read_safetensors(bundle.path)
    if hasattr(reference, "sanitize"):
        weights = reference.sanitize(weights)
    reference.load_weights(list(weights.items()), strict=True)
    del weights
    execution = None
    if precision.startswith("affine_q"):
        bits = int(precision[-1])
        for model in (bundle.model, reference):
            nn.quantize(model, group_size=64, bits=bits, mode="affine")
        bundle = replace(bundle, quantization={"bits": bits, "group_size": 64})
        execution = "float16"
    elif precision != "stored":
        execution = precision
    configure_execution_dtype(bundle, execution)
    if execution:
        reference.set_dtype(
            getattr(mx, execution),
            predicate=lambda d: (
                d
                in {
                    mx.float32,
                    mx.float16,
                    mx.bfloat16,
                }
            ),
        )
    bundle.model.eval()
    reference.eval()
    mx.eval(bundle.model.parameters(), reference.parameters())
    return bundle, reference, module


@contextmanager
def attention_control(mode):
    """Same mathematical attention, independent FP32 reduction diagnostic only."""
    import mlx.core as mx

    original = mx.fast.scaled_dot_product_attention

    def explicit(q, k, v, *, scale, mask=None, **kwargs):
        from mlx_one.ops.attention import attention_reference

        sinks = kwargs.pop("sinks", None)
        if sinks is not None:
            raise ValueError("diagnostic attention does not implement attention sinks")
        if kwargs:
            raise ValueError(f"unsupported diagnostic attention arguments: {kwargs}")
        if mode == "contiguous_rowwise":
            q, k, v = (mx.contiguous(x) for x in (q, k, v))
        return attention_reference(
            q, k, v, scale=scale, mask=mask, stable=mode in {"rowwise", "contiguous_rowwise"}
        ).astype(q.dtype)

    try:
        if mode != "canonical":
            mx.fast.scaled_dot_product_attention = explicit
        yield
    finally:
        mx.fast.scaled_dot_product_attention = original


@contextmanager
def linear_control(mode):
    """Diagnostic only: keep a fixed FP32 row reduction for source-weight linears."""
    import mlx.core as mx
    import mlx.nn as nn

    original, embedding_original = nn.Linear.__call__, nn.Embedding.as_linear

    def linear(value, weight, bias=None):
        rows = value.reshape(-1, 1, value.shape[-1]).astype(mx.float32)
        result = mx.gather_mm(
            rows,
            weight.astype(mx.float32).T,
            lhs_indices=mx.arange(rows.shape[0]),
            rhs_indices=mx.zeros((rows.shape[0],), mx.int32),
        )
        result = result.reshape(*value.shape[:-1], weight.shape[0])
        if bias is not None:
            result += bias.astype(mx.float32)
        return result.astype(value.dtype)

    def call(module, value):
        return linear(value, module.weight, module.bias if "bias" in module else None)

    def as_linear(module, value):
        return linear(value, module.weight)

    try:
        if mode == "rowwise":
            nn.Linear.__call__, nn.Embedding.as_linear = call, as_linear
        yield
    finally:
        nn.Linear.__call__, nn.Embedding.as_linear = original, embedding_original


def first_layer_control(model, native, tokens, length, cache_factory, dtype):
    """Capture identical first-attention operands across full and split evaluation."""
    import mlx.core as mx
    import numpy as np

    from mlx_one.ops.attention import attention_reference

    original = mx.fast.scaled_dot_product_attention
    captures = []

    def capture(q, k, v, *, scale, mask=None, **kwargs):
        output = original(q, k, v, scale=scale, mask=mask, **kwargs)
        if not captures:
            captures.append((q, k, v, output, mask, scale))
            mx.eval(q, k, v, output)
        return output

    def forward(ids, cache=None):
        output = model(ids, cache=cache)
        return output.logits if native else output

    try:
        mx.fast.scaled_dot_product_attention = capture
        mx.eval(forward(tokens[:, :length]))
        full = captures.pop()
        cache = cache_factory()
        mx.eval(forward(tokens[:, : length - 1], cache))
        captures.clear()
        mx.eval(forward(tokens[:, length - 1 : length], cache))
        split = captures.pop()
    finally:
        mx.fast.scaled_dot_product_attention = original
    q, k, v, output, mask, scale = full
    sq, sk, sv, so, _, _ = split
    # Query only the last row with the FULL path's K/V. No cache is involved.
    same_operands = original(q[:, :, -1:], k, v, scale=scale, mask=mask[..., -1:, :])
    dense = attention_reference(q, k, v, scale=scale, mask=mask)
    row_dense = attention_reference(q[:, :, -1:], k, v, scale=scale, mask=mask[..., -1:, :])
    # CPU float64 independently checks the attention equation, mask and GQA layout.
    cq, ck, cv = (snapshot(x).astype(np.float64) for x in (q, k, v))
    repeats = cq.shape[1] // ck.shape[1]
    ck, cv = (np.repeat(x, repeats, axis=1) for x in (ck, cv))
    scores = cq @ ck.swapaxes(-1, -2) * scale
    cmask = snapshot(mask)
    scores = np.where(cmask, scores, -np.inf) if mask.dtype == mx.bool_ else scores + cmask
    scores -= scores.max(-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(-1, keepdims=True)
    cpu = probabilities @ cv
    return {
        "q_last": compare(snapshot(q[:, :, -1:]), snapshot(sq), dtype),
        "k": compare(snapshot(k), snapshot(sk), dtype),
        "v": compare(snapshot(v), snapshot(sv), dtype),
        "first_attention_full_vs_cached": compare(snapshot(output[:, :, -1:]), snapshot(so), dtype),
        "equal_operands_sdpa_full_vs_one_query": compare(
            snapshot(output[:, :, -1:]), snapshot(same_operands), dtype
        ),
        "equal_operands_dense_full_vs_one_query": compare(
            snapshot(dense[:, :, -1:]), snapshot(row_dense), dtype
        ),
        "cpu_float64_vs_full_sdpa": compare(cpu[:, :, -1:], snapshot(output[:, :, -1:]), dtype),
        "cpu_float64_vs_one_query_sdpa": compare(cpu[:, :, -1:], snapshot(same_operands), dtype),
        "cpu_float64_vs_dense": compare(cpu[:, :, -1:], snapshot(dense[:, :, -1:]), dtype),
        "operand_dtypes": [str(x.dtype) for x in (q, k, v, output)],
        "query_shape": list(q.shape),
        "key_shape": list(k.shape),
        "mask_shape": list(mask.shape),
        "scale": scale,
        "absolute_query_position": length - 1,
    }


def last_token_event(name, value, family):
    """Select the semantic token axis, including norms before head transpose."""
    if value.ndim == 3:
        return value[:, -1:]
    if value.ndim != 4:
        return value
    operation = name.split("#", 1)[0]
    pretranspose_norm = (
        family == "qwen3"
        and operation.endswith((".q_norm", ".k_norm"))
        or family == "lfm2"
        and operation.endswith((".q_layernorm", ".k_layernorm"))
        or family == "qwen3_5"
        and operation.endswith((".q_norm", ".k_norm", ".linear_attn.norm"))
    )
    return value[:, -1:] if pretranspose_norm else value[:, :, -1:]


def prefix_trace(model, family, native, tokens, length, cache_factory, dtype):
    from layerwise import capture_modules, compare_events

    def forward(ids, cache=None):
        import mlx.core as mx

        result = model(ids, cache=cache)
        mx.eval(result.logits if native else result)

    with capture_modules(model, family, projection_inputs=True) as full:
        forward(tokens[:, :length])
    cache = cache_factory()
    if length > 1:
        forward(tokens[:, : length - 1], cache)
    with capture_modules(model, family, projection_inputs=True) as split:
        forward(tokens[:, length - 1 : length], cache)

    def last(events):
        return [(name, last_token_event(name, x, family)) for name, x in events if x.ndim in {3, 4}]

    return compare_events(last(full), last(split), dtype)


def run(checkpoint, precision, lengths, mode, trace=False, linear_mode="canonical"):
    import mlx.core as mx
    import numpy as np
    from local_reference import reference_module

    from mlx_one.server.model_manager import ModelManager

    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    manager = ModelManager()
    bundle = manager.load(checkpoint, offline=True)

    def compute():
        nonlocal bundle
        bundle, reference, source = load_reference(bundle, precision)
        seed = bundle.tokenizer.encode("The capital of France is")
        count = max(lengths)
        ids = (seed * ((count + len(seed) - 1) // len(seed)))[:count]
        tokens = mx.array([ids])
        factories = {
            "native": bundle.model.make_cache,
            "reference": lambda: reference_module("cache").make_prompt_cache(reference),
        }
        models = {"native": bundle.model, "reference": reference}
        dtype = None
        observed_dtypes = {which: set() for which in models}

        def forward(which, inputs, cache=None):
            nonlocal dtype
            output = models[which](inputs, cache=cache)
            logits = output.logits if which == "native" else output
            dtype = str(logits.dtype).rsplit(".", 1)[-1]
            observed_dtypes[which].add(dtype)
            return snapshot(logits)

        paths = {key: [] for key in "ABCD"}
        caches = {key: factory() for key, factory in factories.items()}
        with attention_control(mode), linear_control(linear_mode):
            for position in range(count):
                for which, full, cached in (("native", "A", "C"), ("reference", "B", "D")):
                    # Explicit cache=None; every prefix starts at absolute position zero.
                    paths[full].append(forward(which, tokens[:, : position + 1])[:, -1:])
                    paths[cached].append(
                        forward(which, tokens[:, position : position + 1], caches[which])[:, -1:]
                    )
            paths = {key: np.concatenate(value, axis=1) for key, value in paths.items()}
            pairwise = {}
            for left, right in PAIRS:
                positions = [
                    compare(paths[left][:, i : i + 1], paths[right][:, i : i + 1], dtype)
                    for i in range(count)
                ]
                pairwise[f"{left}_{right}"] = {
                    "all_positions": compare(paths[left], paths[right], dtype),
                    "first_nonzero_position": next(
                        (i + 1 for i, r in enumerate(positions) if r["changed_elements"]), None
                    ),
                    "first_failure_position": next(
                        (i + 1 for i, r in enumerate(positions) if r["status"] != "PASS"), None
                    ),
                    "positions": positions,
                }
            chunks = []
            for length in lengths:
                for partition in partitions(length):
                    row = {"length": length, "partition": partition}
                    for which, full in (("native", "A"), ("reference", "B")):
                        cache, start, outputs = factories[which](), 0, []
                        for width in partition:
                            outputs.append(forward(which, tokens[:, start : start + width], cache))
                            start += width
                        row[which] = compare(
                            paths[full][:, :length], np.concatenate(outputs, axis=1), dtype
                        )
                        row[which + "_offsets"] = cache_offsets(cache)
                    chunks.append(row)
        control = None
        traces = None
        cross_trace = None
        cross_trace_length = None
        if trace and mode == linear_mode == "canonical":
            from layerwise import capture_modules, compare_events

            cross_trace_length = pairwise["A_B"]["first_nonzero_position"] or count
            with capture_modules(
                reference, bundle.architecture, projection_inputs=True
            ) as ref_events:
                mx.eval(reference(tokens[:, :cross_trace_length]))
            with capture_modules(
                bundle.model, bundle.architecture, projection_inputs=True
            ) as native_events:
                mx.eval(bundle.model(tokens[:, :cross_trace_length]).logits)
            cross_trace = compare_events(ref_events, native_events, dtype)
        if trace and bundle.architecture in {"gemma3_text", "phi3"}:
            first = pairwise["A_C"]["first_nonzero_position"] or count
            first = max(first, 2)
            if mode == linear_mode == "canonical":
                control = first_layer_control(
                    bundle.model, True, tokens, first, factories["native"], dtype
                )
            with attention_control(mode), linear_control(linear_mode):
                traces = {
                    which: prefix_trace(
                        model,
                        bundle.architecture,
                        which == "native",
                        tokens,
                        max(2, pairwise[pair]["first_nonzero_position"] or count),
                        factories[which],
                        dtype,
                    )
                    for which, model, pair in (
                        ("native", bundle.model, "A_C"),
                        ("reference", reference, "B_D"),
                    )
                }
        return {
            "schema_version": "four-path-oracle-v1",
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "architecture": bundle.architecture,
            "precision": precision,
            "actual_logits_dtype": dtype,
            "actual_logits_dtypes": {key: sorted(value) for key, value in observed_dtypes.items()},
            "attention_mode": mode,
            "linear_mode": linear_mode,
            "qualification_eligible": mode == linear_mode == "canonical",
            "token_ids": ids,
            "lengths": lengths,
            "bos_added": False,
            "position_contract": "zero-based absolute positions, explicit cache=None for A/B",
            "offset_metadata_contract": (
                "None means the reference ArraysCache has no absolute offset attribute; "
                "the explicit partition cursor tracks tokens and KV layers retain RoPE offsets."
            ),
            "snapshot_contract": "mx.eval followed by independent NumPy copy before cache update",
            "paths": {
                "A": "native fresh full prefix",
                "B": "reference fresh full prefix",
                "C": "native single-token cache",
                "D": "reference single-token cache",
            },
            "pairwise": pairwise,
            "chunks": chunks,
            "first_layer_control": control,
            "first_divergence_trace": traces,
            "cross_implementation_trace": cross_trace,
            "cross_trace_prefix_length": cross_trace_length,
            "reference_source": {
                "path": source.__file__,
                "sha256": hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
            },
            "status": "PASS"
            if all(r["all_positions"]["status"] == "PASS" for r in pairwise.values())
            and all(
                row[which]["status"] == "PASS"
                for row in chunks
                for which in ("native", "reference")
            )
            else "FAIL_NUMERICAL",
        }

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--precision", default="stored")
    parser.add_argument("--lengths", default="1,2,3,7,15,16,17,32,64")
    parser.add_argument(
        "--attention",
        choices=("canonical", "dense", "rowwise", "contiguous_rowwise"),
        default="canonical",
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--linear", choices=("canonical", "rowwise"), default="canonical")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new evidence file")
    report = run(
        args.checkpoint,
        args.precision,
        [int(n) for n in args.lengths.split(",")],
        args.attention,
        args.trace,
        args.linear,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["checkpoint"], report["precision"], report["status"], flush=True)


if __name__ == "__main__":
    main()
