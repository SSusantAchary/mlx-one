"""Bounded long-context numerical sensitivity and immutable task evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sample(value):
    import mlx.core as mx

    mx.eval(value)
    return np.array(value.astype(mx.float32)).astype(np.float64)


def sensitivity(left, right, target, dtype, policy):
    from mlx_one.ops.numerical_trace import compare_tensors

    a, b = left.ravel(), right.ravel()
    delta = b - a
    # Remove a common shift: softmax probabilities and all odds are invariant to it.
    centered = delta - (delta.max() + delta.min()) / 2
    error = float(np.abs(centered).max())

    def logsoftmax(x):
        x = x - x.max()
        return x - np.log(np.exp(x).sum())

    la, lb = logsoftmax(a), logsoftmax(b)
    pa, pb = np.exp(la), np.exp(lb)
    top = np.argsort(a)[-2:]
    margin = float(a[top[-1]] - a[top[-2]])
    same = int(a.argmax()) == int(b.argmax())
    budget = policy["precision_budgets"][dtype]
    kl = max(0.0, float(np.sum(pa * (la - lb))))
    return {
        **compare_tensors(left, right, dtype=dtype),
        "rms_logit_error": float(np.sqrt(np.mean(delta**2))),
        "max_centered_logit_error": error,
        "mean_centered_abs_error": float(np.abs(centered).mean()),
        "top1_agreement": same,
        "top1_ids": [int(a.argmax()), int(b.argmax())],
        "top5_overlap": len(set(np.argsort(a)[-5:]) & set(np.argsort(b)[-5:])) / 5,
        "top_two_margin": margin,
        "high_margin_flip": not same and margin > 2 * error,
        "kl_nats": kl,
        "total_variation": float(np.abs(pa - pb).sum() / 2),
        "fixture_target": target,
        "nll": [-float(la[target]), -float(lb[target])],
        "nll_difference": float(la[target] - lb[target]),
        "proposed_status": "PASS"
        if error <= budget["max_centered_logit"]
        and kl <= budget["max_kl_nats"]
        and (same or margin <= 2 * error)
        and np.isfinite(right).all()
        else "FAIL",
    }


def run(args, save):
    import mlx.core as mx
    from four_path_oracle import load_reference
    from local_reference import reference_module

    from mlx_one.core.cache import configure_kv_caches
    from mlx_one.engine.reporting import tensor_dtypes
    from mlx_one.server.model_manager import ModelManager

    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    manager = ModelManager()
    bundle = manager.load(args.checkpoint, offline=True)
    report = {
        "timestamp": time.time(),
        "checkpoint": bundle.model_id,
        "revision": bundle.revision,
        "precision": args.precision,
        "mode": "reference_dense",
        "batch_size": 1,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src/mlx_one").rglob("*.py")
        },
        "rows": [],
    }
    try:
        bundle, reference, module = manager._pool.executor.call(
            bundle.model_id, lambda: load_reference(bundle, args.precision)
        )
        assert Path(module.__file__).resolve().is_relative_to(ROOT / "references_libs/mlx-lm-main")
        report["reference"] = {
            "file": module.__file__,
            "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        }
        # Quality scoring uses the legacy "reference" key for task results.
        # Keep source provenance independent so that scoring cannot replace it.
        report["reference_source"] = dict(report["reference"])
        if args.mode == "quality":
            return quality(bundle, reference, manager, args, report, save)
        policy = json.loads(args.policy.read_text())
        dtype = (
            "float16"
            if args.precision.startswith("affine")
            else next(
                x
                for x in tensor_dtypes(bundle.model.parameters())
                if x in policy["precision_budgets"]
            )
        )
        report["gate_dtype"] = dtype
        report["policy_sha256"] = hashlib.sha256(args.policy.read_bytes()).hexdigest()
        seed = bundle.tokenizer.encode("The red book was on the table. The city is calm. ")
        corpus = (seed * 500)[:2050]
        report["input_ids"] = corpus
        report["state_profile"] = "FP16 KV; architecture-owned recurrent/conv state"
        report["semantic_audit"] = []
        report["attention_visibility_audit"] = []
        if bundle.architecture == "gpt2":
            windows = [None] * len(bundle.model.transformer.h)
        elif bundle.architecture == "lfm2":
            windows = [None for layer in bundle.model.model.layers if layer.is_attention_layer]
        elif bundle.architecture == "gemma3_text":
            windows = [
                bundle.model.config.sliding_window if kind == "sliding_attention" else None
                for kind in bundle.model.config.layer_types
            ]
        else:
            windows = [getattr(bundle.model.config, "sliding_window", None)] * len(
                bundle.model.model.layers
            )

        class StateAudit:
            def __init__(self, owner, layer):
                self.owner, self.layer, self.previous = owner, layer, None

            def __getattr__(self, name):
                return getattr(self.owner, name)

            def update(self, *values):
                before = self.owner.offset
                if len(values) == 1:
                    incoming = values[0]
                    history = self.owner.kernel_size - 1
                    prefix = self.previous
                    if prefix is None:
                        prefix = mx.zeros((1, history, incoming.shape[-1]), dtype=incoming.dtype)
                    expected = mx.concatenate((prefix, incoming), axis=1)
                    returned = self.owner.update(incoming)
                    exact = bool(mx.array_equal(returned, expected))
                    self.previous = expected[:, -history:] if history else expected[:, :0]
                    exact &= bool(mx.array_equal(self.owner.values, self.previous))
                    length, elements = incoming.shape[1], expected.size
                else:
                    stored = tuple(x.astype(mx.float16) for x in values)
                    expected = (
                        stored
                        if self.previous is None
                        else tuple(
                            mx.concatenate((old, new), axis=2)
                            for old, new in zip(self.previous, stored, strict=True)
                        )
                    )
                    returned = self.owner.update(*values)
                    exact = all(
                        bool(mx.array_equal(a, b)) for a, b in zip(returned, expected, strict=True)
                    )
                    window = getattr(self.owner, "max_size", None)
                    self.previous = (
                        tuple(x[:, :, -window:] for x in expected) if window else expected
                    )
                    length, elements = values[0].shape[2], sum(x.size for x in expected)
                assert exact and self.owner.offset == before + length
                report["semantic_audit"].append(
                    {
                        "layer": self.layer,
                        "offset_before": before,
                        "input_tokens": length,
                        "offset_after": self.owner.offset,
                        "exact_logical_elements_compared": elements,
                        "status": "PASS",
                    }
                )
                return returned

        class StorageView:
            def __init__(self, owner):
                self.owner = owner

            def __getattr__(self, name):
                return getattr(self.owner, name)

            def update_and_fetch(self, k, v):
                dtype = k.dtype
                k, v = self.owner.update_and_fetch(k.astype(mx.float16), v.astype(mx.float16))
                return k.astype(dtype), v.astype(dtype)

        def forward(native, n, parts, cached):
            model = bundle.model if native else reference
            cache = None
            if cached:
                cache = (
                    configure_kv_caches(model.make_cache(), "f16", "f16")
                    if native
                    else [
                        StorageView(c) if hasattr(c, "update_and_fetch") else c
                        for c in reference_module("cache").make_prompt_cache(model)
                    ]
                )
                if native:
                    cache = tuple(StateAudit(c, layer) for layer, c in enumerate(cache))
            begin = 0
            for width in parts:
                if cache is not None and native:
                    assert all(c.offset == begin for c in cache)
                original_attention = mx.fast.scaled_dot_product_attention
                attention_index = 0

                def checked_attention(
                    q,
                    k,
                    v,
                    *,
                    scale,
                    mask=None,
                    begin=begin,
                    width=width,
                    original_attention=original_attention,
                    **kwargs,
                ):
                    nonlocal attention_index
                    window = windows[attention_index]
                    attention_index += 1
                    query = mx.arange(begin, begin + width)[:, None]
                    key = mx.arange(begin + width - k.shape[2], begin + width)[None, :]
                    expected = key <= query
                    if window is not None:
                        expected &= key > query - window
                    if mask is None:
                        observed = mx.ones(expected.shape, dtype=mx.bool_)
                    elif isinstance(mask, str):
                        assert mask == "causal"
                        observed = mx.arange(k.shape[2])[None, :] <= (
                            mx.arange(width)[:, None] + k.shape[2] - width
                        )
                    else:
                        observed = mask if mask.dtype == mx.bool_ else mx.isfinite(mask)
                    shape = (1, q.shape[1], width, k.shape[2])
                    exact = bool(
                        mx.array_equal(
                            mx.broadcast_to(observed, shape), mx.broadcast_to(expected, shape)
                        )
                    )
                    assert exact
                    report["attention_visibility_audit"].append(
                        {
                            "layer_attention_index": attention_index - 1,
                            "query_position_range": [begin, begin + width - 1],
                            "key_position_range": [begin + width - k.shape[2], begin + width - 1],
                            "window": window,
                            "visible_positions_exact": exact,
                            "status": "PASS",
                        }
                    )
                    return original_attention(q, k, v, scale=scale, mask=mask, **kwargs)

                if native:
                    mx.fast.scaled_dot_product_attention = checked_attention
                try:
                    out = model(mx.array([corpus[begin : begin + width]]), cache=cache)
                finally:
                    mx.fast.scaled_dot_product_attention = original_attention
                logits = out.logits if native else out
                last = sample(logits[:, -1:, :])
                begin += width
                del out, logits
            offsets = [int(c.offset) if hasattr(c, "offset") else None for c in cache or []]
            assert all(x == n for x in offsets if x is not None)
            if cache is not None and native:
                for c in cache:
                    c.reset()
                    assert c.offset == 0
            del cache
            gc.collect()
            mx.clear_cache()
            return last, offsets

        def compute():
            for n in [16, 128, 512, 1024, 2048]:
                if n > bundle.context_length:
                    report["rows"].append({"context": n, "status": "UNSUPPORTED"})
                    save(report)
                    continue
                full_n, _ = forward(True, n, [n], False)
                full_r, _ = forward(False, n, [n], False)
                row = {
                    "context": n,
                    "target_token": corpus[n],
                    "partitions": [],
                    "cross_fresh": sensitivity(full_r, full_n, corpus[n], dtype, policy),
                }
                partitions = [
                    [n],
                    [n - 1, 1],
                    [n // 3, n // 3, n - 2 * (n // 3)],
                    [128] * (n // 128) + ([n % 128] if n % 128 else []),
                ]
                for parts in partitions:
                    cached_n, offsets_n = forward(True, n, parts, True)
                    cached_r, offsets_r = forward(False, n, parts, True)
                    row["partitions"].append(
                        {
                            "parts": parts,
                            "native_offsets": offsets_n,
                            "reference_offsets": offsets_r,
                            "native_full_vs_cached": sensitivity(
                                full_n, cached_n, corpus[n], dtype, policy
                            ),
                            "reference_full_vs_cached": sensitivity(
                                full_r, cached_r, corpus[n], dtype, policy
                            ),
                            "cross_cached": sensitivity(
                                cached_r, cached_n, corpus[n], dtype, policy
                            ),
                        }
                    )
                row["status"] = "MEASURED"
                report["rows"].append(row)
                save(report)
                print(bundle.model_id, args.precision, n, flush=True)
            report["status"] = "MEASURED_NOT_RELEASE_APPROVED"
            return report

        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def quality(bundle, reference, manager, args, report, save):
    import mlx.core as mx
    from local_reference import reference_module
    from quality import evaluate

    from mlx_one import Engine
    from mlx_one.engine.cache_specs import CacheConfig

    report["fixture_sha256"] = hashlib.sha256(args.fixture.read_bytes()).hexdigest()

    class ReferenceEngine:
        def stream(self, prompt, *, options):
            def generate():
                cache = reference_module("cache").make_prompt_cache(reference)

                class View:
                    def __init__(self, owner):
                        self.owner = owner

                    def __getattr__(self, name):
                        return getattr(self.owner, name)

                    def update_and_fetch(self, k, v):
                        dtype = k.dtype
                        k, v = self.owner.update_and_fetch(
                            k.astype(mx.float16), v.astype(mx.float16)
                        )
                        return k.astype(dtype), v.astype(dtype)

                cache = [View(c) if hasattr(c, "update_and_fetch") else c for c in cache]
                logits = reference(mx.array([bundle.tokenizer.encode(prompt)]), cache=cache)
                ids, finish = [], "length"
                for step in range(options.max_tokens):
                    token = int(mx.argmax(logits[0, -1]))
                    if token in bundle.eos_token_ids:
                        finish = "stop"
                        break
                    ids.append(token)
                    if step + 1 < options.max_tokens:
                        logits = reference(mx.array([[token]]), cache=cache)
                return [
                    *[SimpleNamespace(text="", token_id=i, finish_reason=None) for i in ids],
                    SimpleNamespace(
                        text=bundle.tokenizer.decode(ids), token_id=None, finish_reason=finish
                    ),
                ]

            return iter(manager._pool.executor.call(bundle.model_id, generate))

    fixture = json.loads(args.fixture.read_text())
    selected = [c for c in fixture["cases"] if c["instruction"] == (bundle.architecture != "gpt2")]
    # Evaluate the exact selected immutable fixture, with progress after each prompt.
    temp = args.output.parent / (args.output.stem + "-selected-fixture.json")
    temp.write_text(json.dumps({**fixture, "cases": selected}, indent=2))
    execution = "float16" if args.precision.startswith("affine") else None
    with Engine(
        bundle,
        execution_dtype=execution,
        inference_mode="reference_dense",
        cache_config=CacheConfig(prefix_cache_enabled=False),
    ) as engine:
        report["native"] = evaluate(
            bundle, engine, temp, chat_applicable=bundle.architecture != "gpt2"
        )
        report["cleanup"] = engine.runtime_stats()["cache_stats"]
        repeat = args.output.parent / (args.output.stem + "-repeat-fixture.json")
        repeat.write_text(json.dumps({**fixture, "cases": selected[:3]}, indent=2))
        report["native_repeat"] = evaluate(
            bundle, engine, repeat, chat_applicable=bundle.architecture != "gpt2"
        )
        save(report)
    report["reference"] = evaluate(
        bundle, ReferenceEngine(), temp, chat_applicable=bundle.architecture != "gpt2"
    )
    pairs = list(zip(report["native"]["cases"], report["reference"]["cases"], strict=True))
    report["behavior_comparison"] = [
        {
            "id": a["id"],
            "native_status": a["status"],
            "reference_status": b["status"],
            "text_identical": a["text"] == b["text"],
            "token_ids_identical": a["token_ids"] == b["token_ids"],
            "finish_identical": a["finish_reason"] == b["finish_reason"],
        }
        for a, b in pairs
    ]
    report["framework_specific_failures"] = [
        a["id"] for a, b in pairs if b["status"] == "PASS" and a["status"] != "PASS"
    ]
    report["deterministic_repeat"] = all(
        a["token_ids"] == b["token_ids"]
        for a, b in zip(
            report["native"]["cases"][:3], report["native_repeat"]["cases"], strict=True
        )
    )
    report["status"] = (
        "PASS"
        if not report["framework_specific_failures"]
        and report["deterministic_repeat"]
        and report["cleanup"]["active_handles"] == 0
        else "FAIL"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--precision", default="stored")
    parser.add_argument("--mode", choices=["stability", "quality"], default="stability")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(data):
        args.output.write_text(json.dumps(data, indent=2) + "\n")

    try:
        result = run(args, save)
    except Exception as exc:
        import traceback

        result = {
            "status": "BLOCKED_ENVIRONMENT",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp": time.time(),
        }
    save(result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
