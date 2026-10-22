"""Diagnostic A/B/C/D KV precision isolation; default production is never modified."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from four_path_oracle import load_reference, snapshot
from numerical_metrics import distribution_metrics

ROOT = Path(__file__).resolve().parents[2]


def dtype(x):
    return str(x.dtype).rsplit(".", 1)[-1]


def desc(x):
    return {
        "shape": list(x.shape),
        "dtype": dtype(x),
        "strides": list(x.strides) if hasattr(x, "strides") else None,
    }


def legal_partitions(n):
    choices = [[n]]
    if n > 1:
        choices.extend([[n - 1, 1], [1, n - 1]])
    if n >= 3:
        choices.append([n // 3, n // 3, n - 2 * (n // 3)])
    return list(dict.fromkeys(tuple(p) for p in choices))


def run(args, save):
    import mlx.core as mx

    from mlx_one.core.cache import configure_kv_caches
    from mlx_one.server.model_manager import ModelManager

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("real Metal GPU required")
    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    manager = ModelManager()
    bundle = manager.load(args.checkpoint, offline=True)
    report = {
        "timestamp": time.time(),
        "checkpoint": bundle.model_id,
        "revision": bundle.revision,
        "profile": args.precision,
        "rows": [],
        "diagnostic_only": True,
        "default_production_changed": False,
        "tf32_env": os.environ.get("MLX_ENABLE_TF32"),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_tool_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in [
                "numerical_isolation.py",
                "numerical_metrics.py",
                "precision_operations.py",
                "four_path_oracle.py",
                "local_reference.py",
            ]
        },
        "kernel_control": args.kernel_control,
        "cache_control": args.cache_control,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src/mlx_one").rglob("*.py")
        },
    }

    def compute():
        nonlocal bundle
        sys.path.insert(0, str(ROOT / "scripts/qualification"))
        from numerical_contract_v2 import require_profile_alignment

        def identity(ids, policy):
            return {
                "checkpoint_revision": bundle.revision,
                "weight_profile": args.precision,
                "kv_policy": policy,
                "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            }

        bundle, reference, module = load_reference(bundle, args.precision)
        reference_path = Path(module.__file__).resolve()
        if not reference_path.is_relative_to(ROOT / "references_libs/mlx-lm-main"):
            raise RuntimeError("wrong reference provenance")
        report["reference"] = {
            "module": str(reference_path),
            "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "semantic_version": "unknown: unversioned local archive",
        }
        report["weight_dtypes"] = sorted(
            {
                dtype(x)
                for _, x in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
                    bundle.model.parameters()
                )
            }
        )
        vocab = bundle.model.config.vocab_size
        prompts = {
            "historical": "The red book was on the table. The city is calm. ",
            "factual": ("Paris is the capital of France. Water freezes at zero degrees Celsius. "),
            "code": (
                "def fibonacci(n):\n    a, b = 0, 1\n"
                "    for _ in range(n):\n        a, b = b, a+b\n"
            ),
            "json": '{"people":[{"name":"Ana","age":23}],"valid":true,"count":1}\n',
            "arithmetic": (
                "Question: What is 17 plus 28? Answer: 45. "
                "Question: What is 9 times 6? Answer: 54. "
            ),
            "prose": (
                "At dawn the small boat left the harbor, carrying letters across the quiet bay. "
            ),
            "repeated": " red red red red red red blue ",
            "retrieval": (
                "The secret archive code is BLUE-7319. Remember the code. The story continues. "
            ),
        }
        fixtures = {}
        max_n = max(args.contexts) + 17
        for name, prompt in prompts.items():
            ids = bundle.tokenizer.encode(prompt)
            fixtures[name] = (ids * ((max_n + len(ids) - 1) // len(ids)))[:max_n]
        fixtures["random"] = np.random.default_rng(9621).integers(0, vocab, max_n).tolist()
        report["fixtures"] = fixtures
        report["rows"] = []
        active_dtype = None

        class AuditCache:
            def __init__(self, owner, records, layer):
                self.owner, self.records, self.layer = owner, records, layer
                self.previous = None

            def __getattr__(self, key):
                return getattr(self.owner, key)

            def update(self, *values):
                before = self.owner.offset
                incoming = [desc(x) for x in values]
                output = self.owner.update(*values)
                length = values[0].shape[2] if len(values) == 2 else values[0].shape[1]
                if self.owner.offset != before + length:
                    raise AssertionError("wrong absolute cache offset")
                if len(values) == 2:
                    stored_dtype = output[0].dtype
                    expected = tuple(x.astype(stored_dtype) for x in values)
                    if self.previous is not None:
                        expected = tuple(
                            mx.concatenate((p, x), axis=2)
                            for p, x in zip(self.previous, expected, strict=True)
                        )
                    if not all(
                        bool(mx.array_equal(a, b)) for a, b in zip(output, expected, strict=True)
                    ):
                        raise AssertionError(
                            "logical KV contents differ from expected representation"
                        )
                    window = getattr(self.owner, "max_size", None)
                    self.previous = (
                        tuple(x[:, :, -window:] for x in expected) if window else expected
                    )
                    state = [desc(x) for x in self.previous]
                else:
                    incoming_value = values[0]
                    history = self.owner.kernel_size - 1
                    prefix = self.previous
                    if prefix is None:
                        prefix = mx.zeros(
                            (1, history, incoming_value.shape[-1]), dtype=incoming_value.dtype
                        )
                    expected = mx.concatenate((prefix, incoming_value), axis=1)
                    if not bool(mx.array_equal(output, expected)):
                        raise AssertionError("convolution history differs from expected append")
                    self.previous = expected[:, -history:] if history else expected[:, :0]
                    if not bool(mx.array_equal(self.owner.values, self.previous)):
                        raise AssertionError("convolution retained state mismatch")
                    state = [desc(self.owner.values)]
                self.records.append(
                    {
                        "layer": self.layer,
                        "offset_before": before,
                        "offset_after": self.owner.offset,
                        "incoming": incoming,
                        "state": state,
                        "exact_append": "PASS",
                    }
                )
                return output

        def forward(ids, parts, mode, capture=False, model=None):
            nonlocal active_dtype
            model = model or bundle.model
            caches = None
            if mode in {"B", "C"}:
                caches = model.make_cache()
                if mode == "C":
                    caches = configure_kv_caches(caches, "f16", "f16")
                elif args.cache_control == "same_allocator":
                    from mlx_one.core.cache import KVCache, QuantizedKVCache

                    class NaturalStorage(QuantizedKVCache):
                        def _append_component(self, current, values, bits, dtype, previous):
                            if bits is not None:
                                raise ValueError("diagnostic natural state is floating only")
                            return self._append_array(current, values, previous)

                    caches = tuple(
                        NaturalStorage(None, None)
                        if isinstance(c, KVCache) and getattr(c, "supports_quantization", True)
                        else c
                        for c in caches
                    )
            states, attention, control = [], [], []
            if caches is not None:
                caches = tuple(AuditCache(c, states, i) for i, c in enumerate(caches))
            start = 0
            peak_state = 0
            last = None
            for width in parts:
                original = mx.fast.scaled_dot_product_attention
                layer_index = 0

                def sdpa(
                    q,
                    k,
                    v,
                    *,
                    scale,
                    mask=None,
                    original=original,
                    start=start,
                    width=width,
                    **kwargs,
                ):
                    nonlocal layer_index, active_dtype
                    index = layer_index
                    layer_index += 1
                    active_dtype = dtype(q)
                    if mode == "D":
                        k, v = (x.astype(mx.float16).astype(q.dtype) for x in (k, v))
                    output = original(q, k, v, scale=scale, mask=mask, **kwargs)
                    window = None
                    if bundle.architecture == "gemma3_text":
                        if bundle.model.config.layer_types[index] == "sliding_attention":
                            window = bundle.model.config.sliding_window
                    elif bundle.architecture == "phi3":
                        window = bundle.model.config.sliding_window
                    query = np.arange(start, start + width)[:, None]
                    key = np.arange(start + width - k.shape[2], start + width)[None, :]
                    expected = key <= query
                    if window:
                        expected &= key > query - window
                    if mask is None:
                        observed = np.ones(expected.shape, dtype=bool)
                    elif isinstance(mask, str):
                        if mask != "causal":
                            raise ValueError("unknown mask")
                        observed = (
                            np.arange(k.shape[2])[None, :]
                            <= np.arange(width)[:, None] + k.shape[2] - width
                        )
                    else:
                        material = snapshot(mask)
                        observed = material if material.dtype == bool else np.isfinite(material)
                    compact = np.squeeze(observed)
                    visible = np.array_equal(np.broadcast_to(compact, expected.shape), expected)
                    if not visible:
                        raise AssertionError("wrong absolute attention visibility")
                    attention.append(
                        {
                            "layer_attention_index": index,
                            "q": desc(q),
                            "k": desc(k),
                            "v": desc(v),
                            "output": desc(output),
                            "query_range": [start, start + width - 1],
                            "key_range": [int(key.min()), int(key.max())],
                            "window": window,
                            "visibility": "PASS",
                        }
                    )
                    if capture and index in {0, 5}:
                        from precision_operations import attention_control

                        control.append(
                            attention_control(
                                q,
                                k,
                                v,
                                output,
                                mask,
                                scale,
                                label=f"{mode}/layer{index}/offset{start}",
                                sdpa=original,
                            )
                        )
                    return output

                mx.fast.scaled_dot_product_attention = sdpa
                try:
                    out = model(mx.array([ids[start : start + width]]), cache=caches)
                    logits = out.logits
                    mx.eval(logits)
                    last = snapshot(logits[:, -1:, :])
                    actual_top1 = int(mx.argmax(logits[0, -1]))
                    if actual_top1 != int(np.argmax(last)):
                        raise AssertionError("CPU token selection does not match actual MLX argmax")
                    final_dtype = dtype(logits)
                    if caches:
                        peak_state = max(peak_state, sum(c.nbytes for c in caches))
                    del out, logits
                finally:
                    mx.fast.scaled_dot_product_attention = original
                start += width
            if start != len(ids):
                raise ValueError("partition does not cover tokens")
            if caches:
                for c in caches:
                    if c.offset != len(ids):
                        raise AssertionError("offset mismatch")
                    c.reset()
                    if c.offset != 0:
                        raise AssertionError("reset failed")
            del caches
            mx.clear_cache()
            gc.collect()
            return last, {
                "mode": mode,
                "state": states,
                "attention": attention,
                "logit_dtype": final_dtype,
                "last_logit_sha256": hashlib.sha256(last.tobytes()).hexdigest(),
                "actual_top1_id": actual_top1,
                "peak_state_bytes": peak_state,
                "independent_controls": control,
                "comparison_identity": identity(
                    ids,
                    {"A": "none", "B": "natural", "C": "production_f16", "D": "rounded_full"}[mode],
                ),
            }

        def reference_forward(ids, parts, production):
            from local_reference import reference_module

            cache = (
                reference_module("cache").make_prompt_cache(reference)
                if parts is not None
                else None
            )

            class View:
                def __init__(self, owner):
                    self.owner = owner

                def __getattr__(self, key):
                    return getattr(self.owner, key)

                def update_and_fetch(self, k, v):
                    original_dtype = k.dtype
                    k, v = self.owner.update_and_fetch(k.astype(mx.float16), v.astype(mx.float16))
                    return k.astype(original_dtype), v.astype(original_dtype)

            if production and cache:
                cache = [View(c) if hasattr(c, "update_and_fetch") else c for c in cache]
            begin = 0
            for width in parts or [len(ids)]:
                out = reference(mx.array([ids[begin : begin + width]]), cache=cache)
                mx.eval(out)
                last = snapshot(out[:, -1:, :])
                if int(mx.argmax(out[0, -1])) != int(np.argmax(last)):
                    raise AssertionError("reference MLX argmax and comparison tooling disagree")
                begin += width
                del out
            del cache
            mx.clear_cache()
            gc.collect()
            return last, identity(
                ids, "none" if parts is None else "production_f16" if production else "natural"
            )

        for name in args.fixtures:
            for n in args.contexts:
                if n > bundle.context_length:
                    report["rows"].append({"fixture": name, "context": n, "status": "UNSUPPORTED"})
                    continue
                if name == "near_tie":
                    rng = np.random.default_rng(9621 + n)
                    chosen = None
                    for attempt in range(32):
                        candidate = rng.integers(0, vocab, n + 1).tolist()
                        logits = bundle.model(mx.array([candidate[:n]])).logits
                        values = snapshot(logits[:, -1:, :]).ravel()
                        top = np.partition(values, -2)[-2:]
                        margin = float(top.max() - top.min())
                        if chosen is None or margin < chosen[0]:
                            chosen = (margin, candidate, attempt)
                        del logits
                    ids = chosen[1][:n]
                    target = chosen[1][n]
                    report.setdefault("near_tie_selection", []).append(
                        {
                            "context": n,
                            "seed": 9621 + n,
                            "candidates": 32,
                            "chosen_index": chosen[2],
                            "margin": chosen[0],
                            "declared_target_margin": 0.02,
                            "found": chosen[0] <= 0.02,
                            "selected_before_cached_comparison": True,
                            "token_ids": chosen[1],
                        }
                    )
                elif name == "retrieval":
                    intro = bundle.tokenizer.encode("Archive code: BLUE-7319.\n")
                    query = bundle.tokenizer.encode("\nRecall the archive code:")
                    answer = bundle.tokenizer.encode(" BLUE-7319.")
                    filler = bundle.tokenizer.encode("The quiet town continued its daily work. ")
                    room = n - len(intro) - len(query)
                    if room >= 0:
                        ids = (
                            intro
                            + (filler * ((room + len(filler) - 1) // len(filler)))[:room]
                            + query
                        )
                        target = answer[0]
                    else:
                        short = intro + query + answer
                        ids = short[:n]
                        target = short[n]
                else:
                    ids = fixtures[name][:n]
                    target = fixtures[name][n]
                capture = args.controls and name == "historical" and n in {16, 128, 512, 513}
                full, ad = forward(ids, [n], "A", capture=capture)
                dround, dd = forward(ids, [n], "D")
                row = {
                    "fixture": name,
                    "context": n,
                    "input_ids": ids,
                    "next_token": target,
                    "A": ad,
                    "D": dd,
                    "A_vs_D": distribution_metrics(full, dround, target, active_dtype),
                    "partitions": [],
                }
                ref_full, ref_identity = reference_forward(ids, None, False)
                require_profile_alignment(ref_identity, ad["comparison_identity"])
                row["reference_scope"] = (
                    "RAW_REFERENCE_SPECIFICATION_GAP"
                    if bundle.architecture == "phi3" and n > bundle.model.config.sliding_window
                    else "APPLICABLE"
                )
                row["cross_A"] = distribution_metrics(ref_full, full, target, active_dtype)
                for parts in legal_partitions(n):
                    matched, bd = forward(ids, parts, "B", capture=capture and parts[-1] == 1)
                    production, cd = forward(ids, parts, "C", capture=capture and parts[-1] == 1)
                    ref_b, ref_b_identity = reference_forward(ids, parts, False)
                    ref_c, ref_c_identity = reference_forward(ids, parts, True)
                    require_profile_alignment(ref_b_identity, bd["comparison_identity"])
                    require_profile_alignment(ref_c_identity, cd["comparison_identity"])
                    row["partitions"].append(
                        {
                            "parts": parts,
                            "B": bd,
                            "C": cd,
                            "A_vs_B": distribution_metrics(full, matched, target, active_dtype),
                            "A_vs_C": distribution_metrics(full, production, target, active_dtype),
                            "B_vs_C": distribution_metrics(
                                matched, production, target, active_dtype
                            ),
                            "D_vs_C": distribution_metrics(
                                dround, production, target, active_dtype
                            ),
                            "cross_B": distribution_metrics(ref_b, matched, target, active_dtype),
                            "cross_C": distribution_metrics(
                                ref_c, production, target, active_dtype
                            ),
                        }
                    )
                report["rows"].append(row)
                save(report)
                print(bundle.model_id, args.precision, name, n, flush=True)
        report["status"] = "MEASURED_NOT_RELEASE_APPROVED"
        return report

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--precision", default="stored")
    parser.add_argument("--contexts", type=int, nargs="+", default=[16, 32, 128, 512, 1024, 2048])
    parser.add_argument("--fixtures", nargs="+", default=["historical"])
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--kernel-control", choices=["canonical", "fixed_row"], default="canonical")
    parser.add_argument(
        "--cache-control", choices=["model_default", "same_allocator"], default="model_default"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(value):
        args.output.write_text(json.dumps(value, indent=2) + "\n")

    from four_path_oracle import attention_control, linear_control

    with (
        attention_control("rowwise" if args.kernel_control == "fixed_row" else "canonical"),
        linear_control("rowwise" if args.kernel_control == "fixed_row" else "canonical"),
    ):
        result = run(args, save)
    save(result)


if __name__ == "__main__":
    main()
