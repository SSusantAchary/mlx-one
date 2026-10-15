"""Committed-boundary native hybrid lifecycle qualification (no public resume API)."""
from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

from mlx_one.engine.batching import ForwardRequest, execute_compatible
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.components import load_snapshot, save_snapshot
from mlx_one.engine.hybrid import HybridRunner
from mlx_one.engine.parity import compare_logits

LIFECYCLE_SCHEMA = "native-hybrid-lifecycle-v1"


def adopt_snapshot(runtime, handle, staged, *, cancel=None):
    """Validate/evaluate all components before one owned bundle replacement."""
    import mlx.core as mx

    try:
        runtime._validate_handle(handle)
        specs = runtime.plan.state_specs
        if len(staged.entries) != len(specs) or len({c.offset for c in staged.entries}) != 1:
            raise ValueError("snapshot topology or component positions mismatch")
        for component, spec in zip(staged.entries, specs, strict=True):
            component.validate(spec)
        from mlx_one.engine.transactions import component_tensors

        mx.eval(*(v for c in staged.entries for v in component_tensors(c)))
        runtime._validate_handle(handle)
        callback = handle.metadata.get("is_cancelled")
        if ((cancel is not None and cancel.is_set())
                or callable(callback) and callback() or handle.metadata.get("cancelled")):
            return False
        old = handle.bundle
        computed,reserved = handle.computed_tokens,handle.reserved_bytes
        committed = handle.reservation.committed_bytes if handle.reservation else None
        handle.bundle = staged
        handle.computed_tokens = staged.offset
        try:
            runtime.reconcile(handle)
        except BaseException:
            handle.bundle = old
            handle.computed_tokens,handle.reserved_bytes = computed,reserved
            if handle.reservation is not None:
                handle.reservation.committed_bytes = committed
            raise
        old.release()
        return True
    finally:
        if handle.bundle is not staged:
            staged.release()


def hybrid_lifecycle(bundle, profile, output, *, reload_bundle=None):
    """Independent native oracle for resume/fork/cancel/disposal/reload."""
    import mlx.core as mx

    dtype = "bfloat16" if profile.cache_type == "bf16" else "float16"
    from mlx_one.inference_baseline import TOLERANCES

    atol, rtol = TOLERANCES[dtype] if profile.atol is None else (profile.atol, profile.rtol)
    config = CacheConfig(backend="paged" if profile.inference_mode == "paged" else "hybrid",
                         prefix_cache_enabled=False,
                         key_type=profile.cache_type, value_type=profile.cache_type,
                         block_size_tokens=profile.page_tokens)
    runtime = CacheManager(bundle, config)
    oracle = CacheManager(bundle, CacheConfig(prefix_cache_enabled=False))
    runtimes = [runtime, oracle]
    identity = {"weights": runtime.weight_identity, "revision": bundle.revision,
                "topology": runtime.plan.topology_digest, "storage_dtype": dtype}
    runner, comparisons, checks = HybridRunner(), [], {}
    seed = tuple(bundle.tokenizer.encode("Native lifecycle shared prefix. " * 64))
    prefix = seed[:profile.page_tokens]
    if len(prefix) < profile.page_tokens:
        raise ValueError("lifecycle fixture must fill a complete page")
    handles = [runtime.create_handle(name, key_type=profile.cache_type,
                                     value_type=profile.cache_type) for name in ("donor", "peer")]
    references = {}
    def independent(name, tokens):
        if name not in references:
            references[name] = oracle.create_handle(name, key_type=profile.cache_type,
                                                     value_type=profile.cache_type)
        result = bundle.model(mx.array([tokens]), cache=references[name].bundle.entries)
        mx.eval(result.logits)
        return result.logits[:, -1:]
    def forward(owner, entries, tokens, *, phase="decode", cancel=None, active_bundle=None):
        return ForwardRequest(active_bundle or owner.bundle, tuple(tokens), entries.bundle.entries,
            phase, prefix, owner, entries, request_id=entries.request_id, cancel=cancel)
    def compare_state(actual, expected, name):
        for index, (a, b) in enumerate(zip(actual.entries, expected.entries, strict=True)):
            if a.offset != b.offset:
                checks[f"{name}_positions"] = False
            if hasattr(a, "_blocks"):
                if hasattr(b, "_blocks"):
                    blocks = [r.block.state for r in b._blocks]
                    expected_values = tuple(mx.concatenate([v[i] for v in blocks], axis=2)
                                            for i in range(2))
                elif hasattr(b, "_keys"):
                    expected_values = (b._materialize(b._keys, b.key_bits, b.offset),
                                       b._materialize(b._values, b.value_bits, b.offset))
                else:
                    expected_values = (b.keys, b.values)
                start = 0
                for ref in a._blocks:
                    block = ref.block
                    for i, value in enumerate(block.state):
                        comparisons.append({"stage": name, "layer": index, "page": start,
                            **compare_logits(value[:, :, :block.token_count],
                                expected_values[i][:, :, start:start + block.token_count],
                                atol=atol, rtol=rtol)})
                    start += block.token_count
                continue
            attrs = (("conv_state","recurrent_state")
                     if getattr(a,"checkpoint_semantics",None) is not None else ("keys","values"))
            for attr in attrs:
                left, right = getattr(a, attr, None), getattr(b, attr, None)
                if right is None and hasattr(b, "_keys"):
                    stored = b._keys if attr == "keys" else b._values
                    bits = b.key_bits if attr == "keys" else b.value_bits
                    right = b._materialize(stored, bits, b.offset)
                if left is None and right is None:
                    continue
                if attr == "keys" or getattr(a, "kind", "") in {"full_kv", "sliding_kv"}:
                    n = min(a.offset, getattr(a, "max_size", a.offset))
                    left, right = left[:, :, :n], right[:, :, :b.offset][:, :, -n:]
                comparisons.append({"stage": name, "layer": index, "tensor": attr,
                    **compare_logits(left, right, atol=atol, rtol=rtol)})
    try:
        requests = tuple(forward(runtime, h,
                                 prefix if i == 0 else prefix[:max(1, len(prefix)//2)],
                                 phase="prefill") for i, h in enumerate(handles))
        outputs = execute_compatible(runner,requests)
        for i, request in enumerate(requests):
            expected = independent(request.request_id, request.token_ids)
            comparisons.append({"stage": "initial", **compare_logits(
                outputs.logits[i], expected, atol=atol, rtol=rtol)})
        save_snapshot(Path(output) / "snapshot", handles[0].bundle, identity=identity)
        specs = tuple(replace(s, dtype=dtype) if s.kind in {"full_kv", "sliding_kv"} else s
                      for s in runtime.plan.state_specs)
        restored = load_snapshot(Path(output) / "snapshot", identity=identity, expected_specs=specs,
                                 page_pool=(runtime.block_pool
                                            if config.backend == "paged" else None))
        checks["pause_resume"] = adopt_snapshot(runtime, handles[0], restored)
        fork = runtime.create_handle("fork", key_type=profile.cache_type,
                                     value_type=profile.cache_type)
        checks["fork"] = adopt_snapshot(runtime, fork, handles[0].bundle.clone())
        independent("fork", prefix)
        for step in range(3):
            tokens = [seed[len(prefix) + step], seed[len(prefix) + step + 4]]
            result = execute_compatible(runner,tuple(forward(runtime, h, (token,))
                for h, token in zip((handles[0], fork), tokens, strict=True)))
            for index, (h, token) in enumerate(zip((handles[0], fork), tokens, strict=True)):
                expected = independent(h.request_id, (token,))
                comparisons.append({"stage": "resumed_fork", **compare_logits(
                    result.logits[index], expected, atol=atol, rtol=rtol)})
                compare_state(h.bundle, references[h.request_id].bundle, "resumed_fork")
        cancel = threading.Event()
        before = handles[0].bundle.clone()
        class CancelAtEvaluation:
            def __getattr__(self, name):
                return getattr(bundle.model, name)
            def __call__(self, *args, **kwargs):
                result = bundle.model(*args, **kwargs)
                mx.eval(result.logits)
                cancel.set()
                return result
        wrapped = replace(bundle, model=CancelAtEvaluation())
        # Both requests share the wrapped bundle; only the donor is cancelled.
        result = execute_compatible(runner,tuple(forward(runtime, h, (seed[-1],),
            cancel=cancel if i == 0 else None, active_bundle=wrapped)
            for i, h in enumerate(handles)))
        compare_state(handles[0].bundle, before, "cancelled")
        expected = independent("peer", (seed[-1],))
        comparisons.append({"stage": "peer_continues", **compare_logits(
            result.logits[1], expected, atol=atol, rtol=rtol)})
        checks["cancellation_isolation"] = handles[0].bundle.offset == before.offset
        before.release()
        stale = handles[0]
        runtime.close()
        runtime.close()
        checks["repeated_disposal"] = runtime.stats().active_handles == 0
        if reload_bundle is None:
            checks["checkpoint_reload"] = None
        else:
            fresh_bundle = reload_bundle()
            fresh = CacheManager(fresh_bundle, config)
            runtimes.append(fresh)
            checks["checkpoint_reload"] = (fresh.weight_identity == identity["weights"]
                                            and fresh.plan.topology_digest == identity["topology"])
            try:
                fresh._validate_handle(stale)
                checks["stale_handle_rejected"] = False
            except (ValueError, RuntimeError):
                checks["stale_handle_rejected"] = True
            handle = fresh.create_handle("reloaded", key_type=profile.cache_type,
                                         value_type=profile.cache_type)
            adopt_snapshot(fresh, handle, load_snapshot(
                Path(output) / "snapshot", identity=identity,
                expected_specs=specs, page_pool=fresh.block_pool
                if config.backend == "paged" else None))
            result = execute_compatible(runner,(forward(fresh, handle, (seed[-2],)),)).logits[0]
            independent("reloaded", prefix)
            expected = independent("reloaded", (seed[-2],))
            comparisons.append({"stage": "reloaded", **compare_logits(result, expected,
                                                                        atol=atol, rtol=rtol)})
    finally:
        for owner in runtimes:
            owner.close()
    passed = all(c["within_tolerance"] for c in comparisons) and all(checks.values())
    return {"schema_version": LIFECYCLE_SCHEMA,
            "status": "passed" if passed else "pending" if None in checks.values() else "failed",
            "checks": checks, "comparisons": comparisons,
            "cancel_boundary": "evaluated-native-forward-before-commit",
            "identity": identity, "cleanup": [r.stats().to_dict() for r in runtimes]}
