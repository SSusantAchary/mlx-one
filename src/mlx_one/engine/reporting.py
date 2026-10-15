"""Backend-free native runtime diagnostics. Never materialize block KV for reporting."""

from __future__ import annotations

from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from mlx_one.engine.cache import cache_allocated_nbytes, cache_nbytes
from mlx_one.engine.modes import resolved_mode
from mlx_one.registry import InferenceFeatureRegistry, InferenceScope


def normalize_dtype(dtype: Any) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def tensor_dtypes(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (tuple, list)):
        values = value
    else:
        dtype = getattr(value, "dtype", None)
        return (normalize_dtype(dtype),) if dtype is not None else ()
    return tuple(sorted({dtype for item in values for dtype in tensor_dtypes(item)}))


def describe_cache(entries, plan) -> list[dict[str, Any]]:
    layers = []
    for spec, cache in zip(plan.state_specs, entries, strict=True):
        if hasattr(cache, "_blocks"):
            tensors = [ref.block.state for ref in cache._blocks]
        else:
            attributes = vars(cache) if hasattr(cache, "__dict__") else {}
            tensors = {key: value for key, value in attributes.items()
                       if key in {"keys", "values", "_keys", "_values", "state",
                                  "conv_state", "recurrent_state"}}
        layers.append({
            "layer_id": spec.layer_id, "kind": spec.kind,
            "actual_dtypes": tensor_dtypes(tensors),
            "logical_bytes": cache_nbytes(cache),
            "allocated_bytes": cache_allocated_nbytes(cache),
            "committed_tokens": int(getattr(cache, "offset", 0)),
        })
    return layers


def component_capabilities(bundle, runtime, mode):
    if runtime is None:
        return []
    specs = runtime.plan.state_specs
    if getattr(runtime, "hybrid_state", False):
        from mlx_one.engine.components import hybrid_components

        components = hybrid_components(bundle, runtime.config.key_type,
            pool=runtime.block_pool if mode == "paged" else None)
        result = []
        for spec, component in zip(specs, components, strict=True):
            caps = component.capabilities()
            result.append({**asdict(replace(spec, can_batch=caps.can_batch,
                can_page=caps.can_page, can_snapshot=caps.can_snapshot,
                can_quantize=caps.can_quantize, can_share_prefix=caps.can_share_prefix)),
                "component": asdict(caps), "committed_position_semantics": "real_input_tokens"})
        return result
    return [asdict(replace(s, can_page=True)) if mode == "paged" and s.kind == "full_kv"
            else asdict(s) for s in specs]


def native_report(bundle, runtime=None, *, backend="dense", hardware=None,
                  source_revision=None, registry=None, mode=None, execution_dtype=None
                  ) -> dict[str, Any]:
    selected_mode = mode or resolved_mode(backend)
    plan = runtime.plan if runtime is not None else None
    parameters = getattr(bundle.model, "parameters", None)
    weights = tensor_dtypes(parameters()) if callable(parameters) else ()
    layers = runtime.state_observations() if runtime is not None else []
    kv_dtypes = sorted({d for layer in layers if layer["kind"] in {"full_kv", "sliding_kv"}
                        for d in layer["actual_dtypes"]})
    try:
        mlx_version = version("mlx")
    except PackageNotFoundError:
        mlx_version = None
    features = (registry or getattr(runtime, "qualification_registry", None)
                or InferenceFeatureRegistry())
    if hardware is None or source_revision is None:
        from mlx_one.engine.selection import host_identity, source_identity

        hardware = hardware or host_identity()
        source_revision = source_revision or source_identity()
    scope = InferenceScope(
        str(bundle.model_id), bundle.revision, selected_mode,
        plan.topology_digest if plan else None, ",".join(weights) or None,
        ",".join(kv_dtypes) or None, mlx_version, hardware, source_revision,
    )
    for name in ("dense_execution",):
        if features.find(name, scope) is None:
            features.register_experimental(name, scope)
    block = bool(plan and plan.architecture == "qwen2" and plan.block_compatible
                 and not getattr(bundle, "quantization", {}))
    if block:
        for name in ("block_storage", "block_prefix_reuse"):
            if features.find(name, scope) is None:
                features.register_experimental(name, scope)
    runner_scope = replace(scope, execution_path="causal_runner_equal_offset_dense")
    runner_features = {}
    if plan:
        runner_names = ["chunked_prefill"]
        if plan.state_specs and all(spec.can_batch for spec in plan.state_specs):
            runner_names.append("equal_offset_batched_decode")
        for name in runner_names:
            if features.find(name, runner_scope) is None:
                features.register_experimental(name, runner_scope)
            runner_features[name] = features.report(name, runner_scope)
    if selected_mode in {"experimental_dense_batch", "paged", "experimental_hybrid_batch"}:
        names = ["continuous_batching", "batched_chunked_prefill"]
        if selected_mode == "experimental_dense_batch":
            names.append("batched_dense_decode")
        for name in names:
            if features.find(name, scope) is None:
                features.register_experimental(name, scope)
    added_features = {"paged": ("paged_attention", "paged_prefill"),
                      "experimental_hybrid_batch": ("hybrid_prefix_restore", "hybrid_batching")}
    names = added_features.get(selected_mode, ())
    if getattr(runtime, "hybrid_state", False):
        names = (*names, "hybrid_prefix_restore", "hybrid_batching")
    for name in names:
        if features.find(name, scope) is None:
            features.register_experimental(name, scope)
    inventory = ("dense_execution", "block_storage", "block_prefix_reuse",
                 "continuous_batching", "batched_dense_decode", "batched_chunked_prefill",
                 "paged_attention", "paged_prefill", "hybrid_prefix_restore", "hybrid_batching",
                 "quantized_paged_kv", "long_context")
    qualification = {name: features.report(name, scope) for name in inventory}
    details = {}
    for name, claim in qualification.items():
        available = claim["status"] != "not_implemented"
        if name in {"quantized_paged_kv", "long_context"}:
            available = claim.get("hardware_correctness_passed", False)
        details[name] = {"implemented": True, "available_for_opt_in": bool(available),
                         "checkpoint_correctness_passed": claim.get(
                             "checkpoint_correctness_passed", False),
                         "hardware_correctness_passed": claim.get(
                             "hardware_correctness_passed", False),
                         "performance_qualified": claim.get("performance_qualified", False),
                         "reason": claim.get("validation_reason", "stale native evidence")}
    from mlx_one.engine.adapters import execution_contract
    from mlx_one.engine.selection import compatibility_digest

    contract = execution_contract(bundle, plan, selected_mode) if plan else None
    return {
        "execution_contract": contract,
        "compatibility_digest": compatibility_digest(runtime) if runtime else None,
        "runtime_schema_version": "native-runtime-v2",
        "feature_details": details,
        "weights_identity": getattr(runtime, "weight_identity", None),
        "engine_mode": selected_mode, "configured_execution_dtype": execution_dtype,
        "model_revision": bundle.revision,
        "scope": asdict(scope), "actual_weight_dtypes": weights,
        "layers": layers,
        "arena": (runtime.block_pool.stats()
                  if runtime and runtime.config.backend == "paged" else None),
        "qualified_workload_envelope": getattr(runtime, "qualified_envelope", None),
        "memory_admission": getattr(runtime, "capacity_diagnostics", None),
        "execution_memory_forecast": getattr(runtime, "last_execution_forecast", None),
        "component_capabilities": component_capabilities(bundle, runtime, selected_mode),
        "configured_cache_precision": runtime.precision_configuration() if runtime else None,
        "capabilities": {"continuous_batching": selected_mode in {
            "experimental_dense_batch", "paged", "experimental_hybrid_batch"},
                         "paged_attention": selected_mode == "paged",
                         "hybrid_prefix_restore": bool(getattr(runtime, "hybrid_state", False))},
        "runner_qualification": runner_features,
        "qualification": qualification,
        "latency": {"ttft_p95_ms": None, "decode_gap_p95_ms": None},
        "throughput": {"aggregate_decode_tok_s": None},
    }
