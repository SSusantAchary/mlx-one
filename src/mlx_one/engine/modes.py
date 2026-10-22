"""Explicit selection of existing execution paths; no automatic backend substitution."""

from __future__ import annotations

from typing import Literal

from mlx_one.engine.cache_specs import CACHE_BACKENDS

InferenceMode = Literal["reference_dense", "native_optimized", "experimental_block",
                        "experimental_dense_batch",
                        "paged", "experimental_hybrid_batch", "auto"]

ACTOR_MODES = {"experimental_dense_batch", "paged", "experimental_hybrid_batch", "auto"}


class ModelCapabilityUnsupported(ValueError):
    code = "MODEL_CAPABILITY_UNSUPPORTED"


def resolve_backend(mode: str | None, backend: str | None) -> str:
    modes = {"reference_dense": "dense", "native_optimized": "dense", "experimental_block": "block",
             "experimental_dense_batch": "dense", "paged": "paged",
             "experimental_hybrid_batch": "hybrid", "auto": "dense"}
    if mode is not None and mode not in modes:
        raise ModelCapabilityUnsupported(f"unsupported inference mode: {mode}")
    if backend is not None and backend not in CACHE_BACKENDS:
        raise ValueError("unsupported cache backend")
    if mode is None and backend in {"paged", "hybrid"}:
        raise ModelCapabilityUnsupported("paged/hybrid backends require an explicit inference mode")
    selected = modes.get(mode) if mode else None
    if selected and backend is not None and selected != backend:
        raise ValueError("inference mode conflicts with explicit cache backend")
    return selected or backend or "dense"


def resolved_mode(backend: str) -> str:
    return "experimental_block" if backend == "block" else "reference_dense"


def validate_mode(bundle, plan, mode: str | None) -> None:
    if getattr(getattr(bundle, "model", None), "_mlx_one_qwen3_candidate", False):
        from mlx_one.models.language.qwen3.upstream_adapter import validate_candidate_runtime

        validate_candidate_runtime(bundle, mode=mode)
    if mode == "paged":
        from mlx_one.engine.adapters import native_adapter

        try:
            adapter = native_adapter(bundle, plan, allow_bfloat16=True, allow_hybrid=True)
            if not adapter.supports_paging:
                raise ValueError("adapter has no direct paged attention dispatch")
            if (adapter.spec.hybrid and bundle.quantization
                    and bundle.quantization.get("bits") != 8):
                raise ValueError("paged hybrid execution requires native affine INT8 weights")
            if adapter.spec.hybrid and "sliding_kv" in adapter.spec.state_kinds:
                raise ValueError("hybrid paging does not support sliding attention")
            for layer in adapter.spec.layers:
                if layer.state_kind not in {"full_kv","sliding_kv"}:
                    continue
                if (not 1 <= layer.key_dim <= 256 or layer.key_dim != layer.value_dim
                        or not layer.kv_heads or not layer.query_heads
                        or layer.query_heads % layer.kv_heads):
                    raise ValueError(
                        "paged layers require equal K/V dimensions 1..256 and valid GQA")
        except (ValueError, AttributeError) as exc:
            raise ModelCapabilityUnsupported(str(exc)) from exc
    if mode == "experimental_hybrid_batch":
        from mlx_one.engine.adapters import native_adapter

        try:
            adapter = native_adapter(bundle, plan, allow_hybrid=True, allow_bfloat16=True)
            if not adapter.spec.hybrid:
                raise ValueError("hybrid batching requires a supported compound-state adapter")
        except (ValueError, AttributeError) as exc:
            raise ModelCapabilityUnsupported(str(exc)) from exc
    if mode == "experimental_dense_batch":
        from mlx_one.engine.adapters import native_adapter

        try:
            native_adapter(bundle, plan)
        except (ValueError, AttributeError) as exc:
            raise ModelCapabilityUnsupported(str(exc)) from exc
    if mode == "experimental_block" and (
        plan is None or plan.architecture != "qwen2" or not plan.block_compatible
        or bool(getattr(bundle, "quantization", {}))
    ):
        raise ModelCapabilityUnsupported(
            "experimental_block requires unquantized Qwen2/Qwen2.5 attention topology"
        )


def configure_execution_dtype(bundle, execution_dtype: str | None) -> None:
    if execution_dtype is None:
        return
    if execution_dtype not in {"float32", "float16", "bfloat16"}:
        raise ModelCapabilityUnsupported("execution dtype must be float32, float16 or bfloat16")
    quantized = bool(getattr(bundle, "quantization", {}))
    if quantized:
        from mlx_one.engine.adapters import quantized_modules

        try:
            quantized_modules(bundle, allowed_bits=(4, 8))
        except ValueError as exc:
            raise ModelCapabilityUnsupported(str(exc)) from exc
    import mlx.core as mx

    before = None
    previous = getattr(bundle.model, "_mlx_one_execution_policy", {})
    if quantized:
        from mlx_one.engine.reporting import tensor_dtypes

        before = tensor_dtypes(bundle.model.parameters())
        packed = [(m, m.weight) for _, m in bundle.model.named_modules() if hasattr(m, "bits")]
        bundle.model.set_dtype(getattr(mx, execution_dtype), predicate=lambda d:
                               str(d).rsplit(".", 1)[-1] in {"float16", "bfloat16", "float32"})
        if any(m.weight is not original for m, original in packed):
            raise ModelCapabilityUnsupported("floating conversion modified packed weights")
    else:
        bundle.model.set_dtype(getattr(mx, execution_dtype))
    mx.eval(bundle.model.parameters())
    from mlx_one.engine.adapters import EXECUTION_POLICY_VERSION

    bundle.model._mlx_one_execution_policy = {
        "version": EXECUTION_POLICY_VERSION, "dtype": execution_dtype,
        "hybrid_attention_softmax": "float32-before-mask-v1"
            if getattr(bundle, "architecture", None) in {"lfm2", "lfm2_moe", "qwen3_5"} else None,
        "hybrid_affine_projection": "rowwise-gather-qmm-v1"
            if execution_dtype == "float16" and quantized and
            getattr(bundle, "architecture", None) in {"lfm2", "lfm2_moe", "qwen3_5"} else None,
        "quantized_payload_preserved": quantized,
        "affine_int8_fp16_arithmetic": {
            "shared_attention_swiglu_projections": "rowwise-gather-qmm-v1",
            "shared_dense_attention": "rowwise-fp32-gather-mm-v1",
            "ragged_attention_reduction": "compact-real-row-v1",
            "qwen3_logits": "rowwise-gather-qmm-v1"
                if getattr(bundle, "architecture", None) == "qwen3" else "native",
            "activation_dtype": "float16", "kv_dtype": "float16"}
        if execution_dtype == "float16" and quantized
            and getattr(bundle, "architecture", None) in {"qwen2", "qwen3", "llama"} else None,
        "previous_dtypes": previous.get("previous_dtypes", before)
        if previous.get("dtype") == execution_dtype else before}
    if getattr(bundle.model, "_mlx_one_reference_first_math", False):
        bundle.model._mlx_one_execution_policy.update({
            "affine_int8_fp16_arithmetic": None,
            "hybrid_attention_softmax": None,
            "hybrid_affine_projection": None,
            "reference_first_arithmetic": {
                "projection": "mlx-nn-module-dispatch",
                "attention": "mlx-fast-sdpa",
                "activation": "family-reference-order",
                "qualification": "requires-checkpoint-gates",
            },
        })
    if getattr(bundle.model, "_mlx_one_qwen3_candidate", False):
        # This explicit candidate uses upstream module dispatch and fused SDPA,
        # rather than the existing canonical model's gathered reference math.
        bundle.model._mlx_one_execution_policy["affine_int8_fp16_arithmetic"] = None
        bundle.model._mlx_one_execution_policy["qwen3_upstream_native_arithmetic"] = {
            "upstream_commit": "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd",
            "projection": "mlx-nn-module-dispatch",
            "attention": "mlx-fast-sdpa",
            "swiglu": "upstream-shapeless-compiled-silu-times-up",
            "promotion": "CANDIDATE_ONLY",
        }
    named = getattr(bundle.model, "named_modules", None)
    if callable(named):
        for _, module in named():
            module._mlx_one_activation_dtype = execution_dtype


def validate_batch_precision(bundle) -> None:
    from mlx_one.engine.adapters import native_adapter

    try:
        native_adapter(bundle)
    except (ValueError, AttributeError) as exc:
        raise ModelCapabilityUnsupported(str(exc)) from exc
