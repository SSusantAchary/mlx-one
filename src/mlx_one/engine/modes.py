"""Explicit selection of existing execution paths; no automatic backend substitution."""

from __future__ import annotations

from typing import Literal

from mlx_one.engine.cache_specs import CACHE_BACKENDS

InferenceMode = Literal["reference_dense", "experimental_block", "experimental_dense_batch",
                        "paged", "experimental_hybrid_batch", "auto"]

ACTOR_MODES = {"experimental_dense_batch", "paged", "experimental_hybrid_batch", "auto"}


class ModelCapabilityUnsupported(ValueError):
    code = "MODEL_CAPABILITY_UNSUPPORTED"


def resolve_backend(mode: str | None, backend: str | None) -> str:
    modes = {"reference_dense": "dense", "experimental_block": "block",
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
    if mode == "paged":
        from mlx_one.engine.reporting import tensor_dtypes

        parameters = getattr(bundle.model, "parameters", None)
        observed = tensor_dtypes(parameters()) if callable(parameters) else ()
        if (plan is None or plan.architecture != "qwen2" or not plan.block_compatible
                or bool(getattr(bundle, "quantization", {})) or not observed
                or any(d not in {"float16", "bfloat16"} for d in observed)):
            raise ModelCapabilityUnsupported("paged requires unquantized FP16/BF16 Qwen2/Qwen2.5")
    if mode == "experimental_hybrid_batch":
        if (plan is None or plan.architecture != "lfm2" or not plan.layer_specs
                or any(s.kind.value not in {"attention_kv", "conv_state", "rotating_kv"}
                       for s in plan.layer_specs)):
            raise ModelCapabilityUnsupported("hybrid batching requires native LFM2 attention/conv")
    if mode == "experimental_dense_batch":
        if (plan is None or plan.architecture != "qwen2" or not plan.block_compatible
                or bool(getattr(bundle, "quantization", {}))):
            raise ModelCapabilityUnsupported(
                "experimental_dense_batch requires unquantized Qwen2/Qwen2.5")
        validate_batch_precision(bundle)
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
    if execution_dtype not in {"float16", "bfloat16"}:
        raise ModelCapabilityUnsupported("execution dtype must be float16 or bfloat16")
    if getattr(bundle, "quantization", {}):
        raise ModelCapabilityUnsupported("explicit FP16 execution requires unquantized weights")
    import mlx.core as mx

    bundle.model.set_dtype(getattr(mx, execution_dtype))
    mx.eval(bundle.model.parameters())


def validate_batch_precision(bundle) -> None:
    from mlx_one.engine.reporting import tensor_dtypes

    parameters = getattr(bundle.model, "parameters", None)
    observed = tensor_dtypes(parameters()) if callable(parameters) else ()
    if not observed or any(dtype != "float16" for dtype in observed):
        raise ModelCapabilityUnsupported(
            "experimental_dense_batch requires observed FP16 weights; "
            "select execution_dtype=float16"
        )
