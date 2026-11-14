"""Select immutable execution settings before allocating model-scoped storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any

from mlx_one.engine.cache_manager import CacheManager, _digest, _model_identity
from mlx_one.engine.cache_planner import build_cache_plan
from mlx_one.engine.modes import resolve_backend, resolved_mode, validate_mode


@dataclass(frozen=True)
class CachePlanningContext:
    bundle: Any
    config: Any
    plan: Any
    weight_identity: str
    _model_identity: str
    batching_config: Any
    pressure: Any
    mode: str | None = None

    fingerprint = CacheManager.fingerprint

    def for_mode(self, mode):
        return replace(self, mode=mode)

    def state_observations(self):
        return []

    def precision_configuration(self):
        return {"key_type": self.config.key_type, "value_type": self.config.value_type}


@dataclass(frozen=True)
class ResolvedExecutionPlan:
    mode: str
    config: Any
    selection: Any


def construct_runtime(bundle, config, mode, *, batching=None, pressure=None,
                      registry=None, profile=None, mlxboost_config=None):
    from mlx_one.mlxboost import MLXBoostConfig

    boost = mlxboost_config or MLXBoostConfig()
    if not isinstance(boost, MLXBoostConfig):
        raise TypeError("mlxboost_config must be MLXBoostConfig")
    if mode != "mlxboost_b1" and boost.enabled:
        raise ValueError("MLXBoost features require explicit inference_mode=mlxboost_b1")
    boost_reason = None
    requested_mode = mode
    if mode == "mlxboost_b1":
        import sys
        from importlib.metadata import version

        mx = sys.modules.get("mlx.core")
        precision = getattr(bundle.model, "_mlx_one_execution_policy", {})
        dtype = precision.get("dtype")
        gptoss_boost = boost.gptoss_prefill_sort or boost.gptoss_mlx_lm
        if gptoss_boost and dtype is None:
            embedding = getattr(getattr(bundle.model, "model", None), "embed_tokens", None)
            scales = getattr(embedding, "scales", None)
            if scales is not None:
                dtype = str(scales.dtype).rsplit(".", 1)[-1]
        boost, boost_failure = boost.resolve(
            family=bundle.architecture, dtype=dtype,
            quantization=getattr(bundle, "quantization", {}) or {}, backend=config.backend,
            slots=getattr(batching, "max_sequences", 1),
            cache_reuse=config.prefix_cache_enabled and config.prefix_cache_budget_bytes > 0,
            mlx_version=version("mlx"),
            model_config=getattr(bundle.model, "config", None),
            device_info=mx.device_info() if gptoss_boost and mx else None,
            cache_types=(config.key_type, config.value_type),
            metal_available=bool(mx and mx.metal.is_available()
                                 and mx.default_device() == mx.gpu))
        if boost_failure:
            boost_reason = boost_failure
        elif boost.gptoss_mlx_lm:
            boost_reason = "experimental pinned MLX-LM MoE/prefill; source-aligned math"
        elif boost.gptoss_prefill_sort:
            boost_reason = "experimental GPT-OSS prefill sorting; decode remains unchanged"
        else:
            boost_reason = "experimental host lookup; numerical/model qualification is separate"
        if boost_failure is not None:
            mode = "reference_dense"
    weight_identity = _model_identity(bundle)
    context = CachePlanningContext(
        bundle, config, build_cache_plan(bundle, block_size_tokens=config.block_size_tokens),
        weight_identity, _digest(weight_identity + str(id(bundle.model))), batching, pressure,
        mode if mode != "auto" else None)
    selection = {"requested_mode": mode, "resolved_mode": mode or resolved_mode(config.backend),
                 "reason": "explicit configuration", "evidence_scope": None}
    if mode == "auto":
        from mlx_one.engine.selection import select_auto_mode

        selection = select_auto_mode(bundle, context, registry, profile)
    selected = selection["resolved_mode"]
    if boost_reason is not None:
        selection.update(requested_mode=requested_mode, reason=boost_reason,
                         mlxboost_flags=asdict(boost), qualification="EXPERIMENTAL")
    final_config = replace(config, backend=resolve_backend(selected, None))
    validate_mode(bundle, context.plan, selected)
    plan = ResolvedExecutionPlan(selected, final_config, MappingProxyType(selection))
    runtime = CacheManager(bundle, final_config, _planning=context)
    runtime.mlxboost_config = boost
    runtime.execution_plan = plan
    runtime.mode, runtime.requested_mode = selected, requested_mode
    runtime.pressure, runtime.batching_config = pressure, batching
    runtime.qualification_registry, runtime.qualification_profile = registry, profile
    try:
        if profile and (mode == "auto" and selection["evidence_scope"] is not None
                        or config.key_type in {"q4_0", "q8_0"}):
            from mlx_one.engine.selection import validate_execution_envelope

            if batching is None:
                raise ValueError("qualified actor execution requires a bound scheduler")
            validate_execution_envelope(runtime, batching, required=True)
    except BaseException:
        runtime.close()
        raise
    return runtime, dict(plan.selection)
