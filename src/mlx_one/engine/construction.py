"""Select immutable execution settings before allocating model-scoped storage."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
                      registry=None, profile=None):
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
    final_config = replace(config, backend=resolve_backend(selected, None))
    validate_mode(bundle, context.plan, selected)
    plan = ResolvedExecutionPlan(selected, final_config, MappingProxyType(selection))
    runtime = CacheManager(bundle, final_config, _planning=context)
    runtime.execution_plan = plan
    runtime.mode, runtime.requested_mode = selected, mode
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
