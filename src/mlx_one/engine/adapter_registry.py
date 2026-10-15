"""Versioned lazy task/adapter bindings, separate from architecture registration."""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from mlx_one.core.registry import get_registration, registered_model_types, validate_import_path
from mlx_one.engine.contracts import LayerExecutionSpec, TaskSpec, canonical_task

RUNTIME_CONTRACT_VERSION = "registry-runtime-v1"


@dataclass(frozen=True)
class RuntimeRegistration:
    model_type: str
    adapter_path: str | None
    task_specs_path: str
    state_specs_path: str
    contract_version: str
    component_only: bool = False
    integration_status: str = "not_integrated"
    reason: str = "Task service integration has not been implemented"
    parent_model_type: str | None = None
    task_adapters: tuple[tuple[object, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.model_type or not self.contract_version or not self.reason:
            raise ValueError("runtime registration requires identity, version and support reason")
        for path in (self.adapter_path, self.task_specs_path, self.state_specs_path):
            if path is not None:
                validate_import_path(path)
        if len({task for task,_ in self.task_adapters})!=len(self.task_adapters):
            raise ValueError("duplicate task adapter bindings")
        for task,path in self.task_adapters:
            canonical_task(task)
            validate_import_path(path)
        statuses = {"not_integrated", "loader_declared", "architecture_only", "component_only",
                    "integrated"}
        if self.integration_status not in statuses:
            raise ValueError("invalid runtime integration status")
        if self.integration_status == "integrated" and self.adapter_path is None:
            raise ValueError("integrated task service requires an adapter")
        if self.component_only != (self.integration_status == "component_only") or (
                self.component_only != (self.parent_model_type is not None)):
            raise ValueError("nested component requires explicit parent and component-only status")

    def adapter(self,task=None):
        if task is not None:
            selected=dict(self.task_adapters).get(canonical_task(task))
            if selected:
                return _resolve(selected)
        if self.adapter_path is None:
            raise ValueError(f"{self.model_type} task-service adapter unavailable: {self.reason}")
        return _resolve(self.adapter_path)

    def specifications(self) -> tuple[tuple[TaskSpec, ...], tuple[LayerExecutionSpec, ...]]:
        """Resolve selected metadata only. Model-dependent state planning is a later step."""
        architecture = get_registration(self.model_type)
        tasks = _resolve(self.task_specs_path)(architecture)
        states = _resolve(self.state_specs_path)(architecture)
        if not isinstance(tasks, tuple) or not all(isinstance(t, TaskSpec) for t in tasks):
            raise ValueError("runtime task provider must return immutable TaskSpecs")
        if not isinstance(states, tuple) or not all(
                isinstance(s, LayerExecutionSpec) for s in states):
            raise ValueError("runtime state provider must return immutable LayerExecutionSpecs")
        if (len({t.task for t in tasks}) != len(tasks)
                or len({s.layer_id for s in states}) != len(states)):
            raise ValueError("duplicate task or layer specifications")
        declared = {canonical_task(task) for task in architecture.tasks}
        if any(task.task not in declared for task in tasks):
            raise ValueError("runtime task is not declared by architecture registration")
        if self.integration_status == "integrated" and (not architecture.loader_path or not tasks):
            raise ValueError("integrated runtime requires declared loader and typed tasks")
        return tasks, states


_RUNTIME_REGISTRY: dict[str, RuntimeRegistration] = {}


def register_runtime(registration: RuntimeRegistration) -> None:
    existing = _RUNTIME_REGISTRY.get(registration.model_type)
    if existing is not None and existing != registration:
        raise ValueError(f"runtime already registered: {registration.model_type}")
    _RUNTIME_REGISTRY[registration.model_type] = registration


def get_runtime_registration(model_type: str, *, contract_version=None) -> RuntimeRegistration:
    get_registration(model_type)  # Populate built-ins and reject architecture aliases.
    try:
        result = _RUNTIME_REGISTRY[model_type]
    except KeyError as exc:
        raise KeyError(f"architecture has no explicit runtime binding: {model_type}") from exc
    if contract_version is not None and result.contract_version != contract_version:
        raise ValueError("runtime contract version mismatch")
    return result


def registered_runtime_types() -> tuple[str, ...]:
    registered_model_types()
    return tuple(sorted(_RUNTIME_REGISTRY))


def runtime_coverage() -> tuple[dict, ...]:
    """Every architecture needs a binding; an unresolved plugin is never auto-enabled."""
    rows = []
    for model_type in registered_model_types():
        registration = get_runtime_registration(model_type)
        tasks, states = registration.specifications()
        rows.append({"model_type": model_type, "contract_version": registration.contract_version,
            "integration_status": registration.integration_status, "reason": registration.reason,
            "component_only": registration.component_only,
            "parent_model_type": registration.parent_model_type,
            "adapter_available": registration.adapter_path is not None,
            "shared_service_integrated": registration.integration_status == "integrated",
            "tasks": tuple(t.task.value for t in tasks),
            "declared_tasks": tuple(sorted(get_registration(model_type).tasks)),
            "state_spec_count": len(states), "correctness_qualified": False,
            "performance_qualified": False})
    return tuple(rows)


def _resolve(path):
    module, _, attribute = path.partition(":")
    return getattr(importlib.import_module(module), attribute)
