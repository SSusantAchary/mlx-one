"""Lazy, backend-free registry for native model architectures."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelRegistration:
    model_type: str
    config_path: str
    model_path: str
    modality: str
    capabilities: frozenset[str]
    sanitizer_path: str | None = None
    weight_contract_path: str | None = None
    loader_path: str | None = None
    tasks: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.model_type or not self.modality:
            raise ValueError("model registration requires architecture and modality")
        for path in (self.config_path, self.model_path, self.sanitizer_path,
                     self.weight_contract_path, self.loader_path):
            if path is not None:
                validate_import_path(path)
        if not isinstance(self.tasks, frozenset) or not isinstance(self.capabilities, frozenset):
            raise ValueError("registration tasks and capabilities must be immutable")

    def config_class(self) -> type[Any]:
        return _resolve(self.config_path)

    def model_class(self) -> type[Any]:
        return _resolve(self.model_path)

    def sanitizer(self) -> Any:
        if self.sanitizer_path is None:
            raise ValueError(f"model type {self.model_type!r} has no weight sanitizer")
        return _resolve_attribute(self.sanitizer_path)

    def weight_contract(self) -> Any:
        if self.weight_contract_path is None:
            raise ValueError(f"model type {self.model_type!r} has no weight contract")
        return _resolve_attribute(self.weight_contract_path)

    def loader(self) -> Any:
        if self.loader_path is None:
            raise ValueError(f"model type {self.model_type!r} has no native task loader")
        return _resolve_attribute(self.loader_path)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities or capability in self.tasks


_REGISTRY: dict[str, ModelRegistration] = {}
_BUILTINS_LOADED = False


def register_model(registration: ModelRegistration) -> None:
    current = _REGISTRY.get(registration.model_type)
    if current is not None and current != registration:
        raise ValueError(f"model type already registered: {registration.model_type}")
    _REGISTRY[registration.model_type] = registration


def get_registration(model_type: str) -> ModelRegistration:
    _ensure_builtins()
    try:
        return _REGISTRY[model_type]
    except KeyError as exc:
        raise KeyError(f"unsupported native model type: {model_type}") from exc


def registered_model_types() -> tuple[str, ...]:
    _ensure_builtins()
    return tuple(sorted(_REGISTRY))


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from mlx_one.models.registry import register_builtin_models

    register_builtin_models()
    _BUILTINS_LOADED = True


def _resolve(path: str) -> type[Any]:
    return _resolve_attribute(path)


def validate_import_path(path: str) -> None:
    """Validate lazy references without importing even an unselected local plugin."""
    module_name, separator, attribute = path.partition(":")
    if (not separator or not all(part.isidentifier() for part in module_name.split("."))
            or not attribute.isidentifier()):
        raise ValueError(f"invalid registry import path: {path}")


def _resolve_attribute(path: str) -> Any:
    validate_import_path(path)
    module_name, _, attribute = path.partition(":")
    return getattr(importlib.import_module(module_name), attribute)
