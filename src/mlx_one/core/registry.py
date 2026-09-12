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


def _resolve_attribute(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"invalid registry import path: {path}")
    return getattr(importlib.import_module(module_name), attribute)
