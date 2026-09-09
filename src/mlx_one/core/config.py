"""Backend-free validation helpers for native architecture configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ConfigError(ValueError):
    """Raised when model metadata cannot describe a supported architecture."""


def require_positive(name: str, value: int | float) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be positive")


def require_divisible(name: str, value: int, divisor_name: str, divisor: int) -> None:
    require_positive(name, value)
    require_positive(divisor_name, divisor)
    if value % divisor:
        raise ConfigError(f"{name} must be divisible by {divisor_name}")


def validate_rope_scaling(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    result = dict(value)
    rope_type = result.get("rope_type", result.get("type", "default"))
    if rope_type not in {"default", "linear", "mrope"}:
        raise ConfigError(f"unsupported RoPE type: {rope_type}")
    if rope_type == "linear":
        factor = result.get("factor")
        if not isinstance(factor, (int, float)) or factor <= 0:
            raise ConfigError("linear RoPE scaling requires a positive factor")
    if rope_type == "mrope":
        sections = result.get("mrope_section")
        if not isinstance(sections, (list, tuple)) or len(sections) != 3:
            raise ConfigError("mrope requires three mrope_section values")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in sections
        ):
            raise ConfigError("mrope_section values must be positive integers")
    return result


def extras(data: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in known}
