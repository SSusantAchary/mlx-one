"""Reusable layers for native model families."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DecoderLayer": ("mlx_one.models.shared.transformer", "DecoderLayer"),
    "GroupedQueryAttention": (
        "mlx_one.models.shared.transformer",
        "GroupedQueryAttention",
    ),
    "HybridDecoderLayer": ("mlx_one.models.shared.hybrid", "HybridDecoderLayer"),
    "LfmAttention": ("mlx_one.models.shared.hybrid", "LfmAttention"),
    "LfmMLP": ("mlx_one.models.shared.hybrid", "LfmMLP"),
    "SparseMoeBlock": ("mlx_one.models.shared.moe", "SparseMoeBlock"),
    "StackedExperts": ("mlx_one.models.shared.moe", "StackedExperts"),
    "ShortConv": ("mlx_one.models.shared.hybrid", "ShortConv"),
    "SwiGLU": ("mlx_one.models.shared.transformer", "SwiGLU"),
}

__all__ = [
    "DecoderLayer",
    "GroupedQueryAttention",
    "HybridDecoderLayer",
    "LfmAttention",
    "LfmMLP",
    "SparseMoeBlock",
    "StackedExperts",
    "ShortConv",
    "SwiGLU",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
