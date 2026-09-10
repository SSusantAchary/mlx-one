"""Shared bidirectional encoder and sentence-embedding operations."""

from __future__ import annotations

from typing import Any


def bidirectional_attention_mask(attention_mask: Any, dtype: Any) -> Any:
    """Create a broadcastable additive key-padding mask."""

    import mlx.core as mx

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    return mx.where(attention_mask[:, None, None, :], mx.array(0, dtype=dtype), -1e9)


def masked_mean_pool(hidden: Any, attention_mask: Any) -> Any:
    import mlx.core as mx

    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden and attention_mask must be rank three and rank two")
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError("attention_mask must match hidden batch and sequence dimensions")
    mask = attention_mask[..., None].astype(hidden.dtype)
    total = mx.sum(hidden * mask, axis=1)
    count = mx.sum(mask, axis=1)
    return total / mx.maximum(count, mx.array(1e-9, dtype=hidden.dtype))


def l2_normalize(value: Any) -> Any:
    import mlx.core as mx

    norm = mx.sqrt(mx.sum(value * value, axis=-1, keepdims=True))
    return value / mx.maximum(norm, mx.array(1e-12, dtype=value.dtype))


def cosine_similarity(left: Any, right: Any) -> Any:
    """Return the all-pairs cosine similarity matrix for two batches."""

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("cosine inputs must have shape [batch, dimension]")
    if left.shape[-1] != right.shape[-1]:
        raise ValueError("cosine input dimensions must match")
    return l2_normalize(left) @ l2_normalize(right).T


def validate_dropout(name: str, value: float) -> None:
    from mlx_one.core.config import ConfigError

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number in [0, 1)")
    if not 0 <= value < 1:
        raise ConfigError(f"{name} must be a number in [0, 1)")
