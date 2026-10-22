"""Select vocabulary projection inputs without truncating returned hidden states."""
from __future__ import annotations

from enum import Enum

from mlx_one.engine.optimization_policy import current_math


class LogitsMode(str, Enum):
    ALL = "all"
    LAST_ONLY = "last_only"
    NO_LOGITS = "no_logits"


def resolve_logits_mode(mode=None):
    return LogitsMode(mode if mode is not None else current_math().policy.logits)


def logits_input(hidden, mode=None, *, attention_mask=None):
    mode = resolve_logits_mode(mode)
    if mode == LogitsMode.NO_LOGITS:
        return None
    if mode == LogitsMode.ALL:
        return hidden
    if hidden.ndim != 3 or hidden.shape[1] < 1:
        raise ValueError("last-only logits require nonempty [B,T,D] hidden states")
    if attention_mask is None:
        return hidden[:, -1:, :]
    import mlx.core as mx

    mask = attention_mask[:, -hidden.shape[1]:]
    if mask.ndim != 2 or mask.shape != hidden.shape[:2]:
        raise ValueError("logits padding mask must match [B,T]")
    if not bool(mx.all(mx.any(mask != 0, axis=-1)).item()):
        raise ValueError("last-only logits require a real token in every row")
    indices = mx.max(mx.where(mask != 0, mx.arange(hidden.shape[1])[None, :], -1), axis=-1)
    return hidden[mx.arange(hidden.shape[0]), indices][:, None, :]


def project_logits(hidden, projection, mode=None, *, attention_mask=None):
    value = logits_input(hidden, mode, attention_mask=attention_mask)
    if value is None:
        return None
    from mlx_one.ops.profiling import phase

    result = None
    with phase("logits", output=lambda: result):
        result = projection(value)
    return result


def native_projection(module, value, *, as_linear=False):
    """Keep native head arithmetic as the reference; opt in only for packed QMM."""
    from mlx_one.ops.quantized_linear import try_quantized_linear

    used, result = try_quantized_linear(module, value, as_linear=as_linear)
    if used:
        return result
    return module.as_linear(value) if as_linear else module(value)
