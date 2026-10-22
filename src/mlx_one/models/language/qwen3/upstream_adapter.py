"""Qwen3 candidate bridge to existing MLX-One cache ownership and runtime bounds."""

from __future__ import annotations

from mlx_one.core.cache import KVCache, QuantizedKVCache
from mlx_one.engine.modes import ModelCapabilityUnsupported


class DenseCacheView:
    """Borrow the owner's offset/update API; never own, clone or release its state."""

    def __init__(self, owner):
        if type(owner) not in {KVCache, QuantizedKVCache} or (
            isinstance(owner, QuantizedKVCache)
            and (owner.key_bits is not None or owner.value_bits is not None)
        ):
            raise ModelCapabilityUnsupported("Qwen3 candidate requires native dense KV storage")
        self.owner = owner

    @property
    def offset(self):
        return self.owner.offset

    def update_and_fetch(self, keys, values):
        return self.owner.update(keys, values)

    def make_mask(self, length):
        return None if length == 1 else "causal"


def cache_views(cache, layer_count):
    if cache is None:
        return None
    if len(cache) != layer_count:
        raise ValueError("cache count must match decoder layer count")
    if len({id(entry) for entry in cache}) != layer_count:
        raise ValueError("candidate decoder layers require distinct cache owners")
    views = tuple(DenseCacheView(entry) for entry in cache)
    if len({entry.offset for entry in views}) != 1:
        raise ValueError("candidate cache positions must agree across layers")
    return views


def validate_candidate_runtime(
    bundle,
    *,
    mode=None,
    backend="dense",
    key_type="f16",
    value_type="f16",
    max_sequences=1,
    spec_type="none",
    math_policy=None,
):
    if not getattr(bundle.model, "_mlx_one_qwen3_candidate", False):
        return
    from mlx_one.engine.optimization_policy import REFERENCE_MATH

    if (
        mode not in {None, "reference_dense"}
        or backend != "dense"
        or key_type not in {"f16", "bf16"}
        or value_type != key_type
        or max_sequences != 1
        or spec_type != "none"
        or math_policy is not None
        and math_policy != REFERENCE_MATH
    ):
        raise ModelCapabilityUnsupported(
            "Qwen3 upstream candidate supports reference dense B1, native KV and no speculation"
        )


def load_upstream_candidate(model, **kwargs):
    """Explicit development opt-in; the canonical registry and default loader stay intact."""
    from mlx_one.text.loading import load_text_model

    return load_text_model(model, _qwen3_candidate=True, **kwargs)
