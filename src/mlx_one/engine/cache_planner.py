"""Compatibility planner for existing mlx-one model cache implementations."""

from __future__ import annotations

from typing import Any

from mlx_one.engine.cache_specs import CacheKind, CachePlan, LayerCacheSpec


def build_cache_plan(bundle: Any, *, block_size_tokens: int = 32) -> CachePlan:
    model = bundle.model
    custom = getattr(model, "cache_plan", None)
    if callable(custom):
        plan = custom(block_size_tokens=block_size_tokens)
        if not isinstance(plan, CachePlan):
            raise TypeError("model cache_plan() must return CachePlan")
        return plan
    prototype = tuple(model.make_cache())
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", config)
    specs = tuple(_layer_spec(index, cache, text) for index, cache in enumerate(prototype))
    return CachePlan(
        model_id=str(bundle.model_id),
        revision=bundle.revision,
        architecture=str(getattr(bundle, "architecture", type(model).__name__.lower())),
        layer_specs=specs,
        max_context_tokens=int(getattr(bundle, "context_length", 0) or 0),
        block_size_tokens=block_size_tokens,
    )


def _layer_spec(index: int, cache: Any, config: Any) -> LayerCacheSpec:
    name = type(cache).__name__.lower()
    if "encoderdecoder" in name:
        kind = CacheKind.CROSS_ATTENTION
    elif "quantized" in name:
        kind = CacheKind.QUANTIZED_KV
    elif "conv" in name:
        kind = CacheKind.CONV_STATE
    elif "linear" in name or "recurrent" in name or "mamba" in name:
        kind = CacheKind.RECURRENT_STATE
    elif "kvcache" in name or (hasattr(cache, "keys") and hasattr(cache, "values")):
        kind = CacheKind.ATTENTION_KV
    else:
        kind = CacheKind.MODEL_DEFINED

    heads = _per_layer_int(config, index, "num_key_value_heads", "num_kv_heads", "n_head")
    head_dim = _per_layer_int(config, index, "head_dim")
    if head_dim is None:
        hidden = _per_layer_int(config, index, "hidden_size", "model_dim", "n_embd")
        query_heads = _per_layer_int(
            config, index, "num_attention_heads", "num_query_heads", "n_head"
        )
        if hidden and query_heads:
            head_dim = hidden // query_heads
    attention = kind is CacheKind.ATTENTION_KV
    return LayerCacheSpec(
        layer_id=index,
        kind=kind,
        num_kv_heads=heads if attention else None,
        key_head_dim=head_dim if attention else None,
        value_head_dim=head_dim if attention else None,
        dtype="f16" if attention else None,
        max_window_tokens=getattr(cache, "max_size", None),
        quantizable=attention,
        shareable_prefix=attention,
        trimmable=callable(getattr(cache, "trim", None)),
        batchable=attention,
        block_compatible=attention and heads is not None and head_dim is not None,
    )


def _per_layer_int(config: Any, index: int, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        if isinstance(value, (tuple, list)) and index < len(value):
            item = value[index]
            if isinstance(item, int) and not isinstance(item, bool) and item > 0:
                return item
    return None
