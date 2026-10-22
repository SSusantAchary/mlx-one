"""Compatibility planner for existing mlx-one model cache implementations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from mlx_one.engine.cache_specs import CacheKind, CachePlan, LayerCacheSpec
from mlx_one.engine.contracts import LayerExecutionSpec


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
    execution_dtype = getattr(model, "_mlx_one_activation_dtype", None)
    if execution_dtype in {"float16", "bfloat16"}:
        specs = tuple(replace(spec, state_dtypes=(execution_dtype,))
                      if spec.kind is CacheKind.CONV_STATE and not spec.state_dtypes else spec
                      for spec in specs)
    declared = getattr(model, "layer_execution_specs", None)
    if callable(declared):
        layouts = declared()
        if (not isinstance(layouts, tuple) or len(layouts) != len(specs)
                or any(not isinstance(s, LayerExecutionSpec) or s.layer_id != i
                       for i, s in enumerate(layouts))):
            raise ValueError("model layer execution specifications must align with cache topology")
        kinds = {"full_kv": CacheKind.ATTENTION_KV, "sliding_kv": CacheKind.ROTATING_KV,
                 "conv": CacheKind.CONV_STATE, "recurrent": CacheKind.RECURRENT_STATE,
                 "cross_kv": CacheKind.CROSS_ATTENTION}
        specs = tuple(replace(old, kind=kinds[new.state_kind], num_query_heads=new.query_heads,
            num_kv_heads=new.kv_heads, key_head_dim=new.key_dim, value_head_dim=new.value_dim,
            max_window_tokens=new.window_tokens, position_policy=new.position_policy)
            for old, new in zip(specs, layouts, strict=True))
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

    layer_types = getattr(config, "layer_types", ())
    if index < len(layer_types) and layer_types[index] == "sliding_attention":
        kind = CacheKind.ROTATING_KV

    heads = _per_layer_int(config, index, "num_key_value_heads", "num_kv_heads", "n_head")
    head_dim = _per_layer_int(config, index, "head_dim")
    if head_dim is None:
        hidden = _per_layer_int(config, index, "hidden_size", "model_dim", "n_embd")
        query_heads = _per_layer_int(
            config, index, "num_attention_heads", "num_query_heads", "n_head"
        )
        if hidden and query_heads:
            head_dim = hidden // query_heads
    attention = kind in {CacheKind.ATTENTION_KV, CacheKind.ROTATING_KV}
    return LayerCacheSpec(
        layer_id=index,
        kind=kind,
        num_kv_heads=heads if attention else None,
        key_head_dim=head_dim if attention else None,
        value_head_dim=head_dim if attention else None,
        # An empty cache does not establish its eventual tensor dtype. Storage
        # precision is bound by the runtime config and observed at execution.
        dtype=None,
        max_window_tokens=(getattr(config, "sliding_window", None)
                           if kind is CacheKind.ROTATING_KV else getattr(cache, "max_size", None)),
        quantizable=attention and getattr(cache, "supports_quantization", True),
        shareable_prefix=attention,
        trimmable=callable(getattr(cache, "trim", None)),
        batchable=attention and getattr(cache, "supports_batching", True),
        block_compatible=attention and heads is not None and head_dim is not None
        and getattr(cache, "supports_batching", True),
        state_shapes=((1, int(cache.kernel_size) - 1,
                       _per_layer_int(config, index, "hidden_size") or 0),)
        if kind is CacheKind.CONV_STATE and hasattr(cache, "kernel_size")
        else tuple(getattr(cache,"state_shapes",())),
        checkpoint_semantics="exact_committed_boundary" if kind is CacheKind.CONV_STATE
        else getattr(cache,"checkpoint_semantics",None),
        state_dtypes=tuple(getattr(cache,"state_dtypes",())),
        num_query_heads=_per_layer_int(config, index, "num_attention_heads", "num_query_heads",
                                      "n_head") if attention else None,
        position_policy="learned" if getattr(config, "model_type", None) == "gpt2" else "rope",
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
