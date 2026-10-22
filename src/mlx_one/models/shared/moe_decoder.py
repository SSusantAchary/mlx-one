"""Validate MoE decoder state before any layer can mutate a request's cache."""

from mlx_one.core.cache import ConvCache, KVCache, QuantizedKVCache


def validate_decoder_state(model, tokens, cache, attention_mask, position_ids):
    from mlx_one.models.shared.decoder import prepare_decoder

    if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
        raise ValueError("input_ids must be nonempty [batch, sequence]")
    packed_rows = False
    if cache is not None:
        if len(cache) != len(model.layers):
            raise ValueError("cache count must match decoder layers")
        for index, state in enumerate(cache):
            conv = model.config.layer_types[index] == "conv"
            rows = getattr(state, "rows", ())
            if rows and conv and all(isinstance(row, ConvCache) for row in rows):
                if len(rows) != tokens.shape[0] or any(
                    row.kernel_size != model.config.conv_L_cache for row in rows
                ):
                    raise ValueError(f"invalid packed convolution state at layer {index}")
                packed_rows = True
                continue
            if not isinstance(state, ConvCache if conv else (KVCache, QuantizedKVCache)):
                # Qualified engine attention components can own their own dispatch.
                if conv or not callable(getattr(state, "attend", None)):
                    raise ValueError(f"invalid cache state type at layer {index}")
                packed_rows = True
            offset = getattr(state, "offset", None)
            if type(offset) is not int or offset < 0:
                raise ValueError(f"invalid cache position at layer {index}")
            if conv and state.kernel_size != model.config.conv_L_cache:
                raise ValueError(f"invalid convolution cache kernel at layer {index}")
            if isinstance(state, (ConvCache, KVCache, QuantizedKVCache)):
                arrays = (
                    (state.values,)
                    if conv
                    else (state._keys, state._values)
                    if isinstance(state, QuantizedKVCache)
                    else (state.keys, state.values)
                )
                arrays = tuple(value[0] if isinstance(value, tuple) else value for value in arrays)
                if any(value is None for value in arrays):
                    if offset:
                        raise ValueError(f"missing committed cache state at layer {index}")
                elif any(value.shape[0] != tokens.shape[0] for value in arrays):
                    raise ValueError(f"cache batch shape mismatch at layer {index}")
                elif conv:
                    expected = (tokens.shape[0], state.kernel_size - 1, model.config.hidden_size)
                    if state.values.shape != expected:
                        raise ValueError(f"invalid convolution cache shape at layer {index}")
                else:
                    head = getattr(model.config, "head_dim", None) or (
                        model.config.hidden_size // model.config.num_attention_heads
                    )
                    retained = min(offset, getattr(state, "max_size", offset))
                    if any(
                        value.ndim != 4
                        or value.shape[1] != model.config.num_key_value_heads
                        or value.shape[2] < retained
                        or (not isinstance(state, QuantizedKVCache) and value.shape[-1] != head)
                        for value in arrays
                    ):
                        raise ValueError(f"invalid attention cache shape at layer {index}")
            window = (
                model.config.sliding_window
                if model.config.layer_types[index] == "sliding_attention"
                else None
            )
            if isinstance(state, KVCache) and getattr(state, "max_size", None) != window:
                raise ValueError(f"cache sliding window mismatch at layer {index}")
    # Ragged engine components already validate every row and own their masks;
    # their current-chunk padding mask need not span another row's history.
    return prepare_decoder(
        model, tokens, cache, None if packed_rows else attention_mask, position_ids
    )
