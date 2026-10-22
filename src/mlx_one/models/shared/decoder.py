"""Common B1 decoder traversal; layer mathematics and state stay family-owned."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import ModelOutput


class CausalLM(nn.Module):
    """Output-head scaffolding shared by new correctness-first decoder families."""

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=False,
    ):
        hidden, states = self.model(
            input_ids,
            cache=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.config.tie_word_embeddings
            else self.lm_head(hidden)
        )
        cap = getattr(self.config, "final_logit_softcapping", None)
        if cap is not None:
            logits = cap * mx.tanh(logits / cap)
        return ModelOutput(logits, hidden, cache=cache, hidden_states=states)


def prepare_decoder(model, input_ids, cache, attention_mask, position_ids):
    if input_ids.ndim != 2 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must be nonempty [batch, sequence]")
    if cache is not None and (
        len(cache) != len(model.layers) or len({c.offset for c in cache}) != 1
    ):
        raise ValueError("cache count and positions must match decoder layers")
    offset = 0 if cache is None else cache[0].offset
    if offset + input_ids.shape[1] > model.config.max_position_embeddings:
        raise ValueError("input exceeds checkpoint maximum positions")
    if attention_mask is not None:
        if tuple(attention_mask.shape) != (input_ids.shape[0], offset + input_ids.shape[1]):
            raise ValueError("attention_mask must cover the complete cached sequence")
        if position_ids is None:
            position_ids = mx.maximum(mx.cumsum(attention_mask, axis=-1) - 1, 0)
            position_ids = position_ids[:, -input_ids.shape[1] :]
    if position_ids is not None:
        if tuple(position_ids.shape) not in {(1, input_ids.shape[1]), tuple(input_ids.shape)}:
            raise ValueError("position_ids must match the token sequence")
        if mx.any(
            (position_ids < 0) | (position_ids >= model.config.max_position_embeddings)
        ).item():
            raise ValueError("position_ids exceed checkpoint maximum positions")
    return offset, position_ids


def attention_mask_for(length, offset, retained, *, window=None, padding=None, dtype=mx.float32):
    query = mx.arange(offset, offset + length)[:, None]
    key = mx.arange(offset - retained, offset + length)[None]
    allowed = key <= query
    if window is not None:
        allowed = allowed & (key > query - window)
    mask = mx.where(allowed, mx.array(0, dtype), mx.array(-float("inf"), dtype))
    if padding is not None:
        mask = mx.where(padding[:, None, None, -key.shape[1] :] != 0, mask, -float("inf"))
    return mask
