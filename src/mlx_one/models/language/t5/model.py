"""Native T5 encoder-decoder math with shared embeddings and relative attention.

Attention intentionally has no sqrt(head_dim) scaling. Greedy decoding reuses
encoder outputs and recomputes a bounded decoder prefix; no cache is advertised.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.language.t5.config import T5Config, relative_bucket


@dataclass
class T5Output:
    logits: Any
    encoder_last_hidden_state: Any
    decoder_last_hidden_state: Any


class T5Norm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = mx.ones((config.d_model,))
        self.eps = config.layer_norm_epsilon

    def __call__(self, x):
        variance = mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + self.eps) * self.weight


class T5Attention(nn.Module):
    def __init__(self, config, relative=False):
        super().__init__()
        self.config = config
        width = config.num_heads * config.d_kv
        self.q = nn.Linear(config.d_model, width, bias=False)
        self.k = nn.Linear(config.d_model, width, bias=False)
        self.v = nn.Linear(config.d_model, width, bias=False)
        self.o = nn.Linear(width, config.d_model, bias=False)
        self.relative_attention_bias = (
            nn.Embedding(config.relative_attention_num_buckets, config.num_heads)
            if relative
            else None
        )

    def __call__(self, hidden, keys, bias):
        batch, length, _ = hidden.shape
        heads, dim = self.config.num_heads, self.config.d_kv
        q = self.q(hidden).reshape(batch, length, heads, dim).transpose(0, 2, 1, 3)
        k = self.k(keys).reshape(batch, keys.shape[1], heads, dim).transpose(0, 2, 1, 3)
        v = self.v(keys).reshape(batch, keys.shape[1], heads, dim).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2) + bias
        result = mx.softmax(scores, axis=-1, precise=True) @ v
        return self.o(result.transpose(0, 2, 1, 3).reshape(batch, length, heads * dim))


class SelfAttentionLayer(nn.Module):
    def __init__(self, config, relative):
        super().__init__()
        self.SelfAttention = T5Attention(config, relative)
        self.layer_norm = T5Norm(config)

    def __call__(self, hidden, bias):
        value = self.layer_norm(hidden)
        return hidden + self.SelfAttention(value, value, bias)


class CrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config)
        self.layer_norm = T5Norm(config)

    def __call__(self, hidden, encoder, bias):
        return hidden + self.EncDecAttention(self.layer_norm(hidden), encoder, bias)


class DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)

    def __call__(self, hidden):
        return self.wo(nn.relu(self.wi(hidden)))


class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = DenseReluDense(config)
        self.layer_norm = T5Norm(config)

    def __call__(self, hidden):
        return hidden + self.DenseReluDense(self.layer_norm(hidden))


class T5Block(nn.Module):
    def __init__(self, config, decoder, relative):
        super().__init__()
        self.layer = [SelfAttentionLayer(config, relative)]
        if decoder:
            self.layer.append(CrossAttentionLayer(config))
        self.layer.append(FeedForwardLayer(config))


class T5Stack(nn.Module):
    def __init__(self, config, decoder=False):
        super().__init__()
        self.config, self.decoder = config, decoder
        count = config.num_decoder_layers if decoder else config.num_layers
        self.block = [T5Block(config, decoder, i == 0) for i in range(count)]
        self.final_layer_norm = T5Norm(config)

    def __call__(self, hidden, mask, encoder=None, encoder_mask=None):
        length = hidden.shape[1]
        buckets = [
            [
                relative_bucket(
                    k - q,
                    bidirectional=not self.decoder,
                    buckets=self.config.relative_attention_num_buckets,
                    distance=self.config.relative_attention_max_distance,
                )
                for k in range(length)
            ]
            for q in range(length)
        ]
        learned = self.block[0].layer[0].SelfAttention.relative_attention_bias(mx.array(buckets))
        bias = learned.transpose(2, 0, 1)[None]
        allowed = mask[:, None, None, :].astype(mx.bool_)
        if self.decoder:
            causal = mx.arange(length)[:, None] >= mx.arange(length)[None, :]
            allowed = allowed & causal[None, None]
        bias = bias + mx.where(allowed, 0.0, mx.finfo(hidden.dtype).min)
        cross_bias = (
            None
            if encoder is None
            else mx.where(
                encoder_mask[:, None, None, :].astype(mx.bool_), 0.0, mx.finfo(hidden.dtype).min
            )
        )
        for block in self.block:
            hidden = block.layer[0](hidden, bias)
            if self.decoder:
                hidden = block.layer[1](hidden, encoder, cross_bias)
            hidden = block.layer[-1](hidden)
        return self.final_layer_norm(hidden)


class T5ForConditionalGeneration(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config)
        self.decoder = T5Stack(config, decoder=True)

    def encode(self, input_ids, attention_mask=None):
        self._check_ids(input_ids)
        mask = input_ids != self.config.pad_token_id if attention_mask is None else attention_mask
        if mask.shape != input_ids.shape:
            raise ValueError("encoder attention mask must match input IDs")
        return self.encoder(self.shared(input_ids), mask), mask

    def _check_ids(self, ids):
        if ids.ndim != 2 or not all(ids.shape) or ids.shape[1] > self.config.n_positions:
            raise ValueError("T5 token input must be nonempty [batch, length] within n_positions")

    def __call__(
        self,
        input_ids,
        *,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
    ):
        self._check_ids(decoder_input_ids)
        if encoder_outputs is None:
            encoder_outputs, attention_mask = self.encode(input_ids, attention_mask)
        if encoder_outputs.shape[0] != decoder_input_ids.shape[0]:
            raise ValueError("T5 encoder and decoder batch sizes differ")
        if attention_mask is None:
            raise ValueError("reused encoder outputs require their attention mask")
        mask = (
            mx.ones_like(decoder_input_ids)
            if decoder_attention_mask is None
            else decoder_attention_mask
        )
        if mask.shape != decoder_input_ids.shape:
            raise ValueError("decoder attention mask must match input IDs")
        hidden = self.decoder(self.shared(decoder_input_ids), mask, encoder_outputs, attention_mask)
        logits = self.shared.as_linear(hidden * self.config.d_model**-0.5)
        return T5Output(logits, encoder_outputs, hidden)

    def greedy(self, input_ids, *, attention_mask=None, max_new_tokens=32):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens < self.config.n_positions:
            raise ValueError("max_new_tokens must be positive and below n_positions")
        encoder, mask = self.encode(input_ids, attention_mask)
        mx.eval(encoder)
        ids = mx.full((input_ids.shape[0], 1), self.config.decoder_start_token_id, mx.int32)
        done = mx.zeros((input_ids.shape[0],), mx.bool_)
        for _ in range(max_new_tokens):
            output = self(
                input_ids, decoder_input_ids=ids, attention_mask=mask, encoder_outputs=encoder
            )
            token = mx.argmax(output.logits[:, -1], axis=-1)
            token = mx.where(done, self.config.pad_token_id, token)
            ids = mx.concatenate((ids, token[:, None]), axis=1)
            done = done | (token == self.config.eos_token_id)
            mx.eval(ids, done)
            if bool(mx.all(done).item()):
                break
        return ids
