"""Native NomicBERT with publisher fused-QKV and fc11/fc12 weight semantics."""

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import EmbeddingModelOutput
from mlx_one.models.shared.encoder import (
    bidirectional_attention_mask,
    l2_normalize,
    masked_mean_pool,
)


class NomicEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

    def __call__(self, ids, types=None):
        types = mx.zeros_like(ids) if types is None else types
        return self.word_embeddings(ids) + self.token_type_embeddings(types)


class NomicAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.Wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, hidden, mask):
        batch, length, width = hidden.shape
        heads, dim = self.config.num_attention_heads, width // self.config.num_attention_heads
        q, k, v = [
            part.reshape(batch, length, heads, dim).transpose(0, 2, 1, 3)
            for part in mx.split(self.Wqkv(hidden), 3, axis=-1)
        ]
        q = mx.fast.rope(
            q, dims=dim, traditional=False, base=self.config.rotary_emb_base, scale=1.0, offset=0
        )
        k = mx.fast.rope(
            k, dims=dim, traditional=False, base=self.config.rotary_emb_base, scale=1.0, offset=0
        )
        value = mx.fast.scaled_dot_product_attention(q, k, v, scale=dim**-0.5, mask=mask)
        return self.out_proj(value.transpose(0, 2, 1, 3).reshape(batch, length, width))


class NomicMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc11 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc12 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden):
        # Publisher fc11 is the value projection, fc12 is the SiLU gate.
        return self.fc2(self.fc11(hidden) * nn.silu(self.fc12(hidden)))


class NomicLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = NomicAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = NomicMLP(config)

    def __call__(self, hidden, mask):
        hidden = self.norm1(hidden + self.attn(hidden, mask))
        return self.norm2(hidden + self.mlp(hidden))


class NomicEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [NomicLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden, mask, states=False):
        outputs = [hidden] if states else None
        for layer in self.layers:
            hidden = layer(hidden, mask)
            if outputs is not None:
                outputs.append(hidden)
        return hidden, tuple(outputs) if outputs is not None else None


class NomicBertForEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = NomicEmbeddings(config)
        self.emb_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = NomicEncoder(config)

    def __call__(
        self,
        input_ids,
        *,
        attention_mask=None,
        token_type_ids=None,
        output_hidden_states=False,
        dimensions=None,
    ):
        if (
            input_ids.ndim != 2
            or not all(input_ids.shape)
            or input_ids.shape[1] > self.config.max_seq_length
        ):
            raise ValueError(
                "NomicBERT requires nonempty [batch, sequence] within its context bound"
            )
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
        if attention_mask.shape != input_ids.shape:
            raise ValueError("NomicBERT mask must match input IDs")
        dim = self.config.validate_dimensions(dimensions)
        hidden = self.emb_ln(self.embeddings(input_ids, token_type_ids))
        hidden, states = self.encoder(
            hidden, bidirectional_attention_mask(attention_mask, hidden.dtype), output_hidden_states
        )
        vectors = masked_mean_pool(hidden, attention_mask)
        if dimensions is not None:
            # Explicit Matryoshka profile: layer norm the full vector, THEN
            # truncate, THEN renormalize. Default full-size ST output is separate.
            mean = mx.mean(vectors, axis=-1, keepdims=True)
            variance = mx.mean((vectors - mean) ** 2, axis=-1, keepdims=True)
            vectors = (vectors - mean) * mx.rsqrt(variance + 1e-5)
        vectors = l2_normalize(vectors[:, :dim])
        return EmbeddingModelOutput(vectors, hidden, attention_mask, hidden_states=states)
