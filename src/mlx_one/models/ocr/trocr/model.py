"""Native ViT and TrOCR math following Hugging Face's Apache-2.0 implementation.

Reference: transformers/models/{vit,trocr}, with original checkpoint tensor names.
The runtime uses only MLX and MLX-One encoder/decoder caches.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import make_encoder_decoder_caches
from mlx_one.models.ocr._layers import LanguageModelOutput


class Attention(nn.Module):
    def __init__(self, dim, heads, source_dim=None):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(source_dim or dim, dim)
        self.v_proj = nn.Linear(source_dim or dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x, source=None, cache=None):
        def split(value):
            return value.reshape(
                value.shape[0], value.shape[1], self.heads, self.head_dim
            ).transpose(0, 2, 1, 3)

        query = split(self.q_proj(x))
        offset = cache.offset if cache is not None else 0
        if source is not None and cache is not None and cache.cross_keys is not None:
            key, value = cache.cross_keys, cache.cross_values
        else:
            src = x if source is None else source
            key, value = split(self.k_proj(src)), split(self.v_proj(src))
            if cache is not None:
                key, value = (
                    cache.update_self(key, value)
                    if source is None
                    else cache.update_cross(key, value)
                )
        mask = None
        if source is None:
            mask = mx.arange(key.shape[2])[None] <= mx.arange(offset, offset + x.shape[1])[:, None]
        result = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.head_dim**-0.5, mask=mask
        )
        return self.out_proj(result.transpose(0, 2, 1, 3).reshape(x.shape))


class ViTAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = nn.Module()
        for name in ("query", "key", "value"):
            self.attention[name] = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.qkv_bias)
        self.output = nn.Module()
        self.output.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.heads = cfg.num_attention_heads

    def __call__(self, x):
        b, length, dim = x.shape

        def project(name):
            return (
                self.attention[name](x)
                .reshape(b, length, self.heads, dim // self.heads)
                .transpose(0, 2, 1, 3)
            )

        value = mx.fast.scaled_dot_product_attention(
            project("query"), project("key"), project("value"), scale=(dim // self.heads) ** -0.5
        )
        return self.output.dense(value.transpose(0, 2, 1, 3).reshape(b, length, dim))


class ViTLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attention = ViTAttention(cfg)
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.output = nn.Module()
        self.output.dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def __call__(self, x):
        x = x + self.attention(self.layernorm_before(x))
        return x + self.output.dense(nn.gelu(self.intermediate.dense(self.layernorm_after(x))))


class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = nn.Module()
        self.embeddings.patch_embeddings.projection = nn.Conv2d(
            cfg.num_channels, cfg.hidden_size, cfg.patch_size, stride=cfg.patch_size
        )
        self.embeddings.cls_token = mx.zeros((1, 1, cfg.hidden_size))
        self.embeddings.position_embeddings = mx.zeros(
            (1, (cfg.image_size // cfg.patch_size) ** 2 + 1, cfg.hidden_size)
        )
        self.encoder = nn.Module()
        self.encoder.layer = [ViTLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        # add_pooling_layer=False in VisionEncoderDecoderModel; pooler absent.

    def __call__(self, pixels):
        x = self.embeddings.patch_embeddings.projection(pixels)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        cls = mx.broadcast_to(self.embeddings.cls_token, (x.shape[0], 1, x.shape[-1]))
        x = mx.concatenate([cls, x], 1) + self.embeddings.position_embeddings
        for layer in self.encoder.layer:
            x = layer(x)
        return self.layernorm(x)


class DecoderLayer(nn.Module):
    def __init__(self, cfg, source_dim):
        super().__init__()
        dim = cfg.d_model
        self.self_attn = Attention(dim, cfg.decoder_attention_heads)
        self.encoder_attn = Attention(dim, cfg.decoder_attention_heads, source_dim)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, cfg.decoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.decoder_ffn_dim, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def __call__(self, x, source, cache):
        x = self.self_attn_layer_norm(x + self.self_attn(x, cache=cache))
        x = self.encoder_attn_layer_norm(x + self.encoder_attn(x, source, cache))
        return self.final_layer_norm(x + self.fc2(nn.gelu(self.fc1(x))))


class Decoder(nn.Module):
    def __init__(self, cfg, source_dim):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_positions = nn.Embedding(cfg.max_position_embeddings + 2, cfg.d_model)
        if cfg.layernorm_embedding:
            self.layernorm_embedding = nn.LayerNorm(cfg.d_model)
        self.layers = [DecoderLayer(cfg, source_dim) for _ in range(cfg.decoder_layers)]

    def __call__(self, ids, source, cache):
        offset = cache[0].offset if cache else 0
        x = self.embed_tokens(ids) * (
            math.sqrt(self.cfg.d_model) if self.cfg.scale_embedding else 1
        )
        x = x + self.embed_positions(mx.arange(offset + 2, offset + ids.shape[1] + 2))[None]
        if self.cfg.layernorm_embedding:
            x = self.layernorm_embedding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, source, cache[i] if cache else None)
        return x


class TrOCR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ViT(config.encoder)
        self.decoder = nn.Module()
        self.decoder.model = nn.Module()
        source_dim = config.decoder.cross_attention_hidden_size or config.decoder.d_model
        if config.decoder.cross_attention_hidden_size is None and (
            config.encoder.hidden_size != config.decoder.d_model
        ):
            self.enc_to_dec_proj = nn.Linear(config.encoder.hidden_size, config.decoder.d_model)
        self.decoder.model.decoder = Decoder(config.decoder, source_dim)
        # The published checkpoint can include a separate, tied output projection.
        self.decoder.output_projection = nn.Linear(
            config.decoder.d_model, config.decoder.vocab_size, bias=False
        )

    def encode(self, pixels):
        source = self.encoder(pixels)
        return self.enc_to_dec_proj(source) if "enc_to_dec_proj" in self else source

    def decode(self, ids, source, cache):
        hidden = self.decoder.model.decoder(ids, source, cache)
        return LanguageModelOutput(self.decoder.output_projection(hidden))

    def make_cache(self):
        return make_encoder_decoder_caches(self.config.decoder.decoder_layers)

    @staticmethod
    def sanitize(weights):
        weights = dict(weights)
        # A saved tied projection can be omitted by safetensors shared-storage handling.
        key = "decoder.output_projection.weight"
        if key not in weights and "decoder.model.decoder.embed_tokens.weight" in weights:
            weights[key] = weights["decoder.model.decoder.embed_tokens.weight"]
        return weights
