"""Native MLX Liquid LFM2-VL architecture."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import VisionLanguageModelOutput
from mlx_one.models.language.lfm2.model import Lfm2Model
from mlx_one.models.shared.transformer import quick_gelu
from mlx_one.models.vision_language.lfm2_vl.config import Lfm2VLConfig


class VisionAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, value: Any) -> Any:
        batch, length, dim = value.shape
        shape = (batch, length, self.heads, self.head_dim)
        query = self.q_proj(value).reshape(shape).transpose(0, 2, 1, 3)
        key = self.k_proj(value).reshape(shape).transpose(0, 2, 1, 3)
        val = self.v_proj(value).reshape(shape).transpose(0, 2, 1, 3)
        scores = (query @ key.transpose(0, 1, 3, 2)) * self.scale
        hidden = mx.softmax(scores, axis=-1, precise=True) @ val
        return self.out_proj(hidden.transpose(0, 2, 1, 3).reshape(batch, length, dim))


class VisionBlock(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.activation = config.hidden_act
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VisionAttention(config.hidden_size, config.num_attention_heads)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, value: Any) -> Any:
        value = value + self.self_attn(self.layer_norm1(value))
        hidden = self.fc1(self.layer_norm2(value))
        hidden = nn.gelu(hidden) if self.activation == "gelu" else quick_gelu(hidden)
        return value + self.fc2(hidden)


class Lfm2VisionTransformer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        patch_dim = config.num_channels * config.patch_size**2
        self.patch_embedding = nn.Linear(patch_dim, config.hidden_size, bias=True)
        grid = config.image_size // config.patch_size
        self.position_embedding = nn.Embedding(grid * grid, config.hidden_size)
        self.blocks = [VisionBlock(config) for _ in range(config.num_hidden_layers)]
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        merged = config.hidden_size * config.downsample_factor**2
        self.projector_norm = (
            nn.LayerNorm(merged, eps=config.layer_norm_eps)
            if config.use_projection_layernorm
            else None
        )
        self.projector = nn.Linear(merged, config.projection_dim, bias=True)

    def __call__(
        self, pixel_patches: Any, spatial_shapes: Sequence[Sequence[int]]
    ) -> tuple[Any, Any]:
        if pixel_patches.ndim != 3:
            raise ValueError("pixel_patches must have shape [batch, patches, patch_dim]")
        if len(spatial_shapes) != pixel_patches.shape[0]:
            raise ValueError("spatial_shapes must contain one grid per image")
        encoded = []
        projected = []
        for index, shape in enumerate(spatial_shapes):
            height, width = (int(item) for item in shape)
            length = height * width
            hidden = self.patch_embedding(pixel_patches[index : index + 1, :length])
            hidden = hidden + self._position_embeddings(height, width)[None, ...]
            for block in self.blocks:
                hidden = block(hidden)
            hidden = self.post_layernorm(hidden)
            encoded.append(hidden[0])
            merged = pixel_unshuffle(hidden, height, width, self.config.downsample_factor)
            if self.projector_norm is not None:
                merged = self.projector_norm(merged)
            projected.append(self.projector(merged)[0])
        return mx.concatenate(projected, axis=0), mx.concatenate(encoded, axis=0)

    def _position_embeddings(self, height: int, width: int) -> Any:
        base = self.config.image_size // self.config.patch_size
        if height > base or width > base:
            raise ValueError("vision grid exceeds configured image size")
        rows = mx.arange(height) * (base - 1) / max(height - 1, 1)
        cols = mx.arange(width) * (base - 1) / max(width - 1, 1)
        row0 = mx.floor(rows).astype(mx.int32)
        col0 = mx.floor(cols).astype(mx.int32)
        row1 = mx.minimum(row0 + 1, base - 1)
        col1 = mx.minimum(col0 + 1, base - 1)
        row_mix = (rows - row0)[:, None, None]
        col_mix = (cols - col0)[None, :, None]
        table = self.position_embedding.weight.reshape(base, base, -1)
        top = table[row0[:, None], col0[None, :]] * (1 - col_mix)
        top = top + table[row0[:, None], col1[None, :]] * col_mix
        bottom = table[row1[:, None], col0[None, :]] * (1 - col_mix)
        bottom = bottom + table[row1[:, None], col1[None, :]] * col_mix
        return (top * (1 - row_mix) + bottom * row_mix).reshape(height * width, -1)


def pixel_unshuffle(value: Any, height: int, width: int, factor: int) -> Any:
    if height % factor or width % factor:
        raise ValueError("vision grid must be divisible by downsample_factor")
    batch, length, channels = value.shape
    if length != height * width:
        raise ValueError("vision token count does not match spatial shape")
    value = value.reshape(batch, height // factor, factor, width // factor, factor, channels)
    return value.transpose(0, 1, 3, 2, 4, 5).reshape(
        batch, height * width // factor**2, channels * factor**2
    )


class Lfm2VLForConditionalGeneration(nn.Module):
    def __init__(self, config: Lfm2VLConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = Lfm2VisionTransformer(config.vision_config)
        self.language_model = Lfm2Model(config.text_config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.text_config.hidden_size, config.text_config.vocab_size, bias=False
            )

    def __call__(
        self,
        input_ids: Any,
        *,
        pixel_values: Any | None = None,
        spatial_shapes: Sequence[Sequence[int]] | None = None,
        attention_mask: Any | None = None,
        cache: tuple[Any, ...] | None = None,
        output_hidden_states: bool = False,
    ) -> VisionLanguageModelOutput:
        embeddings = self.language_model.embed_tokens(input_ids)
        vision_states = None
        if pixel_values is not None:
            if spatial_shapes is None:
                raise ValueError("spatial_shapes is required with pixel_values")
            features, vision_states = self.vision_model(pixel_values, spatial_shapes)
            embeddings = replace_image_features(
                embeddings, input_ids == self.config.image_token_id, features
            )
        hidden, cache, states = self.language_model(
            None,
            cache=cache,
            input_embeddings=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        logits = (
            self.language_model.embed_tokens.as_linear(hidden)
            if self.config.tie_word_embeddings
            else self.lm_head(hidden)
        )
        return VisionLanguageModelOutput(
            logits,
            hidden,
            cache=cache,
            hidden_states=states,
            vision_hidden_states=vision_states,
        )


def replace_image_features(embeddings: Any, mask: Any, features: Any) -> Any:
    count = int(mx.sum(mask).item())
    if count != features.shape[0]:
        raise ValueError(
            f"image feature count {features.shape[0]} does not match token count {count}"
        )
    indices = mx.maximum(mx.cumsum(mask.reshape(-1).astype(mx.int32)) - 1, 0)
    replacements = features[indices].reshape(embeddings.shape)
    return mx.where(mask[..., None], replacements, embeddings)
