"""Native MLX Qwen2.5-VL architecture."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import VisionLanguageModelOutput
from mlx_one.models.language.qwen2.config import Qwen2Config
from mlx_one.models.language.qwen2.model import Qwen2Model
from mlx_one.models.shared.vision import PatchEmbed, VisionAttention, vision_position_ids
from mlx_one.models.vision_language.qwen2_5_vl.config import Qwen2_5_VLConfig
from mlx_one.models.vision_language.qwen2_vl.model import (
    _replace_features,
    multimodal_position_ids,
)


class Qwen2_5VisionMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, hidden: Any) -> Any:
        return self.down_proj(nn.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Qwen2_5VisionBlock(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config.hidden_size, config.num_heads)
        self.mlp = Qwen2_5VisionMLP(config)

    def __call__(self, hidden: Any, position_ids: Any, lengths: Sequence[int]) -> Any:
        chunks = []
        start = 0
        for length in lengths:
            stop = start + length
            current = hidden[start:stop]
            positions = position_ids[start:stop]
            current = current + self.attn(self.norm1(current), positions)
            chunks.append(current + self.mlp(self.norm2(current)))
            start = stop
        return mx.concatenate(chunks, axis=0)


class Qwen2_5PatchMerger(nn.Module):
    def __init__(self, output_dim: int, context_dim: int, merge_size: int) -> None:
        super().__init__()
        self.merged_dim = context_dim * merge_size**2
        self.ln_q = nn.RMSNorm(context_dim, eps=1e-6)
        self.mlp_0 = nn.Linear(self.merged_dim, self.merged_dim, bias=True)
        self.mlp_2 = nn.Linear(self.merged_dim, output_dim, bias=True)

    def __call__(self, hidden: Any) -> Any:
        if hidden.shape[0] * self.ln_q.weight.shape[0] % self.merged_dim:
            raise ValueError("vision patch count is incompatible with spatial merge size")
        hidden = self.ln_q(hidden).reshape(-1, self.merged_dim)
        return self.mlp_2(nn.gelu(self.mlp_0(hidden)))


class Qwen2_5VisionTransformer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        self.blocks = [Qwen2_5VisionBlock(config) for _ in range(config.depth)]
        self.merger = Qwen2_5PatchMerger(
            config.out_hidden_size, config.hidden_size, config.spatial_merge_size
        )

    def __call__(self, patches: Any, grid_thw: Sequence[Sequence[int]]) -> tuple[Any, Any]:
        hidden = self.patch_embed(patches)
        positions, full_lengths = vision_position_ids(grid_thw, self.config.spatial_merge_size)
        if sum(full_lengths) != hidden.shape[0]:
            raise ValueError("vision grid patch count does not match pixel patches")
        window_index, window_lengths = _window_plan(grid_thw, self.config)
        group = self.config.spatial_merge_size**2
        grouped = hidden.reshape(-1, group, hidden.shape[-1])
        hidden = grouped[mx.array(window_index)].reshape(-1, hidden.shape[-1])
        grouped_positions = positions.reshape(-1, group, 2)
        positions = grouped_positions[mx.array(window_index)].reshape(-1, 2)
        for index, block in enumerate(self.blocks):
            lengths = full_lengths if index in self.config.fullatt_block_indexes else window_lengths
            hidden = block(hidden, positions, lengths)
        merged = self.merger(hidden)
        reverse = sorted(range(len(window_index)), key=window_index.__getitem__)
        return merged[mx.array(reverse)], hidden


def _window_plan(
    grid_thw: Sequence[Sequence[int]], config: Any
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    merge = config.spatial_merge_size
    side = config.window_size // merge // config.patch_size
    indexes: list[int] = []
    lengths: list[int] = []
    offset = 0
    for temporal, height, width in grid_thw:
        temporal, height, width = int(temporal), int(height), int(width)
        if height % merge or width % merge:
            raise ValueError("vision grid must be divisible by spatial_merge_size")
        merged_height, merged_width = height // merge, width // merge
        for time in range(temporal):
            for window_row in range(0, merged_height, side):
                for window_column in range(0, merged_width, side):
                    count = 0
                    for row in range(window_row, min(window_row + side, merged_height)):
                        for column in range(window_column, min(window_column + side, merged_width)):
                            indexes.append(
                                offset
                                + time * merged_height * merged_width
                                + row * merged_width
                                + column
                            )
                            count += 1
                    lengths.append(count * merge**2)
        offset += temporal * merged_height * merged_width
    return tuple(indexes), tuple(lengths)


class Qwen2_5_VLForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen2_5VisionTransformer(config.vision_config)
        text = config.text_config
        dense = Qwen2Config(
            model_type="qwen2_5_vl_text",
            hidden_size=text.hidden_size,
            num_hidden_layers=text.num_hidden_layers,
            intermediate_size=text.intermediate_size,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=text.num_key_value_heads,
            vocab_size=text.vocab_size,
            rms_norm_eps=text.rms_norm_eps,
            max_position_embeddings=text.max_position_embeddings,
            rope_theta=text.rope_theta,
            rope_scaling=text.rope_scaling,
            tie_word_embeddings=config.tie_word_embeddings,
        )
        self.language_model = Qwen2Model(dense)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: Any,
        *,
        pixel_values: Any | None = None,
        pixel_values_videos: Any | None = None,
        image_grid_thw: Sequence[Sequence[int]] | None = None,
        video_grid_thw: Sequence[Sequence[int]] | None = None,
        mm_token_type_ids: Any | None = None,
        attention_mask: Any | None = None,
        cache: tuple[Any, ...] | None = None,
        output_hidden_states: bool = False,
    ) -> VisionLanguageModelOutput:
        embeddings = self.language_model.embed_tokens(input_ids)
        vision_states = []
        for patches, grid, token_id, kind in (
            (pixel_values, image_grid_thw, self.config.image_token_id, "image"),
            (pixel_values_videos, video_grid_thw, self.config.video_token_id, "video"),
        ):
            if patches is None:
                continue
            if grid is None:
                raise ValueError(f"{kind}_grid_thw is required with pixel values")
            features, states = self.visual(patches, grid)
            embeddings = _replace_features(embeddings, input_ids == token_id, features, kind)
            vision_states.append(states)
        if mm_token_type_ids is None and vision_states:
            mm_token_type_ids = mx.zeros_like(input_ids)
            mm_token_type_ids = mx.where(
                input_ids == self.config.image_token_id, 1, mm_token_type_ids
            )
            mm_token_type_ids = mx.where(
                input_ids == self.config.video_token_id, 2, mm_token_type_ids
            )
        if mm_token_type_ids is None:
            offset = 0 if cache is None else cache[0].offset
            base = mx.arange(offset, offset + input_ids.shape[1])[None, :]
            position_ids = mx.stack((base, base, base), axis=0)
            rope_deltas = mx.zeros((input_ids.shape[0], 1), dtype=base.dtype)
        else:
            position_ids, rope_deltas = multimodal_position_ids(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                spatial_merge_size=self.config.vision_config.spatial_merge_size,
                attention_mask=attention_mask,
            )
        hidden, cache, states = self.language_model(
            None,
            cache=cache,
            input_embeddings=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
            vision_hidden_states=mx.concatenate(vision_states) if vision_states else None,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
        )
