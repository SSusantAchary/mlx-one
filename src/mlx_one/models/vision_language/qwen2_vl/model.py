"""Native MLX implementation of the Qwen2-VL architecture."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import VisionLanguageModelOutput
from mlx_one.models.language.qwen2.config import Qwen2Config
from mlx_one.models.language.qwen2.model import Qwen2Model
from mlx_one.models.shared.vision import (
    PatchEmbed,
    PatchMerger,
    VisionBlock,
    vision_position_ids,
)
from mlx_one.models.vision_language.qwen2_vl.config import Qwen2VLConfig


class Qwen2VisionTransformer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(config.hidden_size, config.embed_dim, config.spatial_merge_size)

    def __call__(self, pixel_patches: Any, grid_thw: Sequence[Sequence[int]]) -> tuple[Any, Any]:
        hidden = self.patch_embed(pixel_patches)
        positions, lengths = vision_position_ids(grid_thw, self.spatial_merge_size)
        if sum(lengths) != hidden.shape[0]:
            raise ValueError("vision grid patch count does not match pixel patches")
        for block in self.blocks:
            chunks = []
            start = 0
            for length in lengths:
                stop = start + length
                chunks.append(block(hidden[start:stop], positions[start:stop]))
                start = stop
            hidden = mx.concatenate(chunks, axis=0)
        return self.merger(hidden), hidden


class Qwen2VLForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen2VLConfig) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen2VisionTransformer(config.vision_config)
        text = config.text_config
        dense_config = Qwen2Config(
            model_type="qwen2_vl_text",
            hidden_size=text.hidden_size,
            num_hidden_layers=text.num_hidden_layers,
            intermediate_size=text.intermediate_size,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=text.num_key_value_heads,
            vocab_size=text.vocab_size,
            rms_norm_eps=text.rms_norm_eps,
            max_position_embeddings=text.max_position_embeddings,
            rope_theta=text.rope_theta,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_scaling=text.rope_scaling,
        )
        self.language_model = Qwen2Model(dense_config)
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
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required with pixel_values")
            image_features, image_states = self.visual(pixel_values, image_grid_thw)
            embeddings = _replace_features(
                embeddings, input_ids == self.config.image_token_id, image_features, "image"
            )
            vision_states.append(image_states)
        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("video_grid_thw is required with pixel_values_videos")
            video_features, video_states = self.visual(pixel_values_videos, video_grid_thw)
            embeddings = _replace_features(
                embeddings, input_ids == self.config.video_token_id, video_features, "video"
            )
            vision_states.append(video_states)
        if mm_token_type_ids is None and vision_states:
            mm_token_type_ids = mx.zeros_like(input_ids)
            mm_token_type_ids = mx.where(
                input_ids == self.config.image_token_id, 1, mm_token_type_ids
            )
            mm_token_type_ids = mx.where(
                input_ids == self.config.video_token_id, 2, mm_token_type_ids
            )
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
            vision_hidden_states=mx.concatenate(vision_states, axis=0) if vision_states else None,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
        )


def _replace_features(embeddings: Any, mask: Any, features: Any, kind: str) -> Any:
    flat_mask = mask.reshape(-1)
    count = int(mx.sum(flat_mask).item())
    if count != features.shape[0]:
        raise ValueError(
            f"{kind} feature count {features.shape[0]} does not match token count {count}"
        )
    indices = mx.maximum(mx.cumsum(flat_mask.astype(mx.int32)) - 1, 0)
    replacements = features[indices].reshape(embeddings.shape)
    return mx.where(mask[..., None], replacements, embeddings)


def multimodal_position_ids(
    input_ids: Any,
    *,
    mm_token_type_ids: Any | None,
    image_grid_thw: Sequence[Sequence[int]] | None,
    video_grid_thw: Sequence[Sequence[int]] | None,
    spatial_merge_size: int,
    attention_mask: Any | None,
) -> tuple[Any, Any]:
    batch, length = input_ids.shape
    has_vision = image_grid_thw is not None or video_grid_thw is not None
    if not has_vision:
        base = mx.arange(length)[None, :]
        return mx.broadcast_to(base, (3, batch, length)), mx.zeros((batch, 1), mx.int32)
    if mm_token_type_ids is None:
        raise ValueError("mm_token_type_ids is required for multimodal position ids")
    image_iter = iter(image_grid_thw or ())
    video_iter = iter(video_grid_thw or ())
    batch_positions = []
    deltas = []
    for batch_index in range(batch):
        token_types = [int(item) for item in mm_token_type_ids[batch_index].tolist()]
        valid = (
            [bool(item) for item in attention_mask[batch_index].tolist()]
            if attention_mask is not None
            else [True] * length
        )
        positions = [[0] * length for _ in range(3)]
        cursor = 0
        index = 0
        while index < length:
            if not valid[index]:
                index += 1
                continue
            token_type = token_types[index]
            stop = index
            while stop < length and valid[stop] and token_types[stop] == token_type:
                stop += 1
            segment_length = stop - index
            if token_type == 0:
                for local in range(segment_length):
                    for axis in range(3):
                        positions[axis][index + local] = cursor + local
                cursor += segment_length
            else:
                grid = next(image_iter if token_type == 1 else video_iter)
                temporal, height, width = (int(item) for item in grid)
                merged_h = height // spatial_merge_size
                merged_w = width // spatial_merge_size
                expected = temporal * merged_h * merged_w
                if expected != segment_length:
                    raise ValueError("multimodal token segment does not match grid")
                local = 0
                for time in range(temporal):
                    for row in range(merged_h):
                        for column in range(merged_w):
                            positions[0][index + local] = cursor + time
                            positions[1][index + local] = cursor + row
                            positions[2][index + local] = cursor + column
                            local += 1
                cursor += max(temporal, merged_h, merged_w)
            index = stop
        batch_positions.append(mx.array(positions))
        deltas.append(cursor - sum(valid))
    result = mx.stack(batch_positions, axis=1)
    return result, mx.array(deltas, dtype=mx.int32)[:, None]
