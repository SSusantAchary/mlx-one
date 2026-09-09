"""Shared vision layers used by native vision-language architectures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.shared.transformer import quick_gelu


def _rotate_half(value: Any) -> Any:
    half = value.shape[-1] // 2
    return mx.concatenate((-value[..., half:], value[..., :half]), axis=-1)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 4:
            raise ValueError("vision attention head dimension must be divisible by four")
        self.head_dim = head_dim
        self.theta = theta

    def __call__(self, value: Any, position_ids: Any) -> Any:
        quarter = self.head_dim // 4
        inv_freq = mx.power(
            self.theta, -mx.arange(quarter, dtype=mx.float32) / quarter
        )
        height = position_ids[:, 0].astype(mx.float32)[:, None] * inv_freq
        width = position_ids[:, 1].astype(mx.float32)[:, None] * inv_freq
        frequencies = mx.concatenate((height, width), axis=-1)
        frequencies = mx.concatenate((frequencies, frequencies), axis=-1)[:, None, :]
        return value * mx.cos(frequencies) + _rotate_half(value) * mx.sin(frequencies)


class PatchEmbed(nn.Module):
    """Qwen2-VL patch projection over processor-flattened 3D patches."""

    def __init__(
        self,
        *,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim, bias=False)

    def __call__(self, patches: Any) -> Any:
        if patches.ndim != 2 or patches.shape[-1] != self.patch_dim:
            raise ValueError(
                f"pixel patches must have shape [patches, {self.patch_dim}]"
            )
        return self.proj(patches)


class VisionAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("vision embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.rope = VisionRotaryEmbedding(self.head_dim)

    def __call__(self, hidden: Any, position_ids: Any) -> Any:
        length = hidden.shape[0]
        qkv = self.qkv(hidden).reshape(length, 3, self.num_heads, self.head_dim)
        queries = self.rope(qkv[:, 0], position_ids).transpose(1, 0, 2)[None, ...]
        keys = self.rope(qkv[:, 1], position_ids).transpose(1, 0, 2)[None, ...]
        values = qkv[:, 2].transpose(1, 0, 2)[None, ...]
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        attended = mx.softmax(scores, axis=-1, precise=True) @ values
        attended = attended[0].transpose(1, 0, 2).reshape(length, self.embed_dim)
        return self.proj(attended)


class VisionMlp(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, activation: str) -> None:
        super().__init__()
        if activation not in {"quick_gelu", "gelu", "silu"}:
            raise ValueError(f"unsupported vision activation: {activation}")
        self.activation = activation
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=True)

    def __call__(self, value: Any) -> Any:
        hidden = self.fc1(value)
        if self.activation == "quick_gelu":
            hidden = quick_gelu(hidden)
        elif self.activation == "gelu":
            hidden = nn.gelu(hidden)
        else:
            hidden = nn.silu(hidden)
        return self.fc2(hidden)


class VisionBlock(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = VisionAttention(config.embed_dim, config.num_heads)
        self.mlp = VisionMlp(
            config.embed_dim, int(config.embed_dim * config.mlp_ratio), config.hidden_act
        )

    def __call__(self, hidden: Any, position_ids: Any) -> Any:
        hidden = hidden + self.attn(self.norm1(hidden), position_ids)
        return hidden + self.mlp(self.norm2(hidden))


class PatchMerger(nn.Module):
    def __init__(self, output_dim: int, context_dim: int, merge_size: int) -> None:
        super().__init__()
        self.merge_size = merge_size
        self.merged_dim = context_dim * merge_size**2
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp_0 = nn.Linear(self.merged_dim, self.merged_dim, bias=True)
        self.mlp_2 = nn.Linear(self.merged_dim, output_dim, bias=True)

    def __call__(self, hidden: Any) -> Any:
        group = self.merge_size**2
        if hidden.shape[0] % group:
            raise ValueError("vision patch count must be divisible by merge_size squared")
        hidden = self.ln_q(hidden).reshape(-1, self.merged_dim)
        return self.mlp_2(nn.gelu(self.mlp_0(hidden)))


def vision_position_ids(
    grid_thw: Sequence[Sequence[int]], spatial_merge_size: int
) -> tuple[Any, tuple[int, ...]]:
    positions: list[tuple[int, int]] = []
    lengths = []
    for grid in grid_thw:
        if len(grid) != 3:
            raise ValueError("each vision grid must contain temporal, height, and width")
        temporal, height, width = (int(item) for item in grid)
        if min(temporal, height, width) < 1:
            raise ValueError("vision grid dimensions must be positive")
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError("vision grid must be divisible by spatial_merge_size")
        lengths.append(temporal * height * width)
        for _ in range(temporal):
            for merged_h in range(0, height, spatial_merge_size):
                for merged_w in range(0, width, spatial_merge_size):
                    for local_h in range(spatial_merge_size):
                        for local_w in range(spatial_merge_size):
                            positions.append((merged_h + local_h, merged_w + local_w))
    return mx.array(positions), tuple(lengths)
