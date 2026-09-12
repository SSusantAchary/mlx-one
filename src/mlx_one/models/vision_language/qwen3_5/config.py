"""Strict nested configuration for the dense Qwen3.5 multimodal releases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_divisible, require_positive


@dataclass(frozen=True)
class Qwen3_5TextConfig:
    model_type: str = "qwen3_5_text"
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True
    tie_word_embeddings: bool = False
    use_cache: bool = True
    full_attention_interval: int = 4
    layer_types: tuple[str, ...] | None = None
    mlp_only_layers: tuple[int, ...] = ()
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    mamba_ssm_dtype: str = "float32"
    rope_parameters: Mapping[str, Any] | None = None
    eos_token_id: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "qwen3_5_text":
            raise ConfigError("invalid Qwen3.5 text model_type")
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "full_attention_interval",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
        ):
            require_positive(name, getattr(self, name))
        require_divisible(
            "num_attention_heads",
            self.num_attention_heads,
            "num_key_value_heads",
            self.num_key_value_heads,
        )
        require_divisible(
            "linear_num_value_heads",
            self.linear_num_value_heads,
            "linear_num_key_heads",
            self.linear_num_key_heads,
        )
        if self.hidden_act != "silu":
            raise ConfigError("Qwen3.5 hidden_act must be silu")
        if self.attention_bias or self.attention_dropout != 0:
            raise ConfigError("Qwen3.5 supports bias-free, zero-dropout text attention")
        if not self.attn_output_gate:
            raise ConfigError("Qwen3.5 requires attn_output_gate")
        if self.mlp_only_layers:
            raise ConfigError("Qwen3.5 small multimodal releases have no mlp_only_layers")
        if self.mamba_ssm_dtype != "float32":
            raise ConfigError("Qwen3.5 recurrent state must use float32")
        if self.mtp_num_hidden_layers != 1 or self.mtp_use_dedicated_embeddings:
            raise ConfigError("unsupported Qwen3.5 MTP configuration")
        layer_types = self.layer_types
        if layer_types is None:
            layer_types = tuple(
                "full_attention"
                if (index + 1) % self.full_attention_interval == 0
                else "linear_attention"
                for index in range(self.num_hidden_layers)
            )
        else:
            layer_types = tuple(layer_types)
        if len(layer_types) != self.num_hidden_layers or any(
            item not in {"linear_attention", "full_attention"} for item in layer_types
        ):
            raise ConfigError("layer_types must contain one supported mixer per layer")
        object.__setattr__(self, "layer_types", layer_types)
        rope = dict(self.rope_parameters or {})
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type not in {"default", "mrope"}:
            raise ConfigError(f"unsupported Qwen3.5 RoPE type: {rope_type}")
        partial = float(rope.get("partial_rotary_factor", 0.25))
        rotary_dim = int(self.head_dim * partial)
        if not 0 < partial <= 1 or rotary_dim < 2 or rotary_dim % 2:
            raise ConfigError("partial_rotary_factor must produce a positive even dimension")
        sections = tuple(int(item) for item in rope.get("mrope_section", ()))
        if sections and (len(sections) != 3 or sum(sections) != rotary_dim // 2):
            raise ConfigError("mrope_section must sum to half the partial rotary dimension")
        if sections and not rope.get("mrope_interleaved", False):
            raise ConfigError("Qwen3.5 multimodal RoPE must be interleaved")
        rope.update(
            {
                "rope_type": "default",
                "rope_theta": float(rope.get("rope_theta", 10_000_000.0)),
                "partial_rotary_factor": partial,
                "mrope_section": sections,
                "mrope_interleaved": bool(rope.get("mrope_interleaved", bool(sections))),
            }
        )
        object.__setattr__(self, "rope_parameters", rope)

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rope_parameters["partial_rotary_factor"])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen3_5TextConfig:
        known = set(cls.__dataclass_fields__) - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        if "rope_parameters" not in payload and "rope_scaling" in data:
            payload["rope_parameters"] = data["rope_scaling"]
        payload["extra"] = extras(data, known | {"rope_scaling"})
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Qwen3.5 text configuration: {exc}") from exc


@dataclass(frozen=True)
class Qwen3_5VisionConfig:
    model_type: str = "qwen3_5_vision"
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type not in {"qwen3_5", "qwen3_5_vision"}:
            raise ConfigError("invalid Qwen3.5 vision model_type")
        object.__setattr__(self, "model_type", "qwen3_5_vision")
        for name in (
            "depth",
            "hidden_size",
            "intermediate_size",
            "num_heads",
            "in_channels",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "out_hidden_size",
            "num_position_embeddings",
        ):
            require_positive(name, getattr(self, name))
        require_divisible("hidden_size", self.hidden_size, "num_heads", self.num_heads)
        require_divisible("vision head dimension", self.head_dim, "rotary groups", 4)
        side = int(self.num_position_embeddings**0.5)
        if side * side != self.num_position_embeddings:
            raise ConfigError("num_position_embeddings must be a perfect square")
        if self.hidden_act != "gelu_pytorch_tanh":
            raise ConfigError("Qwen3.5 vision requires gelu_pytorch_tanh")
        if self.deepstack_visual_indexes:
            raise ConfigError("Qwen3.5 0.8B/2B releases do not use deep-stack vision features")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen3_5VisionConfig:
        known = set(cls.__dataclass_fields__) - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Qwen3.5 vision configuration: {exc}") from exc


@dataclass(frozen=True)
class Qwen3_5Config:
    model_type: str
    text_config: Qwen3_5TextConfig
    vision_config: Qwen3_5VisionConfig
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "qwen3_5":
            raise ConfigError(f"Qwen3_5Config cannot represent model_type={self.model_type!r}")
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ConfigError("vision out_hidden_size must equal text hidden_size")
        if self.tie_word_embeddings != self.text_config.tie_word_embeddings:
            raise ConfigError("top-level and text tie_word_embeddings must agree")
        ids = (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids):
            raise ConfigError("multimodal token ids must be non-negative integers")
        if len(set(ids)) != len(ids):
            raise ConfigError("multimodal token ids must be distinct")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen3_5Config:
        known = {
            "model_type",
            "text_config",
            "vision_config",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "tie_word_embeddings",
        }
        text = data.get("text_config")
        vision = data.get("vision_config")
        if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
            raise ConfigError("Qwen3.5 requires text_config and vision_config objects")
        text = dict(text)
        text.setdefault("tie_word_embeddings", data.get("tie_word_embeddings", True))
        payload = {key: value for key, value in data.items() if key in known}
        payload["text_config"] = Qwen3_5TextConfig.from_dict(text)
        payload["vision_config"] = Qwen3_5VisionConfig.from_dict(vision)
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Qwen3.5 configuration: {exc}") from exc
