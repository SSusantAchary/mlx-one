"""Strict Hugging Face-compatible GPT-2 configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras
from mlx_one.models.shared.encoder import validate_dropout


@dataclass(frozen=True)
class GPT2Config:
    model_type: str = "gpt2"
    vocab_size: int = 50257
    n_positions: int = 1024
    n_ctx: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int | None = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | None = 50256
    pad_token_id: int | None = None
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "gpt2":
            raise ConfigError(f"GPT2Config cannot represent model_type={self.model_type!r}")
        for name in ("vocab_size", "n_positions", "n_ctx", "n_embd", "n_layer", "n_head"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        if self.n_ctx != self.n_positions:
            raise ConfigError("n_ctx must equal n_positions")
        if self.n_embd % self.n_head:
            raise ConfigError("n_embd must be divisible by n_head")
        if self.n_inner is not None and (
            isinstance(self.n_inner, bool) or not isinstance(self.n_inner, int) or self.n_inner <= 0
        ):
            raise ConfigError("n_inner must be a positive integer or null")
        if self.activation_function != "gelu_new":
            raise ConfigError("native GPT-2 supports only gelu_new")
        for name in ("resid_pdrop", "embd_pdrop", "attn_pdrop"):
            validate_dropout(name, getattr(self, name))
        for name in ("layer_norm_epsilon", "initializer_range"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ConfigError(f"{name} must be a positive number")
        if not self.scale_attn_weights:
            raise ConfigError("native GPT-2 requires scale_attn_weights")
        if self.scale_attn_by_inverse_layer_idx:
            raise ConfigError("scale_attn_by_inverse_layer_idx is not supported")
        if self.reorder_and_upcast_attn:
            raise ConfigError("reorder_and_upcast_attn is not supported")
        if self.add_cross_attention:
            raise ConfigError("GPT-2 cross-attention is not supported")
        if not self.tie_word_embeddings:
            raise ConfigError("native GPT-2 requires tied word embeddings")
        for name in ("scale_attn_weights", "use_cache", "tie_word_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a boolean")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < self.vocab_size
            ):
                raise ConfigError(f"{name} must be null or within the vocabulary")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def inner_dim(self) -> int:
        return self.n_inner or 4 * self.n_embd

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GPT2Config:
        known = set(cls.__dataclass_fields__) - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid GPT-2 configuration: {exc}") from exc
