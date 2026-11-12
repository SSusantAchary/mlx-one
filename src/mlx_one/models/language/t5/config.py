"""Configuration and scalar relative-position oracle for original T5."""

import math
from dataclasses import dataclass

from mlx_one.core.config import ConfigError, require_positive


@dataclass(frozen=True)
class T5Config:
    model_type: str = "t5"
    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6
    feed_forward_proj: str = "relu"
    tie_word_embeddings: bool = True
    decoder_start_token_id: int = 0
    pad_token_id: int = 0
    eos_token_id: int = 1
    n_positions: int = 512

    def __post_init__(self):
        if (
            self.model_type != "t5"
            or self.feed_forward_proj != "relu"
            or not self.tie_word_embeddings
        ):
            raise ConfigError(
                "native T5 currently supports original tied-embedding ReLU checkpoints"
            )
        for key in (
            "vocab_size",
            "d_model",
            "d_kv",
            "d_ff",
            "num_layers",
            "num_decoder_layers",
            "num_heads",
            "n_positions",
            "relative_attention_num_buckets",
            "relative_attention_max_distance",
            "layer_norm_epsilon",
        ):
            require_positive(key, getattr(self, key))
        if self.relative_attention_num_buckets < 4 or self.relative_attention_num_buckets % 4:
            raise ConfigError("T5 relative bucket count must be divisible by four")
        if self.relative_attention_max_distance <= self.relative_attention_num_buckets:
            raise ConfigError("T5 relative distance must exceed its exact bucket range")
        for token in (self.pad_token_id, self.decoder_start_token_id, self.eos_token_id):
            if type(token) is not int or not 0 <= token < self.vocab_size:
                raise ConfigError("T5 special token IDs must be in the vocabulary")

    @classmethod
    def from_dict(cls, data):
        values = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        values.setdefault("num_decoder_layers", data.get("num_layers", 6))
        return cls(**values)


def relative_bucket(position, *, bidirectional, buckets=32, distance=128):
    """Pure-Python oracle for T5 key-position minus query-position bucketing."""
    sign = 0
    if bidirectional:
        buckets //= 2
        sign = buckets if position > 0 else 0
        position = abs(position)
    else:
        position = max(-position, 0)
    exact = buckets // 2
    if position < exact:
        return sign + position
    return sign + min(
        buckets - 1,
        exact + int(math.log(position / exact) / math.log(distance / exact) * (buckets - exact)),
    )
