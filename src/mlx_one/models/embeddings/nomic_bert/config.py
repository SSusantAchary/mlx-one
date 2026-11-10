"""Explicit configuration for Nomic's post-norm rotary SwiGLU encoder."""

from dataclasses import dataclass

from mlx_one.core.config import ConfigError, require_positive


@dataclass(frozen=True)
class NomicBertConfig:
    model_type: str = "nomic_bert"
    vocab_size: int = 30528
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    rotary_emb_base: float = 1000.0
    max_seq_length: int = 8192
    pad_token_id: int = 0

    def __post_init__(self):
        if self.model_type != "nomic_bert":
            raise ConfigError("NomicBERT requires its own architecture identity")
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "type_vocab_size",
            "max_seq_length",
            "layer_norm_eps",
            "rotary_emb_base",
        ):
            require_positive(name, getattr(self, name))
        if (
            self.hidden_size % self.num_attention_heads
            or (self.hidden_size // self.num_attention_heads) % 2
        ):
            raise ConfigError("NomicBERT requires even rotary head dimensions")

    @classmethod
    def from_dict(cls, data):
        unsupported = any(
            data.get(key, False)
            for key in (
                "prenorm",
                "parallel_block",
                "causal",
                "use_rms_norm",
                "qkv_proj_bias",
                "mlp_fc1_bias",
                "mlp_fc2_bias",
                "rotary_emb_interleaved",
            )
        )
        if unsupported or data.get("rotary_emb_fraction", 1.0) != 1.0:
            raise ConfigError("unsupported NomicBERT block, bias, or rotary layout")
        if (
            data.get("rotary_scaling_factor") is not None
            or data.get("rotary_emb_scale_base") is not None
        ):
            raise ConfigError("scaled NomicBERT rotary profiles require separate qualification")
        if data.get("activation_function", "swiglu") != "swiglu":
            raise ConfigError("native NomicBERT currently supports SwiGLU checkpoints")
        values = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        values.setdefault("max_seq_length", data.get("n_positions", 8192))
        return cls(**values)

    def validate_dimensions(self, dimensions):
        if dimensions is None:
            return self.hidden_size
        if (
            type(dimensions) is not int
            or dimensions not in {64, 128, 256, 512, 768}
            or dimensions > self.hidden_size
        ):
            raise ValueError("Nomic Matryoshka dimensions must be 64, 128, 256, 512, or 768")
        return dimensions
