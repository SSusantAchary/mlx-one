"""Configuration metadata for the native DistilBERT classification encoder."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DistilBertConfig:
    model_type: str = "distilbert"
    vocab_size: int = 30522
    dim: int = 768
    hidden_dim: int = 3072
    n_heads: int = 12
    n_layers: int = 6
    max_position_embeddings: int = 512
    pad_token_id: int = 0
    activation: str = "gelu"
    id2label: dict = field(default_factory=lambda: {"0": "NEGATIVE", "1": "POSITIVE"})

    def __post_init__(self):
        if self.model_type != "distilbert" or self.activation != "gelu":
            raise ValueError("only DistilBERT with exact GELU is implemented")
        if (
            any(
                isinstance(n, bool) or not isinstance(n, int) or n < 1
                for n in (
                    self.vocab_size,
                    self.dim,
                    self.hidden_dim,
                    self.n_heads,
                    self.n_layers,
                    self.max_position_embeddings,
                )
            )
            or self.dim % self.n_heads
        ):
            raise ValueError("invalid DistilBERT encoder dimensions")

    @classmethod
    def from_dict(cls, payload):
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})
