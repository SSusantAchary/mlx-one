"""Backend-free configuration for the token-classification use of BERT."""

from dataclasses import dataclass, field

from mlx_one.models.embeddings.bert.config import BertEmbeddingConfig


@dataclass(frozen=True)
class BertNERConfig(BertEmbeddingConfig):
    vocab_size: int = 28996
    hidden_size: int = 768
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    add_pooling_layer: bool = False
    sentence_max_length: int = 512
    normalize_embeddings: bool = False
    id2label: dict = field(default_factory=lambda: {
        "0": "O", "1": "B-MISC", "2": "I-MISC", "3": "B-PER", "4": "I-PER",
        "5": "B-ORG", "6": "I-ORG", "7": "B-LOC", "8": "I-LOC",
    })
