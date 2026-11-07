"""Native text classification through the shared MLX-One task service."""

from mlx_one.classification.loading import (
    load_classification_model,
    load_token_classification_model,
)
from mlx_one.classification.public import (
    ClassificationEngine,
    TokenClassificationEngine,
    ZeroShotImageEngine,
    classify,
    token_classify,
    zero_shot_classify,
)
from mlx_one.classification.schemas import ClassificationResult, TokenClassificationResult

__all__ = [
    "ZeroShotImageEngine",
    "zero_shot_classify",
    "TokenClassificationEngine",
    "TokenClassificationResult",
    "token_classify",
    "load_token_classification_model",
    "ClassificationEngine",
    "ClassificationResult",
    "classify",
    "load_classification_model",
]
