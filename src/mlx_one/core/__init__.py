"""Low-level contracts for native mlx-one model implementations."""

from mlx_one.core.compatibility import (
    MLXCompatibility,
    MLXCompatibilityManifestError,
    MLXCompatibilityStatus,
    get_mlx_compatibility,
    load_mlx_compatibility_manifest,
)
from mlx_one.core.config import ConfigError
from mlx_one.core.outputs import (
    ASRModelOutput,
    AudioModelOutput,
    EmbeddingModelOutput,
    ModelOutput,
    VisionLanguageModelOutput,
)
from mlx_one.core.registry import ModelRegistration, get_registration, registered_model_types
from mlx_one.core.weights import WeightContract, WeightContractError

__all__ = [
    "ASRModelOutput",
    "AudioModelOutput",
    "ConfigError",
    "EmbeddingModelOutput",
    "ModelOutput",
    "ModelRegistration",
    "MLXCompatibility",
    "MLXCompatibilityManifestError",
    "MLXCompatibilityStatus",
    "VisionLanguageModelOutput",
    "WeightContract",
    "WeightContractError",
    "get_registration",
    "get_mlx_compatibility",
    "load_mlx_compatibility_manifest",
    "registered_model_types",
]
