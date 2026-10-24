"""Public static-image generation result."""

from dataclasses import dataclass

from mlx_one.text.schemas import GenerationResult


@dataclass(frozen=True)
class VLMGenerationResult(GenerationResult):
    token_ids: tuple[int, ...] = ()
    image_metadata: tuple[dict, ...] = ()
