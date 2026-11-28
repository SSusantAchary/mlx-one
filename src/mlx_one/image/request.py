"""Backend-free image request validation and PNG result metadata."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path

from .profiles import ImageProfile


@dataclass(frozen=True)
class ImageRequest:
    """Validated request for one image."""

    prompt: str
    width: int
    height: int
    steps: int
    seed: int

    @classmethod
    def create(
        cls,
        profile: ImageProfile,
        prompt: str,
        width: int,
        height: int,
        steps: int | None,
        seed: int | None,
    ) -> ImageRequest:
        """Apply model-specific limits before a device lease is acquired."""
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or len(prompt) > profile.max_prompt_chars
        ):
            raise ValueError(f"prompt must contain 1–{profile.max_prompt_chars} characters")
        for size in (width, height):
            if (
                type(size) is not int
                or not 64 <= size <= profile.max_side
                or size % profile.size_multiple
            ):
                raise ValueError(
                    f"width and height must be multiples of {profile.size_multiple}, "
                    f"between 64 and {profile.max_side}"
                )
        if width * height > profile.max_pixels:
            raise ValueError(f"image exceeds the {profile.max_pixels}-pixel request limit")
        minimum = 2 if profile.family == "z-image-turbo" else 1
        if steps is None:
            steps = profile.default_steps
        if type(steps) is not int or not minimum <= steps <= profile.max_steps:
            raise ValueError(f"steps must be an integer from {minimum} to {profile.max_steps}")
        if seed is None:
            seed = secrets.randbits(32)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        return cls(prompt, width, height, steps, seed)


@dataclass(frozen=True)
class ImageResult:
    """Encoded PNG and the actual execution metadata for a generation."""

    png: bytes
    metadata: dict

    def save(self, path: str | Path) -> None:
        """Write PNG bytes to a caller-selected path."""
        Path(path).write_bytes(self.png)
