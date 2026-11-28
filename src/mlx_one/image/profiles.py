"""Immutable checkpoint identities and request limits for native image generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files

from mlx_one.engine.contracts import TaskKind


@dataclass(frozen=True)
class ImageProfile:
    """A pinned text-to-image checkpoint and the native execution profiles it can load."""

    model: str
    revision: str
    family: str
    default_steps: int
    task: TaskKind = TaskKind.IMAGE_GENERATION
    size_multiple: int = 16
    max_pixels: int = 1024**2
    max_side: int = 1536
    max_prompt_chars: int = 8192
    max_prompt_tokens: int = 512
    max_steps: int = 20
    precision_profiles: tuple[int | None, ...] = (None, 4, 8)
    qualified_precisions: tuple[int | None, ...] = ()
    q4_model: str | None = None
    q4_revision: str | None = None

    def artifact_revision(self, model: str) -> str:
        """Return the revision for an official checkpoint or its published Q4 bundle."""
        if model == self.q4_model and self.q4_revision is not None:
            return self.q4_revision
        return self.revision

    def precision(self, bits: int | None, *, published: bool = False) -> dict:
        """Return component-level execution and weight precision."""
        if published:
            affine = {
                "mode": "affine",
                "bits": 4,
                "group_size": 64,
                "axis": "input",
                "packed_dtype": "uint32",
                "scales_and_biases": "bfloat16",
                "other_weights": "bfloat16",
            }
            return {
                "text_encoder": {**affine, "scope": "embedding and linear projections"},
                "denoiser": {**affine, "scope": "linear projections"},
                "vae": {**affine, "scope": "decoder mid-block attention projections"},
                "scheduler_latents": "bfloat16" if self.family == "flux2-klein" else "float32",
            }
        return {
            "text_encoder": "bfloat16",
            "denoiser": "bfloat16"
            if bits is None
            else {
                "mode": "affine",
                "bits": bits,
                "group_size": 64,
                "axis": "input",
                "packed_dtype": "uint32",
                "scales_and_biases": "bfloat16",
                "scope": "transformer-block attention and feed-forward linear weights",
                "other_weights": "bfloat16",
            },
            "vae": "bfloat16",
            "scheduler_latents": "bfloat16" if self.family == "flux2-klein" else "float32",
        }


IMAGE_PROFILES = (
    ImageProfile(
        "black-forest-labs/FLUX.2-klein-4B",
        "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "flux2-klein",
        4,
        q4_model="mflux-community/flux2-klein-4b-mflux-q4",
        q4_revision="794cd159538149ad9830848508c31f0ea7088e58",
    ),
    ImageProfile(
        "Tongyi-MAI/Z-Image-Turbo",
        "f332072aa78be7aecdf3ee76d5c247082da564a6",
        "z-image-turbo",
        9,
        q4_model="mflux-community/z-image-turbo-mflux-q4",
        q4_revision="f427e257d8e6ffa03edd4d9ac554a05809da456c",
    ),
)


def image_profile(model: str) -> ImageProfile:
    """Resolve a supported image checkpoint ID without loading MLX."""
    for profile in IMAGE_PROFILES:
        if model == profile.model or model == profile.q4_model:
            return profile
    raise ValueError(f"unknown native image model: {model}")


def artifact_lock(profile: ImageProfile) -> dict:
    """Return the pinned config and tensor-header identities for this checkpoint."""
    path = files("mlx_one.data").joinpath("image_checkpoints.json")
    return json.loads(path.read_text(encoding="utf-8"))[profile.family]


def published_q4_lock(profile: ImageProfile) -> dict:
    """Return source file identities and canonical model configs for the published Q4 bundle."""
    path = files("mlx_one.data").joinpath("image_published_q4.json")
    return json.loads(path.read_text(encoding="utf-8"))[profile.family]
