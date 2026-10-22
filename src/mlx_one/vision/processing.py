"""Qwen2-VL image normalization and patch construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreparedVisionInput:
    """Model-ready prompt tokens, flattened patches, and image grid."""

    input_ids: Any
    pixel_values: Any
    image_grid_thw: tuple[tuple[int, int, int], ...]
    mm_token_type_ids: Any


class QwenImageProcessor:
    """Minimal native Qwen2-VL processor for one or more static images."""

    _MEAN = (0.48145466, 0.4578275, 0.40821073)
    _STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        tokenizer: Any,
        config: Any,
        *,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        max_source_pixels: int = 40_000_000,
        metadata: dict | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        if metadata is not None:
            import math

            mean,std=metadata.get("image_mean"),metadata.get("image_std")
            if (not isinstance(mean,list) or not isinstance(std,list) or len(mean)!=3 or len(std)!=3
                    or any(type(v) not in {int, float} or not math.isfinite(v) for v in mean+std)
                    or any(v<=0 for v in std) or metadata.get("resample",3)!=3
                    or any(metadata.get(k,True) is not True for k in
                           ("do_normalize","do_resize","do_rescale"))
                    or metadata.get("rescale_factor",1/255)!=1/255):
                raise ValueError("unsupported Qwen processor normalization or resize policy")
            vision=config.vision_config
            for name,actual in (("patch_size",vision.patch_size),
                ("temporal_patch_size",vision.temporal_patch_size),
                ("merge_size",vision.spatial_merge_size)):
                if name in metadata and metadata[name]!=actual:
                    raise ValueError("processor patch layout differs from model")
            self._MEAN,self._STD=tuple(mean),tuple(std)
            size=metadata.get("size",{})
            if not isinstance(size, dict):
                raise ValueError("processor size metadata must be an object")
            min_pixels=metadata.get("min_pixels",size.get("shortest_edge",min_pixels))
            max_pixels=metadata.get("max_pixels",size.get("longest_edge",max_pixels))
        if (any(type(value) is not int or value < 1 for value in
                (min_pixels, max_pixels, max_source_pixels)) or min_pixels > max_pixels
                or max_pixels < (config.vision_config.patch_size
                                  * config.vision_config.spatial_merge_size) ** 2):
            raise ValueError("invalid image pixel bounds")
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_source_pixels = max_source_pixels

    def prepare(self, prompt: str, images: list[Any] | tuple[Any, ...]) -> PreparedVisionInput:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("vision prompt must be non-empty text")
        if not images:
            raise ValueError("at least one image is required")
        pieces = prompt.split("<image>")
        if len(pieces) - 1 != len(images):
            raise ValueError("the number of <image> placeholders must match the images")
        import mlx.core as mx

        all_patches = []
        grids = []
        replacement_ids = []
        for image in images:
            patches, grid = self._patches(image, mx)
            all_patches.append(patches)
            grids.append(grid)
            visual_tokens = grid[0] * grid[1] * grid[2]
            merge = self.config.vision_config.spatial_merge_size**2
            replacement_ids.append(
                [self.config.vision_start_token_id]
                + [self.config.image_token_id] * (visual_tokens // merge)
                + [self.config.vision_end_token_id]
            )
        ids = self.tokenizer.encode(pieces[0])
        types = [0] * len(ids)
        for visual_ids, text in zip(replacement_ids, pieces[1:], strict=True):
            ids.extend(visual_ids)
            types.extend(
                [0] + [1] * (len(visual_ids) - 2) + [0]
            )
            suffix = self.tokenizer.encode(text)
            ids.extend(suffix)
            types.extend([0] * len(suffix))
        return PreparedVisionInput(
            input_ids=mx.array([ids], dtype=mx.int32),
            pixel_values=mx.concatenate(all_patches, axis=0),
            image_grid_thw=tuple(grids),
            mm_token_type_ids=mx.array([types], dtype=mx.int32),
        )

    def _patches(self, image: Any, mx: Any) -> tuple[Any, tuple[int, int, int]]:
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise RuntimeError("image processing requires `pip install 'mlx-one[vision]'`") from exc
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                self._validate_source_size(opened)
                source = ImageOps.exif_transpose(opened).convert("RGB")
        elif isinstance(image, Image.Image):
            self._validate_source_size(image)
            source = ImageOps.exif_transpose(image).convert("RGB")
        else:
            raise TypeError("image must be a path or PIL image")
        patch = self.config.vision_config.patch_size
        merge = self.config.vision_config.spatial_merge_size
        factor = patch * merge
        width, height = _smart_resize(
            source.width,
            source.height,
            factor=factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        source = source.resize((width, height), Image.Resampling.BICUBIC)
        pixels = mx.array(list(source.getdata()), dtype=mx.float32).reshape(height, width, 3)
        mean = mx.array(self._MEAN)
        std = mx.array(self._STD)
        pixels = (pixels / 255.0 - mean) / std
        temporal = self.config.vision_config.temporal_patch_size
        pixels = mx.broadcast_to(pixels[None, ...], (temporal, height, width, 3))
        grid_h, grid_w = height // patch, width // patch
        pixels = pixels.reshape(
            temporal,
            grid_h // merge,
            merge,
            patch,
            grid_w // merge,
            merge,
            patch,
            3,
        )
        pixels = pixels.transpose(1, 4, 2, 5, 7, 0, 3, 6)
        patch_dim = 3 * temporal * patch * patch
        return pixels.reshape(-1, patch_dim), (1, grid_h, grid_w)

    def _validate_source_size(self, image: Any) -> None:
        if (image.width < 1 or image.height < 1
                or image.width * image.height > self.max_source_pixels):
            raise ValueError("image exceeds the source pixel budget")


def _smart_resize(
    width: int, height: int, *, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    if width < 1 or height < 1 or max(width, height) / min(width, height) > 200:
        raise ValueError("image dimensions are empty or have an extreme aspect ratio")
    resized_w = max(factor, round(width / factor) * factor)
    resized_h = max(factor, round(height / factor) * factor)
    pixels = resized_w * resized_h
    if pixels > max_pixels:
        scale = math.sqrt((width * height) / max_pixels)
        resized_w = max(factor, math.floor(width / scale / factor) * factor)
        resized_h = max(factor, math.floor(height / scale / factor) * factor)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / (width * height))
        resized_w = max(factor, math.ceil(width * scale / factor) * factor)
        resized_h = max(factor, math.ceil(height * scale / factor) * factor)
    if resized_w * resized_h > max_pixels:
        raise ValueError("patch-aligned image exceeds the pixel budget")
    return resized_w, resized_h
