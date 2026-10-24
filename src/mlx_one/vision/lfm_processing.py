"""Liquid static-image NaFlex packing and checkpoint-configured tiling.

The resize and tile rules follow the MIT-licensed MLX-VLM LFM2-VL processor;
see models/vision_language/UPSTREAM.md. No external inference runtime is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mlx_one.vision.images import MAX_IMAGES, decode_image
from mlx_one.vision.processing import _smart_resize


@dataclass(frozen=True)
class PreparedLFMImage:
    input_ids: object
    pixel_values: object
    spatial_shapes: tuple
    image_metadata: tuple[dict, ...] = ()


class LFMImageProcessor:
    def __init__(self, tokenizer, config, metadata):
        self.tokenizer, self.config = tokenizer, config
        self.legacy = metadata.get("image_processor_type") == "MLXOneLFM2VLImageProcessor"
        if not self.legacy and config.vision_config.model_type != "siglip2_vision_model":
            raise ValueError("unsupported Liquid vision processor contract")
        self.mean, self.std = metadata["image_mean"], metadata["image_std"]
        if (
            len(self.mean) != 3
            or len(self.std) != 3
            or any(not math.isfinite(v) for v in (*self.mean, *self.std))
            or any(v <= 0 for v in self.std)
        ):
            raise ValueError("invalid LFM2-VL normalization metadata")
        if (
            metadata.get("do_normalize", True) is not True
            or metadata.get("do_rescale", True) is not True
        ):
            raise ValueError("unsupported Liquid normalization policy")
        if metadata.get("rescale_factor", 1 / 255) != 1 / 255:
            raise ValueError("unsupported Liquid rescaling factor")
        self.resample = int(metadata.get("resample", 3))
        if self.resample not in (2, 3):
            raise ValueError("unsupported Liquid image resampler")
        self.splitting = metadata.get("do_image_splitting", True) and not self.legacy
        self.thumbnail = metadata.get("use_thumbnail", True)
        self.min_tiles, self.max_tiles = metadata.get("min_tiles", 2), metadata.get("max_tiles", 10)
        self.tile_size = metadata.get("tile_size", 512)
        self.min_tokens, self.max_tokens = (
            metadata.get("min_image_tokens", 64),
            metadata.get("max_image_tokens", 256),
        )
        self.tolerance = metadata.get("max_pixels_tolerance", 2.0)
        self.special = metadata.get("use_image_special_tokens", True) and not self.legacy
        if (
            not 1 <= self.min_tiles <= self.max_tiles <= 10
            or not 1 <= self.min_tokens <= self.max_tokens <= 4096
            or not 1 <= self.tile_size <= 2048
            or not 1 <= self.tolerance <= 4
        ):
            raise ValueError("invalid Liquid image processing budget")
        v = config.vision_config
        if (
            metadata.get("encoder_patch_size", metadata.get("patch_size", v.patch_size))
            != v.patch_size
        ):
            raise ValueError("processor/model patch size mismatch")
        if metadata.get("downsample_factor", v.downsample_factor) != v.downsample_factor:
            raise ValueError("processor/model downsampling mismatch")

    def _layout(self, width, height):
        v = self.config.vision_config
        factor = v.patch_size * v.downsample_factor
        if self.legacy:
            return (1, 1), [(v.image_size, v.image_size)]
        size = _smart_resize(
            width,
            height,
            factor=factor,
            min_pixels=self.min_tokens * factor**2,
            max_pixels=self.max_tokens * factor**2,
        )
        aligned = max(factor, round(width / factor) * factor) * max(
            factor, round(height / factor) * factor
        )
        if not self.splitting or aligned <= self.max_tokens * factor**2 * self.tolerance:
            return (1, 1), [size]
        ratios = sorted(
            {
                (w, h)
                for w in range(1, self.max_tiles + 1)
                for h in range(1, self.max_tiles + 1)
                if self.min_tiles <= w * h <= self.max_tiles
            },
            key=lambda item: (item[0] * item[1], item[0], item[1]),
        )
        best, diff = (1, 1), float("inf")
        for ratio in ratios:
            current = abs(width / height - ratio[0] / ratio[1])
            if current < diff or (
                current == diff and width * height > 0.5 * self.tile_size**2 * ratio[0] * ratio[1]
            ):
                best, diff = ratio, current
        sizes = [(self.tile_size, self.tile_size)] * (best[0] * best[1])
        if self.thumbnail:
            sizes.append(size)
        return best, sizes

    def estimate_patches(self, images):
        patch = self.config.vision_config.patch_size
        return sum(sum(w * h // patch**2 for w, h in self._layout(*im.size)[1]) for im in images)

    def prepare(self, prompt, images):
        import mlx.core as mx
        import numpy as np
        from PIL import Image

        if (
            not isinstance(prompt, str)
            or not prompt
            or len(images) > MAX_IMAGES
            or prompt.count("<image>") != len(images)
        ):
            raise ValueError("image placeholders must match at most eight images")
        v = self.config.vision_config
        if v.num_channels != 3:
            raise ValueError("LFM2-VL processor requires RGB")
        patches, shapes, metadata = [], [], []
        pieces = prompt.split("<image>")
        ids = list(self.tokenizer.encode(pieces[0]))
        for image, text in zip(images, pieces[1:], strict=True):
            source = decode_image(image)
            views = []
            try:
                layout, sizes = self._layout(*source.size)
                cols, rows = layout
                if cols * rows > 1:
                    tiled = source.resize(
                        (cols * self.tile_size, rows * self.tile_size),
                        Image.Resampling(self.resample),
                    )
                    try:
                        for row in range(rows):
                            for col in range(cols):
                                views.append(
                                    tiled.crop(
                                        (
                                            col * self.tile_size,
                                            row * self.tile_size,
                                            (col + 1) * self.tile_size,
                                            (row + 1) * self.tile_size,
                                        )
                                    )
                                )
                    finally:
                        tiled.close()
                    if self.thumbnail:
                        views.append(source.resize(sizes[-1], Image.Resampling(self.resample)))
                else:
                    views.append(source.resize(sizes[0], Image.Resampling(self.resample)))
                visual_count, patch_count = 0, 0
                if self.special:
                    ids += self.tokenizer.encode("<|image_start|>")
                for index, view in enumerate(views):
                    width, height = view.size
                    gh, gw = height // v.patch_size, width // v.patch_size
                    pixels = (
                        mx.array(np.asarray(view).copy(), dtype=mx.float32).reshape(
                            height, width, 3
                        )
                        / 255.0
                    )
                    pixels = (pixels - mx.array(self.mean)) / mx.array(self.std)
                    order = (0, 2, 4, 1, 3) if self.legacy else (0, 2, 1, 3, 4)
                    patches.append(
                        pixels.reshape(gh, v.patch_size, gw, v.patch_size, 3)
                        .transpose(*order)
                        .reshape(gh * gw, -1)
                    )
                    shapes.append((gh, gw))
                    count = math.ceil(gh / v.downsample_factor) * math.ceil(
                        gw / v.downsample_factor
                    )
                    if self.special and cols * rows > 1:
                        marker = (
                            f"<|img_row_{index // cols + 1}_col_{index % cols + 1}|>"
                            if index < cols * rows
                            else "<|img_thumbnail|>"
                        )
                        ids += self.tokenizer.encode(marker)
                    ids += [self.config.image_token_id] * count
                    visual_count += count
                    patch_count += gh * gw
                if self.special:
                    ids += self.tokenizer.encode("<|image_end|>")
                metadata.append(
                    {
                        "source_size": list(source.size),
                        "processed_sizes": [list(s) for s in sizes],
                        "tile_grid": [cols, rows],
                        "patches": patch_count,
                        "visual_tokens": visual_count,
                    }
                )
                ids += self.tokenizer.encode(text)
            finally:
                source.close()
                for view in views:
                    view.close()
        max_patches = max((p.shape[0] for p in patches), default=0)
        pixels = (
            mx.stack([mx.pad(p, ((0, max_patches - p.shape[0]), (0, 0))) for p in patches])
            if patches
            else None
        )
        return PreparedLFMImage(
            mx.array([ids], dtype=mx.int32), pixels, tuple(shapes), tuple(metadata)
        )
