"""SAM3 processor assets and original-image coordinate validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_image(image):
    from PIL import Image, ImageOps

    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return ImageOps.exif_transpose(source).convert("RGB").copy()
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError("frames must be RGB uint8 arrays with shape [height,width,3]")
    return Image.fromarray(value)


def validate_coordinates(value, width, height, *, boxes=False):
    value = np.asarray(value, dtype=np.float32)
    n = 4 if boxes else 2
    if value.ndim != 2 or value.shape[1] != n or not np.isfinite(value).all():
        raise ValueError(f"coordinates must be finite [N,{n}] pixel coordinates")
    if (
        (value[:, 0::2] < 0).any()
        or (value[:, 0::2] > width).any()
        or (value[:, 1::2] < 0).any()
        or (value[:, 1::2] > height).any()
    ):
        raise ValueError("prompt coordinates are outside the original image")
    if boxes and ((value[:, 2] <= value[:, 0]).any() or (value[:, 3] <= value[:, 1]).any()):
        raise ValueError("boxes must have positive width and height in XYXY format")
    return value


def validate_labels(labels, count, *, default=None):
    if labels is None:
        if default is None:
            raise ValueError("point labels are required")
        return np.full(count, default, dtype=np.int64)
    labels = np.asarray(labels)
    if labels.shape != (count,) or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain exactly one 0/1 label per prompt")
    return labels.astype(np.int64)


class Sam3Processor:
    def __init__(self, assets, tokenizer, image_size):
        self.assets = assets.get("image_processor", assets)
        self.tokenizer = tokenizer
        self.image_size = image_size
        size = self.assets.get("size", {"height": image_size, "width": image_size})
        if size != {"height": image_size, "width": image_size}:
            raise ValueError("processor size does not match the checkpoint architecture")
        if self.assets.get("resample", 2) != 2:
            raise ValueError("SAM3 requires the checkpoint bilinear resize processor")
        if self.assets.get("do_pad") or self.assets.get("do_center_crop"):
            raise ValueError("padding/cropping processors are unsupported for SAM3")
        self.mean = np.asarray(self.assets.get("image_mean", [0.5] * 3), dtype=np.float32)
        self.std = np.asarray(self.assets.get("image_std", [0.5] * 3), dtype=np.float32)
        if (
            self.mean.shape != (3,)
            or self.std.shape != (3,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or (self.std <= 0).any()
        ):
            raise ValueError("invalid processor normalization statistics")

    def image(self, image):
        from mlx_one.models.segmentation.sam3._runtime import Tensor

        from .resizing import resize_rgb_uint8

        image = read_image(image)
        resized = resize_rgb_uint8(image, self.image_size, self.image_size)
        pixels = np.asarray(resized, dtype=np.float32) * self.assets.get("rescale_factor", 1 / 255)
        pixels = (pixels - self.mean) / self.std
        return Tensor(pixels.transpose(2, 0, 1)[None]), image

    def text(self, text):
        from mlx_one.models.segmentation.sam3._runtime import ops

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty concept")
        encoding = self.tokenizer.encode(text)
        return ops.tensor([encoding.ids], dtype=ops.long), ops.tensor(
            [encoding.attention_mask], dtype=ops.long
        )

    def concept_boxes(self, boxes, labels, size):
        from mlx_one.models.segmentation.sam3._runtime import Tensor, ops

        h, w = size
        xyxy = validate_coordinates(boxes, w, h, boxes=True)
        labels = validate_labels(labels, len(xyxy), default=1)
        centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2
        wh = xyxy[:, 2:] - xyxy[:, :2]
        normalized = np.concatenate([centers, wh], axis=-1) / np.asarray([w, h, w, h])
        return Tensor(normalized[None].astype(np.float32)), ops.tensor(labels[None], dtype=ops.long)


def mask_boxes(masks):
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    for i, mask in enumerate(masks):
        y, x = np.nonzero(mask)
        if len(x):
            boxes[i] = [x.min(), y.min(), x.max(), y.max()]
    return boxes
