"""Pinned DETR RGB resize/normalization and box postprocessing candidate."""

import numpy as np

from mlx_one.vision.images import decode_image


def preprocess_detr(image):
    from PIL import Image

    decoded = decode_image(image, reject_transparency=True)
    resized = None
    try:
        width, height = decoded.size
        short, long = min(width, height), max(width, height)
        size = 800
        raw_size = None
        if long / short * size > 1333:
            raw_size = 1333 * short / long
            size = round(raw_size)
        long_scale = size if raw_size is None else raw_size
        if width <= height:
            target = (size, int(long_scale * height / width))
        else:
            target = (int(long_scale * width / height), size)
        resized = decoded.resize(target, Image.Resampling.BILINEAR)
        pixels = (np.asarray(resized).astype(np.float64) * (1 / 255)).astype(np.float32)
        pixels = (pixels - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        return pixels, np.ones(pixels.shape[:2], np.int64), (width, height)
    finally:
        if resized is not None:
            resized.close()
        decoded.close()


def postprocess_detr(logits, pred_boxes, target_size, threshold=0.9):
    """Source softmax excludes no-object for max, absolute xyxy, no clipping/NMS."""
    values = np.asarray(logits, dtype=np.float32)
    values = np.exp(values - values.max(axis=-1, keepdims=True))
    values /= values.sum(axis=-1, keepdims=True)
    labels = values[..., :-1].argmax(axis=-1)
    scores = values[..., :-1].max(axis=-1)
    cx, cy, w, h = np.moveaxis(np.asarray(pred_boxes, np.float32), -1, 0)
    boxes = np.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), axis=-1)
    width, height = target_size
    boxes *= np.array([width, height, width, height], np.float32)
    keep = scores > threshold
    return {"scores": scores[keep], "labels": labels[keep], "boxes": boxes[keep]}
