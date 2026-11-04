"""Pinned CLIP slow image-processor behavior using bounded native image decoding."""

import numpy as np

from mlx_one.vision.images import decode_image


def preprocess_clip(image):
    from PIL import Image

    decoded = decode_image(image, reject_transparency=True)
    resized = None
    cropped = None
    try:
        width, height = decoded.size
        if width <= height:
            target = (224, int(height * 224 / width))
        else:
            target = (int(width * 224 / height), 224)
        resized = decoded.resize(target, Image.Resampling.BICUBIC)
        left = (resized.width - 224) // 2
        top = (resized.height - 224) // 2
        cropped = resized.crop((left, top, left + 224, top + 224))
        pixels = (np.asarray(cropped).astype(np.float64) / 255.0).astype(np.float32)
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        return (pixels - mean) / std, (width, height)
    finally:
        if cropped is not None:
            cropped.close()
        if resized is not None:
            resized.close()
        decoded.close()
