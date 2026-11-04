"""Pinned DPT slow processor used by the selected relative-depth checkpoint."""

import numpy as np

from mlx_one.vision.images import decode_image


def preprocess_depth(image):
    from PIL import Image

    decoded = decode_image(image, reject_transparency=True)
    resized = None
    try:
        width, height = decoded.size
        sh, sw = 518 / height, 518 / width
        scale = sw if abs(1 - sw) < abs(1 - sh) else sh
        target_w, target_h = round(scale * width / 14) * 14, round(scale * height / 14) * 14
        if min(target_w, target_h) < 14 or target_w * target_h > 1_000_000:
            raise ValueError("depth image exceeds the candidate processed-pixel budget")
        resized = decoded.resize((target_w, target_h), Image.Resampling.BICUBIC)
        pixels = (np.asarray(resized).astype(np.float64) * (1 / 255)).astype(np.float32)
        pixels = (pixels - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        return pixels, (width, height)
    finally:
        if resized is not None:
            resized.close()
        decoded.close()
