"""Separable antialiased bilinear sampling coefficients (half-pixel centers)."""

import numpy as np


def bilinear_coefficients(input_size, output_size):
    scale = input_size / output_size
    support = max(1.0, scale)
    centers = (np.arange(output_size, dtype=np.float32) + 0.5) * scale
    left = np.maximum(0, (centers - support + 0.5).astype(np.int32))
    right = np.minimum(input_size, (centers + support + 0.5).astype(np.int32))
    count = int((right - left).max())
    indices = left[:, None] + np.arange(count)[None, :]
    weights = np.maximum(
        0, 1 - np.abs((indices.astype(np.float32) - centers[:, None] + 0.5) / support)
    )
    weights = np.where(indices < right[:, None], weights, 0).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.clip(indices, 0, input_size - 1).astype(np.int32), weights


def resize_rgb_uint8(image, height, width):
    """Match the reference tensor resize: float interpolation, then uint8 rounding."""
    value = np.asarray(image, dtype=np.float32)
    if value.shape[:2] == (height, width):
        return value.astype(np.uint8)
    ix, wx = bilinear_coefficients(value.shape[1], width)
    horizontal = np.sum(value[:, ix, :] * wx[None, :, :, None], axis=2)
    iy, wy = bilinear_coefficients(value.shape[0], height)
    vertical = np.sum(horizontal[iy, :, :] * wy[:, :, None, None], axis=1)
    return np.clip(np.rint(vertical), 0, 255).astype(np.uint8)
