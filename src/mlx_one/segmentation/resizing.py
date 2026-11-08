"""Separable antialiased bilinear sampling coefficients (half-pixel centers)."""

import numpy as np


def bilinear_coefficients(input_size, output_size):
    scale = input_size / output_size
    support = max(1.0, scale)
    centers = (np.arange(output_size, dtype=np.float64) + 0.5) * scale
    left = np.maximum(0, (centers - support + 0.5).astype(np.int32))
    right = np.minimum(input_size, (centers + support + 0.5).astype(np.int32))
    count = int((right - left).max())
    indices = left[:, None] + np.arange(count)[None, :]
    weights = np.maximum(
        0, 1 - np.abs((indices.astype(np.float64) - centers[:, None] + 0.5) / support)
    )
    weights = np.where(indices < right[:, None], weights, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.clip(indices, 0, input_size - 1).astype(np.int32), weights


def _fixed_coefficients(input_size, output_size):
    indices, weights = bilinear_coefficients(input_size, output_size)
    maximum = weights.max()
    precision = next(
        (bits for bits in range(22) if int(0.5 + maximum * (1 << (bits + 1))) >= (1 << 15)),
        22,
    )
    return indices, np.floor(0.5 + weights * (1 << precision)).astype(np.int32), precision


def resize_rgb_uint8(image, height, width):
    """Reference uint8 antialiased bilinear: fixed coefficients, round each pass.

    The tensor processor uses signed 16-bit coefficients with the largest safe
    precision and rounds horizontal results to uint8 before the vertical pass.
    A single float interpolation changes pixels and can alter ambiguous masks.
    """
    value = np.asarray(image, dtype=np.uint8)
    if value.shape[:2] == (height, width):
        return value.astype(np.uint8)
    ix, wx, px = _fixed_coefficients(value.shape[1], width)
    horizontal = (np.sum(value[:, ix, :] * wx[None, :, :, None], axis=2) + (1 << (px - 1))) >> px
    horizontal = np.clip(horizontal, 0, 255).astype(np.uint8)
    iy, wy, py = _fixed_coefficients(value.shape[0], height)
    vertical = (np.sum(horizontal[iy, :, :] * wy[:, :, None, None], axis=1) + (1 << (py - 1))) >> py
    return np.clip(vertical, 0, 255).astype(np.uint8)
