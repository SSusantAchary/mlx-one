"""Native NHWC bilinear/bicubic interpolation for source-faithful depth math.

Bicubic follows PyTorch's alpha=-0.75, align_corners=False, no antialias rule.
No source framework is imported; no existing image processor is modified.
"""

import mlx.core as mx


def resize(x, size, *, mode="bilinear", align_corners=False):
    if x.ndim != 4 or mode not in {"bilinear", "bicubic"} or min(size) < 1:
        raise ValueError("resize requires NHWC input and positive H,W target")
    result = x
    # Source bicubic performs horizontal interpolation before vertical.
    for axis, length in [(2, int(size[1])), (1, int(size[0]))]:
        old = result.shape[axis]
        if length == old:
            continue
        if align_corners:
            scale = 0 if length == 1 else (old - 1) / (length - 1)
            coordinate = mx.arange(length, dtype=mx.float32) * scale
        else:
            coordinate = (mx.arange(length, dtype=mx.float32) + 0.5) * (old / length) - 0.5
        if mode == "bilinear":
            coordinate = mx.maximum(coordinate, 0)
            base = mx.floor(coordinate).astype(mx.int32)
            fraction = coordinate - base
            shape = [1] * 4
            shape[axis] = length
            weight = fraction.reshape(shape)
            result = (
                mx.take(result, mx.minimum(base, old - 1), axis=axis) * (1 - weight)
                + mx.take(result, mx.minimum(base + 1, old - 1), axis=axis) * weight
            )
        else:
            base = mx.floor(coordinate).astype(mx.int32)
            fraction = coordinate - base

            def cubic(t):
                t = mx.abs(t)
                a = -0.75
                return mx.where(
                    t <= 1,
                    ((a + 2) * t - (a + 3)) * t * t + 1,
                    mx.where(t < 2, ((a * t - 5 * a) * t + 8 * a) * t - 4 * a, 0),
                )

            terms = []
            shape = [1] * 4
            shape[axis] = length
            for offset in (-1, 0, 1, 2):
                index = mx.clip(base + offset, 0, old - 1)
                weight = cubic(fraction - offset).reshape(shape)
                terms.append(mx.take(result, index, axis=axis) * weight)
            result = terms[0] + terms[1] + terms[2] + terms[3]
    return result
