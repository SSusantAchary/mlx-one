"""Independent NumPy diagnostic oracle. Never imported by the runtime package."""

import numpy as np


def pack(indices):
    values = np.asarray(indices)
    if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values > 7):
        raise ValueError("indices must be integers in [0,7]")
    out = np.zeros((*values.shape[:-1], (values.shape[-1] + 9) // 10), dtype=np.uint32)
    # Deliberately scalar indexing, independent of GPU vectorized decoder.
    for col in range(values.shape[-1]):
        out[..., col // 10] |= values[..., col].astype(np.uint32) << (3 * (col % 10))
    return out


def reconstruct(packed, scales, codebook, inputs):
    words = np.asarray(packed)
    scales, codebook = np.asarray(scales), np.asarray(codebook)
    if words.dtype != np.uint32 or words.shape[-1] != (inputs + 9) // 10:
        raise ValueError("invalid packed storage")
    out = np.empty((*words.shape[:-1], inputs), np.float64)
    for col in range(inputs):
        code = (words[..., col // 10].astype(np.uint64) // (2 ** (3 * (col % 10)))) % 8
        out[..., col] = codebook[code.astype(np.int64)] * scales[..., col // 64].astype(np.float64)
    return out


def rotate_power2(x, signs):
    # Sylvester matrix built independently from the GPU butterfly operation.
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError("diagnostic Sylvester transform requires power of two")
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (np.asarray(x, np.float64) * signs) @ h.T / np.sqrt(n)


def fixture(outputs=67, inputs=128, experts=None, seed=42):
    CODEBOOK = (
        -2.15234375,
        -1.34375,
        -0.755859375,
        -0.2451171875,
        0.2451171875,
        0.755859375,
        1.34375,
        2.15234375,
    )
    rng = np.random.default_rng(seed)
    shape = (outputs, inputs) if experts is None else (experts, outputs, inputs)
    indices = rng.integers(0, 8, shape, dtype=np.uint32)
    scales = rng.uniform(0.005, 0.025, (*shape[:-1], inputs // 64)).astype(np.float16)
    signs = rng.choice([-1, 1], inputs).astype(np.float16)
    return dict(
        weight=pack(indices), scales=scales, codebook=np.array(CODEBOOK, np.float16), signs=signs
    )
