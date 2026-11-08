"""Golden uint8 pixels from the pinned SAM processor's torchvision v2 backend."""

import hashlib

import numpy as np
import pytest

from mlx_one.segmentation.resizing import resize_rgb_uint8


@pytest.mark.parametrize(
    "size,expected",
    [
        ((3, 5), "1dbc2b29ac7df9f3bbb2cbdd36881a79c37a69793cab9214c084e15002c54824"),
        ((15, 19), "8879430d601460690654574bc260bc5177a22b2865c2140495aa47a7c14bc18a"),
        ((3, 19), "11805c362f54b1eaea3022781df94a994d6e1fe2d7e993a9d5f315243dac55ba"),
    ],
)
def test_antialias_and_each_pass_rounding_match_reference(size, expected):
    image = ((np.arange(7 * 11 * 3).reshape(7, 11, 3) * 37 + 11) % 256).astype(np.uint8)
    output = resize_rgb_uint8(image, *size)
    assert output.shape == (*size, 3)
    assert hashlib.sha256(output.tobytes()).hexdigest() == expected
