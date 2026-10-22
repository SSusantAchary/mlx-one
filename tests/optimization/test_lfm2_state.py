import sys
from types import ModuleType

import numpy as np
import pytest

from mlx_one.core.cache import ConvCache


@pytest.fixture
def core(monkeypatch):
    package, mx = ModuleType("mlx"), ModuleType("mlx.core")
    mx.zeros, mx.concatenate = np.zeros, np.concatenate
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return mx


@pytest.mark.parametrize("length", (1, 2, 3, 31, 32, 33, 127, 128, 511, 512, 2048, 8192))
def test_convolution_carry_full_and_chunks_preserve_precision_and_position(core, length):
    rng = np.random.default_rng(12)
    values = rng.normal(size=(1, length, 4)).astype(np.float16)
    full, chunked = ConvCache(3), ConvCache(3)
    combined = full.update(values)
    pieces = []
    for start in range(0, length, 127):
        end = min(start + 127, length)
        piece = chunked.update(values[:, start:end])
        # An independent 3-wide depthwise finite impulse-response fixture.
        windows = np.lib.stride_tricks.sliding_window_view(piece, 3, axis=1)
        pieces.append(windows.sum(-1))
        assert chunked.offset == end
    expected = np.lib.stride_tricks.sliding_window_view(combined, 3, axis=1).sum(-1)
    np.testing.assert_array_equal(np.concatenate(pieces, axis=1), expected)
    np.testing.assert_array_equal(chunked.values, full.values)
    assert chunked.values.dtype == np.float16


def test_fork_failure_reset_and_interleaving_leave_donor_independent(core, monkeypatch):
    donor, peer = ConvCache(3), ConvCache(3)
    values = np.ones((1, 4, 3), dtype=np.float16)
    donor.update(values)
    fork = donor.clone()
    before = donor.values.copy()
    peer.update(values * 7)
    fork.update(values * 2)
    np.testing.assert_array_equal(donor.values, before)
    def failed(*_args, **_kwargs):
        raise MemoryError("injected allocation")
    monkeypatch.setattr(core, "concatenate", failed)
    with pytest.raises(MemoryError):
        donor.update(values)
    np.testing.assert_array_equal(donor.values, before)
    assert donor.offset == 4 and peer.offset == 4 and fork.offset == 8
    fork.reset()
    fork.reset()
    assert fork.offset == fork.nbytes == 0
    np.testing.assert_array_equal(donor.values, before)
