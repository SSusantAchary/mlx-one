import sys
from types import ModuleType

import numpy as np
import pytest

from mlx_one.core.cache import KVCache
from mlx_one.engine.optimization_policy import NativeMathPolicy
from mlx_one.engine.pressure import activation_forecast


@pytest.fixture
def core(monkeypatch):
    package, mx = ModuleType("mlx"), ModuleType("mlx.core")
    allocations = []
    def zeros(shape, dtype):
        allocations.append(shape)
        return np.zeros(shape, dtype=dtype)
    mx.zeros, package.core = zeros, mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return allocations


@pytest.mark.parametrize("width", (256, 512, 1024, 2048))
@pytest.mark.parametrize("offset", (1, 127, 511, 512, 8191))
def test_chunks_preserve_logical_positions_and_existing_reserved_buffer(core, width, offset):
    rng = np.random.default_rng(97)
    keys = rng.normal(size=(1, 2, offset + width + 2, 4)).astype(np.float16)
    values = rng.normal(size=keys.shape).astype(np.float16)
    full, chunked = KVCache(), KVCache()
    full.update(keys, values)
    chunked.update(keys[:, :, :offset], values[:, :, :offset])
    for start in range(offset, keys.shape[2], width):
        end = min(start + width, keys.shape[2])
        chunked.update(keys[:, :, start:end], values[:, :, start:end])
        assert chunked.offset == end
    np.testing.assert_array_equal(chunked.state()[0], full.state()[0])
    np.testing.assert_array_equal(chunked.state()[1], full.state()[1])
    assert chunked.allocated_nbytes >= chunked.nbytes
    # Existing KVCache already grows in 256-token steps. Do not introduce a
    # second buffer owner or silently replace this storage contract.
    count = len(core)
    chunked.update(keys[:, :, :1], values[:, :, :1])
    assert len(core) == count
    chunked.reset()
    chunked.reset()
    assert chunked.offset == chunked.nbytes == chunked.allocated_nbytes == 0


def test_last_only_forecast_reduces_only_verified_vocab_rows():
    from types import SimpleNamespace

    config = SimpleNamespace(hidden_size=64, intermediate_size=128,
                             num_attention_heads=4, vocab_size=32768)
    old = activation_forecast(config, batch=1, chunk=512, history=128)
    policy = NativeMathPolicy(logits="last_only", attention="mlx_sdpa")
    new = activation_forecast(config, batch=1, chunk=512, history=128, math_policy=policy)
    from mlx_one.models.shared.execution import attention_scratch_bound

    overlap = attention_scratch_bound(batch=1, heads=4, head_dim=16, chunk=512, history=128)
    assert new["bytes"] == old["bytes"] - 511 * config.vocab_size * 4 + overlap
    assert new["measured"] is False
    assert new["bytes"] >= 4 * 512 * (128 + 512) * 4
