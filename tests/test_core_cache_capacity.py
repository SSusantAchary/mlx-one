import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from mlx_one.core.cache import KVCache, QuantizedKVCache
from mlx_one.core.outputs import ModelOutput
from mlx_one.engine import CausalLMRunner, EngineRequest, ExecutionBatch, RequestKind, SequenceState
from mlx_one.engine.cache import CacheBundle


def install_fake_mlx(monkeypatch) -> None:
    package = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.zeros = np.zeros
    core.concatenate = np.concatenate
    core.float16 = np.float16
    core.bfloat16 = np.float32
    core.quantize = lambda values, **_: (
        values,
        np.ones_like(values),
        np.zeros_like(values),
    )
    core.dequantize = lambda weights, scales, biases, **_: weights * scales + biases
    core.array = np.array
    core.eval = lambda *values: values
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def test_kv_cache_grows_in_capacity_chunks_and_snapshots_active_state(monkeypatch) -> None:
    install_fake_mlx(monkeypatch)
    cache = KVCache()
    keys = np.ones((1, 2, 3, 4), dtype=np.float16)
    active_keys, active_values = cache.update(keys, keys)
    assert cache.offset == 3
    assert cache.keys.shape[2] == 256
    assert active_keys.shape == active_values.shape == keys.shape

    storage = cache.keys
    cache.update(np.ones((1, 2, 2, 4), dtype=np.float16), keys[:, :, :2, :])
    assert cache.keys is storage
    assert cache.offset == 5
    assert cache.trim(2)
    assert cache.offset == 3

    snapshot = cache.snapshot()
    assert snapshot.keys.shape[2] == 3
    snapshot.update(np.zeros((1, 2, 1, 4), dtype=np.float16), keys[:, :, :1, :])
    assert snapshot.offset == 4
    assert cache.offset == 3


def test_quantized_kv_cache_uses_capacity_growth_for_each_component(monkeypatch) -> None:
    install_fake_mlx(monkeypatch)
    cache = QuantizedKVCache(4, None, group_size=4)
    values = np.ones((1, 2, 3, 4), dtype=np.float16)
    keys, returned_values = cache.update(values, values)
    assert keys.shape[2] == returned_values.shape[2] == 3
    assert cache._keys[0].shape[2] == 256
    assert cache._values.shape[2] == 256
    key_storage = cache._keys[0]
    cache.update(values[:, :, :1, :], values[:, :, :1, :])
    assert cache._keys[0] is key_storage
    assert cache.offset == 4
    assert cache.trim(2)
    assert cache.offset == 2


def test_attention_caches_batch_select_and_merge(monkeypatch) -> None:
    install_fake_mlx(monkeypatch)
    values = np.ones((1, 2, 3, 4), dtype=np.float16)
    first, second = KVCache(), KVCache()
    first.update(values, values)
    second.update(values * 2, values * 2)
    merged = CacheBundle.merge((CacheBundle((first,)), CacheBundle((second,))))
    assert merged.entries[0].state()[0].shape == (2, 2, 3, 4)
    selected = merged.batch_select((1,))
    assert selected.entries[0].state()[0].shape == (1, 2, 3, 4)
    assert np.all(selected.entries[0].state()[0] == 2)


def test_causal_runner_batches_compatible_attention_decode(monkeypatch) -> None:
    install_fake_mlx(monkeypatch)

    class Model:
        calls = 0

        def make_cache(self):
            return (KVCache(),)

        def __call__(self, token_ids, *, cache):
            self.calls += 1
            values = np.ones((token_ids.shape[0], 1, 1, 4), dtype=np.float16)
            cache[0].update(values, values)
            return ModelOutput(np.zeros((token_ids.shape[0], 1, 8)), values, cache)

    model = Model()
    runner = CausalLMRunner(
        SimpleNamespace(
            model=model,
            tokenizer=SimpleNamespace(encode=lambda _: [1], bos_token_id=None),
        )
    )
    sequences = []
    for index in range(2):
        sequence = runner.prepare(
            EngineRequest(RequestKind.TEXT, "model", {"prompt": "p"})
        )
        sequence.transition(SequenceState.PREFILL)
        sequence.computed_tokens = 1
        sequence.cache.entries[0].update(
            np.ones((1, 1, 1, 4), dtype=np.float16),
            np.ones((1, 1, 1, 4), dtype=np.float16),
        )
        sequence.generated_tokens.append(index + 2)
        sequence.transition(SequenceState.DECODING)
        sequences.append(sequence)
    outputs = runner.decode(ExecutionBatch(tuple(sequences), "decode", 2))
    assert runner.capabilities.batchable
    assert model.calls == 1
    assert len(outputs) == 2
    assert all(sequence.cache.offset == 2 for sequence in sequences)
