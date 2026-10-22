from contextvars import ContextVar

from mlx_one.engine.device_executor import DeviceExecutor, device_iterator


def test_lazy_device_iterator_preserves_request_context():
    executor = DeviceExecutor()
    request = ContextVar("test_request", default=None)

    def stream():
        yield request.get(), executor.is_owner
        yield request.get(), executor.is_owner

    try:
        token = request.set("request-a")
        iterator = device_iterator(executor, "model", stream())
        assert next(iterator) == ("request-a", True)
        request.reset(token)
        token = request.set("request-b")
        assert next(iterator) == ("request-a", True)
        request.reset(token)
        iterator.close()
    finally:
        iterator.close()
        executor.close()


def test_observer_scope_restores_after_failure():
    from mlx_one.engine.observation import current_observer, observe_request

    marker = object()
    try:
        with observe_request(marker):
            assert current_observer() is marker
            raise ValueError("injected")
    except ValueError:
        pass
    assert current_observer() is None


def test_close_with_running_submission_does_not_read_closed_public_model(monkeypatch):
    import threading
    import time
    from pathlib import Path

    import mlx_one.engine.public as public
    from mlx_one.text import GenerationChunk, LoadedTextModel

    ready, resume = threading.Event(), threading.Event()

    class Tokenizer:
        def encode(self, _text):
            return [1]

    def chunks(*_args, **_kwargs):
        ready.set()
        assert resume.wait(5)
        yield GenerationChunk(1, "ok", 1)
        yield GenerationChunk(None, "", 1, "stop")

    monkeypatch.setattr(public, "stream_generate", chunks)
    bundle = LoadedTextModel(object(), Tokenizer(), Path('.'), 'test/model', 'revision')
    engine = public.Engine(bundle)
    handle = engine.submit('prompt')
    assert ready.wait(5)
    closer = threading.Thread(target=engine.close)
    closer.start()
    try:
        end = time.monotonic() + 5
        while not engine._closed and time.monotonic() < end:
            time.sleep(.01)
        assert engine._closed
    finally:
        resume.set()
        closer.join(5)
    assert not closer.is_alive()
    assert handle.result(timeout=5).text == 'ok'


def test_incremental_decoding_holds_incomplete_unicode_and_flushes_literal_replacement():
    from mlx_one.text.generation import _decode_piece

    class Bytes:
        def decode(self, ids):
            return bytes(ids).decode('utf-8', errors='replace')

    tokenizer = Bytes()
    generated, emitted, text = [], '', ''
    for token in '€🙂'.encode():
        generated.append(token)
        piece, emitted = _decode_piece(tokenizer, generated, emitted)
        text += piece
        assert '\ufffd' not in text
    piece, emitted = _decode_piece(tokenizer, generated, emitted, final=True)
    assert text + piece == '€🙂'
    replacement = list('\ufffd'.encode('utf-8'))
    piece, emitted = _decode_piece(tokenizer, replacement, '')
    assert piece == ''
    piece, _ = _decode_piece(tokenizer, replacement, emitted, final=True)
    assert piece == '\ufffd'


def test_terminal_text_flush_reaches_http_without_adding_a_token(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import numpy as np
    from fastapi.testclient import TestClient

    from mlx_one.core.cache import KVCache
    from mlx_one.core.outputs import ModelOutput
    from mlx_one.server.app import create_app
    from mlx_one.server.config import ServerConfig
    from mlx_one.server.generation_engine import GenerationEngine
    from mlx_one.text import LoadedTextModel
    from mlx_one.text.chat import ChatTemplate

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 999

        def encode(self, _text):
            return [1, 2]

        def decode(self, ids):
            return bytes(ids).decode('utf-8', errors='replace')

    class Model:
        config = SimpleNamespace(model_type='qwen2', num_hidden_layers=1,
            num_key_value_heads=1, num_attention_heads=1, hidden_size=2,
            max_position_embeddings=128)

        def parameters(self):
            return {'weight': np.ones((2,), dtype=np.float16)}

        def make_cache(self):
            return (KVCache(),)

        def __call__(self, ids, *, cache):
            value = np.zeros((1, 1, ids.shape[1], 2), dtype=np.float16)
            cache[0].update(value, value)
            return ModelOutput(logits=np.zeros((1, ids.shape[1], 256), dtype=np.float16),
                               last_hidden_state=None)

    import sys
    from types import ModuleType

    core = ModuleType('mlx.core')
    core.array, core.zeros, core.concatenate = np.array, np.zeros, np.concatenate
    core.float16, core.bfloat16 = np.float16, np.float32
    core.eval = lambda *_args: None
    core.clear_cache = lambda: None
    package = ModuleType('mlx')
    package.core = core
    monkeypatch.setitem(sys.modules, 'mlx', package)
    monkeypatch.setitem(sys.modules, 'mlx.core', core)
    tokenizer = Tokenizer()
    bundle = LoadedTextModel(Model(), tokenizer, tmp_path, 'test/model', 'a' * 40,
        architecture='qwen2', context_length=128,
        chat_template=ChatTemplate('qwen2', None, tokenizer))
    manager = SimpleNamespace(current_model=lambda: bundle,
        model_info=lambda: SimpleNamespace(id='test/model', context_length=128,
                                          to_dict=lambda: {}), unload=lambda: None)
    engine = GenerationEngine(manager, inference_mode='reference_dense')
    from mlx_one.text import generation

    ids = iter('\ufffd'.encode('utf-8'))
    monkeypatch.setattr(generation, 'select_token', lambda *_args: next(ids))
    config = ServerConfig(inference_mode='reference_dense', warmup=False, cache_prompt=False)
    with TestClient(create_app(manager, engine, config=config, ui_dir=tmp_path)) as client:
        response = client.post('/v1/chat/completions', json={
            'model': 'test/model', 'messages': [{'role': 'user', 'content': 'hi'}],
            'max_tokens': 3, 'stream': True})
        assert response.status_code == 200, response.text
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        assert all('error' not in event for event in events), response.text
        text = ''.join(event['choices'][0].get('delta', {}).get('content', '')
                       for event in events)
        assert text == '\ufffd'
        assert response.text.count('data: [DONE]') == 1
        assert engine.stats.generated_tokens == 3


def test_async_task_cancellation_signals_running_reference_request(monkeypatch):
    import asyncio
    import threading
    import time
    from pathlib import Path
    from types import SimpleNamespace

    import mlx_one.engine.public as public
    from mlx_one.text import GenerationChunk, LoadedTextModel

    ready, release, cleaned = threading.Event(), threading.Event(), threading.Event()

    class Tokenizer:
        def encode(self, _text):
            return [1]

    def chunks(*_args, **kwargs):
        try:
            ready.set()
            assert release.wait(5)
            for index in range(100):
                if kwargs['is_cancelled']():
                    break
                yield GenerationChunk(1, 'x', index + 1)
                time.sleep(.01)
        finally:
            cleaned.set()

    monkeypatch.setattr(public, 'stream_generate', chunks)
    engine = public.Engine(LoadedTextModel(object(), Tokenizer(), Path('.'), 'test/model', 'a'))
    engine._cache_manager = SimpleNamespace(close=lambda: None)

    async def run():
        task = asyncio.create_task(engine.async_generate('prompt'))
        assert await asyncio.to_thread(ready.wait, 5)
        task.cancel()
        release.set()
        with __import__('pytest').raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(cleaned.wait, .2), 'cancelled task left generation running'

    try:
        asyncio.run(run())
    finally:
        release.set()
        engine.close()


def test_active_direct_stream_observes_engine_shutdown(monkeypatch):
    from pathlib import Path

    import mlx_one.engine.public as public
    from mlx_one.text import GenerationChunk, LoadedTextModel

    class Tokenizer:
        def encode(self, _text):
            return [1]

    def chunks(*_args, **kwargs):
        for index in range(3):
            if kwargs['is_cancelled']():
                break
            yield GenerationChunk(1, 'x', index + 1)
        yield GenerationChunk(None, '', 1, 'stop')

    monkeypatch.setattr(public, 'stream_generate', chunks)
    engine = public.Engine(LoadedTextModel(object(), Tokenizer(), Path('.'), 'test/model', 'a'))
    stream = engine.stream('prompt')
    assert next(stream).token_id == 1
    engine.close()
    assert not any(chunk.token_id is not None for chunk in stream)


def test_invalid_context_is_rejected_before_cache_handle_allocation(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pytest

    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.cache_specs import CacheConfig
    from mlx_one.text import LoadedTextModel, stream_generate

    tokenizer = SimpleNamespace(encode=lambda text: [1] * len(text), bos_token_id=1)
    bundle = LoadedTextModel(object(), tokenizer, tmp_path, 'test/model', 'a', context_length=8)
    manager = object.__new__(CacheManager)
    manager.config = CacheConfig()
    manager.requested_mode = 'reference_dense'

    def allocate(*_args, **_kwargs):
        raise AssertionError('invalid context allocated a cache handle')

    manager.create_handle = allocate
    with pytest.raises(ValueError, match='no room'):
        list(stream_generate(bundle, 'xxxxxxxx', _cache_manager=manager))


def test_explicit_lfm_attention_softmax_precision_does_not_depend_on_mask(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType

    import numpy as np

    from mlx_one.core.cache import KVCache

    core = ModuleType('mlx.core')
    core.float16, core.float32 = np.float16, np.float32
    core.zeros, core.concatenate = np.zeros, np.concatenate
    dtypes = []

    def softmax(value, axis, precise):
        dtypes.append(value.dtype)
        result = np.exp(value - value.max(axis=axis, keepdims=True))
        return result / result.sum(axis=axis, keepdims=True)

    core.softmax = softmax
    nn = ModuleType('mlx.nn')
    nn.Module = object
    package = ModuleType('mlx')
    package.core, package.nn = core, nn
    transformer = ModuleType('mlx_one.models.shared.transformer')
    transformer.apply_rope = lambda value, **_kwargs: value
    for name, module in {'mlx': package, 'mlx.core': core, 'mlx.nn': nn,
                         'mlx_one.models.shared.transformer': transformer}.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / 'src/mlx_one/models/shared/hybrid.py'
    spec = importlib.util.spec_from_file_location('_p0_lfm_attention', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    attention = object.__new__(module.LfmAttention)
    attention.num_heads = attention.num_kv_heads = 1
    attention.head_dim, attention.scale, attention.rope_theta = 4, .5, 10000
    attention._mlx_one_activation_dtype = 'float16'
    for name in ('q_proj','k_proj','v_proj','out_proj','q_layernorm','k_layernorm'):
        setattr(attention, name, lambda value: value)
    hidden = np.arange(8, dtype=np.float16).reshape(1,2,4) / 10
    cache = KVCache()
    attention(hidden, mask=np.zeros((2,2), dtype=np.float32), cache=cache)
    attention(hidden[:,-1:], mask=None, cache=cache)
    assert dtypes == [np.float32, np.float32]
    del attention._mlx_one_activation_dtype
    dtypes.clear()
    attention(hidden[:,-1:], mask=None, cache=cache)
    # Native reference arithmetic accumulates low-precision attention in FP32 too.
    assert dtypes == [np.float32]
