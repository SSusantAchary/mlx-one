"""CPU model-boundary tests; native checkpoint parity is a separate required gate."""
import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_one.ops.logits import LogitsMode, logits_input


@pytest.fixture
def core(monkeypatch):
    before = dict(sys.modules)
    package, mx, nn = ModuleType("mlx"), ModuleType("mlx.core"), ModuleType("mlx.nn")
    for name in ("arange", "zeros", "stack", "where", "max", "all", "any"):
        setattr(mx, name, getattr(np, name))
    mx.int32 = np.int32
    class Module(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as error:
                raise AttributeError(name) from error
        def __setattr__(self, name, value):
            self[name] = value
    nn.Module = Module
    package.core, package.nn = mx, nn
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    yield mx
    for name in list(sys.modules):
        if name.startswith("mlx_one.models.") and name not in before:
            del sys.modules[name]


FAMILIES = (
    ("language.qwen2", "Qwen2ForCausalLM", "tie_word_embeddings"),
    ("language.qwen3", "Qwen3ForCausalLM", "tie_word_embeddings"),
    ("language.lfm2", "Lfm2ForCausalLM", "tie_embedding"),
    ("language.gpt2", "GPT2LMHeadModel", None),
    ("vision_language.qwen3_5", "Qwen3_5ForConditionalGeneration", "tie_word_embeddings"),
)


@pytest.mark.parametrize("family,name,tied_flag", FAMILIES)
@pytest.mark.parametrize("length", (1, 2, 128, 512, 2048, 8192))
def test_slice_precedes_vocabulary_projection_and_preserves_hidden(core, family, name,
                                                                   tied_flag, length):
    module = importlib.import_module("mlx_one.models." + family + ".model")
    model = getattr(module, name).__new__(getattr(module, name))
    hidden = np.arange(length * 4, dtype=np.float32).reshape(1, length, 4) / 1000
    calls = []
    class Head:
        def __call__(self, value):
            return hidden
        def as_linear(self, value):
            calls.append(value.shape)
            return value @ np.arange(28, dtype=np.float32).reshape(4, 7)
    head = Head()
    class Backbone:
        embed_tokens = wte = head
        layers = [SimpleNamespace(self_attn=SimpleNamespace())]
        h = [SimpleNamespace(attn=SimpleNamespace())]
        def __call__(self, *_args, **kwargs):
            if family == "language.gpt2":
                return hidden, None, None
            return hidden, kwargs.get("cache"), None
    backbone = Backbone()
    model.config = SimpleNamespace(**({tied_flag: True} if tied_flag else {}))
    model.model = model.transformer = model.language_model = backbone
    if "vision_language" in family:
        model.config.image_token_id = 50
        model.config.video_token_id = 51
    ids = np.zeros((1, length), dtype=np.int32)
    complete = model(ids, logits_mode=LogitsMode.ALL)
    last = model(ids, logits_mode=LogitsMode.LAST_ONLY)
    unused = model(ids, logits_mode=LogitsMode.NO_LOGITS)
    assert calls == [(1, length, 4), (1, 1, 4)]
    assert last.logits.shape == (1, 1, 7)
    # NumPy may select GEMM vs GEMV here. Use the predeclared FP32 contract.
    np.testing.assert_allclose(last.logits, complete.logits[:, -1:], atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(last.logits.argmax(-1), complete.logits[:, -1:].argmax(-1))
    assert unused.logits is None
    assert complete.last_hidden_state is last.last_hidden_state is hidden
    assert unused.last_hidden_state is hidden
    assert last.logits.nbytes * length == complete.logits.nbytes


def test_last_real_position_works_with_right_and_left_padding(core):
    hidden = np.arange(40).reshape(2, 5, 4)
    mask = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]])
    actual = logits_input(hidden, LogitsMode.LAST_ONLY, attention_mask=mask)
    np.testing.assert_array_equal(actual, hidden[[0, 1], [1, 4]][:, None])
    with pytest.raises(ValueError, match="real token"):
        logits_input(hidden, "last_only", attention_mask=np.zeros((2, 5)))
