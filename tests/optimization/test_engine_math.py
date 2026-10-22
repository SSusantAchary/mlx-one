import numpy as np
import pytest
from test_cache_runtime import Model, Tokenizer, install_fake_mlx

from mlx_one.core.outputs import ModelOutput
from mlx_one.engine import public
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.optimization_policy import NativeMathPolicy, current_math
from mlx_one.text.loading import LoadedTextModel


@pytest.fixture
def setup(monkeypatch, tmp_path):
    import sys

    from mlx_one.engine.pressure import PressureTelemetry

    install_fake_mlx(monkeypatch)
    mx = sys.modules["mlx.core"]
    mx.eval, mx.argmax = lambda *_values: None, np.argmax
    monkeypatch.setattr(public, "configure_execution_dtype", lambda bundle, dtype:
                        setattr(bundle.model, "_mlx_one_execution_policy", {"dtype": dtype}))
    monkeypatch.setattr("mlx_one.engine.pressure.sample_telemetry", lambda: PressureTelemetry(
        installed_bytes=1024**3, recommended_bytes=512 * 1024**2, rss_bytes=1024**2))
    class Forward(Model):
        def __init__(self):
            super().__init__()
            self.config.max_position_embeddings = 10000
            self.config.vocab_size = 16
            self.calls = []
            self.fail_at = None
            self.cancel = None
        def __call__(self, ids, *, cache, logits_mode=None):
            self.calls.append((ids.shape[1], str(logits_mode), current_math().policy.logits))
            value = np.ones((1, 2, ids.shape[1], 4), dtype=np.float16)
            cache[0].update(value, value)
            if self.fail_at == len(self.calls):
                raise MemoryError("injected evaluated-forward failure")
            if self.cancel is not None:
                self.cancel.set()
            mode = getattr(logits_mode, "value", logits_mode)
            rows = 1 if mode == "last_only" else ids.shape[1]
            logits = np.zeros((1, rows, 16), dtype=np.float16)
            logits[:, :, 3] = 1
            return ModelOutput(None if mode == "no_logits" else logits,
                               np.zeros((1, ids.shape[1], 16), dtype=np.float16), cache=cache)
    model = Forward()
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = 999
    bundle = LoadedTextModel(model, tokenizer, tmp_path, "fixture", "a", architecture="qwen2",
                             context_length=10000, tokenizer_identity="fixture-tokenizer")
    def engine(**options):
        return public.Engine(bundle, execution_dtype="float16",
            cache_config=CacheConfig(prefix_cache_enabled=False, memory_budget_bytes=64 * 1024**2),
            **options)
    return model, engine


def test_opt_in_b1_chunks_skip_only_unconsumed_heads(setup):
    model, factory = setup
    policy = NativeMathPolicy(logits="last_only", prefill_chunk_size=256)
    with factory(inference_mode="native_optimized", math_policy=policy) as engine:
        result = engine.generate("x" * 600, max_tokens=3)
        assert result.generation_tokens == 3
        assert [call[0] for call in model.calls] == [256, 256, 88, 1, 1]
        assert [call[1].split(".")[-1] for call in model.calls] == [
            "NO_LOGITS", "NO_LOGITS", "LAST_ONLY", "LAST_ONLY", "LAST_ONLY"]
        assert all(call[2] == "last_only" for call in model.calls)
        stats = engine._cache_manager.stats()
        assert stats.active_handles == stats.reserved_bytes == stats.allocated_bytes == 0
    assert current_math().policy.logits == "all"


def test_explicit_reference_chunking_does_not_require_actor(setup):
    model, factory = setup
    with factory(inference_mode="reference_dense",
                 math_policy=NativeMathPolicy(prefill_chunk_size=256)) as engine:
        assert engine._actor is None
        engine.generate("x" * 600, max_tokens=1)
    assert [call[0] for call in model.calls] == [256, 256, 88]
    assert all(call[1] == "None" and call[2] == "all" for call in model.calls)


def test_forward_failure_cleans_owned_state_and_next_request_succeeds(setup):
    model, factory = setup
    with factory(inference_mode="native_optimized",
                 math_policy=NativeMathPolicy(
                     logits="last_only", prefill_chunk_size=256)) as engine:
        model.fail_at = 2
        with pytest.raises(MemoryError, match="injected"):
            engine.generate("x" * 600, max_tokens=3)
        stats = engine._cache_manager.stats()
        assert stats.active_handles == stats.reserved_bytes == stats.allocated_bytes == 0
        assert stats.prefix_entries == 0
        model.fail_at = None
        assert engine.generate("short", max_tokens=2).generation_tokens == 2


def test_cancellation_after_chunk_evaluation_releases_state(setup):
    import threading

    model, factory = setup
    with factory(inference_mode="native_optimized",
                 math_policy=NativeMathPolicy(
                     logits="last_only", prefill_chunk_size=256)) as engine:
        cancelled = threading.Event()
        model.cancel = cancelled
        assert not list(engine.stream("x" * 600, max_tokens=3, cancel=cancelled))
        assert len(model.calls) == 1
        stats = engine._cache_manager.stats()
        assert stats.active_handles == stats.reserved_bytes == stats.allocated_bytes == 0


def test_two_engines_sharing_model_keep_reference_and_optimized_policies_private(setup):
    model, factory = setup
    with factory(inference_mode="native_optimized") as optimized, factory() as reference:
        a, b = optimized.submit("hello", max_tokens=2), reference.submit("world", max_tokens=2)
        assert a.result(timeout=5).text == b.result(timeout=5).text
    assert {call[2] for call in model.calls} == {"all", "last_only"}
    assert current_math().policy.logits == "all"


def test_component_diagnostics_do_not_promote_failed_or_unmeasured_math(setup):
    _, factory = setup
    with factory(inference_mode="native_optimized") as engine:
        components = engine.runtime_stats()["native_math"]["components"]
        assert components["last_logits"]["availability"] == "experimental"
        assert components["fused_attention"]["availability"] == "unavailable"
        assert components["packed_matrix"]["availability"] == "unavailable"
        assert all(value["qualification"] == "unqualified" for value in components.values())


def test_incompatible_last_logits_result_fails_closed_and_releases_ownership(setup, monkeypatch):
    model, factory = setup
    original = type(model).__call__
    def ignores_contract(self, ids, **settings):
        output = original(self, ids, **settings)
        output.logits = np.zeros((1, ids.shape[1], 16), dtype=np.float16)
        return output
    monkeypatch.setattr(type(model), "__call__", ignores_contract)
    with factory(inference_mode="native_optimized") as engine:
        with pytest.raises(ValueError, match="logits"):
            engine.generate("abcdef", max_tokens=1)
        stats = engine._cache_manager.stats()
        assert stats.active_handles == stats.reserved_bytes == stats.allocated_bytes == 0
        monkeypatch.setattr(type(model), "__call__", original)
        assert engine.generate("valid", max_tokens=1).generation_tokens == 1


@pytest.mark.parametrize("settings", ({"execution_dtype": None}, {"spec_type": "draft-mtp"},
    {"cache_config": CacheConfig()}, {"math_policy": NativeMathPolicy(enable_compile=True)}))
def test_ineligible_policy_fails_before_loading(monkeypatch, settings):
    def load(*_args, **_kwargs):
        raise AssertionError("ineligible policy loaded a checkpoint")
    monkeypatch.setattr(public, "load_text_model", load)
    options = {"inference_mode": "native_optimized", "execution_dtype": "float16", **settings}
    with pytest.raises(ValueError):
        public.Engine("unused", **options)


def test_later_shared_bundle_precision_change_rejects_without_new_forward(setup):
    model, factory = setup
    with factory(inference_mode="native_optimized") as engine:
        engine.generate("first", max_tokens=1)
        count = len(model.calls)
        model._mlx_one_execution_policy = {"dtype": "bfloat16"}
        with pytest.raises(ValueError, match="precision changed"):
            engine.generate("later", max_tokens=1)
        assert len(model.calls) == count
        assert engine._cache_manager.stats().active_handles == 0
        assert engine._cache_manager.stats().reserved_bytes == 0


@pytest.mark.parametrize("mode", (None, "reference_dense"))
def test_reference_mode_cannot_be_redefined_by_operator_policy(monkeypatch, mode):
    monkeypatch.setattr(public, "load_text_model", lambda *_args, **_kwargs:
                        pytest.fail("invalid reference policy loaded weights"))
    with pytest.raises(ValueError, match="requires explicit"):
        public.Engine("unused", inference_mode=mode, execution_dtype="float16",
                      math_policy=NativeMathPolicy(logits="last_only"))


@pytest.mark.parametrize("policy", (NativeMathPolicy(attention="tiled_256"),
    NativeMathPolicy(projection="native"), NativeMathPolicy(hybrid="fast")))
def test_unimplemented_specialization_is_not_silently_substituted(monkeypatch, policy):
    monkeypatch.setattr(public, "load_text_model", lambda *_args, **_kwargs:
                        pytest.fail("unavailable policy loaded weights"))
    with pytest.raises(ValueError, match="not implemented"):
        public.Engine("unused", inference_mode="native_optimized", execution_dtype="float16",
                      math_policy=policy)
