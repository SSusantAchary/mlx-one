from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from mlx_one.engine.modes import resolve_backend, resolved_mode
from mlx_one.mlxboost import MLXBoostConfig
from mlx_one.server.config import ServerConfig

FACTS = dict(
    family="qwen2",
    dtype="float16",
    quantization={"bits": 8, "group_size": 64},
    backend="dense",
    slots=1,
    cache_reuse=False,
    mlx_version="0.32.2",
    metal_available=True,
)


def test_default_engine_remains_reference():
    assert resolved_mode(resolve_backend(None, None)) == "reference_dense"
    assert not MLXBoostConfig().enabled
    assert ServerConfig().inference_mode is None
    assert not ServerConfig().mlxboost_config.enabled
    with pytest.raises(FrozenInstanceError):
        MLXBoostConfig().fast_decode = True


@pytest.mark.parametrize(
    "changes",
    [
        dict(family="unknown"),
        dict(dtype="float32"),
        dict(quantization={"bits": 4, "group_size": 64}),
        dict(slots=2),
        dict(cache_reuse=True),
        dict(metal_available=False),
        dict(mlx_version="0.32.4"),
        dict(backend="paged"),
    ],
)
def test_dispatch_eligible_model_precision_shape_and_fallback(changes):
    config = MLXBoostConfig(fast_decode=True)
    resolved, reason = config.resolve(**(FACTS | changes))
    assert not resolved.enabled and reason
    with pytest.raises(ValueError, match="strict_reference"):
        replace(config, strict_reference=True).resolve(**(FACTS | changes))


def test_eligible_flag_off_on_independent_and_optional_kernel_fallback():
    for enabled in [False, True]:
        config = MLXBoostConfig(fast_decode=enabled)
        assert config.resolve(**FACTS) == (config, None)
    config, reason = MLXBoostConfig(experimental_qgemv=True).resolve(**FACTS)
    assert not config.enabled and "unimplemented" in reason
    assert not MLXBoostConfig().experimental_qgemv


@pytest.mark.parametrize(
    "changes",
    [
        {"family": "qwen3", "dtype": "bfloat16"},
        {"family": "llama", "dtype": "bfloat16", "quantization": {}},
    ],
)
def test_bf16_diagnostic_dispatch(changes):
    config = MLXBoostConfig(fast_decode=True)
    assert config.resolve(**(FACTS | changes)) == (config, None)


def test_explicit_mode_and_b1_are_required():
    with pytest.raises(ValueError, match="explicit"):
        ServerConfig(mlxboost_config=MLXBoostConfig(fast_decode=True))
    with pytest.raises(ValueError, match="parallel=1"):
        ServerConfig(inference_mode="mlxboost_b1", parallel=2)
    with pytest.raises(TypeError, match="boolean"):
        MLXBoostConfig(fast_decode=1)


def test_fast_decode_skips_only_impossible_calibration_lookup(monkeypatch):
    from mlx_one.engine import reporting, selection
    from mlx_one.engine.pressure import execution_forecast

    calls = []
    monkeypatch.setattr(selection, "source_identity", lambda: calls.append("source") or "sha")
    monkeypatch.setattr(selection, "host_identity", lambda: "host")
    monkeypatch.setattr(reporting, "native_report", lambda *a, **kw: {"scope": {}})
    records = []
    runtime = SimpleNamespace(
        qualification_registry=SimpleNamespace(records=lambda: tuple(records)),
        mode="mlxboost_b1",
        bundle=object(),
        config=SimpleNamespace(key_type="f16"),
        mlxboost_config=MLXBoostConfig(fast_decode=True),
    )
    model = SimpleNamespace(
        hidden_size=896,
        intermediate_size=4864,
        num_attention_heads=14,
        num_key_value_heads=2,
        num_hidden_layers=24,
        vocab_size=151936,
    )

    def forecast():
        return execution_forecast(runtime, model, batch=1, chunk=1, history=128)

    enabled = forecast()
    assert not calls
    records.append(SimpleNamespace(feature="unrelated"))
    assert forecast() == enabled and not calls
    # A newly registered activation candidate must immediately restore the full
    # scope validation; no stale negative cache is allowed.
    records[:] = [SimpleNamespace(feature="activation_peak", scope={"mismatch": True})]
    monkeypatch.setattr("dataclasses.asdict", lambda x: x)
    assert forecast() == enabled and calls == ["source"]
    records.clear()
    runtime.mlxboost_config = MLXBoostConfig()
    assert forecast() == enabled and calls == ["source", "source"]
    runtime.mlxboost_config = MLXBoostConfig(fast_decode=True)
    runtime.mode = "reference_dense"
    assert forecast() == enabled and calls == ["source", "source", "source"]
