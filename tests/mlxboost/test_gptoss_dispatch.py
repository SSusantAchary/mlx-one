"""GPT-OSS opt-in state and fallback must remain safe without loading Metal."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mlx_one.mlxboost import gptoss


def test_import_and_reference_execution_do_not_initialize_mlx():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from mlx_one.mlxboost.gptoss import try_moe; "
                "assert try_moe(None, None, None, None) is None; "
                "assert 'mlx.core' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_nested_context_restores_request_state_after_exception():
    assert gptoss._context.get() is None
    with gptoss.gptoss_execution("compiled_swiglu") as outer:
        with pytest.raises(RuntimeError, match="test interruption"):
            with gptoss.gptoss_execution("metal_swiglu") as inner:
                assert gptoss._context.get() is inner
                raise RuntimeError("test interruption")
        assert gptoss._context.get() is outer
    assert gptoss._context.get() is None


def test_concurrent_requests_keep_independent_features_and_dispatch_counts():
    async def request(feature, count):
        with gptoss.gptoss_execution(feature) as state:
            state.dispatch["request"] = count
            await asyncio.sleep(0)
            assert gptoss._context.get() is state
            assert state.features == frozenset({feature})
            assert state.dispatch["request"] == count
            return state

    async def run():
        return await asyncio.gather(request("metal_swiglu", 3), request("compiled_swiglu", 7))

    left, right = asyncio.run(run())
    assert left is not right
    assert gptoss._context.get() is None


@pytest.mark.parametrize(
    "features,settings",
    [
        ("unknown", {}),
        ("compiled_moe,metal_swiglu", {}),
        ("metal_swiglu", {"words_per_lane": 2}),
        ("metal_swiglu", {"rows_per_group": 3}),
        ("prefill_sort,prefill_grouped", {}),
        ("mlx_lm_moe,prefill_sort", {}),
        ("mlx_lm_moe,compiled_moe", {}),
    ],
)
def test_invalid_experiments_are_rejected_before_dispatch(features, settings):
    with pytest.raises(ValueError):
        with gptoss.gptoss_execution(features, **settings):
            pytest.fail("Invalid configuration entered execution")
    assert gptoss._context.get() is None


def test_unavailable_candidate_falls_back_once_per_request(monkeypatch):
    calls = []

    def unavailable(*args):
        calls.append(1)
        raise RuntimeError("Compiler unavailable")

    monkeypatch.setattr(gptoss, "moe_forward", unavailable)
    experts = SimpleNamespace(
        swiglu_limit=7, gate_proj=SimpleNamespace(weight=SimpleNamespace(shape=(32, 2880, 360)))
    )
    x = SimpleNamespace(shape=(1, 1, 2880), dtype="bfloat16")
    indices = SimpleNamespace(shape=(1, 1, 4))
    for _ in range(2):
        with gptoss.gptoss_execution("fused_gate_up") as state:
            assert gptoss.try_moe(experts, x, indices, None) is None
            assert gptoss.try_moe(experts, x, indices, None) is None
            assert state.dispatch["stock: candidate compile/runtime failure"] == 1
            assert state.dispatch["stock: previously unavailable candidate"] == 1
    assert len(calls) == 2


def test_prefill_shape_uses_stock_without_attempting_candidate(monkeypatch):
    def forbidden(*args):
        pytest.fail("Prefill should remain on the reference path")

    monkeypatch.setattr(gptoss, "moe_forward", forbidden)
    with gptoss.gptoss_execution("fused_gate_up") as state:
        assert gptoss.try_moe(None, SimpleNamespace(shape=(1, 512, 2880)), None, None) is None
        assert state.dispatch["stock: unsupported shape or architecture"] == 1


def test_attention_experiment_keeps_moe_and_ineligible_attention_on_stock():
    with gptoss.gptoss_execution("gqa_sinks") as state:
        assert gptoss.try_moe(None, None, None, None) is None
        assert gptoss.try_attention(None, None, None, None, full_causal_single_query=False) is None
        assert state.dispatch["attention stock: unsupported attention scope"] == 1
    with gptoss.gptoss_execution("gqa_sinks,compiled_moe") as state:
        assert state.features == frozenset({"gqa_sinks", "compiled_moe"})


def test_prefill_feature_does_not_change_decode_or_other_architectures():
    with gptoss.gptoss_execution("prefill_sort") as state:
        assert gptoss.try_moe(None, SimpleNamespace(shape=(1, 1, 2880)), None, None) is None
        assert gptoss.try_moe(None, SimpleNamespace(ndim=3, shape=(1, 512, 64)), None, None) is None
        assert state.dispatch["prefill stock: unsupported shape or architecture"] == 1


@pytest.mark.parametrize(
    "module,operation,feature,prefix",
    [
        ("gptoss_prefill", "prefill_moe", "prefill_grouped", "prefill stock: "),
        ("gptoss_mlx_lm", "mlx_lm_moe", "mlx_lm_moe", "mlx_lm stock: "),
    ],
)
def test_prefill_runtime_failure_is_recorded_and_falls_back(
    monkeypatch, module, operation, feature, prefix
):
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("Grouped kernel unavailable")

    monkeypatch.setitem(
        sys.modules, f"mlx_one.mlxboost.{module}", SimpleNamespace(**{operation: unavailable})
    )
    x = SimpleNamespace(ndim=3, shape=(1, 512, 2880), dtype="bfloat16")
    indices = SimpleNamespace(shape=(1, 512, 4))
    experts = SimpleNamespace(
        gate_proj=SimpleNamespace(weight=SimpleNamespace(shape=(32, 2880, 360)))
    )
    with gptoss.gptoss_execution(feature) as state:
        assert gptoss.try_moe(experts, x, indices, None) is None
        assert gptoss.try_moe(experts, x, indices, None) is None
        assert state.dispatch[prefix + "candidate compile/runtime failure"] == 1
        assert state.dispatch[prefix + "previously unavailable candidate"] == 1
    assert len(calls) == 1
