"""Independent CPU fixtures for prefix diagnostic coverage and failure semantics."""
from dataclasses import replace

import numpy as np
import pytest
from test_dense_batching import native as native


@pytest.fixture
def diagnostic_model(native, monkeypatch):
    import mlx.core as mx

    for name in ("abs", "max", "mean", "all", "allclose"):
        monkeypatch.setattr(mx, name, getattr(np, name), raising=False)
    mx.float32 = np.float32
    return native


def test_all_four_paths_use_identical_forced_ids_and_real_boundaries(diagnostic_model):
    from mlx_one.engine.parity import prefix_failure_diagnostic

    captured = {}
    result = prefix_failure_diagnostic(diagnostic_model, prompt="abcde" * 2, count=3,
        sink=lambda name, value: captured.update({name: value.copy()}))
    assert result["status"] == "passed" and not result["failure_reproduced"]
    assert result["atol"] == result["rtol"] == .01
    assert len(captured) == 2 * 4 * 4
    for case in result["cases"]:
        assert 0 < case["reused_tokens"] < len(case["prompt_ids"])
        assert case["executed_suffix_ids"] == case["prompt_ids"][case["reused_tokens"]:]
        assert [b["forced_input_id"] for b in case["boundaries"][1:]] == result["forced_token_ids"]
        for boundary in case["boundaries"]:
            assert boundary["committed_positions_aligned"]
            assert set(boundary["components"]) == {
                "fresh_full", "ordinary_cached", "independent_restored", "prefix_index"}
            for components in boundary["components"].values():
                assert all(c["actual_dtypes"] == ("float16",) for c in components)
    assert result["cleanup_passed"] and result["cleanup"]["isolated_snapshots_disposed"]
    assert result["qualification_promoted"] is False


def test_shape_dependent_failure_remains_failed_qualification(diagnostic_model, monkeypatch):
    from mlx_one.engine.parity import prefix_failure_diagnostic
    from mlx_one.runtime_baseline import diagnostic_outcome

    model_type = type(diagnostic_model.model)
    original = model_type.__call__
    def shape_dependent(self, ids, **kwargs):
        output = original(self, ids, **kwargs)
        return (replace(output, logits=output.logits + np.float16(.2))
                if ids.shape[1] > 4 else output)
    monkeypatch.setattr(model_type, "__call__", shape_dependent)
    result = prefix_failure_diagnostic(diagnostic_model, prompt="abcde" * 2, count=3)
    assert result["status"] == "failed" and result["failure_reproduced"]
    outcome = diagnostic_outcome({"status": "completed"}, result)
    assert outcome["readiness"] == "reproduced" and outcome["qualification"] == "failed"
    assert result["cleanup_passed"] and not result["qualification_promoted"]


@pytest.mark.parametrize("failure", ["forward", "observer"])
def test_failed_execution_or_capture_releases_owned_state(diagnostic_model, monkeypatch, failure):
    from mlx_one.engine import cache_manager as cache_module
    from mlx_one.engine.parity import prefix_failure_diagnostic

    managers = []
    original = cache_module.CacheManager
    class ObservedManager(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            managers.append(self)
    monkeypatch.setattr(cache_module, "CacheManager", ObservedManager)
    diagnostic_model.model.fail = failure == "forward"
    def failed_capture(*args):
        raise RuntimeError("injected observer failure")
    with pytest.raises(RuntimeError, match="injected"):
        prefix_failure_diagnostic(diagnostic_model, prompt="abcde" * 2, count=3,
                                  sink=failed_capture if failure == "observer" else None)
    assert len(managers) == 2
    for manager in managers:
        stats = manager.stats()
        assert manager._closed
        assert stats.active_handles == stats.reserved_bytes == stats.prefix_entries == 0


def test_degenerate_prompt_is_rejected_before_cache_allocation(diagnostic_model):
    from mlx_one.engine.parity import prefix_failure_diagnostic

    with pytest.raises(ValueError, match="reusable prompt"):
        prefix_failure_diagnostic(diagnostic_model, prompt="", count=3)
    with pytest.raises(ValueError, match="positive"):
        prefix_failure_diagnostic(diagnostic_model, prompt="abcde", count=0)
