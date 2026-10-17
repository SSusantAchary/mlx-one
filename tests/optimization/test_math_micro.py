from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from qualification.p01_attention_micro import attention_bound, attention_layouts
from qualification.p01_external import comparator_inventory
from qualification.p01_micro import projection_bound, projection_shapes


def test_shape_inventory_covers_distinct_packed_layouts_and_preserves_all_names():
    def module(bits, outputs):
        return SimpleNamespace(bits=bits, group_size=64,
            weight=SimpleNamespace(shape=(outputs, 128 // (32 // bits)), dtype="uint32"),
            scales=SimpleNamespace(dtype="float16"))
    gate, up, head = module(4, 32), module(4, 32), module(8, 128)
    class Model:
        def named_modules(self):
            return [("gate", gate), ("up", up), ("head", head), ("norm", object())]
    groups = projection_shapes(Model())
    assert len(groups) == 2
    assert groups[0]["module"] is gate and groups[0]["names"] == ["gate", "up"]
    assert groups[1]["module"] is head and groups[1]["bits"] == 8
    assert all(group["width"] == 128 for group in groups)


@pytest.mark.parametrize("shape", ((True, 128, 128), (0, 128, 128), (1, -1, 128)))
def test_invalid_microbenchmark_dimensions_cannot_bypass_memory_guard(shape):
    with pytest.raises(ValueError):
        projection_bound(*shape)


def test_microbenchmark_forecast_includes_both_numerical_results_and_cast_overlap():
    assert projection_bound(8192, 128, 151936) >= 8192 * 151936 * 16


def test_incompatible_competitor_environment_remains_blocked_without_importing_engines():
    packages = {"mlx": "0.32.2", "mlx-lm": "0.31.3", "numpy": "2.5.2"}
    def get_version(name):
        if name not in packages:
            raise PackageNotFoundError(name)
        return packages[name]
    report = comparator_inventory(get_version)
    assert report["mlx_lm"] == "AVAILABLE" and report["omlx"] == "BLOCKED"
    assert len(report["omlx_blockers"]) == 3 and report["qualification_promoted"] is False


def test_attention_inventory_keeps_real_gqa_and_explicit_qwen35_dimensions():
    qwen = SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object(),
                           num_heads=16, num_kv_heads=8, head_dim=128)
    hybrid = SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object(),
        config=SimpleNamespace(num_attention_heads=8, num_key_value_heads=2, head_dim=256))
    model = SimpleNamespace(named_modules=lambda: [
        ("qwen0", qwen), ("qwen1", qwen), ("conv", object()), ("hybrid", hybrid)])
    groups = attention_layouts(model)
    assert groups == [dict(heads=16, kv_heads=8, head_dim=128, modules=["qwen0", "qwen1"]),
                      dict(heads=8, kv_heads=2, head_dim=256, modules=["hybrid"])]
    assert attention_bound(groups[0], 8192, 8192) > 16 * 8192**2 * 16


@pytest.mark.parametrize("query,key", ((0, 1), (2, 1), (True, 2)))
def test_attention_probe_cannot_bypass_positive_causal_memory_bounds(query, key):
    with pytest.raises(ValueError):
        attention_bound(dict(heads=8, kv_heads=2, head_dim=64), query, key)


def test_attention_inventory_rejects_missing_explicit_heads():
    module = SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object())
    with pytest.raises(ValueError, match="explicit"):
        attention_layouts(SimpleNamespace(named_modules=lambda: [("attention", module)]))
