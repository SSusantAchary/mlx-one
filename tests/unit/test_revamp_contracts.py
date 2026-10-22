"""Foundation regressions: declarations never grant checkpoint qualification."""

import ast
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from mlx_one.core import NativeConfig, describe_architecture
from mlx_one.core.config import ConfigError, require_divisible, require_positive
from mlx_one.core.registry import get_registration
from mlx_one.engine.cache_specs import CacheKind, LayerCacheSpec, LayerStateSpec


def test_architecture_descriptor_is_lazy_and_explicitly_unqualified():
    registration = get_registration("qwen3_5")
    before = registration.model_path
    descriptor = describe_architecture(registration)
    assert descriptor.supports("text-generation")
    assert descriptor.supports("vision-language-generation")
    assert descriptor.supports("linear-attention")
    assert not descriptor.supports("qualified-batching")
    assert descriptor.qualification == "NOT_QUALIFIED"
    assert descriptor.to_dict()["model_type"] == "qwen3_5"
    assert registration.model_path == before
    assert isinstance(get_registration("gpt2").config_class()(), NativeConfig)


def test_architecture_registry_does_not_depend_on_engine_or_http_modules():
    import mlx_one.models.registry as registry

    tree = ast.parse(Path(registry.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all(not name.startswith(("mlx_one.engine", "mlx_one.server")) for name in imports
               if name)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, None, "16", 0, -1])
def test_invalid_scalar_config_values_are_rejected_before_allocation(value):
    with pytest.raises(ConfigError, match="positive"):
        require_positive("dimension", value)


def test_dimensions_used_as_divisors_must_be_integral():
    with pytest.raises(ConfigError, match="integers"):
        require_divisible("hidden", 8.0, "heads", 2)


@pytest.mark.parametrize("kind", [CacheKind.ATTENTION_KV, CacheKind.ROTATING_KV,
                                  CacheKind.CONV_STATE, CacheKind.RECURRENT_STATE])
@pytest.mark.parametrize("dtype", ["f16", "bf16", "float32", None])
def test_state_conversion_retains_declared_dtype_and_private_carry(kind, dtype):
    spec = LayerCacheSpec(2, kind, dtype=dtype, max_window_tokens=128,
                          state_shapes=((1, 3, 8),), state_dtypes=("float32",),
                          checkpoint_semantics="exact_committed_boundary")
    state = LayerStateSpec.from_cache_spec(spec)
    canonical_dtype = {"f16": "float16", "bf16": "bfloat16"}.get(dtype, dtype)
    assert state.dtype == canonical_dtype
    restored = state.to_cache_spec()
    assert restored.dtype == canonical_dtype
    for field in ("state_shapes", "state_dtypes", "checkpoint_semantics"):
        assert asdict(restored)[field] == asdict(spec)[field]
    if kind in {CacheKind.CONV_STATE, CacheKind.RECURRENT_STATE}:
        assert not state.can_page and not state.can_quantize
