"""Cache mathematics identities must reject state from the superseded decoders."""

import json

import pytest
from test_cache_runtime import bundle
from test_compound_state import component, spec
from test_dense_batching import native as native
from test_paged_hybrid import advanced as advanced

from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.components import load_snapshot, save_snapshot


def test_reference_port_invalidates_stored_weight_prefix_fingerprint(tmp_path):
    loaded = bundle(tmp_path)
    runtime = CacheManager(loaded)
    legacy = runtime.fingerprint(namespace="same-user")
    loaded.model._mlx_one_reference_first_math = True
    current = runtime.fingerprint(namespace="same-user")
    assert current.model_identity == legacy.model_identity
    assert current.execution_identity != legacy.execution_identity
    loaded.model._mlx_one_reference_first_math = False
    assert runtime.fingerprint(namespace="same-user") == legacy
    runtime.close()


def test_previous_delta_transition_snapshot_is_rejected(advanced, tmp_path):
    advanced()
    state = CacheBundle((component(),))
    try:
        save_snapshot(tmp_path, state, identity={"fixture": "transition"})
        path = tmp_path / "state.json"
        document = json.loads(path.read_text())
        document["layers"][0]["compound"]["version"] = "qwen35-gated-delta-v1"
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="transition identity"):
            load_snapshot(tmp_path, identity={"fixture": "transition"}, expected_specs=(spec(),))
    finally:
        state.release()
