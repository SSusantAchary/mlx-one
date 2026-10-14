"""Opt-in native P1 checkpoints and existing Metal specializations, offline only."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

if os.environ.get("MLX_ONE_RUN_P1_CHECKPOINTS") != "1":
    pytest.skip("set MLX_ONE_RUN_P1_CHECKPOINTS=1 on a Metal host", allow_module_level=True)

from mlx_one.engine.adapters import native_adapter
from mlx_one.engine.modes import configure_execution_dtype
from mlx_one.engine.qualification import execution_parity
from mlx_one.p1_profiles import CHECKPOINTS, profile
from mlx_one.text import load_text_model


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("page", [16, 32])
@pytest.mark.parametrize("paged", [False, True])
def test_native_p1_serial_dense_paged_full_state(checkpoint, size, page, paged):
    asset = CHECKPOINTS[checkpoint]
    path = os.environ.get("MLX_ONE_P1_" + checkpoint.upper().replace("-", "_") + "_PATH")
    if asset.conversion_required and not path:
        pytest.skip("missing verified local SmolLM2 INT8 conversion: " + checkpoint)
    mode = "paged" if paged else ("experimental_hybrid_batch" if asset.architecture == "lfm2"
                                 else "experimental_dense_batch")
    item = profile(checkpoint, mode=mode, checkpoint_path=path)
    bundle = load_text_model(path or asset.model_id, revision=item.revision, offline=True)
    configure_execution_dtype(bundle, "float16")
    adapter = native_adapter(bundle, allow_hybrid=asset.architecture == "lfm2")
    if asset.architecture == "qwen3":
        assert adapter.spec.head_dim == 128 and adapter.spec.qk_normalization
        attention = bundle.model.model.layers[0].self_attn
        assert attention.q_norm is not None and attention.k_norm is not None
    result = execution_parity(bundle, mode=mode, batch_size=size, count=32,
        page_tokens=page, prompt="Native INT8 shared prefix and divergent suffixes. " * 32)
    output = os.environ.get("MLX_ONE_P1_RESULTS_DIR")
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"parity-{checkpoint}-{paged}-{page}-{size}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
    assert result["status"] == "passed", result
    assert result["prefix_isolation_passed"] and result["seeded_rng_isolation"]
    assert result["cleanup"]["active_handles"] == result["cleanup"]["reserved_bytes"] == 0


def test_pr00_prefix_diagnostic_supervised():
    """A diagnostic pass reproduces failed parity; it never qualifies execution."""
    command = [sys.executable, "benchmarks/p1_native_qualification.py", "prefix-diagnostic"]
    process = subprocess.run(command, text=True, capture_output=True, timeout=700,
                             env={**os.environ, "HF_HUB_OFFLINE": "1"})
    assert process.returncode == 0, process.stdout + process.stderr
    directory = Path(process.stdout.strip().splitlines()[-1])
    report = json.loads((directory / "diagnostic.json").read_text())
    assert report["readiness"] == "reproduced" and report["qualification"] == "failed"
    assert report["qualification_promoted"] is False
