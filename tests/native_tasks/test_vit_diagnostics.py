"""Regression coverage for ViT diagnostic trace alignment and operator replay."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "local/native_tasks/runs/vit-investigation-final-20261102"
CHECKPOINT = ROOT / "local/native_tasks/checkpoints/vit"


def test_diagnostic_metric_does_not_hide_shape_or_tolerance_errors():
    spec = spec_from_file_location(
        "vit_parity_report", ROOT / "benchmarks/native_tasks/vit_parity_report.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    ref = np.array([1.0, 0.0], dtype=np.float32)
    exact = module.metrics(ref, ref, 1e-4, 1e-4)
    assert exact["passed"] and exact["max_normalized_error"] == 0
    assert not module.metrics(ref, ref + 0.01, 1e-4, 1e-4)["passed"]
    with pytest.raises(ValueError, match="shape mismatch"):
        module.metrics(ref, ref[:1], 1e-4, 1e-4)
    with pytest.raises(ValueError, match="nonfinite"):
        module.metrics(ref, np.array([np.nan, 0.0]), 1e-4, 1e-4)


@pytest.mark.skipif(not (RUN / "hf.npz").exists(), reason="local reference evidence unavailable")
def test_reference_trace_residual_boundaries_and_operator_replay():
    with np.load(RUN / "hf.npz", allow_pickle=False) as ref:
        with np.load(RUN / "operator-replay.npz", allow_pickle=False) as replay:
            assert len(replay.files) == 26
            for key in replay.files:
                assert ref[key].shape == replay[key].shape
                assert np.allclose(ref[key], replay[key], atol=1e-4, rtol=1e-4)
            for i in (4, 5):
                prefix = f"block_{i:02}_"
                np.testing.assert_array_equal(
                    ref[prefix + "residual1"],
                    ref[prefix + "input"] + ref[prefix + "attention_output"],
                )
                np.testing.assert_array_equal(ref[prefix + "residual2"], ref[f"layer_{i:02}"])
                np.testing.assert_array_equal(ref[prefix + "input"], ref[f"layer_{i - 1:02}"])


@pytest.mark.skipif(
    not CHECKPOINT.exists() or not RUN.exists(), reason="local evidence unavailable"
)
def test_manual_native_trace_matches_production_forward():
    import mlx.core as mx

    from mlx_one.vision.vit_classification import load_vit_classifier

    model = load_vit_classifier(CHECKPOINT).model
    pixels = np.load(RUN / "pixels.npy", allow_pickle=False)
    logits = model(mx.array(pixels.transpose(0, 2, 3, 1)))
    mx.eval(logits)
    with np.load(RUN / "mlx-default.npz", allow_pickle=False) as trace:
        np.testing.assert_array_equal(np.asarray(logits), trace["logits"])
        np.testing.assert_array_equal(pixels, trace["pixels"])
