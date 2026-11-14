"""Raw DETR candidate checks; this does not admit blocked detector task APIs."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "local/native_tasks/checkpoints/detr"
RUN = ROOT / "local/native_tasks/runs/m3-20261103"
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="pinned DETR checkpoint absent")


def test_detr_real_cat_trace_matches_isolated_reference():
    import mlx.core as mx

    from mlx_one.models.detection.detr import load_detr
    from mlx_one.vision.detr_processing import postprocess_detr, preprocess_detr

    golden = RUN / "detr-cat-reference.npz"
    if not golden.exists():
        pytest.skip("isolated real-cat trace absent")
    model = load_detr(CKPT)
    image = ROOT / "local/native_tasks/runs/m2-20261103/clip/real-images/chelsea.png"
    pixels, mask, size = preprocess_detr(image)
    actual = model(mx.array(pixels[None]), mx.array(mask[None]))
    mx.eval(actual)
    with np.load(golden, allow_pickle=False) as ref:
        assert np.array_equal(pixels.transpose(2, 0, 1)[None], ref["pixels"])
        for k, value in actual.items():
            assert np.allclose(np.asarray(value), ref[k], atol=1e-4, rtol=1e-4), k
        post = postprocess_detr(
            np.asarray(actual["logits"]), np.asarray(actual["pred_boxes"]), size
        )
        for k, v in post.items():
            assert np.allclose(v, ref["post_" + k], atol=1e-4, rtol=1e-4)
    assert postprocess_detr(np.zeros((1, 100, 92)), np.zeros((1, 100, 4)), size, 1)[
        "boxes"
    ].shape == (0, 4)


def test_detr_strict_hash_rejects_changed_config(tmp_path):
    from mlx_one.models.detection.detr import load_detr

    (tmp_path / "model.safetensors").symlink_to(CKPT / "model.safetensors")
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_detr(tmp_path)
