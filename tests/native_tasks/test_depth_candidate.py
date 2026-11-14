"""Native depth candidate output/interpolation checks; full trace gate separate."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "local/native_tasks/checkpoints/depth-anything"
RUN = ROOT / "local/native_tasks/runs/m5-20261103"


@pytest.mark.skipif(
    not (RUN / "interpolation-reference.npz").exists(), reason="source probes absent"
)
def test_depth_interpolation_matches_pinned_reference_probes():
    import mlx.core as mx

    from mlx_one.models.depth.interpolation import resize

    with np.load(RUN / "interpolation-reference.npz", allow_pickle=False) as ref:
        for key in ref.files:
            if not key.endswith("_input"):
                continue
            name = key.removesuffix("_input")
            parts = name.split("-")
            target = tuple(map(int, parts[1].split("x")))
            mode = parts[2]
            align = parts[3] == "True"
            result = resize(
                mx.array(ref[key].transpose(0, 2, 3, 1)), target, mode=mode, align_corners=align
            )
            assert np.allclose(
                np.asarray(result).transpose(0, 3, 1, 2),
                ref[name + "_output"],
                atol=1e-4,
                rtol=1e-4,
            ), name


@pytest.mark.skipif(
    not (RUN / "depth-cat-reference.npz").exists(), reason="real source depth trace absent"
)
def test_depth_real_candidate_final_map_matches_reference_and_is_relative():
    import mlx.core as mx

    from mlx_one.models.depth.depth_anything import load_depth
    from mlx_one.models.depth.interpolation import resize
    from mlx_one.vision.depth_processing import preprocess_depth

    image = ROOT / "local/native_tasks/runs/m2-20261103/clip/real-images/chelsea.png"
    pixels, size = preprocess_depth(image)
    model = load_depth(CKPT)
    result = model(mx.array(pixels[None]))
    depth = result["predicted_depth"]
    resized = resize(depth[..., None], (size[1], size[0]), mode="bicubic")[0, :, :, 0]
    mx.eval(depth, resized)
    with np.load(RUN / "depth-cat-reference.npz", allow_pickle=False) as ref:
        assert np.array_equal(pixels[None].transpose(0, 3, 1, 2), ref["pixels"])
        assert np.allclose(np.asarray(depth), ref["predicted_depth"], atol=1e-4, rtol=1e-4)
        assert np.allclose(np.asarray(resized), ref["resized_depth"], atol=1e-4, rtol=1e-4)
    assert resized.shape == (size[1], size[0]) and np.isfinite(np.asarray(resized)).all()
    assert float(mx.min(depth).item()) >= 0
    with pytest.raises(ValueError):
        model(mx.zeros((2, 518, 518, 3)))


@pytest.mark.skipif(not CKPT.exists(), reason="pinned depth checkpoint absent")
def test_depth_loader_rejects_changed_assets(tmp_path):
    from mlx_one.models.depth.depth_anything import load_depth

    (tmp_path / "model.safetensors").symlink_to(CKPT / "model.safetensors")
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="asset mismatch"):
        load_depth(tmp_path)
