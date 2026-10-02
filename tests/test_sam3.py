"""Backend-free contracts and opt-in native checkpoint checks."""

import os
import subprocess
import sys

import pytest

from mlx_one.core.registry import get_registration
from mlx_one.inspection import _infer_modality
from mlx_one.models.segmentation.sam3.config import Sam3VideoConfig
from mlx_one.schemas import Modality


def test_sam3_registry_is_lazy():
    source = (
        "from mlx_one.segmentation import ConceptSegmentationResult; "
        "from mlx_one.models.segmentation.sam3.config import Sam3VideoConfig; "
        "import sys; assert 'mlx.core' not in sys.modules; assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", source], check=True)
    for name in ("sam3", "sam3_tracker", "sam3_tracker_video", "sam3_video"):
        entry = get_registration(name)
        assert entry.modality == "segmentation"
        assert bool(entry.loader_path) == (name != "sam3_tracker_video")
        assert not entry.supports("text-generation")


def test_composite_config_roundtrip_and_independence():
    config = Sam3VideoConfig()
    again = Sam3VideoConfig.from_dict(config.to_dict())
    assert again.detector_config.model_type == "sam3"
    assert again.tracker_config.model_type == "sam3_tracker_video"
    assert again.detector_config.vision_config.image_size == 1008
    assert config.detector_config.vision_config is not again.detector_config.vision_config


@pytest.mark.parametrize("model_type", ["sam3", "sam3_tracker", "sam3_video", "sam3_tracker_video"])
def test_segmentation_is_not_a_vlm(model_type):
    assert (
        _infer_modality(None, {"model_type": model_type, "vision_config": {}}, (), None)
        is Modality.SEGMENTATION
    )


@pytest.mark.parametrize(
    "pipeline", ["image-segmentation", "video-segmentation", "mask-generation"]
)
def test_segmentation_pipeline(pipeline):
    assert _infer_modality(pipeline, {}, (), None) is Modality.SEGMENTATION


def test_prompt_validation():
    from mlx_one.segmentation.processing import validate_coordinates, validate_labels

    assert validate_coordinates([[0, 0, 20, 10]], 20, 10, boxes=True).shape == (1, 4)
    assert validate_labels([1, 0], 2).tolist() == [1, 0]
    for invalid in ([[0, 0, 0, 10]], [[0, 0, 21, 10]], [[0, 0, float("nan"), 10]]):
        with pytest.raises(ValueError):
            validate_coordinates(invalid, 20, 10, boxes=True)
    with pytest.raises(ValueError):
        validate_labels([2], 1)


def test_native_tensor_dialect():
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; mx.eval(mx.zeros(1))"],
        capture_output=True,
    )
    if probe.returncode:
        pytest.skip("MLX Metal runtime is unavailable in this environment")
    pytest.importorskip("mlx.core", exc_type=ImportError)
    import numpy as np

    from mlx_one.models.segmentation.sam3._runtime import (
        Conv2d,
        ConvTranspose2d,
        Tensor,
        interpolate,
        ops,
    )

    value = Tensor(np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3))
    assert interpolate(value, size=(4, 6), mode="bilinear", align_corners=False).shape == (
        1,
        1,
        4,
        6,
    )
    assert np.asarray(value.flatten(2)).tolist() == [[[0, 1, 2, 3, 4, 5]]]
    assert ops.ones(2, 3).sum(1).tolist() == [3, 3]
    assert value[value > 2].tolist() == [3, 4, 5]
    assert Conv2d(1, 2, 1)(value).shape == (1, 2, 2, 3)
    assert ConvTranspose2d(1, 2, 2, stride=2)(value).shape == (1, 2, 4, 6)


@pytest.mark.skipif(
    not os.environ.get("MLX_ONE_SAM3_SNAPSHOT"), reason="explicit pinned checkpoint opt-in"
)
@pytest.mark.parametrize("task", ["concept", "interactive", "video"])
def test_real_checkpoint_weight_contract(task):
    from mlx_one.segmentation import load_segmentation_model

    bundle = load_segmentation_model(os.environ["MLX_ONE_SAM3_SNAPSHOT"], task=task, offline=True)
    assert bundle.task == task
    assert len(bundle.model.state_dict()) > 100
    assert bundle.provenance["qualified"] is False
