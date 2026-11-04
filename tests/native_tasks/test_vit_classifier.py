from pathlib import Path

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")
from mlx_one.vision.vit_classification import (  # noqa: E402
    REVISION,
    load_vit_classifier,
    preprocess,
)

CHECKPOINT = Path(__file__).resolve().parents[2] / "local/native_tasks/checkpoints/vit"


def test_processor_shape_and_input_validation():
    image = Image.new("RGB", (23, 37), (0, 128, 255))
    pixels, size = preprocess(image)
    assert size == (23, 37)
    assert pixels.shape == (224, 224, 3) and pixels.dtype == np.float32
    assert np.allclose(pixels[0, 0], [-1, 1 / 255, 1], atol=1e-7)
    with pytest.raises(ValueError):
        preprocess(123)


def test_revision_and_missing_asset_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        load_vit_classifier(tmp_path, revision="main")
    with pytest.raises(ValueError, match="asset"):
        load_vit_classifier(tmp_path, revision=REVISION)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="pinned checkpoint not available locally")
def test_real_checkpoint_inference_is_deterministic():
    model = load_vit_classifier(CHECKPOINT)
    image = Image.new("RGB", (23, 37), (128, 16, 32))
    first = model.predict(image, top_k=5)
    assert first == model.predict(image, top_k=5)
    assert len(first["labels"]) == 5 and first["revision"] == REVISION
    assert all(0 <= row["score"] <= 1 for row in first["labels"])
    with pytest.raises(ValueError):
        model.predict(image, top_k=0)
