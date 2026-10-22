"""Bounded image preprocessing without native model or Metal execution."""

import sys

import numpy as np
import pytest
from PIL import Image
from test_task_vlm_runtime import processor_config

from mlx_one.vision.processing import QwenImageProcessor


def test_placeholder_mismatch_fails_before_mlx_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    processor = QwenImageProcessor(None, processor_config())
    with pytest.raises(ValueError, match="placeholders"):
        processor.prepare("describe", [object()])


@pytest.mark.parametrize("options", [{"min_pixels": True}, {"max_pixels": 15},
    {"max_pixels": float("inf")}, {"max_source_pixels": 0},
    {"metadata": {"image_mean": [0, 0, 0], "image_std": [1, 1, 1], "size": 4}}])
def test_invalid_or_impossible_pixel_budgets_rejected(options):
    with pytest.raises(ValueError, match="pixel bounds|size metadata"):
        QwenImageProcessor(None, processor_config(), **options)


def test_source_budget_checked_before_image_conversion():
    processor = QwenImageProcessor(None, processor_config(), max_source_pixels=16)
    with Image.new("RGB", (8, 8)) as image:
        with pytest.raises(ValueError, match="source pixel budget"):
            processor._patches(image, np)


def test_exif_orientation_applied_before_grid_and_patch_alignment():
    processor = QwenImageProcessor(None, processor_config(), min_pixels=32, max_pixels=32)
    with Image.new("RGB", (8, 4), "red") as image:
        image.getexif()[274] = 6
        patches, grid = processor._patches(image, np)
        assert grid == (1, 4, 2)
        assert patches.shape == (8, 12)
        assert image.size == (8, 4)  # caller-owned image unchanged
        assert np.isfinite(patches).all()
