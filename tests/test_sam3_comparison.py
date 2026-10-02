"""Network-free gates for captured video arrays, including retained detections."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

compare_video = run_path(
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "sam3_compare.py")
)["compare_video"]


def capture(ids=(0,), frame_index=0):
    return {
        f"frame_{frame_index}_masks": np.ones((len(ids), 4, 6), dtype=bool),
        f"frame_{frame_index}_boxes": np.tile([0, 0, 5, 3], (len(ids), 1)).astype(float),
        f"frame_{frame_index}_scores": np.full(len(ids), 0.9),
        f"frame_{frame_index}_object_ids": np.asarray(ids, dtype=int),
    }


def test_fixed_id_mapping_and_empty_detections():
    assert compare_video(capture(), capture(), frames=1)["passed"]
    assert compare_video(capture((7,)), capture(), frames=1, id_mapping={0: 7})["passed"]
    assert compare_video(capture(()), capture(()), frames=1)["passed"]
    assert not compare_video(capture(()), capture(), frames=1)["passed"]


def test_missing_frames_and_identity_swaps_fail():
    assert not compare_video(capture(), capture(), frames=2)["passed"]
    native = capture() | capture((1,), frame_index=1)
    reference = capture() | capture(frame_index=1)
    report = compare_video(native, reference, frames=2)
    assert not report["passed"] and not report["frames"][1]["detections_match"]


@pytest.mark.parametrize("field", ["masks", "scores", "boxes"])
def test_numeric_gate_failures(field):
    native, reference = capture(), capture()
    native[f"frame_0_{field}"][...] = 0
    assert not compare_video(native, reference, frames=1)["passed"]


def test_nonfinite_scores_and_duplicate_ids_fail():
    native = capture()
    native["frame_0_scores"][0] = np.nan
    assert not compare_video(native, capture(), frames=1)["passed"]
    assert not compare_video(capture((0, 0)), capture((0, 0)), frames=1)["passed"]
