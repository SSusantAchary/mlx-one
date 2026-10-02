"""Network-free input/export contracts and small Metal lifecycle tests."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from mlx_one.cli import main
from mlx_one.segmentation.export import ResultWriter
from mlx_one.segmentation.postprocessing import connected_components, mask_nms
from mlx_one.segmentation.processing import Sam3Processor, read_image
from mlx_one.segmentation.resizing import resize_rgb_uint8
from mlx_one.segmentation.schemas import ConceptSegmentationResult


def test_result_export_and_overwrite_refusal(tmp_path):
    masks = np.zeros((1, 4, 6), dtype=bool)
    masks[0, 1:3, 2:5] = True
    result = ConceptSegmentationResult(
        masks, np.array([0.9]), np.array([[2, 1, 4, 2]]), (4, 6), {"qualified": False}
    )
    writer = ResultWriter(tmp_path / "result")
    writer.add(result)
    assert json.loads((writer.root / "manifest.json").read_text())["status"] == "partial"
    writer.complete()
    manifest = json.loads((writer.root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert (writer.root / manifest["results"][0]["masks"][0]).is_file()
    with pytest.raises(FileExistsError):
        ResultWriter(writer.root)
    with pytest.raises(FileExistsError):
        writer.add(result)


def test_empty_result_export(tmp_path):
    result = ConceptSegmentationResult(
        np.zeros((0, 4, 6), dtype=bool), np.zeros(0), np.zeros((0, 4)), (4, 6)
    )
    writer = ResultWriter(tmp_path / "empty")
    writer.add(result)
    writer.complete()
    assert json.loads((writer.root / "manifest.json").read_text())["results"][0]["masks"] == []


def test_cli_help_is_backend_free_and_output_guard(tmp_path):
    runner = CliRunner()
    for command in ("segment", "track"):
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code == 0
        assert "unqualified" in result.output
    from PIL import Image

    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 8)).save(image)
    output = tmp_path / "existing"
    output.mkdir()
    result = runner.invoke(main, ["segment", "not-a-model", str(image), "--output", str(output)])
    assert result.exit_code != 0 and "overwriting is refused" in result.output


def test_antialiased_resize_preserves_constants_and_identity():
    value = np.full((9, 13, 3), 127, dtype=np.uint8)
    assert np.array_equal(resize_rgb_uint8(value, 9, 13), value)
    assert np.array_equal(resize_rgb_uint8(value, 3, 4), np.full((3, 4, 3), 127, dtype=np.uint8))


def test_connected_components_eight_connectivity_and_counts():
    value = np.zeros((1, 1, 5, 5), dtype=bool)
    value[0, 0, 0, 0] = True
    value[0, 0, 1, 1] = True
    value[0, 0, 4, 4] = True
    labels, counts = connected_components(value)
    assert labels[0, 0, 0, 0] == labels[0, 0, 1, 1]
    assert counts[0, 0, 0, 0] == 2 and counts[0, 0, 4, 4] == 1
    assert not counts[~value].any()
    assert mask_nms(
        np.array([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]]), np.array([0.9, 0.8, 0.7]), 0.5
    ).tolist() == [0, 2]


def test_image_and_processor_validation():
    with pytest.raises(ValueError):
        read_image(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        Sam3Processor({"image_std": [float("nan")] * 3}, None, 1008)
    with pytest.raises(ValueError):
        Sam3Processor({"do_pad": True}, None, 1008)


@pytest.fixture
def fake_bundle():
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; mx.eval(mx.zeros(1))"],
        capture_output=True,
    )
    if probe.returncode:
        pytest.skip("MLX Metal runtime is unavailable in this environment")
    pytest.importorskip("mlx.core", exc_type=ImportError)
    from mlx_one.models.segmentation.sam3._runtime import Tensor, ops

    class Processor:
        image_size = 12

        def image(self, image):
            image = read_image(image)
            return Tensor(np.asarray(image).transpose(2, 0, 1)[None].astype(np.float32)), image

        def text(self, text):
            return ops.tensor([[1]]), ops.tensor([[1]])

    class Model:
        hotstart_delay = 2

        def __call__(self, state, frame=None, frame_idx=0):
            if frame is not None:
                state.add_new_frame(frame, frame_idx)
            state.obj_id_to_idx(0)
            state.obj_id_to_prompt_id[0] = 0
            return SimpleNamespace(
                obj_id_to_mask={0: ops.ones(1, 4, 6)},
                obj_id_to_score={0: 0.9},
                obj_id_to_tracker_score={0: 0.9},
                removed_obj_ids=set(),
                suppressed_obj_ids=set(),
                frame_idx=frame_idx,
            )

    return SimpleNamespace(
        task="video", model=Model(), processor=Processor(), provenance={"qualified": False}
    )


def test_streaming_isolation_indices_reset_and_close(fake_bundle):
    from mlx_one.segmentation.video import Sam3VideoSession

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    first = Sam3VideoSession(fake_bundle, streaming=True)
    second = Sam3VideoSession(fake_bundle, streaming=True)
    first.add_prompt("truck")
    second.add_prompt("truck")
    assert first.process_frame(frame).object_ids == (0,)
    assert first._state.num_frames == 1 and not first._state.processed_frames
    with pytest.raises(ValueError):
        first.process_frame(frame, frame_index=3)
    with pytest.raises(ValueError):
        first.process_frame(np.zeros((6, 6, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        first.correct_object(0, 0, points=[[1, 1]], point_labels=[1])
    first.close()
    first.close()
    assert second.process_frame(frame).frame_index == 0
    second.reset()
    second.add_prompt("truck")
    assert second.process_frame(frame).frame_index == 0
    second.close()


def test_offline_replay_prompt_correction_and_cleanup(fake_bundle):
    from mlx_one.segmentation.video import Sam3VideoSession

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    with Sam3VideoSession(fake_bundle, frames=[frame, frame, frame]) as session:
        temporary = Path(session._temp.name)
        prompt = session.add_prompt("truck")
        assert [r.frame_index for r in session.propagate()] == [0, 1, 2]
        session.correct_object(0, 1, points=[[1, 1]], point_labels=[1], prompt_id=prompt)
        assert 1 in session._state.point_inputs_per_obj[0]
        assert [r.frame_index for r in session.propagate(start_frame_index=1)] == [1, 2]
        session.remove_object(0)
        assert not session._state.obj_ids
    assert not temporary.exists()


def test_cancellation_failure_and_budget_cleanup(fake_bundle):
    from mlx_one.segmentation.video import Sam3VideoSession

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    session = Sam3VideoSession(fake_bundle, streaming=True)
    session.cancel()
    session.cancel()
    assert session.closed
    with pytest.raises(RuntimeError):
        session.add_prompt("truck")
    session = Sam3VideoSession(fake_bundle, streaming=True, memory_budget_bytes=1)
    session.add_prompt("truck")
    with pytest.raises(MemoryError):
        session.process_frame(frame)
    assert session.closed

    def failure(*args, **kwargs):
        raise RuntimeError("deliberate inference failure")

    broken = SimpleNamespace(**vars(fake_bundle))
    broken.model = failure
    session = Sam3VideoSession(broken, streaming=True)
    session.add_prompt("truck")
    with pytest.raises(RuntimeError, match="deliberate inference failure"):
        session.process_frame(frame)
    assert session.closed
