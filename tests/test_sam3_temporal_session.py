"""Standalone temporal prompt validation, frame ordering and state ownership."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_sam3_workflows import fake_bundle as fake_bundle

from mlx_one.segmentation.temporal import Sam3TemporalSession


@pytest.fixture
def temporal_bundle(fake_bundle):
    from mlx_one.models.segmentation.sam3._runtime import ops

    class Model:
        def __call__(self, state, frame_idx, frame=None):
            if frame is not None:
                state.add_new_frame(frame, frame_idx)
            return SimpleNamespace(
                frame_idx=frame_idx,
                object_ids=list(state.obj_ids),
                pred_masks=ops.ones(len(state.obj_ids), 1, 4, 6),
                object_score_logits=ops.ones(len(state.obj_ids), 1),
            )

    return SimpleNamespace(
        task="tracking",
        processor=fake_bundle.processor,
        model=Model(),
        provenance={"qualified": False},
    )


def test_streaming_temporal_sessions_keep_object_ids_and_isolate_state(temporal_bundle):
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    first = Sam3TemporalSession(temporal_bundle, streaming=True, original_size=(4, 6))
    second = Sam3TemporalSession(temporal_bundle, streaming=True, original_size=(4, 6))
    root = Path(first._temp.name)
    try:
        first.add_object(4, points=[[2, 2]], point_labels=[1])
        second.add_object(9, box=[1, 1, 4, 3])
        assert first.process_frame(frame).object_ids == (4,)
        assert second.process_frame(frame).object_ids == (9,)
        with pytest.raises(ValueError, match="consecutive"):
            first.process_frame(frame, frame_index=4)
        with pytest.raises(ValueError, match="dimensions"):
            first.process_frame(np.zeros((5, 6, 3), dtype=np.uint8))
        first.reset()
        assert not first._state.obj_ids and second._state.obj_ids == [9]
        first.add_object(7, points=[[1, 1]], point_labels=[1])
        assert first.process_frame(frame).object_ids == (7,)
    finally:
        first.close()
        first.close()
        second.close()
    assert not root.exists()
    with pytest.raises(RuntimeError, match="closed"):
        first.process_frame(frame)


def test_offline_temporal_validation_and_callback(temporal_bundle):
    frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(3)]
    with Sam3TemporalSession(temporal_bundle, frames=frames) as session:
        with pytest.raises(ValueError, match="outside"):
            session.add_object(1, frame_index=8, points=[[1, 1]], point_labels=[1])
        with pytest.raises(ValueError, match="object_id"):
            session.add_object(True, points=[[1, 1]], point_labels=[1])
        with pytest.raises(ValueError, match="outside"):
            session.add_object(1, points=[[9, 1]], point_labels=[1])
        session.add_object(1, box=[1, 1, 5, 3])
        results = list(session.propagate())
        assert [r.frame_index for r in results] == [0, 1, 2]
        assert all(r.object_ids == (1,) and r.masks.shape == (1, 4, 6) for r in results)
        assert all(r.scores.shape == (1,) for r in results)
