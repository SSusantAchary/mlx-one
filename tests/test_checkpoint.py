import json

import pytest

from mlx_one.utils.checkpoint import CheckpointManager


def test_load_missing_checkpoint_returns_none(tmp_path) -> None:
    manager = CheckpointManager(checkpoint_dir=tmp_path)

    assert manager.load() is None


def test_save_load_round_trip_with_metadata(tmp_path) -> None:
    manager = CheckpointManager("eval.json", checkpoint_dir=tmp_path)
    state = {"completed": [1, 2, 3], "score": 0.75}

    saved_path = manager.save(
        state,
        model_name="mlx-community/SmolLM-135M-4bit",
        operation_type="eval",
        progress_percentage=25.5,
    )
    loaded = manager.load()

    assert saved_path == tmp_path / "eval.json"
    assert loaded is not None
    assert loaded["state"] == state
    assert loaded["metadata"]["model_name"] == "mlx-community/SmolLM-135M-4bit"
    assert loaded["metadata"]["operation_type"] == "eval"
    assert loaded["metadata"]["progress_percentage"] == 25.5
    assert "timestamp" in loaded["metadata"]


def test_clear_deletes_checkpoint(tmp_path) -> None:
    manager = CheckpointManager(checkpoint_dir=tmp_path)
    manager.save({"step": 1})

    manager.clear()

    assert manager.load() is None


def test_save_rejects_non_json_state(tmp_path) -> None:
    manager = CheckpointManager(checkpoint_dir=tmp_path)

    with pytest.raises(TypeError, match="JSON serializable"):
        manager.save({"bad": object()})


def test_load_returns_json_payload(tmp_path) -> None:
    manager = CheckpointManager(checkpoint_dir=tmp_path)
    manager.path.write_text(json.dumps({"metadata": {}, "state": {"ok": True}}), encoding="utf-8")

    assert manager.load() == {"metadata": {}, "state": {"ok": True}}
