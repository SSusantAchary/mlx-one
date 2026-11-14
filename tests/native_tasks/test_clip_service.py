"""Real-checkpoint CLIP task contracts for the admitted candidate-label profile."""

import json
import threading
from concurrent.futures import CancelledError
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "local/native_tasks/checkpoints/clip-converted"
RUN = ROOT / "local/native_tasks/runs/m2-20261103/clip"
IMAGE = RUN / "real-images/chelsea.png"
pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists() or not IMAGE.exists(), reason="local pinned CLIP fixture absent"
)


@pytest.fixture
def engine():
    from mlx_one import ZeroShotImageEngine

    with ZeroShotImageEngine(CHECKPOINT) as engine:
        yield engine


def test_clip_source_scores_ranking_repeat_and_duplicate_labels(engine):
    result = engine.classify(IMAGE, ["cat", "dog"])
    assert result.labels[0]["label"] == "cat"
    assert result.score_semantics == "clip-candidate-softmax"
    assert sum(v["score"] for v in result.labels) == pytest.approx(1.0, abs=1e-6)
    assert result.to_dict() == engine.classify(IMAGE, ["cat", "dog"]).to_dict()
    with np.load(RUN / "matrix/chelsea-2/reference.npz", allow_pickle=False) as ref:
        for row in result.labels:
            assert row["score"] == pytest.approx(
                float(ref["probabilities"][0, row["index"]]), abs=1e-4, rel=1e-4
            )
            assert row["logit"] == pytest.approx(
                float(ref["logits_per_image"][0, row["index"]]), abs=1e-4, rel=1e-4
            )
    dup = engine.classify(IMAGE, ["cat", "cat", "dog"])
    assert [v["index"] for v in dup.labels[:2]] == [0, 1]
    assert dup.labels[0]["score"] == dup.labels[1]["score"]
    gray = Image.new("L", (50, 80), 128)
    assert engine.classify(gray, ["cat", "dog"]).image_size == {"width": 50, "height": 80}
    engine.service.unload("classification")
    assert result.to_dict() == engine.classify(IMAGE, ["cat", "dog"]).to_dict()
    assert engine.service.pool.stats()["active_leases"] == 0


def test_clip_bounds_bad_assets_cancellation_and_transparency(engine):
    for labels in [[], [" "], list(map(str, range(100))), ["a" * 513], ["cat", 42], ["\ud800"]]:
        with pytest.raises(ValueError):
            engine.classify(IMAGE, labels)
    with pytest.raises(ValueError):
        engine.classify(IMAGE, ["cat"], top_k=2)
    with pytest.raises(ValueError):
        engine.classify(Image.new("RGBA", (10, 10)), ["cat", "dog"])
    with pytest.raises(OSError):
        engine.classify(b"bad image", ["cat", "dog"])
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        engine.classify(IMAGE, ["cat", "dog"], cancel=cancel)
    assert engine.classify(IMAGE, ["cat", "dog"]).labels
    assert engine.service.pool.stats()["active_leases"] == 0


def test_clip_python_cli_http_match_and_reject_server_paths(engine):
    from click.testing import CliRunner
    from fastapi.testclient import TestClient

    from mlx_one.cli import main
    from mlx_one.server import ModelManager, ServerConfig, create_app

    expected = json.loads(engine.classify(IMAGE, ["cat", "dog"]).to_json())
    cli = CliRunner().invoke(
        main,
        [
            "zero-shot-image",
            str(CHECKPOINT),
            str(IMAGE),
            "--label",
            "cat",
            "--label",
            "dog",
            "--json-output",
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output) == expected
    manager = ModelManager(alias="clip")
    manager.load(CHECKPOINT, offline=True)
    with TestClient(
        create_app(model_manager=manager, config=ServerConfig(api_keys=("secret",), warmup=False))
    ) as c:
        data = {"model": "clip", "candidate_labels": json.dumps(["cat", "dog"])}
        files = {"file": ("cat.png", IMAGE.read_bytes(), "image/png")}
        route = "/v1/tasks/zero-shot-image-classification"
        assert c.post(route, data=data, files=files).status_code == 401
        c.headers["Authorization"] = "Bearer secret"
        response = c.post(route, data=data, files=files)
        assert response.status_code == 200, response.text
        assert response.json() == expected
        assert c.post(route, json={"model": "clip", "image": "/etc/passwd"}).status_code == 415
        assert (
            c.post(route, data={**data, "candidate_labels": "bad"}, files=files).status_code == 422
        )
        assert (
            c.post(route, data=data, files={"file": ("bad.txt", b"bad", "text/plain")}).status_code
            == 415
        )
        assert (
            c.post(route, data=data, files={"file": ("bad.png", b"bad", "image/png")}).status_code
            == 422
        )
        assert c.post(route, data={**data, "model": "wrong"}, files=files).status_code == 404
        assert (
            c.post(
                "/v1/chat/completions",
                json={"model": "clip", "messages": [{"role": "user", "content": "Hello"}]},
            ).status_code
            == 400
        )
        assert c.post(route, data=data, files=files).json() == expected
    with pytest.raises(RuntimeError):
        manager.current_model()
