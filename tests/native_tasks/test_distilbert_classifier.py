from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
from mlx_one.models.classification.distilbert import (  # noqa: E402
    REVISION,
    load_distilbert_classifier,
)

CHECKPOINT = Path(__file__).resolve().parents[2] / "local/native_tasks/checkpoints/distilbert-sst2"


def test_missing_checkpoint_and_revision_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        load_distilbert_classifier(tmp_path, revision="main")
    with pytest.raises(ValueError, match="asset"):
        load_distilbert_classifier(tmp_path, revision=REVISION)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="pinned checkpoint not available locally")
def test_real_checkpoint_batch_padding_and_validation():
    classifier = load_distilbert_classifier(CHECKPOINT)
    texts = ["Excellent!", "This was a remarkably dull and tedious movie."]
    singles = [classifier.predict(text) for text in texts]
    batch = classifier.predict(texts)
    for one, grouped in zip(singles, batch, strict=True):
        assert [row["label"] for row in one] == [row["label"] for row in grouped]
        assert [row["score"] for row in one] == pytest.approx(
            [row["score"] for row in grouped], abs=1e-5
        )
        assert sum(row["score"] for row in grouped) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        classifier.predict([])
    with pytest.raises(ValueError):
        classifier.predict("")
