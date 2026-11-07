"""Pinned NER task and interface tests; source outputs are local immutable goldens."""

import json
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "local/native_tasks/checkpoints/bert-ner"
RUN = ROOT / "local/native_tasks/runs/m2-20261103"
pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="pinned local NER checkpoint absent"
)


@pytest.fixture
def engine():
    from mlx_one.classification import TokenClassificationEngine

    with TokenClassificationEngine(CHECKPOINT) as engine:
        yield engine


def test_ner_spans_padding_unicode_repeats_and_reload(engine):
    text = "Marie Curie lived in Paris."
    result = engine.tag(text, aggregation_strategy="simple")
    entities = result.predictions[0]["entities"]
    assert [(e["entity_group"], e["start"], e["end"]) for e in entities] == [
        ("PER", 0, 11),
        ("LOC", 21, 26),
    ]
    assert result.offsets == "unicode-codepoint-end-exclusive"
    batch = engine.tag([text, "Beyoncé performed in New York."], aggregation_strategy="simple")
    assert batch.predictions[1]["entities"][0]["end"] == 7
    assert batch.predictions[0]["entities"][0]["score"] == pytest.approx(
        entities[0]["score"], abs=1e-4
    )
    repeated = engine.tag("John Smith met John Smith in London.", aggregation_strategy="simple")
    people = [e for e in repeated.predictions[0]["entities"] if e["entity_group"] == "PER"]
    assert len(people) == 2 and people[0]["start"] != people[1]["start"]
    assert engine.tag(text, aggregation_strategy="simple").to_dict() == result.to_dict()
    short = engine.tag("London " * 600, max_length=8)
    assert short.predictions[0]["token_count"] == 8 and short.predictions[0]["truncated"]
    engine.service.unload("classification")
    assert engine.tag(text, aggregation_strategy="simple").to_dict() == result.to_dict()
    assert engine.service.pool.stats()["active_leases"] == 0


def test_ner_negative_options_cancel_deadline_and_tokenizer_isolation(engine):
    from mlx_one.engine.execution_context import RequestTimeoutError

    for text in ["", "  ", [], ["a", "b", "c"], ["a", 42], "a" * 32769]:
        with pytest.raises(ValueError):
            engine.tag(text)
    with pytest.raises(ValueError):
        engine.tag("A", aggregation_strategy="average")
    with pytest.raises(ValueError):
        engine.tag("A", max_length=513)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        engine.tag("Paris.", cancel=cancel)
    with pytest.raises(RequestTimeoutError):
        engine.tag("Paris.", deadline=time.monotonic() - 1)

    def run(limit):
        return engine.tag("London " * 100, max_length=limit).predictions[0]["token_count"]

    with ThreadPoolExecutor(4) as workers:
        assert list(workers.map(run, [8, 16, 8, 16])) == [8, 16, 8, 16]
    assert engine.service.pool.stats()["active_leases"] == 0


@pytest.mark.skipif(
    not (RUN / "ner-b1-hf-tasks.json").exists(), reason="local reference goldens absent"
)
def test_public_ner_matches_pinned_pipeline_entities_and_decoder(engine):
    cases = json.loads((ROOT / "benchmarks/native_tasks/ner_cases.json").read_text())
    expected = json.loads((RUN / "ner-b1-hf-tasks.json").read_text())
    for strategy in ("none", "simple"):
        for i in (0, 1, 16, 19, 20, 23, 24):
            actual = engine.tag(cases[i], aggregation_strategy=strategy).predictions[0]["entities"]
            ref = expected[strategy][i]
            assert len(actual) == len(ref)
            for a, b in zip(actual, ref, strict=True):
                assert {k: v for k, v in a.items() if k != "score"} == {
                    k: v for k, v in b.items() if k != "score"
                }
                assert a["score"] == pytest.approx(b["score"], abs=1e-4, rel=1e-4)


def test_ner_python_cli_http_identity_and_errors(engine):
    from click.testing import CliRunner
    from fastapi.testclient import TestClient

    from mlx_one.cli import main
    from mlx_one.server import ModelManager, ServerConfig, create_app

    text = "Marie Curie lived in Paris."
    expected = json.loads(engine.tag(text, aggregation_strategy="simple").to_json())
    cli = CliRunner().invoke(
        main, ["ner", str(CHECKPOINT), text, "--aggregation-strategy", "simple", "--json-output"]
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output) == expected
    manager = ModelManager(alias="entities")
    manager.load(CHECKPOINT, offline=True)
    with TestClient(
        create_app(model_manager=manager, config=ServerConfig(api_keys=("secret",), warmup=False))
    ) as c:
        body = {"model": "entities", "input": text, "aggregation_strategy": "simple"}
        assert c.post("/v1/tasks/token-classification", json=body).status_code == 401
        c.headers["Authorization"] = "Bearer secret"
        response = c.post("/v1/tasks/token-classification", json=body)
        assert response.status_code == 200, response.text
        assert response.json() == expected
        assert (
            c.post("/v1/tasks/token-classification", json={**body, "model": "wrong"}).status_code
            == 404
        )
        for extra in [
            {"input": "  "},
            {"aggregation_strategy": "max"},
            {"max_length": True},
            {"dtype": "float16"},
        ]:
            assert (
                c.post("/v1/tasks/token-classification", json={**body, **extra}).status_code == 422
            )
        assert (
            c.post(
                "/v1/tasks/text-classification", json={"model": "entities", "input": text}
            ).status_code
            == 400
        )
        assert (
            c.post(
                "/v1/chat/completions",
                json={"model": "entities", "messages": [{"role": "user", "content": text}]},
            ).status_code
            == 400
        )
        assert c.post("/v1/tasks/token-classification", json=body).json() == expected
    with pytest.raises(RuntimeError):
        manager.current_model()
