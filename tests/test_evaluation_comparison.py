import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlx_one.cli import main
from mlx_one.comparison import ComparisonError, compare_results
from mlx_one.evaluation import TEXT_EXACT_MATCH_PROFILE, evaluate_text
from mlx_one.run_store import RunStore
from mlx_one.schemas import Modality, ModelSpec, Operation, RunSpec


def _spec(run_id: str, *, revision: str = "a" * 40, dataset_revision: str = "data-1"):
    return RunSpec(
        run_id=run_id,
        operation=Operation.EVALUATE,
        model=ModelSpec(model_id="example/model", revision=revision, modality=Modality.TEXT),
        profile=TEXT_EXACT_MATCH_PROFILE,
        dataset_revisions={"tiny": dataset_revision},
        generation={"temperature": 0.0},
    )


def test_deterministic_evaluation_and_quality_gate(tmp_path: Path) -> None:
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "one", "reference": "Yes"}),
                json.dumps({"prompt": "two", "reference": "No"}),
            ]
        ),
        encoding="utf-8",
    )
    baseline_predictions = tmp_path / "baseline.json"
    baseline_predictions.write_text('["yes", "wrong"]', encoding="utf-8")
    candidate_predictions = tmp_path / "candidate.json"
    candidate_predictions.write_text('[" YES  ", "no"]', encoding="utf-8")
    store = RunStore(tmp_path / "runs")

    baseline = evaluate_text(
        _spec("baseline"), dataset, store=store, predictions=baseline_predictions
    )
    candidate = evaluate_text(
        _spec("candidate"), dataset, store=store, predictions=candidate_predictions
    )
    report = compare_results(
        baseline,
        candidate,
        gates={"metrics": {"quality.exact_match": {"minimum": 0.9}}},
    )

    assert baseline.metrics["quality.exact_match"]["value"] == 0.5
    assert candidate.metrics["quality.exact_match"]["value"] == 1.0
    assert report.passed
    assert report.metrics["quality.exact_match"]["delta"] == 0.5


def test_comparison_rejects_incompatible_protocols(tmp_path: Path) -> None:
    dataset = tmp_path / "eval.json"
    dataset.write_text('[{"prompt":"x","reference":"x"}]', encoding="utf-8")
    predictions = tmp_path / "predictions.json"
    predictions.write_text('["x"]', encoding="utf-8")
    store = RunStore(tmp_path / "runs")
    baseline = evaluate_text(_spec("a"), dataset, store=store, predictions=predictions)
    candidate = evaluate_text(
        _spec("b", dataset_revision="different"),
        dataset,
        store=store,
        predictions=predictions,
    )

    with pytest.raises(ComparisonError, match="protocols differ"):
        compare_results(baseline, candidate)
    assert not compare_results(baseline, candidate, allow_incompatible=True).passed


def test_comparison_allows_base_and_adapter_identity_to_differ(tmp_path: Path) -> None:
    dataset = tmp_path / "eval.json"
    dataset.write_text('[{"prompt":"x","reference":"x"}]', encoding="utf-8")
    predictions = tmp_path / "predictions.json"
    predictions.write_text('["x"]', encoding="utf-8")
    store = RunStore(tmp_path / "runs")
    baseline_spec = _spec("base")
    adapter_spec = RunSpec.from_dict(
        {
            **_spec("adapter").to_dict(),
            "generation": {"temperature": 0.0, "adapter": True},
        }
    )
    baseline = evaluate_text(baseline_spec, dataset, store=store, predictions=predictions)
    adapter = evaluate_text(adapter_spec, dataset, store=store, predictions=predictions)

    assert compare_results(baseline, adapter).compatible


def test_evaluate_cli_writes_machine_readable_result(tmp_path: Path) -> None:
    dataset = tmp_path / "eval.json"
    dataset.write_text('[{"prompt":"x","reference":"x"}]', encoding="utf-8")
    predictions = tmp_path / "predictions.json"
    predictions.write_text('["x"]', encoding="utf-8")
    runs = tmp_path / "runs"

    result = CliRunner().invoke(
        main,
        [
            "evaluate",
            "--model",
            "example/model",
            "--revision",
            "a" * 40,
            "--dataset",
            str(dataset),
            "--predictions",
            str(predictions),
            "--runs-dir",
            str(runs),
            "--run-id",
            "eval-cli",
            "--json-output",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["metrics"]["quality.exact_match"]["value"] == 1.0
    assert (runs / "eval-cli" / "result.json").is_file()
