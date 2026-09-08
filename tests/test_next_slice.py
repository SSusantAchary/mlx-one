import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_one.evaluation import (
    TEXT_CODING_PROFILE,
    TEXT_INSTRUCTION_PROFILE,
    TEXT_REASONING_PROFILE,
    evaluate_text,
)
from mlx_one.evidence import publish_run_evidence
from mlx_one.registry import CapabilityRegistry, RegistryError
from mlx_one.run_store import RunStore
from mlx_one.schemas import (
    ArtifactRef,
    CapabilitySpec,
    CapabilityStatus,
    DatasetSpec,
    HardwareSpec,
    Modality,
    ModelSpec,
    Operation,
    RunResult,
    RunSpec,
    RunStatus,
    SoftwareProvenance,
)
from mlx_one.workflow import run_text_workflow


def _model() -> ModelSpec:
    return ModelSpec(model_id="example/model", revision="a" * 40, modality=Modality.TEXT)


@pytest.mark.parametrize(
    ("profile", "prediction", "reference", "metric", "score"),
    [
        (TEXT_CODING_PROFILE, "```python\nprint(1)\n```", "print(1)", "code_exact_match", 1.0),
        (TEXT_INSTRUCTION_PROFILE, "alpha beta", "alpha gamma", "token_f1", 0.5),
        (TEXT_REASONING_PROFILE, "Therefore 42", "42", "answer_accuracy", 1.0),
    ],
)
def test_pinned_evaluation_profiles_and_confidence_intervals(
    tmp_path: Path,
    profile: str,
    prediction: str,
    reference: str,
    metric: str,
    score: float,
) -> None:
    dataset = tmp_path / "eval.json"
    dataset.write_text(
        json.dumps(
            [
                {"id": "included", "prompt": "prompt", "reference": reference},
                {
                    "id": "excluded",
                    "prompt": "private control",
                    "reference": "no",
                    "excluded": True,
                    "exclusion_reason": "control",
                },
            ]
        )
    )
    spec = RunSpec(
        run_id="profile-run",
        operation=Operation.EVALUATE,
        model=_model(),
        profile=profile,
        dataset_revisions={"eval": hashlib.sha256(dataset.read_bytes()).hexdigest()},
        generation={"temperature": 0.0},
    )

    result = evaluate_text(
        spec,
        dataset,
        store=RunStore(tmp_path / "runs"),
        generated_predictions=[prediction, "ignored"],
    )

    value = result.metrics[f"quality.{metric}"]
    assert value["value"] == score
    assert value["confidence_interval_95"]["lower"] <= score
    assert result.task_results[0].metadata["excluded_samples"] == 1
    assert result.task_results[0].metadata["samples"][0]["sample_id"] == "included"


def test_schema_migrations_preserve_unknown_legacy_fields() -> None:
    migrated = DatasetSpec.from_dict(
        {
            "schema_version": "0.1",
            "id": "dataset",
            "path": "data.jsonl",
            "revision": "revision",
            "old_policy": "retained",
        }
    )
    artifact = ArtifactRef.from_dict(
        {"location": "artifact.json", "type": "result", "hash": "b" * 64}
    )

    assert migrated.schema_version == "1.0"
    assert migrated.metadata["legacy_fields"] == {"old_policy": "retained"}
    assert artifact.uri == "artifact.json"
    assert artifact.schema_version == "1.0"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        DatasetSpec.from_dict(
            {
                "schema_version": "2.0",
                "dataset_id": "future",
                "source": "future.json",
                "revision": "future",
            }
        )


def test_registry_promotion_requires_matching_successful_run(tmp_path: Path) -> None:
    entry = CapabilitySpec(
        model=_model(),
        backend="mlx-lm",
        operation=Operation.TRAIN,
        status=CapabilityStatus.CANDIDATE,
    )
    spec = RunSpec(
        run_id="train-run",
        operation=Operation.TRAIN,
        model=_model(),
        hardware=HardwareSpec(
            profile_id="m4-air-32gb",
            platform="macOS",
            architecture="arm64",
            chip="Apple M4",
        ),
    )
    failed = RunResult(run_id="train-run", status=RunStatus.FAILED, spec=spec)
    result_path = tmp_path / "result.json"
    result_path.write_text(failed.to_json())
    registry = CapabilityRegistry((entry,))

    with pytest.raises(RegistryError, match="completed"):
        registry.promote_from_result(entry, failed, result_path, "hardware-verified")

    completed = RunResult(
        run_id="train-run",
        status=RunStatus.COMPLETED,
        spec=spec,
        software=SoftwareProvenance(
            python_version="3.12", packages={"mlx-lm": "0.26.3"}
        ),
    )
    result_path.write_text(completed.to_json())
    promoted = registry.promote_from_result(
        entry, completed, result_path, CapabilityStatus.HARDWARE_VERIFIED
    )

    assert promoted.constraints["evidence_run_id"] == "train-run"
    assert promoted.evidence[-1].sha256 == hashlib.sha256(result_path.read_bytes()).hexdigest()
    assert registry.requiring_revalidation({"mlx-lm": "0.27.0"}) == (promoted,)


def test_publish_run_evidence_redacts_paths_and_text(tmp_path: Path) -> None:
    spec = RunSpec(
        run_id="publishable",
        operation=Operation.TRAIN,
        model=_model(),
        hardware=HardwareSpec(
            profile_id="m4-air-32gb",
            platform="macOS",
            architecture="arm64",
            chip="Apple M4",
        ),
        metadata={"output_dir": "/Users/person/private/run"},
    )
    source = tmp_path / "source.json"
    source.write_text(
        RunResult(
            run_id="publishable",
            status=RunStatus.COMPLETED,
            spec=spec,
            software=SoftwareProvenance(
                python_version="3.12", packages={"mlx-lm": "0.31.3"}
            ),
            metrics={"train.loss": {"value": 0.1, "provenance": "measured"}},
            artifacts=(
                ArtifactRef(
                    uri="/Users/person/private/adapter.safetensors",
                    kind="adapter",
                    sha256="c" * 64,
                ),
            ),
        ).to_json()
    )

    output, digest = publish_run_evidence(source, tmp_path / "published.json")
    content = output.read_text()

    assert "/Users/person" not in content
    assert "artifact://publishable/adapter" in content
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()


def test_reference_workflow_resume_skips_completed_stages(tmp_path: Path, monkeypatch) -> None:
    train = tmp_path / "train.yaml"
    train.write_text('schema_version: "1.0"\noutput_dir: ignored\n')
    for name in ("train.jsonl", "eval.jsonl", "prompts.json", "gate.yaml"):
        (tmp_path / name).write_text("[]")
    config = tmp_path / "workflow.yaml"
    output = tmp_path / "workflow-output"
    config.write_text(
        "\n".join(
            [
                'schema_version: "1.0"',
                "model: example/model",
                f"revision: {'a' * 40}",
                f"dataset: {tmp_path / 'train.jsonl'}",
                f"train_config: {train}",
                f"evaluation_dataset: {tmp_path / 'eval.jsonl'}",
                f"prompts: {tmp_path / 'prompts.json'}",
                f"gate: {tmp_path / 'gate.yaml'}",
                f"output_dir: {output}",
            ]
        )
    )
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("mlx_one.workflow.subprocess.run", fake_run)

    run_text_workflow(config)
    initial_count = len(calls)
    run_text_workflow(config, resume=True)

    assert initial_count == 9
    assert len(calls) == initial_count
    assert json.loads((output / "workflow.json").read_text())["status"] == "completed"
