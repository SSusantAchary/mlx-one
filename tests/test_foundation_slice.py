import hashlib
import json
from pathlib import Path

import pytest

from mlx_one import (
    ArtifactRef,
    CalibrationRecord,
    CalibrationStatus,
    CapabilitySpec,
    CapabilityStatus,
    DatasetLayout,
    HardwareSpec,
    MetricProvenance,
    MetricValue,
    Modality,
    ModelSpec,
    Operation,
    RunSpec,
    SoftwareProvenance,
    TrainConfig,
    TrainingMethod,
    WorkloadKind,
    WorkloadSpec,
)
from mlx_one.compatibility import validate_calibration_evidence, validate_capability
from mlx_one.datasets import (
    DatasetError,
    dataset_spec_from_path,
    load_records,
    preview_dataset,
    tokenize_record,
)
from mlx_one.registry import CapabilityRegistry
from mlx_one.run_store import RunStore, RunStoreError
from mlx_one.schemas import RunResult, RunStatus


def _model() -> ModelSpec:
    return ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B",
        revision="a" * 40,
        modality=Modality.TEXT,
    )


def _hardware() -> HardwareSpec:
    return HardwareSpec(
        profile_id="m4-air-32gb",
        platform="macOS",
        architecture="arm64",
        chip="Apple M4",
    )


def test_metric_provenance_cannot_mislabel_unavailable_values() -> None:
    measured = MetricValue(value=1.0, provenance=MetricProvenance.MEASURED)
    unavailable = MetricValue(value=None, provenance=MetricProvenance.NOT_AVAILABLE)

    assert MetricValue.from_dict(measured.to_dict()) == measured
    assert unavailable.value is None
    with pytest.raises(ValueError, match="cannot contain"):
        MetricValue(value=1.0, provenance=MetricProvenance.NOT_AVAILABLE)


def test_invalid_calibration_cannot_be_promoted_to_verified_evidence() -> None:
    record = CalibrationRecord(
        matrix_id="matrix",
        cell_id="cell",
        status=CalibrationStatus.INVALID,
        model=_model(),
        hardware=_hardware(),
        workload=WorkloadSpec(kind=WorkloadKind.INFERENCE),
        software=SoftwareProvenance(python_version="3.12", packages={"mlx": "1"}),
        metrics={"tokens_per_second": 1.0},
        started_at="2026-09-07T00:00:00Z",
        ended_at="2026-09-07T00:00:01Z",
    )

    report = validate_calibration_evidence(record)

    assert not report.valid
    assert "non-success-status" in {issue.code for issue in report.issues}


def test_verified_capability_requires_evidence_checksum() -> None:
    entry = CapabilitySpec(
        model=_model(),
        backend="mlx-lm",
        operation=Operation.BENCHMARK,
        status=CapabilityStatus.HARDWARE_VERIFIED,
        hardware_profile_ids=("m4-air-32gb",),
        evidence=(ArtifactRef(uri="result.json", kind="calibration"),),
    )

    report = validate_capability(entry)

    assert not report.valid
    assert "missing-checksum" in {issue.code for issue in report.issues}


def test_builtin_registry_contains_exact_qwen_calibration_entries() -> None:
    entries = CapabilityRegistry.builtin().entries

    measured = [entry for entry in entries if entry.status is CapabilityStatus.HARDWARE_VERIFIED]
    candidates = [entry for entry in entries if entry.status is CapabilityStatus.CANDIDATE]

    assert len(measured) == 5
    assert len(candidates) == 2
    assert all(len(entry.model.revision) == 40 for entry in entries)
    assert sum(entry.operation is Operation.BENCHMARK for entry in measured) == 3
    assert sum(entry.operation is Operation.TRAIN for entry in measured) == 2
    assert {entry.model.model_id for entry in candidates} == {
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    }
    assert {entry.operation for entry in candidates} == {Operation.EVALUATE}


def test_run_store_is_atomic_and_preserves_lineage(tmp_path: Path) -> None:
    spec = RunSpec(
        run_id="child",
        operation=Operation.EVALUATE,
        model=_model(),
        parent_run_ids=("parent",),
    )
    result = RunResult(run_id="child", status=RunStatus.COMPLETED, spec=spec)
    store = RunStore(tmp_path)

    store.create(spec)
    store.save_result(result)

    assert store.load_spec("child") == spec
    assert store.load_result("child") == result
    with pytest.raises(RunStoreError, match="already exists"):
        store.create(spec)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, **_kwargs) -> list[int]:
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        del tokenize
        text = "".join(f"{item['role']}:{item['content']}|" for item in messages)
        if add_generation_prompt:
            text += "assistant:"
        return self.encode(text)


def test_dataset_pipeline_normalizes_and_masks_response(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"instruction": "Add", "input": "1+1", "output": "2"}) + "\n",
        encoding="utf-8",
    )
    spec = dataset_spec_from_path(path, layout=DatasetLayout.INSTRUCTION)

    records = load_records(spec)
    prepared = tokenize_record(records[0], FakeTokenizer(), max_length=64)
    preview = preview_dataset(spec, FakeTokenizer(), max_length=64)

    assert records == [{"prompt": "Add\n\n1+1", "completion": "2"}]
    assert all(label == -100 for label in prepared.labels[: prepared.prompt_tokens])
    assert prepared.labels[prepared.prompt_tokens] != -100
    assert preview["record_count"] == 1


def test_dataset_hash_detects_changes(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text('[{"text":"hello"}]', encoding="utf-8")
    spec = dataset_spec_from_path(path)
    path.write_text('[{"text":"changed"}]', encoding="utf-8")

    with pytest.raises(DatasetError, match="sha256"):
        load_records(spec)


def test_train_config_is_strict_and_round_trips() -> None:
    config = TrainConfig(output_dir="runs/test", target_modules=("q_proj",))

    assert TrainConfig.from_json(config.to_json()) == config
    with pytest.raises(TypeError):
        TrainConfig.from_dict({**config.to_dict(), "silently_ignored": True})
    with pytest.raises(ValueError, match="only lora and qlora"):
        TrainConfig(output_dir="runs/test", method=TrainingMethod.FULL)


def test_dataset_spec_hash_matches_source(tmp_path: Path) -> None:
    source = tmp_path / "data.jsonl"
    source.write_text('{"text":"hello"}\n', encoding="utf-8")

    spec = dataset_spec_from_path(source)

    assert spec.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
