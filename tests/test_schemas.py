import json
from dataclasses import FrozenInstanceError

import pytest

from mlx_one import (
    SCHEMA_VERSION,
    ArtifactRef,
    CachePolicy,
    CapabilityHint,
    CapabilityHintStatus,
    CompatibilityResult,
    CompatibilityStatus,
    Confidence,
    Failure,
    HardwareSpec,
    InspectionResult,
    Modality,
    ModelSource,
    ModelSpec,
    Operation,
    RunResult,
    RunSpec,
    RunStatus,
    SoftwareProvenance,
    TaskResult,
)


def model_spec(**overrides) -> ModelSpec:
    values = {
        "model_id": "example/model",
        "revision": "a" * 40,
        "modality": Modality.TEXT,
        "architectures": ("ExampleForCausalLM",),
        "parameter_count": 8_000_000_000,
    }
    values.update(overrides)
    return ModelSpec(**values)


def hardware_spec() -> HardwareSpec:
    return HardwareSpec(
        profile_id="m4-air-32gb",
        platform="macOS",
        architecture="arm64",
        chip="Apple M4",
        memory_bytes=32 * 1024**3,
        accelerator="Metal",
    )


def run_spec(**overrides) -> RunSpec:
    values = {
        "run_id": "run-001",
        "operation": Operation.EVALUATE,
        "model": model_spec(),
        "hardware": hardware_spec(),
        "profile": "text-core",
        "dataset_revisions": {"mmlu-pro": "revision-1"},
        "generation": {"temperature": 0.0, "max_tokens": 128},
        "seed": 42,
        "cache_policy": CachePolicy.OFFLINE,
        "created_at": "2026-09-06T12:00:00+00:00",
    }
    values.update(overrides)
    return RunSpec(**values)


def test_model_spec_round_trip_is_versioned_and_size_agnostic() -> None:
    spec = model_spec(
        quantization={"bits": 4},
        model_type="example",
        base_models=("example/base",),
        weight_bytes=4_000_000_000,
        requires_remote_code=True,
    )

    restored = ModelSpec.from_json(spec.to_json())

    assert restored == spec
    assert restored.parameter_count == 8_000_000_000
    assert restored.base_models == ("example/base",)
    assert restored.requires_remote_code is True
    assert restored.to_dict()["schema_version"] == SCHEMA_VERSION


def test_hardware_spec_validates_capacity() -> None:
    with pytest.raises(ValueError, match="memory_bytes cannot be negative"):
        HardwareSpec(
            profile_id="bad",
            platform="macOS",
            architecture="arm64",
            chip="Apple M4",
            memory_bytes=-1,
        )


def test_run_spec_round_trip_restores_nested_records_and_enums() -> None:
    spec = run_spec()

    restored = RunSpec.from_dict(spec.to_dict())

    assert restored == spec
    assert restored.operation is Operation.EVALUATE
    assert restored.cache_policy is CachePolicy.OFFLINE
    assert isinstance(restored.model, ModelSpec)
    assert isinstance(restored.hardware, HardwareSpec)


def test_run_spec_generates_timezone_aware_creation_time() -> None:
    spec = run_spec(created_at="")

    assert "+00:00" in spec.created_at


def test_run_spec_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="UTC offset"):
        run_spec(created_at="2026-09-06T12:00:00")


def test_run_result_round_trip_preserves_evidence() -> None:
    spec = run_spec()
    failure = Failure(code="sample-error", message="one sample failed", retryable=True)
    task = TaskResult(
        task="mmlu-pro",
        metrics={"accuracy": 0.72},
        sample_count=100,
        failures=(failure,),
    )
    artifact = ArtifactRef(
        uri="results/mmlu.json",
        kind="task-result",
        sha256="f" * 64,
        media_type="application/json",
    )
    software = SoftwareProvenance(
        python_version="3.12.11",
        packages={"mlx": "0.32.2", "mlx-one": "0.1.0a1"},
        git_commit="abc123",
    )
    result = RunResult(
        run_id=spec.run_id,
        status=RunStatus.PARTIAL,
        spec=spec,
        metrics={"accuracy": 0.72},
        task_results=(task,),
        failures=(failure,),
        timings={"wall_seconds": 12.5},
        memory={"peak_bytes": 1024},
        software=software,
        artifacts=(artifact,),
        started_at="2026-09-06T12:00:00Z",
        ended_at="2026-09-06T12:01:00Z",
    )

    restored = RunResult.from_json(result.to_json())

    assert restored == result
    assert restored.task_results[0].metrics["accuracy"] == 0.72
    assert restored.artifacts[0].sha256 == "f" * 64


def test_run_result_requires_matching_run_id() -> None:
    with pytest.raises(ValueError, match="run_id must match"):
        RunResult(run_id="other", status=RunStatus.COMPLETED, spec=run_spec())


def test_run_result_rejects_reversed_timestamps() -> None:
    with pytest.raises(ValueError, match="ended_at cannot be earlier"):
        RunResult(
            run_id="run-001",
            status=RunStatus.COMPLETED,
            spec=run_spec(),
            started_at="2026-09-06T12:01:00Z",
            ended_at="2026-09-06T12:00:00Z",
        )


def test_schema_rejects_unknown_version() -> None:
    payload = model_spec().to_dict()
    payload["schema_version"] = "99.0"

    with pytest.raises(ValueError, match="unsupported schema_version"):
        ModelSpec.from_dict(payload)


def test_schema_rejects_non_json_and_non_finite_metadata() -> None:
    with pytest.raises(TypeError, match="finite JSON values"):
        model_spec(metadata={"bad": object()})

    with pytest.raises(TypeError, match="finite JSON values"):
        model_spec(metadata={"bad": float("nan")})


def test_artifact_requires_valid_sha256() -> None:
    with pytest.raises(ValueError, match="64-character"):
        ArtifactRef(uri="result.json", kind="result", sha256="not-a-digest")


def test_compatibility_result_round_trip() -> None:
    result = CompatibilityResult(
        model=model_spec(),
        hardware=hardware_spec(),
        operation=Operation.TRAIN,
        backend="mlx-lm",
        status=CompatibilityStatus.EXPERIMENTAL,
        confidence=Confidence.MEDIUM,
        estimated_peak_memory_bytes=24 * 1024**3,
        reasons=("Estimated from neighboring model families.",),
        recommendations=("Use QLoRA and batch size 1.",),
    )

    restored = CompatibilityResult.from_json(result.to_json())

    assert restored == result
    assert restored.status is CompatibilityStatus.EXPERIMENTAL


def test_inspection_result_round_trip() -> None:
    hint = CapabilityHint(
        adapter="mlx-lm",
        status=CapabilityHintStatus.CANDIDATE,
        operations=(Operation.CONVERT, Operation.EVALUATE),
        installed_version="1.0",
        reason="Backend is installed; architecture support is not verified.",
    )
    result = InspectionResult(
        model=model_spec(),
        source=ModelSource.HUGGING_FACE,
        requested_revision="main",
        capabilities=(hint,),
        warnings=("License metadata is unavailable.",),
        inspected_at="2026-09-06T12:00:00Z",
    )

    restored = InspectionResult.from_json(result.to_json())

    assert restored == result
    assert restored.source is ModelSource.HUGGING_FACE
    assert restored.capabilities[0].status is CapabilityHintStatus.CANDIDATE


def test_records_are_frozen_and_json_is_deterministic() -> None:
    spec = model_spec()

    with pytest.raises(FrozenInstanceError):
        spec.model_id = "changed"  # type: ignore[misc]

    assert json.loads(spec.to_json()) == spec.to_dict()
    assert spec.to_json() == spec.to_json()
