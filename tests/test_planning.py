from pathlib import Path

import pytest

from mlx_one import (
    Confidence,
    FitStatus,
    MemoryEstimate,
    Modality,
    ModelDimensions,
    ModelSpec,
    PlanningError,
    PlanResult,
    Precision,
    TrainingMethod,
    WorkloadKind,
    WorkloadSpec,
    estimate_memory,
    load_hardware_profile,
    plan_inference,
    plan_training,
)


def model_spec(**overrides) -> ModelSpec:
    values = {
        "model_id": "Qwen/example",
        "revision": "a" * 40,
        "modality": Modality.TEXT,
        "model_type": "qwen2",
        "parameter_count": 1_500_000_000,
        "parameter_count_source": "test",
        "dtype": "BF16",
        "dimensions": ModelDimensions(
            hidden_size=1536,
            layer_count=28,
            attention_heads=12,
            key_value_heads=2,
            head_dimension=128,
            intermediate_size=8960,
            vocabulary_size=151936,
        ),
    }
    values.update(overrides)
    return ModelSpec(**values)


def test_inference_estimate_is_monotonic_with_context_and_batch() -> None:
    hardware = load_hardware_profile("m4-air-32gb")
    small = estimate_memory(
        model_spec(),
        hardware,
        WorkloadSpec(kind=WorkloadKind.INFERENCE, context_length=512, batch_size=1),
    )
    large = estimate_memory(
        model_spec(),
        hardware,
        WorkloadSpec(kind=WorkloadKind.INFERENCE, context_length=4096, batch_size=2),
    )

    assert large.central_bytes > small.central_bytes
    assert large.components["kv_cache_bytes"] > small.components["kv_cache_bytes"]


@pytest.mark.parametrize(
    "method",
    [
        TrainingMethod.FULL,
        TrainingMethod.LORA,
        TrainingMethod.QLORA,
        TrainingMethod.SCRATCH,
    ],
)
def test_all_text_training_methods_return_ranges(method: TrainingMethod) -> None:
    result = estimate_memory(
        model_spec(),
        load_hardware_profile("m4-air-32gb"),
        WorkloadSpec(kind=WorkloadKind.TRAIN, method=method),
    )

    assert 0 < result.lower_bytes <= result.central_bytes <= result.upper_bytes
    if method in (TrainingMethod.FULL, TrainingMethod.SCRATCH):
        assert result.confidence is Confidence.LOW
        assert "not yet M4-calibrated" in " ".join(result.warnings)
    else:
        assert result.confidence in {Confidence.MEDIUM, Confidence.HIGH}


def test_quantized_inference_uses_less_weight_memory() -> None:
    hardware = load_hardware_profile("m4-air-32gb")
    bf16 = estimate_memory(
        model_spec(),
        hardware,
        WorkloadSpec(kind=WorkloadKind.INFERENCE, precision=Precision.BF16),
    )
    int4 = estimate_memory(
        model_spec(),
        hardware,
        WorkloadSpec(kind=WorkloadKind.INFERENCE, precision=Precision.INT4),
    )

    assert int4.components["weights_bytes"] < bf16.components["weights_bytes"]


def test_training_rejects_incompatible_base_precision() -> None:
    hardware = load_hardware_profile("m4-air-32gb")
    with pytest.raises(PlanningError, match="floating-point base weights"):
        estimate_memory(
            model_spec(),
            hardware,
            WorkloadSpec(
                kind=WorkloadKind.TRAIN,
                method=TrainingMethod.LORA,
                precision=Precision.INT4,
            ),
        )
    with pytest.raises(PlanningError, match="requires 4-bit"):
        estimate_memory(
            model_spec(),
            hardware,
            WorkloadSpec(
                kind=WorkloadKind.TRAIN,
                method=TrainingMethod.QLORA,
                precision=Precision.BF16,
            ),
        )


def test_unknown_metadata_and_non_text_models_are_honest() -> None:
    hardware = load_hardware_profile("m4-air-32gb")
    missing = estimate_memory(
        model_spec(parameter_count=None, weight_bytes=None),
        hardware,
        WorkloadSpec(kind=WorkloadKind.INFERENCE),
    )
    vlm = estimate_memory(
        model_spec(modality=Modality.VISION_LANGUAGE),
        hardware,
        WorkloadSpec(kind=WorkloadKind.INFERENCE),
    )

    assert missing.fit is FitStatus.UNKNOWN
    assert vlm.fit is FitStatus.UNKNOWN


def test_parameter_count_is_not_a_software_limit() -> None:
    result = estimate_memory(
        model_spec(parameter_count=100_000_000_000),
        load_hardware_profile("m4-air-32gb"),
        WorkloadSpec(kind=WorkloadKind.INFERENCE, precision=Precision.BF16),
    )

    assert result.fit is FitStatus.DOES_NOT_FIT
    assert result.central_bytes is not None


def test_auto_training_selects_least_restrictive_fitting_method() -> None:
    result = plan_training(
        model_spec(parameter_count=3_000_000_000),
        hardware=load_hardware_profile("m4-air-32gb"),
        method=TrainingMethod.AUTO,
    )

    assert result.requested_workload.method is TrainingMethod.AUTO
    assert result.recommended_workload.method in {
        TrainingMethod.FULL,
        TrainingMethod.LORA,
        TrainingMethod.QLORA,
    }
    assert len(result.alternatives) == 2


def test_plan_functions_accept_specs_and_round_trip() -> None:
    result = plan_inference(
        model_spec(),
        hardware=load_hardware_profile("m4-air-32gb"),
        context_length=1024,
    )

    assert PlanResult.from_json(result.to_json()) == result
    assert MemoryEstimate.from_json(result.estimate.to_json()) == result.estimate


def test_plan_output_is_atomic(tmp_path: Path) -> None:
    from mlx_one.planning import write_plan_result

    result = plan_inference(model_spec(), hardware=load_hardware_profile("m4-air-32gb"))
    output = tmp_path / "nested" / "plan.json"

    saved = write_plan_result(result, output)

    assert saved == output
    assert PlanResult.from_json(output.read_text(encoding="utf-8")) == result
