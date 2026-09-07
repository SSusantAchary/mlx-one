"""Metadata-only memory estimation and workload planning."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any

from mlx_one.hardware import detect_hardware, load_hardware_profile
from mlx_one.inspection import inspect_model
from mlx_one.schemas import (
    Confidence,
    FitStatus,
    HardwareSpec,
    MemoryEstimate,
    Modality,
    ModelSpec,
    PlanResult,
    Precision,
    TrainingMethod,
    WorkloadKind,
    WorkloadSpec,
)
from mlx_one.utils.memory import format_bytes

GIB = 1024**3
_PRECISION_BYTES = {
    Precision.BF16: 2.0,
    Precision.FP16: 2.0,
    Precision.FP32: 4.0,
    Precision.INT8: 1.0,
    Precision.INT4: 0.5625,  # Includes a conservative allowance for quantization metadata.
}


class PlanningError(RuntimeError):
    """An actionable failure to construct a workload plan."""


def estimate_memory(
    model: ModelSpec,
    hardware: HardwareSpec,
    workload: WorkloadSpec,
) -> MemoryEstimate:
    """Estimate peak memory without importing a model runtime or loading weights."""
    if model.modality not in (Modality.TEXT, Modality.UNKNOWN):
        return _unknown_estimate(
            workload,
            hardware,
            f"The first estimator supports text models, not {model.modality.value} models.",
        )

    precision = _resolve_precision(model, workload)
    effective = replace(workload, precision=precision)
    if effective.kind is WorkloadKind.TRAIN:
        if effective.method is TrainingMethod.QLORA and precision is not Precision.INT4:
            raise PlanningError("QLoRA planning requires 4-bit base weights")
        if effective.method is not TrainingMethod.QLORA and precision in {
            Precision.INT4,
            Precision.INT8,
        }:
            raise PlanningError(
                f"{effective.method.value} training requires floating-point base weights"
            )
    parameters = model.parameter_count
    weight_bytes = _weight_bytes(model, precision)
    if weight_bytes is None:
        return _unknown_estimate(
            effective,
            hardware,
            "Parameter count and usable weight-size metadata are unavailable.",
        )

    dimensions = model.dimensions
    assumptions: list[str] = []
    warnings: list[str] = []
    if parameters is None:
        parameters = max(int(weight_bytes / _PRECISION_BYTES.get(precision, 2.0)), 1)
        assumptions.append("Parameter count was inferred from weight bytes and precision.")
    if dimensions.hidden_size is None or dimensions.layer_count is None:
        assumptions.append("Activation memory uses a parameter-count fallback.")

    components: dict[str, int | float | str] = {"weights_bytes": weight_bytes}
    if effective.kind is WorkloadKind.INFERENCE:
        kv_cache = _kv_cache_bytes(model, effective, precision)
        activations = _activation_bytes(model, effective, training=False)
        runtime = max(int(weight_bytes * 0.10), 512 * 1024**2)
        central = weight_bytes + kv_cache + activations + runtime
        components.update(
            kv_cache_bytes=kv_cache,
            activation_bytes=activations,
            runtime_overhead_bytes=runtime,
        )
        spread = (0.88, 1.25)
    else:
        method = effective.method
        if method is TrainingMethod.AUTO:
            raise PlanningError("estimate_memory requires a concrete training method")
        activations = _activation_bytes(model, effective, training=True)
        temporary = max(int(weight_bytes * 0.08), 512 * 1024**2)
        if method in (TrainingMethod.FULL, TrainingMethod.SCRATCH):
            trainable = parameters
            gradient_and_optimizer = int(trainable * 12)
            warnings.append(
                "Full and scratch training estimates are analytic and not yet M4-calibrated."
            )
        else:
            trainable = _adapter_parameter_count(model, effective)
            gradient_and_optimizer = int(trainable * 14)
        central = weight_bytes + gradient_and_optimizer + activations + temporary
        components.update(
            trainable_parameter_count=trainable,
            gradient_optimizer_bytes=gradient_and_optimizer,
            activation_bytes=activations,
            temporary_buffer_bytes=temporary,
        )
        spread = (0.78, 1.40)

    calibration = _nearest_calibration(model, hardware, effective, central)
    references: tuple[str, ...] = ()
    confidence = Confidence.LOW
    if calibration is not None:
        central = calibration["central_bytes"]
        spread = calibration["spread"]
        references = (calibration["reference"],)
        confidence = calibration["confidence"]
        assumptions.append("Estimate adjusted using the nearest compatible calibration cell.")

    lower = max(int(central * spread[0]), weight_bytes)
    upper = max(int(central * spread[1]), central)
    budget = hardware.process_memory_budget_bytes
    fit = _fit_status(lower, central, upper, budget)
    if budget is None:
        warnings.append("Hardware process-memory budget is unknown.")
    if effective.context_length > 4096:
        warnings.append("Context length is outside the initial M4 calibration range.")
        confidence = Confidence.LOW
    if hardware.profile_id != "m4-air-32gb" and not hardware.reference_device:
        warnings.append("This hardware has no bundled calibration evidence.")
        confidence = Confidence.LOW

    return MemoryEstimate(
        workload=effective,
        lower_bytes=lower,
        central_bytes=central,
        upper_bytes=upper,
        budget_bytes=budget,
        fit=fit,
        confidence=confidence,
        components=components,
        assumptions=tuple(assumptions),
        warnings=tuple(warnings),
        calibration_references=references,
    )


def plan_inference(
    model: str | Path | ModelSpec,
    *,
    hardware: str | Path | HardwareSpec = "local",
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    precision: Precision | str = Precision.AUTO,
    context_length: int = 2048,
    batch_size: int = 1,
    generation_length: int = 32,
) -> PlanResult:
    """Inspect and plan a text inference workload without loading weights."""
    model_spec = _resolve_model(model, revision, offline, cache_dir)
    hardware_spec = _resolve_hardware(hardware)
    workload = WorkloadSpec(
        kind=WorkloadKind.INFERENCE,
        precision=precision,
        context_length=context_length,
        batch_size=batch_size,
        generation_length=generation_length,
    )
    estimate = estimate_memory(model_spec, hardware_spec, workload)
    return PlanResult(
        model=model_spec,
        hardware=hardware_spec,
        requested_workload=workload,
        recommended_workload=estimate.workload,
        estimate=estimate,
    )


def plan_training(
    model: str | Path | ModelSpec,
    *,
    hardware: str | Path | HardwareSpec = "local",
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    method: TrainingMethod | str = TrainingMethod.AUTO,
    precision: Precision | str = Precision.AUTO,
    context_length: int = 2048,
    batch_size: int = 1,
    lora_rank: int = 8,
    lora_layers: int = 16,
    gradient_checkpointing: bool = False,
    optimizer: str = "adamw",
) -> PlanResult:
    """Inspect and plan a text training workload without loading weights."""
    model_spec = _resolve_model(model, revision, offline, cache_dir)
    hardware_spec = _resolve_hardware(hardware)
    requested = WorkloadSpec(
        kind=WorkloadKind.TRAIN,
        method=method,
        precision=precision,
        context_length=context_length,
        batch_size=batch_size,
        lora_rank=lora_rank,
        lora_layers=lora_layers,
        gradient_checkpointing=gradient_checkpointing,
        optimizer=optimizer,
    )
    if requested.method is not TrainingMethod.AUTO:
        estimate = estimate_memory(model_spec, hardware_spec, requested)
        return PlanResult(
            model=model_spec,
            hardware=hardware_spec,
            requested_workload=requested,
            recommended_workload=estimate.workload,
            estimate=estimate,
        )

    candidates = (
        replace(requested, method=TrainingMethod.FULL, precision=Precision.BF16),
        replace(requested, method=TrainingMethod.LORA, precision=Precision.BF16),
        replace(requested, method=TrainingMethod.QLORA, precision=Precision.INT4),
    )
    estimates = tuple(estimate_memory(model_spec, hardware_spec, item) for item in candidates)
    accepted = {FitStatus.COMFORTABLE, FitStatus.POSSIBLE}
    selected = next((item for item in estimates if item.fit in accepted), estimates[-1])
    alternatives = tuple(item for item in estimates if item is not selected)
    return PlanResult(
        model=model_spec,
        hardware=hardware_spec,
        requested_workload=requested,
        recommended_workload=selected.workload,
        estimate=selected,
        alternatives=alternatives,
    )


def format_plan_result(result: PlanResult) -> str:
    """Format the recommendation and uncertainty range for a terminal."""
    estimate = result.estimate
    workload = result.recommended_workload
    method = workload.method.value if workload.method else "inference"
    lines = [
        "mlx-one plan",
        f"Model: {result.model.model_id}@{result.model.revision}",
        f"Hardware: {result.hardware.profile_id}",
        f"Workload: {workload.kind.value} ({method})",
        f"Precision: {workload.precision.value}",
        f"Batch/context: {workload.batch_size}/{workload.context_length}",
        f"Estimated peak: {_range(estimate)}",
        f"Process budget: {format_bytes(estimate.budget_bytes)}",
        f"Fit: {estimate.fit.value}",
        f"Confidence: {estimate.confidence.value}",
    ]
    if result.requested_workload.method is TrainingMethod.AUTO:
        lines.append(f"Auto-selected method: {method}")
    if estimate.assumptions:
        lines.append("Assumptions:")
        lines.extend(f"  - {item}" for item in estimate.assumptions)
    if estimate.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {item}" for item in estimate.warnings)
    lines.append("Planning estimates are not compatibility guarantees.")
    return "\n".join(lines)


def write_plan_result(result: PlanResult, output: str | Path) -> Path:
    """Atomically write a plan artifact."""
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(f"{result.to_json()}\n", encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        if temporary.exists():
            temporary.unlink()
        raise PlanningError(f"could not write plan result to {path}: {exc}") from exc
    return path


def _resolve_model(
    model: str | Path | ModelSpec,
    revision: str | None,
    offline: bool,
    cache_dir: str | Path | None,
) -> ModelSpec:
    if isinstance(model, ModelSpec):
        return model
    return inspect_model(model, revision=revision, offline=offline, cache_dir=cache_dir).model


def _resolve_hardware(value: str | Path | HardwareSpec) -> HardwareSpec:
    if isinstance(value, HardwareSpec):
        return value
    return detect_hardware() if str(value) == "local" else load_hardware_profile(value)


def _resolve_precision(model: ModelSpec, workload: WorkloadSpec) -> Precision:
    if workload.precision is not Precision.AUTO:
        return workload.precision
    if workload.method is TrainingMethod.QLORA:
        return Precision.INT4
    if workload.kind is WorkloadKind.TRAIN:
        return Precision.BF16
    bits = model.quantization.get("bits")
    if bits == 4:
        return Precision.INT4
    if bits == 8:
        return Precision.INT8
    normalized = (model.dtype or "").lower()
    aliases = {
        "bf16": Precision.BF16,
        "f16": Precision.FP16,
        "fp16": Precision.FP16,
        "f32": Precision.FP32,
        "fp32": Precision.FP32,
    }
    return aliases.get(normalized, Precision.BF16)


def _weight_bytes(model: ModelSpec, precision: Precision) -> int | None:
    if model.parameter_count is not None and precision in _PRECISION_BYTES:
        return int(model.parameter_count * _PRECISION_BYTES[precision])
    return model.weight_bytes


def _kv_cache_bytes(model: ModelSpec, workload: WorkloadSpec, precision: Precision) -> int:
    dimensions = model.dimensions
    tokens = workload.context_length + workload.generation_length
    byte_width = 4 if precision is Precision.FP32 else 2
    if dimensions.layer_count and dimensions.key_value_heads and dimensions.head_dimension:
        return int(
            2
            * dimensions.layer_count
            * dimensions.key_value_heads
            * dimensions.head_dimension
            * tokens
            * workload.batch_size
            * byte_width
        )
    parameters = model.parameter_count or 0
    return int(max(parameters**0.5 * tokens * workload.batch_size * byte_width * 3, 64 * 1024**2))


def _activation_bytes(model: ModelSpec, workload: WorkloadSpec, *, training: bool) -> int:
    dimensions = model.dimensions
    if dimensions.hidden_size and dimensions.layer_count:
        base = (
            workload.batch_size
            * workload.context_length
            * dimensions.hidden_size
            * dimensions.layer_count
            * 2
        )
    else:
        parameters = model.parameter_count or 0
        scaled = parameters**0.5 * workload.context_length * workload.batch_size * 64
        base = int(max(scaled, 64 * 1024**2))
    factor = 10.0 if training else 0.35
    if training and workload.gradient_checkpointing:
        factor = 3.5
    return max(int(base * factor), 64 * 1024**2)


def _adapter_parameter_count(model: ModelSpec, workload: WorkloadSpec) -> int:
    dimensions = model.dimensions
    hidden = dimensions.hidden_size or max(int((model.parameter_count or 1) ** 0.5), 1)
    layers = min(workload.lora_layers, dimensions.layer_count or workload.lora_layers)
    return 8 * workload.lora_rank * hidden * layers


def _fit_status(lower: int, central: int, upper: int, budget: int | None) -> FitStatus:
    if budget is None:
        return FitStatus.UNKNOWN
    if upper <= budget * 0.8:
        return FitStatus.COMFORTABLE
    if upper <= budget:
        return FitStatus.POSSIBLE
    if central <= budget:
        return FitStatus.RISKY
    if lower > budget:
        return FitStatus.DOES_NOT_FIT
    return FitStatus.RISKY


def _unknown_estimate(
    workload: WorkloadSpec, hardware: HardwareSpec, warning: str
) -> MemoryEstimate:
    return MemoryEstimate(
        workload=workload,
        lower_bytes=None,
        central_bytes=None,
        upper_bytes=None,
        budget_bytes=hardware.process_memory_budget_bytes,
        fit=FitStatus.UNKNOWN,
        confidence=Confidence.LOW,
        warnings=(warning,),
    )


def _nearest_calibration(
    model: ModelSpec,
    hardware: HardwareSpec,
    workload: WorkloadSpec,
    analytic: int,
) -> dict[str, Any] | None:
    try:
        resource = resources.files("mlx_one").joinpath(
            "data", "calibrations", "m4-air-32gb-text-v1.json"
        )
        if not resource.is_file():
            return None
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if hardware.profile_id != payload.get("hardware_profile_id"):
        return None
    candidates = []
    for entry in payload.get("entries", []):
        if entry.get("model_type") != model.model_type:
            continue
        if entry.get("kind") != workload.kind.value:
            continue
        if entry.get("method") != (workload.method.value if workload.method else None):
            continue
        if entry.get("precision") != workload.precision.value:
            continue
        if not entry.get("measured_peak_bytes") or not entry.get("analytic_peak_bytes"):
            continue
        distance = abs(entry.get("context_length", 0) - workload.context_length)
        distance += (
            abs(entry.get("parameter_count", 0) - (model.parameter_count or 0))
            / 1_000_000
        )
        candidates.append((distance, entry))
    if not candidates:
        return None
    entry = min(candidates, key=lambda item: item[0])[1]
    exact_size_entries = [
        item[1] for item in candidates if item[1].get("parameter_count") == model.parameter_count
    ]
    exact_context = next(
        (
            item
            for item in exact_size_entries
            if item.get("context_length") == workload.context_length
        ),
        None,
    )
    if exact_context is not None:
        central = exact_context["measured_peak_bytes"]
        entry = exact_context
    elif len(exact_size_entries) >= 2:
        central = int(_linear_calibration_prediction(exact_size_entries, workload.context_length))
    else:
        ratio = entry["measured_peak_bytes"] / entry["analytic_peak_bytes"]
        central = int(analytic * ratio)
    validation = payload.get("validation", {})
    error = max(float(validation.get("p90_relative_error", 0.30)), 0.30)
    exact_size = bool(exact_size_entries)
    return {
        "central_bytes": central,
        "spread": (max(1.0 - error, 0.5), 1.0 + error),
        "reference": entry.get("record", payload.get("matrix_id", "m4-air-32gb-text-v1")),
        "confidence": Confidence.HIGH if exact_size else Confidence.MEDIUM,
    }


def _linear_calibration_prediction(entries: list[dict[str, Any]], context: int) -> float:
    xs = [float(item["context_length"]) for item in entries]
    ys = [float(item["measured_peak_bytes"]) for item in entries]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        return y_mean
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    return max(y_mean + numerator / denominator * (context - x_mean), 1.0)


def _range(estimate: MemoryEstimate) -> str:
    if estimate.central_bytes is None:
        return "unknown"
    lower = format_bytes(estimate.lower_bytes)
    upper = format_bytes(estimate.upper_bytes)
    central = format_bytes(estimate.central_bytes)
    return f"{lower}–{upper} (central {central})"
