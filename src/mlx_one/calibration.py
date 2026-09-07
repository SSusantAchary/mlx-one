"""Resumable, safety-bounded text calibration orchestration."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata, resources
from pathlib import Path
from typing import Any

import yaml

from mlx_one.diagnostics import collect_doctor_report
from mlx_one.hardware import detect_hardware, load_hardware_profile
from mlx_one.planning import estimate_memory
from mlx_one.schemas import (
    CalibrationRecord,
    CalibrationStatus,
    Failure,
    Modality,
    ModelDimensions,
    ModelSpec,
    Precision,
    SoftwareProvenance,
    WorkloadKind,
    WorkloadSpec,
)

GIB = 1024**3
MAX_SWAP_DELTA_BYTES = 512 * 1024**2


class CalibrationError(RuntimeError):
    """An actionable calibration setup or execution failure."""


@dataclass(frozen=True)
class CalibrationCell:
    """One deterministic cell expanded from a calibration matrix."""

    matrix_id: str
    cell_id: str
    model: ModelSpec
    workload: WorkloadSpec
    warm_repetitions: int = 3
    warmup_steps: int = 1
    measured_steps: int = 5
    lora_scale: float = 20.0
    lora_dropout: float = 0.0


def load_calibration_matrix(name_or_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a built-in or custom calibration matrix."""
    path = Path(name_or_path).expanduser()
    if path.exists():
        source = str(path)
        text = path.read_text(encoding="utf-8")
    elif path.suffix.lower() in {".yaml", ".yml"} or "/" in str(name_or_path):
        raise CalibrationError(f"calibration matrix does not exist: {path}")
    else:
        resource = resources.files("mlx_one").joinpath(
            "data", "calibrations", f"{name_or_path}.matrix.yaml"
        )
        if not resource.is_file():
            raise CalibrationError(f"unknown calibration matrix {str(name_or_path)!r}")
        source = str(name_or_path)
        text = resource.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CalibrationError(f"invalid calibration matrix {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CalibrationError(f"calibration matrix {source} must contain a YAML object")
    if payload.get("schema_version") != "1.0":
        raise CalibrationError(f"calibration matrix {source} requires schema_version '1.0'")
    for key in ("matrix_id", "hardware_profile", "contexts", "models"):
        if not payload.get(key):
            raise CalibrationError(f"calibration matrix {source} is missing {key}")
    if not isinstance(payload["contexts"], list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 1
        for item in payload["contexts"]
    ):
        raise CalibrationError("calibration contexts must be positive integers")
    if not isinstance(payload["models"], list) or any(
        not isinstance(item, dict) for item in payload["models"]
    ):
        raise CalibrationError("calibration models must be a list of objects")
    for section in ("inference", "training"):
        if not isinstance(payload.get(section), dict):
            raise CalibrationError(f"calibration matrix {source} is missing {section}")
    required_model_fields = {
        "model_id",
        "revision",
        "model_type",
        "parameter_count",
        "dimensions",
    }
    if any(not required_model_fields <= set(item) for item in payload["models"]):
        raise CalibrationError("each calibration model requires identity and dimensions")
    revisions = [item.get("revision") for item in payload["models"]]
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value)
        for value in revisions
    ):
        raise CalibrationError(
            "calibration model revisions must be immutable 40-character commit SHAs"
        )
    return payload


def expand_calibration_matrix(matrix: dict[str, Any]) -> tuple[CalibrationCell, ...]:
    """Expand the configured inference and PEFT Cartesian matrix."""
    cells: list[CalibrationCell] = []
    inference = matrix["inference"]
    training = matrix["training"]
    for model_data in matrix["models"]:
        model = _matrix_model(model_data)
        short_name = model.model_id.rsplit("/", 1)[-1].lower()
        for context in matrix["contexts"]:
            for precision in inference["precisions"]:
                workload = WorkloadSpec(
                    kind=WorkloadKind.INFERENCE,
                    precision=precision,
                    context_length=context,
                    batch_size=1,
                    generation_length=inference["generation_length"],
                )
                cells.append(
                    CalibrationCell(
                        matrix_id=matrix["matrix_id"],
                        cell_id=f"{short_name}-inference-{precision}-c{context}",
                        model=model,
                        workload=workload,
                        warm_repetitions=inference["warm_repetitions"],
                    )
                )
            for method in training["methods"]:
                precision = Precision.INT4 if method == "qlora" else Precision.BF16
                workload = WorkloadSpec(
                    kind=WorkloadKind.TRAIN,
                    method=method,
                    precision=precision,
                    context_length=context,
                    batch_size=training["batch_size"],
                    lora_rank=training["lora_rank"],
                    lora_layers=training["lora_layers"],
                    gradient_checkpointing=training["gradient_checkpointing"],
                    optimizer=training["optimizer"],
                )
                cells.append(
                    CalibrationCell(
                        matrix_id=matrix["matrix_id"],
                        cell_id=f"{short_name}-{method}-c{context}",
                        model=model,
                        workload=workload,
                        warmup_steps=training["warmup_steps"],
                        measured_steps=training["measured_steps"],
                        lora_scale=training["lora_scale"],
                        lora_dropout=training["lora_dropout"],
                    )
                )
    return tuple(cells)


def run_calibration(
    name_or_path: str | Path,
    *,
    output: str | Path,
    artifacts_dir: str | Path,
    resume: bool = False,
    retry: bool = False,
    dry_run: bool = False,
    select: str | None = None,
) -> dict[str, Any]:
    """Execute selected matrix cells and write sanitized atomic result records."""
    matrix = load_calibration_matrix(name_or_path)
    profile = load_hardware_profile(matrix["hardware_profile"])
    cells = expand_calibration_matrix(matrix)
    if select:
        try:
            pattern = re.compile(select)
        except re.error as exc:
            raise CalibrationError(f"invalid --select regular expression: {exc}") from exc
        cells = tuple(cell for cell in cells if pattern.search(cell.cell_id))
    if dry_run:
        return {
            "matrix_id": matrix["matrix_id"],
            "hardware_profile": profile.profile_id,
            "cell_count": len(cells),
            "cells": [cell.cell_id for cell in cells],
        }

    doctor = collect_doctor_report()
    local = detect_hardware()
    if not doctor.metal_available:
        raise CalibrationError(
            "MLX Metal is unavailable; run calibration in a local macOS terminal"
        )
    if local.chip.lower() != profile.chip.lower() or local.memory_bytes != profile.memory_bytes:
        raise CalibrationError(
            f"matrix requires {profile.chip} with {profile.memory_bytes / GIB:.0f} GiB; "
            f"detected {local.chip} with {_gib(local.memory_bytes)}"
        )

    output_path = Path(output).expanduser().resolve()
    artifact_path = Path(artifacts_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)
    records: list[CalibrationRecord] = []
    for cell in cells:
        record_path = output_path / f"{cell.cell_id}.json"
        existing = _existing_record(record_path)
        if existing is not None and resume:
            if not retry or existing.status not in {
                CalibrationStatus.FAILED,
                CalibrationStatus.INVALID,
                CalibrationStatus.INTERRUPTED,
            }:
                records.append(existing)
                continue
        estimate = estimate_memory(cell.model, profile, cell.workload)
        over_budget = (
            estimate.lower_bytes is not None
            and estimate.lower_bytes > profile.process_memory_budget_bytes
        )
        if over_budget:
            record = _skipped_record(cell, profile, estimate.lower_bytes)
        else:
            model_path = _prepare_artifact(cell, artifact_path)
            record = _run_cell(cell, profile, model_path, output_path)
        _atomic_json(record_path, record.to_dict())
        records.append(record)
    all_records = _records_in_directory(output_path)
    summary = summarize_calibration(
        all_records,
        matrix_id=matrix["matrix_id"],
        profile_id=profile.profile_id,
    )
    _atomic_json(output_path / "summary.json", summary)
    _atomic_json(output_path / "manifest.json", _result_manifest(all_records, summary))
    return summary


def summarize_calibration(
    records: list[CalibrationRecord] | tuple[CalibrationRecord, ...],
    *,
    matrix_id: str,
    profile_id: str,
) -> dict[str, Any]:
    """Build the compact, runtime-consumable calibration summary."""
    entries = []
    for record in records:
        peak = record.memory.get("peak_metal_bytes")
        analytic = record.memory.get("analytic_peak_bytes")
        if record.status is not CalibrationStatus.COMPLETED or not isinstance(peak, int):
            continue
        entries.append(
            {
                "record": f"{record.cell_id}.json",
                "model_type": record.model.model_type,
                "parameter_count": record.model.parameter_count,
                "kind": record.workload.kind.value,
                "method": record.workload.method.value if record.workload.method else None,
                "precision": record.workload.precision.value,
                "context_length": record.workload.context_length,
                "measured_peak_bytes": peak,
                "analytic_peak_bytes": analytic,
                "p90_relative_error": 0.30,
            }
        )
    validation = _cross_validate(entries)
    calibrated_error = max(validation.get("p90_relative_error", 0.30), 0.30)
    for entry in entries:
        entry["p90_relative_error"] = calibrated_error
    return {
        "schema_version": "1.0",
        "matrix_id": matrix_id,
        "hardware_profile_id": profile_id,
        "generated_at": _now(),
        "validation": validation,
        "entries": entries,
    }


def _cross_validate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        key = (
            entry["parameter_count"],
            entry["kind"],
            entry["method"],
            entry["precision"],
        )
        groups[key].append(entry)
    errors = []
    covered = []
    for group in groups.values():
        if len(group) < 3:
            continue
        for held_out in group:
            training = [item for item in group if item is not held_out]
            predicted = _linear_prediction(training, held_out["context_length"])
            measured = held_out["measured_peak_bytes"]
            errors.append(abs(predicted - measured) / measured)
            covered.append(measured <= predicted * 1.30)
    if not errors:
        return {
            "method": "leave-one-context-out-linear",
            "sample_count": 0,
            "median_relative_error": None,
            "p90_relative_error": 0.30,
            "upper_bound_coverage": None,
        }
    ordered = sorted(errors)
    p90_index = int(0.9 * (len(ordered) - 1))
    return {
        "method": "leave-one-context-out-linear",
        "sample_count": len(errors),
        "median_relative_error": sorted(errors)[len(errors) // 2],
        "p90_relative_error": ordered[p90_index],
        "upper_bound_coverage": sum(covered) / len(covered),
    }


def _linear_prediction(entries: list[dict[str, Any]], context_length: int) -> float:
    xs = [float(item["context_length"]) for item in entries]
    ys = [float(item["measured_peak_bytes"]) for item in entries]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        return y_mean
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    slope /= denominator
    return max(y_mean + slope * (context_length - x_mean), 1.0)


def _prepare_artifact(cell: CalibrationCell, root: Path) -> Path:
    precision = cell.workload.precision
    destination = (
        root
        / cell.model.model_id.replace("/", "--")
        / cell.model.revision
        / precision.value
    )
    if (destination / "config.json").is_file():
        return destination
    if destination.exists():
        raise CalibrationError(
            f"incomplete calibration artifact exists at {destination}; move it aside and retry"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="convert-", dir=destination.parent) as temporary:
        converted = Path(temporary) / "model"
        request_path = Path(temporary) / "conversion.json"
        request_path.write_text(
            json.dumps(
                {
                    "model_id": cell.model.model_id,
                    "revision": cell.model.revision,
                    "destination": str(converted),
                    "quantize": precision is Precision.INT4,
                }
            ),
            encoding="utf-8",
        )
        command = [sys.executable, "-m", "mlx_one.conversion_worker", str(request_path)]
        completed = subprocess.run(command, check=False, text=True)
        if completed.returncode != 0:
            raise CalibrationError(
                f"mlx-lm conversion failed for {cell.model.model_id} ({precision.value})"
            )
        converted.replace(destination)
    return destination


def _run_cell(
    cell: CalibrationCell, profile: Any, model_path: Path, output: Path
) -> CalibrationRecord:
    started = _now()
    estimate = estimate_memory(cell.model, profile, cell.workload)
    before_swap = _swap_used_bytes()
    with tempfile.TemporaryDirectory(prefix=f"{cell.cell_id}-", dir=output) as scratch:
        request = {
            "model_path": str(model_path),
            "workload": cell.workload.to_dict(),
            "memory_limit_bytes": profile.process_memory_budget_bytes,
            "warm_repetitions": cell.warm_repetitions,
            "warmup_steps": cell.warmup_steps,
            "measured_steps": cell.measured_steps,
            "lora_scale": cell.lora_scale,
            "lora_dropout": cell.lora_dropout,
            "scratch_dir": scratch,
            "result_path": str(Path(scratch) / "worker-result.json"),
        }
        request_path = Path(scratch) / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        completed, safety_failure = _run_monitored_worker(
            [sys.executable, "-m", "mlx_one.calibration_worker", str(request_path)],
            baseline_swap=before_swap,
        )
        result_path = Path(request["result_path"])
        try:
            worker = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            worker = {}
    after_swap = _swap_used_bytes()
    swap_delta = max((after_swap or 0) - (before_swap or 0), 0)
    status = CalibrationStatus.COMPLETED
    failure = None
    if safety_failure is not None:
        status = CalibrationStatus.INVALID
        failure = Failure(code=safety_failure, message=_safety_message(safety_failure))
    elif completed.returncode != 0 or not worker or worker.get("error"):
        status = CalibrationStatus.FAILED
        failure = Failure(
            code="worker-failed",
            message="Calibration worker failed; rerun the cell locally for diagnostics.",
        )
    elif swap_delta > MAX_SWAP_DELTA_BYTES:
        status = CalibrationStatus.INVALID
        failure = Failure(
            code="excessive-swap",
            message="Swap grew by more than 512 MiB; this cell is not evidence of fit.",
        )
    memory = dict(worker.get("memory", {}))
    memory.update(analytic_peak_bytes=estimate.central_bytes, swap_delta_bytes=swap_delta)
    return CalibrationRecord(
        matrix_id=cell.matrix_id,
        cell_id=cell.cell_id,
        status=status,
        model=cell.model,
        hardware=profile,
        workload=cell.workload,
        software=_software_provenance(),
        metrics=worker.get("metrics", {}),
        memory=memory,
        environment={"thermal_state": _thermal_state()},
        failure=failure,
        started_at=started,
        ended_at=_now(),
    )


def _skipped_record(cell: CalibrationCell, profile: Any, lower_bytes: int) -> CalibrationRecord:
    timestamp = _now()
    return CalibrationRecord(
        matrix_id=cell.matrix_id,
        cell_id=cell.cell_id,
        status=CalibrationStatus.SKIPPED_OVER_BUDGET,
        model=cell.model,
        hardware=profile,
        workload=cell.workload,
        software=_software_provenance(),
        memory={"analytic_lower_bytes": lower_bytes},
        failure=Failure(
            code="preflight-over-budget",
            message="Analytic lower estimate exceeds the configured process-memory budget.",
        ),
        started_at=timestamp,
        ended_at=timestamp,
    )


def _matrix_model(payload: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        model_id=payload["model_id"],
        revision=payload["revision"],
        modality=Modality.TEXT,
        architectures=("Qwen2ForCausalLM",),
        model_type=payload["model_type"],
        parameter_count=payload["parameter_count"],
        parameter_count_source="calibration-matrix",
        dimensions=ModelDimensions.from_dict(payload["dimensions"]),
    )


def _existing_record(path: Path) -> CalibrationRecord | None:
    if not path.is_file():
        return None
    try:
        return CalibrationRecord.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"invalid existing calibration record {path.name}: {exc}") from exc


def _records_in_directory(path: Path) -> list[CalibrationRecord]:
    records = []
    for candidate in sorted(path.glob("*.json")):
        if candidate.name in {"manifest.json", "summary.json"}:
            continue
        record = _existing_record(candidate)
        if record is not None:
            records.append(record)
    return records


def _software_provenance() -> SoftwareProvenance:
    packages = {}
    for package in ("mlx-one", "mlx", "mlx-lm", "transformers", "huggingface-hub"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return SoftwareProvenance(
        python_version=platform.python_version(),
        packages=packages,
        os_version=platform.mac_ver()[0] or platform.platform(),
    )


def _result_manifest(records: list[CalibrationRecord], summary: dict[str, Any]) -> dict[str, Any]:
    items = []
    for record in records:
        serialized = f"{record.to_json()}\n".encode()
        items.append(
            {
                "cell_id": record.cell_id,
                "status": record.status.value,
                "sha256": hashlib.sha256(serialized).hexdigest(),
            }
        )
    return {**summary, "records": items}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _swap_used_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"used = ([0-9.]+)([MGT])", result.stdout)
    if not match:
        return None
    multiplier = {"M": 1024**2, "G": 1024**3, "T": 1024**4}[match.group(2)]
    return int(float(match.group(1)) * multiplier)


def _thermal_state() -> str:
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    text = result.stdout.lower()
    if "failed to get" in text or "error:" in text:
        return "unknown"
    return "warning" if "warning" in text and "no warning" not in text else "nominal"


def _run_monitored_worker(
    command: list[str], *, baseline_swap: int | None
) -> tuple[subprocess.CompletedProcess[str], str | None]:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    safety_failure = None
    while process.poll() is None:
        swap = _swap_used_bytes()
        if baseline_swap is not None and swap is not None:
            if swap - baseline_swap > MAX_SWAP_DELTA_BYTES:
                safety_failure = "excessive-swap"
        free_percent = _memory_pressure_free_percent()
        if free_percent is not None and free_percent < 5:
            safety_failure = "critical-memory-pressure"
        if safety_failure:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        time.sleep(0.5)
    stdout, stderr = process.communicate()
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    return completed, safety_failure


def _memory_pressure_free_percent() -> int | None:
    try:
        result = subprocess.run(
            ["memory_pressure", "-Q"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"System-wide memory free percentage: ([0-9]+)%", result.stdout)
    return int(match.group(1)) if match else None


def _safety_message(code: str) -> str:
    if code == "critical-memory-pressure":
        return "System memory pressure became critical; the worker was terminated."
    return "Swap grew by more than 512 MiB; the worker was terminated."


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _gib(value: int | None) -> str:
    return "unknown memory" if value is None else f"{value / GIB:.0f} GiB"
