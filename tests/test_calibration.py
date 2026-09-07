import json
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest

from mlx_one.calibration import (
    CalibrationError,
    expand_calibration_matrix,
    load_calibration_matrix,
    run_calibration,
)
from mlx_one.hardware import load_hardware_profile
from mlx_one.schemas import (
    CalibrationRecord,
    CalibrationStatus,
    Confidence,
    FitStatus,
    MemoryEstimate,
    SoftwareProvenance,
)


def test_builtin_matrix_is_pinned_and_expands_full_matrix() -> None:
    matrix = load_calibration_matrix("m4-air-32gb-text-v1")
    cells = expand_calibration_matrix(matrix)

    assert len(cells) == 48
    assert len({cell.cell_id for cell in cells}) == 48
    assert all(len(cell.model.revision) == 40 for cell in cells)
    assert sum(cell.workload.kind.value == "inference" for cell in cells) == 24
    assert sum(cell.workload.kind.value == "train" for cell in cells) == 24


def test_matrix_rejects_mutable_revisions(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        """schema_version: '1.0'
matrix_id: bad
hardware_profile: m4-air-32gb
contexts: [512]
inference: {}
training: {}
models:
  - model_id: example/model
    revision: main
    model_type: example
    parameter_count: 1
    dimensions: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(CalibrationError, match="immutable"):
        load_calibration_matrix(matrix)


def test_calibration_dry_run_has_no_metal_or_filesystem_side_effects(tmp_path: Path) -> None:
    output = tmp_path / "results"

    summary = run_calibration(
        "m4-air-32gb-text-v1",
        output=output,
        artifacts_dir=tmp_path / "models",
        dry_run=True,
        select=r"0\.5b-inference-bf16-c512$",
    )

    assert summary["cell_count"] == 1
    assert not output.exists()
    assert not (tmp_path / "models").exists()


def test_worker_module_does_not_import_mlx() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; import mlx_one.calibration_worker; print('mlx' in sys.modules)",
    ]

    completed = subprocess.run(command, capture_output=True, check=True, text=True)

    assert completed.stdout.strip() == "False"


def test_invalid_existing_resume_record_fails_before_overwrite(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "results"
    output.mkdir()
    record = output / "qwen2.5-0.5b-inference-bf16-c512.json"
    record.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(
        "mlx_one.calibration.collect_doctor_report",
        lambda: type("Doctor", (), {"metal_available": True})(),
    )
    monkeypatch.setattr(
        "mlx_one.calibration.detect_hardware",
        lambda: type(
            "Hardware",
            (),
            {"chip": "Apple M4", "memory_bytes": 32 * 1024**3},
        )(),
    )

    with pytest.raises(CalibrationError, match="invalid existing calibration record"):
        run_calibration(
            "m4-air-32gb-text-v1",
            output=output,
            artifacts_dir=tmp_path / "models",
            resume=True,
            select=r"0\.5b-inference-bf16-c512$",
        )

    assert record.read_text(encoding="utf-8") == "{broken"


def test_result_files_do_not_contain_private_paths(tmp_path: Path) -> None:
    payload = json.dumps({"model": "Qwen/Qwen2.5-0.5B", "status": "completed"})

    assert str(Path.home()) not in payload


def test_packaged_m4_measurements_meet_validation_gates() -> None:
    resource = resources.files("mlx_one").joinpath(
        "data", "calibrations", "m4-air-32gb-text-v1.json"
    )
    summary = json.loads(resource.read_text(encoding="utf-8"))
    validation = summary["validation"]

    assert len(summary["entries"]) == 44
    assert validation["median_relative_error"] <= 0.20
    assert validation["p90_relative_error"] <= 0.30
    assert validation["upper_bound_coverage"] >= 0.95


def test_fake_cell_run_writes_and_resumes_atomic_evidence(tmp_path: Path, monkeypatch) -> None:
    profile = load_hardware_profile("m4-air-32gb")
    monkeypatch.setattr(
        "mlx_one.calibration.collect_doctor_report",
        lambda: type("Doctor", (), {"metal_available": True})(),
    )
    monkeypatch.setattr("mlx_one.calibration.detect_hardware", lambda: profile)
    monkeypatch.setattr("mlx_one.calibration._prepare_artifact", lambda *_args: tmp_path)
    calls = []

    def fake_run(cell, hardware, _model_path, _output):
        calls.append(cell.cell_id)
        return CalibrationRecord(
            matrix_id=cell.matrix_id,
            cell_id=cell.cell_id,
            status=CalibrationStatus.COMPLETED,
            model=cell.model,
            hardware=hardware,
            workload=cell.workload,
            software=SoftwareProvenance(python_version="3.12", packages={}),
            metrics={"wall_seconds": 1.0},
            memory={"peak_metal_bytes": 100, "analytic_peak_bytes": 110},
        )

    monkeypatch.setattr("mlx_one.calibration._run_cell", fake_run)
    output = tmp_path / "results"
    arguments = {
        "output": output,
        "artifacts_dir": tmp_path / "models",
        "select": r"0\.5b-inference-bf16-c512$",
    }

    run_calibration("m4-air-32gb-text-v1", **arguments)
    run_calibration("m4-air-32gb-text-v1", **arguments, resume=True)

    assert calls == ["qwen2.5-0.5b-inference-bf16-c512"]
    assert json.loads((output / "manifest.json").read_text())["records"][0][
        "status"
    ] == "completed"


def test_preflight_skip_does_not_prepare_weights(tmp_path: Path, monkeypatch) -> None:
    profile = load_hardware_profile("m4-air-32gb")
    monkeypatch.setattr(
        "mlx_one.calibration.collect_doctor_report",
        lambda: type("Doctor", (), {"metal_available": True})(),
    )
    monkeypatch.setattr("mlx_one.calibration.detect_hardware", lambda: profile)

    def over_budget(_model, _hardware, workload):
        return MemoryEstimate(
            workload=workload,
            lower_bytes=25 * 1024**3,
            central_bytes=26 * 1024**3,
            upper_bytes=27 * 1024**3,
            budget_bytes=24 * 1024**3,
            fit=FitStatus.DOES_NOT_FIT,
            confidence=Confidence.LOW,
        )

    monkeypatch.setattr("mlx_one.calibration.estimate_memory", over_budget)
    monkeypatch.setattr(
        "mlx_one.calibration._prepare_artifact",
        lambda *_args: pytest.fail("over-budget cells must not prepare weights"),
    )
    output = tmp_path / "results"

    run_calibration(
        "m4-air-32gb-text-v1",
        output=output,
        artifacts_dir=tmp_path / "models",
        select=r"0\.5b-inference-bf16-c512$",
    )

    record = json.loads(next(output.glob("qwen*.json")).read_text())
    assert record["status"] == "skipped-over-budget"
