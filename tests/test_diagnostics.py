from mlx_one import diagnostics
from mlx_one.diagnostics import (
    BackendStatus,
    DoctorReport,
    doctor_report_json,
    format_doctor_report,
)


def test_format_doctor_report_includes_action_for_unsupported_host() -> None:
    report = DoctorReport(
        supported_host=False,
        operating_system="Linux",
        architecture="x86_64",
        chip="Example CPU",
        memory_bytes=None,
        python_version="3.12.1",
        metal_available=False,
        metal_error="unavailable",
        backends=(BackendStatus("mlx", False, None),),
    )

    output = format_doctor_report(report)

    assert "Unified memory: unknown" in output
    assert "mlx: not installed" in output
    assert "requires macOS on Apple Silicon" in output


def test_doctor_json_is_stable_and_serializable() -> None:
    report = DoctorReport(
        supported_host=True,
        operating_system="Darwin",
        architecture="arm64",
        chip="Apple M4",
        memory_bytes=16 * 1024**3,
        python_version="3.13.0",
        metal_available=True,
        metal_error=None,
        backends=(BackendStatus("mlx", True, "1.0"),),
    )

    output = doctor_report_json(report)

    assert '"chip": "Apple M4"' in output
    assert '"name": "mlx"' in output


def test_collect_report_uses_hardware_profile_fallback(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(diagnostics.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(diagnostics, "_sysctl", lambda _name: None)
    monkeypatch.setattr(
        diagnostics,
        "_hardware_profile",
        lambda: {"chip_type": "Apple M4", "physical_memory": "32 GB"},
    )
    monkeypatch.setattr(diagnostics, "_probe_metal", lambda: (True, None))
    monkeypatch.setattr(
        diagnostics,
        "_backend_status",
        lambda name, _module: BackendStatus(name, False, None),
    )

    report = diagnostics.collect_doctor_report()

    assert report.chip == "Apple M4"
    assert report.memory_bytes == 32 * 1024**3
