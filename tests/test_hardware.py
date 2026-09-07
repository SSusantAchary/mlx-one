from pathlib import Path

import pytest

from mlx_one import HardwareSpec
from mlx_one.diagnostics import DoctorReport
from mlx_one.hardware import (
    HardwareProfileError,
    detect_hardware,
    list_hardware_profiles,
    load_hardware_profile,
)


def test_builtin_m4_profile_has_safe_budget() -> None:
    profile = load_hardware_profile("m4-air-32gb")

    assert "m4-air-32gb" in list_hardware_profiles()
    assert profile.memory_bytes == 32 * 1024**3
    assert profile.process_memory_budget_bytes == 24 * 1024**3
    assert profile.system_reserve_bytes == 8 * 1024**3
    assert profile.unified_memory is True
    assert profile.reference_device is True


def test_custom_profile_validation(tmp_path: Path) -> None:
    profile = tmp_path / "custom.yaml"
    profile.write_text(
        """schema_version: '1.0'
profile_id: custom
platform: macOS
architecture: arm64
chip: Apple M9
memory_bytes: 100
process_memory_budget_bytes: 90
system_reserve_bytes: 20
""",
        encoding="utf-8",
    )

    with pytest.raises(HardwareProfileError, match="budget plus system reserve"):
        load_hardware_profile(profile)


def test_missing_and_malformed_profiles_fail(tmp_path: Path) -> None:
    with pytest.raises(HardwareProfileError, match="does not exist"):
        load_hardware_profile(tmp_path / "missing.yaml")
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("[", encoding="utf-8")
    with pytest.raises(HardwareProfileError, match="invalid hardware profile"):
        load_hardware_profile(malformed)


def test_detect_matches_reference_without_private_identifiers(monkeypatch) -> None:
    report = DoctorReport(
        supported_host=True,
        operating_system="Darwin",
        architecture="arm64",
        chip="Apple M4",
        memory_bytes=32 * 1024**3,
        python_version="3.12",
        metal_available=True,
        metal_error=None,
        backends=(),
    )
    monkeypatch.setattr("mlx_one.hardware.collect_doctor_report", lambda: report)

    detected = detect_hardware()
    serialized = detected.to_json().lower()

    assert detected.profile_id == "m4-air-32gb"
    assert detected.provenance == "detected-from-doctor-and-built-in-profile"
    for private_field in ("serial", "uuid", "username", "home"):
        assert private_field not in serialized


def test_detect_unknown_host_uses_quarter_reserve(monkeypatch) -> None:
    report = DoctorReport(
        supported_host=False,
        operating_system="Linux",
        architecture="x86_64",
        chip="Example CPU",
        memory_bytes=16 * 1024**3,
        python_version="3.12",
        metal_available=False,
        metal_error="not available",
        backends=(),
    )
    monkeypatch.setattr("mlx_one.hardware.collect_doctor_report", lambda: report)

    detected = detect_hardware()

    assert detected.profile_id == "local"
    assert detected.system_reserve_bytes == 4 * 1024**3
    assert detected.process_memory_budget_bytes == 12 * 1024**3
    assert detected.accelerator is None


def test_hardware_spec_rejects_impossible_budget() -> None:
    with pytest.raises(ValueError, match="cannot exceed memory_bytes"):
        HardwareSpec(
            profile_id="invalid",
            platform="macOS",
            architecture="arm64",
            chip="Apple M4",
            memory_bytes=10,
            process_memory_budget_bytes=11,
        )
