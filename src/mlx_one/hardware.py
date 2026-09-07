"""Hardware profile discovery and privacy-safe local detection."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from mlx_one.diagnostics import collect_doctor_report
from mlx_one.schemas import HardwareSpec


class HardwareProfileError(RuntimeError):
    """An actionable hardware profile or discovery failure."""


def list_hardware_profiles() -> tuple[str, ...]:
    """Return the identifiers of hardware profiles bundled with mlx-one."""
    root = resources.files("mlx_one").joinpath("data", "hardware")
    names = (
        item.name.removesuffix(".yaml")
        for item in root.iterdir()
        if item.name.endswith(".yaml")
    )
    return tuple(sorted(name for name in names if name))


def load_hardware_profile(profile_or_path: str | Path) -> HardwareSpec:
    """Load a built-in profile identifier or a custom YAML profile."""
    requested = str(profile_or_path)
    path = Path(profile_or_path).expanduser()
    if path.exists():
        if not path.is_file():
            raise HardwareProfileError(f"hardware profile must be a YAML file: {path}")
        return _load_yaml(path.read_text(encoding="utf-8"), source=str(path))
    if path.suffix.lower() in {".yaml", ".yml"} or "/" in requested:
        raise HardwareProfileError(f"hardware profile does not exist: {path}")
    resource = resources.files("mlx_one").joinpath("data", "hardware", f"{requested}.yaml")
    if not resource.is_file():
        available = ", ".join(list_hardware_profiles()) or "none"
        raise HardwareProfileError(
            f"unknown hardware profile {requested!r}; available profiles: {available}"
        )
    return _load_yaml(resource.read_text(encoding="utf-8"), source=requested)


def detect_hardware() -> HardwareSpec:
    """Detect the current host without returning stable machine identifiers."""
    report = collect_doctor_report()
    matched = _matching_profile(report.chip, report.memory_bytes)
    if matched is not None:
        return HardwareSpec.from_dict(
            {
                **matched.to_dict(),
                "os_version": report.operating_system,
                "provenance": "detected-from-doctor-and-built-in-profile",
            }
        )
    reserve = min(8 * 1024**3, report.memory_bytes // 4) if report.memory_bytes else None
    budget = report.memory_bytes - reserve if report.memory_bytes and reserve is not None else None
    return HardwareSpec(
        profile_id="local",
        platform=report.operating_system,
        architecture=report.architecture,
        chip=report.chip,
        memory_bytes=report.memory_bytes,
        accelerator="Metal" if report.metal_available else None,
        accelerator_memory_bytes=report.memory_bytes if report.metal_available else None,
        process_memory_budget_bytes=budget,
        system_reserve_bytes=reserve,
        unified_memory=report.operating_system == "Darwin" and report.architecture == "arm64",
        os_version=report.operating_system,
        provenance="detected-from-doctor",
    )


def format_hardware_profile(profile: HardwareSpec) -> str:
    """Format a hardware profile for terminal presentation."""
    return "\n".join(
        [
            "mlx-one hardware",
            f"Profile: {profile.profile_id} (version {profile.profile_version})",
            f"Host: {profile.platform} {profile.architecture}",
            f"Chip: {profile.chip}",
            f"Form factor: {profile.form_factor or 'unknown'}",
            f"Unified memory: {_format_bytes(profile.memory_bytes)}",
            f"Process budget: {_format_bytes(profile.process_memory_budget_bytes)}",
            f"System reserve: {_format_bytes(profile.system_reserve_bytes)}",
            "CPU/GPU cores: "
            f"{profile.cpu_core_count or 'unknown'}/{profile.gpu_core_count or 'unknown'}",
            f"Accelerator: {profile.accelerator or 'unknown'}",
            f"Reference device: {'yes' if profile.reference_device else 'no'}",
            f"Provenance: {profile.provenance}",
        ]
    )


def _load_yaml(value: str, *, source: str) -> HardwareSpec:
    try:
        payload = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise HardwareProfileError(f"invalid hardware profile {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HardwareProfileError(f"hardware profile {source} must contain a YAML object")
    try:
        return HardwareSpec.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise HardwareProfileError(f"invalid hardware profile {source}: {exc}") from exc


def _matching_profile(chip: str, memory_bytes: int | None) -> HardwareSpec | None:
    if memory_bytes is None:
        return None
    for profile_id in list_hardware_profiles():
        profile = load_hardware_profile(profile_id)
        tolerance = 512 * 1024**2
        memory_matches = abs(profile.memory_bytes - memory_bytes) <= tolerance
        if profile.chip.lower() == chip.lower() and memory_matches:
            return profile
    return None


def _format_bytes(value: int | None) -> str:
    return "unknown" if value is None else f"{value / 1024**3:.1f} GiB"
