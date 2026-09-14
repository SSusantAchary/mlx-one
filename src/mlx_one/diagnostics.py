"""Host and optional-backend diagnostics for MLX One."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata, util
from typing import Any

from mlx_one.core.compatibility import MLXCompatibility, get_mlx_compatibility


@dataclass(frozen=True)
class BackendStatus:
    """Availability and version information for an optional backend."""

    name: str
    installed: bool
    version: str | None


@dataclass(frozen=True)
class DoctorReport:
    """A serializable description of the current MLX One host."""

    supported_host: bool
    operating_system: str
    architecture: str
    chip: str
    memory_bytes: int | None
    python_version: str
    metal_available: bool
    metal_error: str | None
    backends: tuple[BackendStatus, ...]
    mlx_compatibility: MLXCompatibility | None = None
    native_capabilities: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-compatible dictionary."""
        payload = asdict(self)
        if self.mlx_compatibility is not None:
            payload["mlx_compatibility"] = self.mlx_compatibility.to_dict()
        return payload


_BACKENDS = (("mlx", "mlx"),)


def collect_doctor_report() -> DoctorReport:
    """Collect host details without importing MLX in the current process."""
    operating_system = platform.system()
    architecture = platform.machine()
    hardware = _hardware_profile() if operating_system == "Darwin" else {}
    metal_available, metal_error = _probe_metal()
    return DoctorReport(
        supported_host=operating_system == "Darwin" and architecture == "arm64",
        operating_system=operating_system,
        architecture=architecture,
        chip=(
            _sysctl("machdep.cpu.brand_string")
            or _optional_string(hardware.get("chip_type"))
            or platform.processor()
            or "unknown"
        ),
        memory_bytes=(
            _parse_int(_sysctl("hw.memsize"))
            or _parse_memory(_optional_string(hardware.get("physical_memory")))
        ),
        python_version=platform.python_version(),
        metal_available=metal_available,
        metal_error=metal_error,
        backends=tuple(_backend_status(distribution, module) for distribution, module in _BACKENDS),
        mlx_compatibility=get_mlx_compatibility(),
        native_capabilities=_native_capabilities(),
    )


def format_doctor_report(report: DoctorReport) -> str:
    """Format a doctor report for terminal display."""
    memory = _format_bytes(report.memory_bytes)
    lines = [
        "mlx-one doctor",
        "Runtime",
        f"  Host: {report.operating_system} {report.architecture}",
        f"  Chip: {report.chip}",
        f"  Unified memory: {memory}",
        f"  Python: {report.python_version}",
        f"  MLX Metal: {'available' if report.metal_available else 'unavailable'}",
    ]
    compatibility = report.mlx_compatibility
    if compatibility is not None:
        lines.extend(
            [
                "MLX Compatibility",
                f"  Installed: {compatibility.runtime_version or 'not installed'}",
                f"  Latest upstream: {compatibility.latest_upstream}",
                f"  Latest verified: {compatibility.latest_verified or 'none'}",
                "  Window: latest 5 stable releases",
                f"  Status: {compatibility.state.value.upper()}",
                f"  Reason: {compatibility.reason}",
                "Qualified Releases:",
            ]
        )
        for version in compatibility.supported_versions:
            marker = "✓" if version == compatibility.latest_verified else "·"
            lines.append(f"  {marker} {version}")
        if not compatibility.supported_versions:
            lines.append("  none")
    elif report.backends:
        mlx = report.backends[0]
        installed = mlx.version if mlx.installed else "not installed"
        lines.append(f"  mlx: {installed}")
    if report.native_capabilities:
        lines.append("Native Capabilities:")
        lines.extend(
            f"  {name}: {status}" for name, status in sorted(report.native_capabilities.items())
        )
    if not report.supported_host:
        lines.append("Action: mlx-one requires macOS on Apple Silicon for model execution.")
    elif not report.metal_available:
        detail = f" ({report.metal_error})" if report.metal_error else ""
        lines.append(f"Action: verify that MLX can access the Metal device{detail}.")
    return "\n".join(lines)


def doctor_report_json(report: DoctorReport) -> str:
    """Serialize a doctor report as stable, formatted JSON."""
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def _backend_status(distribution: str, module: str) -> BackendStatus:
    installed = util.find_spec(module) is not None
    try:
        version = metadata.version(distribution) if installed else None
    except metadata.PackageNotFoundError:
        version = "unknown" if installed else None
    return BackendStatus(name=distribution, installed=installed, version=version)


def _native_capabilities() -> dict[str, str]:
    """Summarize task readiness without importing model implementations."""
    from mlx_one.core.registry import get_registration, registered_model_types

    registrations = [get_registration(name) for name in registered_model_types()]
    return {
        "asr": "supported"
        if any("transcribe" in item.capabilities for item in registrations)
        else "planned",
        "embeddings": "experimental"
        if any(
            item.modality in {"embedding", "reranking"} and item.loader_path
            for item in registrations
        )
        else "planned",
        "text": "supported"
        if any(
            item.modality == "text" and item.loader_path and item.supports("generate")
            for item in registrations
        )
        else "planned",
        "tts": "planned",
        "vlm": "experimental"
        if any(item.modality == "vision-language" for item in registrations)
        else "planned",
    }


def _probe_metal() -> tuple[bool, str | None]:
    command = [
        sys.executable,
        "-c",
        "import mlx.core.metal as metal; print('1' if metal.is_available() else '0')",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if completed.returncode == 0 and completed.stdout.strip().endswith("1"):
        return True, None
    error = completed.stderr.strip().splitlines()
    return False, error[-1] if error else f"probe exited with status {completed.returncode}"


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _hardware_profile() -> dict[str, object]:
    try:
        completed = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
        entries = payload.get("SPHardwareDataType", [])
        return entries[0] if entries and isinstance(entries[0], dict) else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}


def _parse_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _parse_memory(value: str | None) -> int | None:
    if value is None:
        return None
    parts = value.split()
    if len(parts) != 2:
        return None
    try:
        amount = float(parts[0])
    except ValueError:
        return None
    multipliers = {"GB": 1024**3, "TB": 1024**4}
    multiplier = multipliers.get(parts[1].upper())
    return int(amount * multiplier) if multiplier is not None else None


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 1024**3:.1f} GB"
