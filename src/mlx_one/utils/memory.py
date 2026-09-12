"""Memory helpers for MLX Metal workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class MemorySnapshot:
    """Current memory state for the active Metal device."""

    active_bytes: int
    peak_bytes: int
    total_bytes: int | None
    device_name: str | None = None

    @property
    def available_bytes(self) -> int | None:
        """Return estimated currently available device memory."""
        if self.total_bytes is None:
            return None
        return max(self.total_bytes - self.active_bytes, 0)


def format_bytes(value: int | float | None) -> str:
    """Format bytes as a compact human-readable value."""
    if value is None:
        return "unknown"
    value = float(value)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{value:.0f} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def format_memory(snapshot: MemorySnapshot) -> str:
    """Format active and available memory for terminal output."""
    if snapshot.total_bytes is None:
        return (
            f"{format_bytes(snapshot.active_bytes)} active, "
            f"{format_bytes(snapshot.peak_bytes)} peak"
        )
    return (
        f"{format_bytes(snapshot.active_bytes)} active / "
        f"{format_bytes(snapshot.available_bytes)} available "
        f"of {format_bytes(snapshot.total_bytes)}"
    )


def _load_metal_modules() -> tuple[Any, Any]:
    try:
        import mlx.core as mx
        import mlx.core.metal as metal
    except Exception as exc:  # pragma: no cover - exact MLX failures vary by host.
        raise RuntimeError(f"MLX Metal backend is unavailable: {exc}") from exc
    return mx, metal


def get_active_memory() -> int:
    """Return current active MLX Metal memory in bytes."""
    mx, metal = _load_metal_modules()
    if not metal.is_available():
        raise RuntimeError("MLX Metal backend is unavailable on this host.")
    return int(mx.get_active_memory())


def get_peak_memory() -> int:
    """Return peak MLX Metal memory in bytes."""
    mx, metal = _load_metal_modules()
    if not metal.is_available():
        raise RuntimeError("MLX Metal backend is unavailable on this host.")
    return int(mx.get_peak_memory())


def get_memory_snapshot() -> MemorySnapshot:
    """Return active, peak, and total memory for the Metal device."""
    mx, metal = _load_metal_modules()
    if not metal.is_available():
        raise RuntimeError("MLX Metal backend is unavailable on this host.")

    info = metal.device_info()
    return MemorySnapshot(
        active_bytes=int(mx.get_active_memory()),
        peak_bytes=int(mx.get_peak_memory()),
        total_bytes=_optional_int(info.get("memory")),
        device_name=_optional_str(info.get("name")),
    )


def estimate_model_fit(
    model_size_bytes: int,
    *,
    safety_margin_bytes: int = 2 * BYTES_PER_GB,
    snapshot: MemorySnapshot | None = None,
) -> bool:
    """Estimate whether a model fits in available Metal memory."""
    if model_size_bytes < 0:
        raise ValueError("model_size_bytes must be non-negative")
    if safety_margin_bytes < 0:
        raise ValueError("safety_margin_bytes must be non-negative")

    snapshot = snapshot or get_memory_snapshot()
    if snapshot.available_bytes is None:
        return True
    return model_size_bytes + safety_margin_bytes <= snapshot.available_bytes


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
