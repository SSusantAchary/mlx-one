import pytest

from mlx_one.utils.memory import MemorySnapshot, estimate_model_fit, format_bytes, format_memory


def test_format_bytes() -> None:
    assert format_bytes(None) == "unknown"
    assert format_bytes(0) == "0 B"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(5.2 * 1024**3) == "5.2 GB"


def test_format_memory_with_total() -> None:
    snapshot = MemorySnapshot(
        active_bytes=5 * 1024**3,
        peak_bytes=6 * 1024**3,
        total_bytes=26 * 1024**3,
        device_name="Apple M4",
    )

    assert format_memory(snapshot) == "5.0 GB active / 21.0 GB available of 26.0 GB"


def test_estimate_model_fit() -> None:
    snapshot = MemorySnapshot(
        active_bytes=4 * 1024**3,
        peak_bytes=5 * 1024**3,
        total_bytes=16 * 1024**3,
    )

    assert estimate_model_fit(8 * 1024**3, snapshot=snapshot)
    assert not estimate_model_fit(15 * 1024**3, snapshot=snapshot)


def test_estimate_model_fit_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        estimate_model_fit(-1, snapshot=MemorySnapshot(0, 0, None))
