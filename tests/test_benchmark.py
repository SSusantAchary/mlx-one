from pathlib import Path

from mlx_one import HardwareSpec
from mlx_one.benchmark import benchmark_inference
from mlx_one.run_store import RunStore


def test_benchmark_records_measured_performance_separately(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mlx_one.benchmark.detect_hardware",
        lambda: HardwareSpec(
            profile_id="test",
            platform="macOS",
            architecture="arm64",
            chip="Apple M4",
        ),
    )

    def execute(request):
        assert request["repeats"] == 2
        return {
            "load_seconds": 1.0,
            "sample_count": 2,
            "mean_wall_seconds": 0.5,
            "mean_prompt_tokens_per_second": 100.0,
            "mean_decode_tokens_per_second": 50.0,
            "peak_metal_bytes": 1024,
        }

    result = benchmark_inference(
        "example/model",
        "a" * 40,
        ["hello"],
        store=RunStore(tmp_path / "runs"),
        repeats=2,
        run_id="benchmark-test",
        executor=execute,
    )

    assert result.metrics["performance.decode_tokens_per_second"]["value"] == 50.0
    assert result.metrics["performance.decode_tokens_per_second"]["provenance"] == "measured"
    assert result.memory["peak_metal_bytes"]["unit"] == "bytes"
