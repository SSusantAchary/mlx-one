"""Record cold and synchronized warm performance for a pinned native image profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path

from mlx_one import ImageEngine
from mlx_one.engine.pressure import sample_telemetry
from mlx_one.image import IMAGE_PROFILES

PROMPT = "A botanical scientific illustration of a fiddlehead fern on ivory paper, no text."
SIZES = ((512, 512), (768, 512), (1024, 576))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=[profile.model for profile in IMAGE_PROFILES])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "results", type=Path, help="New output directory for PNGs and benchmark JSON"
    )
    parser.add_argument("--quantization", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    args.results.mkdir()
    import mlx.core as mx

    host = subprocess.run(["sysctl", "-n", "hw.model"], check=True, capture_output=True, text=True)
    record = {
        "host": host.stdout.strip(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "device": mx.device_info(),
        "model": args.model,
        "checkpoint_storage_bytes": sum(
            p.stat().st_size for p in args.checkpoint.rglob("*") if p.is_file()
        ),
        "prompt": PROMPT,
        "seed": 42,
        "results": [],
        "status": "incomplete",
    }
    path = args.results / "benchmark.json"
    try:
        with ImageEngine(
            args.model,
            quantization=args.quantization,
            checkpoint=args.checkpoint,
            idle_ttl=600,
            offline=True,
        ) as engine:
            for width, height in SIZES:
                size = f"{width}x{height}"
                engine.unload()
                runs = {}
                for label in ("cold", "warmup", "warm1", "warm2", "warm3"):
                    if sample_telemetry().os_pressure in {"warning", "critical"}:
                        record.update(status="stopped", reason="host memory pressure")
                        return
                    mx.reset_peak_memory()
                    result = engine.generate(PROMPT, width=width, height=height, seed=42)
                    runs[label] = {
                        **result.metadata,
                        "png_sha256": hashlib.sha256(result.png).hexdigest(),
                    }
                    if label == "warm1":
                        result.save(args.results / f"{size}.png")
                warm = [runs[f"warm{i}"]["timings_seconds"]["total"] for i in range(1, 4)]
                record["results"].append(
                    {
                        "size": size,
                        "runs": runs,
                        "warm_median_seconds": statistics.median(warm),
                        "warm_range_seconds": [min(warm), max(warm)],
                    }
                )
        record["status"] = "measured"
    finally:
        path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
