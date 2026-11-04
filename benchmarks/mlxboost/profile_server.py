"""Benchmark-only device-owner profile of the existing HTTP server."""

from __future__ import annotations

import cProfile
import os
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlx_one.server.scheduler import GenerationScheduler  # noqa: E402

original = GenerationScheduler._step
profile = cProfile.Profile()
output = Path(os.environ["MLXBOOST_PROFILE_DIR"])
output.mkdir(parents=True, exist_ok=True)


def step(self, work):
    if not self.device_executor.is_owner:
        return original(self, work)
    result = profile.runcall(original, self, work)
    if not result:
        profile.dump_stats(str(output / "http-device.pstats"))
        with (output / "http-device.txt").open("w") as handle:
            pstats.Stats(profile, stream=handle).strip_dirs().sort_stats("cumulative").print_stats(
                80
            )
    return result


GenerationScheduler._step = step

if __name__ == "__main__":
    from mlx_one.cli import main

    main()
