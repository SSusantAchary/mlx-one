"""Extend, without modifying, the frozen competitor HTTP runner.

This run uses current local source for MLX-One. Other stacks and all prompt,
sampling, EOS, usage and isolation rules are inherited from the original script.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--boost", choices=["off", "on"])
    args, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location(
        "competitors", ROOT / "scripts/compare_competators.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    original = benchmark.server_command

    def command(backend, python, model, folder, port):
        result = original(backend, python, model, folder, port)
        if backend == "mlx-one":
            if args.boost is not None:
                result[result.index("reference_dense")] = "mlxboost_b1"
                if args.boost == "on":
                    result.append("--mlxboost-fast-decode")
            if args.profile:
                os.environ["MLXBOOST_PROFILE_DIR"] = str(folder / "profiles")
                return [
                    str(python),
                    str(ROOT / "benchmarks/mlxboost/profile_server.py"),
                    *result[1:],
                ]
            return [str(python), str(ROOT / "benchmarks/mlxboost/local_server.py"), *result[1:]]
        return result

    benchmark.server_command = command
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    sys.argv = [str(ROOT / "scripts/compare_competators.py"), *remaining]
    return benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
