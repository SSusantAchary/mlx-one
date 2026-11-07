"""Serialize final source/forward diagnostics after API/wheel qualification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 2400
    while True:
        path = args.integrity / "stages.json"
        stages = json.loads(path.read_text()) if path.exists() else []
        if stages and stages[-1]["name"] == "wheel-cli":
            break
        if time.monotonic() > deadline:
            raise TimeoutError("post campaign did not finish")
        time.sleep(3)
    results = []
    for operation, models in [
        ("reference_probe", ["qwen2", "qwen3", "smollm2", "lfm2"]),
        ("diagnose", ["qwen2", "qwen3", "smollm2"]),
        ("qualify", ["qwen2", "qwen3", "smollm2"]),
    ]:
        for model in models:
            name = f"{operation}-{model}"
            print("ANCILLARY", name, flush=True)
            command = [
                sys.executable,
                ROOT / f"benchmarks/mlxboost/{operation}.py",
                "--model",
                model,
                "--output",
                args.output / name,
            ]
            with (args.output / f"{name}.log").open("w") as log:
                status = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600
                ).returncode
            results.append(
                {
                    "name": name,
                    "command": [str(x) for x in command],
                    "exit_code": status,
                    "numerical_failure_is_not_a_pass": operation == "qualify",
                }
            )
            (args.output / "stages.json").write_text(json.dumps(results, indent=2) + "\n")
            if status and not (operation == "qualify" and status == 2):
                print("Worker failed; preserve evidence and continue independent work", flush=True)


if __name__ == "__main__":
    main()
