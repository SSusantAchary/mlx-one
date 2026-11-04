"""Run gated, serialized requalification and the frozen HTTP protocol.

Every stage uses a new directory. This command never changes environments or
downloads models. Native tests and HTTP measurement processes do not overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "local/competitor-comparison/venvs/client/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stages = []
    deadline = time.monotonic() + 900
    while True:
        paths = [
            ROOT / f"local/mlxboost/runs/ablation-v2-{m}-20261101/results.json"
            for m in ["qwen2", "qwen3", "smollm2"]
        ]
        if all(p.exists() for p in paths):
            if not all(
                json.loads(p.read_text())["state_parity"]
                and json.loads(p.read_text())["exact_output_parity"]
                for p in paths
            ):
                raise RuntimeError("flag parity failed; HTTP optimization runs blocked")
            break
        if time.monotonic() > deadline:
            raise TimeoutError("ablation gate did not complete")
        time.sleep(2)

    def run(name, command, *, env=None, timeout=2400):
        print("STAGE", name, flush=True)
        with (output / f"{name}.log").open("w") as log:
            status = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout
            ).returncode
        stages.append({"stage": name, "command": [str(x) for x in command], "exit_code": status})
        (output / "stages.json").write_text(json.dumps(stages, indent=2) + "\n")
        if status:
            raise RuntimeError(f"stage failed: {name}; retained log: {output / (name + '.log')}")

    run(
        "native-tests",
        [sys.executable, "-m", "pytest", "tests/mlxboost/test_native.py", "--no-cov", "-q"],
        env={**os.environ, "MLXBOOST_NATIVE_TESTS": "1"},
    )
    runner = ROOT / "benchmarks/mlxboost/http_experiment.py"
    for name, flags, backends in [
        ("original-protocol-fast", ["--boost", "on"], []),
        ("original-protocol-off", ["--boost", "off"], ["--backends", "mlx-one"]),
    ]:
        destination = output / name
        run(
            name,
            [
                CLIENT,
                runner,
                *flags,
                "run",
                *backends,
                "--output",
                destination,
                "--report",
                destination / "report.md",
            ],
        )
    if args.expanded:
        prepared = output / "expanded-inputs"
        prepared.mkdir()
        shutil.copy2(ROOT / "local/competitor-comparison/environments.json", prepared)
        run(
            "expanded-prepare",
            [
                CLIENT,
                ROOT / "scripts/compare_competators.py",
                "--directory",
                prepared,
                "prepare",
                "--cases",
                "55",
            ],
        )
        for model, order in [("qwen2", ["on", "off"]), ("qwen3", ["off", "on"])]:
            for flag in order:
                name = f"expanded-{model}-{flag}"
                destination = output / name
                run(
                    name,
                    [
                        CLIENT,
                        runner,
                        "--boost",
                        flag,
                        "--directory",
                        prepared,
                        "run",
                        "--models",
                        model,
                        "--backends",
                        "mlx-one",
                        "--rounds",
                        "5",
                        "--repetitions",
                        "10",
                        "--output",
                        destination,
                        "--report",
                        destination / "report.md",
                    ],
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
