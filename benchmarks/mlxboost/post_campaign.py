"""Serialize API/wheel/regression work after all timed GPU measurements finish."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 3600
    while True:
        stages_file = args.confirmation / "stages.json"
        stages = json.loads(stages_file.read_text()) if stages_file.exists() else []
        if stages and stages[-1]["exit_code"]:
            raise RuntimeError("timed campaign failed; inspect its retained log")
        if stages and stages[-1]["stage"] == "expanded-qwen3-on":
            break
        if time.monotonic() > deadline:
            raise TimeoutError("timed campaign did not finish; post work not started")
        time.sleep(3)
    results = []
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["HF_HUB_OFFLINE"] = "1"

    def run(name, command, *, cwd=ROOT, required=True):
        print("POST", name, flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=1200
            )
        results.append(
            {"name": name, "command": [str(x) for x in command], "exit_code": result.returncode}
        )
        (args.output / "stages.json").write_text(json.dumps(results, indent=2) + "\n")
        if required and result.returncode:
            raise RuntimeError("post stage failed: " + name)

    run(
        "api-integrity",
        [
            sys.executable,
            ROOT / "benchmarks/mlxboost/integrity.py",
            "--output",
            args.output / "api-integrity",
        ],
        required=False,
    )
    run("regression", [sys.executable, "-m", "pytest", "tests", "--no-cov", "-q"], required=False)
    run(
        "wheel-build",
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            args.output / "dist",
        ],
    )
    environment = args.output / "wheel-env"
    run("wheel-venv", [sys.executable, "-m", "venv", environment])
    python = environment / "bin/python"
    # Reproduce the already isolated native-only server dependency lock, without
    # upgrading or installing anything in either working environment.
    original = json.loads((ROOT / "local/competitor-comparison/environments.json").read_text())
    packages = original["environments"]["mlx-one"]["packages"]
    requirements = args.output / "native-lock.txt"
    requirements.write_text(
        "\n".join(
            f"{name}=={version}"
            for name, version in packages.items()
            if name.lower().replace("_", "-") not in {"mlx-one", "pip", "setuptools"}
        )
        + "\n"
    )
    wheel = next((args.output / "dist").glob("*.whl"))
    run(
        "wheel-install",
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "-r",
            requirements,
            wheel,
        ],
    )
    run("wheel-pip-check", [python, "-m", "pip", "check"])
    models = json.loads((ROOT / "local/competitor-comparison/models.json").read_text())["models"]
    run(
        "wheel-smoke",
        [
            python,
            ROOT / "benchmarks/mlxboost/wheel_smoke.py",
            "--model",
            models[0]["path"],
            "--output",
            args.output / "wheel-smoke.json",
        ],
        cwd=args.output,
    )
    run("wheel-cli", [environment / "bin/mlx-one", "serve", "--help"], cwd=args.output)


if __name__ == "__main__":
    main()
