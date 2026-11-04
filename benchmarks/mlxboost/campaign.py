"""Serialize native diagnostics after the original HTTP baseline exits its servers."""

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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 1200
    while True:
        result = json.loads(args.baseline.read_text())
        if len(result["sessions"]) == 27 and all(
            s["status"] in {"PASS", "FAIL"} for s in result["sessions"]
        ):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("baseline has not completed; native GPU work was not started")
        time.sleep(2)
    args.output.mkdir(parents=True, exist_ok=False)
    statuses = []
    for operation in ["diagnose", "qualify"]:
        for model in ["qwen2", "qwen3", "smollm2"]:
            command = [
                sys.executable,
                str(ROOT / f"benchmarks/mlxboost/{operation}.py"),
                "--model",
                model,
                "--output",
                str(args.output / f"{operation}-{model}"),
            ]
            print("Running", operation, model, flush=True)
            with (args.output / f"{operation}-{model}.log").open("w") as handle:
                status = subprocess.run(
                    command, stdout=handle, stderr=subprocess.STDOUT, timeout=600
                ).returncode
            statuses.append({"command": command, "exit_code": status})
            (args.output / "campaign.json").write_text(json.dumps(statuses, indent=2) + "\n")
    return int(any(s["exit_code"] for s in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
