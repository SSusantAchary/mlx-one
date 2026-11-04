"""Create isolated, normally resolved environments for the competitor benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OMLX_COMMIT = "9de79bc39a01c1f3fec1e849903aaf0dd5478d7a"  # official v0.7.3
MLX_LM_VERSION = "0.32.2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "local/competitor-comparison")
    args = parser.parse_args()
    directory = args.directory.resolve()
    logs = directory / "setup-logs"
    logs.mkdir(parents=True, exist_ok=True)
    packages = {
        "client": ["httpx==0.28.1", "psutil>=5.9", "tokenizers==0.23.2", "pytest"],
        "mlx-one": [str(ROOT) + "[server]"],
        "mlx-lm": [f"mlx-lm=={MLX_LM_VERSION}", "mlx==0.32.4", "mlx-metal==0.32.4"],
        "omlx": [f"omlx @ git+https://github.com/jundot/omlx.git@{OMLX_COMMIT}"],
    }
    receipt = {"created_at": time.time(), "python": sys.version, "environments": {}}
    for name, requirements in packages.items():
        environment = directory / "venvs" / name
        if not (environment / "bin/python").exists():
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        python = environment / "bin/python"
        command = [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *requirements,
        ]
        print(f"Installing {name}; log: {logs / (name + '.log')}", flush=True)
        with (logs / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((logs / (name + ".log")).read_text()[-6000:], flush=True)
            raise SystemExit(result.returncode)
        check = subprocess.run([str(python), "-m", "pip", "check"], capture_output=True, text=True)
        (logs / (name + "-pip-check.log")).write_text(check.stdout + check.stderr)
        if check.returncode:
            raise RuntimeError(f"{name} has unsatisfied dependencies: {check.stdout}")
        freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
        (logs / (name + "-freeze.txt")).write_text(freeze)
        inventory = json.loads(
            subprocess.check_output([str(python), "-m", "pip", "list", "--format=json"], text=True)
        )
        receipt["environments"][name] = {
            "python": str(python),
            "requirements": requirements,
            "packages": {item["name"]: item["version"] for item in inventory},
            "pip_check": "PASS",
            "system_site_packages": False,
        }
        (directory / "environments.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(name + " ready", flush=True)


if __name__ == "__main__":
    main()
