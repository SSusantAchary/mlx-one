"""Reject new Ruff findings while keeping each recorded debt item visible."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "src",
            "tests",
            "benchmarks/model_parity",
            "scripts/qualification",
            "scripts/release",
            "--output-format",
            "json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    findings = json.loads(result.stdout)
    current = Counter(
        (str(Path(r["filename"]).relative_to(ROOT)), r["code"], r["message"]) for r in findings
    )
    baseline = Counter(
        tuple(r) for r in json.loads((ROOT / "scripts/release/lint-debt.json").read_text())
    )
    new = current - baseline
    print(
        json.dumps(
            {
                "findings": len(findings),
                "new_findings": list(new.elements()),
                "debt_tracker": "docs/qualification/alpha_release/lint-debt.md",
            },
            indent=2,
        )
    )
    return int(bool(new))


if __name__ == "__main__":
    raise SystemExit(main())
