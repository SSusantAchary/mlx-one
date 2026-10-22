"""Freeze an isolated local candidate, build once, and verify an offline wheel install."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--wait-final-checks", action="store_true")
    args = parser.parse_args()
    run = args.run
    if args.wait_final_checks:
        deadline = time.monotonic() + 5 * 3600
        while not (run / "final-checks/complete.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("final checks did not finish")
            time.sleep(5)
    rows = []
    for key in ["M01", "M03", "M09"]:
        path = next((run / "hardware" / key).glob("*/result.json"))
        data = json.loads(path.read_text())
        row = next(r for r in data["rows"] if r["requested_context"] == 128)
        samples = row["samples"]

        def median(name, samples=samples):
            return statistics.median(s[name] for s in samples)

        rows.append(
            f"| {data['checkpoint']} / {data['precision']} | {samples[0]['input_tokens']} / "
            f"{samples[0]['generated_tokens']} | {median('time_to_first_token_ms'):.2f} | "
            f"{median('prefill_tokens_per_second'):.2f} | "
            f"{median('decode_tokens_per_second'):.2f} | "
            f"{median('peak_inference_mlx_bytes') / 1024**3:.3f} | "
            f"{median('peak_cache_state_bytes') / 1024**2:.3f} |"
        )
    readme = ROOT / "README.md"
    section = (
        "\nMeasured on the declared M4 host, dense B1, FP16 KV, prefix reuse disabled; "
        "median of three warm requests after loading/conversion and shape warmup. These are "
        "experimental measurements, not release certification or comparison claims.\n\n"
        "| Checkpoint / profile | Actual input/output tokens | TTFT ms | Prefill tok/s | "
        "Decode tok/s | Peak inference MLX GiB | Peak state MiB |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    readme.write_text(readme.read_text() + section)
    candidate = Path("/tmp") / ("mlx-one-alpha-candidate-" + run.name)
    if candidate.exists():
        raise FileExistsError(candidate)
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", "--local", str(ROOT), str(candidate)],
        check=True,
    )
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".DS_Store", "*.egg-info", "*.local.json"
    )
    for folder in [
        "src",
        "tests",
        "scripts",
        "benchmarks",
        "examples",
        "requirements",
        ".github",
        "ui",
    ]:
        target = candidate / folder
        if target.exists():
            shutil.rmtree(target)
        selected_ignore = (
            shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "results")
            if folder == "benchmarks"
            else shutil.ignore_patterns(
                "node_modules", ".svelte-kit", "test-results", "playwright-report", ".DS_Store"
            )
            if folder == "ui"
            else ignore
        )
        shutil.copytree(ROOT / folder, target, ignore=selected_ignore)
    for name in ["README.md", "pyproject.toml", "MANIFEST.in", "LICENSE", "CHANGELOG.md"]:
        shutil.copy2(ROOT / name, candidate / name)
    subprocess.run(["git", "add", "-A"], cwd=candidate, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Codex",
            "-c",
            "user.email=codex@openai.com",
            "commit",
            "-q",
            "-m",
            "Freeze current source for local alpha qualification",
        ],
        cwd=candidate,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate, text=True).strip()
    production = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "src/mlx_one").rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and not p.name.endswith((".pyc", ".DS_Store"))
    }
    for name, expected in production.items():
        if hashlib.sha256((candidate / name).read_bytes()).hexdigest() != expected:
            raise ValueError("candidate source differs: " + name)
    (run / "frozen-candidate.json").write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "path": str(candidate),
                "commit": commit,
                "source_base_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "package_source_sha256": production,
                "original_working_tree_preserved": True,
            },
            indent=2,
        )
    )
    dist = run / "dist"
    dist.mkdir()

    def execute(name, command, cwd=candidate, timeout=600, environment=None):
        start = time.time()
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment or {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        (run / (name + ".log")).write_text(result.stdout + result.stderr)
        (run / (name + "-command.json")).write_text(
            json.dumps(
                {
                    "timestamp": start,
                    "command": command,
                    "cwd": str(cwd),
                    "exit_code": result.returncode,
                    "duration": time.time() - start,
                    "source_commit": commit,
                },
                indent=2,
            )
        )
        print(name, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(name + " failed; see its log")
        return result

    execute(
        "build-distributions",
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist)],
    )
    artifacts = sorted(dist.iterdir())
    execute("twine-check", [sys.executable, "-m", "twine", "check", *map(str, artifacts)])
    execute(
        "package-audit",
        [
            sys.executable,
            str(candidate / "scripts/release/package_audit.py"),
            "--dist",
            str(dist),
            "--output",
            str(run / "package-audit.json"),
        ],
    )
    if json.loads((run / "package-audit.json").read_text())["status"] != "PASS":
        raise RuntimeError("distribution payload audit failed")
    fresh = Path("/tmp") / ("mlx-one-alpha-wheel-" + run.name)
    venv.EnvBuilder(with_pip=True).create(fresh)
    wheel = next(dist.glob("*.whl"))
    environment = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(PYTHONDONTWRITEBYTECODE="1", PIP_CACHE_DIR=str(fresh / "pip-cache"))
    execute(
        "wheel-bootstrap-pip",
        [
            str(fresh / "bin/python"),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            "/tmp/mlx-one-reference-wheelhouse",
            "pip==26.2.1",
        ],
        cwd=Path("/tmp"),
        environment=environment,
    )
    execute(
        "wheel-install",
        [
            str(fresh / "bin/python"),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            "/tmp/mlx-one-reference-wheelhouse",
            "-c",
            str(candidate / "requirements/alpha-tested-constraints.txt"),
            str(wheel) + "[server]",
        ],
        cwd=Path("/tmp"),
        environment=environment,
    )
    execute(
        "wheel-pip-check",
        [str(fresh / "bin/python"), "-m", "pip", "check"],
        cwd=Path("/tmp"),
        environment=environment,
    )
    packages = execute(
        "wheel-pip-list",
        [str(fresh / "bin/python"), "-m", "pip", "list", "--format=json"],
        cwd=Path("/tmp"),
        environment=environment,
    )
    (run / "wheel-environment.json").write_text(
        json.dumps(
            {"venv": str(fresh), "source_commit": commit, "packages": json.loads(packages.stdout)},
            indent=2,
        )
    )
    execute(
        "wheel-cli",
        [str(fresh / "bin/mlx-one"), "--version"],
        cwd=Path("/tmp"),
        environment=environment,
    )
    for key in ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10"]:
        execute(
            "wheel-" + key,
            [
                str(fresh / "bin/python"),
                str(candidate / "scripts/release/wheel_smoke.py"),
                "--assets",
                str(run / "checkpoint-manifest.json"),
                "--model",
                key,
                "--output",
                str(run / ("wheel-" + key + ".json")),
            ],
            cwd=Path("/tmp"),
            environment=environment,
        )
    execute(
        "wheel-http",
        [
            str(fresh / "bin/python"),
            str(candidate / "scripts/release/wheel_smoke.py"),
            "--assets",
            str(run / "checkpoint-manifest.json"),
            "--model",
            "M03",
            "--http",
            "--output",
            str(run / "wheel-http.json"),
        ],
        cwd=Path("/tmp"),
        environment=environment,
    )
    (run / "package-complete.json").write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "status": "PASS",
                "source_commit": commit,
                "wheel": wheel.name,
                "artifacts": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
