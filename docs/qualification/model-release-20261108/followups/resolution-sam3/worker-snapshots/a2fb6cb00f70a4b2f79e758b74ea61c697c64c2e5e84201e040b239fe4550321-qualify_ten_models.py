"""One pinned exact checkpoint per supervised process; source-bound failure reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHECKS = (
    "inventory",
    "architecture",
    "weights",
    "numerical",
    "generation",
    "python-api",
    "cli",
    "http",
    "streaming",
    "hardware",
    "quality",
    "all",
)


def manifest(path):
    result = yaml.safe_load(path.read_text())
    if result.get("schema_version") != "ten-model-manifest-v1":
        raise ValueError("invalid ten-model manifest schema")
    models = result.get("models", [])
    if (
        len(models) != 10
        or len({r["repo"] for r in models}) != 10
        or {r["id"] for r in models} != {f"M{i:02d}" for i in range(1, 11)}
    ):
        raise ValueError("manifest must retain all ten unique exact targets")
    return models


def resolve_cached(repo, hub):
    base = hub / ("models--" + repo.replace("/", "--"))
    configs = sorted(base.glob("snapshots/*/config.json"))
    complete = []
    for config in configs:
        files = list(config.parent.glob("*.safetensors"))
        index = config.parent / "model.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
            if not shards or not all((config.parent / name).is_file() for name in shards):
                continue
        if files and (
            (config.parent / "tokenizer.json").is_file()
            or all((config.parent / name).is_file() for name in ("vocab.json", "merges.txt"))
        ):
            complete.append(config)
    if not complete:
        return None
    # Prefer the cached main pointer; never infer a repository from a derivative name.
    main = base / "refs/main"
    pinned = main.read_text().strip() if main.is_file() else None
    selected = next((p for p in complete if p.parent.name == pinned), complete[0])
    return selected.parent.resolve()


def supervise(command, log, *, timeout, rss_ceiling):
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    started = time.monotonic()
    peak, outcome = 0, "completed"
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    outcome = "timeout"
                    break
                sample = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True
                )
                if sample.stdout.strip():
                    peak = max(peak, int(sample.stdout.strip()) * 1024)
                if peak > rss_ceiling:
                    outcome = "memory_ceiling"
                    break
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        code = process.wait()
    return {
        "status": outcome,
        "exit_code": code,
        "peak_sampled_rss_bytes": peak,
        "elapsed_seconds": time.monotonic() - started,
        "timeout_seconds": timeout,
        "rss_ceiling_bytes": rss_ceiling,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("ten_model_manifest.yaml")
    )
    parser.add_argument("--model", action="append", help="M01..M10; default all ten")
    parser.add_argument("--check", action="append", choices=CHECKS, default=[])
    parser.add_argument(
        "--precision",
        choices=["stored", "float32", "float16", "bfloat16", "affine_q4", "affine_q8"],
        default="stored",
    )
    parser.add_argument("--hub", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--rss-gib", type=float, default=20)
    parser.add_argument("--contexts", default="16,128,512")
    parser.add_argument("--outputs", default="1,16")
    args = parser.parse_args()
    models = manifest(args.manifest)
    if args.model and set(args.model) - {r["id"] for r in models}:
        parser.error("unknown model selection")
    if args.timeout <= 0 or not 0 < args.rss_gib <= 22:
        parser.error("positive timeout and RSS limit <=22 GiB required")
    if args.output.exists():
        parser.error("output already exists; use a unique qualification directory")
    args.output.mkdir(parents=True)
    source = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for base in ("src/mlx_one", "scripts/qualification", "benchmarks/model_parity")
        for p in sorted((ROOT / base).rglob("*.py"))
    }
    for name in ("ten_model_manifest.yaml", "ten_model_quality_v1.json"):
        fixture = Path(__file__).with_name(name)
        source[str(fixture.relative_to(ROOT))] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    (args.output / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    rows = []
    for model in models:
        row = {**model, "qualification": "NOT_QUALIFIED", "status": "NOT_RUN"}
        path = resolve_cached(model["repo"], args.hub)
        row.update(path=str(path) if path else None, revision=path.name if path else None)
        if args.model and model["id"] not in args.model:
            rows.append(row)
            continue
        if path is None:
            row.update(status="BLOCKED_ASSET", reason="exact cached safetensors snapshot absent")
        elif args.check and set(args.check) <= {"inventory", "architecture"}:
            try:
                from mlx_one.core.registry import get_registration

                data = json.loads((path / "config.json").read_text())
                registration = get_registration(data["model_type"])
                registration.config_class().from_dict(data)
                row.update(
                    status="PASS",
                    checks={
                        "inventory": {"status": "PASS"},
                        "architecture": {"status": "PASS", "model_type": data["model_type"]},
                    },
                    native_execution="NOT_RUN",
                )
            except Exception as exc:
                row.update(status="FAIL_LOADING", reason=f"{type(exc).__name__}: {exc}")
            (args.output / (model["id"] + ".json")).write_text(json.dumps(row, indent=2) + "\n")
        else:
            config = {
                **row,
                "checks": args.check or ["all"],
                "precision": args.precision,
                "contexts": [int(x) for x in args.contexts.split(",")],
                "outputs": [int(x) for x in args.outputs.split(",")],
            }
            request = args.output / (model["id"] + "-request.json")
            request.write_text(json.dumps(config, indent=2) + "\n")
            result = args.output / (model["id"] + ".json")
            command = [
                sys.executable,
                str(Path(__file__).with_name("ten_model_worker.py")),
                str(request),
                str(result),
            ]
            supervision = supervise(
                command,
                args.output / (model["id"] + ".log"),
                timeout=args.timeout,
                rss_ceiling=int(args.rss_gib * 1024**3),
            )
            if result.exists():
                row.update(json.loads(result.read_text()))
            else:
                row.update(status="BLOCKED_ENVIRONMENT", reason="worker did not finish", checks={})
            row["supervision"] = supervision
            if supervision["status"] != "completed" or supervision["exit_code"] != 0:
                row.update(status="BLOCKED_ENVIRONMENT", qualification="NOT_QUALIFIED")
        rows.append(row)
        (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(model["id"], model["repo"], row["status"], flush=True)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    text = "# Ten exact checkpoint qualification\n\n| ID | Checkpoint | Status | Qualification |\n"
    text += "|---|---|---|---|\n"
    text += (
        "\n".join(
            f"| {r['id']} | {r['repo']} | {r['status']} | {r['qualification']} |" for r in rows
        )
        + "\n"
    )
    (args.output / "summary.md").write_text(text)
    return 0 if all(r["status"] in {"PASS", "NOT_RUN"} for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
