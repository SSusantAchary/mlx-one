"""Sequential shared-Qwen qualification over the existing supervised workers.

Discovery never imports MLX. This driver stops failed dependencies and freezes
the runtime/tooling for a campaign; it cannot promote registry capabilities.
Later hybrid phases intentionally require a separate reviewed source freeze.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from mlx_one.inference_baseline import source_provenance
from mlx_one.p0_engine_qualification import atomic_json, digest, utc
from qualification.p01_experiments import main as experiment_main
from qualification.p01_experiments import validate_sources
from qualification.p01_math import WIDTHS, await_prerequisites, canonical_digest, write_integrity

MODELS = ("mlx-community/Qwen3-0.6B-8bit", "mlx-community/Qwen2.5-0.5B-Instruct-8bit")


def inventory(models, widths=WIDTHS, *, repetitions=10):
    if (not models or set(models) - set(MODELS) or len(set(models)) != len(models)
            or not widths or set(widths) - set(WIDTHS) or len(set(widths)) != len(widths)
            or type(repetitions) is not int or repetitions < 10):
        raise ValueError("shared phase requires unique Qwen checkpoints/widths and >=10 repeats")
    steps = []
    for workload in ("parity", "release", "profile", "engine"):
        for model in models:
            depends = [s["id"] for s in steps if s["checkpoint"] == model
                       and s["workload"] in {"parity", "release"}]
            record = {"checkpoint": model, "workload": workload,
                "widths": [128] if workload in {"release", "profile"} else list(widths),
                "routes": ["optimized"] if workload in {"release", "profile"}
                          else ["reference", "optimized"],
                "repetitions": repetitions if workload == "engine" else 1,
                "independent_runs": 2, "depends": depends, "status": "NOT_RUN"}
            record["id"] = canonical_digest(record)[:20]
            steps.append(record)
    return steps


def run_steps(steps, execute, *, validate, persist):
    """Failures skip dependent work; supervision/environment failures stop all work."""
    outcomes = {s["id"]: s for s in steps}
    for step in steps:
        if any(outcomes[dependency]["status"] != "PASS" for dependency in step["depends"]):
            step.update(status="SKIPPED", reason="mandatory checkpoint dependency failed")
            persist()
            continue
        try:
            validate()
            step.update(status="running", started=utc())
            persist()
            step.update(execute(step))
            if step["status"] not in {"PASS", "FAIL", "BLOCKED"}:
                raise ValueError("qualification step returned an invalid outcome")
        except Exception as error:
            step.update(status="BLOCKED", reason=str(error), error_type=type(error).__name__)
        finally:
            step["finished"] = utc()
            persist()
        if step["status"] == "BLOCKED":
            return "BLOCKED"
    return "PASS" if all(s["status"] == "PASS" for s in steps) else "FAIL"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=list(WIDTHS))
    parser.add_argument("--prompts", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--after", action="append", type=Path, default=[])
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("qualification/results/p01_math/shared-campaign"))
    args = parser.parse_args(argv)
    try:
        steps = inventory(args.model or MODELS, args.prompt_tokens, repetitions=args.repetitions)
        prompts = dict(record.split("=", 1) for record in args.prompts)
        if len(prompts) != len(args.prompts) or set(prompts) != {s["checkpoint"] for s in steps}:
            raise ValueError("one frozen --prompts MODEL=PATH per selected checkpoint required")
        if any(not Path(path).is_file() for path in prompts.values()):
            raise ValueError("prompt manifest is missing")
    except ValueError as error:
        parser.error(str(error))
    root = args.output.resolve() / str(time.time_ns())
    root.mkdir(parents=True)
    source = source_provenance(Path.cwd())
    tools = {str(p.resolve()): digest(p) for p in Path(__file__).parent.glob("p01*.py")}
    frozen_inputs = {str(p.resolve()): digest(p) for p in
                     [args.baseline / "environment.json", args.baseline / "download_manifest.json",
                      *(Path(p) for p in prompts.values())]}
    report = {"schema_version": "native-math-shared-campaign-v1", "source": source,
        "tool_sources": tools, "inputs": frozen_inputs, "steps": steps,
        "status": "QUEUED" if args.run else "NOT_RUN", "timestamp": utc(),
        "qualification_promoted": False,
        "performance_qualification": "PENDING; worker success is not a materiality/protected gate",
        "later_phases": "NOT_RUN; separate source freeze required"}
    def persist():
        atomic_json(root / "manifest.json", report)
        write_integrity(root)
    persist()
    print(root, flush=True)
    if not args.run:
        return 0
    def validate():
        validate_sources({**tools, **frozen_inputs}, source)
    def execute(step):
        output = root / step["id"]
        arguments = ["--baseline", str(args.baseline), "--model", step["checkpoint"],
            "--route", *step["routes"], "--workload", step["workload"], "--prompt-tokens",
            *map(str, step["widths"]), "--projection", "mlx_qmm", "--repetitions",
            str(step["repetitions"]), "--independent-runs", str(step["independent_runs"]),
            "--output", str(output), "--run"]
        if step["workload"] != "release":
            arguments += ["--prompt-manifest", prompts[step["checkpoint"]]]
        if step["workload"] == "engine":
            for prerequisite in steps:
                if prerequisite["checkpoint"] == step["checkpoint"]:
                    for proof in prerequisite.get("proofs", []):
                        arguments += ["--correctness-proof", proof]
        atomic_json(root / f"{step['id']}-command.json",
                    {"argv": [sys.executable, "-m", "qualification.p01_experiments", *arguments]})
        error = None
        with ((root / f"{step['id']}.log").open("w") as log,
              redirect_stdout(log), redirect_stderr(log)):
            try:
                code = experiment_main(arguments)
            except Exception as caught:
                error, code = str(caught), 1
        manifests = sorted(output.glob("*/manifest.json"))
        if not manifests:
            return {"status": "BLOCKED", "reason": error or "worker manifest missing"}
        path = manifests[-1]
        data = json.loads(path.read_text())
        statuses = {task["status"] for task in data["tasks"]}
        # A numerical/model FAIL is retained; missing telemetry, watchdog,
        # timeout, source mismatch or other supervision failure halts the driver.
        status = ("BLOCKED" if "BLOCKED" in statuses or not statuses <= {"PASS", "FAIL", "NOT_RUN"}
                  else "FAIL" if "FAIL" in statuses else "PASS" if not code and statuses == {"PASS"}
                  else "BLOCKED")
        proofs = [str(Path(t["artifact"]) / "data/experiment.json") for t in data["tasks"]
                  if step["workload"] == "parity" and t["status"] == "PASS"]
        return {"status": status, "artifact": str(path), "proofs": proofs, "reason": error}
    try:
        await_prerequisites(args.after)
        report["status"] = "running"
        report["status"] = run_steps(steps, execute, validate=validate, persist=persist)
    except BaseException as error:
        report.update(status="BLOCKED", reason=str(error), error_type=type(error).__name__)
        raise
    finally:
        persist()
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
