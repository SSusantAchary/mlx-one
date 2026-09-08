"""Resumable orchestration for the reference text lifecycle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from mlx_one.evaluation import TEXT_CODING_PROFILE, TEXT_PROFILES
from mlx_one.schemas import TrainConfig
from mlx_one.training import load_train_config, save_train_config


class WorkflowError(RuntimeError):
    """Raised when a workflow configuration or stage fails."""


def run_text_workflow(config_path: str | Path, *, resume: bool = False) -> Path:
    """Run or resume the reference inspect-to-compare workflow."""
    source = Path(config_path)
    config = _load_config(source)
    config_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    output = Path(config["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "workflow.json"
    state = (
        _load_state(state_path)
        if resume and state_path.exists()
        else {
            "schema_version": "1.0",
            "status": "running",
            "model": config["model"],
            "revision": config["revision"],
            "config_sha256": config_sha256,
            "stages": {},
        }
    )
    if state.get("config_sha256") != config_sha256:
        raise WorkflowError("workflow configuration changed; start a new output directory")
    train_config = load_train_config(config["train_config"])
    canonical_train = TrainConfig.from_dict(
        {**train_config.to_dict(), "output_dir": str(output / "training")}
    )
    canonical_train_path = output / "train.yaml"
    save_train_config(canonical_train, canonical_train_path)
    runs = output / "runs"
    checkpoint = output / "training" / "adapters" / "mlx-one-checkpoint.json"
    exported = output / "exported-adapter"
    profile = config.get("evaluation_profile", TEXT_CODING_PROFILE)
    max_tokens = str(config.get("max_tokens", 128))
    commands = [
        (
            "inspect",
            [
                "inspect",
                config["model"],
                "--revision",
                config["revision"],
                "--output",
                str(output / "inspection.json"),
            ],
        ),
        (
            "plan",
            [
                "plan",
                "train",
                "--model",
                config["model"],
                "--revision",
                config["revision"],
                "--hardware",
                config.get("hardware", "m4-air-32gb"),
                "--method",
                canonical_train.method.value,
                "--context-length",
                str(canonical_train.max_seq_length),
                "--output",
                str(output / "plan.json"),
            ],
        ),
        (
            "data",
            [
                "data",
                "validate",
                "--dataset",
                config["dataset"],
                "--layout",
                config.get("dataset_layout", "auto"),
                "--max-seq-length",
                str(canonical_train.max_seq_length),
            ],
        ),
        (
            "train",
            [
                "train",
                "--model",
                config["model"],
                "--revision",
                config["revision"],
                "--dataset",
                config["dataset"],
                "--config",
                str(canonical_train_path),
            ],
        ),
        (
            "export",
            [
                "export",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(exported),
                "--smoke-prompt",
                config.get("smoke_prompt", "Reply with OK."),
                "--max-tokens",
                "16",
            ],
        ),
        (
            "evaluate-base",
            [
                "evaluate",
                "--model",
                config["model"],
                "--revision",
                config["revision"],
                "--dataset",
                config["evaluation_dataset"],
                "--profile",
                profile,
                "--max-tokens",
                max_tokens,
                "--runs-dir",
                str(runs),
                "--run-id",
                "base-eval",
                "--json-output",
            ],
        ),
        (
            "evaluate-adapter",
            [
                "evaluate",
                "--model",
                config["model"],
                "--revision",
                config["revision"],
                "--dataset",
                config["evaluation_dataset"],
                "--adapter",
                str(exported),
                "--profile",
                profile,
                "--max-tokens",
                max_tokens,
                "--runs-dir",
                str(runs),
                "--run-id",
                "adapter-eval",
                "--json-output",
            ],
        ),
        (
            "benchmark",
            [
                "benchmark",
                "inference",
                "--model",
                config["model"],
                "--revision",
                config["revision"],
                "--prompts",
                config["prompts"],
                "--runs-dir",
                str(runs),
                "--run-id",
                "base-benchmark",
            ],
        ),
        (
            "compare",
            [
                "compare",
                str(runs / "base-eval" / "result.json"),
                str(runs / "adapter-eval" / "result.json"),
                "--gate",
                config["gate"],
                "--format",
                "markdown",
                "--output",
                str(output / "report.md"),
            ],
        ),
    ]
    try:
        for name, arguments in commands:
            if state["stages"].get(name, {}).get("status") == "completed":
                continue
            _run_stage(name, arguments, output, state, state_path)
    except WorkflowError:
        state["status"] = "failed"
        _atomic_json(state_path, state)
        raise
    state["status"] = "completed"
    _atomic_json(state_path, state)
    return output / "report.md"


def _load_config(path: str | Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WorkflowError(f"cannot read workflow configuration: {exc}") from exc
    required = {
        "schema_version",
        "model",
        "revision",
        "dataset",
        "train_config",
        "evaluation_dataset",
        "prompts",
        "gate",
        "output_dir",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise WorkflowError("workflow configuration is missing required fields")
    allowed = required | {
        "hardware",
        "dataset_layout",
        "evaluation_profile",
        "smoke_prompt",
        "max_tokens",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise WorkflowError(f"unsupported workflow fields: {', '.join(sorted(unknown))}")
    if payload["schema_version"] != "1.0":
        raise WorkflowError("workflow requires schema_version '1.0'")
    if not isinstance(payload["revision"], str) or len(payload["revision"]) not in range(40, 65):
        raise WorkflowError("workflow requires an immutable model revision")
    if payload.get("evaluation_profile", TEXT_CODING_PROFILE) not in TEXT_PROFILES:
        raise WorkflowError("workflow evaluation_profile is unsupported")
    return payload


def _run_stage(
    name: str,
    arguments: list[str],
    output: Path,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "mlx_one.cli", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    log = output / f"{name}.log"
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    state["stages"][name] = {
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "log": log.name,
    }
    _atomic_json(state_path, state)
    if completed.returncode:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "failed"
        raise WorkflowError(f"workflow stage {name!r} failed: {detail}")


def _load_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot resume workflow: {exc}") from exc
    if not isinstance(state, dict) or state.get("schema_version") != "1.0":
        raise WorkflowError("workflow state requires schema_version '1.0'")
    return state


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
