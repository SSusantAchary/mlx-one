"""Native mlx-one SFT orchestration over the upstream MLX execution engine."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from mlx_one.datasets import dataset_spec_from_path, load_records
from mlx_one.run_store import RunStore
from mlx_one.schemas import (
    ArtifactRef,
    CheckpointMetadata,
    Failure,
    MetricProvenance,
    MetricValue,
    Modality,
    ModelSpec,
    Operation,
    RunResult,
    RunSpec,
    RunStatus,
    TrainConfig,
    TrainingMethod,
)


class TrainingError(RuntimeError):
    """Raised for unsupported or failed SFT workflows."""


@dataclass(frozen=True)
class BackendTrainingResult:
    adapter_path: Path
    loss: float | None
    tokens_per_second: float | None
    stdout: str = ""


class TrainingBackend(Protocol):
    def train(
        self,
        *,
        model_id: str,
        revision: str,
        data_dir: Path,
        config: TrainConfig,
        resume_adapter: Path | None,
    ) -> BackendTrainingResult: ...


class MLXTrainingBackend:
    """Lazy, isolated adapter for mlx-lm's maintained LoRA implementation."""

    def train(
        self,
        *,
        model_id: str,
        revision: str,
        data_dir: Path,
        config: TrainConfig,
        resume_adapter: Path | None,
    ) -> BackendTrainingResult:
        adapter_dir = Path(config.output_dir).expanduser().resolve() / "adapters"
        request = {
            "model_id": model_id,
            "revision": revision,
            "mlx_lm_config": {
                "train": True,
                "test": False,
                "data": str(data_dir),
                "fine_tune_type": "lora",
                "optimizer": config.optimizer,
                "seed": config.seed,
                "num_layers": config.lora_layers,
                "batch_size": config.batch_size,
                "iters": config.max_steps,
                "learning_rate": config.learning_rate,
                "steps_per_report": 1,
                "steps_per_eval": config.eval_steps or config.max_steps + 1,
                "grad_accumulation_steps": config.gradient_accumulation_steps,
                "adapter_path": str(adapter_dir),
                "save_every": config.save_steps,
                "max_seq_length": config.max_seq_length,
                "grad_checkpoint": config.gradient_checkpointing,
                "mask_prompt": True,
                "resume_adapter_file": str(resume_adapter) if resume_adapter else None,
                "lora_parameters": {
                    "rank": config.lora_rank,
                    "dropout": config.lora_dropout,
                    "scale": config.lora_alpha / config.lora_rank,
                    **({"keys": list(config.target_modules)} if config.target_modules else {}),
                },
            },
        }
        with tempfile.TemporaryDirectory(prefix="mlx-one-train-") as temporary:
            request_path = Path(temporary) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "mlx_one.training_worker", str(request_path)],
                capture_output=True,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            detail = (
                completed.stderr.strip().splitlines()[-1]
                if completed.stderr.strip()
                else "worker failed"
            )
            raise TrainingError(f"MLX training failed: {detail}")
        adapter = adapter_dir / "adapters.safetensors"
        if not adapter.is_file():
            raise TrainingError("MLX training completed without producing adapter weights")
        losses = [float(item) for item in re.findall(r"Train loss ([0-9.eE+-]+)", completed.stdout)]
        rates = [float(item) for item in re.findall(r"It/sec ([0-9.eE+-]+)", completed.stdout)]
        return BackendTrainingResult(
            adapter_path=adapter,
            loss=losses[-1] if losses else None,
            tokens_per_second=rates[-1] if rates else None,
            stdout=completed.stdout,
        )


class SFTTrainer:
    """Backend-neutral SFT entry point; MLX is the initial execution backend."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        train_dataset: str | Path,
        args: TrainConfig,
        backend: TrainingBackend | None = None,
    ) -> None:
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
            raise TrainingError("training requires an immutable 40-64 character revision")
        self.model = model
        self.revision = revision.lower()
        self.train_dataset = Path(train_dataset)
        self.args = args
        self.backend = backend or MLXTrainingBackend()
        if args.method is TrainingMethod.QLORA:
            explicitly_quantized = bool(args.metadata.get("base_quantized"))
            looks_quantized = "4bit" in model.lower() or "4-bit" in model.lower()
            if not explicitly_quantized and not looks_quantized:
                raise TrainingError(
                    "qlora requires a verified 4-bit base; use a 4-bit artifact or set "
                    "metadata.base_quantized after inspection"
                )

    def train(self, *, resume_from_checkpoint: str | Path | None = None) -> RunResult:
        """Normalize data, execute isolated training, and persist terminal evidence."""
        output = Path(self.args.output_dir).expanduser().resolve()
        store = RunStore(output / "runs")
        dataset = dataset_spec_from_path(self.train_dataset, response_only=True)
        run_id = f"train-{uuid.uuid4().hex[:12]}"
        model = ModelSpec(model_id=self.model, revision=self.revision, modality=Modality.TEXT)
        spec = RunSpec(
            run_id=run_id,
            operation=Operation.TRAIN,
            model=model,
            profile="text-sft-v1",
            dataset_revisions={dataset.dataset_id: dataset.revision},
            seed=self.args.seed,
            metadata={
                "train_config": self.args.to_dict(),
                "metric_policy": "explicit-provenance-v1",
            },
        )
        run_dir = store.create(spec)
        data_dir = run_dir / "data"
        _materialize_dataset(load_records(dataset), data_dir)
        started = _now()
        begin = time.perf_counter()
        try:
            backend_result = self.backend.train(
                model_id=self.model,
                revision=self.revision,
                data_dir=data_dir,
                config=self.args,
                resume_adapter=Path(resume_from_checkpoint) if resume_from_checkpoint else None,
            )
        except Exception as exc:
            result = RunResult(
                run_id=run_id,
                status=RunStatus.FAILED,
                spec=spec,
                failures=(Failure(code="training-failed", message=str(exc), retryable=True),),
                started_at=started,
                ended_at=_now(),
            )
            store.save_result(result)
            if isinstance(exc, TrainingError):
                raise
            raise TrainingError(str(exc)) from exc
        elapsed = time.perf_counter() - begin
        adapter_hash = hashlib.sha256(backend_result.adapter_path.read_bytes()).hexdigest()
        config_hash = hashlib.sha256(self.args.to_json(indent=None).encode()).hexdigest()
        adapter = ArtifactRef(
            uri=str(backend_result.adapter_path),
            kind="lora-adapter",
            sha256=adapter_hash,
            media_type="application/x-safetensors",
            metadata={"base_model_id": self.model, "base_revision": self.revision},
        )
        checkpoint = CheckpointMetadata(
            run_id=run_id,
            base_model_id=self.model,
            base_revision=self.revision,
            method=self.args.method,
            step=self.args.max_steps,
            adapter=adapter,
            config_sha256=config_hash,
            resumable_state=("adapter-weights",),
        )
        _atomic_json(
            backend_result.adapter_path.parent / "mlx-one-checkpoint.json", checkpoint.to_dict()
        )
        metrics: dict[str, Any] = {
            "train.loss": _metric(backend_result.loss, "loss", "text-sft-v1"),
            "train.tokens_per_second": _metric(
                backend_result.tokens_per_second, "tokens/second", "text-sft-v1"
            ),
        }
        result = RunResult(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            spec=spec,
            metrics=metrics,
            timings={"wall_seconds": _metric(elapsed, "seconds", "text-sft-v1")},
            artifacts=(adapter,),
            started_at=started,
            ended_at=_now(),
        )
        store.save_result(result)
        return result


def load_train_config(path: str | Path) -> TrainConfig:
    """Load strict JSON or YAML configuration without accepting unknown fields."""
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingError(f"cannot read training configuration: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingError("training configuration must contain an object")
    try:
        return TrainConfig.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"invalid training configuration: {exc}") from exc


def save_train_config(config: TrainConfig, path: str | Path) -> Path:
    """Atomically save canonical YAML that round-trips through TrainConfig."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config.to_dict(), sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target


def export_adapter(checkpoint_path: str | Path, destination: str | Path) -> Path:
    """Verify and atomically copy an adapter bundle without overwriting a target."""
    checkpoint_source = Path(checkpoint_path)
    try:
        checkpoint = CheckpointMetadata.from_dict(json.loads(checkpoint_source.read_text()))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrainingError(f"invalid checkpoint metadata: {exc}") from exc
    adapter_source = Path(checkpoint.adapter.uri)
    if not adapter_source.is_file():
        raise TrainingError("checkpoint adapter is missing")
    if hashlib.sha256(adapter_source.read_bytes()).hexdigest() != checkpoint.adapter.sha256:
        raise TrainingError("checkpoint adapter checksum does not match")
    target = Path(destination)
    if target.exists():
        raise TrainingError(f"export destination already exists: {target}")
    target.mkdir(parents=True)
    temporary = target / "adapters.safetensors.tmp"
    temporary.write_bytes(adapter_source.read_bytes())
    temporary.replace(target / "adapters.safetensors")
    _atomic_json(target / "mlx-one-checkpoint.json", checkpoint.to_dict())
    return target


def _materialize_dataset(records: list[dict[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "train.jsonl"
    target.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8"
    )


def _metric(value: float | None, unit: str, protocol: str) -> dict[str, Any]:
    return MetricValue(
        value=value,
        provenance=MetricProvenance.MEASURED
        if value is not None
        else MetricProvenance.NOT_AVAILABLE,
        unit=unit,
        protocol=protocol,
    ).to_dict()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat()
