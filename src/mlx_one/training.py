"""Native mlx-one SFT orchestration implemented directly with Apple MLX."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from mlx_one.datasets import dataset_spec_from_path, load_records
from mlx_one.hardware import detect_hardware
from mlx_one.provenance import collect_software_provenance
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
    ResumeSemantics,
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
    loss_history: tuple[float, ...] = ()
    peak_memory_bytes: int | None = None
    state_artifacts: tuple[ArtifactRef, ...] = ()
    resumable_state: tuple[str, ...] = ()
    resume_semantics: ResumeSemantics = ResumeSemantics.ADAPTER_ONLY


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
    """Native Apple MLX LoRA/QLoRA training backend."""

    def train(
        self,
        *,
        model_id: str,
        revision: str,
        data_dir: Path,
        config: TrainConfig,
        resume_adapter: Path | None,
    ) -> BackendTrainingResult:
        if config.optimizer.lower() != "adamw":
            raise TrainingError("native training currently supports optimizer='adamw'")
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            from mlx.utils import tree_map

            from mlx_one.datasets import tokenize_record
            from mlx_one.text import load_text_model
            from mlx_one.tuning import apply_lora, load_adapter, save_adapter

            mx.random.seed(config.seed)
            bundle = load_text_model(model_id, revision=revision)
            if bundle.architecture not in {"qwen2", "llama"}:
                raise TrainingError(
                    f"native LoRA training is unsupported for {bundle.architecture!r}"
                )
            if config.method is TrainingMethod.QLORA and not bundle.quantization:
                raise TrainingError("QLoRA requires a native four-bit base checkpoint")
            adapter_root = resume_adapter.parent if resume_adapter is not None else None
            if adapter_root is not None:
                adapter_config = load_adapter(
                    bundle.model,
                    adapter_root,
                    base_model_id=model_id,
                    base_revision=revision,
                )
                target_paths = tuple(adapter_config.get("resolved_targets", ()))
            else:
                target_paths = apply_lora(
                    bundle.model,
                    num_layers=config.lora_layers,
                    rank=config.lora_rank,
                    scale=config.lora_alpha / config.lora_rank,
                    dropout=config.lora_dropout,
                    target_modules=config.target_modules,
                )
            if config.gradient_checkpointing:
                _enable_gradient_checkpointing(bundle.model, mx)
            records = [
                json.loads(line)
                for line in (data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            prepared = []
            for record in records:
                if "messages" in record:
                    if bundle.chat_template is None:
                        raise TrainingError("messages training data requires a chat template")
                    prompt = bundle.chat_template.render(
                        record["messages"][:-1], add_generation_prompt=True
                    )
                    complete = bundle.chat_template.render(
                        record["messages"], add_generation_prompt=False
                    )
                    prompt_ids = bundle.tokenizer.encode(prompt)
                    ids = bundle.tokenizer.encode(complete)
                    cutoff = min(len(prompt_ids), len(ids))
                    labels = [-100] * cutoff + ids[cutoff:]
                    ids, labels = ids[: config.max_seq_length], labels[: config.max_seq_length]
                    if not ids or all(value == -100 for value in labels):
                        raise TrainingError("training record has no supervised tokens")
                    prepared.append((ids, labels))
                else:
                    item = tokenize_record(
                        record,
                        bundle.tokenizer,
                        max_length=config.max_seq_length,
                        response_only=True,
                    )
                    prepared.append((list(item.input_ids), list(item.labels)))
            if len(prepared) < config.batch_size:
                raise TrainingError("training dataset is smaller than batch_size")
            optimizer = optim.AdamW(learning_rate=config.learning_rate)

            def loss_fn(model: Any, inputs: Any, labels: Any) -> tuple[Any, Any]:
                targets = labels[:, 1:]
                mask = targets != -100
                safe_targets = mx.where(mask, targets, 0)
                logits = model(inputs[:, :-1]).logits
                losses = nn.losses.cross_entropy(logits, safe_targets) * mask
                token_count = mask.sum()
                return losses.astype(mx.float32).sum() / mx.maximum(token_count, 1), token_count

            value_and_grad = nn.value_and_grad(bundle.model, loss_fn)
            gradient = None
            losses: list[float] = []
            token_total = 0
            started = time.perf_counter()
            bundle.model.train()
            for step in range(config.max_steps):
                batch = [
                    prepared[(step * config.batch_size + i) % len(prepared)]
                    for i in range(config.batch_size)
                ]
                inputs, labels = _pad_training_batch(batch, mx)
                (loss, token_count), current = value_and_grad(bundle.model, inputs, labels)
                gradient = current if gradient is None else tree_map(
                    lambda left, right: left + right, gradient, current
                )
                update = (
                    (step + 1) % config.gradient_accumulation_steps == 0
                    or step + 1 == config.max_steps
                )
                if update:
                    divisor = (
                        config.gradient_accumulation_steps
                        if (step + 1) % config.gradient_accumulation_steps == 0
                        else (step % config.gradient_accumulation_steps) + 1
                    )
                    gradient = tree_map(
                        lambda value, divisor=divisor: value / divisor, gradient
                    )
                    optimizer.update(bundle.model, gradient)
                    gradient = None
                mx.eval(bundle.model.parameters(), optimizer.state, loss, token_count)
                losses.append(float(loss.item()))
                token_total += int(token_count.item())
            elapsed = time.perf_counter() - started
            adapter_dir = Path(config.output_dir).expanduser().resolve() / "adapters"
            adapter_config = {
                "schema_version": 1,
                "base_model_id": model_id,
                "base_revision": revision,
                "fine_tune_type": config.method.value,
                "num_layers": config.lora_layers,
                "rank": config.lora_rank,
                "scale": config.lora_alpha / config.lora_rank,
                "dropout": config.lora_dropout,
                "target_modules": list(config.target_modules),
                "resolved_targets": list(target_paths),
            }
            adapter = save_adapter(bundle.model, adapter_dir, adapter_config)
            peak = int(mx.get_peak_memory())
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError(f"native MLX training failed: {exc}") from exc
        return BackendTrainingResult(
            adapter_path=adapter,
            loss=losses[-1] if losses else None,
            tokens_per_second=token_total / elapsed if elapsed else None,
            loss_history=tuple(losses),
            peak_memory_bytes=peak,
            resumable_state=(),
            resume_semantics=ResumeSemantics.ADAPTER_ONLY,
        )


def _pad_training_batch(batch: list[tuple[list[int], list[int]]], mx: Any) -> tuple[Any, Any]:
    width = max(len(ids) for ids, _ in batch)
    if width < 2:
        raise TrainingError("training sequences must contain at least two tokens")
    inputs = [ids + [0] * (width - len(ids)) for ids, _ in batch]
    labels = [values + [-100] * (width - len(values)) for _, values in batch]
    return mx.array(inputs), mx.array(labels)


def _enable_gradient_checkpointing(model: Any, mx: Any) -> None:
    """Checkpoint the decoder layer call for the current model class."""
    from mlx_one.tuning import _decoder_layers

    layer_type = type(_decoder_layers(model)[0])
    if hasattr(layer_type, "_mlx_one_uncheckpointed_call"):
        return
    original = layer_type.__call__
    layer_type._mlx_one_uncheckpointed_call = original

    def checkpointed(layer: Any, *args: Any, **kwargs: Any) -> Any:
        def inner(parameters: Any, *inner_args: Any, **inner_kwargs: Any) -> Any:
            layer.update(parameters)
            return original(layer, *inner_args, **inner_kwargs)

        return mx.checkpoint(inner)(layer.trainable_parameters(), *args, **kwargs)

    layer_type.__call__ = checkpointed


def _is_native_gpt2(model: str) -> bool:
    normalized = model.rstrip("/").lower()
    if normalized.rsplit("/", 1)[-1] in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}:
        return True
    path = Path(model).expanduser()
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("model_type") == "gpt2"


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
        if _is_native_gpt2(model):
            raise TrainingError(
                "native GPT-2 training and adapters are not implemented; inference only"
            )
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
        """Normalize data, execute native training, and persist terminal evidence."""
        output = Path(self.args.output_dir).expanduser().resolve()
        store = RunStore(output / "runs")
        dataset = dataset_spec_from_path(self.train_dataset, response_only=True)
        resume_adapter = None
        parent_run_ids: tuple[str, ...] = ()
        if resume_from_checkpoint is not None:
            resume_adapter, parent_run_id = _resolve_resume_checkpoint(
                resume_from_checkpoint, self.model, self.revision
            )
            parent_run_ids = (parent_run_id,)
        run_id = f"train-{uuid.uuid4().hex[:12]}"
        model = ModelSpec(model_id=self.model, revision=self.revision, modality=Modality.TEXT)
        spec = RunSpec(
            run_id=run_id,
            operation=Operation.TRAIN,
            model=model,
            hardware=detect_hardware(),
            profile="text-sft-v1",
            dataset_revisions={dataset.dataset_id: dataset.revision},
            seed=self.args.seed,
            parent_run_ids=parent_run_ids,
            metadata={
                "train_config": self.args.to_dict(),
                "metric_policy": "explicit-provenance-v1",
                "resume_semantics": ResumeSemantics.ADAPTER_ONLY.value
                if resume_adapter
                else "fresh",
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
                resume_adapter=resume_adapter,
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
        adapter_config_path = backend_result.adapter_path.parent / "adapter_config.json"
        adapter_config_artifacts: tuple[ArtifactRef, ...] = ()
        if adapter_config_path.is_file():
            adapter_config_artifacts = (
                ArtifactRef(
                    uri=str(adapter_config_path),
                    kind="adapter-config",
                    sha256=hashlib.sha256(adapter_config_path.read_bytes()).hexdigest(),
                    media_type="application/json",
                ),
            )
        checkpoint = CheckpointMetadata(
            run_id=run_id,
            base_model_id=self.model,
            base_revision=self.revision,
            method=self.args.method,
            step=self.args.max_steps,
            adapter=adapter,
            config_sha256=config_hash,
            resumable_state=("adapter-weights", *backend_result.resumable_state),
            resume_semantics=backend_result.resume_semantics,
            state_artifacts=(*adapter_config_artifacts, *backend_result.state_artifacts),
        )
        checkpoint_path = backend_result.adapter_path.parent / "mlx-one-checkpoint.json"
        _atomic_json(checkpoint_path, checkpoint.to_dict())
        checkpoint_artifact = ArtifactRef(
            uri=str(checkpoint_path),
            kind="checkpoint-metadata",
            sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            media_type="application/json",
            metadata={"resume_semantics": checkpoint.resume_semantics.value},
        )
        metrics: dict[str, Any] = {
            "train.loss": _metric(backend_result.loss, "loss", "text-sft-v1"),
            "train.initial_loss": _metric(
                backend_result.loss_history[0] if backend_result.loss_history else None,
                "loss",
                "text-sft-v1",
            ),
            "train.loss_decreased": _metric(
                float(backend_result.loss_history[-1] < backend_result.loss_history[0])
                if len(backend_result.loss_history) > 1
                else None,
                "boolean",
                "text-sft-v1",
            ),
            "train.tokens_per_second": _metric(
                backend_result.tokens_per_second, "tokens/second", "text-sft-v1"
            ),
            "memory.peak_bytes": _metric(
                float(backend_result.peak_memory_bytes)
                if backend_result.peak_memory_bytes is not None
                else None,
                "bytes",
                "text-sft-v1",
            ),
        }
        nonzero_fraction = _adapter_nonzero_fraction(backend_result.adapter_path)
        metrics["train.adapter_nonzero_fraction"] = _metric(
            nonzero_fraction, "ratio", "text-sft-v1"
        )
        metrics["train.update_verified"] = _metric(
            float(
                len(backend_result.loss_history) > 1
                and backend_result.loss_history[-1] < backend_result.loss_history[0]
                and nonzero_fraction is not None
                and nonzero_fraction > 0
            )
            if len(backend_result.loss_history) > 1 and nonzero_fraction is not None
            else None,
            "boolean",
            "text-sft-v1",
        )
        result = RunResult(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            spec=spec,
            software=collect_software_provenance(),
            metrics=metrics,
            timings={"wall_seconds": _metric(elapsed, "seconds", "text-sft-v1")},
            artifacts=(
                adapter,
                checkpoint_artifact,
                *adapter_config_artifacts,
                *backend_result.state_artifacts,
            ),
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


def export_adapter(
    checkpoint_path: str | Path,
    destination: str | Path,
    *,
    smoke_prompt: str | None = None,
    max_tokens: int = 16,
) -> Path:
    """Verify and atomically copy an adapter bundle without overwriting a target."""
    checkpoint_source = Path(checkpoint_path)
    try:
        checkpoint = CheckpointMetadata.from_dict(json.loads(checkpoint_source.read_text()))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrainingError(f"invalid checkpoint metadata: {exc}") from exc
    adapter_source = _artifact_path(checkpoint_source, checkpoint.adapter.uri)
    if not adapter_source.is_file():
        raise TrainingError("checkpoint adapter is missing")
    if hashlib.sha256(adapter_source.read_bytes()).hexdigest() != checkpoint.adapter.sha256:
        raise TrainingError("checkpoint adapter checksum does not match")
    adapter_config_source = adapter_source.parent / "adapter_config.json"
    if not adapter_config_source.is_file():
        raise TrainingError("checkpoint adapter_config.json is missing")
    target = Path(destination)
    if target.exists():
        raise TrainingError(f"export destination already exists: {target}")
    staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    try:
        (staging / "adapters.safetensors").write_bytes(adapter_source.read_bytes())
        (staging / "adapter_config.json").write_bytes(adapter_config_source.read_bytes())
        exported_checkpoint = CheckpointMetadata(
            run_id=checkpoint.run_id,
            base_model_id=checkpoint.base_model_id,
            base_revision=checkpoint.base_revision,
            method=checkpoint.method,
            step=checkpoint.step,
            adapter=ArtifactRef(
                uri="adapters.safetensors",
                kind=checkpoint.adapter.kind,
                sha256=checkpoint.adapter.sha256,
                media_type=checkpoint.adapter.media_type,
                metadata=checkpoint.adapter.metadata,
            ),
            config_sha256=checkpoint.config_sha256,
            resumable_state=checkpoint.resumable_state,
            resume_semantics=checkpoint.resume_semantics,
            state_artifacts=checkpoint.state_artifacts,
            created_at=checkpoint.created_at,
        )
        _atomic_json(staging / "mlx-one-checkpoint.json", exported_checkpoint.to_dict())
        _atomic_json(
            staging / "manifest.json",
            {
                "schema_version": "1.0",
                "base_model_id": checkpoint.base_model_id,
                "base_revision": checkpoint.base_revision,
                "files": {
                    "adapters.safetensors": checkpoint.adapter.sha256,
                    "adapter_config.json": hashlib.sha256(
                        adapter_config_source.read_bytes()
                    ).hexdigest(),
                },
            },
        )
        if smoke_prompt is not None:
            from mlx_one.generation import generate_predictions

            predictions = generate_predictions(
                checkpoint.base_model_id,
                checkpoint.base_revision,
                [smoke_prompt],
                max_tokens=max_tokens,
                adapter_path=staging,
            )
            if len(predictions) != 1 or not predictions[0].strip():
                raise TrainingError("exported adapter smoke test produced no output")
            _atomic_json(
                staging / "smoke-test.json",
                {
                    "schema_version": "1.0",
                    "base_model_id": checkpoint.base_model_id,
                    "base_revision": checkpoint.base_revision,
                    "prompt_sha256": hashlib.sha256(smoke_prompt.encode()).hexdigest(),
                    "output_sha256": hashlib.sha256(predictions[0].encode()).hexdigest(),
                    "max_tokens": max_tokens,
                    "passed": True,
                },
            )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _resolve_resume_checkpoint(
    checkpoint_path: str | Path, model_id: str, revision: str
) -> tuple[Path, str]:
    source = Path(checkpoint_path)
    try:
        checkpoint = CheckpointMetadata.from_dict(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrainingError(f"invalid resume checkpoint: {exc}") from exc
    if checkpoint.base_model_id != model_id or checkpoint.base_revision != revision:
        raise TrainingError("resume checkpoint base model ancestry does not match")
    adapter = _artifact_path(source, checkpoint.adapter.uri)
    if not adapter.is_file():
        raise TrainingError("resume checkpoint adapter is missing")
    if hashlib.sha256(adapter.read_bytes()).hexdigest() != checkpoint.adapter.sha256:
        raise TrainingError("resume checkpoint adapter checksum does not match")
    return adapter, checkpoint.run_id


def _artifact_path(metadata_path: Path, uri: str) -> Path:
    artifact = Path(uri)
    return artifact if artifact.is_absolute() else metadata_path.parent / artifact


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


def _adapter_nonzero_fraction(path: Path) -> float | None:
    try:
        from safetensors import safe_open

        nonzero = 0
        total = 0
        with safe_open(path, framework="numpy") as weights:
            for key in weights.keys():
                tensor = weights.get_tensor(key)
                nonzero += int((tensor != 0).sum())
                total += int(tensor.size)
        return nonzero / total if total else None
    except Exception:
        return None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat()
