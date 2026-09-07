import json
from pathlib import Path

import pytest

from mlx_one.schemas import TrainConfig, TrainingMethod
from mlx_one.training import (
    BackendTrainingResult,
    SFTTrainer,
    TrainingError,
    export_adapter,
    load_train_config,
    save_train_config,
)


class FakeBackend:
    def train(self, *, model_id, revision, data_dir, config, resume_adapter):
        assert model_id == "Qwen/Qwen2.5-Coder-1.5B"
        assert len(revision) == 40
        assert (data_dir / "train.jsonl").is_file()
        assert resume_adapter is None
        adapter = Path(config.output_dir) / "adapters" / "adapters.safetensors"
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(b"safe adapter")
        return BackendTrainingResult(adapter, loss=0.125, tokens_per_second=42.0)


def test_native_trainer_records_adapter_lineage_and_exports(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"prompt":"1+1=","completion":"2"}\n', encoding="utf-8")
    output = tmp_path / "output"
    config = TrainConfig(output_dir=str(output), max_steps=2, save_steps=1)
    trainer = SFTTrainer(
        model="Qwen/Qwen2.5-Coder-1.5B",
        revision="a" * 40,
        train_dataset=dataset,
        args=config,
        backend=FakeBackend(),
    )

    result = trainer.train()
    checkpoint = output / "adapters" / "mlx-one-checkpoint.json"
    exported = export_adapter(checkpoint, tmp_path / "export")

    assert result.status.value == "completed"
    assert result.metrics["train.loss"]["provenance"] == "measured"
    assert result.artifacts[0].metadata["base_revision"] == "a" * 40
    assert (exported / "adapters.safetensors").read_bytes() == b"safe adapter"


def test_training_failure_is_preserved_as_terminal_run(tmp_path: Path) -> None:
    class FailingBackend:
        def train(self, **_kwargs):
            raise TrainingError("expected failure")

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    output = tmp_path / "output"
    trainer = SFTTrainer(
        model="Qwen/Qwen2.5-Coder-1.5B",
        revision="a" * 40,
        train_dataset=dataset,
        args=TrainConfig(output_dir=str(output)),
        backend=FailingBackend(),
    )

    with pytest.raises(TrainingError, match="expected failure"):
        trainer.train()

    result_files = list((output / "runs").glob("*/result.json"))
    assert len(result_files) == 1
    assert json.loads(result_files[0].read_text())["status"] == "failed"


def test_training_config_yaml_round_trip_and_qlora_preflight(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    config = TrainConfig(output_dir=str(tmp_path / "out"))
    save_train_config(config, path)

    assert load_train_config(path) == config
    with pytest.raises(TrainingError, match="verified 4-bit base"):
        SFTTrainer(
            model="example/base",
            revision="a" * 40,
            train_dataset=path,
            args=TrainConfig(output_dir=str(tmp_path / "qlora"), method=TrainingMethod.QLORA),
        )
