"""Small TRL-shaped facade sharing mlx-one's native training contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from mlx_one.schemas import TrainConfig as SFTConfig
from mlx_one.training import SFTTrainer as NativeSFTTrainer
from mlx_one.training import TrainingBackend


class SFTTrainer:
    """Translate the documented TRL-shaped subset to the native trainer."""

    def __init__(
        self,
        *,
        model: Any,
        train_dataset: str | Path,
        args: SFTConfig,
        backend: TrainingBackend | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported SFTTrainer arguments: {names}")
        model_name = (
            model if isinstance(model, str) else getattr(model, "_mlx_one_model_name", None)
        )
        revision = getattr(model, "_mlx_one_revision", None)
        if not model_name or not revision:
            raise ValueError(
                "compatibility training requires a model loaded with "
                "FastLanguageModel.from_pretrained(revision=<exact SHA>)"
            )
        peft = getattr(model, "_mlx_one_peft_config", None)
        if peft:
            args = replace(args, **peft)
        self._native = NativeSFTTrainer(
            model=model_name,
            revision=revision,
            train_dataset=train_dataset,
            args=args,
            backend=backend,
        )

    def train(self, *, resume_from_checkpoint: str | Path | None = None):
        return self._native.train(resume_from_checkpoint=resume_from_checkpoint)


__all__ = ["SFTConfig", "SFTTrainer"]
