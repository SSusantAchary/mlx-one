"""Explicitly limited Unsloth-style loading and PEFT facade."""

from __future__ import annotations

from typing import Any


def _load_bundle(model_name: str, revision: str | None) -> Any:
    from mlx_one.text import load_text_model

    return load_text_model(model_name, revision=revision)


def _apply_lora(model: Any, **kwargs: Any) -> tuple[str, ...]:
    from mlx_one.tuning import apply_lora

    return apply_lora(model, **kwargs)


class FastLanguageModel:
    """Supported subset of Unsloth's model-loading convenience API."""

    @staticmethod
    def from_pretrained(
        *,
        model_name: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported FastLanguageModel arguments: {names}")
        if max_seq_length < 1:
            raise ValueError("max_seq_length must be at least 1")
        bundle = _load_bundle(model_name, revision)
        model, tokenizer = bundle.model, bundle.tokenizer
        model._mlx_one_model_name = model_name
        model._mlx_one_revision = revision
        model._mlx_one_max_seq_length = max_seq_length
        model._mlx_one_load_in_4bit = load_in_4bit
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model: Any,
        *,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported PEFT arguments: {names}")
        if r < 1 or lora_alpha <= 0 or not 0 <= lora_dropout < 1:
            raise ValueError("invalid LoRA rank, alpha, or dropout")
        keys = tuple(target_modules or ())
        layers = getattr(getattr(model, "model", model), "layers", ())
        layer_count = len(layers)
        _apply_lora(
            model,
            num_layers=layer_count,
            rank=r,
            scale=lora_alpha / r,
            dropout=lora_dropout,
            target_modules=keys,
        )
        model._mlx_one_peft_config = {
            "lora_rank": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": keys,
        }
        return model
