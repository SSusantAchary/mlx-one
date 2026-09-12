"""Single-model lifecycle management for the native server."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from mlx_one.text import LoadedTextModel, load_text_model


@dataclass(frozen=True)
class ModelMetadata:
    id: str
    architecture: str
    context_length: int
    parameter_count: int | None
    quantization: dict[str, Any]
    revision: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelManager:
    """Own exactly one model bundle for the lifetime of a server process."""

    def __init__(self, *, alias: str | None = None, context_length: int | None = None) -> None:
        self._bundle: LoadedTextModel | None = None
        self._lock = RLock()
        self._alias = alias
        self._context_length = context_length

    def load(
        self,
        model_id: str | Path,
        *,
        revision: str | None = None,
        offline: bool = False,
        cache_dir: str | Path | None = None,
        tokenizer_source: str | Path | None = None,
    ) -> LoadedTextModel:
        with self._lock:
            if self._bundle is not None:
                raise RuntimeError("a model is already loaded")
            self._bundle = load_text_model(
                model_id,
                revision=revision,
                offline=offline,
                cache_dir=cache_dir,
                tokenizer_source=tokenizer_source,
            )
            if (
                self._context_length is not None
                and self._context_length > self._bundle.context_length
            ):
                native = self._bundle.context_length
                self._bundle = None
                raise ValueError(
                    f"context length {self._context_length} exceeds model maximum {native}"
                )
            return self._bundle

    def unload(self) -> None:
        with self._lock:
            self._bundle = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def current_model(self) -> LoadedTextModel:
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no model is loaded")
            return self._bundle

    def list_models(self) -> tuple[ModelMetadata, ...]:
        with self._lock:
            return () if self._bundle is None else (self.model_info(),)

    def model_info(self) -> ModelMetadata:
        bundle = self.current_model()
        return ModelMetadata(
            id=self._alias or bundle.model_id,
            architecture=bundle.architecture,
            context_length=self._context_length or bundle.context_length,
            parameter_count=bundle.parameter_count,
            quantization=dict(bundle.quantization),
            revision=bundle.revision,
        )
