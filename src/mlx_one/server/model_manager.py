"""Single-model lifecycle management for the native server."""

from __future__ import annotations

import gc
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from mlx_one.engine.model_pool import ModelPool
from mlx_one.text import LoadedTextModel, load_text_model

if TYPE_CHECKING:
    from mlx_one.models.audio.whisper.loading import LoadedWhisper


@dataclass(frozen=True)
class ModelMetadata:
    id: str
    architecture: str
    context_length: int
    parameter_count: int | None
    quantization: dict[str, Any]
    revision: str | None
    tasks: tuple[str, ...] = ("text-generation",)
    modalities: tuple[str, ...] = ("text",)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelManager:
    """Own exactly one model bundle for the lifetime of a server process."""

    def __init__(self, *, alias: str | None = None, context_length: int | None = None) -> None:
        self._pool = ModelPool()
        self._bundle_lease = None
        self._bundle_key = None
        self._unloading = False
        self._bundle: LoadedTextModel | None = None
        self._draft_bundle: LoadedTextModel | None = None
        self._transcription_bundle: LoadedWhisper | None = None
        self._transcription_bytes = 0
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
        checkpoint_path: str | Path | None = None,
        expected_files: dict[str, str] | None = None,
    ) -> LoadedTextModel:
        with self._lock:
            if self._bundle is not None or self._unloading:
                raise RuntimeError("a model is already loaded or draining")
            self._bundle_key = str(model_id) + "@" + str(revision) + ":" + str(checkpoint_path)
            identity_options = {}
            if checkpoint_path is not None:
                identity_options["checkpoint_path"] = checkpoint_path
            if expected_files is not None:
                identity_options["expected_files"] = expected_files
            self._bundle_lease = self._pool.acquire(self._bundle_key, lambda: _load_serving_model(
                model_id,
                revision=revision,
                offline=offline,
                cache_dir=cache_dir,
                tokenizer_source=tokenizer_source,
                **identity_options,
            ), measure=lambda bundle: getattr(bundle, "resident_bytes", 0))
            self._bundle = self._bundle_lease.runtime.value
            if (
                self._context_length is not None
                and self._context_length > self._bundle.context_length
            ):
                native = self._bundle.context_length
                self._bundle = None
                self._bundle_lease.release()
                self._pool.unload(self._bundle_key)
                raise ValueError(
                    f"context length {self._context_length} exceeds model maximum {native}"
                )
            return self._bundle

    def load_draft(
        self,
        model_id: str | Path,
        *,
        revision: str | None = None,
        offline: bool = False,
        cache_dir: str | Path | None = None,
        tokenizer_source: str | Path | None = None,
    ) -> LoadedTextModel:
        """Load one private draft model after verifying token-ID compatibility."""

        with self._lock:
            if self._bundle is None:
                raise RuntimeError("load the primary model before its draft model")
            if self._draft_bundle is not None:
                raise RuntimeError("a draft model is already loaded")
            draft = self._pool.executor.call(id(self), lambda: load_text_model(
                model_id,
                revision=revision,
                offline=offline,
                cache_dir=cache_dir,
                tokenizer_source=tokenizer_source,
            ))
            if _tokenizer_fingerprint(draft.tokenizer) != _tokenizer_fingerprint(
                self._bundle.tokenizer
            ):
                raise ValueError(
                    "draft and primary models must use identical token-ID semantics"
                )
            self._draft_bundle = draft
            return draft

    def unload(self) -> None:
        with self._lock:
            if self._unloading:
                raise RuntimeError("model is already draining")
            self._unloading = True
            lease, key = self._bundle_lease, self._bundle_key
        try:
            if lease is not None:
                lease.release()
                self._pool.unload(key)
        finally:
            with self._lock:
                self._transcription_bundle = None
                self._transcription_bytes = 0
                self._draft_bundle = None
                self._bundle = None
                self._bundle_lease = None
                self._unloading = False
        gc.collect()
        try:
            mx = sys.modules.get("mlx.core")
            if mx is not None:
                self._pool.executor.call(id(self), mx.clear_cache)
        except Exception:
            pass

    def current_model(self) -> LoadedTextModel:
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no model is loaded")
            return self._bundle

    def model_lease(self):
        with self._lock:
            if self._unloading:
                raise RuntimeError("model is draining")
            bundle = self.current_model()
            return self._pool.acquire(self._bundle_key, lambda: bundle)

    def draft_model(self) -> LoadedTextModel | None:
        with self._lock:
            return self._draft_bundle

    def load_transcription(
        self,
        model_id: str | Path,
        *,
        revision: str | None = None,
        offline: bool = False,
        cache_dir: str | Path | None = None,
    ) -> LoadedWhisper:
        """Load one private native Whisper bundle for speech-to-text requests."""

        from mlx_one.models.audio.whisper.loading import load_whisper

        with self._lock:
            if self._bundle is None:
                raise RuntimeError("load the primary model before its transcription model")
            if self._transcription_bundle is not None:
                raise RuntimeError("a transcription model is already loaded")
            before = _active_memory()
            bundle = self._pool.executor.call(id(self), lambda: load_whisper(
                model_id,
                revision=revision,
                offline=offline,
                cache_dir=cache_dir,
            ))
            self._transcription_bundle = bundle
            after = _active_memory()
            if before is not None and after is not None:
                self._transcription_bytes = max(0, after - before)
            return bundle

    def transcription_model(self) -> LoadedWhisper | None:
        with self._lock:
            return self._transcription_bundle

    @property
    def transcription_bytes(self) -> int:
        return self._transcription_bytes

    def list_models(self) -> tuple[ModelMetadata, ...]:
        with self._lock:
            return () if self._bundle is None else (self.model_info(),)

    def model_info(self) -> ModelMetadata:
        bundle = self.current_model()
        if hasattr(bundle, "audio"):
            from mlx_one.core.registry import get_registration

            registration = get_registration(bundle.architecture)
            return ModelMetadata(id=self._alias or bundle.model_id,
                architecture=bundle.architecture, context_length=bundle.context_length,
                parameter_count=None, quantization=bundle.quantization, revision=bundle.revision,
                tasks=tuple(sorted(registration.tasks)), modalities=("audio",))
        tasks = ("text-generation",)
        vision = hasattr(bundle, "processor")
        if vision:
            tasks += ("vision-language",)
        if self.transcription_model() is not None:
            tasks += ("transcription",)
        return ModelMetadata(
            id=self._alias or bundle.model_id,
            architecture=bundle.architecture,
            context_length=self._context_length or bundle.context_length,
            parameter_count=bundle.parameter_count,
            quantization=dict(bundle.quantization),
            revision=bundle.revision,
            tasks=tasks,
            modalities=("text", "image") if vision else ("text",),
        )


def _active_memory() -> int | None:
    try:
        mx = sys.modules.get("mlx.core")
        getter = getattr(mx, "get_active_memory", None)
        return int(getter()) if callable(getter) else None
    except Exception:
        return None


def _tokenizer_fingerprint(tokenizer: Any) -> tuple[object, ...]:
    vocabulary: Any = getattr(tokenizer, "encoder", None)
    if vocabulary is None:
        native = getattr(tokenizer, "_tokenizer", None)
        get_vocab = getattr(native, "get_vocab", None)
        vocabulary = get_vocab(with_added_tokens=True) if callable(get_vocab) else None
    if not isinstance(vocabulary, dict):
        raise ValueError("cannot verify draft tokenizer vocabulary")
    normalized = tuple(
        sorted((str(token), int(token_id)) for token, token_id in vocabulary.items())
    )
    return (
        normalized,
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    )


def _load_serving_model(model_id, **options):
    import json

    from mlx_one.audio.loading import audio_model_type
    from mlx_one.core.registry import get_registration
    from mlx_one.server.audio import serving_audio

    local = Path(model_id).expanduser()
    if (local / "config.json").is_file():
        payload = json.loads((local / "config.json").read_text())
        family = audio_model_type(payload)
        if get_registration(family).modality in {"asr", "tts", "audio-language"}:
            return serving_audio(model_id, local, **{k: v for k, v in options.items()
                if k in {"revision", "offline", "cache_dir"}})

    from mlx_one.text.loading import TextModelLoadError
    from mlx_one.vision.loading import attach_vision_processor, load_vlm_model

    try:
        bundle = load_text_model(model_id, **options)
    except TextModelLoadError as exc:
        if "has no native text loader" not in str(exc):
            raise
        if (
            options.get("checkpoint_path")
            or options.get("expected_files")
            or options.get("tokenizer_source")
        ):
            raise ValueError(
                "VLM serving requires tokenizer and processor in the same checkpoint"
            ) from exc
        return load_vlm_model(
            model_id,
            **{k: v for k, v in options.items() if k in {"revision", "offline", "cache_dir"}},
        )
    return attach_vision_processor(bundle)
