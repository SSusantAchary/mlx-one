"""Bounded Qwen accumulated-utterance sessions and incremental local-file PCM.

Rollback behavior is adapted from QwenLM/Qwen3-ASR, Apache-2.0 (see model NOTICE).
These are not causal audio-encoder caches or append-only transcript deltas.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from array import array
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mlx_one.audio.loading import load_audio_model
from mlx_one.audio.moonshine import AudioResourceLimitError
from mlx_one.audio.qwen import check_resources, decode_qwen, provenance
from mlx_one.audio.schemas import ASRDecodeOptions, AudioStreamEvent, QwenASROptions
from mlx_one.models.audio.parakeet.loading import LoadedParakeet
from mlx_one.models.audio.qwen3_asr.loading import LoadedQwenAudio
from mlx_one.models.audio.qwen3_asr.processing import language_name, parse_output


class QwenAudioStream:
    """One session, one explicitly bounded utterance, one shared read-only model."""

    def __init__(
        self,
        bundle: LoadedQwenAudio,
        *,
        model_name: str,
        sample_rate: int,
        language: str | None,
        options: ASRDecodeOptions | None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        if bundle.model.config.model_type != "qwen3_asr":
            raise ValueError("only Qwen ASR supports accumulated-utterance streaming")
        if isinstance(sample_rate, bool) or sample_rate != 16000:
            raise ValueError("audio streams require an explicit sample_rate=16000")
        if options is not None and not isinstance(options, ASRDecodeOptions):
            raise ValueError("Qwen streaming requires ASRDecodeOptions or QwenASROptions")
        self._bundle = bundle
        self._model_name = model_name
        self._language = language_name(language)
        self._options = options or QwenASROptions()
        self._cancelled = cancelled
        self._closed = False
        self._finished = False
        self._pending = array("f")
        self._audio = array("f")
        self._raw = ""
        self._chunks = 0
        self._sequence = 0
        self._received = 0
        self._last_language = None
        self._last_text = ""

    def __enter__(self) -> QwenAudioStream:
        self._ensure_open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("audio stream is closed")

    def _check(self) -> None:
        self._ensure_open()
        try:
            check_resources(self._cancelled)
        except Exception:
            self.close()
            raise

    def _prefix(self, final_tail: bool) -> str:
        if self._chunks < 2:
            return ""
        ids = self._bundle.tokenizer.encode(self._raw, add_special_tokens=False).ids
        if final_tail:
            # Preserve the official finish-time max(1, ...) behavior separately.
            return self._bundle.tokenizer.decode(
                ids[: max(1, len(ids) - 5)], skip_special_tokens=False
            )
        rollback = 5
        while True:
            end = max(0, len(ids) - rollback)
            prefix = (
                self._bundle.tokenizer.decode(ids[:end], skip_special_tokens=False) if end else ""
            )
            if "\ufffd" not in prefix or not end:
                return prefix
            rollback += 1

    def _decode(self, *, final_tail: bool) -> None:
        import mlx.core as mx

        self._raw, _ = decode_qwen(
            self._bundle,
            mx.array(self._audio, dtype=mx.float32),
            language=self._language,
            context=getattr(self._options, "context", ""),
            prefix=self._prefix(final_tail),
            max_tokens=self._options.max_tokens or 512,
            offline_padding=False,
            cancelled=self._cancelled,
        )
        self._last_language, self._last_text = parse_output(self._raw, self._language)
        self._chunks += 1

    def _event(self, kind: str) -> AudioStreamEvent:
        event = AudioStreamEvent(
            kind,
            self._last_text,
            self._sequence,
            self._received / 16000 if kind == "final" else len(self._audio) / 16000,
            self._model_name,
            self._bundle.revision,
            language=self._last_language,
            provenance={
                **provenance(self._bundle),
                "streaming": "accumulated-utterance",
                "hypothesis": "complete-current",
                "chunk_seconds": 2.0,
                "initial_unfixed_chunks": 2,
                "rollback_tokens": 5,
                "utterance_limit_seconds": 30.0,
            },
        )
        self._sequence += 1
        return event

    def push(self, pcm: Any, *, sample_rate: int | None = None) -> tuple[AudioStreamEvent, ...]:
        self._check()
        if self._finished:
            raise RuntimeError("finish has been called; reset before subsequent PCM input")
        if sample_rate is not None and (isinstance(sample_rate, bool) or sample_rate != 16000):
            raise ValueError("PCM rate cannot change within a session")
        if isinstance(pcm, (str, bytes, bytearray)) or getattr(pcm, "ndim", 1) != 1:
            raise ValueError(
                "PCM input must be a mono sequence of float samples, not encoded bytes"
            )
        if self._received + len(pcm) > 30 * 16000:
            raise AudioResourceLimitError(
                "Qwen utterance exceeds 30 seconds; finish/reset explicitly"
            )
        try:
            samples = pcm.tolist() if hasattr(pcm, "tolist") else pcm
            if str(getattr(pcm, "dtype", "")).rsplit(".", 1)[-1] == "int16":
                # The upstream stream accepts explicitly typed signed PCM16.
                # Untyped sequences represent floating PCM, not encoded bytes.
                samples = (value / 32768.0 for value in samples)
            values = array("f", samples)
        except (TypeError, OverflowError) as exc:
            raise ValueError("PCM input must contain finite float samples") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("PCM input must contain finite float samples")
        self._pending.extend(values)
        self._received += len(values)
        events = []
        try:
            while len(self._pending) >= 32000:
                self._check()
                self._audio.extend(self._pending[:32000])
                del self._pending[:32000]
                self._decode(final_tail=False)
                events.append(self._event("partial"))
        except Exception:
            self.close()
            raise
        return tuple(events)

    def finish(self) -> AudioStreamEvent:
        self._check()
        if self._finished:
            raise RuntimeError("finish has already been called; reset to start another utterance")
        try:
            if self._pending:
                self._audio.extend(self._pending)
                self._pending = array("f")
                self._decode(final_tail=True)
            event = self._event("final")
            self._finished = True
            self._audio = array("f")
            self._raw = ""
            return event
        except Exception:
            self.close()
            raise

    def reset(self) -> None:
        self._check()
        self._pending, self._audio = array("f"), array("f")
        self._raw, self._last_text = "", ""
        self._last_language = None
        self._received = self._chunks = self._sequence = 0
        self._finished = False

    def close(self) -> None:
        self._pending, self._audio = array("f"), array("f")
        self._raw, self._last_text = "", ""
        self._bundle = None
        self._cancelled = self._options = None
        self._closed = True

    cancel = close


def open_audio_stream(
    model: str | Path | LoadedQwenAudio,
    *,
    sample_rate: int,
    language: str | None = None,
    options: ASRDecodeOptions | None = None,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> QwenAudioStream:
    if isinstance(sample_rate, bool) or sample_rate != 16000:
        raise ValueError("audio streams require an explicit sample_rate=16000")
    if isinstance(model, LoadedParakeet):
        raise ValueError("Parakeet TDT is offline-only; native streaming is not implemented")
    bundle = (
        model
        if isinstance(model, LoadedQwenAudio)
        else load_audio_model(model, revision=revision, offline=offline, cache_dir=cache_dir)
    )
    if not isinstance(bundle, LoadedQwenAudio):
        raise ValueError(
            "native streaming is currently implemented only for Qwen ASR; "
            "original Moonshine stays offline-only"
        )
    return QwenAudioStream(
        bundle,
        model_name=str(bundle.path) if isinstance(model, LoadedQwenAudio) else str(model),
        sample_rate=sample_rate,
        language=language,
        options=options,
        cancelled=cancelled,
    )


@contextmanager
def decoded_pcm_chunks(
    path: str | Path, *, chunk_samples: int = 16000, cancelled: Callable[[], bool] | None = None
) -> Iterator[Iterator[array]]:
    """Incremental mono float32/16-kHz decoding; context exit always reaps FFmpeg."""
    if (
        isinstance(chunk_samples, bool)
        or not isinstance(chunk_samples, int)
        or not 1 <= chunk_samples <= 32000
    ):
        raise ValueError("decoder chunk_samples must be in [1, 32000]")
    source = Path(path)
    if not source.is_file():
        raise ValueError("audio file does not exist")
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required for incremental local-file decoding")
    command = [
        executable,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-",
    ]
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
        try:

            def chunks() -> Iterator[array]:
                while True:
                    check_resources(cancelled)
                    data = process.stdout.read(chunk_samples * 4)
                    if not data:
                        break
                    if len(data) % 4:
                        raise RuntimeError("FFmpeg returned malformed float32 PCM")
                    samples = array("f")
                    samples.frombytes(data)
                    yield samples
                if process.wait():
                    stderr.seek(0)
                    detail = stderr.read(4096).decode(errors="replace")
                    raise RuntimeError(f"FFmpeg could not decode audio: {detail}")

            yield chunks()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
