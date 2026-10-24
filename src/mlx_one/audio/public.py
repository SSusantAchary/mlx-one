"""Managed Python audio API using the shared task actor and request-owned sessions."""

from __future__ import annotations

import threading

from mlx_one.audio.input import AudioInput, prepare_audio
from mlx_one.engine.batching import BatchingConfig
from mlx_one.engine.service import InferenceService


class AudioEngine:
    """One resident checkpoint; serialize native inference and release it on close."""

    def __init__(self, model, **load_options):
        self.service = InferenceService(
            batching_config=BatchingConfig(
                max_sequences=1, max_scheduled_tokens=8192, task_transient_budget_bytes=12 * 1024**3
            )
        )
        self.service.register_task("audio", model, **load_options)

    def _input(self, audio, sample_rate, cancelled=None):
        return prepare_audio(audio, sample_rate=sample_rate, max_seconds=180, cancelled=cancelled)

    def transcribe(self, audio, *, sample_rate=None, cancel=None, **options):
        event = cancel or threading.Event()
        pcm = self._input(audio, sample_rate, event.is_set)
        result = self.service.run_task(
            "audio",
            "audio.transcribe",
            {"audio": pcm, "sample_rate": 16000, **options},
            cancel=event,
        ).output
        return apply_offset(result, audio.offset_seconds if isinstance(audio, AudioInput) else 0)

    def detect_language(self, audio, *, sample_rate=None, cancel=None):
        event = cancel or threading.Event()
        pcm = self._input(audio, sample_rate, event.is_set)
        return self.service.run_task(
            "audio", "audio.detect_language", {"audio": pcm, "sample_rate": 16000}, cancel=event
        ).output

    def align(self, audio, text, *, language, sample_rate=None, cancel=None):
        event = cancel or threading.Event()
        pcm = self._input(audio, sample_rate, event.is_set)
        result = self.service.run_task(
            "audio",
            "audio.align",
            {"audio": pcm, "text": text, "language": language, "sample_rate": 16000},
            cancel=event,
        ).output
        return apply_offset(result, audio.offset_seconds if isinstance(audio, AudioInput) else 0)

    def generate_speech(self, text, *, cancel=None, **options):
        return self.service.run_task(
            "audio", "audio.generate_speech", {"text": text, **options}, cancel=cancel
        ).output

    def speech_to_speech(self, audio, *, sample_rate=None, cancel=None, **options):
        event = cancel or threading.Event()
        pcm = self._input(audio, sample_rate, event.is_set)
        return self.service.run_task(
            "audio",
            "audio.speech_to_speech",
            {"audio": pcm, "sample_rate": 16000, **options},
            cancel=event,
        ).output

    def stream_generate(self, text="", *, audio=None, sample_rate=None, cancel=None, **options):
        event = cancel or threading.Event()
        payload = {"text": text, **options}
        task = "audio.generate_speech"
        if audio is not None:
            payload.update(audio=self._input(audio, sample_rate, event.is_set), sample_rate=16000)
            task = "audio.speech_to_speech"
        yield from self.service.stream_task("audio", task, payload, cancel=event)

    def stream_transcribe(
        self, audio_source, *, sample_rate=16000, language=None, options=None, cancel=None
    ):
        """Qwen bounded accumulated utterance hypotheses; not causal encoder streaming."""
        event = cancel or threading.Event()
        with self.service.open_audio_session(
            "audio", sample_rate=sample_rate, language=language, options=options, cancel=event
        ) as session:
            for chunk in audio_source:
                yield from session.push(chunk)
            yield session.finish()

    def close(self):
        self.service.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["AudioEngine", "AudioInput"]


def apply_offset(result, offset):
    """Translate clip-local spans into the supplied timeline without altering duration."""
    from dataclasses import replace

    from mlx_one.audio.schemas import AudioAlignment, SpeechTranscript, TranscriptionResult

    if not offset:
        return result
    if isinstance(result, AudioAlignment):
        return replace(
            result,
            offset_seconds=offset,
            units=tuple(
                replace(unit, start=unit.start + offset, end=unit.end + offset)
                for unit in result.units
            ),
        )
    if isinstance(result, TranscriptionResult):
        return replace(
            result,
            offset_seconds=offset,
            segments=tuple(
                replace(
                    segment,
                    start=segment.start + offset,
                    end=segment.end + offset,
                    words=tuple(
                        replace(word, start=word.start + offset, end=word.end + offset)
                        for word in segment.words
                    ),
                )
                for segment in result.segments
            ),
        )
    if isinstance(result, SpeechTranscript):
        return replace(
            result,
            offset_seconds=offset,
            timestamps=tuple(
                {**span, "start": span["start"] + offset, "end": span["end"] + offset}
                for span in result.timestamps
            )
            if result.timestamps is not None
            else None,
        )
    raise ValueError("audio output does not support timeline offsets")
