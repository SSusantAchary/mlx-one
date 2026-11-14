"""Bounded 16-kHz Silero facade and independent request-owned streaming state."""

from __future__ import annotations

import io
import json
import math
import threading
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mlx_one.audio.input import MAX_BYTES, prepare_audio
from mlx_one.audio.vad_postprocessing import get_speech_timestamps_from_probs
from mlx_one.engine.batching import BatchingConfig
from mlx_one.engine.service import InferenceService

MAX_SAMPLES = 16000 * 600
REVISION = "60b7ffa243625ebdc1070275a29f18c87843786a"


def prepare_vad(source, sample_rate=16000):
    if type(sample_rate) is not int or sample_rate != 16000:
        raise ValueError("VAD admits only 16000-Hz PCM; resampling is not qualified")
    if isinstance(source, (str, Path, bytes, bytearray)):
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser()
            if path.stat().st_size > MAX_BYTES:
                raise ValueError("VAD WAV exceeds 25 MiB")
            source = path.read_bytes()
        if len(source) > MAX_BYTES:
            raise ValueError("VAD WAV exceeds 25 MiB")
        with wave.open(io.BytesIO(source), "rb") as f:
            if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
                raise ValueError("VAD WAV must be mono PCM16 at 16000 Hz")
            if f.getnframes() > MAX_SAMPLES:
                raise ValueError("VAD audio exceeds 600 s")
            samples = f.readframes(f.getnframes())
            if len(samples) != f.getnframes() * 2:
                raise ValueError("truncated VAD WAV")
            source = np.frombuffer(samples, dtype="<i2")
    if np.ndim(source) != 1:
        raise ValueError("VAD requires mono PCM; channel mixing is not qualified")
    # Existing input processor, with no resampling at the admitted 16-kHz rate.
    return prepare_audio(source, sample_rate=16000, max_seconds=600)


def validate_options(options):
    defaults = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
        "speech_pad_ms": 30,
    }
    if set(options) - set(defaults):
        raise ValueError("unknown VAD options")
    result = {**defaults, **options}
    for k, v in result.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"VAD {k} must be a finite number")
        if k == "threshold":
            if not 0 < v < 1:
                raise ValueError("VAD threshold must be between 0 and 1")
        elif not 0 <= v <= 60000:
            raise ValueError(f"VAD {k} must be between 0 and 60000 ms")
    return result


@dataclass(frozen=True)
class VADResult:
    segments: tuple[dict, ...]
    probabilities: tuple[float, ...]
    sample_count: int
    options: dict
    model: str = "snakers4/silero-vad-16k"
    revision: str = REVISION
    source_commit: str = REVISION
    task: str = "voice-activity-detection"
    schema_version: str = "mlx-one-vad-v1"
    sample_rate: int = 16000
    frame_samples: int = 512
    timestamp_units: str = "samples; end exclusive"

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), allow_nan=False)


def vad_result(probs, sample_count, options):
    segments = get_speech_timestamps_from_probs(
        probs, sampling_rate=16000, audio_length_samples=sample_count, **options
    )
    return VADResult(tuple(segments), tuple(probs), sample_count, dict(options))


def load_vad_model(path, *, revision=None, offline=True, cache_dir=None):
    if not offline or cache_dir is not None:
        raise ValueError("Silero requires the explicit locally pinned source assets")
    from mlx_one.models.audio.silero.model import load_silero

    return load_silero(path, revision=revision or REVISION)


class VADEngine:
    """Facade over the existing shared task actor; no model-owned recurrent state."""

    def __init__(self, model, *, timeout=60.0, queue_size=8, **load_options):
        self.service = InferenceService(
            batching_config=BatchingConfig(
                max_sequences=1, max_scheduled_tokens=18750, timeout=timeout, queue_size=queue_size
            )
        )
        self.service.register_task("vad", model, **load_options)

    def submit(self, audio, *, sample_rate=16000, cancel=None, deadline=None, **options):
        options = validate_options(options)
        pcm = prepare_vad(audio, sample_rate)
        return self.service.submit_task(
            "vad",
            "voice-activity-detection",
            {"pcm": pcm, "options": options},
            cancel=cancel,
            deadline=deadline,
        )

    def detect(self, audio, *, sample_rate=16000, cancel=None, deadline=None, **options):
        return (
            self.submit(audio, sample_rate=sample_rate, cancel=cancel, deadline=deadline, **options)
            .result()
            .output
        )

    def session(self, **options):
        return VADSession(self, validate_options(options))

    def close(self):
        self.service.close()
        with self.service._lock:
            self.service._task_models.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class VADSession:
    """512-sample chunks; finalize accepts an optional final 1–511-sample tail.

    Sessions share immutable model weights and hold private host state. Calls on
    one session serialize under its lock. Independent sessions cannot exchange
    recurrent state. Finalization returns source-equivalent offline timestamps.
    """

    def __init__(self, engine, options):
        self.engine, self.options = engine, options
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self.state = np.zeros((2, 1, 128), np.float32)
            self.context = np.zeros((1, 64), np.float32)
            self.probabilities, self.sample_count, self.finalized = [], 0, False

    def _process(self, pcm, count, cancel=None, deadline=None):
        if self.finalized:
            raise ValueError("VAD session is finalized; reset before processing")
        if self.sample_count + count > MAX_SAMPLES:
            raise ValueError("VAD session exceeds 600 s")
        output = self.engine.service.run_task(
            "vad",
            "voice-activity-detection",
            {"pcm": pcm, "state": self.state, "context": self.context},
            cancel=cancel,
            deadline=deadline,
        ).output
        self.state, self.context = output["state"], output["context"]
        self.probabilities.extend(output["probabilities"])
        self.sample_count += count
        return output["probabilities"][0]

    def process_chunk(self, pcm, *, sample_rate=16000, cancel=None, deadline=None):
        pcm = prepare_vad(pcm, sample_rate)
        if len(pcm) != 512:
            raise ValueError("VAD streaming chunks require 512 samples; pass tail to finalize")
        with self._lock:
            return self._process(pcm, 512, cancel, deadline)

    def finalize(self, tail=None, *, cancel=None, deadline=None):
        with self._lock:
            if self.finalized:
                raise ValueError("VAD session already finalized")
            if tail is not None:
                pcm = prepare_vad(tail)
                if not 1 <= len(pcm) < 512:
                    raise ValueError("VAD final tail must contain 1–511 samples")
                self._process(np.pad(pcm, (0, 512 - len(pcm))), len(pcm), cancel, deadline)
            self.finalized = True
            return vad_result(self.probabilities, self.sample_count, self.options)
