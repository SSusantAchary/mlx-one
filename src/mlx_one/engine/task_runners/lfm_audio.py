"""LFM speech tasks use the existing bounded inference actor and model pool."""

from __future__ import annotations

from dataclasses import dataclass

from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.task_runners.speech import AudioTaskAdapter
from mlx_one.engine.tasks import PreparedTask, TaskFailure


@dataclass(frozen=True)
class AudioLanguageAvailability:
    architecture_implemented: bool = True
    task_loader: bool = True
    audio_to_text: bool = True
    text_to_audio: bool = True
    speech_to_speech: bool = True
    codec_streaming: bool = True
    reason: str = "Native implementation; exact checkpoint and speech quality validation pending"


class LFM2AudioAdapter(AudioTaskAdapter):
    family = "lfm2_audio"
    supported_tasks = (
        TaskKind.TRANSCRIPTION,
        TaskKind.SPEECH_GENERATION,
        TaskKind.SPEECH_TO_SPEECH,
    )

    @staticmethod
    def availability():
        return AudioLanguageAvailability()

    def __init__(self, bundle):
        if getattr(bundle, "architecture", None) != self.family:
            raise ValueError("LFM audio adapter/checkpoint mismatch")
        self.bundle = bundle

    def prepare(self, task, payload):
        task = TaskKind(task)
        if task not in self.supported_tasks:
            raise ValueError("unsupported LFM audio task")
        if task != TaskKind.SPEECH_GENERATION:
            return super().prepare(task, payload)
        text = payload.get("text")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 16000:
            raise ValueError("speech generation requires 1 to 16000 text characters")
        return PreparedTask(
            task, dict(payload), (max(1, len(text) // 4),), (self.family, task.value), 512 * 1024**2
        )

    def run_steps(self, prepared, controls):
        from mlx_one.audio.lfm import generate_speech
        from mlx_one.audio.service import transcribe

        outputs = []
        for item, control in zip(prepared, controls, strict=True):
            try:
                function = transcribe if item.task == TaskKind.TRANSCRIPTION else generate_speech
                output = function(self.bundle, cancelled=control.cancelled, **item.payload)
            except Exception as exc:
                output = TaskFailure(exc)
            outputs.append(output)
            yield None
        return tuple(outputs)

    def stream(self, prepared, cancel):
        from mlx_one.audio.lfm import stream_lfm

        task = {
            TaskKind.TRANSCRIPTION: "transcribe",
            TaskKind.SPEECH_GENERATION: "synthesize",
            TaskKind.SPEECH_TO_SPEECH: "speech-to-speech",
        }[prepared.task]
        yield from stream_lfm(self.bundle, task=task, cancelled=cancel.is_set, **prepared.payload)
