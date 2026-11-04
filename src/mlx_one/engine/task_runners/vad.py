"""Silero chunk/state execution on the existing bounded native task actor."""

from __future__ import annotations

from concurrent.futures import CancelledError

import numpy as np

from mlx_one.audio.vad import MAX_SAMPLES, vad_result, validate_options
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.task_runners.base import NativeTaskAdapter
from mlx_one.engine.tasks import PreparedTask, TaskFailure


class SileroVADAdapter(NativeTaskAdapter):
    family = "silero_vad"
    supported_tasks = (TaskKind.VOICE_ACTIVITY_DETECTION,)

    def __init__(self, bundle):
        if bundle.architecture != self.family:
            raise ValueError("VAD adapter/checkpoint mismatch")
        self.bundle = bundle

    def prepare(self, task, payload):
        if TaskKind(task) != self.supported_tasks[0]:
            raise ValueError("Silero supports voice-activity-detection only")
        if set(payload) - {"pcm", "options", "state", "context"}:
            raise ValueError("unknown VAD payload fields")
        pcm = np.array(payload["pcm"], dtype=np.float32, copy=True)
        if pcm.ndim != 1 or len(pcm) > MAX_SAMPLES or not np.isfinite(pcm).all():
            raise ValueError("VAD requires bounded finite mono PCM")
        state = payload.get("state")
        context = payload.get("context")
        if (state is None) != (context is None):
            raise ValueError("VAD state and context must be supplied together")
        if state is not None:
            state, context = np.array(state, dtype=np.float32), np.array(context, dtype=np.float32)
            if (
                state.shape != (2, 1, 128)
                or context.shape != (1, 64)
                or not np.isfinite(state).all()
                or not np.isfinite(context).all()
                or pcm.shape != (512,)
            ):
                raise ValueError("invalid VAD streaming state/chunk")
        data = {
            "pcm": pcm,
            "options": validate_options(payload.get("options", {})),
            "state": state,
            "context": context,
        }
        chunks = max(1, (len(pcm) + 511) // 512)
        return PreparedTask(
            self.supported_tasks[0], data, (chunks,), (id(pcm),), pcm.nbytes + 2 * 1024**2
        )

    def run_steps(self, prepared, controls):
        import mlx.core as mx

        results = []
        for request, control in zip(prepared, controls, strict=True):
            d = request.payload
            state = mx.zeros((2, 1, 128)) if d["state"] is None else mx.array(d["state"])
            context = mx.zeros((1, 64)) if d["context"] is None else mx.array(d["context"])
            probs = []
            padded = np.pad(d["pcm"], (0, -len(d["pcm"]) % 512))
            for offset in range(0, len(padded), 512):
                if control.cancelled():
                    results.append(TaskFailure(CancelledError("VAD cancelled")))
                    break
                chunk = mx.array(padded[offset : offset + 512][None])
                prob, state = self.bundle.model(mx.concatenate((context, chunk), axis=1), state)
                context = chunk[:, -64:]
                mx.eval(prob, state, context)
                probs.append(float(prob.item()))
                yield
            else:
                if d["state"] is not None:
                    results.append(
                        {
                            "probabilities": probs,
                            "state": np.asarray(state).copy(),
                            "context": np.asarray(context).copy(),
                        }
                    )
                else:
                    results.append(vad_result(probs, len(d["pcm"]), d["options"]))
        return tuple(results)
