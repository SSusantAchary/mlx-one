"""Bounded mono PCM16 WAV VAD route on existing auth/leases/task lifecycle."""

from __future__ import annotations

import threading
import wave
from concurrent.futures import CancelledError
from dataclasses import dataclass

from fastapi import Request


@dataclass(frozen=True)
class VADServingBundle:
    vad: object
    chat_template: object = None
    parameter_count: object = None

    def __getattr__(self, name):
        return getattr(self.vad, name)


def install_vad_routes(app, manager, services, auth_dependency):
    from mlx_one.engine.cache_errors import CacheCapacityError
    from mlx_one.engine.execution_context import QueueFullError, RequestTimeoutError
    from mlx_one.server.app import _error, _validate_task_model
    from mlx_one.server.ocr import _run_request

    @app.post("/v1/tasks/voice-activity-detection")
    async def detect(request: Request, auth=auth_dependency):
        if auth is not None:
            return auth
        if request.headers.get("content-type", "").partition(";")[0] != "multipart/form-data":
            return _error(415, "VAD requires a multipart PCM16 WAV upload", "invalid_request_error")
        event = threading.Event()
        try:
            async with request.form(max_files=1, max_fields=5, max_part_size=1024) as form:
                if set(form) - {
                    "file",
                    "model",
                    "threshold",
                    "min_speech_duration_ms",
                    "min_silence_duration_ms",
                    "speech_pad_ms",
                }:
                    raise ValueError("unknown VAD fields")
                if any(len(form.getlist(k)) != 1 for k in form):
                    raise ValueError("duplicate VAD fields")
                invalid = _validate_task_model(manager, form.get("model"))
                if invalid is not None:
                    return invalid
                service = services.get("voice-activity-detection")
                if service is None:
                    return _error(400, "loaded model has no VAD task", "capability_error")
                upload = form.get("file")
                if not hasattr(upload, "read"):
                    raise ValueError("one WAV file is required")
                if upload.content_type not in {"audio/wav", "audio/x-wav", "audio/wave"}:
                    return _error(415, "VAD accepts PCM16 WAV only", "invalid_request_error")
                data = await upload.read(25 * 1024**2 + 1)
                if len(data) > 25 * 1024**2:
                    return _error(413, "VAD upload exceeds 25 MiB", "invalid_request_error")
                options = {k: float(form[k]) for k in form if k not in {"file", "model"}}

            def run():
                with manager.model_lease():
                    return service.detect(data, cancel=event, **options)

            return (await _run_request(request, run, event)).to_dict()
        except QueueFullError:
            return _error(429, "VAD queue is full", "rate_limit_error")
        except RequestTimeoutError as exc:
            return _error(408, str(exc), "timeout_error")
        except CancelledError:
            return _error(499, "VAD cancelled", "cancelled_error")
        except CacheCapacityError as exc:
            return _error(429, str(exc), "capacity_error")
        except (ValueError, TypeError, wave.Error, EOFError) as exc:
            return _error(422, str(exc), "invalid_request_error")
        except RuntimeError as exc:
            return _error(503, str(exc), "capacity_error")
