"""Bounded non-chat tasks on the existing native task actor and HTTP lifecycle."""

from __future__ import annotations

import base64
import io
import threading
from concurrent.futures import CancelledError
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, StrictStr


@dataclass(frozen=True)
class NativeTaskServingBundle:
    """Keep a task bundle under the server's existing model lease."""

    native_task: Any
    task_name: str
    context_length: int
    chat_template: Any = None
    parameter_count: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native_task, name)


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: StrictStr = Field(min_length=1, max_length=1024)


class TextToTextRequest(TaskRequest):
    input: StrictStr | list[StrictStr]
    max_length: int = Field(default=512, ge=2, le=512, strict=True)
    max_new_tokens: int = Field(default=32, ge=1, le=511, strict=True)


class ImageClassificationRequest(TaskRequest):
    images: list[StrictStr] = Field(min_length=1, max_length=8)
    top_k: int = Field(default=5, ge=1, le=1000, strict=True)


class SpeechRequest(TaskRequest):
    input: StrictStr = Field(min_length=1, max_length=8192)
    voice: StrictStr = "af_heart"
    language: StrictStr = "en-us"
    phonemes: bool = Field(default=False, strict=True)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


def _decode_image(value: str) -> Any:
    from PIL import Image

    prefix, separator, payload = value.partition(",")
    if not separator or prefix not in {
        "data:image/png;base64",
        "data:image/jpeg;base64",
        "data:image/webp;base64",
    }:
        raise ValueError("images must be PNG/JPEG/WebP base64 data URIs")
    try:
        raw = base64.b64decode(payload, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 32_000_000:
                raise ValueError("image exceeds the 32-megapixel input bound")
            return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError("invalid or oversized image data URI") from exc


def install_native_task_routes(app, manager, services, auth_dependency) -> None:
    """Install explicit T5, supervised ViT, and Kokoro routes; no chat fallback."""
    from mlx_one.engine.cache_errors import CacheCapacityError
    from mlx_one.engine.execution_context import QueueFullError, RequestTimeoutError
    from mlx_one.server.app import _error, _validate_task_model
    from mlx_one.server.ocr import _run_request

    async def execute(body, request, task, payload, auth):
        if auth is not None:
            return auth
        invalid = _validate_task_model(manager, body.model)
        if invalid is not None:
            return invalid
        service = services.get(task)
        if service is None:
            return _error(400, f"loaded model has no {task} task", "capability_error")
        event = threading.Event()

        def run():
            with manager.model_lease():
                inputs = payload()  # Decode bounded images only on an authorized task request.
                return service.run_task("primary", task, inputs, cancel=event).output

        try:
            result = await _run_request(request, run, event)
            if task == "speech-generation":
                import wave

                import numpy as np

                output = io.BytesIO()
                with wave.open(output, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(result.sample_rate)
                    wav.writeframes(
                        (np.clip(result.waveform, -1, 1) * 32767).astype("<i2").tobytes()
                    )
                return Response(output.getvalue(), media_type="audio/wav")
            return result.to_dict()
        except QueueFullError:
            return _error(429, "native task queue is full", "rate_limit_error")
        except RequestTimeoutError as exc:
            return _error(408, str(exc), "timeout_error")
        except CancelledError:
            return _error(499, "native task cancelled", "cancelled_error")
        except CacheCapacityError as exc:
            return _error(429, str(exc), "capacity_error")
        except (ValueError, UnicodeError) as exc:
            return _error(422, str(exc), "invalid_request_error")
        except RuntimeError as exc:
            return _error(503, str(exc), "capacity_error")

    @app.post("/v1/tasks/text-to-text-generation")
    async def text_to_text(body: TextToTextRequest, request: Request, auth=auth_dependency):
        return await execute(
            body,
            request,
            "text-to-text-generation",
            lambda: dict(
                texts=body.input, max_length=body.max_length, max_new_tokens=body.max_new_tokens
            ),
            auth,
        )

    @app.post("/v1/tasks/image-classification")
    async def image_classification(
        body: ImageClassificationRequest, request: Request, auth=auth_dependency
    ):
        return await execute(
            body,
            request,
            "image-classification",
            lambda: dict(images=[_decode_image(image) for image in body.images], top_k=body.top_k),
            auth,
        )

    @app.post("/v1/tasks/speech-generation")
    async def speech_generation(body: SpeechRequest, request: Request, auth=auth_dependency):
        return await execute(
            body,
            request,
            "speech-generation",
            lambda: dict(
                text=body.input,
                voice=body.voice,
                language=body.language,
                phonemes=body.phonemes,
                speed=body.speed,
            ),
            auth,
        )
