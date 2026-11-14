"""Typed classification route using the existing auth, lifecycle and task actor."""

from __future__ import annotations

import threading
from concurrent.futures import CancelledError
from dataclasses import dataclass

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator


@dataclass(frozen=True)
class ClassificationServingBundle:
    classification: object
    chat_template: object = None
    parameter_count: object = None

    def __getattr__(self, name):
        return getattr(self.classification, name)


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: StrictStr = Field(min_length=1, max_length=1024)
    input: StrictStr | list[StrictStr]
    max_length: int = Field(default=512, ge=2, le=512, strict=True)
    top_k: int = Field(default=2, ge=1, le=2, strict=True)

    @model_validator(mode="after")
    def validate_texts(self):
        from mlx_one.classification.schemas import validate_inputs

        validate_inputs(self.input, self.max_length, self.top_k)
        return self


def install_classification_routes(app, manager, services, auth_dependency):
    from mlx_one.engine.cache_errors import CacheCapacityError
    from mlx_one.engine.execution_context import QueueFullError, RequestTimeoutError
    from mlx_one.server.app import _error, _validate_task_model
    from mlx_one.server.ocr import _run_request

    @app.post("/v1/tasks/text-classification")
    async def classify(body: ClassificationRequest, request: Request, auth=auth_dependency):
        if auth is not None:
            return auth
        invalid = _validate_task_model(manager, body.model)
        if invalid is not None:
            return invalid
        service = services.get("text-classification")
        if service is None:
            return _error(400, "loaded model has no text-classification task", "capability_error")
        event = threading.Event()

        def run():
            # Keep the serving-model lease until bounded native execution stops,
            # even if the HTTP client has already disconnected.
            with manager.model_lease():
                return service.classify(
                    body.input, max_length=body.max_length, top_k=body.top_k, cancel=event
                )

        try:
            result = await _run_request(request, run, event)
            return result.to_dict()
        except QueueFullError:
            return _error(429, "classification queue is full", "rate_limit_error")
        except RequestTimeoutError as exc:
            return _error(408, str(exc), "timeout_error")
        except CancelledError:
            return _error(499, "classification cancelled", "cancelled_error")
        except CacheCapacityError as exc:
            return _error(429, str(exc), "capacity_error")
        except (ValueError, UnicodeError) as exc:
            return _error(422, str(exc), "invalid_request_error")
        except RuntimeError as exc:
            return _error(503, str(exc), "capacity_error")


@dataclass(frozen=True)
class TokenClassificationServingBundle:
    ner: object
    chat_template: object = None
    parameter_count: object = None

    def __getattr__(self, name):
        return getattr(self.ner, name)


class TokenClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: StrictStr = Field(min_length=1, max_length=1024)
    input: StrictStr | list[StrictStr]
    max_length: int = Field(default=512, ge=2, le=512, strict=True)
    aggregation_strategy: str = Field(default="none", pattern="^(none|simple)$")

    @model_validator(mode="after")
    def validate_texts(self):
        from mlx_one.classification.schemas import validate_inputs

        validate_inputs(self.input, self.max_length)
        return self


def install_token_classification_routes(app, manager, services, auth_dependency):
    from mlx_one.engine.cache_errors import CacheCapacityError
    from mlx_one.engine.execution_context import QueueFullError, RequestTimeoutError
    from mlx_one.server.app import _error, _validate_task_model
    from mlx_one.server.ocr import _run_request

    @app.post("/v1/tasks/token-classification")
    async def tag(body: TokenClassificationRequest, request: Request, auth=auth_dependency):
        if auth is not None:
            return auth
        invalid = _validate_task_model(manager, body.model)
        if invalid is not None:
            return invalid
        service = services.get("token-classification")
        if service is None:
            return _error(400, "loaded model has no token-classification task", "capability_error")
        event = threading.Event()

        def run():
            with manager.model_lease():
                return service.tag(
                    body.input,
                    max_length=body.max_length,
                    aggregation_strategy=body.aggregation_strategy,
                    cancel=event,
                )

        try:
            result = await _run_request(request, run, event)
            return result.to_dict()
        except QueueFullError:
            return _error(429, "NER queue is full", "rate_limit_error")
        except RequestTimeoutError as exc:
            return _error(408, str(exc), "timeout_error")
        except CancelledError:
            return _error(499, "NER cancelled", "cancelled_error")
        except CacheCapacityError as exc:
            return _error(429, str(exc), "capacity_error")
        except ValueError as exc:
            return _error(422, str(exc), "invalid_request_error")
        except RuntimeError as exc:
            return _error(503, str(exc), "capacity_error")


@dataclass(frozen=True)
class ZeroShotServingBundle:
    zero_shot: object
    chat_template: object = None
    parameter_count: object = None

    def __getattr__(self, name):
        return getattr(self.zero_shot, name)


def install_zero_shot_routes(app, manager, services, auth_dependency):
    import json

    from mlx_one.engine.cache_errors import CacheCapacityError
    from mlx_one.engine.execution_context import QueueFullError, RequestTimeoutError
    from mlx_one.server.app import _error, _validate_task_model
    from mlx_one.server.ocr import _run_request

    @app.post("/v1/tasks/zero-shot-image-classification")
    async def zero_shot(request: Request, auth=auth_dependency):
        if auth is not None:
            return auth
        service = services.get("zero-shot-image-classification")
        if service is None:
            return _error(
                400, "loaded model has no admitted zero-shot image task", "capability_error"
            )
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            return _error(415, "multipart image upload required", "invalid_request_error")
        event = threading.Event()
        try:
            from PIL.Image import DecompressionBombError

            async with request.form(max_files=1, max_fields=4, max_part_size=20 * 1024**2) as form:
                if set(form) - {"model", "file", "candidate_labels", "top_k"}:
                    raise ValueError("unknown zero-shot image fields")
                invalid = _validate_task_model(manager, str(form.get("model", "")))
                if invalid is not None:
                    return invalid
                file = form.get("file")
                if not callable(getattr(file, "read", None)):
                    raise ValueError("image file upload required; paths and URLs are not accepted")
                if file.content_type not in {"image/png", "image/jpeg", "image/webp"}:
                    return _error(415, "upload must be PNG, JPEG or WebP", "invalid_request_error")
                content = await file.read(20 * 1024**2 + 1)
                if len(content) > 20 * 1024**2:
                    return _error(413, "image upload exceeds 20 MiB", "invalid_request_error")
                raw_labels = str(form.get("candidate_labels", ""))
                if len(raw_labels.encode("utf-8")) > 16384:
                    raise ValueError("candidate label JSON exceeds its byte budget")
                labels = json.loads(raw_labels)
                top_k = int(str(form["top_k"])) if "top_k" in form else None

            def run():
                with manager.model_lease():
                    return service.classify(content, labels, top_k=top_k, cancel=event)

            return (await _run_request(request, run, event)).to_dict()
        except QueueFullError:
            return _error(429, "zero-shot image queue is full", "rate_limit_error")
        except RequestTimeoutError as exc:
            return _error(408, str(exc), "timeout_error")
        except CancelledError:
            return _error(499, "zero-shot image request cancelled", "cancelled_error")
        except CacheCapacityError as exc:
            return _error(429, str(exc), "capacity_error")
        except ImportError:
            return _error(503, "zero-shot image task requires the vision extra", "capability_error")
        except (ValueError, OSError, DecompressionBombError) as exc:
            return _error(422, str(exc), "invalid_request_error")
        except RuntimeError as exc:
            return _error(503, str(exc), "capacity_error")
