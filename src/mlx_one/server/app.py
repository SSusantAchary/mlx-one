"""FastAPI application and OpenAI-compatible native inference routes."""

from __future__ import annotations

import hmac
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from mlx_one.diagnostics import collect_doctor_report
from mlx_one.server.config import ServerConfig
from mlx_one.server.generation_engine import GenerationEngine, GenerationEvent
from mlx_one.server.model_manager import ModelManager
from mlx_one.server.scheduler import GenerationScheduler, QueueFullError, RequestTimeoutError
from mlx_one.server.schemas import ChatCompletionRequest
from mlx_one.text import TextGenerationOptions


class BodyLimitMiddleware:
    def __init__(self, app: Any, limit: int = 1024 * 1024) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = 0
        if declared > self.limit:
            response = _error(413, "request body exceeds 1 MiB", "invalid_request_error")
            await response(scope, receive, send)
            return
        consumed = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > self.limit:
                raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            response = _error(413, "request body exceeds 1 MiB", "invalid_request_error")
            await response(scope, receive, send)


class _BodyTooLarge(Exception):
    pass


def create_app(
    model_manager: ModelManager | None = None,
    generation_engine: GenerationEngine | None = None,
    scheduler: GenerationScheduler | None = None,
    ui_dir: str | Path | None = None,
    config: ServerConfig | None = None,
) -> FastAPI:
    settings = config or ServerConfig(warmup=False, cache_prompt=False, timeout=0)
    manager = model_manager or ModelManager(
        alias=settings.alias, context_length=settings.context_length
    )
    engine = generation_engine or GenerationEngine(manager)
    jobs = scheduler or GenerationScheduler(
        max_pending=settings.queue_size,
        parallel=settings.parallel,
        timeout=settings.timeout,
    )
    bearer = HTTPBearer(auto_error=False)
    bearer_dependency = Depends(bearer)

    async def authorize(
        credentials: HTTPAuthorizationCredentials | None = bearer_dependency,
    ) -> None | JSONResponse:
        if not settings.api_keys:
            return None
        supplied = (
            credentials.credentials
            if credentials and credentials.scheme.lower() == "bearer"
            else ""
        )
        valid = False
        for key in settings.api_keys:
            valid |= hmac.compare_digest(supplied, key)
        if not valid:
            return _error(
                401,
                "invalid or missing API key",
                "authentication_error",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None

    authorization_dependency = Depends(authorize)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        def release() -> None:
            clear_caches = getattr(engine, "clear_caches", None)
            if callable(clear_caches):
                clear_caches()
            manager.unload()

        jobs.close(finalizer=release)

    app = FastAPI(title="mlx-one", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, limit=1024 * 1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(422, str(exc), "invalid_request_error")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "runtime": "mlx", "platform": "apple-silicon"}

    @app.get("/v1/models")
    async def models(auth: None | JSONResponse = authorization_dependency) -> Any:
        if auth is not None:
            return auth
        return {
            "object": "list",
            "data": [
                {
                    "id": item.id,
                    "object": "model",
                    "owned_by": "mlx-one",
                    "mlx": {
                        "architecture": item.architecture,
                        "parameters": item.parameter_count,
                        "quantization": item.quantization or None,
                        "context_length": item.context_length,
                        "revision": item.revision,
                    },
                }
                for item in manager.list_models()
            ],
        }

    @app.get("/v1/runtime")
    async def runtime(auth: None | JSONResponse = authorization_dependency) -> Any:
        if auth is not None:
            return auth
        result: dict[str, Any] = {"backend": "mlx", "device": "gpu"}
        try:
            report = collect_doctor_report()
            result.update({"chip": report.chip, "platform": report.operating_system.lower()})
        except Exception:
            result.update({"chip": None, "platform": "apple-silicon"})
        try:
            result["model"] = manager.model_info().to_dict()
        except RuntimeError:
            result["model"] = None
        result.update(
            {key: value for key, value in engine.stats.to_dict().items() if value is not None}
        )
        result["server"] = settings.public_dict()
        result["scheduler"] = jobs.stats()
        return result

    @app.post("/v1/chat/completions")
    async def chat(
        request: Request,
        body: ChatCompletionRequest,
        auth: None | JSONResponse = authorization_dependency,
    ) -> Any:
        if auth is not None:
            return auth
        try:
            bundle = manager.current_model()
            exposed_model = manager.model_info().id
            if body.model != exposed_model:
                return _error(404, f"model {body.model!r} is not loaded", "invalid_request_error")
            effective_reasoning = body.reasoning or settings.reasoning
            if effective_reasoning == "on" and (
                bundle.chat_template is None
                or "enable_thinking" not in (bundle.chat_template.template or "")
            ):
                return _error(
                    400,
                    "loaded chat template does not support explicit reasoning control",
                    "invalid_request_error",
                )
            stop = (body.stop,) if isinstance(body.stop, str) else tuple(body.stop or ())
            options = TextGenerationOptions(
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                seed=body.seed,
                stop=stop,
            )
            messages = [item.model_dump(exclude_none=True) for item in body.messages]
            stream = jobs.schedule(
                lambda cancel: engine.stream(
                    model=body.model,
                    messages=messages,
                    options=options,
                    cancel=cancel,
                    context_length=manager.model_info().context_length,
                    context_shift=settings.context_shift,
                    reasoning=effective_reasoning,
                    reasoning_budget=(
                        body.reasoning_budget
                        if body.reasoning_budget is not None
                        else settings.reasoning_budget
                    ),
                    reasoning_format=settings.reasoning_format,
                    reasoning_preserve=settings.reasoning_preserve,
                    cache_prompt=settings.cache_prompt,
                    cache_reuse=settings.cache_reuse,
                    cache_entries=settings.parallel,
                    cache_idle_slots=settings.cache_idle_slots,
                    cache_type_k=settings.cache_type_k,
                    cache_type_v=settings.cache_type_v,
                    spec_type=settings.spec_type,
                    spec_draft_n_max=settings.spec_draft_n_max,
                )
            )
        except QueueFullError:
            return _error(429, "generation queue is full", "rate_limit_error")
        except (RuntimeError, ValueError) as exc:
            return _error(400, str(exc), "invalid_request_error")
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if body.stream:
            try:
                first_event = await anext(stream)
            except RequestTimeoutError as exc:
                await stream.aclose()
                return _error(408, str(exc), "timeout_error")
            except Exception as exc:
                await stream.aclose()
                return StreamingResponse(
                    _streaming_failure(exc), media_type="text/event-stream"
                )
            return StreamingResponse(
                _sse(
                    request,
                    stream,
                    completion_id,
                    created,
                    body.model,
                    first_event=first_event,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        content = ""
        reasoning_content = ""
        terminal: GenerationEvent | None = None
        try:
            async for event in stream:
                content += event.text
                reasoning_content += event.reasoning_content
                if event.finish_reason is not None:
                    terminal = event
        except RequestTimeoutError as exc:
            return _error(408, str(exc), "timeout_error")
        except Exception as exc:
            return _error(500, str(exc), "server_error")
        metrics = terminal.metrics if terminal and terminal.metrics else {}
        prompt_tokens = int(metrics.get("prompt_tokens", 0))
        generated_tokens = int(metrics.get("generated_tokens", 0))
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    **({"reasoning_content": reasoning_content} if reasoning_content else {}),
                },
                "finish_reason": terminal.finish_reason if terminal else "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated_tokens,
                "total_tokens": prompt_tokens + generated_tokens,
            },
            "mlx": metrics,
        }

    assets = Path(ui_dir) if ui_dir is not None else Path(str(files("mlx_one").joinpath("ui/dist")))

    @app.get("/{asset_path:path}", include_in_schema=False)
    async def static(asset_path: str) -> Any:
        root = assets.resolve()
        candidate = (root / asset_path).resolve()
        if candidate != root and root not in candidate.parents:
            return _error(404, "asset not found", "not_found_error")
        if candidate.is_file():
            return FileResponse(candidate)
        index = root / "index.html"
        if index.is_file():
            return FileResponse(index)
        return _error(503, "mlx-one UI assets are not installed", "server_error")

    return app


async def _sse(
    request: Request,
    stream: AsyncIterator[GenerationEvent],
    completion_id: str,
    created: int,
    model: str,
    *,
    first_event: GenerationEvent | None = None,
) -> AsyncIterator[str]:
    first = True
    disconnected = False
    try:
        async def events() -> AsyncIterator[GenerationEvent]:
            if first_event is not None:
                yield first_event
            async for item in stream:
                yield item

        async for event in events():
            if await request.is_disconnected():
                disconnected = True
                break
            if not event.text and not event.reasoning_content and event.finish_reason is None:
                continue
            delta: dict[str, str] = {}
            if first:
                delta["role"] = "assistant"
                first = False
            if event.text:
                delta["content"] = event.text
            if event.reasoning_content:
                delta["reasoning_content"] = event.reasoning_content
            payload: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": event.finish_reason}
                ],
            }
            if event.metrics is not None:
                payload["mlx"] = event.metrics
            yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    except Exception as exc:
        timeout = isinstance(exc, RequestTimeoutError)
        payload = {
            "error": {
                "message": str(exc),
                "type": "timeout_error" if timeout else "server_error",
                "code": 408 if timeout else 500,
            }
        }
        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    finally:
        await stream.aclose()
    if not disconnected:
        yield "data: [DONE]\n\n"


def _error(
    status: int,
    message: str,
    error_type: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        headers=headers,
        content={"error": {"message": message, "type": error_type, "code": status}},
    )


async def _streaming_failure(exc: Exception) -> AsyncIterator[str]:
    payload = {
        "error": {"message": str(exc), "type": "server_error", "code": 500}
    }
    yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"
