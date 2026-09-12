"""FastAPI application and OpenAI-compatible native inference routes."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from mlx_one.diagnostics import collect_doctor_report
from mlx_one.server.generation_engine import GenerationEngine, GenerationEvent
from mlx_one.server.model_manager import ModelManager
from mlx_one.server.scheduler import GenerationScheduler, QueueFullError
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
) -> FastAPI:
    manager = model_manager or ModelManager()
    engine = generation_engine or GenerationEngine(manager)
    jobs = scheduler or GenerationScheduler(max_pending=8)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        jobs.close()
        manager.unload()

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
    async def models() -> dict[str, Any]:
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
    async def runtime() -> dict[str, Any]:
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
        return result

    @app.post("/v1/chat/completions")
    async def chat(request: Request, body: ChatCompletionRequest) -> Any:
        try:
            loaded = manager.current_model()
            if body.model != loaded.model_id:
                return _error(404, f"model {body.model!r} is not loaded", "invalid_request_error")
            stop = (body.stop,) if isinstance(body.stop, str) else tuple(body.stop or ())
            options = TextGenerationOptions(
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                seed=body.seed,
                stop=stop,
            )
            messages = [item.model_dump() for item in body.messages]
            stream = jobs.schedule(
                lambda cancel: engine.stream(
                    model=body.model, messages=messages, options=options, cancel=cancel
                )
            )
        except QueueFullError:
            return _error(429, "generation queue is full", "rate_limit_error")
        except (RuntimeError, ValueError) as exc:
            return _error(400, str(exc), "invalid_request_error")
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if body.stream:
            return StreamingResponse(
                _sse(request, stream, completion_id, created, body.model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        content = ""
        terminal: GenerationEvent | None = None
        try:
            async for event in stream:
                content += event.text
                if event.finish_reason is not None:
                    terminal = event
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
                "message": {"role": "assistant", "content": content},
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
) -> AsyncIterator[str]:
    first = True
    disconnected = False
    try:
        async for event in stream:
            if await request.is_disconnected():
                disconnected = True
                break
            delta: dict[str, str] = {}
            if first:
                delta["role"] = "assistant"
                first = False
            if event.text:
                delta["content"] = event.text
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
        payload = {"error": {"message": str(exc), "type": "server_error", "code": 500}}
        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    finally:
        await stream.aclose()
    if not disconnected:
        yield "data: [DONE]\n\n"


def _error(status: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type, "code": status}},
    )
