"""FastAPI application and OpenAI-compatible native inference routes."""

from __future__ import annotations

import asyncio
import base64
import hmac
import inspect
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
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
from mlx_one.engine.memory import AdmissionStatus
from mlx_one.server.admission import MemoryAdmission
from mlx_one.server.config import ServerConfig
from mlx_one.server.generation_engine import GenerationEngine, GenerationEvent
from mlx_one.server.media import ImageResolver, MediaError
from mlx_one.server.model_manager import ModelManager
from mlx_one.server.scheduler import GenerationScheduler, QueueFullError, RequestTimeoutError
from mlx_one.server.schemas import ChatCompletionRequest, EmbeddingRequest, RerankRequest
from mlx_one.server.transcription import MAX_AUDIO_BYTES, AudioUploadError
from mlx_one.text import TextGenerationOptions


class BodyLimitMiddleware:
    def __init__(self, app: Any, limit: int = 1024 * 1024) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", ""))
        limit = (
            32 * 1024**2
            if path == "/v1/chat/completions"
            else 26 * 1024**2
            if path == "/v1/audio/transcriptions"
            else self.limit
        )
        headers = dict(scope.get("headers", ()))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = 0
        if declared > limit:
            response = _error(413, "request body exceeds its route limit", "invalid_request_error")
            await response(scope, receive, send)
            return
        if path == "/v1/chat/completions":
            await self._chat_request(scope, receive, send, limit)
            return
        consumed = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            scope.setdefault("state", {})["body_bytes"] = consumed
            if consumed > limit:
                raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            response = _error(413, "request body exceeds its route limit", "invalid_request_error")
            await response(scope, receive, send)

    async def _chat_request(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
        limit: int,
    ) -> None:
        content = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            content.extend(message.get("body", b""))
            if len(content) > limit:
                response = _error(
                    413,
                    "request body exceeds its route limit",
                    "invalid_request_error",
                )
                await response(scope, receive, send)
                return
            more_body = bool(message.get("more_body", False))
        scope.setdefault("state", {})["body_bytes"] = len(content)
        if len(content) > self.limit and not _is_multimodal_chat_body(content):
            response = _error(413, "text request body exceeds 1 MiB", "invalid_request_error")
            await response(scope, receive, send)
            return
        delivered = False

        async def replay() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(content), "more_body": False}

        await self.app(scope, replay, send)


class _BodyTooLarge(Exception):
    pass


def _is_multimodal_chat_body(content: bytes) -> bool:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return False
    messages = payload.get("messages", ()) if isinstance(payload, dict) else ()
    return any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in message["content"]
        )
        for message in messages
    )


def _cache_bits(cache_type: str) -> int:
    return {"q4_0": 4, "q8_0": 8}.get(cache_type, 16)


def create_app(
    model_manager: ModelManager | None = None,
    generation_engine: GenerationEngine | None = None,
    scheduler: GenerationScheduler | None = None,
    ui_dir: str | Path | None = None,
    config: ServerConfig | None = None,
    task_services: Mapping[str, Any] | None = None,
    image_resolver: ImageResolver | None = None,
    memory_admission: MemoryAdmission | None = None,
) -> FastAPI:
    settings = config or ServerConfig(warmup=False, cache_prompt=False, timeout=0)
    manager = model_manager or ModelManager(
        alias=settings.alias, context_length=settings.context_length
    )
    engine = generation_engine or GenerationEngine(
        manager,
        prefix_cache_bytes=settings.prefix_cache_bytes,
        cache_backend=settings.cache_backend,
        cache_block_size=settings.cache_block_size,
        cache_memory_budget_bytes=settings.cache_memory_budget_bytes,
    )
    jobs = scheduler or GenerationScheduler(
        max_pending=settings.queue_size,
        parallel=settings.parallel,
        timeout=settings.timeout,
    )
    bearer = HTTPBearer(auto_error=False)
    services = dict(task_services or {})
    images = image_resolver or ImageResolver()
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
            close_engine = getattr(engine, "close", None)
            if callable(close_engine):
                close_engine()
            else:
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
                        "tasks": list(item.tasks),
                        "modalities": list(item.modalities),
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
        cache_stats = getattr(engine, "cache_stats", None)
        result["cache"] = cache_stats() if callable(cache_stats) else {}
        if memory_admission is not None:
            result["memory_budget"] = memory_admission.stats()
        transcription = services.get("transcription")
        capability = getattr(transcription, "capabilities", None)
        result["transcription"] = (
            capability()
            if callable(capability)
            else {
                "enabled": transcription is not None,
                "accepted_formats": [],
                "max_bytes": MAX_AUDIO_BYTES,
                "language_detection": transcription is not None,
            }
        )
        if result["transcription"]["enabled"]:
            measured = getattr(manager, "transcription_bytes", 0)
            result["transcription"]["model_memory_bytes"] = measured or None
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
            multimodal = any(not isinstance(item.content, str) for item in body.messages)
            if not multimodal and getattr(request.state, "body_bytes", 0) > 1024 * 1024:
                return _error(413, "text request body exceeds 1 MiB", "invalid_request_error")
            selected_engine = services.get("vision") if multimodal else engine
            if selected_engine is None:
                return _error(
                    400,
                    "the loaded model does not support vision chat requests",
                    "capability_error",
                )
            if not multimodal and effective_reasoning == "on" and (
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
            if multimodal:
                messages = await _resolve_message_images(messages, images)
            admission_check = None
            if memory_admission is not None and not multimodal:
                prompt = bundle.chat_template.render(messages)
                prompt_ids = tuple(bundle.tokenizer.encode(prompt))
                prompt_tokens = len(prompt_ids)

                def admission_check():
                    cache_stats = (
                        engine.cache_stats()
                        if callable(getattr(engine, "cache_stats", None))
                        else {}
                    )
                    reused_tokens = 0
                    preview = getattr(engine, "preview_prefix_tokens", None)
                    if (
                        settings.cache_prompt
                        and callable(preview)
                        and (settings.cache_idle_slots or settings.parallel == 1)
                    ):
                        reused_tokens = preview(
                            prompt_ids,
                            cache_type_k=settings.cache_type_k,
                            cache_type_v=settings.cache_type_v,
                            context_length=manager.model_info().context_length,
                            minimum_tokens=settings.cache_reuse,
                            cache_idle_slots=settings.cache_idle_slots,
                        )
                    return memory_admission.decide(
                        bundle,
                        max(prompt_tokens - reused_tokens, 0) + body.max_tokens,
                        key_bits=_cache_bits(settings.cache_type_k),
                        value_bits=_cache_bits(settings.cache_type_v),
                        prefix_cache_bytes=int(
                            cache_stats.get(
                                "prefix_bytes", cache_stats.get("bytes", 0)
                            )
                        ),
                    )

                decision = admission_check()
                if decision.status is AdmissionStatus.REJECT:
                    return _error(
                        503,
                        decision.reason or "request exceeds unified-memory capacity",
                        "capacity_error",
                    )
            stream = jobs.schedule(
                lambda cancel: selected_engine.stream(
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
                ),
                admission=admission_check,
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

    @app.post("/v1/embeddings")
    async def embeddings(
        body: EmbeddingRequest,
        auth: None | JSONResponse = authorization_dependency,
    ) -> Any:
        if auth is not None:
            return auth
        invalid = _validate_task_model(manager, body.model)
        if invalid is not None:
            return invalid
        service = services.get("embedding")
        if service is None:
            return _error(400, "the loaded model does not support embeddings", "capability_error")
        values = [body.input] if isinstance(body.input, str) else body.input
        try:
            result = await _call_service(
                jobs,
                service,
                values,
                input_type=body.input_type,
                dimensions=body.dimensions,
            )
        except QueueFullError:
            return _error(429, "task queue is full", "rate_limit_error")
        except RequestTimeoutError as exc:
            return _error(408, str(exc), "timeout_error")
        except Exception as exc:
            return _error(500, str(exc), "server_error")
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        vectors = payload.get("embeddings", ())
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": list(vector)}
                for index, vector in enumerate(vectors)
            ],
            "model": body.model,
            "usage": {
                "prompt_tokens": payload.get("prompt_tokens", 0),
                "total_tokens": payload.get("prompt_tokens", 0),
            },
            "mlx": {
                "dimensions": payload.get("dimensions"),
                "input_type": payload.get("input_type", body.input_type),
            },
        }

    @app.post("/v1/rerank")
    async def rerank_endpoint(
        body: RerankRequest,
        auth: None | JSONResponse = authorization_dependency,
    ) -> Any:
        if auth is not None:
            return auth
        invalid = _validate_task_model(manager, body.model)
        if invalid is not None:
            return invalid
        service = services.get("rerank")
        if service is None:
            return _error(400, "the loaded model does not support reranking", "capability_error")
        try:
            result = await _call_service(
                jobs,
                service,
                body.query,
                body.documents,
                instruction=body.instruction,
                top_k=body.top_n,
            )
        except QueueFullError:
            return _error(429, "task queue is full", "rate_limit_error")
        except RequestTimeoutError as exc:
            return _error(408, str(exc), "timeout_error")
        except Exception as exc:
            return _error(500, str(exc), "server_error")
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        items = payload.get("items", ())
        return {
            "id": f"rerank-{uuid.uuid4().hex}",
            "object": "list",
            "model": body.model,
            "results": [dict(item) for item in items],
        }

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
        request: Request,
        auth: None | JSONResponse = authorization_dependency,
    ) -> Any:
        if auth is not None:
            return auth
        service = services.get("transcription")
        if service is None:
            return _error(
                400,
                "the loaded model does not support transcription",
                "capability_error",
            )
        try:
            form = await request.form()
            model = str(form.get("model", ""))
            invalid = _validate_task_model(manager, model)
            if invalid is not None:
                return invalid
            uploaded = form.get("file")
            if uploaded is None or not hasattr(uploaded, "read"):
                return _error(422, "file is required", "invalid_request_error")
            audio = await uploaded.read()
            if len(audio) > MAX_AUDIO_BYTES:
                return _error(413, "audio file exceeds 25 MiB", "invalid_request_error")
            result = await _call_service(
                jobs,
                service,
                audio,
                media_type=str(getattr(uploaded, "content_type", "") or ""),
                language=form.get("language"),
                task=str(form.get("task", "transcribe")),
                request=request,
            )
        except QueueFullError:
            return _error(429, "task queue is full", "rate_limit_error")
        except RequestTimeoutError as exc:
            return _error(408, str(exc), "timeout_error")
        except AudioUploadError as exc:
            return _error(415, str(exc), "invalid_request_error")
        except ClientDisconnected:
            return _error(499, "client disconnected", "cancelled_error")
        except Exception as exc:
            return _error(500, str(exc), "server_error")
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        return {
            "text": payload.get("text", ""),
            **({"language": payload["language"]} if payload.get("language") else {}),
            "mlx": {
                key: value
                for key, value in payload.items()
                if key in {"duration", "segments", "timings"}
            },
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


def _validate_task_model(manager: ModelManager, requested: str) -> JSONResponse | None:
    try:
        loaded = manager.model_info().id
    except RuntimeError:
        return _error(503, "no model is loaded", "capacity_error")
    if requested != loaded:
        return _error(404, f"model {requested!r} is not loaded", "invalid_request_error")
    return None


async def _call_service(
    scheduler: GenerationScheduler,
    service: Callable[..., Any],
    *args: Any,
    request: Request | None = None,
    **kwargs: Any,
) -> Any:
    cancel = threading.Event()
    try:
        parameters = inspect.signature(service).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "cancel" in parameters:
        kwargs["cancel"] = cancel
    worker = asyncio.create_task(
        asyncio.to_thread(
            scheduler.execute,
            lambda: service(*args, **kwargs),
            apply_timeout=True,
        )
    )
    try:
        while not worker.done():
            done, _ = await asyncio.wait({worker}, timeout=0.1)
            if done:
                break
            if request is not None and await request.is_disconnected():
                cancel.set()
                try:
                    await worker
                except Exception:
                    pass
                raise ClientDisconnected
        result = await worker
    except BaseException:
        cancel.set()
        raise
    return await result if inspect.isawaitable(result) else result


class ClientDisconnected(RuntimeError):
    pass


async def _resolve_message_images(
    messages: list[dict[str, Any]], resolver: ImageResolver
) -> list[dict[str, Any]]:
    references = [
        part["image_url"]["url"]
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    if len(references) > 4:
        raise MediaError("chat requests support at most four images")
    resolved = iter(await asyncio.gather(*(resolver.resolve(source) for source in references)))
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") != "image_url":
                parts.append(part)
                continue
            image = next(resolved)
            encoded = base64.b64encode(image.content).decode("ascii")
            parts.append(
                {
                    **part,
                    "image_url": {
                        **part["image_url"],
                        "url": f"data:{image.media_type};base64,{encoded}",
                    },
                    "mlx_digest": image.source_digest,
                }
            )
        normalized.append({**message, "content": parts})
    return normalized


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
