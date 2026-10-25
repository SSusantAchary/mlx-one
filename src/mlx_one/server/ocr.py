"""Authenticated OCR uploads sharing the native task actor and model lifecycle."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, replace

from fastapi import Request
from fastapi.responses import StreamingResponse


@dataclass(frozen=True)
class OCRServingBundle:
    ocr: object
    chat_template: object = None
    parameter_count: object = None

    def __getattr__(self, name):
        return getattr(self.ocr, name)


async def _run_request(request, function, event, on_cancel=None):
    worker = asyncio.create_task(asyncio.to_thread(function))
    try:
        while not worker.done():
            await asyncio.wait({worker}, timeout=0.1)
            if await request.is_disconnected():
                event.set()
                raise asyncio.CancelledError()
        return await worker
    finally:
        if not worker.done():
            event.set()

            # Drain in the background: buffers belong to the worker until GPU work stops.
            def drain(done):
                if not done.cancelled():
                    done.exception()
                    if on_cancel is not None:
                        on_cancel()

            worker.add_done_callback(drain)


def install_ocr_routes(app, manager, services, auth_dependency):

    from mlx_one.ocr.documents import parse_page_range
    from mlx_one.ocr.schemas import OCRInput
    from mlx_one.server.app import _error, _validate_task_model

    @app.post("/v1/ocr")
    async def ocr(request: Request, auth=auth_dependency):
        if auth is not None:
            return auth
        service = services.get("ocr")
        if service is None:
            return _error(400, "loaded model has no native OCR task", "capability_error")
        event = threading.Event()
        try:
            async with request.form(max_files=1, max_fields=20, max_part_size=32 * 1024**2) as form:
                invalid = _validate_task_model(manager, str(form.get("model", "")))
                if invalid is not None:
                    return invalid
                uploaded = form.get("file")
                if not callable(getattr(uploaded, "read", None)):
                    raise ValueError(
                        "file upload is required; server-side paths and URLs are rejected"
                    )
                content = await uploaded.read(32 * 1024**2 + 1)
                if len(content) > 32 * 1024**2:
                    return _error(413, "OCR upload exceeds 32 MiB", "invalid_request_error")
                if not content.startswith(
                    (b"%PDF-", b"\x89PNG", b"\xff\xd8", b"II*\0", b"MM\0*", b"RIFF")
                ):
                    return _error(415, "unsupported OCR document type", "invalid_request_error")
                region = json.loads(str(form["region"])) if "region" in form else None
                options = OCRInput(
                    content,
                    task=str(form.get("task", "ocr.recognize")),
                    output_format=str(form.get("output_format", "json")),
                    pages=parse_page_range(str(form.get("page_range", ""))),
                    dpi=int(str(form.get("dpi", "144"))),
                    max_tokens=int(str(form.get("max_tokens", "512"))),
                    pdf_policy=str(form.get("pdf_policy", "visual")),
                    region=tuple(region) if region else None,
                    limits=replace(
                        service.limits,
                        max_pages=min(
                            service.limits.max_pages, int(str(form.get("max_pages", "100")))
                        ),
                    ),
                )
                if str(form.get("stream", "false")).lower() not in {"false", "true"}:
                    raise ValueError("stream must be true or false")
                streaming = str(form.get("stream", "false")).lower() == "true"
            if not streaming:
                result = await _run_request(
                    request, lambda: service.recognize(options, cancel=event), event
                )
                return result.to_dict()

            async def frames():
                iterator = service.stream(options, cancel=event)

                def step():
                    return next(iterator, None)

                try:
                    while True:
                        item = await _run_request(request, step, event, on_cancel=iterator.close)
                        if item is None:
                            break
                        yield "event: " + item["event"] + "\ndata: " + json.dumps(item) + "\n\n"
                except Exception as exc:
                    yield (
                        "event: error\ndata: "
                        + json.dumps({"message": str(exc), "schema_version": "mlx-one-ocr-v1"})
                        + "\n\n"
                    )
                finally:
                    event.set()
                    # Closing a still executing generator is illegal. Advance/close on
                    # the worker only after its current bounded page has returned.
                    try:
                        await asyncio.to_thread(iterator.close)
                    except ValueError:
                        pass

            return StreamingResponse(frames(), media_type="text/event-stream")
        except TimeoutError as exc:
            return _error(408, str(exc), "timeout_error")
        except (ValueError, OSError) as exc:
            return _error(400, str(exc), "invalid_request_error")
        except Exception as exc:
            from mlx_one.engine.execution_context import QueueFullError, RequestTimeoutError

            if isinstance(exc, QueueFullError):
                return _error(429, str(exc), "rate_limit_error")
            if isinstance(exc, RequestTimeoutError):
                return _error(408, str(exc), "timeout_error")
            return _error(500, str(exc), "server_error")
