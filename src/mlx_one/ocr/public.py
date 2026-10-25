"""Python OCR engine, sequential documents and stable page-level stream events."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict

from mlx_one.engine.batching import BatchingConfig
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.service import InferenceService
from mlx_one.ocr.documents import check_control, iter_pages
from mlx_one.ocr.loading import LoadedOCR, load_ocr_model
from mlx_one.ocr.schemas import OCRInput, OCRLimits, OCRPage, OCRResult


class OCREngine:
    def __init__(self, model, *, task="ocr.recognize", limits=None, **load_options):
        self.task, self.limits = task, limits or OCRLimits()
        self.service = InferenceService(
            batching_config=BatchingConfig(
                max_sequences=1,
                max_scheduled_tokens=32768,
                task_transient_budget_bytes=self.limits.memory_budget_bytes,
            )
        )
        self.service.register_task(
            "ocr",
            model,
            loader=load_ocr_model if not (isinstance(model, LoadedOCR)) else None,
            **load_options,
        )
        self._closed = False

    @property
    def bundle(self):
        _, lease = self.service._task_lease("ocr", TaskKind.OCR_RECOGNIZE)
        try:
            return lease.runtime.value
        finally:
            lease.release()

    @property
    def capabilities(self):
        return self.bundle.capabilities

    def _request(self, source, options):
        if isinstance(source, OCRInput):
            if options:
                raise ValueError("typed OCRInput cannot be combined with request options")
            return source
        return OCRInput(
            source,
            task=options.pop("task", self.task),
            limits=options.pop("limits", self.limits),
            **options,
        )

    def stream(self, source, *, cancel=None, **options):
        if self._closed:
            raise RuntimeError("OCR engine is closed")
        request = self._request(source, options)
        cap = self.capabilities
        if request.task not in cap.tasks:
            raise ValueError(f"{cap.family} does not support {request.task}")
        if request.language is not None and not cap.language_hint:
            raise ValueError("selected checkpoint has no validated language-hint processor")
        if request.output_format == "markdown" and not cap.markdown:
            raise ValueError("selected checkpoint does not produce Markdown")
        if cap.line_only and request.task == "ocr.document":
            raise ValueError("TrOCR recognizes cropped single lines, not full-page documents")
        event = cancel or threading.Event()
        deadline = time.monotonic() + request.limits.timeout_seconds
        started, pages = time.perf_counter(), []
        iterator = iter_pages(request, cancel=event, deadline=deadline)
        try:
            for page in iterator:
                check_control(event, deadline)
                if cap.line_only and (page.metadata.get("pdf_size_points") or request.pages):
                    raise ValueError("TrOCR requires individual cropped line images")
                yield {
                    "event": "page_started",
                    "page": page.index,
                    "schema_version": "mlx-one-ocr-v1",
                }
                if page.embedded_text is not None:
                    if request.region or request.task in {"ocr.detect", "ocr.table", "ocr.formula"}:
                        raise ValueError("embedded PDF text cannot supply OCR crops or structure")
                    output = OCRPage(
                        page.index,
                        page.original_size,
                        page.rendered_size,
                        page.embedded_text,
                        metadata=page.metadata,
                    )
                else:
                    output = self.service.run_task(
                        "ocr",
                        TaskKind(request.task),
                        {
                            "image": page.image,
                            "request": request,
                            "page": page.index,
                            "metadata": page.metadata,
                        },
                        cancel=event,
                        deadline=deadline,
                    ).output
                check_control(event, deadline)
                pages.append(output)
                yield {
                    "event": "page_completed",
                    "page": asdict(output),
                    "schema_version": "mlx-one-ocr-v1",
                }
            result = OCRResult(
                "\n\n".join(p.text for p in pages),
                "\n\n".join(p.markdown for p in pages)
                if all(p.markdown is not None for p in pages)
                else None,
                pages,
                self.bundle.model_id,
                request.task,
                request.language,
                {"total_seconds": time.perf_counter() - started},
                {
                    "processed_pages": len(pages),
                    "page_count": pages[0].metadata.get("source_page_count", len(pages)),
                    "revision": self.bundle.revision,
                    "capabilities": {**asdict(cap), "tasks": sorted(cap.tasks)},
                    "measurement_type": "python-engine",
                    "coordinate_convention": "top-left pixel xyxy, zero-based pages",
                    "checkpoint_status": "PENDING_HARDWARE_TEST",
                },
            )
            yield {
                "event": "document_completed",
                "result": result.to_dict(),
                "schema_version": result.schema_version,
            }
        finally:
            iterator.close()

    def recognize(self, image, *, cancel=None, **options):
        request = self._request(image, options)
        final = None
        for event in self.stream(request, cancel=cancel):
            if event["event"] == "document_completed":
                final = event["result"]
        pages = [OCRPage(**p) for p in final.pop("pages")]
        # The event transport is JSON-shaped; recover typed layout objects for Python callers.
        from mlx_one.ocr.schemas import LayoutElement

        for page in pages:
            page.elements = [
                LayoutElement(**e) if isinstance(e, dict) else e for e in page.elements
            ]
        return OCRResult(pages=pages, **final)

    def parse_document(self, document, **options):
        return self.recognize(document, task=options.pop("task", "ocr.document"), **options)

    def batch(self, images, **options):
        # Consume bounded iterators lazily; each item is a separate ordered document.
        for index, image in enumerate(images):
            if index >= self.limits.max_pages:
                raise ValueError("OCR batch exceeds maximum image count")
            yield self.recognize(image, task="ocr.batch", **options)

    def close(self):
        if not self._closed:
            self._closed = True
            self.service.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
