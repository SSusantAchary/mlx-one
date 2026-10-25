"""Bounded sequential local image/TIFF/PDF ingestion. Never fetch remote URLs."""

from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from mlx_one.ocr.schemas import OCRInput

_PDF_LOCK = threading.Lock()


@dataclass
class DocumentPage:
    index: int
    image: object | None
    original_size: tuple[int, int]
    rendered_size: tuple[int, int]
    embedded_text: str | None
    metadata: dict


@dataclass(frozen=True)
class PixelTransform:
    """Map model coordinates through explicit padding, scaling and a source crop."""

    crop: tuple[float, float, float, float]
    model_size: tuple[int, int]
    padding: tuple[float, float] = (0, 0)
    content_size: tuple[int, int] | None = None

    def to_original(self, x, y):
        x0, y0, x1, y1 = self.crop
        w, h = self.content_size or self.model_size
        return [
            (x - self.padding[0]) * (x1 - x0) / w + x0,
            (y - self.padding[1]) * (y1 - y0) / h + y0,
        ]

    def to_model(self, x, y):
        x0, y0, x1, y1 = self.crop
        w, h = self.content_size or self.model_size
        return [
            (x - x0) * w / (x1 - x0) + self.padding[0],
            (y - y0) * h / (y1 - y0) + self.padding[1],
        ]


def check_control(cancel, deadline):
    if cancel is not None and cancel.is_set():
        raise CancelledError("OCR request cancelled")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("OCR request timed out")


def parse_page_range(value):
    """CLI/HTTP ranges are one-based inclusive; Python page indices are zero-based."""
    if not value:
        return None
    pages = []
    for term in value.split(","):
        pieces = term.split("-")
        if len(pieces) not in {1, 2} or not all(p.strip().isdigit() for p in pieces):
            raise ValueError("page range must look like 1-5,8")
        first, last = int(pieces[0]), int(pieces[-1])
        if first < 1 or last < first or last - first > 10000:
            raise ValueError("invalid or excessive page range")
        pages.extend(range(first - 1, last))
    if len(set(pages)) != len(pages):
        raise ValueError("page range contains duplicates")
    return tuple(sorted(pages))


def _selection(count, request):
    if count < 1 or count > request.limits.max_pages:
        raise ValueError("document page count exceeds max_pages; no pages were skipped")
    pages = tuple(range(count)) if request.pages is None else tuple(sorted(request.pages))
    if max(pages) >= count:
        raise ValueError("selected page is outside the document")
    return pages


def _check_size(width, height, request):
    if min(width, height) < 1 or width * height > request.limits.max_pixels:
        raise ValueError("decoded page exceeds pixel limit")
    if width * height * 4 > min(
        request.limits.max_rendered_bytes, request.limits.memory_budget_bytes
    ):
        raise ValueError("page exceeds rendered-byte or memory limit")


def iter_pages(request: OCRInput, *, cancel=None, deadline=None):
    """The yielded image is borrowed until the next iteration; close the iterator early."""
    from PIL import Image, ImageOps

    value = request.source
    check_control(cancel, deadline)
    if isinstance(value, (str, Path)):
        if str(value).startswith(("https://", "http://", "file://")):
            raise ValueError("OCR accepts local documents only; URL fetching is disabled")
        path = Path(value).expanduser()
        if path.stat().st_size > request.limits.max_file_bytes:
            raise ValueError("document exceeds encoded file limit")
        # At most one bounded encoded document, never an unbounded list of raster pages.
        data = path.read_bytes()
    elif isinstance(value, (bytes, bytearray)):
        data = bytes(value)
        if len(data) > request.limits.max_file_bytes:
            raise ValueError("document exceeds encoded file limit")
    elif isinstance(value, Image.Image):
        data = None
    else:
        raise ValueError("source must be a local path, bytes or Pillow image")
    if data is not None and data.startswith(b"%PDF-"):
        yield from _pdf_pages(data, request, cancel, deadline)
        return
    owned = data is not None
    source = Image.open(BytesIO(data)) if owned else value
    try:
        if source.format not in {None, "PNG", "JPEG", "TIFF", "WEBP"}:
            raise ValueError("supported images: PNG, JPEG, TIFF, WebP")
        if source.format != "TIFF" and getattr(source, "n_frames", 1) != 1:
            raise ValueError("animated image input is unsupported")
        total_bytes = 0
        for index in _selection(getattr(source, "n_frames", 1), request):
            check_control(cancel, deadline)
            source.seek(index)
            _check_size(*source.size, request)
            started = time.perf_counter()
            oriented = ImageOps.exif_transpose(source)
            image = None
            try:
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    with oriented.convert("RGBA") as rgba:
                        image = Image.new("RGB", rgba.size, "white")
                        image.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    image = oriented.convert("RGB")
                total_bytes += image.width * image.height * 3
                if total_bytes > request.limits.max_rendered_bytes:
                    raise ValueError("document exceeds total rendered-byte limit")
                yield DocumentPage(
                    index,
                    image,
                    image.size,
                    image.size,
                    None,
                    {
                        "path": "visual",
                        "exif_oriented": True,
                        "source_page_count": getattr(source, "n_frames", 1),
                        "decode_seconds": time.perf_counter() - started,
                    },
                )
            finally:
                if image is not None:
                    image.close()
                if oriented is not source:
                    oriented.close()
    finally:
        if owned:
            source.close()


def _pdf_pages(data, request, cancel, deadline):
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pip install 'mlx-one[ocr]'") from exc
    # PDFium is not thread-safe. Serialize its lifetime, including between pages.
    while not _PDF_LOCK.acquire(timeout=0.1):
        check_control(cancel, deadline)
    doc = None
    total_bytes = 0
    try:
        doc = pdfium.PdfDocument(data)
        for index in _selection(len(doc), request):
            check_control(cancel, deadline)
            page = doc[index]
            try:
                width, height = page.get_size()
                scale = request.dpi / 72
                import math

                size = (math.ceil(width * scale), math.ceil(height * scale))
                _check_size(*size, request)
                metadata = {
                    "pdf_size_points": [width, height],
                    "source_page_count": len(doc),
                    "dpi": request.dpi,
                    "rotation": page.get_rotation(),
                    "cropbox_points": page.get_cropbox(),
                    "coordinate_space": "rendered page pixels at requested DPI",
                }
                embedded = None
                if request.pdf_policy != "visual":
                    textpage = page.get_textpage()
                    try:
                        total_bytes += textpage.count_chars() * 4
                        if total_bytes > request.limits.max_rendered_bytes:
                            raise ValueError("embedded text exceeds request byte budget")
                        embedded = textpage.get_text_range()
                    finally:
                        textpage.close()
                # Deterministic page-level policy: any non-whitespace embedded text wins.
                # Never append OCR output to the embedded text on the same page.
                if request.pdf_policy == "embedded_text" or (
                    request.pdf_policy == "auto" and embedded and embedded.strip()
                ):
                    metadata["path"] = "embedded_text"
                    yield DocumentPage(index, None, size, size, embedded or "", metadata)
                    continue
                started = time.perf_counter()
                bitmap = page.render(scale=scale)
                image = None
                try:
                    image = bitmap.to_pil().convert("RGB")
                    _check_size(*image.size, request)
                    total_bytes += image.width * image.height * 3
                    if total_bytes > request.limits.max_rendered_bytes:
                        raise ValueError("document exceeds total rendered-byte limit")
                    metadata.update(path="visual", render_seconds=time.perf_counter() - started)
                    yield DocumentPage(index, image, image.size, image.size, None, metadata)
                finally:
                    if image is not None:
                        image.close()
                    bitmap.close()
            finally:
                page.close()
    finally:
        if doc is not None:
            doc.close()
        _PDF_LOCK.release()
