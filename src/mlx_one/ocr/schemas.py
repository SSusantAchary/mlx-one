"""Versioned OCR contracts; absent predictions remain absent."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TASKS = frozenset(
    {
        "ocr.recognize",
        "ocr.detect",
        "ocr.document",
        "ocr.batch",
        "ocr.handwriting",
        "ocr.formula",
        "ocr.table",
        "ocr.region",
    }
)
FORMATS = frozenset({"text", "json", "markdown"})


@dataclass(frozen=True)
class OCRLimits:
    max_file_bytes: int = 32 * 1024**2
    max_pages: int = 100
    max_pixels: int = 20_000_000
    max_dpi: int = 300
    max_tiles: int = 9
    max_rendered_bytes: int = 256 * 1024**2
    memory_budget_bytes: int = 8 * 1024**3
    timeout_seconds: float = 300

    def __post_init__(self):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("OCR limits must be finite and positive")


@dataclass(frozen=True)
class OCRInput:
    source: Any
    task: str = "ocr.recognize"
    pages: tuple[int, ...] | None = None
    region: tuple[float, float, float, float] | None = None
    language: str | None = None
    output_format: str = "text"
    dpi: int = 144
    pdf_policy: str = "visual"
    max_tokens: int = 512
    limits: OCRLimits = field(default_factory=OCRLimits)

    def __post_init__(self):
        if self.task not in TASKS or self.output_format not in FORMATS:
            raise ValueError("unsupported OCR task or output format")
        if self.pdf_policy not in {"visual", "embedded_text", "auto"}:
            raise ValueError("PDF policy must be visual, embedded_text or auto")
        if not 36 <= self.dpi <= self.limits.max_dpi:
            raise ValueError("DPI outside request limits")
        if isinstance(self.max_tokens, bool) or not 1 <= self.max_tokens <= 16384:
            raise ValueError("max_tokens must be between 1 and 16384")
        if self.pages is not None and (
            not self.pages
            or len(set(self.pages)) != len(self.pages)
            or any(type(p) is not int or p < 0 for p in self.pages)
        ):
            raise ValueError("pages must be unique zero-based nonnegative indices")
        if self.region is not None and (
            len(self.region) != 4
            or not all(math.isfinite(v) for v in self.region)
            or self.region[0] < 0
            or self.region[1] < 0
            or self.region[2] <= self.region[0]
            or self.region[3] <= self.region[1]
        ):
            raise ValueError("region must be a finite, ordered pixel box")


@dataclass
class LayoutElement:
    type: str
    text: str
    page: int
    polygon: list[list[float]] | None = None
    bbox: list[float] | None = None
    reading_order: int | None = None
    confidence: float | None = None
    structure: dict | None = None


@dataclass
class OCRPage:
    page: int
    original_size: tuple[int, int]
    rendered_size: tuple[int, int]
    text: str
    markdown: str | None = None
    elements: list[LayoutElement] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class OCRResult:
    text: str
    markdown: str | None
    pages: list[OCRPage]
    model: str
    task: str
    language: str | None
    timings: dict
    metadata: dict
    schema_version: str = "mlx-one-ocr-v1"

    def to_dict(self):
        return asdict(self)

    def format(self, output_format):
        if output_format == "json":
            return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2)
        if output_format == "markdown":
            if self.markdown is None:
                raise ValueError("selected model/task does not produce Markdown")
            return self.markdown
        if output_format == "text":
            return self.text
        raise ValueError("unsupported OCR output format")

    def save(self, path):
        path = Path(path).expanduser()
        extension = {".json": "json", ".md": "markdown", ".txt": "text"}.get(path.suffix.lower())
        if extension is None:
            raise ValueError("output suffix must be .json, .md or .txt")
        path.write_text(self.format(extension) + "\n", encoding="utf-8")
