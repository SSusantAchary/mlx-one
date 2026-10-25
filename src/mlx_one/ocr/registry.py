"""Explicit, task-scoped OCR capabilities. No generic-chat aliases for recognizers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OCRCapabilities:
    family: str
    tasks: frozenset[str]
    markdown: bool = False
    geometry: bool = False
    line_only: bool = False
    language_hint: bool = False
    crop_first: bool = True


CAPABILITIES = {
    "trocr": OCRCapabilities(
        "trocr", frozenset({"ocr.recognize", "ocr.region", "ocr.batch"}), line_only=True
    ),
    "got_ocr2": OCRCapabilities(
        "got_ocr2",
        frozenset({"ocr.recognize", "ocr.document", "ocr.region", "ocr.batch"}),
        markdown=True,
    ),
    "florence2": OCRCapabilities(
        "florence2",
        frozenset({"ocr.recognize", "ocr.detect", "ocr.region", "ocr.document", "ocr.batch"}),
        geometry=True,
    ),
    "paddleocr_vl": OCRCapabilities(
        "paddleocr_vl",
        frozenset(
            {"ocr.recognize", "ocr.document", "ocr.region", "ocr.table", "ocr.formula", "ocr.batch"}
        ),
        markdown=True,
    ),
    "lightonocr": OCRCapabilities(
        "lightonocr",
        frozenset({"ocr.recognize", "ocr.document", "ocr.region", "ocr.batch"}),
        markdown=True,
    ),
    "deepseek_ocr": OCRCapabilities(
        "deepseek_ocr",
        frozenset({"ocr.recognize", "ocr.document", "ocr.region", "ocr.batch"}),
        markdown=True,
    ),
}


def ocr_family(config):
    kind = config.get("model_type")
    if kind == "vision-encoder-decoder" and config.get("decoder", {}).get("model_type") == "trocr":
        return "trocr"
    if kind in {"GOT", "got"}:
        return "got_ocr2"
    if kind == "got_ocr2":
        raise ValueError(
            "HF GOT variant has a different vision/prompt layout; use the pinned "
            "mlx-community/GOT-OCR2_0-4bit checkpoint"
        )
    if kind == "mistral3" and "LightOnOCRForConditionalGeneration" in config.get(
        "architectures", []
    ):
        return "lightonocr"
    if kind == "deepseekocr":
        return "deepseek_ocr"
    if kind in CAPABILITIES:
        return kind
    raise ValueError(f"unsupported native OCR model type: {kind!r}")


def capabilities(family, *, handwriting=False):
    cap = CAPABILITIES[family]
    if handwriting and family == "trocr":
        from dataclasses import replace

        cap = replace(cap, tasks=cap.tasks | {"ocr.handwriting"})
    return cap
