"""Parse only structure and coordinates actually emitted by an OCR model."""

import re
from html.parser import HTMLParser

from mlx_one.ocr.schemas import LayoutElement


def florence_regions(text, transform, page):
    text = re.sub(r"</?s>|<pad>", "", text)
    pattern = re.compile(r"([^<>]*?)((?:<loc_\d+>){8})")
    elements = []
    cursor = 0
    for match in pattern.finditer(text):
        if text[cursor : match.start()].strip():
            raise ValueError("unparsed Florence region output")
        values = [int(v) for v in re.findall(r"<loc_(\d+)>", match[2])]
        if any(v > 999 for v in values):
            raise ValueError("Florence coordinates must be in the 1000-bin vocabulary")
        w, h = transform.model_size
        polygon = [
            transform.to_original((values[i] + 0.5) * w / 1000, (values[i + 1] + 0.5) * h / 1000)
            for i in range(0, 8, 2)
        ]
        area = (
            abs(
                sum(
                    polygon[i][0] * polygon[(i + 1) % 4][1]
                    - polygon[(i + 1) % 4][0] * polygon[i][1]
                    for i in range(4)
                )
            )
            / 2
        )
        if area == 0:
            raise ValueError("Florence emitted a degenerate polygon")
        xs, ys = zip(*polygon, strict=True)
        elements.append(
            LayoutElement(
                "text_line",
                match[1],
                page,
                polygon,
                [min(xs), min(ys), max(xs), max(ys)],
                len(elements),
            )
        )
        cursor = match.end()
    if text[cursor:].strip() or not elements and text.strip():
        raise ValueError("malformed Florence OCR_WITH_REGION output")
    return elements


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.rows, self.row, self.cell = [], None, None, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            if self.rows is not None:
                raise ValueError("nested table markup is unsupported")
            self.rows = []
        elif tag == "tr" and self.rows is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            spans = {key: int(attrs.get(key, 1)) for key in ("rowspan", "colspan")}
            if any(not 1 <= v <= 1000 for v in spans.values()):
                raise ValueError("invalid table span")
            self.cell = {"text": "", "header": tag == "th", **spans}

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.rows is not None:
            self.tables.append({"format": "html", "rows": self.rows})
            self.rows = None


def parse_output(bundle, request, raw, prepared, page):
    raw = raw.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
    elements, tables, formulas = [], [], []
    if bundle.architecture == "florence2" and request.task in {"ocr.detect", "ocr.region"}:
        elements = florence_regions(raw, prepared.transform, page)
        raw = "\n".join(e.text for e in elements)
    if request.task == "ocr.table" and "<table" in raw.lower():
        parser = TableParser()
        parser.feed(raw)
        if parser.rows is not None or parser.cell is not None:
            raise ValueError("incomplete model-generated HTML table")
        tables = parser.tables
    if request.task == "ocr.formula":
        formulas = [raw] if raw.strip() else []
    markdown = raw if request.output_format == "markdown" and bundle.capabilities.markdown else None
    return raw, markdown, elements, tables, formulas
