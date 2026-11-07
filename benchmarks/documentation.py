"""Presentation helpers for relocating generated reports without changing evidence."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def relative_links(markdown: str, destination: Path) -> str:
    """Rebase repository-root links emitted by the existing report renderers."""

    def replace(match):
        url = match[1]
        parts = urlsplit(url)
        if parts.scheme or parts.netloc or not parts.path:
            return match[0]
        path = Path(os.path.relpath(ROOT / parts.path, destination.parent)).as_posix()
        suffix = ("?" + parts.query if parts.query else "") + (
            "#" + parts.fragment if parts.fragment else ""
        )
        return "](" + path + suffix + ")"

    return re.sub(r"\]\(([^\s)]+)\)", replace, markdown)


def replace_block(text: str, start: str, end: str, content: str) -> str:
    """Require one existing, ordered marker pair; never insert into a homepage."""
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
        raise ValueError("document must contain exactly one ordered generated marker pair")
    return text[: text.index(start)] + content.rstrip() + text[text.index(end) + len(end) :]
