"""Verify rendered site links and anchors, including the GitHub Pages path prefix."""

from __future__ import annotations

import argparse
import json
from functools import cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "a" and values.get("name"):
            self.ids.add(values["name"])
        for key in ("href", "src", "action"):
            if values.get(key):
                self.links.append(values[key])


@cache
def parse(path):
    document = Document()
    document.feed(path.read_text())
    return document


def check(directory):
    directory = directory.resolve()
    pages = sorted(directory.rglob("*.html"))
    errors, links = [], 0
    for page in pages:
        for url in parse(page).links:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path and not parts.fragment:
                continue
            path = unquote(parts.path)
            if path.startswith("/mlx-one/"):
                target = directory / path[len("/mlx-one/"):]
            elif path.startswith("/"):
                target = directory / path.lstrip("/")
            else:
                target = (page.parent / path).resolve() if path else page
            if target.is_dir():
                target /= "index.html"
            links += 1
            if not target.is_relative_to(directory) or not target.is_file():
                errors.append(f"{page.relative_to(directory)}: missing {url}")
            elif parts.fragment and target.suffix == ".html" and unquote(parts.fragment) not in parse(target).ids:
                errors.append(f"{page.relative_to(directory)}: missing anchor {url}")
    report = {"html_pages": len(pages), "local_links": links, "errors": sorted(set(errors))}
    if not pages:
        report["errors"].append("no generated HTML pages")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).parent / "dist")
    raise SystemExit(bool(check(parser.parse_args().directory)["errors"]))
