"""Check landing-page documentation, local links, anchors, fences and ownership.

Uses only the standard library. External URLs are counted, never requested.
Pass explicit Markdown paths to check a different selection.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from functools import cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
REPO_LINKS = {
    "github.com": (
        "/SSusantAchary/mlx-one/blob/main/",
        "/SSusantAchary/mlx-one/tree/main/",
    ),
    "raw.githubusercontent.com": ("/SSusantAchary/mlx-one/main/",),
}
OWNERS = {
    "README.md": [
        ("model-performance-summary:start", "model-performance-summary:end"),
        ("MLX-ONE-VLM-BENCHMARK-START", "MLX-ONE-VLM-BENCHMARK-END"),
        ("MLX-ONE-AUDIO-BENCHMARK-START", "MLX-ONE-AUDIO-BENCHMARK-END"),
        ("MLX-ONE-OCR-BENCHMARK-START", "MLX-ONE-OCR-BENCHMARK-END"),
    ],
}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])
        if tag == "a" and values.get("name"):
            self.ids.append(values["name"])
        key = "href" if tag == "a" else "src" if tag == "img" else None
        if key and values.get(key):
            self.links.append(values[key])


@cache
def parse(path):
    text = path.read_text()
    prose, errors, snippets = [], [], []
    fence, language, code = None, None, []
    for line in text.splitlines():
        mark = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if (
                mark
                and mark[1][0] == fence[0]
                and len(mark[1]) >= len(fence)
                and not mark[2].strip()
            ):
                snippets.append((language, "\n".join(code)))
                fence = None
                code = []
            else:
                code.append(line)
        elif mark:
            fence, language = mark[1], mark[2].strip()
        else:
            prose.append(line)
    if fence:
        errors.append("unclosed code fence")
    for language, code in snippets:
        if language in {"python", "py"}:
            try:
                ast.parse(code)
            except SyntaxError as exc:
                errors.append(f"invalid Python example: {exc}")
    content = "\n".join(prose)
    parsed = Links()
    parsed.feed(content)
    seen = {}
    for line in prose:
        heading = re.match(r"^ {0,3}#{1,6}\s+(.+?)(?:\s+#+)?$", line)
        if heading:
            title = re.sub(r"!?\[([^]]*)\]\([^)]*\)", r"\1", heading[1])
            slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            parsed.ids.append(slug + (f"-{n}" if n else ""))
    if len(parsed.ids) != len(set(parsed.ids)):
        errors.append("duplicate HTML/heading anchor")
    for table in re.findall(r"(?:^\|.*\|\s*$\n?){2,}", content, re.M):
        widths = [
            len(re.split(r"(?<!\\)\|", row.strip().strip("|")))
            for row in table.strip().splitlines()
        ]
        if len(set(widths)) != 1:
            errors.append("inconsistent Markdown table columns")
    parsed.links += re.findall(r"\]\(<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\)", content)
    references = dict(re.findall(r"^\s*\[([^]]+)\]:\s*<?([^\s>]+)>?", content, re.M))
    for label in re.findall(r"!?\[[^]]+\]\[([^]]+)\]", content):
        if label not in references:
            errors.append(f"missing link reference: {label}")
        else:
            parsed.links.append(references[label])
    return text, parsed, errors, len([s for s in snippets if s[0] in {"python", "py"}])


def default_files():
    return [ROOT / "README.md"]


def check(files):
    errors, external, local, examples = [], set(), 0, 0
    for path in files:
        if not path.is_file():
            errors.append(f"{path}: missing document")
            continue
        text, parsed, syntax_errors, count = parse(path)
        label = path.relative_to(ROOT).as_posix()
        examples += count
        errors += [f"{label}: {error}" for error in syntax_errors]
        for start, end in OWNERS.get(label, []):
            a, b = f"<!-- {start} -->", f"<!-- {end} -->"
            if text.count(a) != 1 or text.count(b) != 1 or text.index(a) >= text.index(b):
                errors.append(f"{label}: invalid generated markers {start}")
        for url in parsed.links:
            parts = urlsplit(url)
            prefix = next(
                (v for v in REPO_LINKS.get(parts.netloc, ()) if parts.path.startswith(v)),
                None,
            )
            if prefix is not None:
                target = (ROOT / unquote(parts.path.removeprefix(prefix))).resolve()
            elif parts.scheme or parts.netloc:
                external.add(url)
                continue
            else:
                target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            local += 1
            if not target.exists():
                errors.append(f"{label}: missing target {url}")
            elif (
                parts.fragment
                and target.suffix == ".md"
                and unquote(parts.fragment) not in parse(target)[1].ids
            ):
                errors.append(f"{label}: missing anchor {url}")
        if label == "README.md":
            forbidden = [
                "Experimental; bounded functional smoke",
                "Text checkpoint status",
                "docs/benchmarks/inventory.json",
                "<!-- mlx-one:verified-task-results:start -->",
                "<!-- mlx-one:mlxboost-summary:start -->",
            ]
            for value in forbidden:
                if value in text:
                    errors.append(f"README.md: detailed/experimental content returned: {value}")
    return dict(
        files=len(files),
        local_links=local,
        external_urls=len(external),
        python_examples=examples,
        errors=errors,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="*")
    args = parser.parse_args()
    report = check([p.resolve() for p in args.files] if args.files else default_files())
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
