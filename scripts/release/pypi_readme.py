"""Generate the PyPI description from README.md with working absolute source/image links."""

from __future__ import annotations

import argparse
import posixpath
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "https://github.com/SSusantAchary/mlx-one"
RAW = "https://raw.githubusercontent.com/SSusantAchary/mlx-one/main"
IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def render(text, root=ROOT):
    def absolute(url):
        parts = urlsplit(url)
        if parts.scheme or parts.netloc or not parts.path:
            return url
        relative = posixpath.normpath(unquote(parts.path))
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or not target.exists():
            raise ValueError(f"invalid PyPI README source link: {url}")
        suffix = ("?" + parts.query if parts.query else "") + (
            "#" + parts.fragment if parts.fragment else ""
        )
        encoded = quote(relative, safe="/")
        if target.suffix.lower() in IMAGES:
            return f"{RAW}/{encoded}{suffix}"
        kind = "tree" if target.is_dir() else "blob"
        return f"{REPOSITORY}/{kind}/main/{encoded}{suffix}"

    def prose(value):
        value = re.sub(r"\]\(([^\s)]+)\)", lambda m: "](" + absolute(m[1]) + ")", value)
        return re.sub(
            r'(href|src)="([^\"]+)"',
            lambda m: m[1] + '="' + absolute(m[2]) + '"',
            value,
        )

    # Installation commands and example inputs remain byte-for-byte source text.
    chunks = re.split(r"(```[\s\S]*?```|~~~[\s\S]*?~~~)", text)
    result = "".join(prose(chunk) if i % 2 == 0 else chunk for i, chunk in enumerate(chunks))
    return "<!-- Generated from README.md by scripts/release/pypi_readme.py. -->\n\n" + result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the description is stale")
    args = parser.parse_args()
    destination = ROOT / "README.pypi.md"
    content = render((ROOT / "README.md").read_text())
    if args.check:
        if not destination.is_file() or destination.read_text() != content:
            parser.exit(1, "README.pypi.md is stale; run python scripts/release/pypi_readme.py\n")
        print("PyPI README matches its source; local source and image links are absolute.")
    else:
        destination.write_text(content)
        print("Generated README.pypi.md from README.md.")


if __name__ == "__main__":
    main()
