"""Stage documentation from its repository sources without duplicating maintained facts."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from insights import fallback_markdown, make_data, modality_cards_html, summary_html

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
GENERATED = HERE / ".generated"
REPOSITORY = "https://github.com/SSusantAchary/mlx-one"

# One destination per maintained source. New model-family READMEs are discovered below.
PAGES = {
    "website/pages/index.md": "index.md",
    "website/pages/insights.md": "insights/index.md",
    "website/pages/guides/generate.md": "guides/generate.md",
    "website/pages/guides/retrieval.md": "guides/retrieval.md",
    "website/pages/guides/memory.md": "guides/memory.md",
    "docs/getting-started.md": "getting-started.md",
    "docs/ui.md": "guides/serve.md",
    "docs/VLM_TESTING.md": "guides/vision.md",
    "docs/AUDIO_TESTING.md": "guides/audio.md",
    "docs/OCR_TESTING.md": "guides/ocr.md",
    "docs/models/index.md": "models/index.md",
    "docs/benchmarks/inventory.md": "reference/checkpoints.md",
    "docs/model-admission.md": "reference/admission.md",
    "docs/api/index.md": "reference/api.md",
    "docs/api/openai-compatibility.md": "reference/http.md",
    "docs/architecture/index.md": "reference/architecture.md",
    "docs/REPOSITORY_SUMMARY.md": "reference/repository.md",
    "docs/roadmap.md": "reference/roadmap.md",
    "CHANGELOG.md": "reference/changelog.md",
    "docs/benchmarks/text-performance.md": "performance/text.md",
    "docs/benchmarks/vlm/performance.md": "performance/vision.md",
    "docs/benchmarks/audio/performance.md": "performance/audio.md",
    "docs/benchmarks/ocr/performance.md": "performance/ocr.md",
    "docs/benchmarks/http-performance.md": "performance/http.md",
    "docs/benchmarks/README.md": "performance/methodology.md",
    "docs/experimental/README.md": "experimental/index.md",
    "docs/experimental/native-tasks.md": "experimental/native-tasks.md",
    "docs/experimental/mlxboost.md": "experimental/mlxboost.md",
    "docs/experimental/inference-runtime.md": "experimental/runtime.md",
    "docs/experimental/known-limitations.md": "experimental/limitations.md",
}

README_ANCHORS = {
    "quick-start": "getting-started.md",
    "installation": "getting-started.md#installation",
    "supported-modalities-and-models": "models/index.md",
    "model-specific-documentation": "models/index.md",
    "architecture": "reference/architecture.md",
    "python-cli-and-http-apis": "reference/api.md",
    "cli": "reference/api.md#cli",
    "python-api": "reference/api.md#python-api",
    "documentation": "index.md#find-your-guide",
    "citation": "reference/citation.md",
    "native-inference-engine-and-mlxboost": "experimental/mlxboost.md",
}


def stage():
    pages = dict(PAGES)
    for source in sorted((ROOT / "src/mlx_one/models").glob("*/*/README.md")):
        category, family = source.parent.parent.name, source.parent.name
        pages[source.relative_to(ROOT).as_posix()] = f"models/{category}/{family}.md"
    targets = set(pages.values()) | {"performance/index.md", "reference/citation.md"}
    if GENERATED.exists():
        # Only disposable output owned by this builder is removed.
        shutil.rmtree(GENERATED)
    GENERATED.mkdir(parents=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    sources, assets = {}, set()

    def record(path):
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

    insights = make_data(record)

    def relative(target, destination):
        return posixpath.relpath(target, posixpath.dirname(destination) or ".")

    def rewrite_url(url, source, destination):
        parts = urlsplit(url)
        if parts.scheme or parts.netloc or not parts.path:
            return url
        path = (source.parent / unquote(parts.path)).resolve()
        suffix = ("?" + parts.query if parts.query else "") + (
            "#" + parts.fragment if parts.fragment else ""
        )
        if not path.is_relative_to(ROOT):
            raise ValueError(f"link leaves the repository: {source.relative_to(ROOT)}: {url}")
        key = path.relative_to(ROOT).as_posix()
        if key == "README.md":
            if parts.fragment in README_ANCHORS:
                target, _, fragment = README_ANCHORS[parts.fragment].partition("#")
                return relative(target, destination) + ("#" + fragment if fragment else "")
            if parts.fragment and any(word in parts.fragment for word in ("performance", "latency", "handwriting")):
                return relative("performance/index.md", destination) + suffix
            if not parts.fragment:
                return relative("index.md", destination)
        if key in pages:
            return relative(pages[key], destination) + suffix
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
            asset = "assets/repository/" + key
            output = GENERATED / asset
            output.parent.mkdir(parents=True, exist_ok=True)
            if asset not in assets:
                shutil.copyfile(path, output)
                record(path)
                assets.add(asset)
            return relative(asset, destination) + suffix
        # Authored site pages can link directly to another generated site destination.
        local = posixpath.normpath(posixpath.join(posixpath.dirname(destination), unquote(parts.path)))
        if local in {"assets/data/insights.json", "source-manifest.json"}:
            return parts.path + suffix
        if local in targets and not path.exists():
            return parts.path + suffix
        if not path.exists():
            raise ValueError(f"missing documentation link: {source.relative_to(ROOT)}: {url}")
        kind = "tree" if path.is_dir() else "blob"
        return f"{REPOSITORY}/{kind}/{revision}/{quote(key, safe='/')}" + suffix

    def render(text, source, destination):
        def prose(value):
            value = re.sub(
                r"\]\(([^\s)]+)\)",
                lambda m: "](" + rewrite_url(m[1], source, destination) + ")",
                value,
            )
            return re.sub(
                r'(href|src)="([^\"]+)"',
                lambda m: m[1] + '="' + rewrite_url(m[2], source, destination) + '"',
                value,
            )
        # Fenced code contains commands, fixtures and URLs, never navigation to rewrite.
        chunks = re.split(r"(```[\s\S]*?```|~~~[\s\S]*?~~~)", text)
        return "".join(prose(chunk) if i % 2 == 0 else chunk for i, chunk in enumerate(chunks))

    for name, destination in pages.items():
        source = ROOT / name
        record(source)
        text = source.read_text().replace("<!-- insights:summary -->", summary_html(insights))
        text = text.replace("<!-- insights:modalities -->", modality_cards_html(insights))
        text = text.replace("<!-- insights:catalog -->", fallback_markdown(insights))
        content = render(text, source, destination)
        content += f"\n\n---\n\n[Edit this page]({REPOSITORY}/edit/main/{quote(name, safe='/')})\n"
        output = GENERATED / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)

    # The README remains the owner of the public all-modality performance selection.
    source = ROOT / "README.md"
    record(source)
    readme = source.read_text()
    performance = readme[readme.index("<!-- model-performance-summary:start -->"):
                         readme.index("## Supported Modalities and Models")]
    (GENERATED / "performance/index.md").write_text(
        "# Performance across all modalities\n\n"
        "Recorded measurements retain their original hardware, precision and workload. "
        "The sections below come directly from the project README; the source data and "
        "detailed reports remain authoritative.\n\n"
        + render(performance, source, "performance/index.md")
    )
    citation = ROOT / "CITATION.bib"
    record(citation)
    record(ROOT / "LICENSE")
    (GENERATED / "reference/citation.md").write_text(
        "# Citation and license\n\nIf you use MLX-One in your work, please cite it as:\n\n"
        "```bibtex\n" + citation.read_text().strip() + "\n```\n\n"
        f"[Download CITATION.bib]({REPOSITORY}/blob/{revision}/CITATION.bib)\n\n"
        "## License\n\nMLX-One is released under the "
        f"[Apache License 2.0]({REPOSITORY}/blob/{revision}/LICENSE). "
        "Model checkpoints retain their separate terms.\n"
    )
    styles = GENERATED / "assets/stylesheets"
    styles.mkdir(parents=True)
    shutil.copyfile(HERE / "assets/stylesheets/extra.css", styles / "extra.css")
    record(HERE / "assets/stylesheets/extra.css")
    scripts = GENERATED / "assets/javascripts"
    scripts.mkdir(parents=True)
    script = HERE / "assets/javascripts/insights.js"
    shutil.copyfile(script, scripts / script.name)
    record(script)
    data = GENERATED / "assets/data"
    data.mkdir(parents=True)
    (data / "insights.json").write_text(json.dumps(insights, indent=2, allow_nan=False) + "\n")
    brand = GENERATED / "assets/brand"
    brand.mkdir(parents=True)
    logo = ROOT / "ui/static/mlx-one-logo.png"
    shutil.copyfile(logo, brand / "mark.png")
    record(logo)
    manifest = {"source_commit": revision, "source_sha256": dict(sorted(sources.items())),
                "pages": dict(sorted(pages.items())), "generated_pages": len(targets)}
    (GENERATED / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Staged {len(targets)} pages and {len(assets)} repository images from original sources.")
    return manifest


if __name__ == "__main__":
    stage()
