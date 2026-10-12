"""Freeze original Microsoft TTS source and its separately supplied text tokenizer.

This inspects source/assets only. It never imports upstream inference code or
declares publisher voice files consented merely because they are public/licensed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

from audio_checkpoint_contract import sha256

SOURCE = "6065c5224e1fee5b2eaeef15f79ee8ef7de947d0"
TREE = "fb03d34019e571e4be18298fa2b5437b5f5a4e01"
TOKENIZER = "8faed761d45a263340a0528343f099c05c9a4323"


def source_contract(source: Path, tokenizer: Path, publisher_commit: dict) -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=source, check=True)
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=source, text=True
    ).strip()
    if head != SOURCE or tree != TREE:
        raise ValueError("source differs from the frozen pre-removal TTS commit/tree")
    if (
        publisher_commit.get("sha") != SOURCE
        or publisher_commit.get("commit", {}).get("tree", {}).get("sha") != TREE
        or publisher_commit.get("html_url")
        != f"https://github.com/microsoft/VibeVoice/commit/{SOURCE}"
    ):
        raise ValueError("original publisher commit identity differs from preserved source")
    if tokenizer.name != TOKENIZER:
        raise ValueError("supplemental tokenizer directory must be the frozen Hub snapshot")
    relative = ["README.md", "LICENSE", "pyproject.toml"]
    relative += sorted(
        p.relative_to(source).as_posix() for p in (source / "vibevoice").rglob("*.py")
    )
    assets = {
        name: {"bytes": (source / name).stat().st_size, "sha256": sha256(source / name)}
        for name in relative
    }
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "LICENSE",
    )
    tokenizer_assets = {
        name: {"bytes": (tokenizer / name).stat().st_size, "sha256": sha256(tokenizer / name)}
        for name in tokenizer_files
    }
    from tokenizers import Tokenizer

    backend = Tokenizer.from_file(str(tokenizer / "tokenizer.json"))
    specials = {
        s: backend.token_to_id(s)
        for s in (
            "<|vision_start|>",
            "<|vision_end|>",
            "<|vision_pad|>",
            "<|image_pad|>",
            "<|endoftext|>",
        )
    }
    if any(value is None for value in specials.values()) or len(set(specials.values())) != len(
        specials
    ):
        raise ValueError("original processor's speech/padding special IDs are missing or ambiguous")
    config = json.loads((tokenizer / "tokenizer_config.json").read_text())
    voices = {
        p.relative_to(source).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
            "publisher_available": True,
            "recorded_consent": False,
            "qualification_permission_evidence": "WIP — not established by availability or license",
        }
        for p in sorted((source / "demo/voices").glob("*.wav"))
    }
    return {
        "status": "SOURCE_AND_TOKENIZER_PINNED_NOT_QUALIFIED",
        "checkpoint": "microsoft/VibeVoice-1.5B",
        "checkpoint_revision": "c00898d257e6b46004e3e2866a47534085fb685a",
        "reference_source": "https://github.com/microsoft/VibeVoice",
        "reference_commit": SOURCE,
        "reference_tree": TREE,
        "archive_transport": "https://github.com/rsxdalv/VibeVoice/tree/archive",
        "publisher_verification_url": f"https://api.github.com/repos/microsoft/VibeVoice/commits/{SOURCE}",
        "publisher_commit": publisher_commit,
        "source_assets": assets,
        "source_license": "MIT",
        "tokenizer_model": "Qwen/Qwen2.5-1.5B",
        "tokenizer_revision": TOKENIZER,
        "tokenizer_assets": tokenizer_assets,
        "speech_special_ids": specials,
        "tokenizer_eos": config.get("eos_token"),
        "publisher_voice_files": voices,
        "qualification": {
            "production_inference": "WIP / not implemented",
            "reference_dependency_lock": "WIP — isolated environment not yet frozen",
            "processor_parity": "WIP — assets/IDs inspected, original processor not executed",
            "voice_permission_and_intended_use": "WIP — affected synthesis fixtures not qualified",
            "reference_evaluator": "frozen isolated Qwen3-ASR 1.7B reference; not native fallback",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    request = urllib.request.Request(
        f"https://api.github.com/repos/microsoft/VibeVoice/commits/{SOURCE}",
        headers={"User-Agent": "mlx-one-qualification", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.load(response)
    # Persist identity, provenance and verification without a large file-diff payload.
    publisher = {key: metadata[key] for key in ("sha", "html_url", "commit", "parents")}
    result = source_contract(args.source, args.tokenizer, publisher)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({"output": str(args.output), "status": result["status"]}))


if __name__ == "__main__":
    main()
