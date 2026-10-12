"""Record pinned audio assets and safe tensor headers without loading a backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from safetensors import safe_open

from mlx_one.audio.weights import select_safetensors, validate_tensor_index


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract(
    root: Path,
    model_id: str,
    revision: str,
    license_id: str,
    *,
    reference_source: str = "https://github.com/huggingface/transformers",
    reference_commit: str = "cc832f9055ba11c8c55f918ab4bda9472b910d48",
    supplemental_assets: tuple[Path, ...] = (),
    required_assets: tuple[str, ...] = (),
    weight_layout: str = "auto",
) -> dict:
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("checkpoint revision must be a full commit SHA")
    if not re.fullmatch("[0-9a-f]{40}", reference_commit):
        raise ValueError("reference revision must be a full commit SHA")
    for name in required_assets:
        relative = PurePosixPath(name)
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or ":" in name
            or not (root / name).is_file()
        ):
            raise ValueError(f"required checkpoint asset is invalid or missing: {name}")
    files, mapping = select_safetensors(root, layout=weight_layout)
    names, tensors = {}, {}
    for path in files:
        with safe_open(path, framework="numpy") as handle:
            names[path.name] = tuple(handle.keys())
            for name in handle.keys():
                value = handle.get_slice(name)
                tensors[name] = {
                    "shape": value.get_shape(),
                    "dtype": value.get_dtype(),
                    "file": path.name,
                }
    validate_tensor_index(files, mapping, names)
    assets = [
        root / name
        for name in (
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors.index.json",
            "README.md",
            "chat_template.json",
            "processor_config.json",
            "LICENSE",
        )
        if (root / name).is_file()
    ]
    assets.extend(root / name for name in required_assets if root / name not in assets)
    return {
        "model": model_id,
        "revision": revision,
        "checkpoint_license": license_id,
        "reference_source": reference_source,
        "reference_commit": reference_commit,
        "weight_layout": weight_layout,
        "config": json.loads((root / "config.json").read_text()),
        "preprocessing": (
            json.loads((root / "preprocessor_config.json").read_text())
            if (root / "preprocessor_config.json").is_file()
            else None
        ),
        "supplemental_assets": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in supplemental_assets
        },
        "assets": {
            path.relative_to(root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (*files, *assets)
        },
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--reference-source", default="https://github.com/huggingface/transformers")
    parser.add_argument("--reference-commit", default="cc832f9055ba11c8c55f918ab4bda9472b910d48")
    parser.add_argument("--supplemental-asset", type=Path, action="append", default=[])
    parser.add_argument("--required-asset", action="append", default=[])
    parser.add_argument("--weight-layout", choices=("auto", "canonical", "indexed"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    result = contract(
        args.directory,
        args.model,
        args.revision,
        args.license,
        reference_source=args.reference_source,
        reference_commit=args.reference_commit,
        supplemental_assets=tuple(args.supplemental_asset),
        required_assets=tuple(args.required_asset),
        weight_layout=args.weight_layout,
    )
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"output": str(args.output), "tensors": len(result["tensors"])}))


if __name__ == "__main__":
    main()
