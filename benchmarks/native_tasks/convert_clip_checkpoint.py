#!/usr/bin/env python3
"""Offline audited conversion of the exact published CLIP artifact to safetensors.

Run only in the reference environment. The original artifact is verified against
its pinned Hub LFS SHA-256 before torch.load(weights_only=True). Production never
loads pickle or imports this tool. Every tensor is round-trip checked.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save

EXPECTED_HASH = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
EXPECTED_SIZE = 605247071
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    original = args.source / "pytorch_model.bin"
    if original.stat().st_size != EXPECTED_SIZE or digest(original) != EXPECTED_HASH:
        raise ValueError("original CLIP artifact size/hash does not match the pinned Hub blob")
    args.destination.mkdir(parents=True, exist_ok=False)
    state = torch.load(original, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("expected a tensor-only state dictionary")
    tensors = {k: v.detach().contiguous().clone() for k, v in state.items()}
    output = args.destination / "model.safetensors"
    with output.open("xb") as f:
        f.write(save(tensors))
    with safe_open(output, framework="pt", device="cpu") as restored:
        if set(restored.keys()) != set(state):
            raise ValueError("round-trip tensor inventory mismatch")
        if not all(torch.equal(restored.get_tensor(k), v) for k, v in state.items()):
            raise ValueError("round-trip tensor value mismatch")
    for name in [
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "README.md",
    ]:
        shutil.copyfile(args.source / name, args.destination / name)
    report = {
        "model_id": "openai/clip-vit-base-patch32",
        "source_revision": REVISION,
        "source_sha256": EXPECTED_HASH,
        "source_size": EXPECTED_SIZE,
        "conversion": "torch.load(weights_only=True,map_location=cpu); no dtype/name changes",
        "torch_version": torch.__version__,
        "round_trip_exact": True,
        "files": {x.name: digest(x) for x in args.destination.iterdir() if x.is_file()},
        "tensor_inventory": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in state.items()
        },
    }
    with (args.destination / "conversion.json").open("x") as f:
        json.dump(report, f, indent=2)
    print("converted tensors", len(state), "sha256", report["files"]["model.safetensors"])


if __name__ == "__main__":
    main()
