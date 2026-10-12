"""Isolate the aligner classifier using frozen reference hidden states.

This is not a checkpoint qualification or a production reference fallback.
Reference and native operators run separately; weights/activations remain in a
temporary NPZ, never in repository qualification results.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from audio_qualification import digest
from qwen_audio_layer_probe import MODEL, validate_reference
from qwen_audio_probe import score_decision
from qwen_audio_qualification import PINS, SOURCE


def validate_operator_metadata(metadata, fixture):
    if (
        metadata.get("model") != MODEL
        or metadata.get("revision") != PINS[MODEL]
        or metadata.get("reference_source_commit") != SOURCE
        or metadata.get("fixture_sha256") != digest(fixture)
    ):
        raise ValueError("classifier fixture differs from frozen contract")


def reference(args):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    metadata = json.loads(args.reference.read_text())
    validate_reference(metadata, args.fixture)
    if importlib.metadata.version("torch") != "2.7.1":
        raise ValueError("classifier reference requires pinned torch==2.7.1")
    torch.set_num_threads(1)
    root = Path(snapshot_download(MODEL, revision=PINS[MODEL], local_files_only=True))
    # Read only the classifier tensor, not another copy of the full model.
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor("thinker.lm_head.weight")
    with np.load(args.fixture, allow_pickle=False) as fixture:
        hidden = torch.from_numpy(fixture["model.norm.output"]).to(torch.bfloat16)
        positions = np.flatnonzero(fixture["input_ids"][0] == 151705)
    if weight.shape != (5000, 1024) or weight.dtype != torch.bfloat16:
        raise ValueError("classifier differs from frozen tensor shape/dtype")
    with torch.inference_mode():
        logits = torch.nn.functional.linear(hidden, weight)[0, positions].float().numpy()
    with args.operator_fixture.open("xb") as stream:
        np.savez(
            stream,
            hidden=hidden.float().numpy(),
            weight=weight.float().numpy(),
            positions=positions,
            logits=logits,
        )
    return {
        "scope": "isolated BF16 timestamp classifier, not qualification",
        "model": MODEL,
        "revision": PINS[MODEL],
        "reference_source_commit": SOURCE,
        "input_fixture_sha256": digest(args.fixture),
        "fixture_sha256": digest(args.operator_fixture),
        "reference_sha256": digest(args.reference),
        "torch": importlib.metadata.version("torch"),
        "decisions": [score_decision(row) for row in logits],
    }


def native(args):
    metadata = json.loads(args.reference.read_text())
    validate_operator_metadata(metadata, args.operator_fixture)
    import mlx.core as mx

    with np.load(args.operator_fixture, allow_pickle=False) as fixture:
        hidden = mx.array(fixture["hidden"]).astype(mx.bfloat16)
        weight = mx.array(fixture["weight"]).astype(mx.bfloat16)
        positions = mx.array(fixture["positions"])
        logits = mx.take((hidden @ weight.T)[0], positions, axis=0)
        actual = np.array(logits.astype(mx.float32))
        expected = fixture["logits"]
    native_decisions = [score_decision(row) for row in actual]
    reference_decisions = [score_decision(row) for row in expected]
    return {
        "scope": metadata["scope"],
        "model": MODEL,
        "revision": PINS[MODEL],
        "reference_sha256": digest(args.reference),
        "fixture_sha256": digest(args.operator_fixture),
        "different_values": int(np.count_nonzero(actual != expected)),
        "maximum_logit_difference": float(np.max(np.abs(actual - expected))),
        "class_mismatches": [
            i
            for i, (a, b) in enumerate(zip(reference_decisions, native_decisions, strict=True))
            if a["argmax"] != b["argmax"]
        ],
        "reference_decisions": reference_decisions,
        "native_decisions": native_decisions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "native"), required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--operator-fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.backend == "reference" and args.operator_fixture.exists()):
        parser.error("output targets must not exist")
    if args.backend == "reference" and args.fixture is None:
        parser.error("reference requires --fixture")
    report = reference(args) if args.backend == "reference" else native(args)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {"output": str(args.output), "class_mismatches": report.get("class_mismatches")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
