"""Persist the pinned Nemotron oracle in a separate validation environment.

No native inference or qualification claim: these outputs precede the MLX port.
Initial capture covers offline English with explicit or automatic language ID;
streaming, other language subsets and native comparisons remain WIP.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import subprocess
import time
from pathlib import Path

from audio_qualification import BUDGET, digest, fixtures

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
REVISION = "ea30d66debe3740a08b573244286791d423d6b3e"
SOURCE = "469230357aab0f2b303b0d638c1f8d06edb14184"


def capture(args) -> dict:
    manifest, records = fixtures(args.manifest, args.limit)
    contract = json.loads(args.contract.read_text())
    if (
        contract["model"] != MODEL
        or contract["revision"] != REVISION
        or contract["reference_commit"] != SOURCE
        or contract["weight_layout"] != "canonical"
    ):
        raise ValueError("Nemotron checkpoint contract differs from frozen target")
    for name, entry in contract["assets"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name or ":" in name:
            raise ValueError("unsafe contract asset path")
        if digest(args.directory / name) != entry["sha256"]:
            raise ValueError(f"checkpoint checksum differs: {name}")
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoProcessor, Nemotron3_5AsrForRNNT

    checkout = Path(transformers.__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=checkout, check=True)
    if head != SOURCE or importlib.metadata.version("torch") != "2.7.1":
        raise ValueError("Nemotron reference source/dependencies differ from frozen target")
    torch.set_num_threads(4)
    started = time.perf_counter()
    model, loading = Nemotron3_5AsrForRNNT.from_pretrained(
        str(args.directory),
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"reference checkpoint loading is not strict: {loading}")
    model.eval()
    processor = AutoProcessor.from_pretrained(str(args.directory), local_files_only=True)
    if processor.default_num_lookahead_tokens != 3:
        raise ValueError("initial Nemotron reference must retain three-token lookahead")
    loaded = time.perf_counter()
    results = []
    for record in records:
        waveform, rate = sf.read(args.manifest.parent / record["audio"], dtype="float32")
        if rate != 16000 or waveform.ndim != 1:
            raise ValueError("fixed reference clips require mono 16-kHz input")
        before = time.perf_counter()
        inputs = processor(
            waveform, sampling_rate=rate, language=args.language, return_tensors="pt"
        )
        if inputs["num_lookahead_tokens"] != 3:
            raise ValueError("processor lookahead differs from model contract")
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, num_beams=1)
        result = {
            "id": record["id"],
            "anchor": record["anchor"],
            "reference": record["reference"],
            "prediction": processor.batch_decode(output.sequences, skip_special_tokens=True)[0],
            "tokens": output.sequences[0].tolist(),
            "duration": len(waveform) / rate,
            "seconds": time.perf_counter() - before,
        }
        results.append(result)
        print(json.dumps({"stage": "clip", **result}, ensure_ascii=False), flush=True)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET:
            raise RuntimeError("reference exceeded the 24 GiB process budget")
        del output, inputs, waveform
    return {
        "status": "REFERENCE_CAPTURED_NOT_QUALIFIED",
        "model": MODEL,
        "revision": REVISION,
        "reference_source_commit": SOURCE,
        "backend": "reference",
        "dtype": "float32",
        "mode": "offline",
        "language_prompt": args.language,
        "num_lookahead_tokens": 3,
        "manifest_sha256": digest(args.manifest),
        "contract_sha256": digest(args.contract),
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "predictions": results,
        "load_seconds": loaded - started,
        "peak_process_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "scope": "reference outputs only; native inference/comparisons remain WIP",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--language", choices=("en-US", "auto"), default="en-US")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    result = capture(args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(args.output), "status": result["status"]}))


if __name__ == "__main__":
    main()
