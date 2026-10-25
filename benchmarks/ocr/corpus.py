"""Reproducible, resumable 100-page OmniAI corpus benchmark (separate from OCR-S*)."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import resource
import statistics
import subprocess
import time
from pathlib import Path

from benchmarks.ocr.metrics import normalize
from benchmarks.ocr.run import hash_files, measure

DATASET = "getomni-ai/ocr-benchmark"
REVISION = "4ed0d95271ca00107726230f7a0944ed9e90d897"
SEED = 20261025
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/benchmarks/ocr/omni100-20261025"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(work):
    import os
    from concurrent.futures import ThreadPoolExecutor

    os.environ["HF_XET_CACHE"] = str(work / "xet-cache")
    os.environ["HF_HUB_CACHE"] = str(work / "hub-cache")
    from huggingface_hub import hf_hub_download
    from PIL import Image

    def download(name):
        return Path(
            hf_hub_download(
                DATASET, name, repo_type="dataset", revision=REVISION, local_dir=work / "dataset"
            )
        )

    metadata = download("test/metadata.jsonl")
    population = sorted(
        [json.loads(s) for s in metadata.read_text().splitlines()], key=lambda r: int(r["id"])
    )
    if len(population) != 1000 or len({r["id"] for r in population}) != 1000:
        raise ValueError("unexpected pinned dataset population")
    selected = random.Random(SEED).sample(population, 100)

    def fixture(row):
        name = "test/" + row["file_name"]
        path = download(name)
        reference = json.loads(row["true_markdown_output"])
        if not isinstance(reference, str):
            raise ValueError("reference must decode to a string")
        # This dataset JSON-encodes Markdown with literal backslash-n separators.
        # Decode those explicitly, leaving Unicode and other backslashes untouched.
        reference = reference.replace("\\n", "\n").replace("\\t", "\t")
        with Image.open(path) as image:
            size = list(image.size)
        return {
            "id": str(row["id"]),
            "path": name,
            "sha256": sha(path),
            "size": size,
            "reference": reference,
            "source_reference": row["true_markdown_output"],
            "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
            "metadata": json.loads(row["metadata"]),
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        fixtures = list(pool.map(fixture, selected))
    manifest = {
        "schema_version": "mlx-one-ocr-omni100-sample-v1",
        "dataset": DATASET,
        "revision": REVISION,
        "split": "test",
        "license": "mit (dataset card)",
        "population": 1000,
        "seed": SEED,
        "sampling": "random.Random(seed).sample(numeric-id-sorted population, 100)",
        "reference_decoding": "JSON string decode; literal backslash n/t to newline/tab",
        "fixtures": fixtures,
        "metadata_sha256": sha(metadata),
    }
    write(EVIDENCE / "sample.json", manifest)
    print("Pinned 100 unique images:", sha(EVIDENCE / "sample.json"), flush=True)


def accuracy(reference, prediction):
    from rapidfuzz.distance import Levenshtein

    r, p = normalize(reference), normalize(prediction)
    words = r.split()
    return {
        "character_edits": Levenshtein.distance(r, p),
        "reference_characters": len(r),
        "word_edits": Levenshtein.distance(words, p.split()),
        "reference_words": len(words),
    }


def summarize(samples):
    success = [s for s in samples if s["status"] == "OK"]
    scored = [s["accuracy"] for s in samples]
    chars = sum(s["reference_characters"] for s in scored)
    words = sum(s["reference_words"] for s in scored)
    elapsed = sum(s["latency_seconds"] for s in samples)
    return {
        "attempted": len(samples),
        "completed": len(success),
        "failed": len(samples) - len(success),
        "truncated": sum(s.get("truncated", False) for s in success),
        "median_latency_seconds": statistics.median(s["latency_seconds"] for s in success)
        if success
        else None,
        "completed_pages_per_second": len(success) / elapsed if elapsed else None,
        "corpus_cer": sum(s["character_edits"] for s in scored) / chars if chars else None,
        "corpus_wer": sum(s["word_edits"] for s in scored) / words if words else None,
        "max_peak_mlx_memory_gib": max((s["peak_mlx_memory_gib"] for s in success), default=None),
        "actual_output_tokens": sum(s.get("output_tokens", 0) for s in success),
        "inference_seconds": elapsed,
    }


def run(args):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx_one.ocr.loading import load_ocr_model
    from mlx_one.ocr.schemas import OCRInput, OCRLimits

    inventory = json.loads((EVIDENCE.parent / "inventory.json").read_text())["checkpoints"]
    entry = next(e for e in inventory if e["id"] == args.id)
    hand = args.cohort == "hand10"
    evidence = EVIDENCE.parent / "hand10-20261025" if hand else EVIDENCE
    budget = 128 if hand else 1023
    manifest = json.loads((evidence / "sample.json").read_text())
    if entry["family"] == "trocr" and not hand:
        raise ValueError("page corpus has no annotated line crops; TrOCR is incompatible")
    model = args.work / "models" / entry["id"]
    for asset in list(model.glob("*.safetensors")) + [model / "config.json"]:
        meta = model / ".cache/huggingface/download" / (asset.name + ".metadata")
        if not meta.is_file() or meta.read_text().splitlines()[0] != entry["revision"]:
            raise ValueError("checkpoint revision mismatch: " + asset.name)
    record = {
        "schema_version": "mlx-one-ocr-hand10-v1" if hand else "mlx-one-ocr-omni100-v1",
        "status": "RUNNING",
        "checkpoint": entry["checkpoint"],
        "revision": entry["revision"],
        "id": entry["id"],
        "name": entry["name"],
        "precision_label": entry["precision"],
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "sample_sha256": sha(evidence / "sample.json"),
        "hardware": {
            "chip": mx.device_info()["device_name"],
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
        },
        "software": {
            "mlx": importlib.metadata.version("mlx"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "rapidfuzz": importlib.metadata.version("rapidfuzz"),
        },
        "measurement_type": "direct-model",
        "task": entry["primary_task"],
        "protocol": {
            "batch_size": 1,
            "warmups": 1,
            "warmup_max_tokens": 16,
            "repetitions_per_image": 1,
            "images": len(manifest["fixtures"]),
            "max_tokens": budget,
            "timeout_seconds": 180,
            "temperature": 0,
            "output_format": "text",
            "backend": "native MLX/Metal",
            "timing_scope": (
                "image open/decode, model preprocessing, encode, prefill and decode; "
                "excludes scoring and load"
            ),
            "accuracy": (
                "micro CER/WER on NFKC/whitespace normalized "
                + ("plain-text" if hand else "Markdown")
                + " references; failures count as empty output"
            ),
        },
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_sha256": hash_files(list((ROOT / "src/mlx_one").rglob("*.py"))),
        "runner_sha256": hash_files(list((ROOT / "benchmarks/ocr").glob("*.py"))),
        "samples": [],
        "checkpoint_files": json.loads((EVIDENCE / (args.id + "-assets.json")).read_text()),
    }
    output = evidence / (args.id + ".json")
    write(output, record)
    try:
        started = time.perf_counter()
        bundle = load_ocr_model(model, revision=entry["revision"], offline=True)
        mx.synchronize()
        record["load_seconds"] = time.perf_counter() - started
        record["precision"] = bundle.quantization or {
            "mode": "floating",
            "dtypes": sorted(
                {
                    str(v.dtype).rsplit(".", 1)[-1]
                    for _, v in tree_flatten(bundle.model.parameters())
                }
            ),
        }

        def request(fixture, max_tokens=None):
            max_tokens = budget if max_tokens is None else max_tokens
            image = args.work / ("handwriting" if hand else "dataset") / fixture["path"]
            if sha(image) != fixture["sha256"]:
                raise ValueError("fixture hash mismatch")
            return OCRInput(
                image,
                task=entry["primary_task"],
                max_tokens=max_tokens,
                limits=OCRLimits(timeout_seconds=180),
            )

        warm = time.perf_counter()
        measure(bundle, request(manifest["fixtures"][0], 16))
        mx.synchronize()
        record["warmup_seconds"] = time.perf_counter() - warm
        for fixture in manifest["fixtures"]:
            req = request(fixture)
            mx.synchronize()
            mx.reset_peak_memory()
            began = time.perf_counter()
            sample = {"id": fixture["id"], "image_sha256": fixture["sha256"]}
            try:
                result = measure(bundle, req)
                mx.synchronize()
                elapsed = time.perf_counter() - began
                sample.update(
                    status="OK",
                    latency_seconds=elapsed,
                    transcript=result.text,
                    output_tokens=sum(p.metadata.get("output_tokens", 0) for p in result.pages),
                    truncated=any(p.metadata.get("truncated", False) for p in result.pages),
                    pages=[{"timings": p.timings, "metadata": p.metadata} for p in result.pages],
                    peak_mlx_memory_gib=mx.get_peak_memory() / 1024**3,
                    peak_process_memory_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024**3,
                )
            except Exception as exc:
                mx.synchronize()
                sample.update(
                    status="FAILED",
                    latency_seconds=time.perf_counter() - began,
                    transcript="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            sample["accuracy"] = accuracy(fixture["reference"], sample["transcript"])
            record["samples"].append(sample)
            record["metrics"] = summarize(record["samples"])
            write(output, record)
            print(
                entry["id"],
                len(record["samples"]),
                fixture["id"],
                sample["status"],
                round(sample["latency_seconds"], 2),
                sample.get("output_tokens"),
                flush=True,
            )
            mx.clear_cache()
        record["status"] = "MEASURED" if record["metrics"]["completed"] else "FAILED"
    except Exception as exc:
        import traceback

        traceback.print_exc()
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    write(output, record)
    if record["status"] == "FAILED":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument(
        "--work", type=Path, required=True, help="Dedicated task-owned scratch directory"
    )
    parser.add_argument("--cohort", choices=["omni100", "hand10"], default="omni100")
    parser.add_argument("--id", choices=[f"O{i:02}" for i in range(1, 8)])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.work)
    elif args.id:
        result = 0
        try:
            run(args)
        except SystemExit as exc:
            result = exc.code
        if (
            args.cohort == "omni100"
            and (args.work / "handwriting").is_dir()
            and (EVIDENCE.parent / "hand10-20261025/sample.json").is_file()
        ):
            import gc

            gc.collect()
            args.cohort = "hand10"
            run(args)
        if result:
            raise SystemExit(result)
    else:
        parser.error("run needs --id")


if __name__ == "__main__":
    main()
