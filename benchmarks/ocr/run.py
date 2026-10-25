"""One checkpoint per process, fresh load, one warmup, five measured native OCR requests."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path

from benchmarks.ocr.metrics import auxiliary_metrics, text_metrics, validate


def hash_files(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def measure(bundle, request):
    import threading

    from mlx_one.ocr.documents import iter_pages
    from mlx_one.ocr.generation import stream_page
    from mlx_one.ocr.schemas import OCRResult

    pages = []
    event = threading.Event()
    deadline = time.monotonic() + request.limits.timeout_seconds
    iterator = iter_pages(request, cancel=event, deadline=deadline)
    try:
        for page in iterator:
            stream = stream_page(
                bundle,
                page.image,
                request,
                page=page.index,
                page_metadata=page.metadata,
                cancelled=lambda: time.monotonic() >= deadline,
            )
            try:
                while True:
                    try:
                        next(stream)
                    except StopIteration as stopped:
                        pages.append(stopped.value)
                        break
            finally:
                stream.close()
    finally:
        iterator.close()
    return OCRResult(
        "\n\n".join(p.text for p in pages),
        None,
        pages,
        bundle.model_id,
        request.task,
        None,
        {},
        {"measurement_type": "direct-model"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, help="Original Hugging Face checkpoint ID")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("docs/benchmarks/ocr/fixtures/manifest.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="ocr.recognize")
    parser.add_argument("--format", choices=["text", "markdown"], default="text")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument(
        "--measurement-type", choices=["direct-model", "python-engine"], default="direct-model"
    )
    args = parser.parse_args()
    import mlx.core as mx

    from mlx_one.ocr import OCREngine, OCRInput
    from mlx_one.ocr.loading import load_ocr_model

    manifest = json.loads(args.manifest.read_text())
    fixture = next((f for f in manifest["fixtures"] if f["id"] == args.workload), None)
    if fixture is None:
        parser.error("workload is absent; provide actual fixture and transcript first")
    path = Path(fixture["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != fixture["sha256"]:
        parser.error("workload fixture hash mismatch")
    request = OCRInput(
        path, task=args.task, output_format=args.format, max_tokens=args.max_tokens, dpi=args.dpi
    )
    # Request incompatibility must fail before allocating or warming a model.
    if args.model.name != args.revision:
        for asset in list(args.model.glob("*.safetensors")) + [args.model / "config.json"]:
            meta = args.model / ".cache/huggingface/download" / (asset.name + ".metadata")
            if not meta.is_file() or meta.read_text().splitlines()[0] != args.revision:
                parser.error(
                    "benchmark revision must match hf download metadata "
                    "for every weight/config file"
                )
    loaded = time.perf_counter()
    bundle = load_ocr_model(args.model, revision=args.revision, offline=True)
    load_seconds = time.perf_counter() - loaded
    cap = bundle.capabilities
    if args.task not in cap.tasks or args.format == "markdown" and not cap.markdown:
        parser.error("unsupported checkpoint task/format")
    if cap.line_only and args.workload not in {"OCR-S1", "OCR-S8", "OCR-S9-LINE"}:
        parser.error("TrOCR benchmark requires a cropped line workload")
    engine = OCREngine(bundle) if args.measurement_type == "python-engine" else None

    def infer():
        return engine.recognize(request) if engine else measure(bundle, request)

    samples = []
    try:
        infer()
        mx.synchronize()
        for _ in range(5):
            mx.synchronize()
            mx.reset_peak_memory()
            started = time.perf_counter()
            result = infer()
            mx.synchronize()
            elapsed = time.perf_counter() - started
            samples.append(
                {
                    "latency_seconds": elapsed,
                    "pages": len(result.pages),
                    "peak_mlx_memory_gib": mx.get_peak_memory() / 1024**3,
                    "peak_process_memory_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024**3,
                    "output_tokens": sum(p.metadata.get("output_tokens", 0) for p in result.pages),
                    "page_details": [
                        {"timings": p.timings, "metadata": p.metadata} for p in result.pages
                    ],
                    "transcript": result.text,
                    "accuracy": text_metrics(fixture["reference"], result.text),
                    "quality": auxiliary_metrics(
                        fixture,
                        result,
                        geometry=cap.geometry and request.task in {"ocr.detect", "ocr.region"},
                        structured_table=request.task == "ocr.table",
                        formula=request.task == "ocr.formula",
                    ),
                }
            )
    finally:
        if engine:
            engine.close()
    from mlx.utils import tree_flatten

    actual_dtypes = sorted(
        {
            str(value.dtype).rsplit(".", 1)[-1]
            for _, value in tree_flatten(bundle.model.parameters())
        }
    )
    info = mx.device_info()
    memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    record = {
        "schema_version": "mlx-one-ocr-benchmark-v1",
        "status": "MEASURED",
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "precision": bundle.quantization or {"mode": "floating", "dtypes": actual_dtypes},
        "task": args.task,
        "workload": args.workload,
        "fixture_kind": fixture["kind"],
        "reference": fixture["reference"],
        "fixture_sha256": fixture["sha256"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "measurement_type": args.measurement_type,
        "hardware": {"chip": info.get("device_name", "Apple Silicon"), "memory_bytes": memory},
        "software": {
            "mlx": importlib.metadata.version("mlx"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "protocol": {
            "batch_size": 1,
            "warmups": 1,
            "repetitions": 5,
            "statistic": "median",
            "dpi": args.dpi,
            "pdf_policy": "visual",
            "max_tokens": args.max_tokens,
            "output_format": args.format,
            "backend": "native MLX/Metal",
            "timing_scope": (
                "document decode/render, model preprocessing, encode, prefill and decode"
            ),
        },
        "load_seconds": load_seconds,
        "samples": samples,
        "workload_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "checkpoint_sha256": hash_files(
            list(args.model.glob("*.safetensors")) + list(args.model.glob("*.json"))
        ),
        "implementation_sha256": hash_files(list(Path("src/mlx_one").rglob("*.py"))),
        "runner_sha256": hash_files(list(Path("benchmarks/ocr").glob("*.py"))),
        "metrics": {
            "latency_seconds": statistics.median(s["latency_seconds"] for s in samples),
            "pages_per_second": statistics.median(
                s["pages"] / s["latency_seconds"] for s in samples
            ),
            "cer": statistics.median(s["accuracy"]["normalized"]["cer"] for s in samples),
            "peak_mlx_memory_gib": statistics.median(s["peak_mlx_memory_gib"] for s in samples),
        },
    }
    validate(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
