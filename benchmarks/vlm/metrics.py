"""Strict evidence validation; source-test numbers cannot enter public VLM tables."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/benchmarks/vlm"
METRICS = (
    "preprocessing_ms",
    "visual_prefill_ms",
    "prefill_tps",
    "decode_tps",
    "ttft_ms",
    "total_ms",
    "peak_mlx_memory_gib",
)


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(record, plan=None):
    plan = plan or read(EVIDENCE / "benchmark_plan.json")
    require(record.get("task") == "vision_language", "wrong measurement task")
    require(set(record.get("metrics", {})) == set(METRICS), "incomplete metrics schema")
    models = {m["checkpoint"]: m for m in plan["checkpoints"]}
    require(record["checkpoint"] in models, "checkpoint is not inventoried")
    model = models[record["checkpoint"]]
    require(record["revision"] == model["revision"], "unrecognized checkpoint revision")
    require(record["workload"] in plan["workloads"], "unrecognized workload")
    require(record["protocol"] == record["workload"].lower() + "-v1", "protocol mismatch")
    if record["status"] == "NOT_RUN":
        require(
            not record.get("samples") and all(v is None for v in record["metrics"].values()),
            "unexecuted records cannot contain measurements",
        )
        return record
    require(
        record["status"] == "MEASURED" and record.get("synthetic") is False,
        "only actual checkpoint measurements are publishable",
    )
    require(
        record["measurement_type"] == "direct-model" and record["backend"] == "native-mlx-metal",
        "direct inference must remain separate from Engine/CLI/HTTP",
    )
    hardware = record["hardware"]
    require(
        all(hardware.get(k) for k in ("device", "memory_bytes", "os", "machine")),
        "hardware provenance missing",
    )
    require(
        record.get("mlx_version") and record.get("python_version"), "runtime provenance missing"
    )
    require(
        record["batch_size"] == 1
        and record["warmups"] == 1
        and record["repetitions"] == 5
        and record["statistic"] == "median",
        "invalid publication protocol",
    )
    require(
        record["precision"] == model["precision"]
        and record["architecture"] == model["architecture"],
        "model format mismatch",
    )
    require(
        record["weight_sha256"] == model["weight_sha256"],
        "weight content does not match pinned checkpoint",
    )
    if model.get("tokenizer_sha256"):
        require(
            record.get("tokenizer_sha256") == model["tokenizer_sha256"],
            "tokenizer content differs from pinned publisher file",
        )
    require(
        record.get("asset_git_blob") == model["asset_git_blob"], "tokenizer Git provenance mismatch"
    )
    require(
        re.fullmatch("[0-9a-f]{64}", record.get("tokenizer_sha256", "")),
        "tokenizer checksum missing",
    )
    require(
        record["asset_sha256"] == model["asset_sha256"],
        "processor/config/tokenizer metadata provenance mismatch",
    )
    require(
        re.fullmatch("[0-9a-f]{64}", record.get("implementation_sha256", ""))
        and re.fullmatch("[0-9a-f]{64}", record.get("runner_sha256", "")),
        "implementation provenance missing",
    )
    require(
        type(hardware["memory_bytes"]) is int and hardware["memory_bytes"] > 0,
        "physical memory provenance must be actual positive bytes",
    )
    require(hardware["machine"] == "arm64", "native Apple Silicon hardware required")
    require(
        record.get("generation_options")
        == {
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "seed": 0,
            "stop": [],
        },
        "generation settings mismatch",
    )
    require(
        type(record.get("model_load_ms")) in (int, float) and record["model_load_ms"] > 0,
        "actual model load timing missing",
    )
    from benchmarks.vlm.workloads import workload_metadata

    expected = workload_metadata(record["workload"])
    require(
        record["image_sha256"] == expected["image_sha256"]
        and record["source_sizes"] == expected["source_sizes"],
        "workload images differ from the versioned source",
    )
    if "image_provenance" in expected:
        require(
            record.get("image_provenance") == expected["image_provenance"],
            "pinned HF dataset image provenance mismatch",
        )
    require(
        record["prompt"] == expected["prompt"] and record["max_tokens"] == 128,
        "workload prompt/output budget mismatch",
    )
    require(
        record.get("functional_outcome") == "inference_completed",
        "functional inference did not complete",
    )
    require(len(record["samples"]) == 5, "five actual repetitions required")
    for sample in record["samples"]:
        count = sample["output_tokens"]
        require(
            type(count) is int and 0 <= count <= 128 and count == len(sample["token_ids"]),
            "invalid actual generated count",
        )
        require(
            all(type(t) is int and t >= 0 for t in sample["token_ids"]), "invalid output token IDs"
        )
        require(
            sample["finish_reason"] in {"stop", "length"} and sample["cleanup"] is True,
            "generation/cleanup incomplete",
        )
        require(
            type(sample["prompt_tokens"]) is int and sample["prompt_tokens"] > 0,
            "prompt count missing",
        )
        require(
            len(sample["image_metadata"]) == len(record["source_sizes"]),
            "image processing metadata missing",
        )
        require(sample["text_prompt_tokens"] >= 0, "negative text/framing token count")
        require(
            type(sample.get("peak_process_memory_bytes")) is int
            and sample["peak_process_memory_bytes"] > 0,
            "process memory missing",
        )
        if sample["finish_reason"] == "length":
            require(count == 128, "length termination must use the actual protocol output budget")
        for image, source_size in zip(
            sample["image_metadata"], record["source_sizes"], strict=True
        ):
            require(image["source_size"] == source_size, "source image metadata mismatch")
            require(
                image["visual_tokens"] > 0
                and image["patches"] >= image["visual_tokens"]
                and image["processed_sizes"],
                "visual dimensions/counts missing",
            )
        for key in METRICS:
            value = sample[key]
            if key in {"decode_tps", "ttft_ms"} and count < (2 if key == "decode_tps" else 1):
                require(value is None, "unmeasurable token speed/latency must be null")
            else:
                require(
                    type(value) in (float, int) and math.isfinite(value) and value > 0,
                    "invalid metric " + key,
                )
        require(
            math.isclose(
                sample["prefill_tps"],
                sample["prompt_tokens"] * 1000 / sample["visual_prefill_ms"],
                rel_tol=1e-7,
            ),
            "prefill rate must use actual fused input tokens and synchronized combined prefill",
        )
        if count > 1:
            require(
                sample["decode_ms"] > 0
                and math.isclose(
                    sample["decode_tps"], (count - 1) * 1000 / sample["decode_ms"], rel_tol=1e-7
                ),
                "decode rate must use actual continuation steps",
            )
        require(
            sample["text_prompt_tokens"]
            == sample["prompt_tokens"] - sum(m["visual_tokens"] for m in sample["image_metadata"]),
            "text/visual token accounting mismatch",
        )
        require(
            sample["vision_encoder_ms"] is None
            and sample["vision_projection_ms"] is None
            and sample["language_prefill_ms"] is None,
            "this protocol measures combined visual/language prefill",
        )
    for key in METRICS:
        values = [s[key] for s in record["samples"]]
        actual = statistics.median(values) if all(v is not None for v in values) else None
        require(record["metrics"][key] == actual, "summary is not per-column median: " + key)
    return record
