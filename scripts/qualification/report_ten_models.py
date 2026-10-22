"""Compile immutable campaign outputs; failed profiles are never promoted."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from itertools import product
from pathlib import Path

from qualify_ten_models import manifest


def validate(document, schema, path="$"):
    """Validate the type/required/enum/items/properties subset used by our schema."""
    kinds = {"object": dict, "array": list, "string": str, "null": type(None)}
    expected = schema.get("type")
    allowed = [expected] if isinstance(expected, str) else expected
    if allowed and not any(isinstance(document, kinds[k]) for k in allowed):
        raise ValueError(f"{path}: invalid type")
    if "enum" in schema and document not in schema["enum"]:
        raise ValueError(f"{path}: invalid enum value")
    if isinstance(document, dict):
        if set(schema.get("required", [])) - document.keys():
            raise ValueError(f"{path}: required fields missing")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and document.keys() - properties.keys():
            raise ValueError(f"{path}: unexpected fields")
        for key, value in document.items():
            if key in properties:
                validate(value, properties[key], f"{path}.{key}")
    if isinstance(document, list) and "items" in schema:
        for index, value in enumerate(document):
            validate(value, schema["items"], f"{path}[{index}]")


def metric_median(row, key):
    values = [s[key] for s in row.get("samples", []) if s.get(key) is not None]
    return statistics.median(values) if values else None


def coverage(documents, hardware):
    contexts = (16, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    outputs = (1, 16, 64, 128, 256)
    precisions = ("float32", "float16", "bfloat16", "affine_q4", "affine_q8")
    measured = {}
    for row in hardware:
        precision = row["source_precision"]
        if precision == "stored":
            precision = next(
                (p for p in ("bfloat16", "float16", "float32") if p in row["weight_dtypes"]), None
            )
        key = (row["id"], precision, row["input_tokens"], row["output_budget"], 1)
        if row["status"] == "PASS":
            measured[key] = row["evidence"]
    cells = []
    for document in documents:
        limit = document["checkpoint"].get("context_tokens")
        for precision, context, output, batch in product(precisions, contexts, outputs, (1, 2, 4)):
            key = (document["checkpoint_id"], precision, context, output, batch)
            status, reason = "NOT_RUN", "no completed measurement for this exact cell"
            if batch != 1:
                reason = "batching is not qualified by this B1 cohort"
            if limit is not None and context + output > limit:
                status, reason = "UNSUPPORTED", "checkpoint context limit"
            elif key in measured:
                status, reason = "PASS", "completed hardware request and cleanup only"
            cells.append(
                {
                    "id": key[0],
                    "precision": precision,
                    "input_tokens": context,
                    "output_budget": output,
                    "batch_size": batch,
                    "status": status,
                    "reason": reason,
                    "evidence": measured.get(key),
                }
            )
    return {
        "schema_version": "ten-model-hardware-coverage-v1",
        "qualification": "NOT_QUALIFIED",
        "pass_meaning": "Hardware request/cleanup only; numerical, API and quality gates separate",
        "counts": dict(Counter(cell["status"] for cell in cells)),
        "cells": cells,
        "off_grid_executed": [r for r in hardware if r["input_tokens"] not in contexts],
    }


def compile_results(run, destination):
    if destination.exists():
        raise ValueError("output exists; choose a new report directory")
    destination.mkdir(parents=True)
    models = manifest(Path(__file__).with_name("ten_model_manifest.yaml"))
    schema = json.loads(Path(__file__).with_name("ten_model_result.schema.json").read_text())
    documents, hardware = [], []
    license_path = run / "checkpoint-licenses.json"
    licenses = json.loads(license_path.read_text()) if license_path.is_file() else []
    for model in models:
        files = sorted(run.glob(f"*/{model['id']}.json"))
        profiles, revisions, checkpoint = [], set(), {}
        for path in files:
            row = json.loads(path.read_text())
            if row.get("schema_version") == "ten-model-result-v1":
                continue  # An earlier aggregate is evidence, not another model execution.
            summary = path.parent / "summary.json"
            if summary.is_file():
                supervised = next(
                    (r for r in json.loads(summary.read_text()) if r["id"] == model["id"]), None
                )
                if supervised:
                    row.update(supervised)
            outcome = row.get("supervision", {}).get("status")
            partial = row.get("checks", {}).get("hardware", {})
            if outcome in {"timeout", "memory_ceiling"} and partial.get("status") == "NOT_RUN":
                partial["snapshot_reason"] = partial.get("reason")
                partial["status"] = "BLOCKED_ENVIRONMENT"
                partial["reason"] = f"incomplete: supervisor {outcome}; completed rows retained"
            if row.get("repo") != model["repo"]:
                raise ValueError("exact target identity changed")
            if row.get("revision"):
                revisions.add(row["revision"])
            if row.get("checkpoint"):
                checkpoint = row["checkpoint"]
            precision = dict(row.get("precision_profile", {}))
            public_metrics = (
                row.get("checks", {})
                .get("http", {})
                .get("responses", {})
                .get("completion", {})
                .get("mlx", {})
            )
            if "actual_logits_dtypes" in public_metrics:
                precision["observed_public_logits_dtypes"] = public_metrics["actual_logits_dtypes"]
            numerical = row.get("checks", {}).get("numerical", {})
            if "native_state_dtypes" in numerical:
                precision["observed_direct_model_state_dtypes"] = sorted(
                    {dtype for layer in numerical["native_state_dtypes"] for dtype in layer}
                )
            profiles.append(
                {
                    "name": path.parent.name,
                    "status": row["status"],
                    "precision": precision,
                    "checks": row.get("checks", {}),
                    "evidence": str(path.relative_to(run)),
                }
            )
            for cell in row.get("checks", {}).get("hardware", {}).get("rows", []):
                if "samples" not in cell:
                    continue
                hardware.append(
                    {
                        "id": model["id"],
                        "profile": path.parent.name,
                        "input_tokens": cell["input_tokens"],
                        "output_budget": cell["output_budget"],
                        "ttft_ms": metric_median(cell, "time_to_first_token_ms"),
                        "prefill_tps": metric_median(cell, "prefill_tokens_per_second"),
                        "decode_tps": metric_median(cell, "decode_tokens_per_second"),
                        "peak_mlx_gib": max(s["peak_mlx_bytes"] for s in cell["samples"]) / 1024**3,
                        "actual_generated_tokens": [
                            s["actual_generated_tokens"] for s in cell["samples"]
                        ],
                        "source_precision": row.get("precision_profile", {}).get("requested"),
                        "weight_dtypes": row.get("precision_profile", {}).get(
                            "actual_weight_dtypes", []
                        ),
                        "cold_source_load_seconds": row.get("cold_load_seconds"),
                        "peak_rss_gib": (
                            row["peak_rss_bytes"] / 1024**3
                            if row.get("peak_rss_bytes") is not None
                            else None
                        ),
                        "performance_qualified": False,
                        "status": cell["status"],
                        "evidence": str(path.relative_to(run)),
                    }
                )
        if len(revisions) > 1:
            raise ValueError("profiles mix exact checkpoint revisions")
        document = {
            "schema_version": "ten-model-result-v1",
            "checkpoint_id": model["id"],
            "model_id": model["repo"],
            "architecture": model["family"],
            "checkpoint_revision": next(iter(revisions), None),
            "checkpoint": checkpoint,
            "license": next((r for r in licenses if r["id"] == model["id"]), {}),
            "qualification": "NOT_QUALIFIED",
            "profiles": profiles,
            "evidence_paths": [p["evidence"] for p in profiles],
            "limitations": [
                "A passing scoped profile is not all-gates release qualification.",
                "Only listed numerical, quality, API and hardware workloads were executed.",
                "Peak MLX is the process lifetime high-water mark, including reference comparison.",
                "Local Transformers source inspected; PyTorch numerical execution not performed.",
            ],
        }
        for directory, suffix in (
            ("primitive-calls", "calls"),
            ("quantized-projection-traces", "trace"),
            ("sse-contract-final", "sse"),
        ):
            auxiliary = run / directory / f"{model['id']}-{suffix}.json"
            if auxiliary.is_file():
                document["evidence_paths"].append(str(auxiliary.relative_to(run)))
        validate(document, schema)
        (destination / (model["id"] + ".json")).write_text(json.dumps(document, indent=2) + "\n")
        documents.append(document)
    (destination / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    (destination / "hardware-coverage.json").write_text(
        json.dumps(coverage(documents, hardware), indent=2) + "\n"
    )
    return documents, hardware


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    documents, hardware = compile_results(args.run, args.output)
    print(
        json.dumps(
            {
                "checkpoint_records": len(documents),
                "hardware_cells": len(hardware),
                "qualified": 0,
                "alpha": "NO-GO",
            }
        )
    )


if __name__ == "__main__":
    main()
