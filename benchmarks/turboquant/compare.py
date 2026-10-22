#!/usr/bin/env python3
"""Compare completed JSON reports, never launch simultaneous models."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    native = json.loads(a.native.read_text())
    reference = json.loads(a.reference.read_text())
    if (
        native["schema_version"] != reference["schema_version"]
        or native["mlx"] != reference["mlx"]
        or native["platform"] != reference["platform"]
        or native["device"] != reference["device"]
    ):
        p.error("schema/MLX/host mismatch")
    comparisons = []
    if native["schema_version"] == "tq3-operators-v1":
        refs = {(x["operation"], tuple(x["shape"])): x for x in reference["results"]}
        for row in native["results"]:
            other = refs[(row["operation"], tuple(row["shape"]))]
            if row.get("fixture_sha256") != other.get("fixture_sha256"):
                p.error("synthetic packed fixtures differ")
            comparisons.append(
                dict(
                    operation=row["operation"],
                    shape=row["shape"],
                    native_median_ms=row["median_ms"],
                    reference_median_ms=other["median_ms"],
                    throughput_ratio=other["median_ms"] / row["median_ms"],
                    native_peak_increment_bytes=row["peak_increment_bytes"],
                    reference_peak_increment_bytes=other["peak_increment_bytes"],
                )
            )
        status = "SYNTHETIC_ONLY_NO_FULL_MODEL_PARITY_CLAIM"
    else:
        for key in [
            "config_sha256",
            "index_sha256",
            "token_file_sha256",
            "interface",
            "weight_sha256",
            "checkpoint_revision",
        ]:
            if native[key] != reference[key]:
                p.error(f"full model mismatch: {key}")
        for key in ["max_tokens", "temperature", "top_p", "seed", "kv_dtype", "warmups"]:
            if native["settings"][key] != reference["settings"][key]:
                p.error(f"benchmark setting mismatch: {key}")
        refs = {x["input_tokens"]: x for x in reference["results"]}
        for row in native["results"]:
            other = refs[row["input_tokens"]]
            if row["prompt_token_sha256"] != other["prompt_token_sha256"]:
                p.error("prompt IDs differ")
            counts = [x["generated_tokens"] for x in row["samples"]]
            refcounts = [x["generated_tokens"] for x in other["samples"]]
            if counts != refcounts:
                comparisons.append(
                    dict(
                        input_tokens=row["input_tokens"],
                        status="INCOMPARABLE_EOS_COUNTS",
                        native_counts=counts,
                        reference_counts=refcounts,
                    )
                )
                continue
            nt, rt = row["warm_decode_tps_median"], other["warm_decode_tps_median"]
            npref = sum(x["prefill_tokens_per_second"] for x in row["samples"]) / len(
                row["samples"]
            )
            rpref = sum(x["prefill_tokens_per_second"] for x in other["samples"]) / len(
                other["samples"]
            )
            comparisons.append(
                dict(
                    input_tokens=row["input_tokens"],
                    decode_ratio=nt / rt if nt and rt else None,
                    prefill_ratio=npref / rpref,
                    target_met=bool(nt and rt and nt / rt >= 0.9),
                )
            )
        status = (
            "SYNTHETIC_DRIVER_TEST_NO_MODEL_PARITY_CLAIM"
            if native.get("full_checkpoint_performance_status") == "SYNTHETIC_DRIVER_TEST"
            else "MEASURED_LOCAL_CHECKPOINT"
        )
        if status.startswith("SYNTHETIC"):
            for row in comparisons:
                row["target_met"] = None
    a.output.write_text(json.dumps(dict(status=status, comparisons=comparisons), indent=2) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
