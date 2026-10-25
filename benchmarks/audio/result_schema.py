"""Validate measured audio records before publishing any metric."""

from __future__ import annotations

import hashlib
import math
import re
import statistics

SCHEMA = "mlx-one-audio-benchmark-v1"
PROTOCOL = {
    "batch_size": 1,
    "warmups": 1,
    "repetitions": 5,
    "statistic": "median",
    "clock": "perf_counter_with_mlx_synchronize",
    "playback_pacing": False,
}
METRICS = (
    "processing_seconds",
    "audio_seconds",
    "real_time_factor",
    "audio_seconds_per_second",
    "peak_mlx_gib",
    "peak_rss_gib",
    "wer",
    "cer",
    "time_to_first_audio_ms",
    "output_audio_seconds",
    "alignment_mean_error_ms",
    "alignment_median_error_ms",
    "alignment_p95_error_ms",
    "alignment_max_error_ms",
    "unmatched_words",
    "first_partial_ms",
    "final_result_ms",
    "finalization_ms",
    "revised_partials",
    "buffer_peak_bytes",
)


def digest_file(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate(record):
    if record.get("schema") != SCHEMA or record.get("status") not in {"NOT_RUN", "MEASURED"}:
        raise ValueError("unsupported audio benchmark schema/status")
    for field in ("checkpoint", "revision", "task", "workload", "precision", "name"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError("record requires " + field)
    if not re.fullmatch("[0-9a-f]{40}", record["revision"]):
        raise ValueError("checkpoint revision must be an immutable 40-character SHA")
    if record["protocol"] != PROTOCOL:
        raise ValueError("record uses a different audio measurement protocol")
    if set(record["metrics"]) != set(METRICS):
        raise ValueError("incomplete audio metric fields")
    if record["status"] == "NOT_RUN":
        if record.get("runs") or any(v is not None for v in record["metrics"].values()):
            raise ValueError("unrun records cannot contain invented measurements")
        return record
    for field in ("hardware", "software", "implementation_sha256", "checkpoint_assets", "input"):
        if not record.get(field):
            raise ValueError("measured record requires " + field)
    if record.get("backend") != "native-mlx-metal" or record.get("measurement_type") not in {
        "direct-model-prepared-pcm",
        "python-engine",
        "full-file-pipeline",
    }:
        raise ValueError("unknown execution backend/measurement type")
    if not record["hardware"].get("chip") or not record["hardware"].get("memory_bytes"):
        raise ValueError("hardware chip/memory identity required")
    if not record["software"].get("mlx"):
        raise ValueError("MLX version required")
    if not re.fullmatch("[0-9a-f]{64}", record["implementation_sha256"]):
        raise ValueError("invalid implementation identity")
    if not record["checkpoint_assets"] or any(
        not re.fullmatch("[0-9a-f]{64}", value) for value in record["checkpoint_assets"].values()
    ):
        raise ValueError("invalid checkpoint file evidence")
    runs = record.get("runs", [])
    if len(runs) != PROTOCOL["repetitions"]:
        raise ValueError("five actual measured repetitions are required")
    for metric in METRICS:
        values = [row.get(metric) for row in runs]
        for value in values:
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("invalid metric " + metric)
        present = [v for v in values if v is not None]
        expected = statistics.median(present) if len(present) == len(runs) else None
        if record["metrics"][metric] != expected:
            raise ValueError("published metric is not the median of actual runs: " + metric)
    for row in runs:
        duration = (
            row["output_audio_seconds"]
            if record["task"] in {"synthesize", "speech-to-speech"}
            else row["audio_seconds"]
        )
        if not duration or row["processing_seconds"] <= 0:
            raise ValueError("actual processed audio duration and time are required")
        if not math.isclose(row["real_time_factor"], row["processing_seconds"] / duration):
            raise ValueError("RTF denominator does not match actual audio duration")
        if not math.isclose(row["audio_seconds_per_second"], duration / row["processing_seconds"]):
            raise ValueError("audio throughput does not match actual duration")
    return record
