"""Explicitly download and prepare the pinned, small LibriSpeech benchmark fixture.

Requires optional pyarrow and FFmpeg. No model weights are downloaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path

from benchmarks.audio.result_schema import digest_file

DATASET = "hf-internal-testing/librispeech_asr_dummy"
REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
PARQUET = "clean/validation-00000-of-00001.parquet"
PARQUET_SHA = "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603"
SAMPLE = "1272-128104-0003"


def prepare(output, cache_dir):
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if output.exists():
        raise ValueError("output must be a new directory")
    parquet = Path(
        hf_hub_download(
            DATASET, PARQUET, repo_type="dataset", revision=REVISION, cache_dir=cache_dir
        )
    )
    if digest_file(parquet) != PARQUET_SHA:
        raise ValueError("dataset checksum mismatch")
    rows = pq.read_table(parquet).to_pylist()
    row = next(r for r in rows if r["id"] == SAMPLE)
    flac = row["audio"]["bytes"]
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        input=flac,
        capture_output=True,
        check=True,
    ).stdout
    pcm = np.frombuffer(decoded, dtype="<f4")
    if len(pcm) != 158400 or not np.isfinite(pcm).all():
        raise ValueError("unexpected source audio duration or samples")
    original_seconds = len(pcm) / 16000
    pcm = np.pad(pcm, (0, 160000 - len(pcm)))
    output.mkdir(parents=True)
    audio = output / "librispeech-10s.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    transcript = output / "librispeech.txt"
    transcript.write_text(row["text"] + "\n")
    text = json.loads(Path("benchmarks/audio/workloads.json").read_text())["TTS-S1"]["text"]
    (output / "tts.txt").write_text(text + "\n")
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": "validation",
        "configuration": "clean",
        "sample_id": SAMPLE,
        "parquet_sha256": digest_file(parquet),
        "original_flac_sha256": hashlib.sha256(flac).hexdigest(),
        "original_seconds": original_seconds,
        "prepared_seconds": 10,
        "transform": "FFmpeg decode mono 16k; trailing zero padding to 10s; PCM16 WAV",
        "transcript": row["text"],
        "language": "English",
        "audio_sha256": digest_file(audio),
        "text_sha256": digest_file(transcript),
        "license": "LibriSpeech CC BY 4.0",
        "license_source": "https://www.openslr.org/12/",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    question = next(r for r in rows if r["id"] == "1272-135031-0012")
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        input=question["audio"]["bytes"],
        capture_output=True,
        check=True,
    ).stdout
    pcm = np.frombuffer(decoded, dtype="<f4")
    if len(pcm) != 32640 or not np.isfinite(pcm).all():
        raise ValueError("unexpected spoken-question duration or samples")
    question_audio = output / "sts-question.wav"
    with wave.open(str(question_audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    (output / "sts-question.txt").write_text(question["text"] + "\n")
    question_manifest = {
        **manifest,
        "sample_id": question["id"],
        "original_seconds": 2.04,
        "prepared_seconds": 2.04,
        "transform": "FFmpeg decode mono 16k; PCM16 WAV; no padding",
        "transcript": question["text"],
        "audio_sha256": digest_file(question_audio),
        "text_sha256": digest_file(output / "sts-question.txt"),
        "original_flac_sha256": hashlib.sha256(question["audio"]["bytes"]).hexdigest(),
        "workload": "STS-HF-Q1",
    }
    (output / "sts-question-manifest.json").write_text(
        json.dumps(question_manifest, indent=2) + "\n"
    )
    print(output / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output, args.cache_dir)


if __name__ == "__main__":
    main()
