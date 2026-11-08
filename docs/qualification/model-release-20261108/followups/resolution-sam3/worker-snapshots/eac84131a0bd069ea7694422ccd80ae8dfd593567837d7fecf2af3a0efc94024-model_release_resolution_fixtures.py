"""Prepare pinned independent alignment gold and natural multi-object SAM video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    fixtures = args.work.resolve() / "fixtures"
    args.evidence.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    gold = fixtures / "alignment-gold"
    gold.mkdir(parents=True, exist_ok=True)
    source = dict(
        dataset="kylelovesllms/timit_asr_ipa",
        revision="d696c0d9c28ab0ff17f3d1ce817688ffe95464a1",
        file="data/test-00000-of-00001.parquet",
    )
    shard = Path(
        hf_hub_download(
            source["dataset"],
            source["file"],
            repo_type="dataset",
            revision=source["revision"],
            local_dir=gold,
        )
    )
    records = []
    for index, row in enumerate(pq.read_table(shard).slice(0, 10).to_pylist()):
        audio = gold / f"{index:02d}.wav"
        audio.write_bytes(row["audio"]["bytes"])
        words = [
            dict(text=w["utterance"], start=w["start"] / 16000, end=w["stop"] / 16000)
            for w in row["word_detail"]
        ]
        records.append(
            dict(
                index=index,
                id=row["id"],
                timit_path=row["timit_path"],
                audio=str(audio),
                audio_sha256=sha(audio),
                duration=row["duration"],
                text=" ".join(w["text"] for w in words),
                words=words,
            )
        )
    manifest = dict(
        schema_version="alignment-gold-v1",
        source={**source, "path": str(shard)},
        selection="first 10 test rows, selected before aligner inference",
        annotation="TIMIT word_detail start/stop sample indices; independent corpus annotations",
        gates=dict(
            endpoint_mean_absolute_error_max_seconds=0.08,
            endpoint_p95_error_max_seconds=0.2,
            word_coverage_min=1.0,
        ),
        records=records,
    )
    write(gold / "source.json", manifest["source"])
    write(gold / "manifest.json", manifest)
    write(
        args.evidence / "resolution-alignment-fixtures.json",
        {**manifest, "source": {**manifest["source"], "file_sha256": sha(shard)}},
    )

    natural = fixtures / "sam-natural"
    frames = natural / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    video = Path(
        hf_hub_download(
            "hf-internal-testing/sam2-fixtures",
            "bedroom.mp4",
            repo_type="dataset",
            revision="ee201e1a2ba9f944f26ab4b3219a32cf6dd27252",
            local_dir=natural,
        )
    )
    if not list(frames.glob("*.png")):
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(video),
                "-frames:v",
                "32",
                str(frames / "%03d.png"),
            ],
            check=True,
        )
    paths = sorted(frames.glob("*.png"))
    if len(paths) != 32:
        raise ValueError("natural video preparation requires exactly the first 32 frames")
    manifest = dict(
        dataset="hf-internal-testing/sam2-fixtures",
        revision="ee201e1a2ba9f944f26ab4b3219a32cf6dd27252",
        file="bedroom.mp4",
        sha256=sha(video),
        frames=[dict(path=str(path), sha256=sha(path)) for path in paths],
        selection="first 32 video frames, frozen before model inference",
        prompts=dict(
            concept="person",
            objects=[dict(id=1, point=[400, 200], label=1), dict(id=2, point=[240, 260], label=1)],
        ),
        gates=dict(
            mask_iou_min=0.98,
            score_max_absolute_error=0.02,
            box_max_error_pixels=2,
            stable_explicit_object_ids=True,
            foreground_each_frame=True,
        ),
    )
    write(natural / "manifest.json", manifest)
    write(args.evidence / "resolution-sam-video-fixture.json", manifest)


if __name__ == "__main__":
    main()
