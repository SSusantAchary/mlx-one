"""Prepare immutable public ASR fixtures; datasets/pyarrow are validation-only.

This script intentionally does not import MLX, download model code, or alter the
production dependency set. A populated output directory is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET = "openslr/librispeech_asr"
REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
SOURCE = "clean/test/0000.parquet"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_LANGUAGES = {"fr_fr": "French", "de_de": "German", "cmn_hans_cn": "Chinese"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--dataset", choices=["librispeech", "fleurs"], default="librispeech")
    parser.add_argument("--language", choices=FLEURS_LANGUAGES)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.count < 16 or args.output.exists():
        parser.error("count must be >=16 and output must not exist")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if args.dataset == "fleurs" and args.language is None:
        parser.error("FLEURS requires an explicit --language")
    dataset = DATASET if args.dataset == "librispeech" else "google/fleurs"
    revision = REVISION if args.dataset == "librispeech" else FLEURS_REVISION
    asset = (
        SOURCE
        if args.dataset == "librispeech"
        else f"parquet-data/{args.language}/test-00000-of-00001.parquet"
    )
    source = Path(
        hf_hub_download(
            dataset, asset, repo_type="dataset", revision=revision, local_files_only=args.offline
        )
    )
    if args.dataset == "librispeech":
        rows = sorted(pq.read_table(source).to_pylist(), key=lambda item: item["id"])[: args.count]
    else:
        # FLEURS sentence IDs can repeat across recordings. Pin source row as a
        # secondary sort key and part of fixture identity; never deduplicate audio.
        table = pq.ParquetFile(source)
        ids = table.read(columns=["id"]).column("id").to_pylist()
        selected = sorted(range(len(ids)), key=lambda i: (ids[i], i))[: args.count]
        wanted = set(selected)
        found, offset = {}, 0
        for batch in table.iter_batches(batch_size=32):
            hits = [i for i in range(batch.num_rows) if offset + i in wanted]
            for i in hits:
                row = batch.slice(i, 1).to_pylist()[0]
                row["source_row"] = offset + i
                found[offset + i] = row
            offset += batch.num_rows
            if len(found) == len(wanted):
                break
        rows = [found[i] for i in selected]
    if len(rows) != args.count:
        raise ValueError("dataset shard has fewer records than requested")
    args.output.mkdir(parents=True)
    records = []
    for index, row in enumerate(rows):
        # IDs are metadata, not untrusted filesystem paths.
        extension = ".flac" if args.dataset == "librispeech" else ".wav"
        target = args.output / f"clip-{index:03d}{extension}"
        with target.open("xb") as destination:
            destination.write(row["audio"]["bytes"])
        records.append(
            {
                "id": (
                    row["id"]
                    if args.dataset == "librispeech"
                    else f"{args.language}:{row['id']}:{row['source_row']}"
                ),
                "audio": target.name,
                "audio_sha256": digest(target),
                "reference": row["text"] if args.dataset == "librispeech" else row["transcription"],
                "anchor": index < 16 if args.dataset == "librispeech" else False,
                **(
                    {
                        "language": FLEURS_LANGUAGES[args.language],
                        "source_row": row["source_row"],
                        "publisher_audio_path": row["audio"].get("path"),
                    }
                    if args.dataset == "fleurs"
                    else {}
                ),
            }
        )
    manifest = {
        "dataset": dataset,
        "revision": revision,
        "source": asset,
        "source_sha256": digest(source),
        "records": records,
        **(
            {
                "license": "cc-by-4.0",
                "language": FLEURS_LANGUAGES[args.language],
                "selection": (
                    "first count test recordings sorted by (sentence ID, source row); "
                    "no deduplication"
                ),
            }
            if args.dataset == "fleurs"
            else {}
        ),
    }
    with (args.output / "manifest.json").open("x") as destination:
        json.dump(manifest, destination, indent=2)
    print(json.dumps({"fixtures": str(args.output), "count": len(records)}), flush=True)


if __name__ == "__main__":
    main()
