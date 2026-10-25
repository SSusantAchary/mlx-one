"""Select 10 pinned test-set handwriting lines, fetching image/text Parquet columns only."""

import argparse
import hashlib
import io
import random

from benchmarks.ocr.corpus import EVIDENCE, SEED, sha, write

HAND_EVIDENCE = EVIDENCE.parent / "hand10-20261025"
DATASET = "toghrultahirov/handwritten_text_ocr"
REVISION = "3762d848764659233b02278e97751de068a1eebf"


def prepare(work):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download
    from PIL import Image

    fs = HfFileSystem()
    root = f"datasets/{DATASET}@{REVISION}"
    names = [f"data/test-{i:05}-of-00003.parquet" for i in range(3)]
    selected = random.Random(SEED).sample(range(2915), 10)
    fixtures, groups, offset = {}, [], 0
    for name in names:
        with fs.open(root + "/" + name, "rb", block_size=1024**2) as source:
            parquet = pq.ParquetFile(source)
            file_rows = parquet.metadata.num_rows
            local_offset = 0
            for group in range(parquet.num_row_groups):
                count = parquet.metadata.row_group(group).num_rows
                picks = [
                    i
                    for i in selected
                    if offset + local_offset <= i < offset + local_offset + count
                ]
                if picks:
                    table = parquet.read_row_group(group, columns=["text", "image"])
                    groups.append(
                        {
                            "path": name,
                            "row_group": group,
                            "rows": count,
                            "columns_read": ["text", "image"],
                        }
                    )
                    for index in picks:
                        row = table.slice(index - offset - local_offset, 1).to_pylist()[0]
                        image = row["image"]["bytes"]
                        if not isinstance(image, bytes) or not row["text"]:
                            raise ValueError("missing original image or ground-truth text")
                        with Image.open(io.BytesIO(image)) as im:
                            size, fmt = list(im.size), im.format.lower()
                        relative = f"test/{index}.{fmt}"
                        path = work / "handwriting" / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(image)
                        fixtures[index] = {
                            "id": str(index),
                            "path": relative,
                            "size": size,
                            "sha256": sha(path),
                            "reference": row["text"],
                            "reference_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                            "parquet": name,
                            "row_group": group,
                            "file_row": index - offset,
                        }
                local_offset += count
            offset += file_rows
    if offset != 2915 or len(fixtures) != 10:
        raise ValueError("unexpected pinned handwriting population")
    card = hf_hub_download(
        DATASET,
        "README.md",
        repo_type="dataset",
        revision=REVISION,
        local_dir=work / "handwriting-card",
    )
    write(
        HAND_EVIDENCE / "sample.json",
        {
            "schema_version": "mlx-one-ocr-hand10-sample-v1",
            "dataset": DATASET,
            "revision": REVISION,
            "split": "test",
            "population": 2915,
            "seed": SEED,
            "sampling": "random.Random(seed).sample(range(2915), 10); original Parquet order",
            "license": (
                "not declared in pinned dataset card; references only, images not redistributed"
            ),
            "card_sha256": sha(card),
            "parquet_reads": groups,
            "fixtures": [fixtures[i] for i in selected],
        },
    )
    print("Pinned 10 test-set handwriting lines", selected, flush=True)


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.work)


if __name__ == "__main__":
    main()
