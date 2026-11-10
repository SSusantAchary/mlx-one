"""Capture two pinned CC0 image rows with range reads in a reference environment."""

import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

REPO = "AIOmarRehan/AnimalsDataset"
REVISION = "676c9639be89ef4847325dcdbf141051319129f2"
FILE = "data/train-00000-of-00001.parquet"


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        super().__init__()
        self.url, self.size, self.position = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        )
        return self.position

    def read(self, n=-1):
        n = self.size - self.position if n < 0 else min(n, self.size - self.position)
        if n <= 0:
            return b""
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{self.position + n - 1}"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise RuntimeError("fixture retrieval requires an HTTP range response")
            data = response.read(n)
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true", timeout=60
    ) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION or metadata["cardData"]["license"] != "cc0-1.0":
        raise ValueError("fixture identity or license differs from the inspected inventory")
    size = next(file["size"] for file in metadata["siblings"] if file["rfilename"] == FILE)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILE}"
    parquet = pq.ParquetFile(RangeFile(url, size))
    rows = next(parquet.iter_batches(batch_size=2)).to_pylist()
    args.output.mkdir(exist_ok=True, parents=True)
    record = dict(
        dataset=REPO, revision=REVISION, license="cc0-1.0", file=FILE, rows=[0, 1], files=[]
    )
    for index, row in enumerate(rows):
        image = next(
            value for value in row.values() if isinstance(value, dict) and "bytes" in value
        )
        data = image["bytes"]
        name = f"{index}.jpg"
        (args.output / name).write_bytes(data)
        record["files"].append(
            dict(
                local_name=name,
                sha256=hashlib.sha256(data).hexdigest(),
                row=index,
                labels={k: v for k, v in row.items() if not isinstance(v, dict)},
            )
        )
    (args.output / "dataset.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
