"""Explicitly fetch four pinned HF documentation images; never fetch model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

MANIFEST = Path(__file__).with_name("hf-images.json")


def prepare(output_dir, *, source_dir=None):
    from PIL import Image, ImageOps

    metadata = json.loads(MANIFEST.read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for item in metadata["sources"]:
        if source_dir is None:
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(
                    metadata["dataset"],
                    item["path"],
                    repo_type="dataset",
                    revision=metadata["revision"],
                    local_dir=output_dir / "originals",
                )
            )
        else:
            path = Path(source_dir) / item["path"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("source image checksum mismatch: " + item["path"])
        sources[item["path"]] = data
    for job in metadata["workloads"].values():
        for item in job["images"]:
            with Image.open(BytesIO(sources[item["source"]])) as opened:
                source = ImageOps.exif_transpose(opened).convert("RGB")
                processed = ImageOps.pad(
                    source,
                    tuple(item["size"]),
                    method=Image.Resampling.LANCZOS,
                    color="white",
                    centering=(0.5, 0.5),
                )
                buffer = BytesIO()
                processed.save(buffer, format="PNG")
                data = buffer.getvalue()
                source.close()
                processed.close()
            if item.get("sha256") and hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(
                    "prepared image checksum mismatch; use the recorded Pillow version"
                )
            (output_dir / item["file"]).write_bytes(data)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output_dir))


if __name__ == "__main__":
    main()
