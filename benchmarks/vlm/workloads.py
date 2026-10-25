"""Versioned, deterministic source images; checkpoint processors retain their own sizes."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

PROMPT = "Describe the visible shapes, colors, and their spatial arrangement."
OCR_TEXT = "MLX ONE\nInvoice 1042\nTotal 123.45 USD"


HF_MANIFEST = Path(__file__).with_name("hf-images.json")
LEGACY_NAMES = ("VLM-S1", "VLM-S2", "VLM-S3", "VLM-S4")
HF_NAMES = ("VLM-HF-S1", "VLM-HF-S2", "VLM-HF-S3", "VLM-HF-S4")
WORKLOAD_NAMES = (*LEGACY_NAMES, *HF_NAMES)


def workload_metadata(name):
    if name in HF_NAMES:
        manifest = json.loads(HF_MANIFEST.read_text())
        job = manifest["workloads"][name]
        return {
            "name": name,
            "protocol": name.lower() + "-v1",
            "source_sizes": [item["size"] for item in job["images"]],
            "image_sha256": [item["sha256"] for item in job["images"]],
            "prompt": job["prompt"],
            "expected_text": job["expected_text"],
            "image_provenance": {
                key: manifest[key]
                for key in (
                    "dataset",
                    "revision",
                    "sources",
                    "license",
                    "transformation",
                    "pillow_version",
                )
            },
        }
    job = workload(name)
    return {key: value for key, value in job.items() if key != "images"}


def workload(name, *, image_dir=None):
    if name in HF_NAMES:
        if image_dir is None:
            raise ValueError(
                "HF workloads require --image-dir; run benchmarks.vlm.prepare_images first"
            )
        metadata = workload_metadata(name)
        manifest = json.loads(HF_MANIFEST.read_text())
        images = [
            (Path(image_dir) / item["file"]).read_bytes()
            for item in manifest["workloads"][name]["images"]
        ]
        if [hashlib.sha256(data).hexdigest() for data in images] != metadata["image_sha256"]:
            raise ValueError("prepared HF image hashes do not match the pinned workload")
        return {**metadata, "images": images}

    if name not in {"VLM-S1", "VLM-S2", "VLM-S3", "VLM-S4"}:
        raise ValueError("unknown VLM workload")
    from PIL import Image

    directory = Path(__file__).with_name("images")
    images = [path.read_bytes() for path in sorted(directory.glob(name + "-*.png"))]
    if len(images) != (4 if name == "VLM-S3" else 1):
        raise ValueError("versioned workload images are missing")
    sizes = []
    for data in images:
        with Image.open(BytesIO(data)) as image:
            sizes.append(list(image.size))
    return {
        "name": name,
        "protocol": name.lower() + "-v1",
        "images": images,
        "source_sizes": sizes,
        "image_sha256": [hashlib.sha256(im).hexdigest() for im in images],
        "prompt": "Extract all text exactly, preserving line breaks."
        if name == "VLM-S4"
        else PROMPT,
        "expected_text": OCR_TEXT if name == "VLM-S4" else None,
    }
