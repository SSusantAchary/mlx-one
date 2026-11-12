"""Experimental one-page QA candidate using the existing VLM task service.

This internal diagnostic interface is not release-qualified or task-registered.
It preserves existing Qwen model math and enforces the exact prior Q4 checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from mlx_one.text.schemas import TextGenerationOptions
from mlx_one.utils.hashing import sha256_file
from mlx_one.vision.loading import load_vlm_model
from mlx_one.vision.public import VisionLanguageEngine

MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
REVISION = "46d4cf06a06ffc1a766c214174f9cbed2f45bcab"
SOURCE_COMMIT = "8cb5963cc22174954e7dca2c0a3320b7dc2f4edc"
HASHES = {
    "model.safetensors": "636982419c940321ac0f7793dc9dc3575a4ee4843a6b167b2e8c3d3cd25dacf4",
    "config.json": "7ed631bd2786d251cb38bd2a6a8a78c31bc1316f728bffa84b2d8954d3cfcd63",
    "preprocessor_config.json": "f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0",
    "tokenizer.json": "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
    "tokenizer_config.json": "2f58f4bbd7bbce15d683f525954ef3a92cd82f5e06415a9c513859bf8ab72436",
}
MAX_PIXELS = 28 * 28 * 256


def load_document_bundle(path, *, revision=REVISION, offline=True):
    if revision != REVISION or not offline:
        raise ValueError("document QA requires the exact local existing Q4 checkpoint")
    root = Path(path)
    for name, expected in HASHES.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"document QA checkpoint asset mismatch: {name}")
    bundle = load_vlm_model(root, revision=REVISION, offline=True)
    from mlx_one.vision.processing import Qwen25ImageProcessor

    metadata = json.loads((root / "preprocessor_config.json").read_text())
    metadata["max_pixels"] = MAX_PIXELS
    processor = Qwen25ImageProcessor(bundle.tokenizer, bundle.model.config, metadata=metadata)
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return replace(bundle, model_id=MODEL_ID, processor=processor, processor_identity=identity)


@dataclass(frozen=True)
class DocumentQAResult:
    answer: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    finish_reason: str
    image_metadata: tuple[dict, ...]
    model: str = MODEL_ID
    revision: str = REVISION
    source_commit: str = SOURCE_COMMIT
    page_index: int = 1
    task: str = "document-question-answering"
    schema_version: str = "mlx-one-document-qa-candidate-v1"
    qualification: str = "EXPERIMENTAL; numerical/task gates incomplete"

    def to_dict(self):
        return asdict(self)


class DocumentQAEngine:
    """One image page, one bounded question, greedy answer, no invented citations."""

    def __init__(self, checkpoint):
        self.engine = VisionLanguageEngine(checkpoint, loader=load_document_bundle)

    def ask(self, image, question, *, max_tokens=32, cancel=None):
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question.encode("utf-8")) > 8192
            or "<image>" in question
            or "<|" in question
        ):
            raise ValueError(
                "question must be nonempty text up to8192 UTF-8 bytes without special tokens"
            )
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 64
        ):
            raise ValueError("document answer requires1–64 tokens")
        if isinstance(image, (str, Path)) and Path(image).suffix.lower() == ".pdf":
            raise ValueError(
                "document QA candidate accepts one image page; PDF/multi-page not admitted"
            )
        result = self.engine.generate(
            prompt=question,
            images=(image,),
            options=TextGenerationOptions(max_tokens=max_tokens, temperature=0),
            cancel=cancel,
        )
        return DocumentQAResult(
            result.text,
            result.token_ids,
            result.prompt_tokens,
            result.finish_reason,
            result.image_metadata,
        )

    def close(self):
        self.engine.close()
        with self.engine.service._lock:
            self.engine.service._task_models.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
