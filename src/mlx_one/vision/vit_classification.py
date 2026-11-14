"""Experimental native ViT classifier for one immutable HF checkpoint.

Encoder layers are shared with TrOCR. This module is intentionally separate
from the task registry until checkpoint, processor and task qualification finish.
Transformers v4.57.1 source: 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc
(licensed Apache-2.0). No reference framework is used at runtime.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_one.models.ocr.trocr.model import ViT
from mlx_one.vision.images import decode_image

MODEL_ID = "google/vit-base-patch16-224"
REVISION = "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"
_HASHES = {
    "config.json": "abdf73cc4ce79495a83f32a8c9265daa7a1363f5a3efa4704c88a6e7f45ad19d",
    "preprocessor_config.json": "90d1ae427c27d2b9dcb01f6a62ba96aa67aac2de1697c59cd212055b5035c4a2",
    "model.safetensors": "1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d",
}


class ViTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit = ViT(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(self, pixels):
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (224, 224, 3):
            raise ValueError("expected NHWC pixel tensor with shape [batch, 224, 224, 3]")
        return self.classifier(self.vit(pixels)[:, 0])


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedViTClassifier:
    model: ViTClassifier
    labels: dict
    model_id: str = MODEL_ID
    revision: str = REVISION

    def predict(self, image, *, top_k=5):
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= len(self.labels)
        ):
            raise ValueError("top_k must be an integer between 1 and the number of labels")
        pixels, original_size = preprocess(image)
        logits = self.model(mx.array(pixels[None]))
        scores = mx.softmax(logits, axis=-1, precise=True)[0]
        indices = mx.argsort(-scores)[:top_k]
        mx.eval(scores, indices)
        return {
            "labels": [
                {"label": self.labels[str(i)], "score": float(scores[i].item()), "index": i}
                for i in indices.tolist()
            ],
            "model": self.model_id,
            "revision": self.revision,
            "image_size": {"width": original_size[0], "height": original_size[1]},
        }


def preprocess(image):
    """Decode using MLX-One limits and return HF-normalized NHWC float32 pixels."""
    from PIL import Image

    decoded = decode_image(image)
    try:
        size = decoded.size
        resized = decoded.resize((224, 224), resample=Image.Resampling.BILINEAR)
        try:
            values = np.asarray(resized, dtype=np.float32)
            # HF's rescale computes in float64 then casts to float32.
            values = (values.astype(np.float64) / 255.0).astype(np.float32)
            values = (values - np.float32(0.5)) / np.float32(0.5)
            return values, size
        finally:
            resized.close()
    finally:
        decoded.close()


def load_vit_classifier(path, *, revision=REVISION):
    """Load the exact local, hash-verified safetensors checkpoint; no network access."""
    if revision != REVISION:
        raise ValueError("unsupported ViT checkpoint revision")
    root = Path(path)
    for name, expected_hash in _HASHES.items():
        if not (root / name).is_file() or _digest(root / name) != expected_hash:
            raise ValueError(f"missing or unexpected checkpoint asset: {name}")
    payload = json.loads((root / "config.json").read_text())
    labels = payload["id2label"]
    cfg = SimpleNamespace(**{**payload, "num_labels": len(labels)})
    model = ViTClassifier(cfg)
    tensors = mx.load(str(root / "model.safetensors"))
    target = dict(tree_flatten(model.parameters()))
    if set(tensors) != set(target):
        raise ValueError(
            f"checkpoint tensor mismatch: missing={sorted(set(target) - set(tensors))}, "
            f"unexpected={sorted(set(tensors) - set(target))}"
        )
    for name, value in tensors.items():
        if name == "vit.embeddings.patch_embeddings.projection.weight":
            value = value.transpose(0, 2, 3, 1)
        if value.shape != target[name].shape or value.dtype != mx.float32:
            raise ValueError(f"unexpected shape or dtype for {name}")
        tensors[name] = value
    model.load_weights(list(tensors.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return LoadedViTClassifier(model, labels)
