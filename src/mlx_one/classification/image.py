"""Native supervised image classification with an explicit publisher processor."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlx_one.models.classification.vit_config import ViTConfig
from mlx_one.models.classification.vit_weights import sanitize_weights, weight_contract
from mlx_one.retrieval.loading import _read_json, _read_safetensors, _resolve


@dataclass(frozen=True)
class LoadedImageClassifier:
    model: Any
    processor: dict
    model_id: str
    revision: str | None
    path: Path
    architecture: str = "vit"


@dataclass(frozen=True)
class ImageClassificationResult:
    predictions: tuple[dict, ...]
    model: str
    revision: str | None
    task: str = "image-classification"
    score_semantics: str = "supervised-class-softmax"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


def load_image_classification_model(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> LoadedImageClassifier:
    """Load a strict ViT classifier and its fixed-resolution RGB processor."""
    root = _resolve(model, revision=revision, offline=offline, cache_dir=cache_dir)
    config = ViTConfig.from_dict(_read_json(root / "config.json"))
    processor = _read_json(root / "preprocessor_config.json")
    size = processor.get("size", config.image_size)
    if size not in (config.image_size, {"height": config.image_size, "width": config.image_size}):
        raise ValueError("ViT processor dimensions conflict with the trained position grid")
    if processor.get("do_center_crop", False):
        raise ValueError("this ViT processor does not implement center-crop profiles")
    weights = sanitize_weights(_read_safetensors(root), config)
    weight_contract(config).validate(weights)
    from mlx_one.models.classification.vit import ViTForImageClassification

    native = ViTForImageClassification(config)
    native.load_weights(list(weights.items()), strict=True)
    native.eval()
    return LoadedImageClassifier(native, processor, str(model), revision, root)


def prepare_image(image: Any, config: ViTConfig, processor: dict[str, Any]) -> np.ndarray:
    """Match ViT's configured PIL resize, rescale, and channel normalization."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            if opened.width * opened.height > 32_000_000:
                raise ValueError("image exceeds the 32-megapixel input bound")
            image = opened.convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise ValueError(
                "array images must be uint8 RGB/HWC; normalized neural tensors use model.forward"
            )
        image = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        if image.width * image.height > 32_000_000:
            raise ValueError("image exceeds the 32-megapixel input bound")
        image = image.convert("RGB")
    else:
        raise ValueError("image must be a PIL image, local path, or uint8 array")
    if image.width * image.height > 32_000_000:
        raise ValueError("image exceeds the 32-megapixel input bound")
    if processor.get("do_resize", True):
        image = image.resize(
            (config.image_size, config.image_size), resample=processor.get("resample", 2)
        )
    value = np.asarray(image).astype(np.float32)
    if processor.get("do_rescale", True):
        value = (value.astype(np.float64) * processor.get("rescale_factor", 1 / 255)).astype(
            np.float32
        )
    if processor.get("do_normalize", True):
        mean = np.asarray(processor.get("image_mean", [0.5] * 3), np.float32)
        std = np.asarray(processor.get("image_std", [0.5] * 3), np.float32)
        if (
            mean.shape != (3,)
            or std.shape != (3,)
            or np.any(std <= 0)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(std))
        ):
            raise ValueError("ViT normalization requires three finite channels and positive scales")
        value = (value - mean) / std
    return value.transpose(2, 0, 1)


def classify_image(
    model: str | Path | LoadedImageClassifier,
    images: Any,
    *,
    top_k: int = 5,
    revision: str | None = None,
    offline: bool = False,
) -> ImageClassificationResult:
    """Classify one image or a bounded batch against the checkpoint's label set."""
    bundle = (
        model
        if isinstance(model, LoadedImageClassifier)
        else load_image_classification_model(model, revision=revision, offline=offline)
    )
    values = images if isinstance(images, (list, tuple)) else [images]
    if (
        not 1 <= len(values) <= 8
        or type(top_k) is not int
        or not 1 <= top_k <= bundle.model.config.num_labels
    ):
        raise ValueError(
            "image classification requires 1–8 images and top_k within the class count"
        )
    pixels = np.stack(
        [prepare_image(image, bundle.model.config, bundle.processor) for image in values]
    )
    import mlx.core as mx

    output = bundle.model(mx.array(pixels))
    scores = mx.softmax(output.logits, axis=-1, precise=True)
    mx.eval(scores, output.logits)
    predictions = []
    for row, logits in zip(scores.tolist(), output.logits.tolist(), strict=True):
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))[:top_k]
        predictions.append(
            {
                "labels": [
                    dict(
                        index=i,
                        label=bundle.model.config.id2label.get(str(i), str(i)),
                        score=row[i],
                        logit=logits[i],
                    )
                    for i in order
                ]
            }
        )
    return ImageClassificationResult(tuple(predictions), bundle.model_id, bundle.revision)
