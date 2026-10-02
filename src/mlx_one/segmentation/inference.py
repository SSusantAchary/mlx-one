"""Native SAM3 image workflows with host-side, original-resolution results."""

from __future__ import annotations

import numpy as np

from .loading import LoadedSegmentationModel, load_segmentation_model
from .processing import mask_boxes, validate_coordinates, validate_labels
from .schemas import ConceptSegmentationResult, InteractiveSegmentationResult


def _bundle(model, task, kwargs):
    bundle = (
        model
        if isinstance(model, LoadedSegmentationModel)
        else load_segmentation_model(model, task=task, **kwargs)
    )
    if bundle.task != task:
        raise ValueError(f"this operation requires a {task} bundle, not {bundle.task}")
    return bundle


def segment_image(
    model,
    image,
    *,
    text=None,
    boxes=None,
    box_labels=None,
    threshold=0.3,
    mask_threshold=0.5,
    **load_options,
):
    from mlx_one.models.segmentation.sam3._runtime import interpolate

    if not 0 <= threshold <= 1 or not 0 <= mask_threshold <= 1:
        raise ValueError("thresholds must be in [0,1]")
    bundle = _bundle(model, "concept", load_options)
    pixels, image = bundle.processor.image(image)
    size = image.height, image.width
    inputs = dict(pixel_values=pixels)
    ids, attention = bundle.processor.text(text or ("visual" if boxes is not None else ""))
    inputs.update(input_ids=ids, attention_mask=attention)
    if boxes is not None:
        inputs["input_boxes"], inputs["input_boxes_labels"] = bundle.processor.concept_boxes(
            boxes, box_labels, size
        )
    elif box_labels is not None:
        raise ValueError("box_labels requires boxes")
    outputs = bundle.model(**inputs)
    scores = outputs.pred_logits.sigmoid()[0] * outputs.presence_logits.sigmoid()[0]
    keep = np.flatnonzero(np.asarray(scores) > threshold)
    boxes_out = np.asarray(outputs.pred_boxes[0])[keep] * np.asarray(
        [size[1], size[0], size[1], size[0]]
    )
    if len(keep):
        masks = interpolate(
            outputs.pred_masks[0][keep].sigmoid().unsqueeze(1),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        masks = np.asarray(masks) > mask_threshold
    else:
        masks = np.zeros((0, *size), dtype=bool)
    return ConceptSegmentationResult(
        masks, np.asarray(scores)[keep], boxes_out.astype(np.float32), size, bundle.provenance
    )


def predict_masks(
    model,
    image,
    *,
    points=None,
    point_labels=None,
    boxes=None,
    mask=None,
    multimask=True,
    **load_options,
):
    from mlx_one.models.segmentation.sam3._runtime import Tensor, interpolate, ops

    bundle = _bundle(model, "interactive", load_options)
    pixels, image = bundle.processor.image(image)
    h, w = image.height, image.width
    target = bundle.processor.image_size
    inputs = dict(pixel_values=pixels, multimask_output=multimask)
    if points is not None:
        points = validate_coordinates(points, w, h)
        labels = validate_labels(point_labels, len(points))
        inputs.update(
            input_points=Tensor(
                (points * np.asarray([target / w, target / h]))[None, None].astype(np.float32)
            ),
            input_labels=ops.tensor(labels[None, None], dtype=ops.long),
        )
    elif point_labels is not None:
        raise ValueError("point_labels requires points")
    if boxes is not None:
        boxes = validate_coordinates(boxes, w, h, boxes=True)
        if len(boxes) != 1:
            raise ValueError("interactive prediction accepts one box per prompt group")
        inputs["input_boxes"] = Tensor(
            (boxes * np.asarray([target / w, target / h, target / w, target / h]))[None].astype(
                np.float32
            )
        )
    if mask is not None:
        value = np.asarray(mask, dtype=np.float32)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("refinement mask must be finite 2-D logits")
        mask_size = 4 * (target // bundle.model.config.prompt_encoder_config.patch_size)
        inputs["input_masks"] = interpolate(
            Tensor(value[None, None]),
            size=(mask_size, mask_size),
            mode="bilinear",
            align_corners=False,
        )
    if points is None and boxes is None and mask is None:
        raise ValueError("interactive prediction requires points, a box, or a mask")
    outputs = bundle.model(**inputs)
    logits = outputs.pred_masks[0, 0]
    masks = interpolate(
        logits.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(1)
    masks = np.asarray(masks) > 0
    return InteractiveSegmentationResult(
        masks,
        np.asarray(outputs.iou_scores[0, 0]),
        mask_boxes(masks),
        (h, w),
        np.asarray(logits),
        bundle.provenance,
    )


def generate_masks(
    model,
    image,
    *,
    points_per_side=32,
    predicted_iou_threshold=0.88,
    stability_threshold=0.95,
    **load_options,
):
    """Single-image point-grid automatic mask generation with mask-IoU NMS."""
    from .processing import read_image

    if not isinstance(points_per_side, int) or points_per_side < 1:
        raise ValueError("points_per_side must be positive")
    if not 0 <= predicted_iou_threshold <= 1 or not 0 <= stability_threshold <= 1:
        raise ValueError("thresholds must be in [0,1]")
    bundle = _bundle(model, "interactive", load_options)
    image = read_image(image)
    masks = []
    scores = []
    logits = []
    # Reuse vision features through the architecture's explicit image_embeddings API.
    from mlx_one.models.segmentation.sam3._runtime import Tensor, interpolate, ops

    pixels, _ = bundle.processor.image(image)
    vision = bundle.model.get_image_embeddings(pixels)
    scale = bundle.processor.image_size
    for y in (np.arange(points_per_side) + 0.5) / points_per_side:
        for x in (np.arange(points_per_side) + 0.5) / points_per_side:
            output = bundle.model(
                image_embeddings=vision,
                input_points=Tensor([[[[x * scale, y * scale]]]]),
                input_labels=ops.tensor([[[1]]], dtype=ops.long),
                multimask_output=True,
            )
            for quality, low in zip(
                np.asarray(output.iou_scores[0, 0]), output.pred_masks[0, 0], strict=True
            ):
                low_host = np.asarray(low)
                union = (low_host > -1).sum()
                intersection = (low_host > 1).sum()
                if (
                    quality < predicted_iou_threshold
                    or not union
                    or intersection / union < stability_threshold
                ):
                    continue
                full = (
                    np.asarray(
                        interpolate(
                            low.unsqueeze(0).unsqueeze(0),
                            size=(image.height, image.width),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                    )
                    > 0
                )
                masks.append(full)
                scores.append(quality)
                logits.append(low_host)
    keep = []
    for index in np.argsort(-np.asarray(scores)):
        if all(
            (masks[index] & masks[j]).sum() / max(1, (masks[index] | masks[j]).sum()) <= 0.7
            for j in keep
        ):
            keep.append(index)
    shape = (image.height, image.width)
    selected = np.stack([masks[i] for i in keep]) if keep else np.zeros((0, *shape), dtype=bool)
    return InteractiveSegmentationResult(
        selected,
        np.asarray(scores, dtype=np.float32)[keep],
        mask_boxes(selected),
        shape,
        np.stack([logits[i] for i in keep]) if keep else None,
        bundle.provenance,
    )
