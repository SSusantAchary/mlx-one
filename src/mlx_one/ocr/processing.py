"""Checkpoint-specific OCR processors. No universal resize or invented layout."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from mlx_one.ocr.documents import PixelTransform


@dataclass
class PreparedOCR:
    ids: tuple[int, ...]
    kwargs: dict
    transform: PixelTransform
    metadata: dict


def _pixels(image, size, metadata, *, channels_first=False):
    with image.resize(size, resample=Image.Resampling(metadata.get("resample", 3))) as resized:
        value = np.asarray(resized, dtype=np.float32) * metadata.get("rescale_factor", 1 / 255)
    if metadata.get("do_normalize", True):
        value = (value - np.array(metadata.get("image_mean", [0.5] * 3), np.float32)) / np.array(
            metadata.get("image_std", [0.5] * 3), np.float32
        )
    return mx.array(value.transpose(2, 0, 1)[None] if channels_first else value[None])


def prepare_ocr(bundle, image, request):
    family, config, tok = bundle.architecture, bundle.config, bundle.tokenizer
    region = request.region or (0, 0, image.width, image.height)
    if region[2] > image.width or region[3] > image.height:
        raise ValueError("crop region extends beyond page dimensions")
    # Integer crop coordinates are explicit and recorded; avoid hidden Pillow rounding.
    if any(int(c) != c for c in region):
        raise ValueError("crop region must use integer source pixel coordinates")
    cropped = image.crop(region)
    try:
        metadata = bundle.processor_metadata
        if family == "trocr":
            size = (config.encoder.image_size,) * 2
            kwargs = {"pixel_values": _pixels(cropped, size, metadata)}
            ids = [config.decoder.decoder_start_token_id]
            vision_tokens = (size[0] // config.encoder.patch_size) ** 2 + 1
        elif family == "got_ocr2":
            size = (config.vision_config.img_size,) * 2
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
            kwargs = {
                "pixel_values": _pixels(
                    cropped, size, {"image_mean": mean, "image_std": std, "resample": 3}
                )
            }
            count = (size[0] // config.vision_config.patch_size // 4) ** 2
            mode = "OCR with format:" if request.output_format == "markdown" else "OCR:"
            prompt = (
                "<|im_start|>system\n        You should follow the instructions carefully "
                "and explain your answers in detail.<|im_end|><|im_start|>user\n<img>"
                + "<imgpad>" * count
                + "</img>\n"
                + mode
                + "<|im_end|><|im_start|>assistant\n"
            )
            ids = tok.encode(prompt)
            if ids.count(config.im_patch_token) != count:
                raise ValueError("GOT prompt tokenizer does not preserve image placeholders")
            vision_tokens = count
        elif family == "florence2":
            source_size = metadata.get("size", {"height": 768, "width": 768})
            if isinstance(source_size, int):
                size = (source_size,) * 2
            else:
                size = (source_size.get("width", 768), source_size.get("height", 768))
            kwargs = {"pixel_values": _pixels(cropped, size, metadata, channels_first=True)}
            regions = request.task in {"ocr.detect", "ocr.region"}
            prompt = (
                "What is the text in the image, with regions?"
                if regions
                else "What is the text in the image?"
            )
            kwargs["encoder_input_ids"] = mx.array([tok.encode(prompt, add_special_tokens=True)])
            ids = [config.text_config.decoder_start_token_id]
            vision_tokens = (size[0] // 32) * (size[1] // 32) + 1
        elif family == "paddleocr_vl":
            from mlx_one.vision.processing import _smart_resize

            v = config.vision_config
            width, height = _smart_resize(
                cropped.width,
                cropped.height,
                factor=v.patch_size * v.spatial_merge_size,
                min_pixels=metadata.get("min_pixels", 112896),
                max_pixels=metadata.get("max_pixels", 1003520),
            )
            size = (width, height)
            pixels = _pixels(cropped, size, metadata)
            gh, gw = height // v.patch_size, width // v.patch_size
            patches = pixels[0].reshape(gh, v.patch_size, gw, v.patch_size, 3)
            patches = patches.transpose(0, 2, 4, 1, 3).reshape(
                1, gh * gw, 3, v.patch_size, v.patch_size
            )
            count = gh * gw // v.spatial_merge_size**2
            prompt = {"ocr.table": "Table Recognition:", "ocr.formula": "Formula Recognition:"}.get(
                request.task, "OCR:"
            )
            ids = tok.encode(
                "<|begin_of_sentence|>User: <|IMAGE_START|>"
                + "<|IMAGE_PLACEHOLDER|>" * count
                + "<|IMAGE_END|>"
                + prompt
                + "\nAssistant:\n"
            )
            if ids.count(config.image_token_id) != count:
                raise ValueError("Paddle OCR prompt image-token count differs from processor grid")
            kwargs = {"pixel_values": patches, "image_grid_thw": mx.array([[1, gh, gw]])}
            vision_tokens = count
        elif family == "lightonocr":
            settings = metadata.get("image_processor", metadata)
            maximum = settings.get("size", {}).get("longest_edge", config.vision_config.image_size)
            factor = config.vision_config.patch_size * config.spatial_merge_size
            scale = min(1, maximum / max(cropped.size))
            scaled_size = (math.floor(dim * scale) for dim in cropped.size)
            size = tuple(max(factor, math.ceil(dim / factor) * factor) for dim in scaled_size)
            count = (size[0] // factor) * (size[1] // factor)
            # LightOnOCR has no natural-language OCR instruction in its published template.
            prompt = (
                "<|im_start|>system<|im_end|>\n<|im_start|>user\n"
                + "<|image_pad|>" * count
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            from mlx_one.text.chat import ChatTemplate

            template = ChatTemplate.from_directory(bundle.path, family, tok)
            if template.template:
                # Reuse the sandbox compiler, with OCR's explicitly constructed
                # single-image message. Text chat validation intentionally rejects
                # structured image content and must stay strict for text models.
                prompt = str(
                    ChatTemplate._compile(template.template).render(
                        messages=[{"role": "user", "content": [{"type": "image"}]}],
                        add_vision_id=False,
                        add_generation_prompt=True,
                        bos_token=getattr(tok, "bos_token", "") or "",
                        eos_token=getattr(tok, "eos_token", "") or "",
                    )
                )
                if prompt.count("<|image_pad|>") != 1:
                    raise ValueError("LightOnOCR template must emit exactly one image placeholder")
                prompt = prompt.replace("<|image_pad|>", "<|image_pad|>" * count)
            ids = tok.encode(prompt)
            if ids.count(config.image_token_index) != count:
                raise ValueError("LightOnOCR prompt image-token count differs from processor grid")
            kwargs = {
                "pixel_values": _pixels(cropped, size, settings, channels_first=True),
                "image_sizes": mx.array([[size[1], size[0]]]),
            }
            vision_tokens = count
        elif family == "deepseek_ocr":
            ids, kwargs, size, vision_tokens = _deepseek(bundle, cropped, request)
        else:
            raise ValueError("no native OCR processor for checkpoint")
        pixels = size[0] * size[1]
        if pixels > request.limits.max_pixels:
            raise ValueError("processed image exceeds pixel budget")
        if len(ids) + request.max_tokens > bundle.context_length:
            raise ValueError("OCR prompt and generation exceed checkpoint context")
        return PreparedOCR(
            tuple(ids),
            kwargs,
            PixelTransform(region, size),
            {
                "processed_size": size,
                "visual_tokens": vision_tokens,
                "prompt_tokens": len(ids),
                "crop_first": request.region is not None,
                "source_crop": region,
            },
        )
    finally:
        cropped.close()


def _deepseek(bundle, image, request):
    params = bundle.config.vision_config.params or {}
    base, local = params.get("base_size", 1024), params.get("local_size", 640)
    grid = (1, 1)
    if max(image.size) > local:
        candidates = [
            (w, h)
            for w in range(1, request.limits.max_tiles + 1)
            for h in range(1, request.limits.max_tiles + 1)
            if 2 <= w * h <= request.limits.max_tiles
        ]
        if candidates:
            candidates.sort(key=lambda t: (t[0] * t[1], t[0], t[1]))
            best = float("inf")
            for pair in candidates:
                error = abs(image.width / image.height - pair[0] / pair[1])
                if error < best or (
                    error == best
                    and image.width * image.height > 0.5 * local**2 * pair[0] * pair[1]
                ):
                    grid, best = pair, error
    global_view = ImageOps.pad(image, (base, base), color=(127,) * 3)
    crops = []
    try:
        pixels = _pixels(global_view, (base, base), {}, channels_first=True)
        if grid != (1, 1):
            with image.resize((grid[0] * local, grid[1] * local)) as enlarged:
                for row in range(grid[1]):
                    for col in range(grid[0]):
                        crop = enlarged.crop(
                            (col * local, row * local, (col + 1) * local, (row + 1) * local)
                        )
                        try:
                            crops.append(_pixels(crop, (local, local), {}, channels_first=True)[0])
                        finally:
                            crop.close()
        local_pixels = mx.stack(crops) if crops else mx.zeros((0, 3, local, local))
        global_grid, local_grid = math.ceil(base / 64), math.ceil(local / 64)
        count = (
            global_grid * (global_grid + 1)
            + 1
            + ((local_grid * grid[0] + 1) * local_grid * grid[1] if crops else 0)
        )
        instruction = (
            "<|grounding|>Convert the document to markdown."
            if (request.output_format == "markdown")
            else "Free OCR."
        )
        # The pinned original OCR checkpoint uses the plain conversation, not VL2 chat roles.
        before = [bundle.tokenizer.bos_token_id if bundle.tokenizer.bos_token_id is not None else 0]
        after = bundle.tokenizer.encode("\n" + instruction)
        ids = before + [bundle.config.image_token_index] * count + after
        mask = [False] * len(before) + [True] * count + [False] * len(after)
        return (
            ids,
            {
                "pixel_values": (local_pixels, pixels),
                "images_spatial_crop": mx.array([grid]),
                "images_seq_mask": mx.array([mask]),
            },
            (base, base),
            count,
        )
    finally:
        global_view.close()
