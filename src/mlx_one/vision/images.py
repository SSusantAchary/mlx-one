"""Bounded local image decoding shared by native VLM entry points."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

MAX_IMAGES = 8
MAX_IMAGE_BYTES = 20 * 1024**2
MAX_SOURCE_PIXELS = 20_000_000
MAX_REQUEST_PIXELS = 22_000_000


def decode_image(value):
    """Return an owned, orientation-correct RGB image. Never fetch URLs."""
    from PIL import Image, ImageOps

    if isinstance(value, Image.Image):
        source, owned = value, False
    elif isinstance(value, (bytes, bytearray)):
        if len(value) > MAX_IMAGE_BYTES:
            raise ValueError("encoded image exceeds 20 MiB")
        source, owned = Image.open(BytesIO(value)), True
    elif isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("encoded image exceeds 20 MiB")
        source, owned = Image.open(path), True
    else:
        raise ValueError("image must be a local path, PNG/JPEG/WebP bytes, or PIL image")
    try:
        if source.format is not None and source.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("only PNG, JPEG and WebP images are supported")
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("animated images and video are unsupported")
        if min(source.size) < 1 or source.width * source.height > MAX_SOURCE_PIXELS:
            raise ValueError("image exceeds the decoded pixel budget")
        oriented = ImageOps.exif_transpose(source)
        try:
            # A white background makes transparent documents readable.
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                result = background.convert("RGB")
                rgba.close()
                background.close()
                return result
            return oriented.convert("RGB")
        finally:
            if oriented is not source:
                oriented.close()
    finally:
        if owned:
            source.close()
