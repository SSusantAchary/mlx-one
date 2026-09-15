import asyncio
import base64

import pytest

from mlx_one.server.app import _resolve_message_images
from mlx_one.server.media import (
    ImageResolver,
    MediaError,
    ResolvedImage,
    _is_public_address,
)

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


def test_data_url_is_validated_and_hashed() -> None:
    pytest.importorskip("PIL")
    source = "data:image/png;base64," + base64.b64encode(PNG_1X1).decode()
    image = asyncio.run(ImageResolver().resolve(source))
    assert image.content == PNG_1X1
    assert image.media_type == "image/png"
    assert len(image.source_digest) == 64


def test_data_url_rejects_invalid_or_mismatched_content() -> None:
    pytest.importorskip("PIL")
    with pytest.raises(MediaError, match="invalid base64"):
        asyncio.run(ImageResolver().resolve("data:image/png;base64,!!!"))
    source = "data:image/jpeg;base64," + base64.b64encode(PNG_1X1).decode()
    with pytest.raises(MediaError, match="do not match"):
        asyncio.run(ImageResolver().resolve(source))


def test_public_address_filter_rejects_local_networks() -> None:
    assert _is_public_address("8.8.8.8")
    assert not _is_public_address("127.0.0.1")
    assert not _is_public_address("169.254.169.254")
    assert not _is_public_address("::1")


def test_message_images_are_normalized_after_validation() -> None:
    class Resolver:
        async def resolve(self, source: str) -> ResolvedImage:
            assert source == "https://example.com/image.png"
            return ResolvedImage(PNG_1X1, "image/png", "digest")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
            ],
        }
    ]
    normalized = asyncio.run(_resolve_message_images(messages, Resolver()))  # type: ignore[arg-type]
    image = normalized[0]["content"][1]
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert image["mlx_digest"] == "digest"


def test_message_images_are_limited_before_fetching() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"https://e.test/{index}"}}
                for index in range(5)
            ],
        }
    ]
    with pytest.raises(MediaError, match="four images"):
        asyncio.run(_resolve_message_images(messages, ImageResolver()))
