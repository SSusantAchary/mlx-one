"""Bounded image retrieval for OpenAI-style multimodal requests."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import ipaddress
import socket
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.parse import urljoin, urlparse

SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class MediaError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedImage:
    content: bytes
    media_type: str
    source_digest: str


class ImageResolver:
    def __init__(
        self,
        *,
        maximum_bytes: int = 20 * 1024**2,
        maximum_redirects: int = 3,
        client_factory: Any | None = None,
    ) -> None:
        if maximum_bytes < 1 or maximum_redirects < 0:
            raise ValueError("image resolver limits are invalid")
        self.maximum_bytes = maximum_bytes
        self.maximum_redirects = maximum_redirects
        self._client_factory = client_factory

    async def resolve(self, source: str) -> ResolvedImage:
        if source.startswith("data:"):
            media_type, content = _decode_data_url(source, self.maximum_bytes)
        else:
            media_type, content = await self._fetch_https(source)
        image = _validated_image(content, media_type)
        return ResolvedImage(image, media_type, hashlib.sha256(image).hexdigest())

    async def _fetch_https(self, source: str) -> tuple[str, bytes]:
        import httpx

        factory = self._client_factory or (
            lambda: httpx.AsyncClient(
                follow_redirects=False,
                timeout=httpx.Timeout(10.0, connect=3.0),
                trust_env=False,
            )
        )
        current = source
        try:
            async with factory() as client:
                for redirect in range(self.maximum_redirects + 1):
                    await _validate_public_https_url(current)
                    async with client.stream("GET", current) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location or redirect == self.maximum_redirects:
                                raise MediaError("image redirect limit exceeded")
                            current = urljoin(current, location)
                            continue
                        response.raise_for_status()
                        media_type = response.headers.get("content-type", "").split(
                            ";", 1
                        )[0]
                        if media_type not in SUPPORTED_IMAGE_TYPES:
                            detail = media_type or "missing"
                            raise MediaError(f"unsupported image content type: {detail}")
                        declared = response.headers.get("content-length")
                        if declared:
                            try:
                                too_large = int(declared) > self.maximum_bytes
                            except ValueError as exc:
                                raise MediaError(
                                    "remote image has invalid length metadata"
                                ) from exc
                            if too_large:
                                raise MediaError("remote image exceeds the size limit")
                        content = bytearray()
                        async for chunk in response.aiter_bytes():
                            content.extend(chunk)
                            if len(content) > self.maximum_bytes:
                                raise MediaError("remote image exceeds the size limit")
                        return media_type, bytes(content)
        except MediaError:
            raise
        except httpx.HTTPError as exc:
            raise MediaError(f"remote image request failed: {exc}") from exc
        raise MediaError("image redirect limit exceeded")


def _decode_data_url(source: str, maximum_bytes: int) -> tuple[str, bytes]:
    header, separator, payload = source.partition(",")
    if not separator or not header.endswith(";base64"):
        raise MediaError("image data URL must use base64 encoding")
    media_type = header[5:-7].lower()
    if media_type not in SUPPORTED_IMAGE_TYPES:
        raise MediaError(f"unsupported image content type: {media_type or 'missing'}")
    if len(payload) > ((maximum_bytes + 2) // 3) * 4 + 4:
        raise MediaError("image data URL exceeds the size limit")
    try:
        content = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaError("image data URL contains invalid base64") from exc
    if len(content) > maximum_bytes:
        raise MediaError("image data URL exceeds the size limit")
    return media_type, content


async def _validate_public_https_url(source: str) -> None:
    parsed = urlparse(source)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise MediaError("remote images require an HTTPS URL without credentials")
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise MediaError("remote image URL has an invalid port") from exc
    try:
        records = await asyncio.to_thread(
            socket.getaddrinfo,
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise MediaError("remote image host cannot be resolved") from exc
    addresses = {record[4][0] for record in records}
    if not addresses or any(not _is_public_address(value) for value in addresses):
        raise MediaError("remote image host resolves to a non-public address")


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value.split("%", 1)[0])
    return address.is_global


def _validated_image(content: bytes, media_type: str) -> bytes:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError("image processing requires `pip install 'mlx-one[vision]'`") from exc
    try:
        with Image.open(BytesIO(content)) as image:
            expected = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
            if expected.get(image.format) != media_type:
                raise MediaError("image bytes do not match the declared content type")
            pixels = int(image.width) * int(image.height)
            if Image.MAX_IMAGE_PIXELS is not None and pixels > Image.MAX_IMAGE_PIXELS:
                raise MediaError("image dimensions exceed the safe pixel limit")
            image.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise MediaError(f"invalid or unsafe image: {exc}") from exc
    return content
