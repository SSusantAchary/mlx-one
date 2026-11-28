"""OpenAI-compatible image request fields and cancellation bridge."""

from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field


class ImageGenerationRequest(BaseModel):
    """Accept the supported OpenAI fields and bounded local extensions."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=8192)
    size: str = Field(default="512x512", pattern=r"^[1-9][0-9]{1,3}x[1-9][0-9]{1,3}$")
    n: int = Field(default=1, ge=1, le=1)
    response_format: str = Field(default="b64_json", pattern=r"^b64_json$")
    seed: int | None = None
    steps: int | None = None

    def dimensions(self) -> tuple[int, int]:
        """Return width and height in the request's documented order."""
        match = re.fullmatch(r"(\d+)x(\d+)", self.size)
        if match is None:
            raise ValueError("size must use WIDTHxHEIGHT")
        return int(match[1]), int(match[2])


@dataclass
class ImageRequestAbandoned(Exception):
    """The client disconnected or its service timeout elapsed."""

    status: int
    message: str


async def wait_for_image(request: Request, item: ImageGenerationRequest, engine, timeout: float):
    """Keep the event loop responsive while the device retains its active lease."""
    width, height = item.dimensions()
    stop = threading.Event()
    work = engine.submit(
        item.prompt, width=width, height=height, steps=item.steps, seed=item.seed, cancel=stop
    )
    result = asyncio.wrap_future(work)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout if timeout else None
    try:
        while not result.done():
            if await request.is_disconnected():
                raise ImageRequestAbandoned(499, "client disconnected")
            if deadline is not None and loop.time() >= deadline:
                raise ImageRequestAbandoned(408, "image generation timed out")
            wait = 0.1 if deadline is None else min(0.1, max(0.0, deadline - loop.time()))
            await asyncio.wait({result}, timeout=wait)
        return result.result()
    finally:
        if not result.done():
            stop.set()
            work.cancel()
            result.cancel()
