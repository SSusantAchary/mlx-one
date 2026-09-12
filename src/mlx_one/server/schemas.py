"""Validated HTTP request schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)
    reasoning_content: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    seed: int = 0
    stop: str | list[str] | None = None
    stream: bool = False
    reasoning: Literal["auto", "on", "off"] | None = None
    reasoning_budget: int | None = Field(default=None, ge=-1)

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value: str | list[str] | None) -> str | list[str] | None:
        values = [value] if isinstance(value, str) else value
        if values is not None and (
            len(values) > 8 or any(not item or len(item) > 256 for item in values)
        ):
            raise ValueError("stop accepts at most eight non-empty strings of 256 characters")
        return value
