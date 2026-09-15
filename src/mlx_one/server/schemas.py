"""Validated HTTP request schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str = Field(min_length=1)


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1)
    detail: Literal["auto", "low", "high"] = "auto"


class ImageContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[TextContentPart | ImageContentPart, Field(discriminator="type")]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]
    reasoning_content: str | None = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str | list[ContentPart]) -> str | list[ContentPart]:
        if isinstance(value, str):
            if not value:
                raise ValueError("message content cannot be empty")
        elif not value or len(value) > 64:
            raise ValueError("message content requires one to 64 parts")
        return value


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


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    input: str | list[str]
    dimensions: int | None = Field(default=None, ge=1)
    input_type: Literal["query", "document"] = "document"

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str | list[str]) -> str | list[str]:
        values = [value] if isinstance(value, str) else value
        if not values or len(values) > 256 or any(not item for item in values):
            raise ValueError("input requires one to 256 non-empty strings")
        return value


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=256)
    instruction: str | None = None
    top_n: int | None = Field(default=None, ge=1)

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("documents must be non-empty strings")
        return value
