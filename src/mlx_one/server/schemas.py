"""Validated HTTP request schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


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


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str | None = Field(default=None, max_length=65536)
    parameters: dict[str, JsonValue] = Field(default_factory=lambda: {"type": "object"})
    strict: bool | None = None

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value):
        from mlx_one.text.harmony_tools import validate_schema
        validate_schema(value)
        return value


class FunctionTool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function"]
    function: FunctionDefinition


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    arguments: str = Field(max_length=1048576)

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, value: str) -> str:
        from mlx_one.text.harmony_tools import load_arguments
        load_arguments(value)
        return value


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=256)
    type: Literal["function"]
    function: FunctionCall


class SelectedFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=64)


class ToolChoiceObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function"]
    function: SelectedFunction


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = False


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, min_length=1, max_length=128)
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=256)
    reasoning_content: str | None = None

    @model_validator(mode="after")
    def validate_message(self):
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages may contain tool_calls")
        if self.role == "tool":
            if not self.tool_call_id or not isinstance(self.content, str):
                raise ValueError("tool messages require tool_call_id and text content")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid on tool messages")
        elif self.content is None or self.content == "":
            if not (self.role == "assistant" and self.tool_calls):
                raise ValueError("message content cannot be empty")
        if isinstance(self.content, list) and (not self.content or len(self.content) > 64):
            raise ValueError("message content requires one to 64 parts")
        return self


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    temperature: float = Field(default=0.7, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    seed: int = 0
    stop: str | list[str] | None = None
    stream: bool = False
    reasoning: Literal["auto", "on", "off"] | None = None
    reasoning_budget: int | None = Field(default=None, ge=-1)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    tools: list[FunctionTool] | None = Field(default=None, max_length=128)
    tool_choice: Literal["none", "auto", "required"] | ToolChoiceObject | None = None
    parallel_tool_calls: bool | None = None
    stream_options: StreamOptions | None = None

    @model_validator(mode="after")
    def validate_tools(self):
        names = [tool.function.name for tool in self.tools or []]
        if len(set(names)) != len(names):
            raise ValueError("tool function names must be unique")
        if isinstance(self.tool_choice, ToolChoiceObject):
            if self.tool_choice.function.name not in names:
                raise ValueError("selected function is not present in tools")
        if self.tool_choice == "required" and not names:
            raise ValueError("required tool choice requires tools")
        pending = {}
        seen = set()
        for message in self.messages:
            if message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError("tool_call_id has no pending assistant tool call")
                del pending[message.tool_call_id]
            else:
                if pending:
                    raise ValueError("assistant tool calls must be followed by their tool results")
                for call in message.tool_calls or []:
                    if call.id in seen:
                        raise ValueError("tool call IDs must be unique in conversation history")
                    seen.add(call.id)
                    pending[call.id] = call.function.name
        if pending:
            raise ValueError("missing tool results for assistant tool calls")
        return self

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value: str | list[str] | None) -> str | list[str] | None:
        values = [value] if isinstance(value, str) else value
        if values is not None and (
            len(values) > 8 or any(not item or len(item) > 256 for item in values)
        ):
            raise ValueError("stop accepts at most eight non-empty strings of 256 characters")
        return value


class CompletionRequest(BaseModel):
    """B1 raw-text subset of the completions API."""

    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    seed: int = 0
    stop: str | list[str] | None = None
    stream: bool = False

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value: str | list[str] | None) -> str | list[str] | None:
        return ChatCompletionRequest.validate_stop(value)


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
