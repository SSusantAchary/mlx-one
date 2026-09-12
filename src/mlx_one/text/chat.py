"""Safe chat-template loading and family fallbacks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

ChatMessage = Mapping[str, str]


class ChatTemplate:
    def __init__(self, model_type: str, template: str | None, tokenizer: Any) -> None:
        self.model_type = model_type
        self.template = template
        self.tokenizer = tokenizer
        if template:
            self._compile(template)

    @classmethod
    def from_directory(cls, root: Path, model_type: str, tokenizer: Any) -> ChatTemplate:
        path = root / "tokenizer_config.json"
        payload: dict[str, object] = {}
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read chat template metadata: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError("tokenizer_config.json must contain an object")
            payload = value
        template = payload.get("chat_template")
        if isinstance(template, list):
            candidates = [item for item in template if isinstance(item, dict)]
            preferred = next((item for item in candidates if item.get("name") == "default"), None)
            preferred = preferred or (candidates[0] if candidates else None)
            template = preferred.get("template") if preferred else None
        if template is not None and not isinstance(template, str):
            raise ValueError("chat_template must be a string or a named template list")
        template_path = root / "chat_template.jinja"
        if template is None and template_path.is_file():
            try:
                template = template_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"cannot read chat_template.jinja: {exc}") from exc
        return cls(model_type, template, tokenizer)

    def render(self, messages: Sequence[ChatMessage], **template_options: Any) -> str:
        _validate_messages(messages)
        if not self.template:
            return _fallback(self.model_type, messages)
        compiled = self._compile(self.template)
        eos = getattr(self.tokenizer, "eos_token", "") or ""
        bos = getattr(self.tokenizer, "bos_token", "") or ""
        try:
            token_values = dict(getattr(self.tokenizer, "special_tokens", {}))
            token_values.update(
                {
                    "messages": list(messages),
                    "add_generation_prompt": True,
                    "bos_token": bos,
                    "eos_token": eos,
                    "tools": None,
                }
            )
            token_values.update(template_options)
            return str(compiled.render(**token_values))
        except Exception as exc:
            raise ValueError(f"chat template rendering failed: {exc}") from exc

    @staticmethod
    def _compile(template: str) -> Any:
        try:
            from jinja2 import StrictUndefined
            from jinja2.sandbox import SandboxedEnvironment

            environment = SandboxedEnvironment(
                undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True
            )
            environment.filters["tojson"] = lambda value: json.dumps(
                value, ensure_ascii=False, separators=(",", ":")
            )

            def raise_exception(message: object) -> None:
                raise ValueError(str(message))

            environment.globals["raise_exception"] = raise_exception
            environment.globals["strftime_now"] = lambda format_string: datetime.now().strftime(
                str(format_string)
            )
            return environment.from_string(template)
        except Exception as exc:
            raise ValueError(f"invalid chat template: {exc}") from exc


def _validate_messages(messages: Sequence[ChatMessage]) -> None:
    if not messages:
        raise ValueError("messages cannot be empty")
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("message role must be system, user, or assistant")
        if not isinstance(message.get("content"), str):
            raise ValueError("message content must be text")


def _fallback(model_type: str, messages: Sequence[ChatMessage]) -> str:
    if model_type in {"qwen2", "qwen2_moe", "qwen3", "qwen3_5", "lfm2", "lfm2_moe"}:
        body = "".join(
            f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in messages
        )
        return body + "<|im_start|>assistant\n"
    if model_type == "openelm":
        labels = {"system": "System", "user": "User", "assistant": "Assistant"}
        body = "".join(f"### {labels[item['role']]}:\n{item['content']}\n" for item in messages)
        return body + "### Assistant:\n"
    labels = {"system": "System", "user": "User", "assistant": "Assistant"}
    body = "".join(f"{labels[item['role']]}: {item['content']}\n" for item in messages)
    return body + "Assistant:"
