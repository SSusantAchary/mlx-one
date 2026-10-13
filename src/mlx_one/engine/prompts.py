"""One prepared prompt for preview, adoption, execution, and publication."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedPrompt:
    messages: tuple
    text: str
    token_ids: tuple[int, ...]
    context_tokens: int
    rendering_options: dict
    semantic_boundary: int = 0


def prepare_chat(bundle, messages, output_tokens, context_tokens, *, context_shift=False,
                 rendering_options=None):
    context_tokens = context_tokens or bundle.context_length
    if output_tokens >= context_tokens:
        raise ValueError("max_tokens leaves no room inside the configured context")
    settings = dict(rendering_options or {})
    selected = [dict(message) for message in messages]
    while True:
        text = bundle.chat_template.render(selected, **settings)
        tokens = tuple(bundle.tokenizer.encode(text))
        if len(tokens) + output_tokens <= context_tokens:
            break
        if not context_shift:
            raise ValueError("messages and requested output exceed the configured context")
        start = 0
        while start < len(selected) and selected[start]["role"] in {"system", "developer"}:
            start += 1
        if len(selected) - start <= 1:
            raise ValueError("the system message or newest turn cannot fit in the context")
        end = start + 1
        while end < len(selected) and selected[end]["role"] == "assistant":
            end += 1
        del selected[start:end]
    boundary = 0
    for end in range(1, len(selected)):
        if selected[end - 1]["role"] not in {"system", "developer", "assistant"}:
            continue
        try:
            candidate = bundle.chat_template.render(selected[:end],
                                                    **(settings | {"add_generation_prompt": False}))
            prefix = tuple(bundle.tokenizer.encode(candidate))
        except ValueError:
            continue
        if prefix and len(prefix) < len(tokens) and tokens[:len(prefix)] == prefix:
            boundary = max(boundary, len(prefix))
    return PreparedPrompt(tuple(selected), text, tokens, context_tokens, settings, boundary)
