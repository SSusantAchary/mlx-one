"""Checkpoint-specific completion tokenization, shared by API and serving paths."""

from __future__ import annotations


def completion_token_ids(bundle, prompt: str) -> list[int]:
    tokens = list(bundle.tokenizer.encode(prompt))
    architecture = getattr(bundle, "architecture", None)
    if architecture in {"openelm", "lfm2"}:
        # These publisher recipes require an initial BOS for completion.
        # Chat templates may already render it; never insert it twice.
        bos = bundle.tokenizer.bos_token_id
        if isinstance(bos, bool) or not isinstance(bos, int) or bos < 0:
            raise ValueError(f"{architecture} completion requires a compatible tokenizer BOS token")
        if not tokens or tokens[0] != bos:
            tokens.insert(0, bos)
    return tokens
