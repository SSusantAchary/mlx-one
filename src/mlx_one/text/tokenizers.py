"""Tokenizer contracts and Hugging Face tokenizer.json adapter."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol


class TextTokenizer(Protocol):
    bos_token: str | None
    eos_token: str | None
    bos_token_id: int | None
    eos_token_id: int | None
    special_tokens: Mapping[str, str]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str: ...


class HFTokenizerAdapter:
    """Small mlx-one interface over the standalone Hugging Face tokenizers package."""

    def __init__(
        self,
        tokenizer: object,
        *,
        bos_token: str | None,
        eos_token: str | None,
        bos_token_id: int | None,
        eos_token_id: int | None,
        special_tokens: Mapping[str, str],
    ) -> None:
        self._tokenizer = tokenizer
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.special_tokens = dict(special_tokens)

    @classmethod
    def from_directory(cls, directory: str | Path) -> HFTokenizerAdapter:
        root = Path(directory)
        tokenizer_path = root / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise ValueError(f"tokenizer.json is missing from {root}")
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            config = _json_object(root / "tokenizer_config.json", required=False)
            special = _json_object(root / "special_tokens_map.json", required=False)
        except Exception as exc:
            raise ValueError(f"cannot load tokenizer assets: {exc}") from exc
        bos = _token_text(config.get("bos_token", special.get("bos_token")))
        eos = _token_text(config.get("eos_token", special.get("eos_token")))
        token_values = {
            key: text
            for key in set(config) | set(special)
            if key.endswith("_token")
            and (text := _token_text(config.get(key, special.get(key)))) is not None
        }
        return cls(
            tokenizer,
            bos_token=bos,
            eos_token=eos,
            bos_token_id=tokenizer.token_to_id(bos) if bos is not None else None,
            eos_token_id=tokenizer.token_to_id(eos) if eos is not None else None,
            special_tokens=token_values,
        )

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        return str(
            self._tokenizer.decode(
                [int(value) for value in token_ids], skip_special_tokens=skip_special_tokens
            )
        )


def _token_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def _json_object(path: Path, *, required: bool) -> dict[str, object]:
    if not path.exists() and not required:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value
