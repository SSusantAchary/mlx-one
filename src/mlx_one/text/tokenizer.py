"""Native GPT-2 byte-level BPE tokenizer."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import regex


def bytes_to_unicode() -> dict[int, str]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = list(values)
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, (chr(value) for value in characters), strict=True))


class GPT2Tokenizer:
    _PATTERN = regex.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )

    def __init__(
        self,
        vocabulary: dict[str, int],
        merges: Iterable[tuple[str, str]],
        *,
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        unk_token: str = "<|endoftext|>",
    ) -> None:
        self.encoder = dict(vocabulary)
        self.decoder = {value: key for key, value in self.encoder.items()}
        if len(self.decoder) != len(self.encoder):
            raise ValueError("GPT-2 vocabulary contains duplicate token IDs")
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        self.bpe_ranks = {pair: index for index, pair in enumerate(merges)}
        self._cache: dict[str, tuple[str, ...]] = {}
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.special_tokens = {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "unk_token": unk_token,
        }
        self.bos_token_id = self.token_id(bos_token)
        self.eos_token_id = self.token_id(eos_token)

    @classmethod
    def from_directory(cls, directory: str | Path) -> GPT2Tokenizer:
        root = Path(directory)
        vocabulary = _json_object(root / "vocab.json")
        tokenizer_config = _json_object(root / "tokenizer_config.json", required=False)
        try:
            lines = (root / "merges.txt").read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"cannot read GPT-2 merges.txt: {exc}") from exc
        merges = []
        for line in lines:
            if not line or line.startswith("#"):
                continue
            pair = line.split()
            if len(pair) != 2:
                raise ValueError("invalid GPT-2 BPE merge")
            merges.append((pair[0], pair[1]))
        return cls(
            {str(key): int(value) for key, value in vocabulary.items()},
            merges,
            bos_token=_token_text(tokenizer_config.get("bos_token"), "<|endoftext|>"),
            eos_token=_token_text(tokenizer_config.get("eos_token"), "<|endoftext|>"),
            unk_token=_token_text(tokenizer_config.get("unk_token"), "<|endoftext|>"),
        )

    def token_id(self, token: str) -> int:
        try:
            return self.encoder[token]
        except KeyError as exc:
            raise ValueError(f"GPT-2 token is not available: {token}") from exc

    def bpe(self, token: str) -> tuple[str, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token)
        if len(word) < 2:
            return word
        while True:
            pairs = set(zip(word, word[1:], strict=False))
            pair = min(pairs, key=lambda value: self.bpe_ranks.get(value, float("inf")))
            if pair not in self.bpe_ranks:
                break
            first, second = pair
            combined: list[str] = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    combined.extend(word[index:])
                    break
                combined.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index + 1] == second:
                    combined.append(first + second)
                    index += 2
                else:
                    combined.append(word[index])
                    index += 1
            word = tuple(combined)
            if len(word) == 1:
                break
        if len(self._cache) < 65536:
            self._cache[token] = word
        return word

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        result: list[int] = []
        unknown = self.token_id(self.unk_token)
        for piece in self._PATTERN.findall(text):
            encoded = "".join(self.byte_encoder[value] for value in piece.encode("utf-8"))
            for token in self.bpe(encoded):
                result.append(self.encoder.get(token, unknown))
        return result

    def token_bytes(self, token_id: int) -> bytes:
        token = self.decoder.get(int(token_id), self.unk_token)
        if token in {self.bos_token, self.eos_token}:
            return b""
        return bytes(self.byte_decoder[character] for character in token)

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        raw = bytearray()
        pieces: list[str] = []

        def flush() -> None:
            if raw:
                pieces.append(raw.decode("utf-8", errors="replace"))
                raw.clear()

        for token_id in token_ids:
            token = self.decoder.get(int(token_id), self.unk_token)
            if skip_special_tokens and token in {self.bos_token, self.eos_token}:
                continue
            if token in {self.bos_token, self.eos_token, self.unk_token}:
                flush()
                pieces.append(token)
                continue
            try:
                raw.extend(self.byte_decoder[character] for character in token)
            except KeyError:
                flush()
                pieces.append(token)
        flush()
        return "".join(pieces)


def _token_text(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    raise ValueError("GPT-2 tokenizer special tokens must be strings or token objects")


def _json_object(path: Path, *, required: bool = True) -> dict[str, object]:
    if not path.exists() and not required:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read GPT-2 tokenizer asset {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"GPT-2 tokenizer asset {path.name} must contain an object")
    return value
