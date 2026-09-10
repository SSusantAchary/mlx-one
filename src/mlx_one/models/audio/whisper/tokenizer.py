"""Small Hugging Face/OpenAI compatible byte-BPE tokenizer for Whisper."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import regex


def bytes_to_unicode() -> dict[int, str]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    characters = list(visible)
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(visible, (chr(value) for value in characters), strict=True))


class WhisperTokenizer:
    """Whisper byte BPE with explicit special-token handling."""

    _PATTERN = regex.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        regex.IGNORECASE,
    )

    def __init__(
        self,
        vocabulary: dict[str, int],
        merges: Iterable[tuple[str, str]],
        added_tokens: dict[str, int] | None = None,
    ) -> None:
        self.encoder = dict(vocabulary)
        self.encoder.update(added_tokens or {})
        self.decoder = {value: key for key, value in self.encoder.items()}
        if len(self.decoder) != len(self.encoder):
            raise ValueError("Whisper vocabulary contains duplicate token IDs")
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        self.bpe_ranks = {pair: index for index, pair in enumerate(merges)}
        self._bpe_cache: dict[str, tuple[str, ...]] = {}
        self.special_tokens = {
            token: value for token, value in self.encoder.items() if token.startswith("<|")
        }
        special = sorted(self.special_tokens, key=len, reverse=True)
        self._special_pattern = regex.compile(
            "(" + "|".join(regex.escape(token) for token in special) + ")"
        ) if special else None

    @classmethod
    def from_directory(cls, directory: str | Path) -> WhisperTokenizer:
        root = Path(directory)
        vocabulary = _json_object(root / "vocab.json")
        added = _json_object(root / "added_tokens.json", required=False)
        try:
            lines = (root / "merges.txt").read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"cannot read Whisper merges.txt: {exc}") from exc
        merges = []
        for line in lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError("invalid Whisper BPE merge")
            merges.append((parts[0], parts[1]))
        return cls(
            {str(key): int(value) for key, value in vocabulary.items()},
            merges,
            {str(key): int(value) for key, value in added.items()},
        )

    def token_id(self, token: str) -> int:
        try:
            return self.encoder[token]
        except KeyError as exc:
            raise ValueError(f"Whisper token is not available: {token}") from exc

    def bpe(self, token: str) -> tuple[str, ...]:
        if token in self._bpe_cache:
            return self._bpe_cache[token]
        word = tuple(token)
        if len(word) < 2:
            return word
        while True:
            pairs = set(zip(word, word[1:], strict=False))
            pair = min(pairs, key=lambda value: self.bpe_ranks.get(value, float("inf")))
            if pair not in self.bpe_ranks:
                break
            first, second = pair
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
        if len(self._bpe_cache) < 65536:
            self._bpe_cache[token] = word
        return word

    def encode(self, text: str, *, allow_special: bool = False) -> list[int]:
        pieces = [text]
        if allow_special and self._special_pattern is not None:
            pieces = [piece for piece in self._special_pattern.split(text) if piece]
        result: list[int] = []
        for piece in pieces:
            if allow_special and piece in self.special_tokens:
                result.append(self.special_tokens[piece])
                continue
            for match in self._PATTERN.findall(piece):
                encoded = "".join(self.byte_encoder[value] for value in match.encode("utf-8"))
                for token in self.bpe(encoded):
                    try:
                        result.append(self.encoder[token])
                    except KeyError as exc:
                        raise ValueError(
                            f"Whisper vocabulary cannot encode token {token!r}"
                        ) from exc
        return result

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        chunks: list[str] = []
        for token_id in token_ids:
            token = self.decoder.get(int(token_id), "")
            if skip_special and token in self.special_tokens:
                continue
            chunks.append(token)
        raw = bytearray()
        literal: list[str] = []
        for character in "".join(chunks):
            if character in self.byte_decoder:
                raw.append(self.byte_decoder[character])
            else:
                literal.append(character)
        return raw.decode("utf-8", errors="replace") + "".join(literal)

    def split_to_word_tokens(self, token_ids: Iterable[int]) -> tuple[list[str], list[list[int]]]:
        words: list[str] = []
        word_tokens: list[list[int]] = []
        current: list[int] = []
        for token_id in token_ids:
            token = int(token_id)
            piece = self.decode([token])
            if current and (piece.startswith(" ") or piece.strip() in _PUNCTUATION):
                words.append(self.decode(current))
                word_tokens.append(current)
                current = [token]
            else:
                current.append(token)
        if current:
            words.append(self.decode(current))
            word_tokens.append(current)
        return _merge_punctuation(words, word_tokens)


_PUNCTUATION = set(".,!?:;)]}、。！？，：；）】》")


def _merge_punctuation(
    words: list[str], word_tokens: list[list[int]]
) -> tuple[list[str], list[list[int]]]:
    prepended = "\"'“¿([{-"
    appended = "\"'.。,，!！?？:：”)]}、"
    for index in range(1, len(words)):
        if words[index].startswith(tuple(appended)) and words[index - 1]:
            words[index - 1] += words[index]
            word_tokens[index - 1].extend(word_tokens[index])
            words[index], word_tokens[index] = "", []
    for index in range(len(words) - 2, -1, -1):
        if words[index].endswith(tuple(prepended)) and words[index + 1]:
            words[index + 1] = words[index] + words[index + 1]
            word_tokens[index + 1] = word_tokens[index] + word_tokens[index + 1]
            words[index], word_tokens[index] = "", []
    combined = [
        (word, tokens)
        for word, tokens in zip(words, word_tokens, strict=True)
        if word
    ]
    return [word for word, _ in combined], [tokens for _, tokens in combined]


def _json_object(path: Path, *, required: bool = True) -> dict[str, object]:
    if not path.exists() and not required:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Whisper tokenizer asset {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Whisper tokenizer asset {path.name} must contain an object")
    return value
