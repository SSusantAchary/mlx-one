"""Native Qwen3 NFC-normalized byte-level BPE tokenizer."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path

import regex

from mlx_one.text.tokenizer import bytes_to_unicode


class Qwen3Tokenizer:
    _PATTERN = regex.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )

    def __init__(
        self,
        vocabulary: dict[str, int],
        merges: Iterable[tuple[str, str]],
        added_tokens: dict[str, int],
        *,
        special_tokens: Iterable[str] = (),
        post_prefix: Sequence[int] = (),
        post_suffix: Sequence[int] = (),
        pad_token: str = "<|endoftext|>",
    ) -> None:
        self.encoder = {**vocabulary, **added_tokens}
        self.decoder = {value: key for key, value in self.encoder.items()}
        if len(self.decoder) != len(self.encoder):
            raise ValueError("Qwen3 vocabulary contains duplicate token IDs")
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        self.bpe_ranks = {pair: index for index, pair in enumerate(merges)}
        self._cache: dict[str, tuple[str, ...]] = {}
        self.added_tokens = frozenset(added_tokens)
        self.special_tokens = frozenset(special_tokens)
        self.post_prefix = tuple(post_prefix)
        self.post_suffix = tuple(post_suffix)
        self.pad_token = pad_token
        self.pad_token_id = self.token_id(pad_token)
        special = sorted(added_tokens, key=len, reverse=True)
        self._special_pattern = (
            regex.compile("(" + "|".join(regex.escape(token) for token in special) + ")")
            if special
            else None
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> Qwen3Tokenizer:
        root = Path(directory)
        payload = _json_object(root / "tokenizer.json")
        model = payload.get("model")
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise ValueError("Qwen3 tokenizer.json requires a BPE model")
        vocabulary = model.get("vocab")
        merges_data = model.get("merges")
        if not isinstance(vocabulary, dict) or not isinstance(merges_data, list):
            raise ValueError("Qwen3 tokenizer BPE vocabulary or merges are missing")
        if any(
            not isinstance(token, str)
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token, token_id in vocabulary.items()
        ):
            raise ValueError("Qwen3 BPE vocabulary must map strings to non-negative IDs")
        merges = []
        for item in merges_data:
            pair = item.split() if isinstance(item, str) else item
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError("invalid Qwen3 BPE merge")
            merges.append((str(pair[0]), str(pair[1])))
        added: dict[str, int] = {}
        special: set[str] = set()
        added_data = payload.get("added_tokens", [])
        if not isinstance(added_data, list):
            raise ValueError("Qwen3 added_tokens must be an array")
        for item in added_data:
            if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                raise ValueError("invalid Qwen3 added token")
            token_id = item.get("id")
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError("Qwen3 added token IDs must be non-negative integers")
            added[item["content"]] = token_id
            if item.get("special") is True:
                special.add(item["content"])
        config = _json_object(root / "tokenizer_config.json", required=False)
        pad_token = _token_text(config.get("pad_token"), "<|endoftext|>")
        encoder = {**vocabulary, **added}
        post_prefix, post_suffix = _post_processor_tokens(
            payload.get("post_processor"), encoder
        )
        return cls(
            {str(key): int(value) for key, value in vocabulary.items()},
            merges,
            added,
            special_tokens=special,
            post_prefix=post_prefix,
            post_suffix=post_suffix,
            pad_token=pad_token,
        )

    def token_id(self, token: str) -> int:
        try:
            return self.encoder[token]
        except KeyError as exc:
            raise ValueError(f"Qwen3 token is not available: {token}") from exc

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

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        if max_length is not None and (
            isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1
        ):
            raise ValueError("max_length must be null or a positive integer")
        normalized = unicodedata.normalize("NFC", text)
        pieces = (
            self._special_pattern.split(normalized)
            if self._special_pattern is not None
            else [normalized]
        )
        result: list[int] = []
        for piece in pieces:
            if not piece:
                continue
            if piece in self.added_tokens:
                result.append(self.encoder[piece])
                continue
            for segment in self._PATTERN.findall(piece):
                encoded = "".join(
                    self.byte_encoder[value] for value in segment.encode("utf-8")
                )
                for token in self.bpe(encoded):
                    try:
                        result.append(self.encoder[token])
                    except KeyError as exc:
                        raise ValueError(
                            f"Qwen3 tokenizer vocabulary misses byte token {token!r}"
                        ) from exc
        prefix = self.post_prefix if add_special_tokens else ()
        suffix = self.post_suffix if add_special_tokens else ()
        if max_length is not None:
            capacity = max_length - len(prefix) - len(suffix)
            if capacity < 0:
                raise ValueError("max_length is smaller than the tokenizer special-token template")
            result = result[:capacity]
        return [*prefix, *result, *suffix]

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        raw = bytearray()
        output: list[str] = []

        def flush() -> None:
            if raw:
                output.append(raw.decode("utf-8", errors="replace"))
                raw.clear()

        for token_id in token_ids:
            token = self.decoder.get(int(token_id))
            if token is None:
                raise ValueError(f"unknown Qwen3 token id: {token_id}")
            if token in self.added_tokens:
                if skip_special_tokens and token in self.special_tokens:
                    continue
                flush()
                output.append(token)
                continue
            raw.extend(self.byte_decoder[character] for character in token)
        flush()
        return "".join(output)

    def pad(
        self, sequences: Sequence[Sequence[int]], *, padding_side: str = "left"
    ) -> tuple[list[list[int]], list[list[bool]]]:
        if not sequences:
            raise ValueError("token batch cannot be empty")
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be left or right")
        width = max(len(sequence) for sequence in sequences)
        if width == 0:
            raise ValueError("token sequences cannot be empty")
        ids: list[list[int]] = []
        masks: list[list[bool]] = []
        for sequence in sequences:
            padding = width - len(sequence)
            if padding_side == "left":
                ids.append([self.pad_token_id] * padding + list(sequence))
                masks.append([False] * padding + [True] * len(sequence))
            else:
                ids.append(list(sequence) + [self.pad_token_id] * padding)
                masks.append([True] * len(sequence) + [False] * padding)
        return ids, masks


def _token_text(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    raise ValueError("Qwen3 tokenizer special tokens must be strings or token objects")


def _post_processor_tokens(
    value: object, encoder: dict[str, int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if value is None:
        return (), ()
    if not isinstance(value, dict) or value.get("type") != "TemplateProcessing":
        raise ValueError("Qwen3 tokenizer supports only TemplateProcessing post-processors")
    template = value.get("single")
    if not isinstance(template, list):
        raise ValueError("Qwen3 tokenizer post-processor requires a single template")
    prefix: list[int] = []
    suffix: list[int] = []
    target = prefix
    saw_sequence = False
    for item in template:
        if not isinstance(item, dict):
            raise ValueError("invalid Qwen3 tokenizer post-processor item")
        if "Sequence" in item:
            if saw_sequence:
                raise ValueError("Qwen3 tokenizer post-processor has multiple input sequences")
            saw_sequence = True
            target = suffix
            continue
        special = item.get("SpecialToken")
        if not isinstance(special, dict) or not isinstance(special.get("id"), str):
            raise ValueError("unsupported Qwen3 tokenizer post-processor item")
        try:
            target.append(encoder[special["id"]])
        except KeyError as exc:
            raise ValueError("Qwen3 post-processor references an unknown token") from exc
    if not saw_sequence:
        raise ValueError("Qwen3 tokenizer post-processor does not contain the input sequence")
    return tuple(prefix), tuple(suffix)


def _json_object(path: Path, *, required: bool = True) -> dict[str, object]:
    if not path.exists() and not required:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Qwen3 tokenizer asset {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Qwen3 tokenizer asset {path.name} must contain an object")
    return value
