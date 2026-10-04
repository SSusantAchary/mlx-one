"""Known-text units and timestamp repair matching the pinned Qwen processor.

Copyright 2026 The Alibaba Qwen team. SPDX-License-Identifier: Apache-2.0.
Adapted from qwen_asr/inference/qwen3_forced_aligner.py; see NOTICE.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import unicodedata
from pathlib import Path

KOREAN_SHA256 = "cc157a9ab78f92bf11ed98a69b285768d6627cf5cd57a581e6c06693effce37d"


def tokenizer_version(name: str, required: str) -> None:
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(f"alignment requires the optional {name}=={required} tokenizer") from exc
    if installed != required:
        raise ValueError(f"unsupported alignment tokenizer {name}=={installed}; require {required}")


def kept(char: str) -> bool:
    return char == "'" or unicodedata.category(char).startswith(("L", "N"))


def cjk(char: str) -> bool:
    code = ord(char)
    return any(
        lo <= code <= hi
        for lo, hi in (
            (0x4E00, 0x9FFF),
            (0x3400, 0x4DBF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B820, 0x2CEAF),
            (0xF900, 0xFAFF),
        )
    )


def text_units(text: str, language: str, root: Path) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise ValueError("alignment text must be a string")
    if language == "Japanese":
        tokenizer_version("nagisa", "0.2.11")
        try:
            import nagisa
        except ImportError as exc:
            raise ValueError(
                "Japanese alignment requires the optional nagisa==0.2.11 tokenizer"
            ) from exc
        return tuple(
            clean
            for word in nagisa.tagging(text).words
            if (clean := "".join(ch for ch in word if kept(ch)))
        )
    if language == "Korean":
        asset = root / "korean_dict_jieba.dict"
        if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != KOREAN_SHA256:
            raise ValueError(
                "Korean alignment requires the pinned korean_dict_jieba.dict supplemental asset"
            )
        tokenizer_version("soynlp", "0.0.493")
        try:
            from soynlp.tokenizer import LTokenizer
        except ImportError as exc:
            raise ValueError(
                "Korean alignment requires the optional soynlp==0.0.493 tokenizer"
            ) from exc
        scores = {line.split()[0]: 1.0 for line in asset.read_text().splitlines() if line.strip()}
        return tuple(
            clean
            for word in LTokenizer(scores=scores).tokenize(text)
            if (clean := "".join(ch for ch in word if kept(ch)))
        )
    # The reference uses whitespace splitting even for Chinese/Cantonese,
    # strips punctuation first, then emits each ideograph as its own unit.
    result = []
    for word in text.split():
        buffer = ""
        for char in (ch for ch in word if kept(ch)):
            if cjk(char):
                if buffer:
                    result.append(buffer)
                    buffer = ""
                result.append(char)
            else:
                buffer += char
        if buffer:
            result.append(buffer)
    return tuple(result)


def fix_timestamps(data: list[int]) -> list[int]:
    """Reference longest nondecreasing subsequence and nearest/interpolated repair."""
    if not data:
        return []
    count = len(data)
    dp, parent = [1] * count, [-1] * count
    for i in range(1, count):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i], parent[i] = dp[j] + 1, j
    normal = set()
    index = dp.index(max(dp))
    while index != -1:
        normal.add(index)
        index = parent[index]
    result = list(data)
    i = 0
    while i < count:
        if i in normal:
            i += 1
            continue
        j = i
        while j < count and j not in normal:
            j += 1
        left = result[i - 1] if i else None
        right = result[j] if j < count else None
        for k in range(i, j):
            if left is None:
                result[k] = right
            elif right is None:
                result[k] = left
            elif j - i <= 2:
                result[k] = left if k - (i - 1) <= j - k else right
            else:
                result[k] = left + (right - left) / (j - i + 1) * (k - i + 1)
        i = j
    return [int(value) for value in result]
