"""Qwen-specific tokenizer, chat prompts, and validated 128-Mel preprocessing.

Processing behavior adapted from QwenLM/Qwen3-ASR (Apache-2.0); see NOTICE.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from mlx_one.models.audio.qwen3_asr.config import QwenASRConfig, audio_token_length

LANGUAGES = frozenset(
    (
        "Chinese English Cantonese Arabic German French Spanish Portuguese Indonesian Italian "
        "Korean Russian Thai Vietnamese Japanese Turkish Hindi Malay Dutch Swedish Danish Finnish "
        "Polish Czech Filipino Persian Greek Romanian Hungarian Macedonian"
    ).split()
)
PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def language_name(value: str | None, supported: Any = LANGUAGES) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Qwen language must be a supported language name")
    name = value.strip().capitalize()
    if name not in supported:
        raise ValueError(
            f"unsupported Qwen language: {value}; "
            "use names such as English, French, German, Chinese"
        )
    return name


def read_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path.name} contains duplicate key: {key}")
            result[key] = value
        return result

    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be an object")
    return data


def tokenizer_from_directory(root: Path, config: QwenASRConfig) -> Any:
    from tokenizers import (
        AddedToken,
        Regex,
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
    )

    metadata = read_json(root / "tokenizer_config.json")
    if metadata.get("tokenizer_class") != "Qwen2Tokenizer" or metadata.get(
        "add_prefix_space", False
    ):
        raise ValueError("unsupported Qwen audio tokenizer")
    vocab = read_json(root / "vocab.json")
    if set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("Qwen base vocabulary must have unique contiguous IDs")
    merges = []
    for line in (root / "merges.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("invalid Qwen BPE merge")
        merges.append(tuple(parts))
    tokenizer = Tokenizer(models.BPE(vocab, merges))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    added = metadata.get("added_tokens_decoder")
    if not isinstance(added, dict):
        raise ValueError("Qwen added-token asset is missing")
    for token_id, record in sorted(added.items(), key=lambda item: int(item[0])):
        token = AddedToken(**record)
        if record.get("special", False):
            tokenizer.add_special_tokens([token])
        else:
            tokenizer.add_tokens([token])
        if tokenizer.token_to_id(record["content"]) != int(token_id):
            raise ValueError("Qwen added token IDs do not match checkpoint assets")
    if max(tokenizer.get_vocab().values()) >= config.text_config.vocab_size:
        raise ValueError("Qwen tokenizer exceeds the model vocabulary")
    for name, value in (
        ("<|audio_pad|>", config.audio_token_id),
        ("<|audio_start|>", config.audio_start_token_id),
        ("<|audio_end|>", config.audio_end_token_id),
    ):
        if tokenizer.token_to_id(name) != value:
            raise ValueError("Qwen audio token IDs disagree with configuration")
    if (
        config.model_type == "qwen3_forced_aligner"
        and tokenizer.token_to_id("<timestamp>") != config.timestamp_token_id
    ):
        raise ValueError("timestamp token ID disagrees with configuration")
    return tokenizer


def validate_preprocessor(data: dict[str, Any]) -> None:
    expected = {
        "feature_extractor_type": "WhisperFeatureExtractor",
        "processor_class": "Qwen3ASRProcessor",
        "feature_size": 128,
        "hop_length": 160,
        "n_fft": 400,
        "chunk_length": 30,
        "dither": 0.0,
        "padding_value": 0.0,
        "padding_side": "right",
    }
    if (
        any(data.get(k) != v for k, v in expected.items())
        or data.get("sampling_rate", 16000) != 16000
    ):
        raise ValueError("unsupported Qwen audio preprocessing contract")
    if data.get("do_normalize", False):
        raise ValueError("Qwen audio must not use waveform variance normalization")


def preprocess(waveform: Any, *, offline_padding: bool) -> tuple[Any, int]:
    import mlx.core as mx

    from mlx_one.models.audio.whisper.processing import mel_filterbank

    # Qwen explicitly ships WhisperFeatureExtractor, but uses dynamic longest
    # padding, no 30-second trimming, and a family-specific convolution length.
    if offline_padding and waveform.shape[0] < 8000:
        waveform = mx.pad(waveform, ((0, 8000 - waveform.shape[0]),))
    if waveform.shape[0] <= 200:
        raise ValueError(
            "nonempty Qwen streaming tails require more than 200 total "
            "utterance samples for reflect STFT"
        )
    # Preserve the reference hann_window operation order: one rounded angular
    # increment, rather than multiplying 2*pi and dividing each GPU position.
    signal = mx.pad(waveform.astype(mx.float32), ((200, 200),), mode="reflect")
    frames = mx.stack(
        [signal[i * 160 : i * 160 + 400] for i in range(1 + (signal.shape[0] - 400) // 160)]
    )
    window = 0.5 - 0.5 * mx.cos(mx.arange(400, dtype=mx.float32) * (2 * math.pi / 400))
    powers = mx.abs(mx.fft.rfft(frames * window, axis=-1)) ** 2
    spectrum = mx.log10(mx.maximum(powers[:-1] @ mel_filterbank(128).T, 1e-10))
    mel = ((mx.maximum(spectrum, mx.max(spectrum) - 8.0) + 4.0) / 4.0).T
    return mel, audio_token_length(mel.shape[-1])


def render_prompt(
    template: str, context: str, language: str | None, count: int, prefix: str = ""
) -> str:
    from jinja2.sandbox import SandboxedEnvironment

    # Render checkpoint assets verbatim, including their literal backslash-ns.
    rendered = (
        SandboxedEnvironment()
        .from_string(template)
        .render(
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ],
            add_generation_prompt=True,
        )
    )
    if rendered.count("<|audio_pad|>") != 1:
        raise ValueError("unsupported Qwen checkpoint chat template")
    rendered = rendered.replace("<|audio_pad|>", "<|audio_pad|>" * count)
    return rendered + (f"language {language}<asr_text>" if language else "") + prefix


def fix_repetitions(text: str, threshold: int = 20) -> str:
    # Match the pinned reference's character then pattern repetition cleanup.
    cleaned, i = [], 0
    while i < len(text):
        end = i + 1
        while end < len(text) and text[end] == text[i]:
            end += 1
        cleaned.append(text[i] if end - i > threshold else text[i:end])
        i = end
    text = "".join(cleaned)
    i = 0
    while i <= len(text) - threshold * 2:
        for width in range(1, 21):
            if i + width * threshold > len(text):
                break
            pattern = text[i : i + width]
            if text[i : i + width * threshold] == pattern * threshold:
                end = i + width * threshold
                while text[end : end + width] == pattern:
                    end += width
                return text[:i] + pattern + fix_repetitions(text[end:], threshold)
        i += 1
    return text


def parse_output(raw: str, language: str | None) -> tuple[str | None, str]:
    value = fix_repetitions(raw.strip())
    if not value:
        return None, ""
    if language:
        return language, value
    if "<asr_text>" not in value:
        return None, value
    meta, text = value.split("<asr_text>", 1)
    if "language none" in meta.lower():
        return None, text.strip()
    for line in meta.splitlines():
        if line.strip().lower().startswith("language "):
            name = line.strip()[9:].strip().capitalize()
            return name or None, text.strip()
    return None, text.strip()
