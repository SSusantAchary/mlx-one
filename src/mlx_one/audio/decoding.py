"""OpenAI-Whisper-style decoding, timestamps, and word alignment."""

from __future__ import annotations

import math
import random
import re
import zlib
from dataclasses import dataclass
from typing import Any

from mlx_one.audio.schemas import TranscriptionWord, WhisperDecodeOptions


@dataclass(frozen=True)
class DecodingResult:
    tokens: tuple[int, ...]
    text: str
    language: str
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    cross_attentions: tuple[Any, ...] | None = None
    token_logprobs: tuple[float, ...] = ()


def compression_ratio(text: str) -> float:
    encoded = text.encode("utf-8")
    return len(encoded) / len(zlib.compress(encoded)) if encoded else 0.0


def language_tokens(tokenizer: Any) -> dict[str, int]:
    result = {}
    for token, token_id in tokenizer.special_tokens.items():
        match = re.fullmatch(r"<\|([a-z]{2,3})\|>", token)
        if match and match.group(1) not in {"sot", "eot"}:
            result[match.group(1)] = token_id
    return result


def detect_language(
    model: Any, audio_features: Any, tokenizer: Any
) -> tuple[str, dict[str, float]]:
    import mlx.core as mx

    languages = language_tokens(tokenizer)
    if not languages:
        return "en", {"en": 1.0}
    sot = tokenizer.token_id("<|startoftranscript|>")
    output = model(None, mx.array([[sot]]), encoder_outputs=audio_features)
    logits = output.logits[0, -1]
    ids = list(languages.values())
    probabilities = mx.softmax(logits[mx.array(ids)], axis=-1)
    mx.eval(probabilities)
    scores = {
        language: float(probabilities[index].item())
        for index, language in enumerate(languages)
    }
    detected = max(scores, key=scores.__getitem__)
    return detected, scores


def initial_tokens(
    tokenizer: Any,
    *,
    language: str,
    task: str,
    without_timestamps: bool,
    prompt_tokens: list[int] | None = None,
) -> list[int]:
    tokens = [tokenizer.token_id("<|startoftranscript|>")]
    language_token = f"<|{language}|>"
    if language_token in tokenizer.special_tokens:
        tokens.append(tokenizer.token_id(language_token))
    tokens.append(tokenizer.token_id(f"<|{task}|>"))
    if without_timestamps:
        tokens.append(tokenizer.token_id("<|notimestamps|>"))
    if prompt_tokens:
        sot_prev = tokenizer.special_tokens.get("<|startofprev|>")
        tokens = ([sot_prev] if sot_prev is not None else []) + prompt_tokens + tokens
    return tokens


def decode_with_fallback(
    model: Any,
    audio_features: Any,
    tokenizer: Any,
    *,
    language: str,
    task: str,
    word_timestamps: bool,
    options: WhisperDecodeOptions,
    prompt_tokens: list[int] | None = None,
) -> DecodingResult:
    last: DecodingResult | None = None
    for index, temperature in enumerate(options.temperatures):
        result = decode_segment(
            model,
            audio_features,
            tokenizer,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            options=options,
            temperature=temperature,
            prompt_tokens=prompt_tokens,
            seed=options.seed + index,
        )
        last = result
        needs_fallback = False
        if options.compression_ratio_threshold is not None:
            needs_fallback |= result.compression_ratio > options.compression_ratio_threshold
        if options.logprob_threshold is not None:
            needs_fallback |= result.avg_logprob < options.logprob_threshold
        silent = (
            options.no_speech_threshold is not None
            and result.no_speech_prob > options.no_speech_threshold
            and (
                options.logprob_threshold is None
                or result.avg_logprob < options.logprob_threshold
            )
        )
        if not needs_fallback or silent:
            return result
    assert last is not None
    return last


def decode_segment(
    model: Any,
    audio_features: Any,
    tokenizer: Any,
    *,
    language: str,
    task: str,
    word_timestamps: bool,
    options: WhisperDecodeOptions,
    temperature: float,
    prompt_tokens: list[int] | None,
    seed: int,
) -> DecodingResult:
    import mlx.core as mx

    prefix = initial_tokens(
        tokenizer,
        language=language,
        task=task,
        without_timestamps=False,
        prompt_tokens=prompt_tokens,
    )
    eos = tokenizer.token_id("<|endoftext|>")
    maximum = options.max_tokens or max(1, model.config.max_target_positions - len(prefix))
    if temperature == 0 and options.beam_size and options.beam_size > 1:
        tokens, logprobs, no_speech = _beam_search(
            model, audio_features, prefix, eos, tokenizer, options, maximum
        )
    else:
        candidates = options.best_of if temperature > 0 and options.best_of else 1
        samples = [
            _sample_sequence(
                model,
                audio_features,
                prefix,
                eos,
                tokenizer,
                options,
                maximum,
                temperature,
                seed + candidate,
            )
            for candidate in range(candidates)
        ]
        tokens, logprobs, no_speech = max(
            samples, key=lambda item: sum(item[1]) / max(len(item[1]), 1)
        )
    generated = tokens[len(prefix) :]
    generated_logprobs = logprobs
    if generated and generated[-1] == eos:
        generated = generated[:-1]
        generated_logprobs = generated_logprobs[:-1]
    text = tokenizer.decode(generated).strip()
    final_attention = None
    if word_timestamps and tokens:
        output = model(
            None,
            mx.array([tokens]),
            encoder_outputs=audio_features,
            output_attentions=True,
        )
        final_attention = output.cross_attentions
        mx.eval(*final_attention)
    average = (sum(logprobs) / (len(logprobs) + 1)) if logprobs else -float("inf")
    return DecodingResult(
        tuple(generated),
        text,
        language,
        temperature,
        average,
        compression_ratio(text),
        no_speech,
        final_attention,
        tuple(generated_logprobs),
    )


def _sample_sequence(
    model: Any,
    audio: Any,
    prefix: list[int],
    eos: int,
    tokenizer: Any,
    options: WhisperDecodeOptions,
    maximum: int,
    temperature: float,
    seed: int,
) -> tuple[list[int], list[float], float]:
    import mlx.core as mx

    rng = random.Random(seed)
    tokens = list(prefix)
    logprobs: list[float] = []
    no_speech = 0.0
    cache = model.make_cache()
    for step in range(maximum):
        decoder_input = tokens if step == 0 else [tokens[-1]]
        output = model(
            None,
            mx.array([decoder_input]),
            encoder_outputs=audio,
            cache=cache,
            use_cache=True,
        )
        logits = output.logits[0, -1]
        logits = apply_logit_filters(logits, tokens, tokenizer, options, step)
        if step == 0 and "<|nospeech|>" in tokenizer.special_tokens:
            no_speech = float(mx.softmax(logits)[tokenizer.token_id("<|nospeech|>")].item())
        scaled = logits if temperature == 0 else logits / temperature
        probabilities = mx.softmax(scaled, axis=-1)
        mx.eval(probabilities)
        if temperature == 0:
            token = int(mx.argmax(probabilities).item())
        else:
            values = probabilities.tolist()
            token = rng.choices(range(len(values)), weights=values, k=1)[0]
        probability = max(float(probabilities[token].item()), 1e-20)
        logprobs.append(math.log(probability))
        tokens.append(token)
        if token == eos:
            break
    return tokens, logprobs, no_speech


def _beam_search(
    model: Any,
    audio: Any,
    prefix: list[int],
    eos: int,
    tokenizer: Any,
    options: WhisperDecodeOptions,
    maximum: int,
) -> tuple[list[int], list[float], float]:
    import mlx.core as mx

    width = options.beam_size or 1
    beams: list[tuple[list[int], list[float]]] = [(list(prefix), [])]
    finished: list[tuple[list[int], list[float]]] = []
    no_speech = 0.0
    for step in range(maximum):
        candidates: list[tuple[list[int], list[float]]] = []
        for tokens, scores in beams:
            output = model(None, mx.array([tokens]), encoder_outputs=audio)
            logits = apply_logit_filters(output.logits[0, -1], tokens, tokenizer, options, step)
            probabilities = mx.softmax(logits, axis=-1)
            if step == 0 and "<|nospeech|>" in tokenizer.special_tokens:
                no_speech = float(probabilities[tokenizer.token_id("<|nospeech|>")].item())
            indices = mx.argpartition(
                -probabilities, kth=min(width, probabilities.shape[0] - 1)
            )[:width]
            mx.eval(probabilities, indices)
            for token in indices.tolist():
                probability = max(float(probabilities[token].item()), 1e-20)
                candidate = (tokens + [int(token)], scores + [math.log(probability)])
                (finished if token == eos else candidates).append(candidate)
        if not candidates:
            break
        beams = sorted(candidates, key=lambda item: _rank(item, options), reverse=True)[:width]
        if len(finished) >= math.ceil(width * options.patience):
            break
    pool = finished or beams
    return (*max(pool, key=lambda item: _rank(item, options)), no_speech)


def _rank(item: tuple[list[int], list[float]], options: WhisperDecodeOptions) -> float:
    _, scores = item
    total = sum(scores)
    if options.length_penalty is None:
        return total / max(len(scores), 1)
    return total / (((5 + len(scores)) / 6) ** options.length_penalty)


def apply_logit_filters(
    logits: Any,
    tokens: list[int],
    tokenizer: Any,
    options: WhisperDecodeOptions,
    step: int,
) -> Any:
    import mlx.core as mx

    suppressed = set(options.suppress_tokens or ())
    if options.suppress_tokens is None:
        suppressed.update(getattr(options, "config_suppress_tokens", ()) or ())
    if step == 0 and options.suppress_blank:
        suppressed.add(tokenizer.encoder.get(" ", -1))
    valid = [value for value in suppressed if 0 <= value < logits.shape[0]]
    if valid:
        logits = logits.at[mx.array(valid)].add(-float("inf"))
    timestamp_begin = tokenizer.special_tokens.get("<|0.00|>")
    if timestamp_begin is not None:
        timestamps = [token for token in tokens if token >= timestamp_begin]
        if len(timestamps) >= 2 and timestamps[-1] < timestamps[-2]:
            logits = logits.at[:timestamp_begin].add(-float("inf"))
        if step == 0 and options.max_initial_timestamp is not None:
            maximum = timestamp_begin + round(options.max_initial_timestamp / 0.02)
            if maximum + 1 < logits.shape[0]:
                logits = logits.at[maximum + 1 :].add(-float("inf"))
    return logits


def timestamp_segments(
    tokens: tuple[int, ...], tokenizer: Any
) -> list[tuple[float, float, tuple[int, ...]]]:
    timestamp_begin = tokenizer.special_tokens.get("<|0.00|>")
    if timestamp_begin is None:
        return [(0.0, 30.0, tokens)] if tokens else []
    result: list[tuple[float, float, tuple[int, ...]]] = []
    start: float | None = None
    text_tokens: list[int] = []
    for token in tokens:
        if token >= timestamp_begin:
            value = (token - timestamp_begin) * 0.02
            if start is None:
                start = value
            else:
                result.append((start, value, tuple(text_tokens)))
                start = value
                text_tokens = []
        else:
            text_tokens.append(token)
    if text_tokens:
        result.append((start or 0.0, 30.0, tuple(text_tokens)))
    return result


def median_filter(values: list[list[float]], width: int) -> list[list[float]]:
    if width < 1 or width % 2 == 0:
        raise ValueError("median filter width must be a positive odd number")
    radius = width // 2
    result = []
    for row in values:
        filtered = []
        for index in range(len(row)):
            window = row[max(0, index - radius) : min(len(row), index + radius + 1)]
            filtered.append(sorted(window)[len(window) // 2])
        result.append(filtered)
    return result


def dynamic_time_warp(cost: list[list[float]]) -> tuple[list[int], list[int]]:
    if not cost or not cost[0]:
        return [], []
    rows, columns = len(cost), len(cost[0])
    matrix = [[float("inf")] * (columns + 1) for _ in range(rows + 1)]
    trace = [[0] * (columns + 1) for _ in range(rows + 1)]
    matrix[0][0] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            choices = (
                matrix[row - 1][column - 1],
                matrix[row - 1][column],
                matrix[row][column - 1],
            )
            direction = min(range(3), key=choices.__getitem__)
            matrix[row][column] = cost[row - 1][column - 1] + choices[direction]
            trace[row][column] = direction
    text, time = [], []
    row, column = rows, columns
    while row > 0 and column > 0:
        text.append(row - 1)
        time.append(column - 1)
        direction = trace[row][column]
        if direction != 2:
            row -= 1
        if direction != 1:
            column -= 1
    return text[::-1], time[::-1]


def align_words(
    token_ids: tuple[int, ...],
    cross_attentions: tuple[Any, ...] | None,
    tokenizer: Any,
    *,
    alignment_heads: list[list[int]] | None,
    median_width: int,
    time_offset: float = 0.0,
    token_logprobs: tuple[float, ...] = (),
) -> tuple[TranscriptionWord, ...]:
    if not token_ids or not cross_attentions:
        return ()
    import mlx.core as mx

    heads = alignment_heads or [
        [layer, head]
        for layer in range(len(cross_attentions) // 2, len(cross_attentions))
        for head in range(cross_attentions[layer].shape[1])
    ]
    matrices = [cross_attentions[layer][0, head, -len(token_ids) :, :] for layer, head in heads]
    weights = mx.mean(mx.stack(matrices), axis=0)
    centered = weights - mx.mean(weights, axis=-1, keepdims=True)
    weights = centered / (
        mx.sqrt(mx.mean(centered**2, axis=-1, keepdims=True)) + 1e-6
    )
    mx.eval(weights)
    filtered = median_filter(weights.tolist(), median_width)
    text_path, time_path = dynamic_time_warp([[-value for value in row] for row in filtered])
    words, groups = tokenizer.split_to_word_tokens(token_ids)
    boundaries = []
    probabilities = []
    consumed = 0
    for group in groups:
        group_start = consumed
        consumed += len(group)
        matching = [
            time
            for text, time in zip(text_path, time_path, strict=True)
            if text < consumed
        ]
        boundaries.append(max(matching, default=0))
        scores = token_logprobs[group_start:consumed]
        probabilities.append(
            math.exp(sum(scores) / len(scores)) if scores else 0.0
        )
    result = []
    previous = 0
    for word, boundary, probability in zip(
        words, boundaries, probabilities, strict=True
    ):
        result.append(
            TranscriptionWord(
                word=word,
                start=time_offset + previous * 0.02,
                end=time_offset + max(previous, boundary) * 0.02,
                probability=probability,
            )
        )
        previous = max(previous, boundary)
    return tuple(result)
