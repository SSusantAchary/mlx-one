"""Native greedy, sampled, batched, and streaming text generation."""

from __future__ import annotations

import codecs
import math
import random
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

from mlx_one.text.loading import LoadedTextModel, load_text_model
from mlx_one.text.schemas import GenerationChunk, GenerationResult, TextGenerationOptions


def stream_generate(
    model: str | Path | LoadedTextModel,
    prompt: str,
    *,
    options: TextGenerationOptions | None = None,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> Iterator[GenerationChunk]:
    """Yield native generation chunks for one prompt."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty text")
    bundle = (
        model
        if isinstance(model, LoadedTextModel)
        else load_text_model(model, revision=revision, offline=offline, cache_dir=cache_dir)
    )
    yield from _stream_bundle(bundle, prompt, options or TextGenerationOptions())


def generate(
    model: str | Path | LoadedTextModel,
    prompt_or_prompts: str | Sequence[str],
    *,
    options: TextGenerationOptions | None = None,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> GenerationResult | tuple[GenerationResult, ...]:
    """Generate one completion or an independent batch of completions."""

    bundle = (
        model
        if isinstance(model, LoadedTextModel)
        else load_text_model(model, revision=revision, offline=offline, cache_dir=cache_dir)
    )
    if isinstance(prompt_or_prompts, str):
        return _generate_one(bundle, prompt_or_prompts, options or TextGenerationOptions())
    prompts = tuple(prompt_or_prompts)
    if not prompts:
        raise ValueError("prompt batch cannot be empty")
    return tuple(
        _generate_one(bundle, prompt, options or TextGenerationOptions()) for prompt in prompts
    )


def _generate_one(
    bundle: LoadedTextModel, prompt: str, options: TextGenerationOptions
) -> GenerationResult:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty text")
    started = time.perf_counter()
    prompt_ids = bundle.tokenizer.encode(prompt)
    chunks = tuple(_stream_bundle(bundle, prompt, options))
    terminal = chunks[-1]
    content = tuple(chunk for chunk in chunks if chunk.token_id is not None)
    return GenerationResult(
        prompt=prompt,
        text="".join(chunk.text for chunk in chunks),
        prompt_tokens=len(prompt_ids) or 1,
        generation_tokens=len(content),
        finish_reason=terminal.finish_reason or "length",
        model=bundle.model_id,
        revision=bundle.revision,
        timings={"wall_seconds": time.perf_counter() - started},
    )


def _stream_bundle(
    bundle: LoadedTextModel, prompt: str, options: TextGenerationOptions
) -> Iterator[GenerationChunk]:
    import mlx.core as mx

    prompt_ids = bundle.tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [bundle.tokenizer.bos_token_id]
    if len(prompt_ids) >= bundle.model.config.n_positions:
        raise ValueError("prompt leaves no room inside the GPT-2 context window")
    maximum = min(options.max_tokens, bundle.model.config.n_positions - len(prompt_ids))
    stop_tokens = tuple(tuple(bundle.tokenizer.encode(value)) for value in options.stop)
    if any(not value for value in stop_tokens):
        raise ValueError("stop sequences must encode to at least one token")
    pending: list[int] = []
    hold = max((len(value) for value in stop_tokens), default=1)
    cache = bundle.model.make_cache()
    randomizer = random.Random(options.seed)
    tokens = list(prompt_ids)
    emitted = 0
    finish_reason = "length"
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    for step in range(maximum):
        current = tokens if step == 0 else [tokens[-1]]
        output = bundle.model(mx.array([current]), cache=cache)
        logits = output.logits[0, -1]
        token = _select_token(logits, options, randomizer)
        if token == bundle.tokenizer.eos_token_id:
            finish_reason = "stop"
            break
        tokens.append(token)
        pending.append(token)
        matched = next(
            (
                value
                for value in stop_tokens
                if len(pending) >= len(value) and tuple(pending[-len(value) :]) == value
            ),
            None,
        )
        if matched is not None:
            del pending[-len(matched) :]
            finish_reason = "stop"
            break
        while len(pending) >= hold:
            emitted += 1
            current_token = pending.pop(0)
            text = decoder.decode(bundle.tokenizer.token_bytes(current_token), final=False)
            yield GenerationChunk(current_token, text, emitted)
    for current_token in pending:
        emitted += 1
        text = decoder.decode(bundle.tokenizer.token_bytes(current_token), final=False)
        yield GenerationChunk(current_token, text, emitted)
    trailing = decoder.decode(b"", final=True)
    if trailing:
        yield GenerationChunk(None, trailing, emitted)
    yield GenerationChunk(None, "", emitted, finish_reason)


def _select_token(logits: object, options: TextGenerationOptions, randomizer: random.Random) -> int:
    import mlx.core as mx

    if options.temperature == 0:
        return int(mx.argmax(logits).item())
    probabilities = mx.softmax(logits / options.temperature, axis=-1)
    mx.eval(probabilities)
    ranked = sorted(enumerate(probabilities.tolist()), key=lambda item: item[1], reverse=True)
    if options.top_k is not None:
        ranked = ranked[: options.top_k]
    if options.top_p < 1.0:
        cutoff = options.top_p * sum(probability for _, probability in ranked)
        selected = []
        cumulative = 0.0
        for item in ranked:
            selected.append(item)
            cumulative += item[1]
            if cumulative >= cutoff:
                break
        ranked = selected
    total = sum(value for _, value in ranked)
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("sampling probabilities are invalid")
    threshold = randomizer.random() * total
    cumulative = 0.0
    for token, probability in ranked:
        cumulative += probability
        if threshold <= cumulative:
            return int(token)
    return int(ranked[-1][0])
