"""Native greedy, sampled, batched, cancellable, and streaming text generation."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from mlx_one.text.loading import LoadedTextModel, load_text_model
from mlx_one.text.sampling import select_token
from mlx_one.text.schemas import GenerationChunk, GenerationResult, TextGenerationOptions

CancelCheck = Callable[[], bool]


def stream_generate(
    model: str | Path | LoadedTextModel,
    prompt: str,
    *,
    options: TextGenerationOptions | None = None,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    tokenizer_source: str | Path | None = None,
    is_cancelled: CancelCheck | None = None,
) -> Iterator[GenerationChunk]:
    """Yield native generation chunks for one prompt."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty text")
    bundle = (
        model
        if isinstance(model, LoadedTextModel)
        else load_text_model(
            model,
            revision=revision,
            offline=offline,
            cache_dir=cache_dir,
            tokenizer_source=tokenizer_source,
        )
    )
    yield from _stream_bundle(bundle, prompt, options or TextGenerationOptions(), is_cancelled)


def stream_chat(
    model: LoadedTextModel,
    messages: Sequence[Mapping[str, str]],
    *,
    options: TextGenerationOptions | None = None,
    is_cancelled: CancelCheck | None = None,
) -> Iterator[GenerationChunk]:
    if model.chat_template is None:
        raise ValueError("loaded model has no chat-template layer")
    prompt = model.chat_template.render(messages)
    yield from _stream_bundle(model, prompt, options or TextGenerationOptions(), is_cancelled)


def generate(
    model: str | Path | LoadedTextModel,
    prompt_or_prompts: str | Sequence[str],
    *,
    options: TextGenerationOptions | None = None,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    tokenizer_source: str | Path | None = None,
) -> GenerationResult | tuple[GenerationResult, ...]:
    """Generate one completion or an independent batch of completions."""

    bundle = (
        model
        if isinstance(model, LoadedTextModel)
        else load_text_model(
            model,
            revision=revision,
            offline=offline,
            cache_dir=cache_dir,
            tokenizer_source=tokenizer_source,
        )
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
    chunks = tuple(_stream_bundle(bundle, prompt, options, None))
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
    bundle: LoadedTextModel,
    prompt: str,
    options: TextGenerationOptions,
    is_cancelled: CancelCheck | None,
) -> Iterator[GenerationChunk]:
    import mlx.core as mx

    cancelled = is_cancelled or (lambda: False)
    prompt_ids = bundle.tokenizer.encode(prompt)
    if not prompt_ids:
        bos = bundle.tokenizer.bos_token_id
        if bos is None:
            raise ValueError("empty prompt requires a tokenizer BOS token")
        prompt_ids = [bos]
    context_length = bundle.context_length or _legacy_context_length(bundle)
    if len(prompt_ids) >= context_length:
        raise ValueError("prompt leaves no room inside the model context window")
    maximum = min(options.max_tokens, context_length - len(prompt_ids))
    stop_tokens = tuple(tuple(bundle.tokenizer.encode(value)) for value in options.stop)
    if any(not value for value in stop_tokens):
        raise ValueError("stop sequences must encode to at least one token")
    pending: list[int] = []
    generated: list[int] = []
    decoded = ""
    hold = max((len(value) for value in stop_tokens), default=1)
    cache = bundle.model.make_cache()
    randomizer = random.Random(options.seed)
    tokens = list(prompt_ids)
    emitted = 0
    finish_reason = "length"
    eos_token_id = bundle.tokenizer.eos_token_id
    if eos_token_id is None:
        config = getattr(bundle.model.config, "text_config", bundle.model.config)
        eos_token_id = getattr(config, "eos_token_id", None)
    for step in range(maximum):
        if cancelled():
            finish_reason = "stop"
            break
        current = tokens if step == 0 else [tokens[-1]]
        output = bundle.model(mx.array([current]), cache=cache)
        logits = output.logits[0, -1]
        token = select_token(logits, options, randomizer)
        if eos_token_id is not None and token == eos_token_id:
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
            generated.append(pending.pop(0))
            current_text = bundle.tokenizer.decode(generated)
            piece = (
                current_text[len(decoded) :] if current_text.startswith(decoded) else current_text
            )
            decoded = current_text
            yield GenerationChunk(generated[-1], piece, emitted)
    for current_token in pending:
        emitted += 1
        generated.append(current_token)
        current_text = bundle.tokenizer.decode(generated)
        piece = current_text[len(decoded) :] if current_text.startswith(decoded) else current_text
        decoded = current_text
        yield GenerationChunk(current_token, piece, emitted)
    yield GenerationChunk(None, "", emitted, finish_reason)


def _legacy_context_length(bundle: LoadedTextModel) -> int:
    config = getattr(bundle.model.config, "text_config", bundle.model.config)
    for name in ("max_position_embeddings", "max_context_length", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("model configuration does not declare a context length")
