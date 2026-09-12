"""Native greedy, sampled, batched, cancellable, and streaming text generation."""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from mlx_one.text.loading import LoadedTextModel, load_text_model
from mlx_one.text.sampling import select_token
from mlx_one.text.schemas import GenerationChunk, GenerationResult, TextGenerationOptions

CancelCheck = Callable[[], bool]
CacheCallback = Callable[[tuple[int, ...], tuple[object, ...]], None]


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
    template_options: Mapping[str, object] | None = None,
    context_length: int | None = None,
    cache: tuple[object, ...] | None = None,
    cached_prompt_ids: Sequence[int] = (),
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    on_cache_update: CacheCallback | None = None,
    spec_type: str = "none",
    spec_draft_n_max: int = 3,
    speculative_stats: dict[str, int] | None = None,
    reasoning_budget: int = -1,
    yield_steps: bool = False,
) -> Iterator[GenerationChunk]:
    if model.chat_template is None:
        raise ValueError("loaded model has no chat-template layer")
    prompt = model.chat_template.render(messages, **dict(template_options or {}))
    yield from _stream_bundle(
        model,
        prompt,
        options or TextGenerationOptions(),
        is_cancelled,
        context_length=context_length,
        cache=cache,
        cached_prompt_ids=cached_prompt_ids,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        on_cache_update=on_cache_update,
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max,
        speculative_stats=speculative_stats,
        reasoning_budget=reasoning_budget,
        yield_steps=yield_steps,
    )


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
    *,
    context_length: int | None = None,
    cache: tuple[object, ...] | None = None,
    cached_prompt_ids: Sequence[int] = (),
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    on_cache_update: CacheCallback | None = None,
    spec_type: str = "none",
    spec_draft_n_max: int = 3,
    speculative_stats: dict[str, int] | None = None,
    reasoning_budget: int = -1,
    yield_steps: bool = False,
) -> Iterator[GenerationChunk]:
    import mlx.core as mx

    from mlx_one.core.cache import configure_kv_caches

    cancelled = is_cancelled or (lambda: False)
    prompt_ids = bundle.tokenizer.encode(prompt)
    if not prompt_ids:
        bos = bundle.tokenizer.bos_token_id
        if bos is None:
            raise ValueError("empty prompt requires a tokenizer BOS token")
        prompt_ids = [bos]
    context_length = context_length or bundle.context_length or _legacy_context_length(bundle)
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
    cached = tuple(int(value) for value in cached_prompt_ids)
    if cached and tuple(prompt_ids[: len(cached)]) != cached:
        raise ValueError("cached prompt does not prefix the rendered prompt")
    if cache is None:
        cache = configure_kv_caches(bundle.model.make_cache(), cache_type_k, cache_type_v)
    randomizer = random.Random(options.seed)
    tokens = list(prompt_ids)
    think_open = tuple(bundle.tokenizer.encode("<think>"))
    think_close = tuple(bundle.tokenizer.encode("</think>"))
    reasoning_active = bool(think_open and tuple(prompt_ids[-len(think_open) :]) == think_open)
    reasoning_count = 0
    forced_tokens: deque[int] = deque()
    closing_reasoning = False
    if reasoning_active and reasoning_budget == 0:
        forced_tokens.extend(think_close)
        closing_reasoning = True
    emitted = 0
    finish_reason = "length"
    eos_token_id = bundle.tokenizer.eos_token_id
    if eos_token_id is None:
        config = getattr(bundle.model.config, "text_config", bundle.model.config)
        eos_token_id = getattr(config, "eos_token_id", None)
    produced = 0
    cache_safe = True
    cache_token_count = len(cached)
    while produced < maximum:
        if cancelled():
            finish_reason = "stop"
            break
        current = tokens[len(cached) :] if produced == 0 else [tokens[-1]]
        if not current:
            raise ValueError("prompt cache must leave at least one token for prefill")
        output = bundle.model(mx.array([current]), cache=cache)
        cache_token_count += len(current)
        logits = output.logits[0, -1]
        using_forced = bool(forced_tokens)
        token = (
            forced_tokens.popleft()
            if using_forced
            else select_token(logits, options, randomizer)
        )
        candidates = [token]
        step_yielded = False
        if (
            spec_type == "draft-mtp"
            and options.temperature == 0
            and hasattr(bundle.model, "mtp")
            and produced + 1 < maximum
            and not forced_tokens
            and not (reasoning_active and reasoning_budget >= 0)
        ):
            candidates, cache, drafted, accepted = _speculative_candidates(
                bundle.model,
                cache,
                token,
                output.last_hidden_state[:, -1:],
                min(spec_draft_n_max, maximum - produced - 1),
                options,
                randomizer,
            )
            cache_safe = False
            if speculative_stats is not None:
                speculative_stats["drafted_tokens"] += drafted
                speculative_stats["accepted_draft_tokens"] += accepted
        stopped = False
        for candidate in candidates[: maximum - produced]:
            if eos_token_id is not None and candidate == eos_token_id:
                finish_reason = "stop"
                stopped = True
                break
            tokens.append(candidate)
            produced += 1
            opened = bool(
                think_open and tuple(tokens[-len(think_open) :]) == think_open
            )
            closed = bool(
                think_close and tuple(tokens[-len(think_close) :]) == think_close
            )
            if opened:
                reasoning_active = True
                reasoning_count = 0
            elif closed:
                reasoning_active = False
                closing_reasoning = False
            elif reasoning_active and not using_forced:
                reasoning_count += 1
                if (
                    reasoning_budget >= 0
                    and reasoning_count >= reasoning_budget
                    and not closing_reasoning
                ):
                    forced_tokens.extend(think_close)
                    closing_reasoning = True
            pending.append(candidate)
            matched = next(
                (
                    value
                    for value in stop_tokens
                    if len(pending) >= len(value)
                    and tuple(pending[-len(value) :]) == value
                ),
                None,
            )
            if matched is not None:
                del pending[-len(matched) :]
                finish_reason = "stop"
                stopped = True
                break
            while len(pending) >= hold:
                emitted += 1
                generated.append(pending.pop(0))
                current_text = bundle.tokenizer.decode(generated)
                piece = (
                    current_text[len(decoded) :]
                    if current_text.startswith(decoded)
                    else current_text
                )
                decoded = current_text
                step_yielded = True
                yield GenerationChunk(generated[-1], piece, emitted)
        if stopped:
            break
        if yield_steps and not step_yielded:
            yield GenerationChunk(None, "", emitted)
    for current_token in pending:
        emitted += 1
        generated.append(current_token)
        current_text = bundle.tokenizer.decode(generated)
        piece = current_text[len(decoded) :] if current_text.startswith(decoded) else current_text
        decoded = current_text
        yield GenerationChunk(current_token, piece, emitted)
    if on_cache_update is not None and tokens and cache_safe:
        on_cache_update(tuple(tokens[:cache_token_count]), cache)
    yield GenerationChunk(None, "", emitted, finish_reason)


def _speculative_candidates(
    model: object,
    cache: tuple[object, ...],
    first: int,
    hidden: object,
    maximum: int,
    options: TextGenerationOptions,
    randomizer: random.Random,
) -> tuple[list[int], tuple[object, ...], int, int]:
    import mlx.core as mx

    from mlx_one.core.cache import clone_caches

    drafts: list[int] = []
    mtp_cache = model.make_mtp_cache()
    mtp_hidden = hidden
    previous = first
    position_offset = cache[0].offset
    for _ in range(maximum):
        mtp_hidden = model.mtp(
            model.language_model.embed_tokens(mx.array([[previous]])),
            mtp_hidden,
            cache=mtp_cache,
            position_offset=position_offset,
        )
        logits = (
            model.language_model.embed_tokens.as_linear(mtp_hidden)
            if model.config.tie_word_embeddings
            else model.lm_head(mtp_hidden)
        )
        previous = select_token(logits[0, -1], options, randomizer)
        drafts.append(previous)
    verification = clone_caches(cache)
    verified = model(mx.array([[first, *drafts]]), cache=verification)
    accepted: list[int] = []
    for index, draft in enumerate(drafts):
        actual = select_token(verified.logits[0, index], options, randomizer)
        if actual != draft:
            corrected = [first, *accepted, actual]
            committed = clone_caches(cache)
            model(mx.array([corrected[:-1]]), cache=committed)
            return corrected, committed, len(drafts), len(accepted)
        accepted.append(draft)
    bonus = select_token(verified.logits[0, len(drafts)], options, randomizer)
    return [first, *drafts, bonus], verification, len(drafts), len(drafts)


def _legacy_context_length(bundle: LoadedTextModel) -> int:
    config = getattr(bundle.model.config, "text_config", bundle.model.config)
    for name in ("max_position_embeddings", "max_context_length", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("model configuration does not declare a context length")
