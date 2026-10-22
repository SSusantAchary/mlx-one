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


def _qualification_eos_ids(eos_ids, *, ignore_eos=False, runtime=None):
    if not ignore_eos:
        return eos_ids
    if not getattr(runtime, "harness_only", False):
        raise ValueError("fixed output EOS policy requires the qualification harness")
    return set()


def _decode_piece(tokenizer, token_ids, committed_text, *, final=False, harmony_prompt=None):
    """Commit only stable text; incomplete byte sequences can revise trailing U+FFFD."""
    if harmony_prompt is not None:
        from mlx_one.text.harmony import final_text

        text = final_text(tokenizer, token_ids, prompt=harmony_prompt, final=final)
    else:
        text = tokenizer.decode(token_ids)
    stable = text if final else text.rstrip("\ufffd")
    if not stable.startswith(committed_text):
        raise RuntimeError("tokenizer revised previously committed streaming text")
    return stable[len(committed_text):], stable


def stream_generate(
    model: str | Path | LoadedTextModel,
    prompt: str,
    *,
    options: TextGenerationOptions | None = None,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    tokenizer_source: str | Path | None = None,
    adapter_path: str | Path | None = None,
    is_cancelled: CancelCheck | None = None,
    _cache_manager: object | None = None,
    _cache_namespace: str | None = None,
    _cooperative: bool = False,
    _prefill_chunk_size: int = 512,
    _spec_type: str = "none",
    _spec_draft_n_max: int = 3,
    _draft_bundle: object | None = None,
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
            adapter_path=adapter_path,
        )
    )
    resolved = options or TextGenerationOptions()
    if _cache_manager is None:
        yield from _stream_bundle(bundle, prompt, resolved, is_cancelled,
            spec_type=_spec_type,spec_draft_n_max=_spec_draft_n_max,_draft_bundle=_draft_bundle)
        return
    from mlx_one.engine.cache_errors import PrefixReuseUnsupported
    from mlx_one.engine.cache_manager import CacheManager

    if not isinstance(_cache_manager, CacheManager):
        raise TypeError("_cache_manager must be a CacheManager")
    token_ids = tuple(int(value) for value in bundle.tokenizer.encode(prompt))
    context_limit = bundle.context_length or _legacy_context_length(bundle)
    effective_prompt_tokens = len(token_ids) or 1
    if effective_prompt_tokens >= context_limit:
        raise ValueError("prompt leaves no room inside the model context window")
    if not token_ids and bundle.tokenizer.bos_token_id is None:
        raise ValueError("empty prompt requires a tokenizer BOS token")
    output_budget = min(resolved.max_tokens, context_limit - effective_prompt_tokens)
    if (_cache_manager.config.backend in {"paged", "hybrid"}
            or getattr(_cache_manager, "requested_mode", None) == "auto"):
        from mlx_one.engine.selection import validate_context_admission

        validate_context_admission(_cache_manager, len(token_ids), resolved.max_tokens)
    key_type, value_type = _cache_manager.config.key_type, _cache_manager.config.value_type
    handle = _cache_manager.create_handle(f"engine-{time.time_ns()}",
                                         key_type=key_type, value_type=value_type)
    handle.metadata["is_cancelled"] = is_cancelled
    fingerprint = _cache_manager.fingerprint(namespace=_cache_namespace,
                                             key_type=key_type, value_type=value_type)
    try:
        try:
            reused = _cache_manager.adopt_prefix(handle, fingerprint, token_ids)
        except PrefixReuseUnsupported:
            reused = 0
        requested = _cache_manager.estimate_growth(
            handle,
            prompt_tokens=effective_prompt_tokens,
            max_new_tokens=output_budget,
        )
        _cache_manager.reserve(
            handle,
            tokens=max(effective_prompt_tokens - reused, 0) + output_budget,
            requested_bytes=requested,
        )
        yield from _stream_bundle(
            bundle,
            prompt,
            resolved,
            is_cancelled,
            cache=handle.bundle.entries,
            cached_prompt_ids=token_ids[:reused],
            _cooperative=_cooperative, _prefill_chunk_size=_prefill_chunk_size,
            _cache_runtime=_cache_manager, _cache_handle=handle,
            spec_type=_spec_type,spec_draft_n_max=_spec_draft_n_max,
            _draft_bundle=_draft_bundle,
            on_cache_update=lambda tokens, _: _cache_manager.publish_prefix(
                handle, fingerprint, tokens
            ),
        )
        if handle.reservation is not None:
            _cache_manager.commit(
                handle.reservation,
                min(handle.bundle.allocated_nbytes, requested),
            )
    finally:
        if not handle.closed:
            _cache_manager.release(handle)


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
    on_execution: Callable[[dict[str, object]], None] | None = None,
    _cooperative: bool = False,
    _prefill_chunk_size: int = 512,
    _cache_runtime: object | None = None,
    _cache_handle: object | None = None,
    _math_observer: object | None = None,
    _ignore_eos: bool = False,
) -> Iterator[GenerationChunk]:
    if model.chat_template is None:
        raise ValueError("loaded model has no chat-template layer")
    prepared = (_cache_handle.metadata.get("prepared_prompt")
                if _cache_handle is not None else None)
    prompt = (prepared.text if prepared is not None else
              model.chat_template.render(messages, **dict(template_options or {})))
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
        on_execution=on_execution,
        _cooperative=_cooperative, _prefill_chunk_size=_prefill_chunk_size,
        _cache_runtime=_cache_runtime, _cache_handle=_cache_handle,
        _math_observer=_math_observer,
        _ignore_eos=_ignore_eos,
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
    adapter_path: str | Path | None = None,
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
            adapter_path=adapter_path,
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


def _stream_bundle(bundle, prompt, options, is_cancelled, **settings):
    """Apply text stops even when a stop starts or ends inside a model token."""
    iterator = _stream_bundle_impl(bundle, prompt, options, is_cancelled, **settings)
    if not options.stop:
        yield from iterator
        return
    buffered = ""
    try:
        for chunk in iterator:
            buffered += chunk.text
            matches = [buffered.find(stop) for stop in options.stop if stop in buffered]
            if matches:
                yield GenerationChunk(chunk.token_id, buffered[:min(matches)],
                                      chunk.generated_tokens, "stop")
                return
            if chunk.finish_reason is not None:
                yield GenerationChunk(chunk.token_id, buffered, chunk.generated_tokens,
                                      chunk.finish_reason)
                return
            hold = max((length for stop in options.stop for length in range(1, len(stop))
                        if buffered.endswith(stop[:length])), default=0)
            ready = buffered[:-hold] if hold else buffered
            buffered = buffered[-hold:] if hold else ""
            yield GenerationChunk(chunk.token_id, ready, chunk.generated_tokens)
    finally:
        iterator.close()


def _stream_bundle_impl(
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
    on_execution: Callable[[dict[str, object]], None] | None = None,
    _cooperative: bool = False,
    _prefill_chunk_size: int = 512,
    _cache_runtime: object | None = None,
    _cache_handle: object | None = None,
    _forced_token_ids: tuple[int, ...] = (),
    _prepared_input_ids: tuple[int, ...] | None = None,
    _draft_bundle: object | None = None,
    _math_policy: object | None = None,
    _math_observer: object | None = None,
    _ignore_eos: bool = False,
) -> Iterator[GenerationChunk]:
    import mlx.core as mx

    from mlx_one.core.cache import configure_kv_caches
    from mlx_one.engine.observation import current_observer
    from mlx_one.engine.optimization_policy import REFERENCE_MATH, math_execution
    from mlx_one.ops.logits import LogitsMode

    math_policy = _math_policy or getattr(_cache_runtime, "math_policy", REFERENCE_MATH)
    request_observer = _math_observer or current_observer()
    if on_execution is None:
        on_execution = request_observer

    cancelled = is_cancelled or (lambda: False)
    if spec_type not in {"none","draft-mtp","draft-model"} or spec_draft_n_max<1:
        raise ValueError("unsupported speculation policy or draft budget")
    if spec_type=="draft-mtp":
        from mlx_one.engine.speculation import validate_mtp

        validate_mtp(bundle,temperature=options.temperature)
    if spec_type=="draft-model":
        from mlx_one.engine.speculation import validate_external

        validate_external(bundle,_draft_bundle,temperature=options.temperature)
    tokenization_started = time.perf_counter()
    prepared = (_cache_handle.metadata.get("prepared_prompt")
                if _cache_handle is not None else None)
    prompt_ids = (list(_prepared_input_ids) if _prepared_input_ids is not None else
                  list(prepared.token_ids) if prepared is not None else
                  bundle.tokenizer.encode(prompt))
    if on_execution is not None:
        on_execution({"kind": "tokenization", "duration_ms":
                      (time.perf_counter() - tokenization_started) * 1000})
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
    pending_times: deque[float] = deque()
    generated: list[int] = []
    decoded = ""
    harmony_prompt = prompt if bundle.architecture == "gpt_oss" else None
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
    forced_tokens: deque[int] = deque(_forced_token_ids)
    closing_reasoning = False
    if reasoning_active and reasoning_budget == 0:
        forced_tokens.extend(think_close)
        closing_reasoning = True
    emitted = 0
    finish_reason = "length"
    eos_token_ids = set(bundle.eos_token_ids)
    if bundle.tokenizer.eos_token_id is not None:
        eos_token_ids.add(bundle.tokenizer.eos_token_id)
    if not eos_token_ids:
        config = getattr(bundle.model.config, "text_config", bundle.model.config)
        configured = getattr(config, "eos_token_id", None)
        if isinstance(configured, int) and not isinstance(configured, bool):
            eos_token_ids.add(configured)
        elif isinstance(configured, (list, tuple)):
            eos_token_ids.update(
                value
                for value in configured
                if isinstance(value, int) and not isinstance(value, bool)
            )
    eos_token_ids = _qualification_eos_ids(eos_token_ids, ignore_eos=_ignore_eos,
                                         runtime=_cache_runtime)
    produced = 0
    prompt_cache_published = False
    while produced < maximum:
        if cancelled():
            finish_reason = "stop"
            break
        current = tokens[len(cached) :] if produced == 0 else [tokens[-1]]
        if not current:
            raise ValueError("prompt cache must leave at least one token for prefill")
        phase = "prefill" if produced == 0 else "decode"
        width = (_prefill_chunk_size if _cooperative and produced == 0 else
                 math_policy.prefill_chunk_size if produced == 0 and
                 math_policy.prefill_chunk_size is not None else len(current))
        start = 0
        while start < len(current):
            stop = min(start + width, len(current))
            if (phase == "prefill" and _cache_runtime is not None
                    and getattr(_cache_runtime, "hybrid_state", False) and stop == len(current)):
                page = _cache_runtime.config.block_size_tokens
                boundary = (len(prompt_ids) - 1) // page * page
                aligned = boundary - cache[0].offset
                if 0 < aligned < stop - start:
                    stop = start + aligned
            boundary = (_cache_handle.metadata.get("semantic_boundary", 0)
                        if _cache_handle is not None and phase == "prefill" else 0)
            relative = boundary - len(cached)
            if start < relative < stop:
                stop = relative
            part = tuple(current[start:stop])
            start = stop
            if (math_policy != REFERENCE_MATH or
                    getattr(_cache_runtime, "mode", None) == "native_optimized"):
                from mlx_one.engine.modes import ModelCapabilityUnsupported

                precision = getattr(bundle.model, "_mlx_one_execution_policy", {})
                if precision.get("dtype") != "float16":
                    raise ModelCapabilityUnsupported(
                        "model precision changed after native math policy was frozen")
            forward_started = time.perf_counter()
            if _cooperative:
                from mlx_one.engine.batching import ForwardRequest

                output = yield ForwardRequest(bundle, part, cache, phase, tuple(prompt_ids),
                                              _cache_runtime, _cache_handle, options=options,
                                              generated_ids=tuple(tokens[len(prompt_ids):]))
            else:
                from contextlib import ExitStack

                from mlx_one.engine.block_kv import BlockKVCache
                from mlx_one.engine.cache import CacheBundle

                transactional = any(isinstance(entry, BlockKVCache) for entry in cache)
                staged = None
                try:
                    with ExitStack() as execution:
                        if _cache_runtime is not None:
                            from mlx_one.engine.pressure import execution_forecast

                            amount = execution_forecast(
                                _cache_runtime, bundle.model.config, batch=1,
                                chunk=len(part), history=cache[0].offset)["bytes"]
                            if transactional:
                                amount += CacheBundle(cache).allocated_nbytes
                            execution.enter_context(_cache_runtime.transient_reservation(amount))
                        staged = CacheBundle(cache).clone() if transactional else None
                        forward_started = time.perf_counter()
                        with math_execution(math_policy, family=bundle.architecture,
                                            phase=phase, observer=request_observer):
                            mode = None
                            if math_policy.logits == "last_only":
                                mode = (LogitsMode.NO_LOGITS if start < len(current) and
                                        not getattr(request_observer, "capture_state", False)
                                        else LogitsMode.LAST_ONLY)
                            kwargs = {"cache": staged.entries if staged else cache}
                            if mode is not None:
                                kwargs["logits_mode"] = mode
                            output = bundle.model(mx.array([part]), **kwargs)
                            if mode is not None:
                                from mlx_one.engine.modes import ModelCapabilityUnsupported

                                if (output.last_hidden_state is None
                                        or output.last_hidden_state.shape[:2] != (1, len(part))):
                                    raise ModelCapabilityUnsupported(
                                        "native logits selection must retain complete hidden rows")
                                if (mode == LogitsMode.LAST_ONLY and (output.logits is None
                                        or output.logits.shape[:2] != (1, 1))):
                                    raise ModelCapabilityUnsupported(
                                        "native last-only logits must have shape [1,1,V]")
                                if mode == LogitsMode.NO_LOGITS and output.logits is not None:
                                    raise ModelCapabilityUnsupported(
                                        "unused intermediate logits must not be projected")
                            # NO_LOGITS is allowed only for unused intermediate chunks.
                            if output.logits is not None:
                                mx.eval(output.logits)
                            else:
                                from mlx_one.engine.transactions import component_tensors

                                state = [tensor for entry in kwargs["cache"]
                                         for tensor in component_tensors(entry)]
                                mx.eval(output.last_hidden_state, state)
                        if cancelled():
                            return
                        if staged is not None:
                            for original, draft in zip(cache, staged.entries, strict=True):
                                original.replace_from(draft)
                finally:
                    if staged is not None:
                        staged.release()
            if _cache_handle is not None:
                _cache_handle.computed_tokens = cache[0].offset
                _cache_runtime.reconcile(_cache_handle)
                if boundary and cache[0].offset == boundary and not cancelled():
                    fingerprint = _cache_handle.metadata.get("fingerprint")
                    if fingerprint is not None:
                        _cache_runtime.stage_prefix(_cache_handle, fingerprint,
                                                    tuple(prompt_ids[:boundary]))
            if on_execution is not None:
                elapsed_ms = (time.perf_counter() - forward_started) * 1000
                on_execution({"kind": "forward", "phase": phase,
                              "duration_ms": getattr(output, "forward_duration_ms", elapsed_ms),
                              "at": time.perf_counter(),
                              "batch_size": getattr(output, "batch_size", 1),
                              "input_tokens": len(part), "logits_dtype": (str(output.logits.dtype)
                                  if output.logits is not None else "not_projected")})
            if request_observer is not None and getattr(request_observer, "capture_bytes", False):
                request_observer({"kind": "cache_bytes",
                                  "bytes": sum(int(getattr(entry, "nbytes", 0))
                                               for entry in cache)})
            if request_observer is not None and getattr(request_observer, "capture_state", False):
                request_observer({"kind": "state", "logits": output.logits,
                                  "cache": cache, "input_ids": part, "phase": phase})
        if cancelled():
            return
        if on_cache_update is not None and produced == 0 and not prompt_cache_published:
            staged = (_cache_runtime.publish_staged_prefix(
                _cache_handle, maximum_entries=_cache_handle.metadata.get("maximum_prefix_entries"))
                if _cache_handle is not None else False)
            if not staged:
                on_cache_update(tuple(prompt_ids), cache)
            prompt_cache_published = True
        logits = output.logits[0, -1]
        using_forced = bool(forced_tokens)
        if request_observer is not None and getattr(request_observer, "profile_math", False):
            from mlx_one.ops.profiling import phase as math_phase

            with math_phase("sampling", observer=request_observer):
                token = (forced_tokens.popleft() if using_forced else
                         select_token(logits, options, randomizer))
        else:
            token = (forced_tokens.popleft() if using_forced else
                     select_token(logits, options, randomizer))
        candidates = [token]
        step_yielded = False
        proposal=None
        step_events=[]
        if spec_type not in {"none","draft-mtp","draft-model"}:
            raise ValueError("unsupported speculation policy")
        if spec_type=="draft-mtp":
            from mlx_one.engine.speculation import SpeculativeProposal, validate_mtp

            validate_mtp(bundle,temperature=options.temperature)
            if (produced+1<maximum and not forced_tokens
                    and not (reasoning_active and reasoning_budget>=0)):
                proposal=SpeculativeProposal(bundle,cache,runtime=_cache_runtime,
                    handle=_cache_handle,cancelled=cancelled,observer=on_execution)
                proposal.propose_mtp(token,output.last_hidden_state[:,-1:],
                                    min(spec_draft_n_max,maximum-produced-1))
                if cancelled():
                    proposal.close()
                    return
                candidates=proposal.candidates or [token]
                if speculative_stats is not None:
                    for key,value in (("drafted_tokens",proposal.drafted),
                                      ("accepted_draft_tokens",proposal.accepted)):
                        speculative_stats[key]=speculative_stats.get(key,0)+value
        if (spec_type=="draft-model" and produced+1<maximum and not forced_tokens
                and not (reasoning_active and reasoning_budget>=0)):
            from mlx_one.engine.speculation import (
                SpeculativeProposal,
                accept_verification,
                prepare_external,
            )

            proposal=SpeculativeProposal(bundle,cache,runtime=_cache_runtime,
                handle=_cache_handle,cancelled=cancelled,observer=on_execution)
            try:
                ids=prepare_external(proposal,_draft_bundle,tuple(tokens),token,
                                     min(spec_draft_n_max,maximum-produced-1))
                if cancelled():
                    proposal.close()
                    return
                if _cooperative:
                    from mlx_one.engine.batching import ForwardRequest

                    verified=yield ForwardRequest(bundle,ids,proposal.verified.entries,'prefill',
                        tuple(prompt_ids),return_all_logits=True,staged_verification=True)
                else:
                    verified=proposal._forward(ids,proposal.verified.entries,'verify')
                accept_verification(proposal,verified.logits)
                candidates=proposal.candidates
            except BaseException:
                proposal.close()
                raise
        before_produced=produced
        try:
            stopped = False
            for candidate in candidates[: maximum - produced]:
                if candidate in eos_token_ids:
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
                pending_times.append(time.perf_counter())
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
                    for _ in matched:
                        pending_times.pop()
                    finish_reason = "stop"
                    stopped = True
                    break
                while len(pending) >= hold:
                    emitted += 1
                    generated.append(pending.pop(0))
                    sampled_at = pending_times.popleft()
                    if on_execution is not None:
                        on_execution({"kind": "token", "at": sampled_at, "token_id": generated[-1]})
                    piece, decoded = _decode_piece(bundle.tokenizer, generated, decoded,
                                                    harmony_prompt=harmony_prompt)
                    step_yielded = True
                    event=GenerationChunk(generated[-1],piece,emitted)
                    if proposal is not None:
                        step_events.append(event)
                    else:
                        yield event
            if proposal is not None:
                proposal.commit(produced-before_produced)
        finally:
            if proposal is not None:
                proposal.close()
        if _cooperative and proposal is not None:
            from mlx_one.engine.batching import GenerationProgress

            yield GenerationProgress(tuple(tokens[len(prompt_ids):]),cache[0].offset)
        for event in step_events:
            yield event
        if stopped:
            break
        if yield_steps and not step_yielded:
            yield GenerationChunk(None, "", emitted)
    for current_token in pending:
        sampled_at = pending_times.popleft()
        if on_execution is not None:
            on_execution({"kind": "token", "at": sampled_at, "token_id": current_token})
        emitted += 1
        generated.append(current_token)
        piece, decoded = _decode_piece(bundle.tokenizer, generated, decoded,
                                       harmony_prompt=harmony_prompt)
        yield GenerationChunk(current_token, piece, emitted)
    trailing, _ = _decode_piece(bundle.tokenizer, generated, decoded, final=True,
                                harmony_prompt=harmony_prompt)
    yield GenerationChunk(None, trailing, emitted, finish_reason)


def _legacy_context_length(bundle: LoadedTextModel) -> int:
    config = getattr(bundle.model.config, "text_config", bundle.model.config)
    for name in ("max_position_embeddings", "max_context_length", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("model configuration does not declare a context length")
