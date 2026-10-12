"""Internal task runners that isolate execution from request scheduling."""

from __future__ import annotations

from typing import Any

from mlx_one.engine.cache import CacheBundle, cache_capabilities
from mlx_one.engine.cache_planner import build_cache_plan
from mlx_one.engine.contracts import (
    BatchPlan,
    EngineRequest,
    Phase,
    RequestKind,
    RunnerCapabilities,
    RunnerOutput,
    SequenceState,
)


class CausalLMRunner:
    """Thin runner over a loaded native causal-language-model bundle."""

    def __init__(self, bundle: Any) -> None:
        self.bundle = bundle
        self._sequences: dict[str, SequenceState] = {}
        prototype = tuple(self.bundle.model.make_cache())
        self._decode_batchable = bool(prototype) and all(
            cache_capabilities(cache).batchable for cache in prototype
        )

    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            frozenset({RequestKind.TEXT}),
            batchable=self._decode_batchable,
            chunked_prefill=True,
            speculative=hasattr(self.bundle.model, "mtp"),
            family=getattr(self.bundle, "architecture", "unknown"),
            supports_batched_decode=self._decode_batchable,
            supported_state_kinds=tuple(dict.fromkeys(
                spec.kind for spec in self.cache_plan().state_specs
            )) if hasattr(self.bundle, "model_id") else (),
        )

    def prepare(self, request: EngineRequest) -> SequenceState:
        if request.kind is not RequestKind.TEXT:
            raise ValueError("causal LM runner accepts text requests only")
        prompt = request.payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("text engine request requires a non-empty prompt")
        tokens = tuple(int(token) for token in self.bundle.tokenizer.encode(prompt))
        if not tokens:
            bos = self.bundle.tokenizer.bos_token_id
            if bos is None:
                raise ValueError("empty tokenization requires a tokenizer BOS token")
            tokens = (int(bos),)
        if request.request_id in self._sequences:
            raise ValueError("request ID is already registered")
        state = SequenceState(
            request_id=request.request_id, input_ids=tokens, prompt_len=len(tokens),
            request=request, cache=self.make_cache(),
            deadline_ns=int(request.deadline * 1e9) if request.deadline is not None else None,
            rng_seed=request.payload.get("seed"),
        )
        self._sequences[request.request_id] = state
        return state

    def make_cache(self) -> CacheBundle:
        return CacheBundle(tuple(self.bundle.model.make_cache()))

    def cache_plan(self, *, block_size_tokens: int = 32):
        return build_cache_plan(self.bundle, block_size_tokens=block_size_tokens)

    def prefill(self, plan: BatchPlan) -> RunnerOutput:
        return self._execute(plan, "prefill")

    def decode(self, plan: BatchPlan) -> RunnerOutput:
        return self._execute(plan, "decode")

    def finalize(self, sequence: SequenceState) -> None:
        if self._sequences.get(sequence.request_id) is not sequence:
            return
        if sequence.cache is not None:
            sequence.cache.release()
        self._sequences.pop(sequence.request_id)

    def release(self) -> None:
        for state in tuple(self._sequences.values()):
            self.finalize(state)

    def _resolve(self, plan: BatchPlan, phase: str) -> tuple[SequenceState, ...]:
        if plan.execution_mode == "paged":
            raise ValueError("runner does not implement paged attention")
        sequences = []
        for slot in plan.sequences:
            state = self._sequences.get(slot.request_id)
            if slot.block_table_id is not None:
                raise ValueError("dense runner does not accept page-table mappings")
            if state is None or slot.phase != phase:
                raise ValueError("unknown request or incompatible batch phase")
            if state.phase.value != phase or state.cancel.is_set():
                raise ValueError("batch requires a live sequence in the scheduled phase")
            if slot.start_pos != state.computed_len:
                raise ValueError("batch start position must match committed state")
            if not isinstance(state.cache, CacheBundle) or state.cache.closed:
                raise ValueError("batch requires a live cache bundle")
            if state.cache.offset != state.computed_len:
                raise ValueError("cache and committed state boundaries differ")
            if phase == "prefill" and slot.input_token_count > state.remaining_prompt_tokens:
                raise ValueError("prefill exceeds remaining prompt tokens")
            if phase == "decode" and state.computed_len != state.prompt_len + len(
                state.generated_ids
            ) - 1:
                raise ValueError("decode requires exactly one uncomputed generated token")
            sequences.append(state)
        return tuple(sequences)

    def _execute(self, plan: BatchPlan, phase: str) -> RunnerOutput:
        sequences = self._resolve(plan, phase)
        import mlx.core as mx

        if phase == "decode" and len(sequences) > 1 and self._decode_batchable:
            logits = self._execute_batched_decode(sequences, mx)
        else:
            logits = []
            for slot, state in zip(plan.sequences, sequences, strict=True):
                token_ids = (
                    state.input_ids[slot.start_pos:slot.start_pos + slot.input_token_count]
                    if phase == "prefill" else (state.generated_ids[-1],)
                )
                try:
                    output = self.bundle.model(mx.array([token_ids]), cache=state.cache.entries)
                    mx.eval(output.logits)
                except Exception as exc:
                    state.error = exc
                    state.transition(Phase.FAILED)
                    self.finalize(state)
                    raise
                state.commit(slot.input_token_count)
                logits.append(output.logits)
        return RunnerOutput(
            tuple(s.request_id for s in plan.sequences), tuple(logits),
            tuple(s.input_token_count for s in plan.sequences),
        )

    def _execute_batched_decode(self, sequences: tuple[SequenceState, ...], mx: Any):
        caches = tuple(sequence.cache for sequence in sequences)
        if len({cache.offset for cache in caches}) != 1:
            raise ValueError("batched decode requires identical cache offsets")
        merged = CacheBundle.merge(caches)
        token_ids = mx.array([[sequence.generated_ids[-1]] for sequence in sequences])
        try:
            output = self.bundle.model(token_ids, cache=merged.entries)
            mx.eval(output.logits)
            selected = [merged.batch_select((index,)) for index in range(len(sequences))]
        except Exception as exc:
            for sequence in sequences:
                sequence.error = exc
                sequence.transition(Phase.FAILED)
                self.finalize(sequence)
            raise
        logits = []
        for index, sequence in enumerate(sequences):
            sequence.cache.release()
            sequence.cache = selected[index]
            sequence.commit(1)
            logits.append(output.logits[index:index + 1])
        merged.release()
        return tuple(logits)


class EncoderRunner:
    """Adapt a native embedding or reranking callable to an engine request."""

    def __init__(self, service: Any, *, reranking: bool = False) -> None:
        self.service = service
        self.reranking = reranking

    def capabilities(self) -> RunnerCapabilities:
        kind = RequestKind.RERANK if self.reranking else RequestKind.EMBEDDING
        return RunnerCapabilities(frozenset({kind}), batchable=True)

    def execute(self, request: EngineRequest) -> Any:
        expected = RequestKind.RERANK if self.reranking else RequestKind.EMBEDDING
        if request.kind is not expected:
            raise ValueError(f"encoder runner cannot execute {request.kind.value} requests")
        return self.service(**dict(request.payload))


class ASRRunner:
    """Adapt native Whisper transcription without decode-scheduler coupling."""

    def __init__(self, service: Any) -> None:
        self.service = service

    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            frozenset({RequestKind.TRANSCRIPTION}), modalities=frozenset({"audio"})
        )

    def execute(self, request: EngineRequest) -> Any:
        if request.kind is not RequestKind.TRANSCRIPTION:
            raise ValueError(f"ASR runner cannot execute {request.kind.value} requests")
        return self.service(**dict(request.payload))


class VLMRunner:
    """Adapt a native VLM service and advertise image-text capability."""

    def __init__(self, service: Any) -> None:
        self.service = service

    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            frozenset({RequestKind.VISION}),
            modalities=frozenset({"text", "image"}),
        )

    def execute(self, request: EngineRequest) -> Any:
        if request.kind is not RequestKind.VISION:
            raise ValueError(f"VLM runner cannot execute {request.kind.value} requests")
        return self.service(**dict(request.payload))
