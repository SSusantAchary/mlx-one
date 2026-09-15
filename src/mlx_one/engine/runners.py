"""Internal task runners that isolate execution from request scheduling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import is_dataclass, replace
from typing import Any

from mlx_one.engine.cache import CacheBundle, cache_capabilities
from mlx_one.engine.contracts import (
    EngineRequest,
    ExecutionBatch,
    RequestKind,
    RunnerCapabilities,
    SequenceContext,
    SequenceState,
)


class CausalLMRunner:
    """Thin runner over a loaded native causal-language-model bundle."""

    def __init__(self, bundle: Any) -> None:
        self.bundle = bundle
        prototype = tuple(self.bundle.model.make_cache())
        self._decode_batchable = bool(prototype) and all(
            cache_capabilities(cache).batchable for cache in prototype
        )

    @property
    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            frozenset({RequestKind.TEXT}),
            batchable=self._decode_batchable,
            chunked_prefill=True,
            speculative=hasattr(self.bundle.model, "mtp"),
        )

    def prepare(self, request: EngineRequest) -> SequenceContext:
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
        return SequenceContext(
            request=request,
            state=SequenceState.WAITING,
            prompt_tokens=tokens,
            cache=self.make_cache(),
        )

    def make_cache(self) -> CacheBundle:
        return CacheBundle(tuple(self.bundle.model.make_cache()))

    def prefill(self, batch: ExecutionBatch) -> Sequence[Any]:
        if batch.phase != "prefill":
            raise ValueError("prefill runner received a non-prefill batch")
        return self._execute(batch)

    def decode(self, batch: ExecutionBatch) -> Sequence[Any]:
        if batch.phase != "decode":
            raise ValueError("decode runner received a non-decode batch")
        return self._execute(batch)

    def finalize(self, sequence: SequenceContext) -> None:
        if sequence.cache is not None and sequence.state in {
            SequenceState.CANCELLED,
            SequenceState.TIMED_OUT,
            SequenceState.FAILED,
        }:
            sequence.cache.release()

    def _execute(self, batch: ExecutionBatch) -> Sequence[Any]:
        import mlx.core as mx

        if (
            batch.phase == "decode"
            and len(batch.sequences) > 1
            and self._decode_batchable
        ):
            return self._execute_batched_decode(batch, mx)
        outputs = []
        for sequence in batch.sequences:
            if not isinstance(sequence.cache, CacheBundle):
                raise TypeError("sequence cache must be a CacheBundle")
            if batch.phase == "prefill":
                start = sequence.computed_tokens
                stop = min(start + batch.token_budget, len(sequence.prompt_tokens))
                token_ids = sequence.prompt_tokens[start:stop]
            else:
                if not sequence.generated_tokens:
                    raise ValueError("decode requires a previously generated token")
                token_ids = (sequence.generated_tokens[-1],)
            output = self.bundle.model(mx.array([token_ids]), cache=sequence.cache.entries)
            mx.eval(output.logits)
            outputs.append(output)
            if batch.phase == "prefill":
                sequence.computed_tokens += len(token_ids)
        return tuple(outputs)

    def _execute_batched_decode(self, batch: ExecutionBatch, mx: Any) -> Sequence[Any]:
        caches = tuple(sequence.cache for sequence in batch.sequences)
        if not all(isinstance(cache, CacheBundle) for cache in caches):
            raise TypeError("sequence cache must be a CacheBundle")
        if len({cache.offset for cache in caches}) != 1:
            raise ValueError("batched decode requires identical cache offsets")
        merged = CacheBundle.merge(caches)
        token_ids = mx.array(
            [[sequence.generated_tokens[-1]] for sequence in batch.sequences]
        )
        output = self.bundle.model(token_ids, cache=merged.entries)
        mx.eval(output.logits)
        outputs = []
        for index, sequence in enumerate(batch.sequences):
            sequence.cache = merged.batch_select((index,))
            logits = output.logits[index : index + 1]
            outputs.append(
                replace(output, logits=logits, cache=sequence.cache.entries)
                if is_dataclass(output)
                else logits
            )
        return tuple(outputs)


class EncoderRunner:
    """Adapt a native embedding or reranking callable to an engine request."""

    def __init__(self, service: Any, *, reranking: bool = False) -> None:
        self.service = service
        self.reranking = reranking

    @property
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

    @property
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

    @property
    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            frozenset({RequestKind.VISION}),
            modalities=frozenset({"text", "image"}),
        )

    def execute(self, request: EngineRequest) -> Any:
        if request.kind is not RequestKind.VISION:
            raise ValueError(f"VLM runner cannot execute {request.kind.value} requests")
        return self.service(**dict(request.payload))
