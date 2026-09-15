"""Deterministic token-budget policy for resumable engine sequences."""

from __future__ import annotations

import time
from collections import deque

from mlx_one.engine.contracts import ExecutionBatch, SequenceContext, SequenceState


class TokenBudgetScheduler:
    """Decode-first policy with bounded progress for the oldest prefill."""

    def __init__(self, *, max_batch_tokens: int = 2048, prefill_chunk_size: int = 512) -> None:
        if max_batch_tokens < 1 or prefill_chunk_size < 1:
            raise ValueError("scheduler token limits must be positive")
        self.max_batch_tokens = max_batch_tokens
        self.prefill_chunk_size = min(prefill_chunk_size, max_batch_tokens)
        self.waiting: deque[SequenceContext] = deque()
        self.active: list[SequenceContext] = []

    def add(self, sequence: SequenceContext) -> None:
        if sequence.state is not SequenceState.WAITING:
            raise ValueError("only waiting sequences can enter the scheduler")
        self.waiting.append(sequence)

    def schedule(self, now: float | None = None) -> tuple[ExecutionBatch, ...]:
        current = time.monotonic() if now is None else now
        self._expire(current)
        while self.waiting:
            sequence = self.waiting.popleft()
            sequence.transition(SequenceState.PREFILL)
            self.active.append(sequence)
        decodes = tuple(
            sequence
            for sequence in self.active
            if sequence.state is SequenceState.DECODING and not sequence.cancel.is_set()
        )
        batches: list[ExecutionBatch] = []
        if decodes:
            batches.append(
                ExecutionBatch(
                    decodes[: self.max_batch_tokens],
                    "decode",
                    min(len(decodes), self.max_batch_tokens),
                )
            )
        prefill = next(
            (
                sequence
                for sequence in self.active
                if sequence.state is SequenceState.PREFILL and not sequence.cancel.is_set()
            ),
            None,
        )
        if prefill is not None:
            budget = min(
                self.prefill_chunk_size,
                self.max_batch_tokens,
                max(prefill.remaining_prompt_tokens, 1),
            )
            batches.append(ExecutionBatch((prefill,), "prefill", budget))
        return tuple(batches)

    def prune(self) -> None:
        self.active = [sequence for sequence in self.active if not sequence.state.terminal]

    def _expire(self, now: float) -> None:
        for sequence in (*self.waiting, *self.active):
            if sequence.cancel.is_set() and not sequence.state.terminal:
                sequence.transition(SequenceState.CANCELLED)
            elif (
                sequence.request.deadline is not None
                and now >= sequence.request.deadline
                and not sequence.state.terminal
            ):
                sequence.transition(SequenceState.TIMED_OUT)
        self.waiting = deque(item for item in self.waiting if not item.state.terminal)
        self.prune()
