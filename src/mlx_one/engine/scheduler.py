"""Deterministic token-budget policy for resumable engine sequences."""

from __future__ import annotations

import time
from collections import deque

from mlx_one.engine.contracts import BatchPlan, Phase, ScheduledSequence, SequenceState


class TokenBudgetScheduler:
    """Decode-first policy with bounded progress for the oldest prefill."""

    def __init__(self, *, max_batch_tokens: int = 2048, prefill_chunk_size: int = 512) -> None:
        if max_batch_tokens < 1 or prefill_chunk_size < 1:
            raise ValueError("scheduler token limits must be positive")
        self.max_batch_tokens = max_batch_tokens
        self.prefill_chunk_size = min(prefill_chunk_size, max_batch_tokens)
        self.waiting: deque[SequenceState] = deque()
        self.active: list[SequenceState] = []

    def add(self, sequence: SequenceState) -> None:
        if sequence.phase is not Phase.WAITING:
            raise ValueError("only waiting sequences can enter the scheduler")
        self.waiting.append(sequence)

    def schedule(self, now: float | None = None) -> tuple[BatchPlan, ...]:
        current = time.monotonic() if now is None else now
        self._expire(current)
        while self.waiting:
            sequence = self.waiting.popleft()
            sequence.transition(Phase.PREFILL)
            self.active.append(sequence)
        decodes = tuple(
            sequence
            for sequence in self.active
            if sequence.phase is Phase.DECODE and not sequence.cancel.is_set()
        )
        batches: list[BatchPlan] = []
        if decodes:
            batches.append(
                BatchPlan(
                    tuple(ScheduledSequence(s.request_id, "decode", s.computed_len, 1)
                          for s in decodes[:self.max_batch_tokens]),
                    min(len(decodes), self.max_batch_tokens),
                )
            )
        prefill = next(
            (
                sequence
                for sequence in self.active
                if sequence.phase is Phase.PREFILL and not sequence.cancel.is_set()
            ),
            None,
        )
        if prefill is not None:
            budget = min(
                self.prefill_chunk_size,
                self.max_batch_tokens,
                max(prefill.remaining_prompt_tokens, 1),
            )
            batches.append(BatchPlan((ScheduledSequence(
                prefill.request_id, "prefill", prefill.computed_len, budget
            ),), budget))
        return tuple(batches)

    def prune(self) -> None:
        self.active = [sequence for sequence in self.active if not sequence.phase.terminal]

    def _expire(self, now: float) -> None:
        for sequence in (*self.waiting, *self.active):
            if sequence.cancel.is_set() and not sequence.phase.terminal:
                sequence.transition(Phase.CANCELLED)
            elif (
                sequence.deadline_ns is not None
                and int(now * 1e9) >= sequence.deadline_ns
                and not sequence.phase.terminal
            ):
                sequence.transition(Phase.TIMED_OUT)
        self.waiting = deque(item for item in self.waiting if not item.phase.terminal)
        self.prune()
