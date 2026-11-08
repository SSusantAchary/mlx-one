"""Bounded deterministic beam decoding with independent native cache branches."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlx_one.core.cache import clone_caches
from mlx_one.text.schemas import GenerationChunk


@dataclass
class _Beam:
    tokens: tuple[int, ...]
    score: float
    cache: tuple


def stream_beam_search(
    forward,
    prompt_ids,
    cache,
    *,
    eos_token_ids,
    max_tokens,
    beam_size=3,
    forced_first_token=None,
    cancelled=lambda: False,
    on_execution=None,
):
    """Yield the winning sequence after EOS-aware, length-normalized search.

    Page OCR consumes completed outputs. Keeping beams private until completion
    prevents committing a prefix that a later beam can revise. Cache tensors are
    immutable and each branch owns its cache objects.
    """
    import time

    if max_tokens < 1 or beam_size < 1 or not prompt_ids:
        raise ValueError("beam decoding requires a prompt and positive bounded sizes")
    active = [_Beam((), 0.0, tuple(cache))]
    finished = []
    eos = set(eos_token_ids)
    for step in range(max_tokens):
        candidates = []
        for beam in active:
            if cancelled():
                raise TimeoutError("beam generation was cancelled or timed out")
            ids = prompt_ids if not beam.tokens else [beam.tokens[-1]]
            started = time.perf_counter()
            output = forward(mx.array([ids]), cache=beam.cache)
            logits = output.logits[0, -1].astype(mx.float32)
            scores = logits - mx.logsumexp(logits)
            mx.eval(scores)
            if not bool(mx.isfinite(scores).all().item()):
                raise ValueError("beam decoding produced non-finite log probabilities")
            if on_execution:
                on_execution(
                    dict(
                        kind="forward",
                        phase="prefill" if not step else "decode",
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
                )
            if step == 0 and forced_first_token is not None:
                choices = [(forced_first_token, 0.0)]
            else:
                count = min(2 * beam_size, scores.size)
                indices = mx.argpartition(-scores, kth=count - 1)[:count]
                choices = list(zip(indices.tolist(), scores[indices].tolist(), strict=True))
            candidates.extend((beam.score + score, token, beam) for token, score in choices)
        candidates.sort(key=lambda row: (-row[0], row[1], row[2].tokens))
        next_active = []
        for rank, (score, token, parent) in enumerate(candidates):
            tokens = (*parent.tokens, int(token))
            if token in eos:
                if rank < beam_size:
                    finished.append((score / len(tokens), tokens, "stop"))
            elif len(next_active) < beam_size:
                next_active.append(_Beam(tokens, score, clone_caches(parent.cache)))
            if len(next_active) == beam_size and rank >= beam_size:
                break
        active = next_active
        finished = sorted(finished, key=lambda row: (-row[0], row[1]))[:beam_size]
        if not active:
            break
        # Match the publisher's early_stopping=False beam scorer heuristic.
        if len(finished) >= beam_size:
            best_possible = max(beam.score for beam in active) / (step + 1)
            if finished[-1][0] >= best_possible:
                active = []
                break
    finished.extend((beam.score / len(beam.tokens), beam.tokens, "length") for beam in active)
    if not finished:
        raise ValueError("beam decoding did not produce a sequence")
    _, tokens, reason = min(finished, key=lambda row: (-row[0], row[1]))
    content = tokens[:-1] if reason == "stop" else tokens
    for index, token in enumerate(content, 1):
        yield GenerationChunk(token, "", index)
    yield GenerationChunk(None, "", len(content), reason, tokens[-1] if reason == "stop" else None)
