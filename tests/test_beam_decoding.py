"""Beam search must retain independent alternatives and honor termination."""

from types import SimpleNamespace

import pytest


class Cache:
    def __init__(self, tokens=()):
        self.tokens = tokens

    def clone(self):
        return Cache(self.tokens)


def test_beams_find_better_completed_sequence_than_greedy_without_sharing_state():
    import mlx.core as mx

    from mlx_one.text.beam import stream_beam_search

    observed = []

    def forward(ids, cache):
        cache[0].tokens += tuple(ids[0].tolist())
        observed.append(cache[0].tokens)
        last = cache[0].tokens[-1]
        probabilities = (
            [0.55, 0.45, 1e-12, 1e-12]
            if last == 9
            else [0.25, 0.25, 0.25, 0.25]
            if last == 0
            else [0.01, 0.01, 0.01, 0.97]
        )
        return SimpleNamespace(logits=mx.log(mx.array([[probabilities]])))

    events = list(
        stream_beam_search(forward, [9], [Cache()], eos_token_ids=[3], max_tokens=3, beam_size=3)
    )
    assert [event.token_id for event in events[:-1]] == [1]
    assert events[-1].finish_reason == "stop"
    assert events[-1].terminal_token_id == 3
    assert (9, 0) in observed and (9, 1) in observed
    assert all(path[0] == 9 and len(path) <= 3 for path in observed)


def test_beam_limit_and_cancellation_are_explicit():
    import mlx.core as mx

    from mlx_one.text.beam import stream_beam_search

    def forward(ids, cache):
        return SimpleNamespace(logits=mx.array([[[2.0, 1.0]]]))

    events = list(
        stream_beam_search(
            forward, [0], [Cache()], eos_token_ids=[], max_tokens=2, forced_first_token=1
        )
    )
    assert events[0].token_id == 1
    assert events[-1].finish_reason == "length"
    assert events[-1].generated_tokens == 2
    with pytest.raises(TimeoutError):
        list(
            stream_beam_search(
                forward, [0], [Cache()], eos_token_ids=[], max_tokens=2, cancelled=lambda: True
            )
        )
