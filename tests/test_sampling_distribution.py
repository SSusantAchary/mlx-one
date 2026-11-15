"""Check seeded compatibility against the original sampler, including CDF edges."""

import math
import random

import numpy as np
import pytest

from mlx_one.text.sampling import _sample_probabilities
from mlx_one.text.schemas import TextGenerationOptions


def original(probabilities, options, rng):
    ranked = sorted(enumerate(probabilities.tolist()), key=lambda item: item[1], reverse=True)
    if options.top_k is not None:
        ranked = ranked[: options.top_k]
    if options.top_p < 1:
        cutoff = options.top_p * sum(p for _, p in ranked)
        selected, cumulative = [], 0.0
        for item in ranked:
            selected.append(item)
            cumulative += item[1]
            if cumulative >= cutoff:
                break
        ranked = selected
    total = sum(p for _, p in ranked)
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("sampling probabilities are invalid")
    threshold, cumulative = rng.random() * total, 0.0
    for token, probability in ranked:
        cumulative += probability
        if threshold <= cumulative:
            return token
    return ranked[-1][0]


@pytest.mark.parametrize("top_k", [None, 1, 7, 500])
@pytest.mark.parametrize("top_p", [1.0, 0.9, 0.01])
def test_seeded_sequence_and_rng_state_match_original(top_k, top_p):
    options = TextGenerationOptions(temperature=0.7, top_k=top_k, top_p=top_p)
    # Include exact ties, zeros, and a large dynamic range.
    values = np.random.default_rng(17).lognormal(0, 7, 257).astype(np.float32)
    values[:10] = 0
    values[10:20] = values[20]
    values /= values.sum()
    before, after = random.Random(42), random.Random(42)
    for _ in range(50):
        assert _sample_probabilities(values, options, after) == original(values, options, before)
    assert before.getstate() == after.getstate()


@pytest.mark.parametrize("top_k,top_p", [(None, 1), (None, 0.9), (40, 0.9)])
def test_gptoss_vocabulary_matches_original(top_k, top_p):
    values = np.random.default_rng(19).random(201088).astype(np.float32)
    values /= values.sum()
    options = TextGenerationOptions(temperature=0.7, top_k=top_k, top_p=top_p)
    for seed in range(3):
        assert _sample_probabilities(values, options, random.Random(seed)) == original(
            values, options, random.Random(seed)
        )


class FixedRandom:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


@pytest.mark.parametrize("threshold", [0, 0.25, np.nextafter(0.25, 1), 0.5, 0.75, 1])
@pytest.mark.parametrize("top_p", [0.5, 1.0])
def test_ties_cdf_boundaries_and_terminal_fallback(threshold, top_p):
    options = TextGenerationOptions(temperature=1, top_p=top_p)
    values = np.array([0.25] * 4 + [0.0], dtype=np.float32)
    assert _sample_probabilities(values, options, FixedRandom(threshold)) == original(
        values, options, FixedRandom(threshold)
    )


@pytest.mark.parametrize("values", [[0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_probability_mass_is_rejected(values):
    with pytest.raises(RuntimeError, match="sampling probabilities are invalid"):
        _sample_probabilities(
            np.array(values), TextGenerationOptions(temperature=1), random.Random()
        )


@pytest.mark.parametrize("top_p", [1.0, 0.9, 0.01])
def test_bf16_integer_ranking_preserves_ties_signed_zero_and_seed(top_p):
    values = np.random.default_rng(23).random(201088).astype(np.float32)
    values /= values.sum()
    # Exactly representable BF16 values, exported to float32 as in select_token.
    values.view(np.uint32)[:] &= np.uint32(0xFFFF0000)
    values[:4] = [0.0, -0.0, 0.0, -0.0]
    options = TextGenerationOptions(temperature=0.7, top_p=top_p)
    for seed in range(4):
        assert _sample_probabilities(values, options, random.Random(seed)) == original(
            values, options, random.Random(seed)
        )
