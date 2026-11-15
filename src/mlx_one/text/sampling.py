"""Native logits sampling shared by every causal language family."""

from __future__ import annotations

import math
import random
from typing import Any

from mlx_one.text.schemas import TextGenerationOptions


def banned_ngram_tokens(history, size):
    """Tokens that would repeat an existing n-gram, including prompt context."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("n-gram size must be a positive integer")
    if len(history) + 1 < size:
        return set()
    prefix = tuple(history[-(size - 1) :]) if size > 1 else ()
    return {
        history[i + size - 1]
        for i in range(len(history) - size + 1)
        if tuple(history[i : i + size - 1]) == prefix
    }


def block_repeated_ngrams(history, logits, *, size):
    import mlx.core as mx

    banned = banned_ngram_tokens(history, size)
    if not banned:
        return logits
    result = mx.array(logits)
    result[mx.array(sorted(banned))] = -float("inf")
    return result


def select_token(logits: Any, options: TextGenerationOptions, randomizer: random.Random) -> int:
    import mlx.core as mx

    if options.temperature == 0:
        return int(mx.argmax(logits).item())
    probabilities = mx.softmax(logits / options.temperature, axis=-1)
    mx.eval(probabilities)
    # NumPy supports the float32 export; BF16 -> FP32 is exact and does not
    # change the probabilities produced by the model's softmax.
    return _sample_probabilities(probabilities.astype(mx.float32), options, randomizer)


def _sample_probabilities(probabilities, options, randomizer):
    """Vectorize ranking/CDF work while preserving the existing Python RNG.

    Stable descending order retains token-ID order for ties. Float64 cumsum
    matches the old sequential Python-float accumulation. Keep Python's sum
    for normalizers: its rounding differs across supported Python versions,
    so replacing it with NumPy's pairwise sum can change seeded boundaries.
    """
    import numpy as np

    values = np.asarray(probabilities)
    if values.dtype == np.float32:
        bits = values.view(np.uint32)
        if np.all((bits & 0xFFFF) == 0) and np.all(values >= 0):
            # BF16 probabilities exported as FP32 have zero low mantissa bits.
            # Their nonnegative IEEE encodings preserve numeric order. Stable
            # uint16 ranking uses NumPy's bounded integer sort, avoiding a
            # vocabulary-sized floating-point comparison sort. No rounding.
            keys = (bits >> 16).astype(np.uint16)
            keys[values == 0] = 0  # +0 and -0 must remain stable ties.
            order = np.argsort(~keys, kind="stable")
        else:
            order = np.argsort(-values, kind="stable")
    else:
        order = np.argsort(-values, kind="stable")
    weights = np.asarray(values, dtype=np.float64)
    if options.top_k is not None:
        order = order[: options.top_k]
    ranked = weights[order]
    cumulative = np.cumsum(ranked, dtype=np.float64)
    if options.top_p < 1.0:
        cutoff = options.top_p * sum(ranked.tolist())
        count = min(int(np.searchsorted(cumulative, cutoff, side="left")) + 1, len(order))
        order, ranked, cumulative = order[:count], ranked[:count], cumulative[:count]
    total = sum(ranked.tolist())
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("sampling probabilities are invalid")
    threshold = randomizer.random() * total
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), len(order) - 1)
    return int(order[index])
