"""Native logits sampling shared by every causal language family."""

from __future__ import annotations

import math
import random
from typing import Any

from mlx_one.text.schemas import TextGenerationOptions


def select_token(logits: Any, options: TextGenerationOptions, randomizer: random.Random) -> int:
    import mlx.core as mx

    if options.temperature == 0:
        return int(mx.argmax(logits).item())
    probabilities = mx.softmax(logits / options.temperature, axis=-1)
    mx.eval(probabilities)
    ranked = sorted(enumerate(probabilities.tolist()), key=lambda item: item[1], reverse=True)
    if options.top_k is not None:
        ranked = ranked[: options.top_k]
    if options.top_p < 1.0:
        cutoff = options.top_p * sum(probability for _, probability in ranked)
        selected: list[tuple[int, float]] = []
        cumulative = 0.0
        for item in ranked:
            selected.append(item)
            cumulative += item[1]
            if cumulative >= cutoff:
                break
        ranked = selected
    total = sum(value for _, value in ranked)
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("sampling probabilities are invalid")
    threshold = randomizer.random() * total
    cumulative = 0.0
    for token, probability in ranked:
        cumulative += probability
        if threshold <= cumulative:
            return int(token)
    return int(ranked[-1][0])
