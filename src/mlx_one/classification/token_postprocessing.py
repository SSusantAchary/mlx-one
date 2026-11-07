"""HF v4.57.1 token-classification NONE/SIMPLE postprocessing (Apache-2.0).

Offsets are Python Unicode code-point indices, end-exclusive. All source tokens
participate in grouping before the O label is filtered, preserving BIO breaks.
"""

from __future__ import annotations

import numpy as np


def entities_from_scores(
    text, encoding, scores, labels, *, aggregation_strategy="none", decoder=None
):
    if aggregation_strategy not in ("none", "simple"):
        raise ValueError("aggregation_strategy must be none or simple")
    if len(encoding.ids) != len(scores):
        raise ValueError("token scores must match the encoded sequence")
    entities = []
    for i, (token, offsets, special, row) in enumerate(
        zip(encoding.tokens, encoding.offsets, encoding.special_tokens_mask, scores, strict=True)
    ):
        if special:
            continue
        start, end = offsets
        index = int(np.argmax(row))
        word = text[start:end] if token == "[UNK]" else token
        entities.append(
            {
                "entity": labels[str(index)],
                "score": float(row[index]),
                "index": i,
                "word": word,
                "start": int(start),
                "end": int(end),
            }
        )
    if aggregation_strategy == "none":
        return [item for item in entities if item["entity"] != "O"]
    groups, current = [], []

    def tag(entity):
        name = entity["entity"]
        return (name[:1], name[2:]) if name.startswith(("B-", "I-")) else ("I", name)

    def finish(group):
        return {
            "entity_group": tag(group[0])[1],
            "score": float(np.mean(np.asarray([v["score"] for v in group], dtype=np.float32))),
            "word": (
                decoder.decode([v["word"] for v in group])
                if decoder is not None
                else " ".join(v["word"] for v in group)
            ),
            "start": group[0]["start"],
            "end": group[-1]["end"],
        }

    for entity in entities:
        bi, label = tag(entity)
        if current and (label != tag(current[-1])[1] or bi == "B"):
            groups.append(finish(current))
            current = []
        current.append(entity)
    if current:
        groups.append(finish(current))
    return [item for item in groups if item["entity_group"] != "O"]
