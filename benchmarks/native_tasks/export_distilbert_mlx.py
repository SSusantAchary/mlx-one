#!/usr/bin/env python3
"""Capture native DistilBERT tokenizer, hidden states and scores."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_one.engine.task_runners.classification import DistilBertClassificationAdapter
from mlx_one.models.classification.distilbert import load_distilbert_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    bundle = load_distilbert_classifier(args.checkpoint)
    model = bundle.model
    trace = {}
    adapter = DistilBertClassificationAdapter(bundle)
    cases = json.loads(args.cases.read_text())
    for index, offset in enumerate(range(0, len(cases), args.batch_size)):
        prepared = adapter.prepare(
            "text-classification",
            {"texts": cases[offset : offset + args.batch_size], "max_length": args.max_length},
        )
        sequences = prepared.payload["sequences"]
        width = max(map(len, sequences))
        ids = mx.array([list(row) + [0] * (width - len(row)) for row in sequences])
        mask = mx.array([[1] * len(row) + [0] * (width - len(row)) for row in sequences])
        x = model.distilbert.embeddings(ids)
        prefix = f"case_{index:02}_"
        trace[prefix + "ids"] = ids
        trace[prefix + "mask"] = mask
        trace[prefix + "layer_00"] = x
        for layer_index, layer in enumerate(model.distilbert.transformer.layer, 1):
            x = layer(x, mask.astype(mx.bool_))
            trace[prefix + f"layer_{layer_index:02}"] = x
        logits = model.classifier(nn.relu(model.pre_classifier(x[:, 0])))
        trace[prefix + "logits"] = logits
        trace[prefix + "probabilities"] = mx.softmax(logits, axis=-1, precise=True)
    mx.eval(*trace.values())
    with args.output.open("xb") as stream:
        np.savez(stream, **{name: np.asarray(value) for name, value in trace.items()})


if __name__ == "__main__":
    main()
