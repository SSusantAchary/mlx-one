#!/usr/bin/env python3
"""Local labeled sentiment evaluation with matched reference/native inputs.

No dataset text is distributed by this script. Read a separately pinned JSON
list of {idx, sentence, label}; preserve outputs and hashes in local evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def wilson(correct, count):
    z = 1.959963984540054
    p = correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - half, center + half]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["reference", "native"])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if not cases or any(row["label"] not in (0, 1) for row in cases):
        raise ValueError("expected labeled binary sentiment examples")
    if args.backend == "reference":
        import torch
        import transformers
        from transformers import AutoTokenizer, DistilBertForSequenceClassification

        if transformers.__version__ != "4.57.1":
            raise ValueError("requires the pinned Transformers 4.57.1 environment")
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        model = (
            DistilBertForSequenceClassification.from_pretrained(
                args.checkpoint,
                local_files_only=True,
                use_safetensors=True,
                attn_implementation="eager",
            )
            .cpu()
            .eval()
        )

        def predict(texts):
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.inference_mode():
                return torch.softmax(model(**encoded).logits, dim=-1).numpy()

        def close():
            pass
    else:
        from mlx_one.classification import ClassificationEngine

        engine = ClassificationEngine(args.checkpoint)

        def predict(texts):
            result = engine.classify(texts)
            return np.asarray(
                [
                    [
                        next(item["score"] for item in row["labels"] if item["index"] == index)
                        for index in (0, 1)
                    ]
                    for row in result.predictions
                ],
                dtype=np.float32,
            )

        close = engine.close
    rows = []
    started = time.perf_counter()
    try:
        for offset in range(0, len(cases), 2):
            batch = cases[offset : offset + 2]
            scores = predict([item["sentence"] for item in batch])
            for item, score in zip(batch, scores, strict=True):
                rows.append(
                    {
                        "idx": item["idx"],
                        "label": item["label"],
                        "prediction": int(np.argmax(score)),
                        "scores": [float(x) for x in score],
                    }
                )
    finally:
        close()
    correct = sum(row["label"] == row["prediction"] for row in rows)
    report = {
        "backend": args.backend,
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "wilson_95_interval": wilson(correct, len(rows)),
        "batch_size": 2,
        "max_length": 512,
        "seconds": time.perf_counter() - started,
        "rows": rows,
    }
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
