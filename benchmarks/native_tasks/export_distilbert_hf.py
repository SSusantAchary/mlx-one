#!/usr/bin/env python3
"""Capture pinned DistilBERT tokenizer, hidden states and scores on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, DistilBertForSequenceClassification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if not 2 <= args.max_length <= 512:
        raise ValueError("max_length must be from 2 through 512")
    if transformers.__version__ != "4.57.1":
        raise ValueError("requires Transformers 4.57.1")
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
    torch.set_num_threads(4)
    trace = {}
    cases = json.loads(args.cases.read_text())
    with torch.inference_mode():
        for index, offset in enumerate(range(0, len(cases), args.batch_size)):
            texts = cases[offset : offset + args.batch_size]
            tokens = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            result = model(**tokens, output_hidden_states=True)
            prefix = f"case_{index:02}_"
            trace[prefix + "ids"] = tokens["input_ids"].numpy()
            trace[prefix + "mask"] = tokens["attention_mask"].numpy()
            for layer, value in enumerate(result.hidden_states):
                trace[prefix + f"layer_{layer:02}"] = value.numpy()
            trace[prefix + "logits"] = result.logits.numpy()
            trace[prefix + "probabilities"] = torch.softmax(result.logits, -1).numpy()
    with args.output.open("xb") as stream:
        np.savez(stream, **trace)


if __name__ == "__main__":
    main()
