#!/usr/bin/env python3
"""Model-specific BERT NER trace adapter for the shared pickle-free comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=["reference", "native"])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("cases", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("tasks", type=Path)
    p.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    args = p.parse_args()
    cases = json.loads(args.cases.read_text())
    trace = {}
    tasks = {"none": [], "simple": []}
    if args.backend == "reference":
        import torch
        import transformers
        from transformers import AutoTokenizer, BertForTokenClassification, pipeline

        if transformers.__version__ != "4.57.1":
            raise ValueError("requires Transformers 4.57.1")
        torch.set_num_threads(4)
        model = (
            BertForTokenClassification.from_pretrained(
                args.checkpoint,
                local_files_only=True,
                use_safetensors=True,
                attn_implementation="eager",
            )
            .cpu()
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        for i, offset in enumerate(range(0, len(cases), args.batch_size)):
            encoded = tokenizer(
                cases[offset : offset + args.batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")
            special = encoded.pop("special_tokens_mask")
            with torch.inference_mode():
                result = model(**encoded, output_hidden_states=True)
            prefix = f"case_{i:02}_"
            for name, value in [
                ("ids", encoded["input_ids"]),
                ("mask", encoded["attention_mask"]),
                ("types", encoded["token_type_ids"]),
                ("offsets", offsets),
                ("special", special),
            ]:
                trace[prefix + name] = value.numpy()
            for layer, value in enumerate(result.hidden_states):
                trace[prefix + f"layer_{layer:02}"] = value.numpy()
            trace[prefix + "logits"] = result.logits.numpy()
            trace[prefix + "probabilities"] = torch.softmax(result.logits, -1).numpy()
        for strategy in tasks:
            task = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy=strategy,
                device=-1,
            )
            tasks[strategy] = task(cases)
    else:
        import mlx.core as mx

        from mlx_one.classification.token_postprocessing import entities_from_scores
        from mlx_one.models.classification.bert_ner import load_bert_ner

        bundle = load_bert_ner(args.checkpoint)
        for i, offset in enumerate(range(0, len(cases), args.batch_size)):
            texts = cases[offset : offset + args.batch_size]
            encoded = bundle.tokenizer.encode_batch(texts)
            width = max(len(e.ids) for e in encoded)
            ids = mx.array([e.ids + [0] * (width - len(e.ids)) for e in encoded])
            masks = mx.array([[1] * len(e.ids) + [0] * (width - len(e.ids)) for e in encoded])
            types = mx.zeros_like(ids)
            result = bundle.model(ids, masks, types, output_hidden_states=True)
            probs = mx.softmax(result.logits, axis=-1, precise=True)
            mx.eval(probs, *result.hidden_states)
            prefix = f"case_{i:02}_"
            for name, value in [
                ("ids", ids),
                ("mask", masks),
                ("types", types),
                ("logits", result.logits),
                ("probabilities", probs),
            ]:
                trace[prefix + name] = np.asarray(value)
            trace[prefix + "offsets"] = np.asarray(
                [e.offsets + [(0, 0)] * (width - len(e.ids)) for e in encoded]
            )
            trace[prefix + "special"] = np.asarray(
                [e.special_tokens_mask + [1] * (width - len(e.ids)) for e in encoded]
            )
            for layer, value in enumerate(result.hidden_states):
                trace[prefix + f"layer_{layer:02}"] = np.asarray(value)
            for text, e, row in zip(texts, encoded, probs.tolist(), strict=True):
                for strategy in tasks:
                    tasks[strategy].append(
                        entities_from_scores(
                            text,
                            e,
                            row[: len(e.ids)],
                            bundle.labels,
                            aggregation_strategy=strategy,
                            decoder=bundle.tokenizer._tokenizer.decoder,
                        )
                    )
    with args.output.open("xb") as f:
        np.savez(f, **trace)

    def number(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    with args.tasks.open("x") as f:
        json.dump(tasks, f, default=number, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
