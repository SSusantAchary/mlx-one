"""Isolated pinned publisher backbone/projection comparison, without production imports."""

import argparse
import json
from pathlib import Path


def main():
    import numpy as np
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    capture = np.load(args.native)
    model, loading = AutoModel.from_pretrained(
        str(args.model),
        local_files_only=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
        attn_implementation="eager",
    )
    assert not any(loading[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys"))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    recipe = json.loads((args.model / "config_sentence_transformers.json").read_text())
    tokenizer.pad_token_id = tokenizer.mask_token_id or tokenizer.eos_token_id
    projection = load_file(str(args.model / "1_Dense/model.safetensors"))["linear.weight"]
    records = []
    for index, sample in enumerate(json.loads(str(capture["samples"]))):
        query = sample["input_type"] == "query"
        limit = recipe["query_length"] if query else min(128, recipe["document_length"])
        tokens = tokenizer(
            [sample["text"], sample["text"]],
            truncation=True,
            max_length=limit - 1,
            padding="max_length" if query else True,
            return_tensors="pt",
        )
        prefix = tokenizer.convert_tokens_to_ids(
            recipe["query_prefix" if query else "document_prefix"]
        )
        ids, attention = tokens["input_ids"], tokens["attention_mask"]
        ids = torch.cat([ids[:, :1], torch.full((2, 1), prefix), ids[:, 1:]], dim=1)
        attention = torch.cat(
            [attention[:, :1], torch.ones((2, 1), dtype=attention.dtype), attention[:, 1:]], dim=1
        )
        if query and recipe["attend_to_expansion_tokens"]:
            attention.fill_(1)
        assert np.array_equal(ids[0].numpy(), capture[f"ids_{index}"])
        assert np.array_equal(attention[0].numpy(), capture[f"attention_{index}"])
        skip = tokenizer.convert_tokens_to_ids(recipe["skiplist_words"])
        scoring = (
            torch.ones_like(attention, dtype=torch.bool)
            if query
            else (attention.bool() & ~torch.isin(ids, torch.tensor(skip)))
        )
        assert np.array_equal(scoring[0].numpy(), capture[f"scoring_{index}"])
        with torch.inference_mode():
            hidden = model(
                input_ids=ids, attention_mask=attention, use_cache=False
            ).last_hidden_state
            features = torch.nn.functional.normalize(
                torch.nn.functional.linear(hidden, projection), dim=-1
            )
            features = features.masked_fill(~scoring[..., None], 0)
        expected = features[0].float().numpy()
        actual = capture[f"vectors_{index}"]
        error = np.abs(actual - expected)
        records.append(
            dict(
                input_type=sample["input_type"],
                tokens=len(actual),
                max_abs=float(error.max()),
                mean_abs=float(error.mean()),
                stock_forward_error=None,
                comparison_method="stock_forward",
            )
        )
    result = dict(
        transformers=transformers.__version__,
        torch=torch.__version__,
        reference_batch_size=2,
        loading_info=loading,
        comparisons=records,
        scope=(
            "independent tokenizer recipe and publisher backbone/projection; "
            "stock AutoModel forward with independent token recipe and bias-free projection"
        ),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
