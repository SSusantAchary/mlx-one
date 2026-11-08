"""Independent publisher Qwen3 embedding forward, tokenizer and final-token pooling."""

import argparse
import json
from pathlib import Path


def main():
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    captured = json.loads(args.capture.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    model, loading = AutoModel.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    assert not any(loading[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys"))
    model.eval()
    vectors = []
    for start in range(0, len(captured["texts"]), 4):
        texts = captured["texts"][start : start + 4]
        tokens = tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        for offset, mask in enumerate(tokens.attention_mask):
            ids = tokens.input_ids[offset][mask.bool()].tolist()
            assert ids == captured["ids"][start + offset]
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            pooled = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)
        vectors.extend(pooled.numpy().tolist())
    a, b = np.asarray(captured["vectors"]), np.asarray(vectors)
    result = dict(
        torch=torch.__version__,
        transformers=transformers.__version__,
        loading_info=loading,
        samples=len(vectors),
        tokenizer_exact=True,
        max_abs=float(np.abs(a - b).max()),
        mean_abs=float(np.abs(a - b).mean()),
        minimum_cosine=float(
            np.min(
                np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
            )
        ),
        reference_execution="float32",
        checkpoint_storage="bfloat16",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
