"""Compare bounded LFM2 native Q4 NLL against its pinned publisher source weights."""

import argparse
import json
from pathlib import Path


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, loading = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        local_files_only=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
        attn_implementation="eager",
    )
    assert not any(loading[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys"))
    model.eval()
    rows = []
    for row in json.loads(args.samples.read_text()):
        ids = torch.tensor([row["ids"]], dtype=torch.long)
        with torch.inference_mode():
            logits = model(ids[:, :-1], use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten())
        rows.append(
            dict(
                reference_nll=float(loss),
                native_nll=row["nll"],
                nll_delta=row["nll"] - float(loss),
                target_tokens=ids.shape[1] - 1,
            )
        )
    result = dict(
        torch=torch.__version__,
        transformers=transformers.__version__,
        loading_info=loading,
        rows=rows,
        scope="native affine Q4 versus publisher source at FP32; five BOS-prefixed excerpts",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
