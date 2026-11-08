"""Isolated publisher reference for OpenELM; never imported by production.

Uses the inspected modeling/configuration sources pinned to the checkpoint.
The caller supplies only campaign-owned local assets and native captures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    captured = np.load(args.native)
    ids = torch.from_numpy(captured["ids"]).long()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        output = model(ids, use_cache=False, output_hidden_states=True)
    expected = output.logits.float().numpy()
    actual = captured["logits"]
    difference = np.abs(actual - expected)
    labels = torch.from_numpy(captured["labels"]).long()
    nll = float(
        torch.nn.functional.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]), labels.reshape(-1)
        )
    )
    report = dict(
        torch=torch.__version__,
        max_abs_logits=float(difference.max()),
        mean_abs_logits=float(difference.mean()),
        reference_nll=nll,
        top1_agreement=float((actual.argmax(-1) == expected.argmax(-1)).mean()),
        layers=[],
    )
    for index, state in enumerate(output.hidden_states):
        reference = state.float().numpy()
        native = captured[f"hidden_{index}"]
        error = np.abs(native - reference)
        report["layers"].append(
            dict(
                index=index,
                max_abs=float(error.max()),
                mean_abs=float(error.mean()),
                reference_abs_max=float(np.abs(reference).max()),
            )
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
