#!/usr/bin/env python3
"""Pinned ViT eager CPU trace. Requires the separate reference environment."""

import argparse
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import ViTForImageClassification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("pixels", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if transformers.__version__ != "4.57.1":
        raise ValueError("requires Transformers 4.57.1")
    model = (
        ViTForImageClassification.from_pretrained(
            args.checkpoint,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="eager",
        )
        .cpu()
        .eval()
    )
    pixels = torch.from_numpy(np.load(args.pixels, allow_pickle=False))
    trace = {"pixels": pixels}
    handles = []

    def capture(key):
        def hook(_module, _args, output):
            trace[key] = output.detach()

        return hook

    handles.append(model.vit.embeddings.register_forward_hook(capture("layer_00")))
    for i, block in enumerate(model.vit.encoder.layer):
        handles.append(block.register_forward_hook(capture(f"layer_{i + 1:02}")))
        if i not in (3, 4):
            continue
        prefix = f"block_{i + 1:02}_"

        def input_hook(_module, inputs, prefix=prefix):
            trace[prefix + "input"] = inputs[0].detach()

        handles.append(block.register_forward_pre_hook(input_hook))

        def attention_input_hook(_module, inputs, prefix=prefix):
            trace[prefix + "attention_context"] = inputs[0].detach()

        handles.append(block.attention.output.dense.register_forward_pre_hook(attention_input_hook))
        for name, module in [
            ("ln1", block.layernorm_before),
            ("query", block.attention.attention.query),
            ("key", block.attention.attention.key),
            ("value", block.attention.attention.value),
            ("attention_output", block.attention.output.dense),
            ("ln2", block.layernorm_after),
            ("mlp1", block.intermediate.dense),
            ("gelu", block.intermediate.intermediate_act_fn),
            ("mlp2", block.output.dense),
        ]:
            handles.append(module.register_forward_hook(capture(prefix + name)))
    handles.append(model.vit.layernorm.register_forward_hook(capture("final_hidden")))
    with torch.inference_mode():
        output = model(pixels)
        trace["logits"] = output.logits
        trace["probabilities"] = torch.softmax(output.logits, dim=-1)
        for i in (4, 5):
            prefix = f"block_{i:02}_"
            trace[prefix + "residual1"] = (
                trace[prefix + "input"] + trace[prefix + "attention_output"]
            )
            trace[prefix + "residual2"] = trace[prefix + "residual1"] + trace[prefix + "mlp2"]

            def split(t):
                return t.reshape(t.shape[0], t.shape[1], 12, 64).permute(0, 2, 1, 3)

            q, k = split(trace[prefix + "query"]), split(trace[prefix + "key"])
            scores = (q @ k.transpose(-1, -2)) * (64**-0.5)
            trace[prefix + "scores"] = scores
            trace[prefix + "probabilities"] = torch.softmax(scores, dim=-1, dtype=torch.float32)
    for h in handles:
        h.remove()
    with args.output.open("xb") as f:
        np.savez(f, **{k: v.detach().numpy() for k, v in trace.items()})


if __name__ == "__main__":
    main()
