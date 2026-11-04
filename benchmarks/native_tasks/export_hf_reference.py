#!/usr/bin/env python3
"""Export ViT CPU FP32 trace in the isolated Transformers 4.57.1 environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--processor-only", action="store_true")
    args = parser.parse_args()
    if transformers.__version__ != "4.57.1":
        raise ValueError("reference exporter requires Transformers 4.57.1")
    processor = ViTImageProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    if args.processor_only:
        with Image.open(args.image) as image:
            pixels = processor(images=image.convert("RGB"), return_tensors="np")["pixel_values"]
        with args.output.open("xb") as stream:
            np.savez(stream, pixels=pixels)
        return
    model = (
        ViTForImageClassification.from_pretrained(
            args.checkpoint,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="eager",
        )
        .to("cpu")
        .eval()
    )
    with Image.open(args.image) as image:
        pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
    captured = {}
    handles = [
        model.vit.embeddings.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("layer_00", output.detach())
        )
    ]
    for index, layer in enumerate(model.vit.encoder.layer, 1):
        handles.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, index=index: captured.__setitem__(
                    f"layer_{index:02}", output.detach()
                )
            )
        )
    handles.append(
        model.vit.layernorm.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("final_hidden", output.detach())
        )
    )
    with torch.inference_mode():
        output = model(pixels)
    for handle in handles:
        handle.remove()
    trace = {"pixels": pixels.numpy()}
    trace.update({key: value.numpy() for key, value in captured.items()})
    trace.update(
        logits=output.logits.numpy(), probabilities=torch.softmax(output.logits, -1).numpy()
    )
    with args.output.open("xb") as stream:
        np.savez(stream, **trace)


if __name__ == "__main__":
    main()
