#!/usr/bin/env python3
"""Pinned DPT processor, DINOv2 blocks/taps, depth neck/fusion/head trace."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=("reference", "native"))
    p.add_argument("checkpoint", type=Path)
    p.add_argument("image", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    stages = {}
    if args.backend == "reference":
        import torch
        import transformers
        from transformers import DepthAnythingForDepthEstimation, DPTImageProcessor

        assert transformers.__version__ == "4.57.1"
        torch.set_num_threads(4)
        model = (
            DepthAnythingForDepthEstimation.from_pretrained(
                args.checkpoint, local_files_only=True, attn_implementation="eager"
            )
            .cpu()
            .eval()
        )
        processor = DPTImageProcessor.from_pretrained(args.checkpoint, local_files_only=True)
        with Image.open(args.image) as image:
            size = image.size
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        stages["pixels"] = inputs["pixel_values"]
        handles = []

        def capture(key):
            def hook(module, inputs, output):
                stages[key] = output.detach()

            return hook

        handles.append(model.backbone.embeddings.register_forward_hook(capture("embeddings")))
        for i, layer in enumerate(model.backbone.encoder.layer, 1):
            handles.append(layer.register_forward_hook(capture(f"block_{i:02}")))
        taps = []
        handles.append(
            model.backbone.layernorm.register_forward_hook(lambda m, i, o: taps.append(o.detach()))
        )
        for i, layer in enumerate(model.neck.reassemble_stage.layers):
            handles.append(layer.register_forward_hook(capture(f"reassemble_{i}")))
        for i, layer in enumerate(model.neck.convs):
            handles.append(layer.register_forward_hook(capture(f"neck_conv_{i}")))
        for i, layer in enumerate(model.neck.fusion_stage.layers):
            handles.append(layer.register_forward_hook(capture(f"fusion_{i}")))
        with torch.inference_mode():
            output = model(**inputs)
            stages["predicted_depth"] = output.predicted_depth
            stages["resized_depth"] = processor.post_process_depth_estimation(
                output, target_sizes=[(size[1], size[0])]
            )[0]["predicted_depth"]
        for i, tap in zip((3, 6, 9, 12), taps, strict=True):
            stages[f"tap_{i:02}"] = tap
        for handle in handles:
            handle.remove()
        stages = {k: v.numpy() for k, v in stages.items()}
    else:
        import mlx.core as mx

        from mlx_one.models.depth.depth_anything import load_depth
        from mlx_one.models.depth.interpolation import resize
        from mlx_one.vision.depth_processing import preprocess_depth

        model = load_depth(args.checkpoint)
        pixels, size = preprocess_depth(args.image)
        stages = model(mx.array(pixels[None]))
        stages["resized_depth"] = resize(
            stages["predicted_depth"][..., None], (size[1], size[0]), mode="bicubic"
        )[0, :, :, 0]
        mx.eval(stages)
        stages = {k: np.asarray(v) for k, v in stages.items()}
        stages["pixels"] = pixels[None].transpose(0, 3, 1, 2)
    keys = (
        ["pixels", "embeddings"]
        + [f"block_{i:02}" for i in range(1, 13)]
        + [f"tap_{i:02}" for i in (3, 6, 9, 12)]
    )
    keys += [f"{name}_{i}" for name in ("reassemble", "neck_conv", "fusion") for i in range(4)]
    keys += ["predicted_depth", "resized_depth"]
    with args.output.open("xb") as stream:
        np.savez(stream, **{k: stages[k] for k in keys})


if __name__ == "__main__":
    main()
