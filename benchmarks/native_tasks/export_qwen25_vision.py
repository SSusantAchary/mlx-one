#!/usr/bin/env python3
"""Exact existing Q4 checkpoint's dense FP16 vision tower vs pinned source.

The vision weights are stored FP16, not quantized. This trace does not claim
parity of the packed Q4 language decoder or of another unquantized checkpoint.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=("reference", "native"))
    p.add_argument("checkpoint", type=Path)
    p.add_argument("processor_trace", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--compute-dtype", choices=("float16", "float32"), default="float16")
    args = p.parse_args()
    with np.load(args.processor_trace, allow_pickle=False) as f:
        pixels, grids = f["pixel_values"], f["image_grid_thw"]
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    trace = {}
    if args.backend == "reference":
        import torch
        from safetensors import safe_open
        from transformers import Qwen2_5_VLConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        torch.set_num_threads(4)
        config = Qwen2_5_VLConfig.from_dict(cfg).vision_config
        config._attn_implementation = "eager"
        # CPU construction also materializes nonpersistent rotary buffers.
        model = Qwen2_5_VisionTransformerPretrainedModel(config)
        weights = {}
        with safe_open(args.checkpoint / "model.safetensors", framework="pt") as f:
            for name in f.keys():
                if name.startswith("vision_tower."):
                    value = f.get_tensor(name)
                    if name.endswith("patch_embed.proj.weight"):
                        value = value.permute(0, 4, 1, 2, 3).contiguous()
                    assert value.dtype == torch.float16
                    if args.compute_dtype == "float32":
                        value = value.float()
                    weights[name.removeprefix("vision_tower.")] = value
        model.load_state_dict(weights, strict=True, assign=True)
        model.eval()
        handles = []

        def capture(name):
            def hook(module, inputs, output):
                trace[name] = output.detach()

            return hook

        handles.append(model.patch_embed.register_forward_hook(capture("patch_embed")))
        for i, block in enumerate(model.blocks):
            handles.append(block.register_forward_hook(capture(f"block_{i + 1:02}")))
        with torch.inference_mode():
            grid = torch.from_numpy(grids)
            index, lengths = model.get_window_index(grid)
            trace["window_index"] = index
            trace["merged"] = model(torch.from_numpy(pixels), grid)
        trace = {
            k: v.float().numpy() if v.is_floating_point() else v.numpy() for k, v in trace.items()
        }
    else:
        import mlx.core as mx

        from mlx_one.models.shared.vision import vision_position_ids
        from mlx_one.models.shared.vlm_weights import canonical_name, patch_weight
        from mlx_one.models.vision_language.qwen2_5_vl.config import Qwen2_5_VLConfig
        from mlx_one.models.vision_language.qwen2_5_vl.model import (
            Qwen2_5VisionTransformer,
            _window_plan,
        )

        config = Qwen2_5_VLConfig.from_dict(cfg).vision_config
        model = Qwen2_5VisionTransformer(config)
        original = mx.load(str(args.checkpoint / "model.safetensors"))
        weights = []
        for name, value in original.items():
            if name.startswith("vision_tower."):
                key = canonical_name(name, "qwen2_5_vl").removeprefix("visual.")
                if key == "patch_embed.proj.weight":
                    value = patch_weight(value, config.in_channels)
                if value.dtype != mx.float16:
                    raise ValueError("vision weights must be exact FP16, not quantized")
                if args.compute_dtype == "float32":
                    value = value.astype(mx.float32)
                weights.append((key, value))
        model.load_weights(weights, strict=True)
        model.eval()
        mx.eval(model.parameters())
        del original
        grid = grids.tolist()
        hidden = model.patch_embed(mx.array(pixels))
        trace["patch_embed"] = hidden
        positions, full_lengths = vision_position_ids(grid, config.spatial_merge_size)
        index, lengths = _window_plan(grid, config)
        trace["window_index"] = mx.array(index)
        group = config.spatial_merge_size**2
        hidden = hidden.reshape(-1, group, hidden.shape[-1])[mx.array(index)].reshape(
            -1, hidden.shape[-1]
        )
        positions = positions.reshape(-1, group, 2)[mx.array(index)].reshape(-1, 2)
        for i, block in enumerate(model.blocks):
            hidden = block(
                hidden, positions, full_lengths if i in config.fullatt_block_indexes else lengths
            )
            trace[f"block_{i + 1:02}"] = hidden
        merged = model.merger(hidden)
        reverse = sorted(range(len(index)), key=index.__getitem__)
        trace["merged"] = merged[mx.array(reverse)]
        mx.eval(trace)
        trace = {
            k: np.asarray(v.astype(mx.float32)) if v.dtype == mx.float16 else np.asarray(v)
            for k, v in trace.items()
        }
    keys = ["patch_embed", "window_index"] + [f"block_{i:02}" for i in range(1, 33)] + ["merged"]
    with args.output.open("xb") as stream:
        np.savez(stream, **{k: trace[k] for k in keys})


if __name__ == "__main__":
    main()
