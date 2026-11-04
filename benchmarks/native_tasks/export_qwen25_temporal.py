#!/usr/bin/env python3
"""Pinned-source regression oracle for Qwen2.5 time positions/frame attention.

Tiny FP32 weights validate verified temporal mathematics defects only. They are
not task-quality or real-checkpoint qualification evidence.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CONFIG = dict(
    depth=2,
    hidden_size=32,
    intermediate_size=64,
    num_heads=4,
    in_channels=3,
    out_hidden_size=16,
    patch_size=2,
    temporal_patch_size=2,
    spatial_merge_size=2,
    window_size=8,
    fullatt_block_indexes=[1],
    tokens_per_second=2,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=("reference", "native"))
    p.add_argument("weights", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    pixels = np.random.default_rng(20261103).normal(size=(32, 24)).astype(np.float32)
    arrays = {}
    if args.backend == "reference":
        import torch
        from transformers import Qwen2_5_VLConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
            Qwen2_5_VLModel,
        )

        torch.manual_seed(20261103)
        cfg = Qwen2_5_VLConfig(vision_config=CONFIG)
        cfg.vision_config._attn_implementation = "eager"
        model = Qwen2_5_VisionTransformerPretrainedModel(cfg.vision_config).cpu().eval()
        with args.weights.open("xb") as stream:
            np.savez(stream, **{k: v.numpy() for k, v in model.state_dict().items()})
        with torch.inference_mode():
            arrays["video_merged"] = model(
                torch.from_numpy(pixels), torch.tensor([[2, 4, 4]])
            ).numpy()
        ids = torch.tensor(
            [
                [42, cfg.vision_start_token_id]
                + [cfg.video_token_id] * 8
                + [cfg.vision_end_token_id, 43]
            ]
        )
        for i, interval in enumerate((None, [0.5], [1.0], [2.5])):
            pos, delta = Qwen2_5_VLModel.get_rope_index(
                SimpleNamespace(config=cfg),
                ids,
                video_grid_thw=torch.tensor([[2, 4, 4]]),
                second_per_grid_ts=interval,
            )
            arrays[f"positions_{i}"], arrays[f"delta_{i}"] = pos.numpy(), delta.numpy()
    else:
        import mlx.core as mx

        from mlx_one.models.vision_language.qwen2_5_vl.config import Qwen2_5_VLVisionConfig
        from mlx_one.models.vision_language.qwen2_5_vl.model import Qwen2_5VisionTransformer
        from mlx_one.models.vision_language.qwen2_5_vl.positions import video_position_ids

        cfg = Qwen2_5_VLVisionConfig.from_dict(CONFIG)
        model = Qwen2_5VisionTransformer(cfg)
        weights = []
        with np.load(args.weights, allow_pickle=False) as stored:
            for key in stored.files:
                value = stored[key]
                if value.ndim == 5:
                    value = value.reshape(value.shape[0], -1)
                key = key.replace("merger.mlp.0.", "merger.mlp_0.").replace(
                    "merger.mlp.2.", "merger.mlp_2."
                )
                weights.append((key, mx.array(value)))
        model.load_weights(weights, strict=True)
        model.eval()
        merged, _ = model(mx.array(pixels), [[2, 4, 4]])
        arrays["video_merged"] = merged
        ids = mx.array([[42, 151652] + [151656] * 8 + [151653, 43]])
        types = mx.array([[0, 0] + [2] * 8 + [0, 0]])
        for i, interval in enumerate((None, [0.5], [1.0], [2.5])):
            pos, delta = video_position_ids(
                ids,
                mm_token_type_ids=types,
                image_grid_thw=None,
                video_grid_thw=[[2, 4, 4]],
                spatial_merge_size=2,
                attention_mask=None,
                second_per_grid_ts=interval,
                tokens_per_second=2,
            )
            arrays[f"positions_{i}"], arrays[f"delta_{i}"] = pos, delta
        mx.eval(arrays)
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
    with args.output.open("xb") as stream:
        np.savez(stream, **arrays)


if __name__ == "__main__":
    main()
