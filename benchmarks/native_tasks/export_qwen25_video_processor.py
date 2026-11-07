#!/usr/bin/env python3
"""Qwen pinned fast video processor vs existing native per-frame packing."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=("reference", "native"))
    p.add_argument("checkpoint", type=Path)
    p.add_argument("frames", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    frames = np.load(args.frames, allow_pickle=False)
    if args.backend == "reference":
        from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

        processor = Qwen2VLVideoProcessor(
            size={"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 256}
        )
        out = processor(videos=[frames], do_sample_frames=False, return_tensors="pt")
        values = {k: np.asarray(out[k]) for k in ["pixel_values_videos", "video_grid_thw"]}
    else:
        import mlx.core as mx

        from mlx_one.models.vision_language.qwen2_5_vl.config import Qwen2_5_VLConfig
        from mlx_one.text.tokenizers import HFTokenizerAdapter
        from mlx_one.vision.processing import Qwen25ImageProcessor
        from mlx_one.vision.video_processing import pack_video_frames

        config = Qwen2_5_VLConfig.from_dict(
            json.loads((args.checkpoint / "config.json").read_text())
        )
        metadata = json.loads((args.checkpoint / "preprocessor_config.json").read_text())
        metadata["max_pixels"] = 28 * 28 * 256
        processor = Qwen25ImageProcessor(
            HFTokenizerAdapter.from_directory(args.checkpoint), config, metadata=metadata
        )
        pixels, grid = pack_video_frames(frames, processor)
        mx.eval(pixels)
        values = {"pixel_values_videos": np.asarray(pixels), "video_grid_thw": np.asarray(grid)}
    with args.output.open("xb") as stream:
        np.savez(stream, **values)


if __name__ == "__main__":
    main()
