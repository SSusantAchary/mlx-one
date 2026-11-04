#!/usr/bin/env python3
"""Source vs existing native Qwen2.5 document processor and prompt trace."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=("reference", "native"))
    p.add_argument("checkpoint", type=Path)
    p.add_argument("image", type=Path)
    p.add_argument("question")
    p.add_argument("output", type=Path)
    p.add_argument("--max-pixels", type=int, default=28 * 28 * 256)
    args = p.parse_args()
    if args.backend == "reference":
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            args.checkpoint, local_files_only=True, use_fast=False, max_pixels=args.max_pixels
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "\n" + args.question}],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=False
        )
        with Image.open(args.image) as image:
            inputs = processor(text=[prompt], images=[image.convert("RGB")], return_tensors="np")
        trace = {
            k: np.asarray(inputs[k])
            for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        }
    else:
        import mlx.core as mx

        from mlx_one.models.vision_language.qwen2_5_vl.config import Qwen2_5_VLConfig
        from mlx_one.text.chat import ChatTemplate
        from mlx_one.text.tokenizers import HFTokenizerAdapter
        from mlx_one.vision.processing import Qwen25ImageProcessor

        config = Qwen2_5_VLConfig.from_dict(
            json.loads((args.checkpoint / "config.json").read_text())
        )
        tokenizer = HFTokenizerAdapter.from_directory(args.checkpoint)
        metadata = json.loads((args.checkpoint / "preprocessor_config.json").read_text())
        metadata["max_pixels"] = args.max_pixels
        processor = Qwen25ImageProcessor(tokenizer, config, metadata=metadata)
        prompt = ChatTemplate.from_directory(args.checkpoint, "qwen2_5_vl", tokenizer).render(
            [{"role": "user", "content": "<image>\n" + args.question}], add_vision_id=False
        )
        inputs = processor.prepare(prompt, [args.image])
        mx.eval(inputs.input_ids, inputs.pixel_values)
        trace = {
            "input_ids": np.asarray(inputs.input_ids),
            "pixel_values": np.asarray(inputs.pixel_values),
            "image_grid_thw": np.asarray(inputs.image_grid_thw),
            "attention_mask": np.ones(inputs.input_ids.shape, np.int64),
        }
    with args.output.open("xb") as stream:
        np.savez(stream, **trace)
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(
            {"prompt": prompt, "max_pixels": args.max_pixels, "backend": args.backend},
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
