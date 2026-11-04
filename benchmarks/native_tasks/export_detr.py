#!/usr/bin/env python3
"""Pinned real-checkpoint DETR processor/backbone/encoder/decoder/box trace."""

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
    if args.backend == "reference":
        import torch
        import transformers
        from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

        assert transformers.__version__ == "4.57.1"
        torch.set_num_threads(4)
        cfg = DetrConfig.from_pretrained(args.checkpoint, local_files_only=True)
        cfg.use_pretrained_backbone = False  # full checkpoint owns every backbone weight
        model = (
            DetrForObjectDetection.from_pretrained(
                args.checkpoint, config=cfg, local_files_only=True, attn_implementation="eager"
            )
            .cpu()
            .eval()
        )
        processor = DetrImageProcessor.from_pretrained(args.checkpoint, local_files_only=True)
        with Image.open(args.image) as f:
            image = f.convert("RGB")
            size = image.size
            inputs = processor(images=image, return_tensors="pt")
        captured, handles = {}, []

        def capture(key, first=False):
            def hook(module, inputs, output):
                captured[key] = output[0] if first else output

            return hook

        for i in range(1, 5):
            handles.append(
                getattr(model.model.backbone.conv_encoder.model, f"layer{i}").register_forward_hook(
                    capture(f"backbone_{i}")
                )
            )
        handles.append(
            model.model.input_projection.register_forward_hook(capture("projection_nchw"))
        )
        handles.append(
            model.model.backbone.position_embedding.register_forward_hook(capture("positions_nchw"))
        )
        for name in ("encoder", "decoder"):
            for i, layer in enumerate(getattr(model.model, name).layers, 1):
                handles.append(layer.register_forward_hook(capture(f"{name}_{i:02}", True)))
        handles.append(model.model.decoder.layernorm.register_forward_hook(capture("decoder_norm")))
        with torch.inference_mode():
            output = model(**inputs)
            post = processor.post_process_object_detection(
                output, threshold=0.9, target_sizes=torch.tensor([[size[1], size[0]]])
            )[0]
            stages = {
                "pixels": inputs["pixel_values"],
                "pixel_mask": inputs["pixel_mask"],
                **captured,
                "logits": output.logits,
                "pred_boxes": output.pred_boxes,
                **{"post_" + k: v for k, v in post.items()},
            }
        for handle in handles:
            handle.remove()
        stages["projected"] = stages.pop("projection_nchw").flatten(2).transpose(1, 2)
        stages["positions"] = stages.pop("positions_nchw").flatten(2).transpose(1, 2)
        feature_shape = stages["backbone_4"].shape[-2:]
        stages["feature_mask"] = torch.nn.functional.interpolate(
            inputs["pixel_mask"][None].float(), size=feature_shape
        )[0].bool()
        stages = {k: v.detach().numpy() for k, v in stages.items()}
    else:
        import mlx.core as mx

        from mlx_one.models.detection.detr import load_detr
        from mlx_one.vision.detr_processing import postprocess_detr, preprocess_detr

        model = load_detr(args.checkpoint)
        pixels, mask, size = preprocess_detr(args.image)
        stages = model(mx.array(pixels[None]), mx.array(mask[None]))
        mx.eval(stages)
        stages = {k: np.asarray(v) for k, v in stages.items()}
        stages.update(pixels=pixels[None].transpose(0, 3, 1, 2), pixel_mask=mask[None])
        stages.update(
            {
                "post_" + k: v
                for k, v in postprocess_detr(stages["logits"], stages["pred_boxes"], size).items()
            }
        )
    # Stable semantic order makes the first divergent boundary meaningful.
    keys = ["pixels", "pixel_mask"] + [f"backbone_{i}" for i in range(1, 5)]
    keys += ["feature_mask", "positions", "projected"]
    keys += [f"{name}_{i:02}" for name in ("encoder", "decoder") for i in range(1, 7)]
    keys += ["decoder_norm", "logits", "pred_boxes", "post_scores", "post_labels", "post_boxes"]
    with args.output.open("xb") as stream:
        np.savez(stream, **{k: stages[k] for k in keys})


if __name__ == "__main__":
    main()
