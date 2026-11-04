#!/usr/bin/env python3
"""Pinned CLIP processor, both towers, projections and similarity trace adapter."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def fixed_siglip_trace(args, texts):
    """Resolved fixed SigLIP2 checkpoint adapter; padding and scores differ from CLIP."""
    if args.backend == "reference":
        import torch
        import transformers
        from transformers import AutoTokenizer, SiglipImageProcessor, SiglipModel

        if transformers.__version__ != "4.57.1":
            raise ValueError("requires Transformers 4.57.1")
        torch.set_num_threads(4)
        model = (
            SiglipModel.from_pretrained(
                args.checkpoint,
                local_files_only=True,
                use_safetensors=True,
                attn_implementation="eager",
            )
            .cpu()
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        processor = SiglipImageProcessor.from_pretrained(args.checkpoint, local_files_only=True)
        with Image.open(args.image) as image:
            pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
        tokens = tokenizer(
            texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        )
        captured, handles = {}, []

        def capture(key):
            def hook(_module, _inputs, output):
                captured[key] = output.detach()

            return hook

        for name, tower in (("vision", model.vision_model), ("text", model.text_model)):
            handles.append(tower.embeddings.register_forward_hook(capture(name + "_layer_00")))
            for index, layer in enumerate(tower.encoder.layers, 1):
                handles.append(layer.register_forward_hook(capture(f"{name}_layer_{index:02}")))
            norm = tower.post_layernorm if name == "vision" else tower.final_layer_norm
            handles.append(norm.register_forward_hook(capture(name + "_last_hidden")))
        with torch.inference_mode():
            result = model(pixel_values=pixels, **tokens)
            trace = {
                "pixels": pixels,
                "ids": tokens["input_ids"],
                "logits_per_image": result.logits_per_image,
                "image_embeds": result.image_embeds,
                "text_embeds": result.text_embeds,
                "image_features": result.vision_model_output.pooler_output,
                "text_features": result.text_model_output.pooler_output,
                "probabilities": torch.sigmoid(result.logits_per_image),
                **captured,
            }
        for handle in handles:
            handle.remove()
        return {key: value.numpy() for key, value in trace.items()}
    import mlx.core as mx

    from mlx_one.models.classification.siglip_fixed import load_fixed_siglip
    from mlx_one.vision.vit_classification import preprocess

    bundle = load_fixed_siglip(args.checkpoint)
    encoded = bundle.tokenizer.encode_batch(texts)
    ids = mx.array([e.ids for e in encoded])
    pixels, _ = preprocess(args.image)
    trace = bundle.model(mx.array(pixels[None]), ids, trace=True)
    trace.update(pixels=mx.array(pixels[None].transpose(0, 3, 1, 2)), ids=ids)
    trace["probabilities"] = mx.sigmoid(trace["logits_per_image"])
    mx.eval(trace)
    return {key: np.asarray(value) for key, value in trace.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=["reference", "native"])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("image", type=Path)
    p.add_argument("labels", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--family", choices=("clip", "fixed-siglip"), default="clip")
    args = p.parse_args()
    labels = json.loads(args.labels.read_text())
    texts = [f"This is a photo of {label}." for label in labels]
    if args.family == "fixed-siglip":
        trace = fixed_siglip_trace(args, texts)
        with args.output.open("xb") as stream:
            np.savez(stream, **trace)
        return
    if args.backend == "reference":
        import torch
        import transformers
        from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

        if transformers.__version__ != "4.57.1":
            raise ValueError("requires Transformers 4.57.1")
        torch.set_num_threads(4)
        model = (
            CLIPModel.from_pretrained(
                args.checkpoint,
                local_files_only=True,
                use_safetensors=True,
                attn_implementation="eager",
            )
            .cpu()
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        processor = CLIPImageProcessor.from_pretrained(args.checkpoint, local_files_only=True)
        with Image.open(args.image) as image:
            pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
        tokens = tokenizer(texts, padding=True, return_tensors="pt")
        with torch.inference_mode():
            result = model(pixel_values=pixels, **tokens, output_hidden_states=True)
            image_features = model.visual_projection(result.vision_model_output.pooler_output)
            text_features = model.text_projection(result.text_model_output.pooler_output)
            text_embedding = model.text_model.embeddings(tokens["input_ids"])
            vision_embedding = model.vision_model.embeddings(pixels)
            trace = {
                "pixels": pixels,
                "ids": tokens["input_ids"],
                "mask": tokens["attention_mask"],
                "logits_per_image": result.logits_per_image,
                "image_embeds": result.image_embeds,
                "text_embeds": result.text_embeds,
                "image_features": image_features,
                "text_features": text_features,
                "vision_last_hidden": result.vision_model_output.last_hidden_state,
                "text_last_hidden": result.text_model_output.last_hidden_state,
                "text_embedding": text_embedding,
                "vision_embedding": vision_embedding,
            }
            trace.update(
                {
                    f"vision_layer_{i:02}": v
                    for i, v in enumerate(result.vision_model_output.hidden_states)
                }
            )
            trace.update(
                {
                    f"text_layer_{i:02}": v
                    for i, v in enumerate(result.text_model_output.hidden_states)
                }
            )
            trace["probabilities"] = torch.softmax(result.logits_per_image, -1)
        trace = {k: v.numpy() for k, v in trace.items()}
    else:
        import mlx.core as mx

        from mlx_one.models.classification.clip import load_clip
        from mlx_one.vision.clip_processing import preprocess_clip

        bundle = load_clip(args.checkpoint)
        bundle.tokenizer.enable_padding(pad_id=49407, pad_token="<|endoftext|>")
        tokens = bundle.tokenizer.encode_batch(texts)
        if any(len(e.ids) > 77 for e in tokens):
            raise ValueError("candidate prompt exceeds CLIP capacity")
        ids = mx.array([e.ids for e in tokens])
        mask = mx.array([e.attention_mask for e in tokens])
        pixels, _ = preprocess_clip(args.image)
        trace = bundle.model(mx.array(pixels[None]), ids, mask, trace=True)
        trace["probabilities"] = mx.softmax(trace["logits_per_image"], axis=-1, precise=True)
        trace.update(pixels=mx.array(pixels[None].transpose(0, 3, 1, 2)), ids=ids, mask=mask)
        mx.eval(trace)
        trace = {k: np.asarray(v) for k, v in trace.items()}
    with args.output.open("xb") as f:
        np.savez(f, **trace)


if __name__ == "__main__":
    main()
