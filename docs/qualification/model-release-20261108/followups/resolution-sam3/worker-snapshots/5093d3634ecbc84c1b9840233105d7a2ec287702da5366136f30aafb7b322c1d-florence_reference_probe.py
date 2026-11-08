"""Compare Florence processor, visual/encoder states and publisher beam generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "reference"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    if args.backend == "native":
        import mlx.core as mx

        from mlx_one.ocr.generation import OCRForward
        from mlx_one.ocr.loading import load_ocr_model
        from mlx_one.ocr.processing import prepare_ocr
        from mlx_one.ocr.schemas import OCRInput

        bundle = load_ocr_model(args.model, offline=True)
        prepared = prepare_ocr(bundle, image, OCRInput(image, max_tokens=512))
        pixels, ids = prepared.kwargs["pixel_values"], prepared.kwargs["encoder_input_ids"]
        visual = bundle.model._encode_image(pixels)
        embeddings = bundle.model.get_input_embeddings(ids, pixels).inputs_embeds
        forward = OCRForward(bundle, prepared)
        arrays = dict(
            pixel_values=np.asarray(pixels),
            input_ids=np.asarray(ids),
            visual_features=np.asarray(visual),
            inputs_embeds=np.asarray(embeddings),
            encoder_states=np.asarray(forward.encoder_state),
        )
        mx.eval(visual, embeddings)
    else:
        import torch

        from florence_publisher.modeling_florence2 import Florence2ForConditionalGeneration
        from florence_publisher.processing_florence2 import Florence2Processor

        torch.set_num_threads(4)
        model = Florence2ForConditionalGeneration.from_pretrained(
            str(args.model),
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        processor = Florence2Processor.from_pretrained(str(args.model), local_files_only=True)
        inputs = processor(text="<OCR>", images=image, return_tensors="pt")
        with torch.inference_mode():
            visual = model._encode_image(inputs.pixel_values)
            embeddings, mask = model._merge_input_ids_with_image_features(
                visual, model.get_input_embeddings()(inputs.input_ids)
            )
            encoder = model.language_model.model.encoder(
                inputs_embeds=embeddings,
                attention_mask=mask,
                return_dict=True,
            ).last_hidden_state
            arrays = dict(
                pixel_values=inputs.pixel_values.numpy(),
                input_ids=inputs.input_ids.numpy(),
                visual_features=visual.numpy(),
                inputs_embeds=embeddings.numpy(),
                encoder_states=encoder.numpy(),
            )
            generated = model.generate(**inputs, num_beams=3, do_sample=False, max_new_tokens=512)
        text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        args.output.with_suffix(".json").write_text(
            json.dumps(dict(text=text, tokens=generated.tolist())) + "\n"
        )
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(
        json.dumps(
            dict(backend=args.backend, shapes={k: list(v.shape) for k, v in arrays.items()})
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
