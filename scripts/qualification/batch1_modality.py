"""Separate-process T5/ViT reference and native checkpoint verification.

The selected mode imports exactly one neural runtime. Reference dependencies are
never imported by the native mode or installed into the inference distribution.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np


def compare(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return dict(status="FAIL", actual_shape=list(a.shape), expected_shape=list(b.shape))
    error = np.abs(a - b)
    return dict(
        status="PASS" if np.allclose(a, b, atol=1e-4, rtol=1e-4) else "FAIL",
        max_abs_error=float(error.max()),
        mean_abs_error=float(error.mean()),
    )


def reference(args):
    import torch
    from transformers import (
        AutoTokenizer,
        T5ForConditionalGeneration,
        ViTForImageClassification,
        ViTImageProcessor,
    )

    torch.set_num_threads(4)
    torch.manual_seed(0)
    if args.task == "t5":
        model = T5ForConditionalGeneration.from_pretrained(args.model, local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        texts = [
            "translate English to German: The house is wonderful.",
            "translate English to French: The book is on the table.",
        ]
        inputs = tokenizer(texts, padding=True, return_tensors="pt")
        decoder = torch.tensor([[0, 2182, 1103, 35], [0, 2182, 1103, 0]])
        with torch.inference_mode():
            result = model(**inputs, decoder_input_ids=decoder)
            generated = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        arrays = {name: value.numpy() for name, value in inputs.items()}
        arrays.update(
            decoder_input_ids=decoder.numpy(),
            logits=result.logits.numpy(),
            encoder_hidden=result.encoder_last_hidden_state.numpy(),
            decoder_hidden=result.decoder_hidden_states[-1].numpy()
            if result.decoder_hidden_states
            else np.empty((0,)),
            generation=generated.numpy(),
        )
        metadata = dict(
            texts=texts, generated_texts=tokenizer.batch_decode(generated, skip_special_tokens=True)
        )
    else:
        from PIL import Image

        model = ViTForImageClassification.from_pretrained(args.model, local_files_only=True).eval()
        processor = ViTImageProcessor.from_pretrained(args.model, local_files_only=True)
        images = [Image.open(args.fixtures / f"{i}.jpg").convert("RGB") for i in range(2)]
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            result = model(**inputs, output_hidden_states=True)
            hidden = model.vit(**inputs).last_hidden_state
        arrays = dict(
            pixel_values=inputs.pixel_values.numpy(),
            logits=result.logits.numpy(),
            embedding_hidden=result.hidden_states[0].numpy(),
            first_layer=result.hidden_states[1].numpy(),
            hidden=hidden.numpy(),
        )
        metadata = dict(
            dataset=json.loads((args.fixtures / "dataset.json").read_text()),
            top5=torch.argsort(result.logits, descending=True)[:, :5].tolist(),
        )
        metadata["attention_implementation"] = model.config._attn_implementation
        eager = ViTForImageClassification.from_pretrained(
            args.model, local_files_only=True, attn_implementation="eager"
        ).eval()
        with torch.inference_mode():
            eager_hidden = eager.vit(**inputs).last_hidden_state
        arrays["hidden_eager"] = eager_hidden.numpy()
        metadata["reference_eager_vs_default_max_error"] = float(
            torch.max(torch.abs(eager_hidden - hidden))
        )
        # Independent higher-precision oracle distinguishes backend rounding
        # from an implementation mismatch; production inference stays FP32 MLX.
        eager = eager.double()
        with torch.inference_mode():
            high = eager.vit(**{k: v.double() for k, v in inputs.items()}).last_hidden_state
        arrays["hidden_fp64"] = high.numpy()
        metadata["publisher_fp32_rounding_max_error"] = float(
            torch.max(torch.abs(hidden.double() - high))
        )
    np.savez_compressed(args.output, **arrays)
    source = Path(inspect.getfile(type(model)))
    metadata.update(
        torch=torch.__version__,
        transformers=importlib.metadata.version("transformers"),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        parameter_count=sum(p.numel() for p in model.parameters()),
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def native(args):
    import mlx.core as mx

    expected = np.load(args.reference)
    metadata = json.loads(args.reference.with_suffix(".json").read_text())
    checks, benchmarks = {}, []
    tick = time.perf_counter()
    if args.task == "t5":
        from mlx_one.seq2seq import generate, load_seq2seq_model

        bundle = load_seq2seq_model(args.model, offline=True)
        mx.eval(bundle.model.parameters())
        load_time = time.perf_counter() - tick
        rows = [bundle.tokenizer.encode(t, add_special_tokens=True) for t in metadata["texts"]]
        ids, mask = bundle.tokenizer.pad(rows)
        checks["input_ids"] = compare(ids, expected["input_ids"])
        checks["attention_mask"] = compare(mask, expected["attention_mask"])
        inputs = mx.array(ids)
        result = bundle.model(
            inputs,
            attention_mask=mx.array(mask),
            decoder_input_ids=mx.array(expected["decoder_input_ids"]),
        )
        checks["encoder_hidden"] = compare(
            np.array(result.encoder_last_hidden_state), expected["encoder_hidden"]
        )
        checks["teacher_forced_logits"] = compare(np.array(result.logits), expected["logits"])
        generated = bundle.model.greedy(inputs, attention_mask=mx.array(mask), max_new_tokens=24)
        checks["greedy_generation"] = compare(np.array(generated), expected["generation"])
        public = generate(bundle, metadata["texts"], max_new_tokens=24)
        checks["public_pipeline"] = dict(
            status="PASS" if list(public.texts) == metadata["generated_texts"] else "FAIL"
        )
        for repeat in range(6):
            start = time.perf_counter()
            result_public = generate(bundle, metadata["texts"], max_new_tokens=24)
            mx.synchronize()
            elapsed = time.perf_counter() - start
            if repeat:
                benchmarks.append(
                    dict(seconds=elapsed, emitted_tokens=sum(map(len, result_public.token_ids)))
                )
        encoder_tick = time.perf_counter()
        encoded, encoder_mask = bundle.model.encode(inputs, mx.array(mask))
        mx.eval(encoded)
        encoder_seconds = time.perf_counter() - encoder_tick
        decoder_tick = time.perf_counter()
        first = bundle.model(
            inputs,
            attention_mask=encoder_mask,
            encoder_outputs=encoded,
            decoder_input_ids=mx.full(
                (len(ids), 1), bundle.model.config.decoder_start_token_id, mx.int32
            ),
        )
        mx.eval(first.logits)
        breakdown = dict(
            encoder_seconds=encoder_seconds,
            actual_encoder_tokens=sum(map(sum, mask)),
            first_decoder_token_seconds=time.perf_counter() - decoder_tick,
            cache="none; bounded decoder prefix recomputation; encoder reused",
        )
        example = public.to_dict()
    else:
        from mlx_one.classification.image import (
            classify_image,
            load_image_classification_model,
            prepare_image,
        )

        bundle = load_image_classification_model(args.model, offline=True)
        mx.eval(bundle.model.parameters())
        load_time = time.perf_counter() - tick
        images = [args.fixtures / f"{i}.jpg" for i in range(2)]
        pixels = np.stack(
            [prepare_image(image, bundle.model.config, bundle.processor) for image in images]
        )
        checks["processor"] = compare(pixels, expected["pixel_values"])
        result = bundle.model(mx.array(pixels))
        initial = bundle.model.vit.embeddings(mx.array(pixels))
        first = bundle.model.vit.encoder.layer[0](initial)
        checks.update(
            logits=compare(np.array(result.logits), expected["logits"]),
            embedding_hidden=compare(np.array(initial), expected["embedding_hidden"]),
            first_layer=compare(np.array(first), expected["first_layer"]),
            hidden=compare(np.array(result.last_hidden_state), expected["hidden"]),
        )
        if "hidden_fp64" in expected:
            exact = expected["hidden_fp64"]
            cpu_error = float(np.max(np.abs(expected["hidden"] - exact)))
            native_error = float(np.max(np.abs(np.array(result.last_hidden_state) - exact)))
            checks["fp64_rounding_diagnostic"] = dict(
                status="PASS" if native_error <= 2 * cpu_error + 1e-5 else "FAIL",
                publisher_fp32_max_error=cpu_error,
                native_fp32_max_error=native_error,
                bound=(
                    "2 * publisher FP32-vs-FP64 max error + 1e-5; independent rounding diagnostic"
                ),
            )
        top = np.argsort(-np.array(result.logits), axis=-1)[:, :5]
        checks["top5_parity"] = dict(status="PASS" if top.tolist() == metadata["top5"] else "FAIL")
        public = classify_image(bundle, images)
        checks["public_pipeline"] = dict(
            status="PASS"
            if [[row["index"] for row in pred["labels"]] for pred in public.predictions]
            == metadata["top5"]
            else "FAIL"
        )
        for batch in (1, 8):
            rows = [images[0]] * batch
            for repeat in range(6):
                start = time.perf_counter()
                classify_image(bundle, rows)
                mx.synchronize()
                if repeat:
                    benchmarks.append(dict(batch_size=batch, seconds=time.perf_counter() - start))
        breakdown = dict(
            scope="full loaded-model Python pipeline including image processing",
            dataset=metadata["dataset"],
        )
        example = public.to_dict()
    forbidden = [
        name
        for name in (
            "torch",
            "transformers",
            "sentence_transformers",
            "FlagEmbedding",
            "kokoro",
            "mlx_lm",
            "mlx_vlm",
        )
        if name in sys.modules
    ]
    checks["runtime_independence"] = dict(
        status="PASS" if not forbidden else "FAIL", imported=forbidden
    )
    report = dict(
        status="PASS" if all(check["status"] == "PASS" for check in checks.values()) else "FAIL",
        checks=checks,
        load_seconds=load_time,
        benchmark_samples=benchmarks,
        breakdown=breakdown,
        example=example,
        device=mx.device_info(),
        peak_mlx_bytes=mx.get_peak_memory(),
    )
    from batch1_interfaces import check_interfaces, package_identity

    identity = package_identity()
    report["package_identity"] = identity
    if args.check_interfaces:
        report["interfaces"] = check_interfaces(
            bundle,
            "text-to-text-generation" if args.task == "t5" else "image-classification",
            images=None if args.task == "t5" else images,
            output_dir=args.output.parent,
        )
        report["installed_distribution"] = dict(
            status="PASS"
            if identity["installed_in_environment"] and not identity["forbidden_distributions"]
            else "FAIL"
        )
        if (
            any(r["status"] != "PASS" for r in report["interfaces"].values())
            or report["installed_distribution"]["status"] != "PASS"
        ):
            report["status"] = "FAIL"
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], checks=checks)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["reference", "native"], required=True)
    parser.add_argument("--task", choices=["t5", "vit"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-interfaces", action="store_true")
    args = parser.parse_args()
    (reference if args.mode == "reference" else native)(args)


if __name__ == "__main__":
    main()
