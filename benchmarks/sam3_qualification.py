"""Explicit native SAM3 smoke runner; passing smoke is NOT parity qualification."""

import argparse
import gc
import json
import resource
import time


def save_arrays(path, **arrays):
    if path:
        import numpy as np

        with open(path, "xb") as destination:
            np.savez_compressed(destination, **arrays)


def reference(args):
    """Reference only: never imported by the package or production CLI."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import (
        Sam3Model,
        Sam3Processor,
        Sam3TrackerModel,
        Sam3TrackerProcessor,
        Sam3VideoModel,
        Sam3VideoProcessor,
    )

    torch.set_num_threads(4)
    classes = {
        "concept": (Sam3Model, Sam3Processor),
        "interactive": (Sam3TrackerModel, Sam3TrackerProcessor),
        "video": (Sam3VideoModel, Sam3VideoProcessor),
    }
    model_class, processor_class = classes[args.task]
    start = time.perf_counter()
    model = (
        model_class.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa"
        )
        .eval()
        .to(args.reference_device)
    )
    processor = processor_class.from_pretrained(args.model, local_files_only=True)
    image = Image.open(args.image).convert("RGB")
    print(
        json.dumps({"stage": "reference_loaded", "seconds": time.perf_counter() - start}),
        flush=True,
    )
    start = time.perf_counter()
    with torch.inference_mode():
        if args.task == "concept":
            inputs = processor(images=image, text=args.text, return_tensors="pt")
            outputs = model(**inputs)
            result = processor.post_process_instance_segmentation(
                outputs, threshold=0.3, target_sizes=[(image.height, image.width)]
            )[0]
            arrays = {key: result[key].cpu().numpy() for key in ("masks", "scores", "boxes")}
            arrays.update(
                pixel_values=inputs.pixel_values.numpy(),
                pred_masks=outputs.pred_masks.numpy(),
                pred_boxes=outputs.pred_boxes.numpy(),
                pred_logits=outputs.pred_logits.numpy(),
                presence_logits=outputs.presence_logits.numpy(),
            )
        elif args.task == "interactive":
            inputs = processor(
                images=image,
                input_points=[[[[image.width / 2, image.height / 2]]]],
                input_labels=[[[1]]],
                return_tensors="pt",
            )
            outputs = model(**inputs)
            masks = processor.post_process_masks(outputs.pred_masks, inputs.original_sizes)[0][0]
            arrays = dict(
                masks=masks.numpy(),
                predicted_iou=outputs.iou_scores[0, 0].numpy(),
                logits=outputs.pred_masks[0, 0].numpy(),
                pixel_values=inputs.pixel_values.numpy(),
            )
        else:
            from transformers.models.sam3_video import modeling_sam3_video

            from mlx_one.segmentation.postprocessing import connected_components, mask_nms

            class HostCV:
                @staticmethod
                def generic_nms(ious, scores, threshold, **kwargs):
                    kept = mask_nms(ious.cpu().numpy(), scores.cpu().numpy(), threshold)
                    return torch.from_numpy(kept).to(scores.device)

                @staticmethod
                def cc_2d(mask, **kwargs):
                    labels, counts = connected_components(mask.cpu().numpy())
                    return torch.from_numpy(labels).to(mask.device), torch.from_numpy(counts).to(
                        mask.device
                    )

            # Reference inference math is unmodified. Supply CPU equivalents for
            # the optional CUDA-only bookkeeping kernels, also tested separately.
            modeling_sam3_video.cv_utils_kernel = HostCV()
            frames = [np.roll(np.asarray(image), index % 8, axis=1) for index in range(args.frames)]
            session = processor.init_video_session(
                video=None if args.streaming else frames,
                inference_device=args.reference_device,
                inference_state_device="cpu",
                video_storage_device="cpu",
                processing_device="cpu",
                dtype=torch.float32,
            )
            processor.add_text_prompt(session, args.text)
            if args.streaming:

                def outputs_iterator():
                    for index, frame in enumerate(frames):
                        inputs = processor(images=Image.fromarray(frame), return_tensors="pt")
                        yield model(
                            session,
                            frame=inputs.pixel_values.to(args.reference_device),
                            frame_idx=index,
                        )

                iterator = outputs_iterator()
            else:
                iterator = model.propagate_in_video_iterator(session)
            arrays = {}
            for output in iterator:
                result = processor.postprocess_outputs(
                    session, output, original_sizes=[image.height, image.width]
                )
                for key in ("masks", "scores", "boxes", "object_ids"):
                    arrays[f"frame_{output.frame_idx}_{key}"] = result[key].cpu().numpy()
                print(
                    json.dumps(
                        {
                            "stage": "reference_video_frame",
                            "index": output.frame_idx,
                            "objects": len(result["object_ids"]),
                        }
                    ),
                    flush=True,
                )
            save_arrays(args.output, **arrays)
            print(
                json.dumps(
                    {
                        "stage": "reference_video_inferred",
                        "frames": args.frames,
                        "seconds": time.perf_counter() - start,
                        "device": args.reference_device,
                        "kernel_policy": "numpy mask-NMS and eight-connected components",
                        "peak_process_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    }
                ),
                flush=True,
            )
            return
    save_arrays(args.output, **arrays)
    print(
        json.dumps(
            {
                "stage": "reference_inferred",
                "seconds": time.perf_counter() - start,
                "mask_count": len(arrays["masks"]),
                "peak_process_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=["concept", "interactive", "video"], default="concept")
    parser.add_argument("--image")
    parser.add_argument("--text", default="car")
    parser.add_argument("--backend", choices=["native", "reference"], default="native")
    parser.add_argument(
        "--output", help="Write validation arrays to a new .npz file (never overwrite)."
    )
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--reference-device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    if args.image == "fixture":
        from huggingface_hub import hf_hub_download

        args.image = hf_hub_download(
            "hf-internal-testing/sam2-fixtures", "truck.jpg", repo_type="dataset"
        )
    if args.backend == "reference":
        reference(args)
        return
    import mlx.core as mx

    from mlx_one.segmentation import load_segmentation_model, predict_masks, segment_image

    start = time.perf_counter()
    mx.reset_peak_memory()
    bundle = load_segmentation_model(args.model, task=args.task, offline=True)
    mx.eval(list(bundle.model.parameters().values()))
    print(
        json.dumps(
            {
                "stage": "loaded",
                "task": args.task,
                "seconds": time.perf_counter() - start,
                "tensor_count": len(bundle.model.state_dict()),
                "peak_mlx_bytes": mx.get_peak_memory(),
            }
        ),
        flush=True,
    )
    if args.image and args.task != "video":
        start = time.perf_counter()
        if args.image == "synthetic":
            import numpy as np
            from PIL import Image

            image = Image.fromarray(
                np.tile(np.linspace(0, 255, 640, dtype=np.uint8)[None, :, None], (480, 1, 3))
            )
        else:
            image = args.image
        if args.task == "concept":
            result = segment_image(bundle, image, text=args.text)
        else:
            from PIL import Image

            from mlx_one.segmentation.processing import read_image

            image = read_image(image)
            point = [image.width / 2, image.height / 2]
            result = predict_masks(bundle, image, points=[point], point_labels=[1])
        print(
            json.dumps(
                {
                    "stage": "inferred",
                    "task": args.task,
                    "seconds": time.perf_counter() - start,
                    "mask_count": len(result.masks),
                    "peak_mlx_bytes": mx.get_peak_memory(),
                    "peak_process_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "qualified": False,
                }
            ),
            flush=True,
        )
        arrays = dict(masks=result.masks, boxes=result.boxes)
        if args.task == "concept":
            arrays["scores"] = result.scores
            if args.output:
                pixels, _ = bundle.processor.image(image)
                ids, attention = bundle.processor.text(args.text)
                outputs = bundle.model(pixel_values=pixels, input_ids=ids, attention_mask=attention)
                import numpy as np

                arrays.update(
                    pixel_values=np.asarray(pixels),
                    pred_masks=np.asarray(outputs.pred_masks),
                    pred_boxes=np.asarray(outputs.pred_boxes),
                    pred_logits=np.asarray(outputs.pred_logits),
                    presence_logits=np.asarray(outputs.presence_logits),
                )
        else:
            arrays.update(predicted_iou=result.predicted_iou, logits=result.low_res_logits)
        save_arrays(args.output, **arrays)
    elif args.image and args.task == "video":
        import numpy as np
        from PIL import Image

        image = Image.open(args.image).convert("RGB")
        # Controlled short clip: reproducible image, no claim of natural-video quality.
        frames = (np.roll(np.asarray(image), index % 8, axis=1) for index in range(args.frames))
        start = time.perf_counter()
        records = []
        with bundle.video_session(
            None if args.streaming else frames, streaming=args.streaming
        ) as session:
            session.add_prompt(args.text)
            iterator = (
                (session.process_frame(frame) for frame in frames)
                if args.streaming
                else session.propagate()
            )
            for result in iterator:
                records.append(result)
                print(
                    json.dumps(
                        {
                            "stage": "video_frame",
                            "index": result.frame_index,
                            "objects": len(result.object_ids),
                        }
                    ),
                    flush=True,
                )
        print(
            json.dumps(
                {
                    "stage": "video_inferred",
                    "frames": len(records),
                    "seconds": time.perf_counter() - start,
                    "peak_mlx_bytes": mx.get_peak_memory(),
                    "peak_process_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "qualified": False,
                }
            ),
            flush=True,
        )
        if args.output:
            arrays = {
                f"frame_{r.frame_index}_{key}": value
                for r in records
                for key, value in dict(
                    masks=r.masks,
                    scores=r.scores,
                    boxes=r.boxes,
                    object_ids=np.asarray(r.object_ids),
                ).items()
            }
            save_arrays(args.output, **arrays)
    del bundle
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
