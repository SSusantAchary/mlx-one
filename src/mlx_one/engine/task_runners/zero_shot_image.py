"""Bounded, source-validated CLIP task adapter using the existing task actor."""

from __future__ import annotations

from tokenizers import Tokenizer

from mlx_one.classification.schemas import ZeroShotImageResult
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.task_runners.base import NativeTaskAdapter
from mlx_one.engine.tasks import PreparedTask
from mlx_one.vision.clip_processing import preprocess_clip


class CLIPZeroShotAdapter(NativeTaskAdapter):
    family = "clip"
    supported_tasks = (TaskKind.ZERO_SHOT_IMAGE_CLASSIFICATION,)

    def __init__(self, bundle):
        if bundle.architecture != self.family:
            raise ValueError("zero-shot adapter/checkpoint mismatch")
        self.bundle = bundle

    def prepare(self, task, payload):
        if TaskKind(task) != TaskKind.ZERO_SHOT_IMAGE_CLASSIFICATION:
            raise ValueError("CLIP adapter supports zero-shot-image-classification only")
        if set(payload) - {"image", "candidate_labels", "top_k"}:
            raise ValueError("unknown zero-shot image options")
        labels = payload.get("candidate_labels")
        if not isinstance(labels, (list, tuple)) or not 1 <= len(labels) <= 10:
            raise ValueError(
                "candidate_labels requires 1–10 labels; the 100-label parity profile failed"
            )
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("candidate labels must be nonempty strings")
        try:
            if any(len(label.encode("utf-8")) > 512 for label in labels):
                raise ValueError("candidate label exceeds 512 UTF-8 bytes")
        except UnicodeError as exc:
            raise ValueError("candidate labels must be valid Unicode") from exc
        top_k = payload.get("top_k")
        if top_k is None:
            top_k = len(labels)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(labels):
            raise ValueError("top_k must be between 1 and the number of candidate labels")
        tokenizer = Tokenizer.from_str(self.bundle.tokenizer.to_str())
        tokenizer.no_truncation()
        tokenizer.enable_padding(pad_id=49407, pad_token="<|endoftext|>")
        encoded = tokenizer.encode_batch([f"This is a photo of {label}." for label in labels])
        if any(len(e.ids) > 77 for e in encoded):
            raise ValueError("candidate prompt exceeds the CLIP 77-token limit")
        pixels, size = preprocess_clip(payload.get("image"))
        data = {
            "pixels": pixels,
            "size": size,
            "labels": tuple(labels),
            "top_k": top_k,
            "ids": tuple(tuple(e.ids) for e in encoded),
            "mask": tuple(tuple(e.attention_mask) for e in encoded),
        }
        length = len(encoded) * len(encoded[0].ids) + 50
        return PreparedTask(
            TaskKind.ZERO_SHOT_IMAGE_CLASSIFICATION,
            data,
            (length,),
            (),
            pixels.nbytes + len(encoded) * 77 * 512 * 40 + 50 * 768 * 40,
        )

    def run_batch(self, prepared):
        import mlx.core as mx

        results = []
        for request in prepared:
            data = request.payload
            output = self.bundle.model(
                mx.array(data["pixels"][None]), mx.array(data["ids"]), mx.array(data["mask"])
            )
            logits = output["logits_per_image"][0]
            scores = mx.softmax(logits, axis=-1, precise=True)
            mx.eval(scores, logits)
            scores, logits = scores.tolist(), logits.tolist()
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            labels = tuple(
                {
                    "label": data["labels"][i],
                    "score": float(scores[i]),
                    "index": i,
                    "logit": float(logits[i]),
                }
                for i in order[: data["top_k"]]
            )
            results.append(
                ZeroShotImageResult(
                    labels,
                    {"width": data["size"][0], "height": data["size"][1]},
                    self.bundle.model_id,
                    self.bundle.revision,
                )
            )
        return tuple(results)
