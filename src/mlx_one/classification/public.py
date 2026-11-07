"""Classification facade backed by MLX-One's existing InferenceService."""

from __future__ import annotations

from mlx_one.engine.batching import BatchingConfig
from mlx_one.engine.service import InferenceService


class ClassificationEngine:
    def __init__(
        self, model, *, revision=None, offline=True, cache_dir=None, timeout=60.0, queue_size=8
    ):
        self.service = InferenceService(
            batching_config=BatchingConfig(
                max_sequences=2, max_scheduled_tokens=1024, queue_size=queue_size, timeout=timeout
            )
        )
        if hasattr(model, "model"):
            if revision is not None and revision != model.revision:
                self.service.close()
                raise ValueError("requested revision differs from the loaded classification model")
            self.service.register_task("classification", model)
        else:
            self.service.register_task(
                "classification", model, revision=revision, offline=offline, cache_dir=cache_dir
            )

    def submit(self, texts, *, max_length=512, top_k=2, cancel=None, deadline=None):
        return self.service.submit_task(
            "classification",
            "text-classification",
            {"texts": texts, "max_length": max_length, "top_k": top_k},
            cancel=cancel,
            deadline=deadline,
        )

    def classify(self, texts, *, max_length=512, top_k=2, cancel=None, deadline=None):
        return (
            self.submit(texts, max_length=max_length, top_k=top_k, cancel=cancel, deadline=deadline)
            .result()
            .output
        )

    def close(self):
        self.service.close()
        # Borrowed serving bundles must not remain retained by closed registrations.
        with self.service._lock:
            self.service._task_models.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def classify(model, texts, *, max_length=512, top_k=2, revision=None, offline=True, cache_dir=None):
    with ClassificationEngine(
        model, revision=revision, offline=offline, cache_dir=cache_dir
    ) as engine:
        return engine.classify(texts, max_length=max_length, top_k=top_k)


class TokenClassificationEngine(ClassificationEngine):
    """NER facade using the same service lifecycle and existing task actor."""

    def submit(
        self, texts, *, max_length=512, aggregation_strategy="none", cancel=None, deadline=None
    ):
        return self.service.submit_task(
            "classification",
            "token-classification",
            {
                "texts": texts,
                "max_length": max_length,
                "aggregation_strategy": aggregation_strategy,
            },
            cancel=cancel,
            deadline=deadline,
        )

    def tag(
        self, texts, *, max_length=512, aggregation_strategy="none", cancel=None, deadline=None
    ):
        return (
            self.submit(
                texts,
                max_length=max_length,
                aggregation_strategy=aggregation_strategy,
                cancel=cancel,
                deadline=deadline,
            )
            .result()
            .output
        )

    def classify(self, *args, **kwargs):
        raise ValueError("use tag() for token classification")


def token_classify(
    model,
    texts,
    *,
    max_length=512,
    aggregation_strategy="none",
    revision=None,
    offline=True,
    cache_dir=None,
):
    with TokenClassificationEngine(
        model, revision=revision, offline=offline, cache_dir=cache_dir
    ) as engine:
        return engine.tag(texts, max_length=max_length, aggregation_strategy=aggregation_strategy)


class ZeroShotImageEngine(ClassificationEngine):
    """Single-image CLIP facade; delegates all execution to the shared service."""

    def submit(self, image, *, candidate_labels, top_k=None, cancel=None, deadline=None):
        return self.service.submit_task(
            "classification",
            "zero-shot-image-classification",
            {"image": image, "candidate_labels": candidate_labels, "top_k": top_k},
            cancel=cancel,
            deadline=deadline,
        )

    def classify(self, image, candidate_labels, *, top_k=None, cancel=None, deadline=None):
        return (
            self.submit(
                image,
                candidate_labels=candidate_labels,
                top_k=top_k,
                cancel=cancel,
                deadline=deadline,
            )
            .result()
            .output
        )


def zero_shot_classify(model, image, candidate_labels, *, top_k=None, revision=None):
    with ZeroShotImageEngine(model, revision=revision) as engine:
        return engine.classify(image, candidate_labels, top_k=top_k)
