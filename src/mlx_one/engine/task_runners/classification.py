"""DistilBERT adapter for the existing task actor and model residency pool."""

from __future__ import annotations

from tokenizers import Tokenizer

from mlx_one.classification.schemas import ClassificationResult, validate_inputs
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.task_runners.base import NativeTaskAdapter
from mlx_one.engine.tasks import PreparedTask


class DistilBertClassificationAdapter(NativeTaskAdapter):
    family = "distilbert"
    supported_tasks = (TaskKind.TEXT_CLASSIFICATION,)

    def __init__(self, bundle):
        if bundle.architecture != self.family:
            raise ValueError("classification adapter/checkpoint mismatch")
        self.bundle = bundle

    def prepare(self, task, payload):
        if TaskKind(task) != TaskKind.TEXT_CLASSIFICATION:
            raise ValueError("checkpoint supports text-classification only")
        if set(payload) - {"texts", "max_length", "top_k"}:
            raise ValueError("unknown classification options")
        limit, top_k = payload.get("max_length", 512), payload.get("top_k", 2)
        texts = validate_inputs(payload.get("texts"), limit, top_k)
        # Request-local tokenizer options: simultaneous requests cannot mutate
        # the checkpoint tokenizer's truncation/padding configuration.
        tokenizer = Tokenizer.from_str(self.bundle.tokenizer._tokenizer.to_str())
        tokenizer.enable_truncation(max_length=limit)
        tokenizer.no_padding()
        encoded = tokenizer.encode_batch(list(texts))
        sequences = tuple(tuple(item.ids) for item in encoded)
        width = max(map(len, sequences))
        data = {
            "sequences": sequences,
            "top_k": top_k,
            "truncated": tuple(bool(item.overflowing) for item in encoded),
        }
        transient = len(texts) * width * (768 * 32 + width * 48)
        return PreparedTask(
            TaskKind.TEXT_CLASSIFICATION, data, tuple(map(len, sequences)), (limit,), transient
        )

    def run_batch(self, prepared):
        import mlx.core as mx

        sequences = [row for request in prepared for row in request.payload["sequences"]]
        width = max(map(len, sequences))
        ids = [list(row) + [0] * (width - len(row)) for row in sequences]
        masks = [[True] * len(row) + [False] * (width - len(row)) for row in sequences]
        logits = self.bundle.model(mx.array(ids), mx.array(masks))
        scores = mx.softmax(logits, axis=-1, precise=True)
        mx.eval(scores)
        rows = scores.tolist()
        results, cursor = [], 0
        for request in prepared:
            predictions = []
            for index, sequence in enumerate(request.payload["sequences"]):
                row = rows[cursor]
                cursor += 1
                order = sorted(range(len(row)), key=lambda i: (-row[i], i))
                predictions.append(
                    {
                        "index": index,
                        "token_count": len(sequence),
                        "truncated": request.payload["truncated"][index],
                        "labels": [
                            {
                                "label": self.bundle.labels[str(i)],
                                "score": float(row[i]),
                                "index": i,
                            }
                            for i in order[: request.payload["top_k"]]
                        ],
                    }
                )
            results.append(
                ClassificationResult(tuple(predictions), self.bundle.model_id, self.bundle.revision)
            )
        return tuple(results)
