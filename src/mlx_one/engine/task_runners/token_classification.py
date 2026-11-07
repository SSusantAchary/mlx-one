"""Source-faithful BERT NER adapter on the existing bounded task actor."""

from __future__ import annotations

from tokenizers import Tokenizer

from mlx_one.classification.schemas import TokenClassificationResult, validate_inputs
from mlx_one.classification.token_postprocessing import entities_from_scores
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.task_runners.base import NativeTaskAdapter
from mlx_one.engine.tasks import PreparedTask


class BertTokenClassificationAdapter(NativeTaskAdapter):
    family = "bert_token_classification"
    supported_tasks = (TaskKind.TOKEN_CLASSIFICATION,)

    def __init__(self, bundle):
        if bundle.architecture != self.family:
            raise ValueError("NER adapter/checkpoint mismatch")
        self.bundle = bundle

    def prepare(self, task, payload):
        if TaskKind(task) != TaskKind.TOKEN_CLASSIFICATION:
            raise ValueError("NER checkpoint supports token-classification only")
        if set(payload) - {"texts", "max_length", "aggregation_strategy"}:
            raise ValueError("unknown NER options")
        limit = payload.get("max_length", 512)
        texts = validate_inputs(payload.get("texts"), limit)
        strategy = payload.get("aggregation_strategy", "none")
        if strategy not in ("none", "simple"):
            raise ValueError("aggregation_strategy must be none or simple")
        tokenizer = Tokenizer.from_str(self.bundle.tokenizer._tokenizer.to_str())
        tokenizer.enable_truncation(max_length=limit)
        tokenizer.no_padding()
        encoded = tuple(tokenizer.encode_batch(list(texts)))
        width = max(len(e.ids) for e in encoded)
        data = {"texts": texts, "encoded": encoded, "strategy": strategy}
        return PreparedTask(
            TaskKind.TOKEN_CLASSIFICATION,
            data,
            tuple(len(e.ids) for e in encoded),
            (limit,),
            len(texts) * width * (768 * 32 + width * 64),
        )

    def run_batch(self, prepared):
        import mlx.core as mx

        encoded = [item for request in prepared for item in request.payload["encoded"]]
        width = max(len(e.ids) for e in encoded)
        ids = mx.array([e.ids + [0] * (width - len(e.ids)) for e in encoded])
        masks = mx.array([[1] * len(e.ids) + [0] * (width - len(e.ids)) for e in encoded])
        output = self.bundle.model(ids, masks)
        scores = mx.softmax(output.logits, axis=-1, precise=True)
        mx.eval(scores)
        rows = scores.tolist()
        result, cursor = [], 0
        for request in prepared:
            predictions = []
            for index, (text, item) in enumerate(
                zip(request.payload["texts"], request.payload["encoded"], strict=True)
            ):
                entities = entities_from_scores(
                    text,
                    item,
                    rows[cursor][: len(item.ids)],
                    self.bundle.labels,
                    aggregation_strategy=request.payload["strategy"],
                    decoder=self.bundle.tokenizer._tokenizer.decoder,
                )
                cursor += 1
                predictions.append(
                    {
                        "index": index,
                        "token_count": len(item.ids),
                        "truncated": bool(item.overflowing),
                        "entities": entities,
                    }
                )
            result.append(
                TokenClassificationResult(
                    tuple(predictions),
                    self.bundle.model_id,
                    self.bundle.revision,
                    request.payload["strategy"],
                )
            )
        return tuple(result)
