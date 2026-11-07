"""Pinned BERT token head reusing the existing native BERT embedding/encoder math.

Transformers v4.57.1, Apache-2.0; checkpoint dslim/bert-base-NER, MIT.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tokenizers import BertWordPieceTokenizer

from mlx_one.models.classification.bert_ner_config import BertNERConfig
from mlx_one.models.embeddings.bert.model import BertEmbeddings, BertEncoder

MODEL_ID = "dslim/bert-base-NER"
REVISION = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
HASHES = {
    "model.safetensors": "b04492186cfb45a64908487a17a9f8d6ddec3a403ef39db5bca688f0fa702a34",
    "tokenizer_config.json": "4c052d60b505817149a29e88c5292d7779ddecfadbfc6208729e61403ec558ba",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "config.json": "a5dc77a0d60dadf5645103017f3c7b8ece4ef1b450bc1ea68e17da1d3ae7e515",
    "vocab.txt": "eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02",
}


@dataclass(frozen=True)
class BertNEROutput:
    logits: object
    hidden_states: tuple | None


class BertForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = nn.Module()
        self.bert.embeddings = BertEmbeddings(config)
        self.bert.encoder = BertEncoder(config)
        self.classifier = nn.Linear(config.hidden_size, len(config.id2label))

    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, output_hidden_states=False
    ):
        if (
            input_ids.ndim != 2
            or not 1 <= input_ids.shape[1] <= self.config.max_position_embeddings
        ):
            raise ValueError("expected [batch, sequence<=512] input IDs")
        mask = mx.ones_like(input_ids) if attention_mask is None else attention_mask
        if mask.shape != input_ids.shape:
            raise ValueError("attention mask must match input IDs")
        hidden = self.bert.embeddings(input_ids, token_type_ids, None)
        # Use the source finfo-min key mask; leave the shared embedding task mask unchanged.
        additive = mx.where(
            mask[:, None, None, :].astype(mx.bool_),
            mx.array(0, hidden.dtype),
            mx.array(mx.finfo(hidden.dtype).min, hidden.dtype),
        )
        hidden, states = self.bert.encoder(hidden, additive, output_hidden_states)
        return BertNEROutput(self.classifier(hidden), states)


@dataclass(frozen=True)
class LoadedBertNER:
    model: BertForTokenClassification
    tokenizer: object
    labels: dict
    model_path: Path
    model_id: str = MODEL_ID
    revision: str = REVISION
    architecture: str = "bert_token_classification"
    context_length: int = 512

    @property
    def quantization(self):
        return {}


def load_bert_ner(path, *, revision=REVISION):
    if revision != REVISION:
        raise ValueError("unsupported BERT NER checkpoint revision")
    root = Path(path)
    for name, expected in HASHES.items():
        digest = hashlib.sha256()
        try:
            with (root / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ValueError(f"missing BERT NER asset: {name}") from exc
        if digest.hexdigest() != expected:
            raise ValueError(f"unexpected BERT NER asset: {name}")
    cfg = BertNERConfig.from_dict(json.loads((root / "config.json").read_text()))
    model = BertForTokenClassification(cfg)
    tensors = mx.load(str(root / "model.safetensors"))
    # HF BertForTokenClassification disables pooling; the published checkpoint
    # nevertheless contains this exact, validated pair of unused pooler tensors.
    unused = {
        "bert.pooler.dense.weight": (cfg.hidden_size, cfg.hidden_size),
        "bert.pooler.dense.bias": (cfg.hidden_size,),
    }
    for name, shape in unused.items():
        if name not in tensors or tensors[name].shape != shape or tensors[name].dtype != mx.float32:
            raise ValueError(f"unexpected source-only NER pooler tensor: {name}")
        tensors.pop(name)

    tensors = {
        k.replace(".attention.self.", ".attention.self_attn."): v for k, v in tensors.items()
    }
    expected = dict(tree_flatten(model.parameters()))
    if set(tensors) != set(expected):
        raise ValueError(
            f"BERT NER tensor inventory mismatch: missing={set(expected) - set(tensors)}, "
            f"unexpected={set(tensors) - set(expected)}"
        )
    for name, value in tensors.items():
        if value.shape != expected[name].shape or value.dtype != mx.float32:
            raise ValueError(f"BERT NER shape or dtype mismatch: {name}")
    model.load_weights(list(tensors.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    tokenizer = BertWordPieceTokenizer(str(root / "vocab.txt"), lowercase=False)
    tokenizer.enable_truncation(max_length=512)
    return LoadedBertNER(model, tokenizer, cfg.id2label, root.resolve())
