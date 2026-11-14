"""DistilBERT sequence classification, Transformers 4.57.1 (Apache-2.0).

Restricted to the pinned SST-2 safetensors checkpoint pending task qualification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tokenizers import BertWordPieceTokenizer

from mlx_one.models.classification.config import DistilBertConfig

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
HASHES = {
    "config.json": "582122c8f414793d131e10022ce9ba04e3811a9da6389137ee2f18665b4f4d15",
    "tokenizer_config.json": "5ab9097b4149371c5fd52b2d6e26cb6f9c07c0d19fcfdda895b1adad6b57c3e0",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "model.safetensors": "7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa",
}


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.dim)
        self.LayerNorm = nn.LayerNorm(cfg.dim, eps=1e-12)

    def __call__(self, ids):
        return self.LayerNorm(
            self.word_embeddings(ids) + self.position_embeddings(mx.arange(ids.shape[1])[None, :])
        )


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.n_heads
        self.q_lin = nn.Linear(cfg.dim, cfg.dim)
        self.k_lin = nn.Linear(cfg.dim, cfg.dim)
        self.v_lin = nn.Linear(cfg.dim, cfg.dim)
        self.out_lin = nn.Linear(cfg.dim, cfg.dim)

    def __call__(self, x, mask):
        batch, length, dim = x.shape

        def split(linear):
            return (
                linear(x)
                .reshape(batch, length, self.heads, dim // self.heads)
                .transpose(0, 2, 1, 3)
            )

        output = mx.fast.scaled_dot_product_attention(
            split(self.q_lin),
            split(self.k_lin),
            split(self.v_lin),
            scale=(dim // self.heads) ** -0.5,
            mask=mask[:, None, None, :],
        )
        return self.out_lin(output.transpose(0, 2, 1, 3).reshape(batch, length, dim))


class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lin1 = nn.Linear(cfg.dim, cfg.hidden_dim)
        self.lin2 = nn.Linear(cfg.hidden_dim, cfg.dim)

    def __call__(self, x):
        return self.lin2(nn.gelu(self.lin1(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = Attention(cfg)
        self.sa_layer_norm = nn.LayerNorm(cfg.dim, eps=1e-12)
        self.ffn = FFN(cfg)
        self.output_layer_norm = nn.LayerNorm(cfg.dim, eps=1e-12)

    def __call__(self, x, mask):
        x = self.sa_layer_norm(x + self.attention(x, mask))
        return self.output_layer_norm(x + self.ffn(x))


class DistilBertClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.distilbert = nn.Module()
        self.distilbert.embeddings = Embeddings(cfg)
        self.distilbert.transformer = nn.Module()
        self.distilbert.transformer.layer = [Block(cfg) for _ in range(cfg.n_layers)]
        self.pre_classifier = nn.Linear(cfg.dim, cfg.dim)
        self.classifier = nn.Linear(cfg.dim, len(cfg.id2label))

    def __call__(self, ids, attention_mask):
        if ids.ndim != 2 or ids.shape != attention_mask.shape or not 1 <= ids.shape[1] <= 512:
            raise ValueError("input IDs and mask must have identical [batch, sequence<=512] shapes")
        x = self.distilbert.embeddings(ids)
        for layer in self.distilbert.transformer.layer:
            x = layer(x, attention_mask.astype(mx.bool_))
        return self.classifier(nn.relu(self.pre_classifier(x[:, 0])))


class LoadedDistilBert:
    def __init__(self, model, tokenizer, labels):
        self.model, self.tokenizer, self.labels = model, tokenizer, labels
        self.model_id, self.revision = MODEL_ID, REVISION
        self.architecture = "distilbert"
        self.context_length = 512
        self.quantization = {}

    def predict(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else texts
        if (
            not isinstance(items, (tuple, list))
            or not items
            or any(not isinstance(text, str) or not text for text in items)
        ):
            raise ValueError("expected a nonempty string or list of nonempty strings")
        encodings = self.tokenizer.encode_batch(items)
        width = max(len(e.ids) for e in encodings)
        ids = [e.ids + [0] * (width - len(e.ids)) for e in encodings]
        masks = [[1] * len(e.ids) + [0] * (width - len(e.ids)) for e in encodings]
        logits = self.model(mx.array(ids), mx.array(masks))
        scores = mx.softmax(logits, axis=-1, precise=True)
        mx.eval(scores)
        rows = [
            [
                {"label": self.labels[str(i)], "score": float(row[i]), "index": i}
                for i in sorted(range(len(row)), key=lambda idx: -row[idx])
            ]
            for row in scores.tolist()
        ]
        return rows[0] if single else rows


def load_distilbert_classifier(path, *, revision=REVISION):
    if revision != REVISION:
        raise ValueError("unsupported DistilBERT checkpoint revision")
    root = Path(path)
    for name, expected in HASHES.items():
        digest = hashlib.sha256()
        try:
            with (root / name).open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ValueError(f"missing checkpoint asset: {name}") from exc
        if digest.hexdigest() != expected:
            raise ValueError(f"unexpected checkpoint asset: {name}")
    config = DistilBertConfig.from_dict(json.loads((root / "config.json").read_text()))
    model = DistilBertClassifier(config)
    tensors = mx.load(str(root / "model.safetensors"))
    expected = dict(tree_flatten(model.parameters()))
    if set(tensors) != set(expected):
        raise ValueError(
            f"checkpoint tensor inventory mismatch: missing={set(expected) - set(tensors)}, "
            f"unexpected={set(tensors) - set(expected)}"
        )
    for name, value in tensors.items():
        if value.shape != expected[name].shape or value.dtype != mx.float32:
            raise ValueError(f"unexpected shape or dtype for {name}")
    model.load_weights(list(tensors.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    tokenizer = BertWordPieceTokenizer(str(root / "vocab.txt"), lowercase=True)
    tokenizer.enable_truncation(max_length=config.max_position_embeddings)
    return LoadedDistilBert(model, tokenizer, config.id2label)
