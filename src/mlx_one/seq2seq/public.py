"""Safe T5 loading and a bounded, native text-to-text Python interface."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mlx_one.models.language.t5.config import T5Config
from mlx_one.models.language.t5.weights import sanitize_weights, weight_contract
from mlx_one.retrieval.loading import _read_json, _read_safetensors, _resolve
from mlx_one.text.tokenizers import HFTokenizerAdapter


@dataclass(frozen=True)
class LoadedSeq2SeqModel:
    model: Any
    tokenizer: HFTokenizerAdapter
    path: Path
    model_id: str
    revision: str | None
    architecture: str = "t5"


@dataclass(frozen=True)
class Seq2SeqResult:
    inputs: tuple[str, ...]
    texts: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]
    finish_reasons: tuple[str, ...]
    model: str
    revision: str | None
    task: str = "text-to-text-generation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


def load_seq2seq_model(
    model: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    cache_dir: str | Path | None = None,
) -> LoadedSeq2SeqModel:
    """Load strict safe T5 tensors; never initialize missing model parameters."""
    root = _resolve(model, revision=revision, offline=offline, cache_dir=cache_dir)
    config = T5Config.from_dict(_read_json(root / "config.json"))
    weights = sanitize_weights(_read_safetensors(root), config)
    weight_contract(config).validate(weights)
    from mlx_one.models.language.t5.model import T5ForConditionalGeneration

    native = T5ForConditionalGeneration(config)
    native.load_weights(list(weights.items()), strict=True)
    native.eval()
    return LoadedSeq2SeqModel(
        native, HFTokenizerAdapter.from_directory(root), root, str(model), revision
    )


def generate(
    model: str | Path | LoadedSeq2SeqModel,
    texts: str | Sequence[str],
    *,
    max_new_tokens: int = 32,
    max_length: int | None = None,
    revision: str | None = None,
    offline: bool = False,
) -> Seq2SeqResult:
    """Greedy T5 text-to-text generation; encoder outputs are reused in decoding."""
    bundle = (
        model
        if isinstance(model, LoadedSeq2SeqModel)
        else load_seq2seq_model(model, revision=revision, offline=offline)
    )
    values = (texts,) if isinstance(texts, str) else tuple(texts)
    if not 1 <= len(values) <= 8 or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("T5 requires one to eight nonempty input texts")
    limit = bundle.model.config.n_positions if max_length is None else max_length
    if type(limit) is not int or not 2 <= limit <= bundle.model.config.n_positions:
        raise ValueError("T5 max_length is outside the configured encoder capacity")
    sequences = [
        bundle.tokenizer.encode(v, add_special_tokens=True, max_length=limit) for v in values
    ]
    ids, mask = bundle.tokenizer.pad(sequences)
    import mlx.core as mx

    output = bundle.model.greedy(
        mx.array(ids), attention_mask=mx.array(mask), max_new_tokens=max_new_tokens
    )
    mx.eval(output)
    rows, reasons = [], []
    for row in output.tolist():
        row = row[1:]  # decoder start is not an emitted token
        if bundle.model.config.eos_token_id in row:
            row = row[: row.index(bundle.model.config.eos_token_id) + 1]
            reasons.append("eos")
        else:
            reasons.append("length")
        rows.append(tuple(row))
    return Seq2SeqResult(
        values,
        tuple(bundle.tokenizer.decode(row) for row in rows),
        tuple(rows),
        tuple(reasons),
        bundle.model_id,
        bundle.revision,
    )
