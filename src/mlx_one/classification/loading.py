"""Exact-checkpoint classification loading using the existing Hub acquisition tool."""

from pathlib import Path

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"


def load_classification_model(model, *, revision=None, offline=True, cache_dir=None):
    from mlx_one.models.classification.distilbert import load_distilbert_classifier

    if revision not in (None, REVISION):
        raise ValueError(f"classification requires immutable SST-2 revision {REVISION}")
    root = Path(model).expanduser()
    if not root.is_dir():
        if str(model) != MODEL_ID:
            raise ValueError(
                "only the pinned DistilBERT SST-2 model ID or a local checkpoint is accepted"
            )
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                MODEL_ID,
                revision=REVISION,
                local_files_only=offline,
                cache_dir=cache_dir,
                allow_patterns=[
                    "config.json",
                    "tokenizer_config.json",
                    "vocab.txt",
                    "model.safetensors",
                ],
            )
        )
    bundle = load_distilbert_classifier(root, revision=REVISION)
    bundle.model_path = root.resolve()
    return bundle


def load_token_classification_model(model, *, revision=None, offline=True, cache_dir=None):
    from mlx_one.models.classification.bert_ner import (
        HASHES,
        load_bert_ner,
    )
    from mlx_one.models.classification.bert_ner import (
        MODEL_ID as ner_model_id,
    )
    from mlx_one.models.classification.bert_ner import (
        REVISION as ner_revision,
    )

    if revision not in (None, ner_revision):
        raise ValueError(f"NER requires immutable checkpoint revision {ner_revision}")
    root = Path(model).expanduser()
    if not root.is_dir():
        if str(model) != ner_model_id:
            raise ValueError("only pinned dslim/bert-base-NER or a local checkpoint is accepted")
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                ner_model_id,
                revision=ner_revision,
                local_files_only=offline,
                cache_dir=cache_dir,
                allow_patterns=list(HASHES),
            )
        )
    return load_bert_ner(root, revision=ner_revision)


def load_zero_shot_image_model(model, *, revision=None, offline=True, cache_dir=None):
    import importlib.util

    if importlib.util.find_spec("PIL") is None:
        raise ImportError("zero-shot image classification requires mlx-one[vision]")
    from mlx_one.models.classification.clip import REVISION as clip_revision
    from mlx_one.models.classification.clip import load_clip

    if revision not in (None, clip_revision):
        raise ValueError("only the pinned, audited original CLIP conversion is admitted")
    root = Path(model).expanduser()
    if not root.is_dir():
        raise ValueError(
            "CLIP requires a local audited safetensors conversion; "
            "run benchmarks/native_tasks/convert_clip_checkpoint.py offline first"
        )
    return load_clip(root, revision=clip_revision)
