"""Offline real-checkpoint probes, executed in the candidate wheel environment.

A functional pass is deliberately separate from release qualification. Input
samples live in a temporary directory; immutable IDs and digests are archived.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
import os
import platform
import struct
import sys
import time
import traceback
from pathlib import Path


class ForbidReferences(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"transformers", "torch", "mlx_lm", "mlx_vlm", "mlx_audio"}:
            raise AssertionError("native production imported reference package: " + fullname)


def serial(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_inventory(root):
    inventory = {}
    for path in root.rglob("*.safetensors"):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 64 * 1024**2:
                raise ValueError("safetensors header exceeds diagnostic bound")
            header = json.loads(stream.read(size))
        inventory[str(path.relative_to(root))] = {
            name: {"shape": value["shape"], "dtype": value["dtype"]}
            for name, value in header.items()
            if name != "__metadata__"
        }
    return inventory


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def text_probe(entry, model, fixtures, report):
    from functional_checkpoint_smoke import run

    profile = entry.get("profile", "")
    if profile in {"affine_q4", "affine_q8", "affine-Q4", "affine-Q8"} or profile.startswith(
        "affine_"
    ):
        from mlx_one.conversion import convert_text_model

        bits = 8 if "8" in profile else 4
        model = convert_text_model(
            model, model.parent / "converted", revision=entry["revision"], quantize=True, bits=bits
        )
        report["conversion"] = json.loads((model / "mlx-one-conversion.json").read_text())
    report["tested_model_path"] = str(model)
    rows = fixtures["text"]["rows"]
    # A bounded natural-text continuation exercises native cache/state/generation.
    # It is not a held-out language quality benchmark or chat/template admission.
    prompt = " ".join(rows[0]["text"].split()[:24])
    chat = entry["model_type"] == "gpt_oss"
    if chat:
        prompt = "Continue this passage briefly: " + prompt
    result = run(
        model,
        prompt=prompt,
        chat=chat,
        max_tokens=512 if chat else 16,
        context=1024 if chat else 512,
        reasoning_effort="low" if chat else None,
    )
    report["functional"] = result
    report["sample_row_ids"] = [rows[0]["row_id"]]
    assert result["status"] == "PASS", result.get("error", result)
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_one.text import load_text_model
    from mlx_one.text.prompts import completion_token_ids

    bundle = load_text_model(model, offline=True)
    if entry["model_type"] == "openelm" and report.get("reference_python"):
        import shutil
        import subprocess

        import numpy as np

        source_root = Path(report["fixtures_directory"]) / "openelm-reference-source"
        provenance = json.loads((source_root / "provenance.json").read_text())
        for name, expected in provenance["files"].items():
            assert name in {"configuration_openelm.py", "modeling_openelm.py"}
            assert digest(source_root / name) == expected
            shutil.copyfile(source_root / name, model / name)
        tokens = completion_token_ids(bundle, rows[0]["text"])[:97]
        ids = mx.array([tokens[:-1]])
        output = bundle.model(ids, output_hidden_states=True)
        mx.eval(output.logits, output.hidden_states)
        capture = model.parent / "native-parity.npz"
        np.savez(
            capture,
            ids=np.asarray(ids),
            labels=np.asarray([tokens[1:]]),
            logits=np.asarray(output.logits),
            **{f"hidden_{i}": np.asarray(x) for i, x in enumerate(output.hidden_states)},
        )
        reference_output = model.parent / "reference-parity.json"
        completed = subprocess.run(
            [
                report["reference_python"],
                str(Path(__file__).with_name("openelm_reference_probe.py")),
                "--model",
                str(model),
                "--native",
                str(capture),
                "--output",
                str(reference_output),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        report["reference_source_provenance"] = provenance
        report["reference_exit_code"] = completed.returncode
        report["reference_log"] = (completed.stdout + completed.stderr)[-6000:]
        if reference_output.exists():
            report["publisher_reference_comparison"] = json.loads(reference_output.read_text())
    losses, counts = [], []
    for row in rows:
        tokens = completion_token_ids(bundle, row["text"])[:97]
        if len(tokens) < 2:
            continue
        logits = bundle.model(mx.array([tokens[:-1]])).logits
        loss = nn.losses.cross_entropy(logits, mx.array([tokens[1:]]), reduction="mean")
        mx.eval(loss)
        losses.append(float(loss.item()))
        counts.append(len(tokens) - 1)
    mean_nll = sum(loss * n for loss, n in zip(losses, counts, strict=True)) / sum(counts)
    report["text_quality"] = dict(
        sample_row_ids=[row["row_id"] for row in rows],
        target_tokens=sum(counts),
        mean_negative_log_likelihood=mean_nll,
        perplexity=math.exp(min(mean_nll, 100)),
        threshold_perplexity=1000,
        scope="five bounded WikiText excerpts; sanity check, not full test-set perplexity",
    )
    assert math.isfinite(mean_nll) and mean_nll < math.log(1000), (
        "bounded WikiText perplexity exceeds 1000"
    )
    report["quality_status"] = "PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY"


def vlm_probe(entry, model, fixtures, report, data):
    from mlx_one.text import TextGenerationOptions
    from mlx_one.vision import VisionLanguageEngine

    outputs = []
    with VisionLanguageEngine(model, revision=entry["revision"], offline=True) as engine:
        for file, expected in [
            ("VLM-HF-S1-1.png", ("cat", "kitten", "feline")),
            ("VLM-HF-S3-2.png", ("car", "vehicle", "automobile")),
        ]:
            result = engine.generate(
                "Describe this image.",
                [data / "images" / file],
                options=TextGenerationOptions(max_tokens=64),
            )
            output = serial(result)
            output["semantic_keyword_match"] = any(w in result.text.lower() for w in expected)
            outputs.append(output)
        report["outputs"] = outputs
        assert all(o["semantic_keyword_match"] for o in outputs), (
            "image description failed fixture keyword check"
        )
        stream = engine.stream(
            "Describe this image.",
            [data / "images/VLM-HF-S1-1.png"],
            options=TextGenerationOptions(max_tokens=32),
        )
        try:
            next(stream)
        finally:
            stream.close()
        again = engine.generate(
            "Describe this image.",
            [data / "images/VLM-HF-S1-1.png"],
            options=TextGenerationOptions(max_tokens=16),
        )
        assert again.text.strip()
    report["early_stream_close_then_reuse"] = True
    report["quality_status"] = "PASS_TWO_IMAGE_SANITY_ONLY"


def retrieval_probe(entry, model, fixtures, report):
    import numpy as np

    from mlx_one.retrieval import embed, load_retrieval_model, rerank

    family = entry["model_type"]
    if family == "lfm2_colbert":
        return colbert_probe(entry, model, fixtures, report)
    bundle = load_retrieval_model(
        model,
        revision=entry["revision"],
        offline=True,
        task="reranking" if family == "qwen3_reranker" else "embedding",
    )
    rows = fixtures["similarity"]["rows"]
    if family == "qwen3_reranker":
        pairs = []
        for row in [r for r in rows if r["score"] >= 0.8][:3]:
            result = rerank(
                bundle,
                row["sentence1"],
                [row["sentence2"], "The stock market closed lower today."],
                max_length=256,
                batch_size=2,
            )
            pairs.append(serial(result))
            assert all(math.isfinite(item.score) for item in result.items)
            assert result.items[0].score > result.items[1].score, "tied reranker scores"
            assert result.items[0].index == 0, "relevant STS-B pair ranked below unrelated control"
        assert pairs, "no positive pairs in frozen fixture"
        report["outputs"] = pairs
        report["quality_status"] = "PASS_THREE_PAIR_SANITY_ONLY"
        return
    texts = [s for row in rows for s in (row["sentence1"], row["sentence2"])]
    result = embed(bundle, texts, batch_size=4, max_length=128)
    vectors = np.asarray(result.embeddings)
    assert np.isfinite(vectors).all() and vectors.ndim == 2 and len(vectors) == len(texts)
    repeated = embed(bundle, [texts[0]], batch_size=1, max_length=128)
    assert np.allclose(vectors[0], repeated.embeddings[0], atol=2e-4, rtol=2e-4), (
        "batch/padding disagreement"
    )
    similarities = [
        float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        for a, b in zip(vectors[::2], vectors[1::2], strict=True)
    ]
    from scipy.stats import spearmanr

    correlation = float(spearmanr(similarities, [r["score"] for r in rows]).statistic)
    report.update(
        dimensions=vectors.shape[1],
        sample_count=len(rows),
        cosine_similarities=similarities,
        spearman=correlation,
        embedding_sha256=hashlib.sha256(vectors.tobytes()).hexdigest(),
        batched_single_agreement=True,
    )
    assert math.isfinite(correlation) and correlation >= 0.5, (
        "STS-B small-sample Spearman below 0.5"
    )
    report["quality_status"] = "PASS_32_PAIR_SANITY_ONLY"


def colbert_probe(entry, model, fixtures, report):
    import subprocess

    import numpy as np

    from mlx_one.engine.contracts import TaskKind
    from mlx_one.engine.service import InferenceService
    from mlx_one.engine.task_runners.retrieval import maxsim
    from mlx_one.retrieval import load_retrieval_model
    from mlx_one.retrieval.colbert import prepare_token_features

    bundle = load_retrieval_model(model, revision=entry["revision"], offline=True)
    rows = [r for r in fixtures["similarity"]["rows"] if r["score"] >= 0.8][:3]
    service = InferenceService()
    service.register_task("colbert", bundle)
    samples, captured, comparisons = [], {}, []
    try:

        def features(texts, input_type):
            return service.run_task(
                "colbert",
                TaskKind.LATE_INTERACTION,
                dict(
                    texts=texts,
                    input_type=input_type,
                    max_length=128,
                ),
            )

        for row in rows:
            query = features([row["sentence1"]], "query")
            documents = features(
                [row["sentence2"], "The stock market closed lower today."], "document"
            )
            scores = maxsim(query, documents)
            assert len(scores) == 2 and all(math.isfinite(v) for v in scores)
            comparisons.append(dict(row_id=row["row_id"], scores=list(scores)))
            assert scores[0] > scores[1], "ColBERT positive pair ranks below unrelated control"
            repeated = features([row["sentence2"]], "document")
            assert np.allclose(
                documents.embeddings[0], repeated.embeddings[0], atol=2e-4, rtol=2e-4
            )
            if not samples:
                for text, kind, output, offset in [
                    (row["sentence1"], "query", query, 0),
                    (row["sentence2"], "document", documents, 0),
                    ("The stock market closed lower today.", "document", documents, 1),
                ]:
                    index = len(samples)
                    ids, attention, scoring = prepare_token_features(
                        bundle.tokenizer,
                        bundle.model.config,
                        [text],
                        input_type=kind,
                        max_length=128,
                    )
                    samples.append(dict(text=text, input_type=kind))
                    captured.update(
                        {
                            f"ids_{index}": np.asarray(ids[0]),
                            f"attention_{index}": np.asarray(attention[0]),
                            f"scoring_{index}": np.asarray(scoring[0]),
                            f"vectors_{index}": np.asarray(output.embeddings[offset]),
                        }
                    )
        report.update(
            outputs=comparisons,
            dimensions=bundle.model.config.projection_dim,
            batched_single_agreement=True,
            public_workflow="InferenceService.run_task(late-interaction), maxsim",
        )
        assert len(comparisons) == 3
        if report.get("reference_python"):
            capture_path = model.parent / "colbert-native.npz"
            np.savez(capture_path, samples=json.dumps(samples), **captured)
            output = model.parent / "colbert-reference.json"
            helper = Path(__file__).with_name("colbert_reference_probe.py")
            completed = subprocess.run(
                [
                    report["reference_python"],
                    str(helper),
                    "--model",
                    str(model),
                    "--native",
                    str(capture_path),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            report.update(
                reference_exit_code=completed.returncode,
                reference_helper_sha256=digest(helper),
                reference_log=(completed.stdout + completed.stderr)[-6000:],
            )
            if output.exists():
                report["publisher_reference_comparison"] = json.loads(output.read_text())
            assert completed.returncode == 0, "ColBERT publisher reference comparison failed"
            assert (
                max(r["max_abs"] for r in report["publisher_reference_comparison"]["comparisons"])
                < 5e-4
            )
        report["quality_status"] = "PASS_THREE_PAIR_LATE_INTERACTION_SANITY_ONLY"
    finally:
        service.close()


def audio_probe(entry, model, fixtures, report, data):
    from mlx_one.audio.evaluation import character_error_rate, word_error_rate
    from mlx_one.audio.public import AudioEngine

    family = entry["model_type"]
    with AudioEngine(model, revision=entry["revision"], offline=True) as engine:
        if family == "qwen3_forced_aligner":
            result = engine.align(
                data / "audio/librispeech-10s.wav",
                fixtures["audio"]["transcript"],
                language="English",
            )
            report["output"] = serial(result)
            assert result.units and not result.unmatched_words
            assert all(0 <= u.start <= u.end <= result.duration + 0.1 for u in result.units)
            report["quality_status"] = "BLOCKED_NO_GOLD_BOUNDARY_ANNOTATIONS"
            return
        result = engine.transcribe(data / "audio/librispeech-10s.wav")
        report["output"] = serial(result)
        report["wer"] = word_error_rate(fixtures["audio"]["transcript"], result.text)
        report["cer"] = character_error_rate(fixtures["audio"]["transcript"], result.text)
        if family == "lfm2_audio":
            import numpy as np

            speech = report["speech_tasks"] = {}
            jobs = {
                "tts": lambda: engine.generate_speech(
                    (data / "audio/tts.txt").read_text().strip(), max_tokens=1024
                ),
                "sts_short": lambda: engine.speech_to_speech(
                    data / "audio/sts-question.wav", max_tokens=1024
                ),
                "sts_10s": lambda: engine.speech_to_speech(
                    data / "audio/librispeech-10s.wav", max_tokens=1024
                ),
            }
            output_dir = Path(report["evidence_directory"]) / "audio-outputs"
            output_dir.mkdir(exist_ok=True)
            for task, execute in jobs.items():
                try:
                    output = execute()
                    assert output.duration > 0 and np.isfinite(output.waveform).all()
                    assert output.duration <= 40.96, "generated speech exceeds bounded duration"
                    path = output_dir / ("lfm2_audio-" + task + ".wav")
                    output.save(path)
                    speech[task] = dict(
                        status="PASS_FUNCTIONAL",
                        seconds=output.duration,
                        sample_rate=output.sample_rate,
                        text=output.text,
                        wav_sha256=digest(path),
                        artifact=str(path.name),
                        quality_review="PENDING_INTELLIGIBILITY_OR_ANSWER_REVIEW",
                    )
                except Exception as exc:
                    speech[task] = dict(status="FAIL", error=str(exc))
                print("SPEECH_TASK", task, speech[task]["status"], flush=True)
            report["quality_status"] = "ASR_MEASURED_SPEECH_REVIEW_REQUIRED"
            assert all(r["status"] == "PASS_FUNCTIONAL" for r in speech.values()), (
                "one or more advertised LFM speech task probes failed"
            )
        else:
            report["quality_status"] = "PASS_ONE_ENGLISH_CLIP_ONLY"
        assert result.text.strip() and report["wer"] <= 0.35, "LibriSpeech fixture WER exceeds 0.35"


def vad_probe(entry, model, fixtures, report, data):
    import numpy as np

    from mlx_one import VADEngine

    with VADEngine(model) as engine:
        speech = engine.detect(data / "audio/librispeech-10s.wav")
        silence = engine.detect(np.zeros(16000, np.float32))
        assert speech.segments and not silence.segments
        report["speech"] = serial(speech)
        report["silence"] = serial(silence)
        session = engine.session()
        for _ in range(3):
            session.process_chunk(np.zeros(512, np.float32))
        assert not session.finalize().segments
        session.reset()
    report["quality_status"] = "PASS_UTTERANCE_PRESENCE_ONLY_NO_BOUNDARY_GOLD"


def classification_probe(entry, model, fixtures, report, data):
    from mlx_one.classification import (
        ClassificationEngine,
        TokenClassificationEngine,
        ZeroShotImageEngine,
    )

    family = entry["model_type"]
    if family == "distilbert":
        outputs, correct = [], 0
        rows = fixtures["sentiment"]["rows"]
        with ClassificationEngine(model, revision=entry["revision"]) as engine:
            for start in range(0, len(rows), 2):
                group = rows[start : start + 2]
                result = engine.classify([r["sentence"] for r in group])
                outputs.append(serial(result))
                for row, pred in zip(group, result.predictions, strict=True):
                    # The contract emits one prediction dictionary per input.
                    label = pred["label"] if "label" in pred else pred["labels"][0]["label"]
                    correct += int((label.upper() == "POSITIVE") == bool(row["label"]))
        report.update(
            outputs=outputs, accuracy=correct / len(rows), correct=correct, count=len(rows)
        )
        assert report["accuracy"] >= 0.8, "SST-2 sample accuracy below 80%"
        report["quality_status"] = "PASS_32_EXAMPLE_SANITY_ONLY"
    elif family == "bert_token_classification":
        with TokenClassificationEngine(model, revision=entry["revision"]) as engine:
            outputs = [
                serial(engine.tag(" ".join(row["tokens"]), aggregation_strategy="simple"))
                for row in fixtures["ner"]["rows"]
            ]
        report["outputs"] = outputs
        assert any(o["predictions"] for o in outputs)
        labels_path = data / "ner-labels/dataset/label.json"
        labels = {value: key for key, value in json.loads(labels_path.read_text()).items()}
        gold, predicted = set(), set()
        for row, output in zip(fixtures["ner"]["rows"], outputs, strict=True):
            offset, current = 0, None
            for word, tag_id in zip(row["tokens"], row["tags"], strict=True):
                tag = labels[tag_id]
                kind = tag[2:] if tag != "O" else None
                if current and (tag == "O" or tag.startswith("B-") or kind != current[2]):
                    gold.add((row["row_id"], *current))
                    current = None
                if kind:
                    current = (
                        (current[0], offset + len(word), kind)
                        if current
                        else (offset, offset + len(word), kind)
                    )
                offset += len(word) + 1
            if current:
                gold.add((row["row_id"], *current))
            for entity in output["predictions"][0]["entities"]:
                predicted.add(
                    (row["row_id"], entity["start"], entity["end"], entity["entity_group"])
                )
        tp, fp, fn = len(gold & predicted), len(predicted - gold), len(gold - predicted)
        report["ner_quality"] = dict(
            examples=len(outputs),
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / max(1, tp + fp),
            recall=tp / max(1, tp + fn),
            strict_span_f1=2 * tp / max(1, 2 * tp + fp + fn),
            label_mapping_sha256=digest(labels_path),
        )
        report["quality_status"] = "MEASURED_STRICT_SPAN_F1_REVIEW_REQUIRED"
    else:
        with ZeroShotImageEngine(model, revision=entry["revision"]) as engine:
            result = engine.classify(data / "images/VLM-HF-S1-1.png", ["cat", "car", "airplane"])
            report["output"] = serial(result)
        assert result.labels[0]["label"] == "cat", "CLIP cat fixture top-1 mismatch"
        report["quality_status"] = "PASS_THREE_CANDIDATE_SANITY_ONLY"


def ocr_probe(entry, model, fixtures, report, data):
    from mlx_one.ocr import OCREngine
    from mlx_one.ocr.schemas import OCRLimits

    image = Path(fixtures["ocr"]["path"])
    page_sample = None
    if entry["model_type"] == "trocr":
        image = data / "ocr-line.png"
    elif (data / "ocr-sample.json").is_file():
        page_sample = json.loads((data / "ocr-sample.json").read_text())
        image = Path(page_sample["image_path"])
        report["ocr_fixture_sha256"] = digest(data / "ocr-sample.json")
        report["ocr_fixture"] = {
            "dataset": page_sample["dataset"],
            "revision": page_sample["revision"],
            "id": page_sample["fixture"]["id"],
            "image_sha256": digest(image),
            "reference_sha256": page_sample["fixture"]["reference_sha256"],
        }
    report["ocr_image_path"] = str(image)
    with OCREngine(
        model, revision=entry["revision"], offline=True, limits=OCRLimits(timeout_seconds=300)
    ) as engine:
        budget = 512 if entry["model_type"] == "florence2" else 1024 if page_sample else 256
        report["ocr_generation_budget"] = budget
        result = engine.recognize(image, max_tokens=budget)
        report["output"] = serial(result)
        assert result.text.strip(), "empty OCR output"
    if entry["model_type"] == "trocr":
        from rapidfuzz.distance import Levenshtein

        normalized = "".join(c for c in result.text.lower() if c.isalnum())
        report["line_cer"] = Levenshtein.distance("cashbill", normalized) / len("cashbill")
        assert report["line_cer"] <= 0.2, "printed HF line CER exceeds 20%"
        report["quality_status"] = "PASS_ONE_PRINTED_LINE_ONLY"
    elif page_sample:
        import unicodedata

        from rapidfuzz.distance import Levenshtein

        reference = " ".join(
            unicodedata.normalize("NFKC", page_sample["fixture"]["reference"]).split()
        )
        hypothesis = " ".join(unicodedata.normalize("NFKC", result.text).split())
        report["page_cer"] = Levenshtein.distance(reference, hypothesis) / len(reference)
        report["page_wer"] = Levenshtein.distance(reference.split(), hypothesis.split()) / len(
            reference.split()
        )
        report["quality_status"] = "MEASURED_ONE_LABELED_PAGE"
        assert report["page_cer"] <= page_sample["normalized_cer_limit"], (
            "labeled HF page normalized CER exceeds declared 25% floor"
        )
        report["quality_status"] = "PASS_ONE_LABELED_PAGE_ONLY"
    else:
        report["quality_status"] = "PENDING_FULL_RECEIPT_TRANSCRIPTION_CER_REVIEW"


def segmentation_probe(entry, model, fixtures, report):
    import numpy as np

    from mlx_one.segmentation import load_segmentation_model, predict_masks, segment_image

    family = entry["model_type"]
    if family == "sam3_tracker_video":
        raise ValueError(
            "registered standalone temporal tracker has no admitted public checkpoint loader"
        )
    task = {"sam3": "concept", "sam3_tracker": "interactive", "sam3_video": "video"}[family]
    bundle = load_segmentation_model(model, task=task, revision=entry["revision"], offline=True)
    image = fixtures["segmentation"]["path"]
    if task == "concept":
        result = segment_image(bundle, image, text="truck")
        report.update(
            mask_count=len(result.masks),
            scores=np.asarray(result.scores).tolist(),
            boxes=np.asarray(result.boxes).tolist(),
        )
        assert len(result.masks) and np.asarray(result.masks).any()
    elif task == "interactive":
        from PIL import Image

        with Image.open(image) as im:
            point = [im.width / 2, im.height / 2]
        result = predict_masks(bundle, image, points=[point], point_labels=[1])
        report.update(
            mask_count=len(result.masks), predicted_iou=np.asarray(result.predicted_iou).tolist()
        )
        assert len(result.masks) and np.asarray(result.masks).any()
    else:
        from PIL import Image

        with Image.open(image) as im:
            pixels = np.asarray(im.convert("RGB"))
        modes = {}
        for streaming in [False, True]:
            frames = (np.roll(pixels, i % 8, axis=1) for i in range(32))
            with bundle.video_session(
                None if streaming else frames, streaming=streaming
            ) as session:
                session.add_prompt("truck")
                outputs = (
                    (session.process_frame(f) for f in frames) if streaming else session.propagate()
                )
                records = [dict(frame=r.frame_index, objects=list(r.object_ids)) for r in outputs]
            assert len(records) == 32
            modes[str(streaming)] = records
        report["modes"] = modes
    report["quality_status"] = "PENDING_SAME_CHECKPOINT_REFERENCE_MASK_COMPARISON"


def interface_probe(entry, model, fixtures, report, data):
    """Exercise installed CLI and HTTP paths while the owned weights still exist."""
    import gc

    import mlx.core as mx
    from click.testing import CliRunner

    from mlx_one.cli import main as cli

    kind, family = entry["kind"], entry["model_type"]
    interfaces = report.setdefault("interfaces", {"python": "PASS"})
    if kind == "segmentation" or family == "lfm2_colbert":
        interfaces["remaining"] = "PENDING_TASK_SPECIFIC_INTERFACE_REPLAY"
        return
    model = Path(report.get("tested_model_path", model))
    audio = data / "audio/librispeech-10s.wav"
    image = data / "images/VLM-HF-S1-1.png"
    if kind == "text":
        args = [
            "generate",
            str(model),
            " ".join(fixtures["text"]["rows"][0]["text"].split()[:24]),
            "--offline",
            "--max-tokens",
            "4",
            "--json-output",
        ]
    elif kind == "vlm":
        args = [
            "vlm",
            "--model",
            str(model),
            "--image",
            str(image),
            "--prompt",
            "Describe this image.",
            "--offline",
            "--max-tokens",
            "16",
            "--json-output",
        ]
    elif kind == "audio" and family != "qwen3_forced_aligner":
        args = ["transcribe", str(model), str(audio), "--offline", "--json-output"]
    elif kind == "audio":
        alignment_path = Path(report["evidence_directory"]) / "alignment-cli.json"
        args = [
            "align",
            str(model),
            str(audio),
            "--offline",
            "--text-file",
            str(data / "audio/librispeech.txt"),
            "--language",
            "English",
            "--output",
            str(alignment_path),
        ]
    elif kind == "vad":
        args = ["vad", str(model), str(audio), "--json-output"]
    elif kind == "retrieval":
        row = fixtures["similarity"]["rows"][0]
        args = (
            [
                "rerank",
                str(model),
                row["sentence1"],
                row["sentence2"],
                "Unrelated stock prices.",
                "--offline",
                "--max-length",
                "256",
                "--json-output",
            ]
            if family == "qwen3_reranker"
            else [
                "embed",
                str(model),
                row["sentence1"],
                "--offline",
                "--max-length",
                "128",
                "--json-output",
            ]
        )
    elif kind == "classification":
        if family == "distilbert":
            args = [
                "classify",
                str(model),
                fixtures["sentiment"]["rows"][0]["sentence"],
                "--json-output",
            ]
        elif family == "bert_token_classification":
            args = [
                "ner",
                str(model),
                " ".join(fixtures["ner"]["rows"][0]["tokens"]),
                "--aggregation-strategy",
                "simple",
                "--json-output",
            ]
        else:
            args = [
                "zero-shot-image",
                str(model),
                str(image),
                "--labels",
                "cat",
                "--labels",
                "car",
                "--json-output",
            ]
    elif kind == "ocr":
        source = Path(report["ocr_image_path"])
        args = [
            "ocr",
            "--model",
            str(model),
            "--image",
            str(source),
            "--offline",
            "--max-tokens",
            "64",
            "--json",
        ]
    else:
        interfaces["remaining"] = "PENDING_ALIGNMENT_CLI_HTTP_REPLAY"
        return
    gc.collect()
    mx.clear_cache()
    result = CliRunner().invoke(cli, args)
    interfaces["cli"] = dict(exit_code=result.exit_code, output=result.output)
    assert result.exit_code == 0, f"installed CLI failed: {result.output}"
    if family == "qwen3_forced_aligner":
        interfaces["cli"]["alignment"] = json.loads(alignment_path.read_text())
    else:
        json.loads(result.output)
    interfaces["cli"]["status"] = "PASS"
    gc.collect()
    mx.clear_cache()
    from fastapi.testclient import TestClient

    from mlx_one.server import ModelManager, ServerConfig, create_app

    manager = ModelManager(alias="qualification")
    manager.load(
        model,
        offline=True,
        **(
            {"retrieval_task": "reranking" if family == "qwen3_reranker" else "embedding"}
            if kind == "retrieval"
            else {}
        ),
    )
    config = ServerConfig(
        api_keys=("qualification-local-test-key",),
        warmup=False,
        timeout=300,
        inference_mode="reference_dense",
        cache_prompt=False,
    )
    with TestClient(create_app(model_manager=manager, config=config)) as client:
        assert client.get("/v1/models").status_code == 401
        client.headers["Authorization"] = "Bearer qualification-local-test-key"
        assert client.get("/v1/models").status_code == 200
        assert client.get("/v1/runtime").status_code == 200
        if kind == "text":
            body = dict(model="qualification", prompt="The city was", max_tokens=4, temperature=0)
            response = client.post("/v1/completions", json=body)
            assert response.status_code == 200, response.text
            stream = client.post("/v1/completions", json={**body, "stream": True})
            assert stream.status_code == 200 and stream.text.count("data: [DONE]") == 1
            chunks = [
                json.loads(line[6:])
                for line in stream.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            assert (
                "".join(c["choices"][0]["text"] for c in chunks)
                == response.json()["choices"][0]["text"]
            )
            interfaces["sse"] = "PASS_GREEDY_EQUALS_NONSTREAMING"
        elif kind == "retrieval":
            row = fixtures["similarity"]["rows"][0]
            endpoint = "/v1/rerank" if family == "qwen3_reranker" else "/v1/embeddings"
            body = (
                dict(
                    model="qualification",
                    query=row["sentence1"],
                    documents=[row["sentence2"]],
                )
                if family == "qwen3_reranker"
                else dict(model="qualification", input=row["sentence1"])
            )
            response = client.post(endpoint, json=body)
        elif kind == "classification" and family != "clip":
            endpoint = (
                "/v1/tasks/text-classification"
                if family == "distilbert"
                else "/v1/tasks/token-classification"
            )
            text = (
                fixtures["sentiment"]["rows"][0]["sentence"]
                if family == "distilbert"
                else " ".join(fixtures["ner"]["rows"][0]["tokens"])
            )
            response = client.post(endpoint, json={"model": "qualification", "input": text})
        elif kind == "classification":
            response = client.post(
                "/v1/tasks/zero-shot-image-classification",
                data={"model": "qualification", "candidate_labels": json.dumps(["cat", "car"])},
                files={"file": ("fixture.png", image.read_bytes(), "image/png")},
            )
        elif kind == "vad":
            response = client.post(
                "/v1/tasks/voice-activity-detection",
                data={"model": "qualification"},
                files={"file": ("fixture.wav", audio.read_bytes(), "audio/wav")},
            )
        elif kind == "audio":
            alignment = family == "qwen3_forced_aligner"
            response = client.post(
                "/v1/audio/alignments" if alignment else "/v1/audio/transcriptions",
                data={
                    "model": "qualification",
                    **(
                        {"text": fixtures["audio"]["transcript"], "language": "English"}
                        if alignment
                        else {}
                    ),
                },
                files={"file": ("fixture.wav", audio.read_bytes(), "audio/wav")},
            )
        elif kind == "vlm":
            import base64

            url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
            body = dict(
                model="qualification",
                max_tokens=32,
                messages=[
                    dict(
                        role="user",
                        content=[
                            dict(type="image_url", image_url=dict(url=url)),
                            dict(type="text", text="Describe this image."),
                        ],
                    )
                ],
            )
            response = client.post("/v1/chat/completions", json=body)
            stream = client.post("/v1/chat/completions", json={**body, "stream": True})
            assert stream.status_code == 200 and stream.text.count("data: [DONE]") == 1
            interfaces["sse"] = "PASS_VISION_COMPLETION"
        elif kind == "ocr":
            source = Path(report["ocr_image_path"])
            response = client.post(
                "/v1/ocr",
                data={"model": "qualification", "max_tokens": "64"},
                files={"file": (source.name, source.read_bytes(), "application/octet-stream")},
            )
        else:
            raise ValueError("missing task-specific HTTP replay")
        interfaces["http"] = dict(status_code=response.status_code, body=response.json())
        assert response.status_code == 200, response.text
        interfaces["http"]["status"] = "PASS"
        interfaces["http_authentication"] = "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve prior evidence")
    entry = json.loads(args.entry.read_text())
    report = dict(
        schema_version="model-release-probe-v1",
        model_type=entry["model_type"],
        checkpoint=entry["repo"],
        revision=entry["revision"],
        started=time.time(),
        release_qualified=False,
        python=sys.version,
        platform=platform.platform(),
        worker_sha256=digest(__file__),
        functional_helper_sha256=digest(Path(__file__).with_name("functional_checkpoint_smoke.py")),
        inputs_sha256=digest(args.fixtures),
        fixtures_directory=str(args.fixtures.parent),
        evidence_directory=str(args.output.parent),
        reference_python=str(args.reference_python) if args.reference_python else None,
    )
    snapshot_dir = args.output.parent / "worker-snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    snapshot = snapshot_dir / (report["worker_sha256"] + ".py")
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    helper_snapshot = snapshot_dir / (report["functional_helper_sha256"] + "-functional.py")
    if not helper_snapshot.exists():
        helper_snapshot.write_bytes(
            Path(__file__).with_name("functional_checkpoint_smoke.py").read_bytes()
        )
    # Separate transfer controllers may download disjoint cohorts concurrently.
    # Native execution is exclusive across them to respect unified-memory limits.
    import fcntl

    gpu_lock = (args.fixtures.parent.parent / ".qualification-gpu.lock").open("a+")
    print("Waiting for the campaign Metal execution lock", flush=True)
    fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX)
    report["gpu_execution_serialized"] = True
    print("Acquired Metal execution lock", flush=True)
    sys.meta_path.insert(0, ForbidReferences())
    try:
        for name in ["transformers", "torch", "mlx-lm", "mlx-vlm", "mlx-audio"]:
            try:
                importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                continue
            raise AssertionError("reference dependency installed in candidate environment: " + name)
        import mlx.core as mx

        import mlx_one

        assert Path(mlx_one.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert mx.metal.is_available() and mx.default_device() == mx.gpu
        mx.set_cache_limit(256 * 1024**2)
        report.update(
            installed_source=mlx_one.__file__,
            mlx=importlib.metadata.version("mlx"),
            device=mx.device_info(),
            reference_packages_absent=True,
            installed_record_sha256=hashlib.sha256(
                importlib.metadata.distribution("mlx-one").read_text("RECORD").encode()
            ).hexdigest(),
        )
        fixtures = json.loads(args.fixtures.read_text())
        data = args.fixtures.parent
        if (args.model / "config.json").is_file():
            report["checkpoint_configuration"] = json.loads(
                (args.model / "config.json").read_text()
            )
        if entry["model_type"] == "openelm":
            import shutil

            tokenizer_root = data / "openelm-tokenizer"
            provenance = json.loads((tokenizer_root / "provenance.json").read_text())
            assert provenance["status"] == "PASS_COMPATIBLE_PUBLIC_TOKENIZER"
            assert args.model.resolve().is_relative_to(data.parent / "models")
            assert set(provenance["files"]) == {
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            }
            for name, expected in provenance["files"].items():
                assert digest(tokenizer_root / name) == expected
                assert not (args.model / name).exists()
                shutil.copyfile(tokenizer_root / name, args.model / name)
            report["tokenizer_provenance"] = provenance
        probe = {
            "text": text_probe,
            "vlm": vlm_probe,
            "retrieval": retrieval_probe,
            "audio": audio_probe,
            "vad": vad_probe,
            "classification": classification_probe,
            "ocr": ocr_probe,
            "segmentation": segmentation_probe,
        }[entry["kind"]]
        if entry["kind"] in {"text", "retrieval", "segmentation"}:
            probe(entry, args.model, fixtures, report)
        else:
            probe(entry, args.model, fixtures, report, data)
        report["functional_status"] = "PASS"
        report["stage"] = "interfaces"
        interface_probe(entry, args.model, fixtures, report, data)
        mx.synchronize()
        report["peak_mlx_bytes"] = mx.get_peak_memory()
        report["status"] = "PASS_FUNCTIONAL"
    except Exception as exc:
        report.update(
            status="FAIL",
            error=str(exc),
            exception=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
        try:
            report["checkpoint_tensor_inventory"] = tensor_inventory(args.model)
        except Exception as metadata_error:
            report["tensor_inventory_error"] = str(metadata_error)
    report["seconds"] = time.time() - report["started"]
    atomic(args.output, report)
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in ["model_type", "status", "quality_status", "seconds", "error"]
            }
        ),
        flush=True,
    )
    gpu_lock.close()
    return 0 if report["status"] == "PASS_FUNCTIONAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
