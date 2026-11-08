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


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(".partial")
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
    rows = fixtures["text"]["rows"]
    # A bounded natural-text continuation exercises native cache/state/generation.
    # It is not a held-out language quality benchmark or chat/template admission.
    prompt = " ".join(rows[0]["text"].split()[:24])
    result = run(model, prompt=prompt, max_tokens=16, context=512)
    report["functional"] = result
    report["sample_row_ids"] = [rows[0]["row_id"]]
    assert result["status"] == "PASS", result.get("error", result)
    report["quality_status"] = "PENDING_HELD_OUT_QUALITY"


def vlm_probe(entry, model, fixtures, report, data):
    from mlx_one import VisionLanguageEngine
    from mlx_one.text import TextGenerationOptions

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
    bundle = load_retrieval_model(
        model,
        revision=entry["revision"],
        offline=True,
        task="reranking" if family == "qwen3_reranker" else "embedding",
    )
    rows = fixtures["similarity"]["rows"]
    if family == "lfm2_colbert":
        # Generic single-vector embedding does not admit token-vector outputs.
        raise ValueError(
            "late-interaction publisher bundle requires a verified token-vector public workflow"
        )
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
        assert result.text.strip() and report["wer"] <= 0.35, "LibriSpeech fixture WER exceeds 0.35"
        report["quality_status"] = "PASS_ONE_ENGLISH_CLIP_ONLY"
        if family == "lfm2_audio":
            report["additional_tasks_pending"] = [
                "TTS intelligibility",
                "STS output-duration and answer correctness",
            ]


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
        report["quality_status"] = "PENDING_STRICT_ENTITY_SPAN_F1"
    else:
        with ZeroShotImageEngine(model, revision=entry["revision"]) as engine:
            result = engine.classify(data / "images/VLM-HF-S1-1.png", ["cat", "car", "airplane"])
            report["output"] = serial(result)
        report["quality_status"] = "PENDING_TOP1_REVIEW"


def ocr_probe(entry, model, fixtures, report, data):
    from mlx_one.ocr import OCREngine
    from mlx_one.ocr.schemas import OCRLimits

    image = Path(fixtures["ocr"]["path"])
    if entry["model_type"] == "trocr":
        image = data / "ocr-line.png"
    with OCREngine(
        model, revision=entry["revision"], offline=True, limits=OCRLimits(timeout_seconds=300)
    ) as engine:
        result = engine.recognize(image, max_tokens=256)
        report["output"] = serial(result)
        assert result.text.strip(), "empty OCR output"
    report["quality_status"] = "PENDING_TRANSCRIPTION_CER_REVIEW"


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        inputs_sha256=digest(args.fixtures),
    )
    snapshot_dir = args.output.parent / "worker-snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    snapshot = snapshot_dir / (report["worker_sha256"] + ".py")
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
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
    return 0 if report["status"] == "PASS_FUNCTIONAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
