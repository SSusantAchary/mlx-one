"""Offline held-out datasets, paired scoring, and qualification coverage."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from pathlib import Path

QUALITY_SCHEMA = "native-held-out-quality-v1"
DEPTH_BANDS = ("start", "quarter", "middle", "three_quarter", "end")
SCORER_VERSION = "nfkc-casefold-whitespace-v1"


def load_quality_manifest(path):
    root = Path(path).resolve().parent
    data = json.loads(Path(path).read_text())
    if (data.get("schema_version") != QUALITY_SCHEMA
            or data.get("split") not in {"test", "validation", "held_out"}
            or not all(data.get(k) for k in ("dataset_id", "provenance", "license", "settings"))):
        raise ValueError("quality requires a versioned, provenance-backed held-out manifest")
    settings = data["settings"]
    minimum = settings.get("min_oracle_retrieval_accuracy")
    if (settings.get("scorer") != SCORER_VERSION or not isinstance(minimum, (float, int))
            or not math.isfinite(minimum) or not 0 < minimum <= 1
            or not settings.get("context_lengths")):
        raise ValueError("declare scorer, context lengths and minimum oracle accuracy")
    records = {}
    for kind in ("documents", "retrieval"):
        descriptor = data[kind]
        file = (root / descriptor["path"]).resolve()
        if not file.is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest() != (
                descriptor["sha256"]):
            raise ValueError("quality dataset path or checksum mismatch")
        rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate quality record IDs")
        for row in rows:
            if kind == "documents" and not isinstance(row.get("text"), str):
                raise ValueError("document text is required")
            if kind == "retrieval" and (
                row.get("context_tokens") not in settings["context_lengths"]
                or row.get("depth_band") not in DEPTH_BANDS
                or not isinstance(row.get("context"), str)
                or not isinstance(row.get("question"), str)
                or not row.get("accepted_answers")
                or any(not isinstance(a, str) or not a.strip() for a in row["accepted_answers"])
            ):
                raise ValueError("retrieval records require context, depth, query and answers")
        records[kind] = rows
    identity = {"schema_version": QUALITY_SCHEMA,
                "manifest_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "dataset_id": data["dataset_id"], "split": data["split"],
                "files": {k: data[k]["sha256"] for k in records},
                "settings": settings, "scorer_version": SCORER_VERSION}
    return {"manifest": data, "identity": identity, **records}


def normalized_answer(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def retrieval_score(output, answers):
    return normalized_answer(output) in {normalized_answer(a) for a in answers}


def paired_logits_metrics(oracle, target, expected_tokens):
    import numpy as np

    left, right = np.asarray(oracle, dtype=np.float64), np.asarray(target, dtype=np.float64)
    ids = np.asarray(expected_tokens, dtype=np.int64)
    if (left.shape != right.shape or left.ndim != 2 or left.shape[0] != len(ids)
            or not len(ids) or np.any(ids < 0) or np.any(ids >= left.shape[1])
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("invalid paired held-out logits or target mapping")
    def nll(values):
        maximum = values.max(axis=-1)
        return maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=-1)) - (
            values[np.arange(len(ids)), ids])
    a, b = nll(left), nll(right)
    error = np.abs(left - right)
    return {"scored_tokens": len(ids), "oracle_nll_sum": float(a.sum()),
            "target_nll_sum": float(b.sum()), "max_error": float(error.max()),
            "absolute_error_sum": float(error.sum()), "logit_count": error.size,
            "greedy_matches": int((left.argmax(-1) == right.argmax(-1)).sum())}


def summarize_quality_documents(rows):
    total = sum(r["scored_tokens"] for r in rows)
    if not total:
        raise ValueError("no held-out tokens were scored")
    return {"schema_version": QUALITY_SCHEMA, "document_count": len(rows),
            "scored_tokens": total, "documents": rows,
            "held_out_nll_mean_delta": sum(r["target_nll_sum"] - r["oracle_nll_sum"]
                                           for r in rows) / total,
            "max_error": max(r["max_error"] for r in rows),
            "mean_error": sum(r["absolute_error_sum"] for r in rows)
                          / sum(r["logit_count"] for r in rows),
            "greedy_agreement_fraction": sum(r["greedy_matches"] for r in rows) / total}


def quality_coverage(documents, retrieval, context_tokens):
    rows = retrieval.get("fixtures", [])
    bands = {band: sum(r.get("depth_band") == band for r in rows) for band in DEPTH_BANDS}
    checks = {"held_out_documents": documents.get("document_count", 0) >= 100,
              "held_out_tokens": documents.get("scored_tokens", 0) >= 100_000,
              "retrieval_cases": len(rows) >= 100,
              "retrieval_depths": all(n >= 20 for n in bands.values()),
              "context_coverage": all(r.get("context_tokens") == context_tokens
                                      and r.get("rendered_tokens", 0) >= .9 * context_tokens
                                      for r in rows)}
    return {"eligible": all(checks.values()), "checks": checks, "depth_counts": bands}


def native_dataset_quality(bundle, profile):
    """Evaluate native teacher-forced windows and retrieval without downloads."""
    import mlx.core as mx
    import numpy as np

    from mlx_one import BatchingConfig, CacheConfig, Engine, MemoryPressureConfig
    from mlx_one.engine.batching import ForwardRequest
    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.hybrid import HybridRunner
    from mlx_one.engine.paged import PagedRunner

    if not getattr(bundle, "tokenizer_identity", None):
        raise ValueError("quality evaluation requires immutable tokenizer identity")
    dataset = load_quality_manifest(profile.quality_manifest)
    if dataset["identity"] != profile.quality_dataset_identity:
        raise ValueError("quality manifest changed after profile creation")
    if profile.context_tokens not in dataset["manifest"]["settings"]["context_lengths"]:
        raise ValueError("quality dataset does not cover the selected context")
    oracle_type = "bf16" if profile.cache_type == "bf16" else "f16"
    oracle = CacheManager(bundle, CacheConfig(prefix_cache_enabled=False))
    backend = "paged" if profile.inference_mode == "paged" else "hybrid"
    target = CacheManager(bundle, CacheConfig(backend=backend, prefix_cache_enabled=False,
                         block_size_tokens=profile.page_tokens,
                         key_type=profile.cache_type, value_type=profile.cache_type))
    target.harness_only = True
    runner = PagedRunner() if backend == "paged" else HybridRunner()
    documents = []
    try:
        for document in dataset["documents"]:
            ids = tuple(bundle.tokenizer.encode(document["text"]))
            measurements = []
            for start in range(0, len(ids) - 1, profile.context_tokens):
                window = ids[start:start + profile.context_tokens]
                if len(window) < 2:
                    continue
                a = oracle.create_handle("oracle", key_type=oracle_type, value_type=oracle_type)
                b = target.create_handle("target", key_type=profile.cache_type,
                                         value_type=profile.cache_type)
                try:
                    for cursor in range(0, len(window) - 1, profile.chunk_tokens):
                        tokens = window[cursor:min(cursor + profile.chunk_tokens, len(window) - 1)]
                        expected = window[cursor + 1:cursor + 1 + len(tokens)]
                        left = bundle.model(mx.array([tokens]), cache=a.bundle.entries).logits
                        request = ForwardRequest(bundle, tokens, b.bundle.entries, "prefill",
                            window, target, b, request_id="quality", return_all_logits=True)
                        right = runner.execute((request,)).logits[0]
                        mx.eval(left, right)
                        measurements.append(paired_logits_metrics(np.array(left)[0],
                                                                   np.array(right)[0], expected))
                finally:
                    oracle.release(a)
                    target.release(b)
            if measurements:
                row = {"id": document["id"], "windows": len(measurements)}
                for key in ("scored_tokens", "oracle_nll_sum", "target_nll_sum",
                            "absolute_error_sum", "logit_count", "greedy_matches"):
                    row[key] = sum(m[key] for m in measurements)
                row["max_error"] = max(m["max_error"] for m in measurements)
                documents.append(row)
        nll = summarize_quality_documents(documents)
    finally:
        oracle.close()
        target.close()
    fixtures = []
    for row in sorted(dataset["retrieval"], key=lambda r: r["id"]):
        if row["context_tokens"] != profile.context_tokens:
            continue
        prompt = row["context"] + "\n" + row["question"]
        rendered_tokens = len(bundle.tokenizer.encode(prompt))
        if rendered_tokens + profile.output_tokens > bundle.context_length:
            raise ValueError("retrieval context plus output exceeds checkpoint limit")
        outputs = {}
        for name, mode, cache_type in (("oracle", "reference_dense", oracle_type),
                                       ("target", profile.inference_mode, profile.cache_type)):
            cache_backend = "dense" if name == "oracle" else backend
            with Engine(bundle, inference_mode=mode,
                        cache_config=CacheConfig(backend=cache_backend,
                            prefix_cache_enabled=False, key_type=cache_type, value_type=cache_type,
                            block_size_tokens=profile.page_tokens),
                        batching_config=BatchingConfig(prefill_chunk_size=profile.chunk_tokens),
                        memory_pressure_config=MemoryPressureConfig(enabled=False)) as engine:
                engine._cache_manager.harness_only = True
                answer = engine.generate(prompt, max_tokens=profile.output_tokens,
                                         temperature=0, seed=profile.seed).text
                outputs[name] = {"text": answer,
                                 "retrieved": retrieval_score(answer, row["accepted_answers"])}
        fixtures.append({"id": row["id"], "depth_band": row["depth_band"],
                         "context_tokens": profile.context_tokens,
                         "rendered_tokens": rendered_tokens,
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "accepted_answers": row["accepted_answers"], **outputs})
    retrieval = {"schema_version": QUALITY_SCHEMA, "fixtures": fixtures,
                 "oracle_accuracy": sum(r["oracle"]["retrieved"] for r in fixtures)
                                    / max(len(fixtures), 1),
                 "target_accuracy": sum(r["target"]["retrieved"] for r in fixtures)
                                    / max(len(fixtures), 1),
                 "min_oracle_accuracy": dataset["manifest"]["settings"][
                     "min_oracle_retrieval_accuracy"]}
    nll["coverage"] = quality_coverage(nll, retrieval, profile.context_tokens)
    nll["dataset_identity"] = {**dataset["identity"],
                               "tokenizer_identity": getattr(bundle, "tokenizer_identity", None)}
    return nll, retrieval
