"""Native-only real-checkpoint verification and pipeline benchmarks for Batch 1."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_one.retrieval import embed, rerank
from mlx_one.retrieval.loading import load_retrieval_model


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return dict(
            status="FAIL", actual_shape=list(actual.shape), expected_shape=list(expected.shape)
        )
    delta = np.abs(actual - expected)
    return dict(
        status="PASS" if np.allclose(actual, expected, atol=1e-4, rtol=1e-4) else "FAIL",
        max_abs_error=float(delta.max()),
        mean_abs_error=float(delta.mean()),
        max_relative_error=float(np.max(delta / np.maximum(np.abs(expected), 1e-12))),
        cosine_similarity=float(
            np.dot(actual.reshape(-1), expected.reshape(-1))
            / max(np.linalg.norm(actual) * np.linalg.norm(expected), 1e-30)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-interfaces", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.input.read_text())
    start = time.perf_counter()
    bundle = load_retrieval_model(args.model, task=spec["task"], offline=True)
    mx.eval(bundle.model.parameters())
    load_seconds = time.perf_counter() - start
    reference = np.load(args.reference)
    pairs = spec["task"] == "reranking"
    if pairs:
        encoded = [
            bundle.tokenizer.encode_pair(a, b, max_length=spec["max_length"])
            for a, b in zip(spec["first"], spec["second"], strict=True)
        ]
        ids, masks = bundle.tokenizer.pad([row[0] for row in encoded])
        types = [row[1] + [0] * (len(ids[0]) - len(row[1])) for row in encoded]
    else:
        ids, masks = bundle.tokenizer.pad(
            [
                bundle.tokenizer.encode(
                    text, add_special_tokens=True, max_length=spec["max_length"]
                )
                for text in spec["first"]
            ]
        )
        types = [[0] * len(ids[0]) for _ in ids]
    inputs = {"input_ids": ids, "attention_mask": masks, "token_type_ids": types}
    checks = {
        name: {"status": "PASS" if np.array_equal(value, reference[name]) else "FAIL"}
        for name, value in inputs.items()
        if name in reference
    }
    extra = {"representations": True} if spec.get("bge_m3") else {}
    output = bundle.model(
        mx.array(ids),
        attention_mask=mx.array(masks),
        token_type_ids=mx.array(types),
        output_hidden_states=True,
        **extra,
    )
    hidden = output.last_hidden_state
    # Capture the same actual native blocks as the isolated oracle.
    from mlx_one.models.shared.encoder import bidirectional_attention_mask

    if bundle.architecture == "nomic_bert":
        initial = bundle.model.emb_ln(bundle.model.embeddings(mx.array(ids), mx.array(types)))
        layer = bundle.model.encoder.layers[0]
    else:
        positions = None
        if bundle.architecture == "xlm-roberta":
            nonpad = (mx.array(ids) != bundle.model.config.pad_token_id).astype(mx.int32)
            positions = mx.cumsum(nonpad, axis=1) * nonpad + bundle.model.config.pad_token_id
        initial = bundle.model.embeddings(mx.array(ids), mx.array(types), positions)
        layer = bundle.model.encoder.layer[0]
    first = layer(initial, bidirectional_attention_mask(mx.array(masks), initial.dtype))
    native_values = output.logits if pairs else output.embeddings
    mx.eval(initial, first, hidden, native_values)
    traces = {
        "embedding_hidden": initial,
        "first_layer": first,
        "hidden": hidden,
        "output": native_values,
    }
    for name, value in traces.items():
        checks[name] = comparison(np.array(value), reference[name])
    if spec.get("bge_m3"):
        checks["learned_sparse_weights"] = comparison(
            np.array(output.sparse_weights), reference["sparse_weights"]
        )
        checks["colbert_embeddings"] = comparison(
            np.array(output.colbert_embeddings), reference["colbert_embeddings"]
        )
    if bundle.architecture == "nomic_bert":
        for dim in (64, 128, 256, 512, 768):
            value = bundle.model(mx.array(ids), attention_mask=mx.array(masks), dimensions=dim)
            checks[f"matryoshka_{dim}"] = comparison(
                np.array(value.embeddings), reference[f"matryoshka_{dim}"]
            )
    if pairs:
        result = rerank(bundle, spec["query"], spec["documents"], max_length=spec["max_length"])
        actual = np.array(
            [
                next(item.score for item in result.items if item.index == i)
                for i in range(len(spec["documents"]))
            ]
        )
        checks["public_pipeline"] = comparison(actual, reference["output"].reshape(-1))
        ranking = [item.index for item in result.items]
        quality = dict(
            expected_top=0, ranked_indices=ranking, status="PASS" if ranking[0] == 0 else "FAIL"
        )
    else:
        queries = embed(bundle, spec["queries"], input_type="query", max_length=spec["max_length"])
        documents = embed(bundle, spec["documents"], max_length=spec["max_length"])
        actual = np.array([*queries.embeddings, *documents.embeddings])
        checks["public_pipeline"] = comparison(actual, reference["output"])
        q = actual[0] / np.linalg.norm(actual[0])
        docs = np.asarray(documents.embeddings)
        similarities = docs @ q / np.linalg.norm(docs, axis=1)
        ranking = np.argsort(-similarities).tolist()
        quality = dict(
            expected_top=0,
            ranked_indices=ranking,
            similarities=similarities.tolist(),
            status="PASS" if ranking[0] == 0 else "FAIL",
        )
    quality.update(
        fixture="batch1-controlled-retrieval-v1",
        samples=len(spec["documents"]),
        recall_at_1=int(ranking[0] == 0),
        reciprocal_rank=1 / (ranking.index(0) + 1),
    )
    if spec.get("bge_m3"):
        from mlx_one.retrieval import encode_multi, hybrid_scores

        q = encode_multi(bundle, spec["queries"][0])
        docs = encode_multi(bundle, spec["documents"])
        ranked = hybrid_scores(q, docs)
        checks["hybrid_ranking"] = dict(
            status="PASS" if ranked[0]["index"] == 0 else "FAIL", scores=ranked
        )
        checks["sparse_special_filter"] = dict(
            status="PASS"
            if all(not {"0", "1", "2", "3"} & set(row) for row in docs.sparse_weights)
            else "FAIL"
        )
        quality["per_mode_rankings"] = {
            mode: [r["index"] for r in sorted(ranked, key=lambda row: -row[mode])]
            for mode in ("dense", "sparse", "colbert", "score")
        }
    args.output.with_name("native-core.json").write_text(
        json.dumps(dict(checks=checks, quality=quality), indent=2, allow_nan=False) + "\n"
    )
    benchmarks = []
    for width in (32, 128, 512):
        limit = getattr(
            bundle.model.config,
            "max_seq_length",
            getattr(bundle.model.config, "sentence_max_length", 0),
        )
        if width > limit:
            benchmarks.append(
                dict(requested_tokens=width, status="NOT_RUN", reason="checkpoint sentence limit")
            )
            continue
        text = " ".join(["hello"] * width)
        for batch in (1, 10, 100) if pairs else (1, 8):

            def run(text=text, batch=batch, width=width):
                if pairs:
                    return rerank(
                        bundle, "query", [text] * batch, max_length=width, batch_size=min(batch, 8)
                    )
                return embed(bundle, [text] * batch, max_length=width, batch_size=batch)

            run()
            mx.synchronize()
            samples = []
            mx.reset_peak_memory()
            for _ in range(5):
                tick = time.perf_counter()
                run()
                mx.synchronize()
                samples.append(time.perf_counter() - tick)
            benchmarks.append(
                dict(
                    status="MEASURED",
                    actual_tokens_per_item=width,
                    batch_size=batch,
                    neural_batch_limit=min(batch, 8) if pairs else batch,
                    repeats=5,
                    warmups=1,
                    p50_seconds=float(np.median(samples)),
                    p95_seconds=float(np.percentile(samples, 95)),
                    items_per_second=batch / float(np.median(samples)),
                    tokens_per_second=batch * width / float(np.median(samples)),
                    peak_mlx_bytes=mx.get_peak_memory(),
                    samples_seconds=samples,
                    scope=(
                        "native Python retrieval pipeline; model already loaded; "
                        "synthetic repeated-token input"
                    ),
                )
            )
    forbidden = [
        name
        for name in (
            "torch",
            "transformers",
            "sentence_transformers",
            "FlagEmbedding",
            "kokoro",
            "mlx_lm",
            "mlx_vlm",
        )
        if name in sys.modules
    ]
    checks["runtime_independence"] = dict(
        status="PASS" if not forbidden else "FAIL", imported=forbidden
    )
    result = dict(
        checks=checks,
        quality=quality,
        benchmarks=benchmarks,
        load_seconds=load_seconds,
        status="PASS"
        if all(row["status"] == "PASS" for row in checks.values()) and quality["status"] == "PASS"
        else "FAIL",
        output_example=result.to_dict() if pairs else documents.to_dict(),
        environment=dict(
            python=platform.python_version(),
            macos=platform.mac_ver()[0],
            mlx=importlib.metadata.version("mlx"),
            device=mx.device_info(),
        ),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    from batch1_interfaces import check_interfaces, package_identity

    identity = package_identity()
    result["package_identity"] = identity
    if args.check_interfaces:
        result["interfaces"] = check_interfaces(bundle, spec["task"], output_dir=args.output.parent)
        result["installed_distribution"] = dict(
            status="PASS"
            if identity["installed_in_environment"] and not identity["forbidden_distributions"]
            else "FAIL"
        )
        if (
            any(r["status"] != "PASS" for r in result["interfaces"].values())
            or result["installed_distribution"]["status"] != "PASS"
        ):
            result["status"] = "FAIL"
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": checks, "quality": quality}))


if __name__ == "__main__":
    main()
