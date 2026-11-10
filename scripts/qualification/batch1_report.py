"""Derive the ten-checkpoint review manifest and benchmark table from saved runs.

This never rewrites individual measurements or grants a package release. The
specific ViT functional decision is separate from its failed hidden diagnostic.
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/qualification/batch-1-sub-6b"
TYPES = [
    "bert",
    "bert",
    "xlm-roberta",
    "bert",
    "t5",
    "vit",
    "xlm-roberta",
    "kokoro",
    "nomic_bert",
    "bert",
]
TASKS = [
    "reranking",
    "embedding",
    "embedding",
    "embedding",
    "text-to-text-generation",
    "image-classification",
    "reranking",
    "speech-generation",
    "embedding",
    "embedding",
]
LIMITS = {
    "B01": "512 tokens including pair specials; scalar logit and separate sigmoid",
    "B02": "384-D CLS + L2; 512 tokens; no default query instruction",
    "B03": "Dense/sparse/ColBERT; fixed-example reference; execution/timing through 512 tokens",
    "B04": "384-D masked mean without normalization; 128-token sentence limit",
    "B05": "Greedy encoder-decoder only; 512 encoder tokens; bounded prefix recomputation",
    "B06": "Fixed 224 RGB supervised classification; no universal hidden-state parity",
    "B07": "Raw paired score; fixed-example reference; execution/timing through 512 tokens",
    "B08": "af_heart en-us only; 510 phonemes, 30 seconds, 24 kHz; no streaming",
    "B09": "768-D normalized default; explicit Matryoshka 64/128/256/512/768; through 512 tokens",
    "B10": "384-D mean + L2; 512 tokens; explicit query:/passage: formatting",
}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def artifact(path):
    return dict(
        path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def main():
    verification = read(EVIDENCE / "verification.json")
    assert verification["native_regression"]["status"] == "PASS"
    assert verification["python_matrix"]["status"] == "PASS"
    rows, benchmarks = [], []
    for i in range(1, 11):
        key = f"B{i:02}"
        cohort = (
            "installed-resource"
            if key == "B08"
            else "installed-replay"
            if key in {"B01", "B02", "B05"}
            else "installed"
        )
        selected = EVIDENCE / cohort / key
        native = read(selected / "native.json")
        metadata = read(EVIDENCE / "metadata" / key / "summary.json")
        if key == "B08":
            checkpoint = read(selected / "assets.json")
            cleanup = selected / "cleanup.json"
            reference = EVIDENCE / key / "reference.json"
        else:
            checkpoint = read(selected / "result.json")
            cleanup, reference = selected / "cleanup.json", selected / "reference.json"
        assert read(cleanup)["status"] == "REMOVED_OWNED_DOWNLOADS"
        assert all(c["status"] == "PASS" for c in native["interfaces"].values())
        assert native["installed_distribution"]["status"] == "PASS"
        if key == "B06":
            decision = read(EVIDENCE / key / "functional-decision.json")
            assert decision["status"] == "PASS_FUNCTIONAL"
            assert native["checks"]["logits"]["status"] == "PASS"
            assert native["checks"]["top5_parity"]["status"] == "PASS"
            assert native["checks"]["hidden"]["status"] == "FAIL"
        else:
            assert native["status"] == "PASS", key
        if key == "B08":
            listening = read(EVIDENCE / key / "listening.json")
            audio = EVIDENCE / key / listening["audio"]
            assert listening["status"] == "PASS"
            assert hashlib.sha256(audio.read_bytes()).hexdigest() == listening["audio_sha256"]
        reference_result = read(reference)
        interfaces = {"python": {"status": "PASS"}, **native["interfaces"]}
        row = dict(
            id=key,
            model_type=TYPES[i - 1],
            model_id=metadata["checkpoint"],
            revision=metadata["revision"],
            license=metadata["license"],
            task=TASKS[i - 1],
            status="SUPPORTED",
            admission_tier="VALIDATED_SCOPED",
            release_approval="NOT_REQUESTED",
            precision="FP32",
            storage_precision="FP32 backbone / FP16 auxiliary heads" if key == "B03" else "FP32",
            source_parameter_count=reference_result.get("parameter_count"),
            native_parameter_count=verification["native_parameter_counts"][key]["elements"],
            native_module=(
                "mlx_one.models.embeddings.bert"
                if TYPES[i - 1] == "bert"
                else {
                    "xlm-roberta": "mlx_one.models.embeddings.xlm_roberta",
                    "nomic_bert": "mlx_one.models.embeddings.nomic_bert",
                    "t5": "mlx_one.models.language.t5",
                    "vit": "mlx_one.models.classification.vit",
                    "kokoro": "mlx_one.models.audio.kokoro",
                }[TYPES[i - 1]]
            ),
            assets=checkpoint["assets"],
            evidence=artifact(selected / "native.json"),
            reference=artifact(reference),
            cleanup=artifact(cleanup),
            interfaces=interfaces,
            numerical_checks=native["checks"],
            quality=native.get("quality", {}),
            known_limitations=[
                LIMITS[key],
                "No quantized profile or external speed comparison admitted",
            ],
            gates={
                **{gate: "PASS" for gate in ("G0", "G1", "G2", "G3", "G4")},
                "G5": "EVIDENCE_READY_FOR_MAINTAINER_REVIEW",
            },
            gate_basis=(
                "Bounded primary functional workflow; see shared verification "
                "and scope-specific diagnostics"
            ),
        )
        if key == "B06":
            row["functional_decision"] = artifact(EVIDENCE / key / "functional-decision.json")
            row["numerical_status"] = "CLASSIFICATION_PASS_WITH_RETAINED_HIDDEN_DIAGNOSTIC_FAIL"
        else:
            row["numerical_status"] = "PASS_FOR_RECORDED_SCOPE"
        if key == "B03":
            row["modes"] = {
                "dense": "PASS",
                "learned_sparse": "PASS",
                "colbert": "PASS",
                "hybrid": "PASS",
            }
            row["conversion"] = artifact(selected / "conversion.json")
        if key == "B08":
            row["conversion"] = artifact(selected / "conversion.json")
            row["listening"] = artifact(EVIDENCE / key / "listening.json")
            row["asr_roundtrip"] = artifact(EVIDENCE / key / "asr-roundtrip.json")
            row["vocoder_diagnostic"] = native["vocoder_diagnostic"]
            row["known_limitations"].append(
                "Human/ASR review references the retained original WAV; "
                "installed synthesis has a separate hash. "
                "No exact stochastic waveform equality claim."
            )
        if key == "B08":
            row["known_limitations"].append(
                "Actor reserves 3 GiB transient memory; configure at least that capacity "
                "in addition to resident weights (server task budget: 4 GiB)."
            )
        for sample in native.get("benchmarks", []):
            benchmarks.append(dict(id=key, model_id=metadata["checkpoint"], **sample))
        for sample in native.get("benchmark_samples", []):
            benchmarks.append(
                dict(
                    id=key,
                    model_id=metadata["checkpoint"],
                    status="MEASURED",
                    scope="loaded native Python pipeline",
                    **sample,
                )
            )
        rows.append(row)
    assert len(rows) == 10 and len({row["id"] for row in rows}) == 10
    catalog = dict(
        schema_version=1,
        campaign="batch-1-sub-6b",
        recorded_date="2026-11-10",
        checkpoint_count=10,
        added_architecture_types=5,
        registered_model_types=45,
        support_standard="mlx-lm-aligned-functional-v1",
        release_approval="NOT_REQUESTED",
        note="Development checkpoint support; the dated 40-type 0.1.1 approval is unchanged.",
        models=rows,
    )
    save(EVIDENCE / "results.json", catalog)
    packaged = {k: v for k, v in catalog.items() if k != "models"}
    packaged["models"] = [
        {
            k: v
            for k, v in row.items()
            if k
            in {
                "id",
                "model_type",
                "model_id",
                "revision",
                "license",
                "task",
                "status",
                "admission_tier",
                "release_approval",
                "precision",
                "storage_precision",
                "native_module",
                "source_parameter_count",
                "native_parameter_count",
                "known_limitations",
                "numerical_status",
                "evidence",
                "functional_decision",
                "modes",
            }
        }
        for row in rows
    ]
    save(ROOT / "src/mlx_one/data/batch_1_sub_6b.json", packaged)
    save(
        EVIDENCE / "benchmarks.json",
        dict(
            hardware="Apple M4, 32 GiB",
            mlx="0.32.2",
            mlxboost="OFF",
            external_comparison="NOT_RUN",
            rows=benchmarks,
        ),
    )
    fields = sorted(
        {k for row in benchmarks for k, value in row.items() if not isinstance(value, (list, dict))}
    )
    with (EVIDENCE / "benchmarks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(benchmarks)
    print("Generated exactly ten scoped checkpoint records and measured benchmark exports")


if __name__ == "__main__":
    main()
