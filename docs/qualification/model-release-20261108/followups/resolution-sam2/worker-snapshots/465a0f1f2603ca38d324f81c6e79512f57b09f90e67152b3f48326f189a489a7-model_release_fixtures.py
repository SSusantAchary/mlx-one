"""Prepare pinned Hugging Face samples; keep raw samples in the temporary work directory."""


def main():
    import argparse
    import hashlib
    import json
    import os
    import shutil
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Prepare pinned task-specific Hugging Face release fixtures."
    )
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    ROOT = Path(__file__).resolve().parents[2]
    WORK = args.work.resolve()
    WORK.mkdir(parents=True, exist_ok=True)
    DATA = WORK / "fixtures"
    DATA.mkdir(exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(WORK / "hf")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_XET_CACHE"] = str(WORK / "xet")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, str(ROOT))

    def sha(p):
        return hashlib.file_digest(Path(p).open("rb"), "sha256").hexdigest()

    records = {}

    def download(repo, rev, name):
        return Path(
            hf_hub_download(
                repo,
                name,
                repo_type="dataset",
                revision=rev,
                local_dir=DATA / "sources" / repo.replace("/", "--"),
            )
        )

    def parquet(key, repo, rev, name, select):
        p = download(repo, rev, name)
        rows = pq.read_table(p).to_pylist()
        selected = select(rows)
        records[key] = {
            "dataset": repo,
            "revision": rev,
            "file": name,
            "file_sha256": sha(p),
            "rows": selected,
        }

    parquet(
        "text",
        "Salesforce/wikitext",
        "b08601e04326c79dfdd32d625aee71d232d685c3",
        "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        lambda rows: [
            dict(row_id=i, text=r["text"]) for i, r in enumerate(rows) if len(r["text"]) > 120
        ][:5],
    )
    parquet(
        "similarity",
        "sentence-transformers/stsb",
        "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308",
        "data/test-00000-of-00001.parquet",
        lambda rows: [dict(row_id=i, **r) for i, r in enumerate(rows)][:32],
    )
    parquet(
        "sentiment",
        "stanfordnlp/sst2",
        "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb",
        "data/validation-00000-of-00001.parquet",
        lambda rows: [dict(row_id=i, **r) for i, r in enumerate(rows)][:32],
    )
    rev = "b18612dee0007b1f7129731dbf2f5f2ed4039ad3"
    p = download("tner/conll2003", rev, "dataset/test.json")
    records["ner"] = {
        "dataset": "tner/conll2003",
        "revision": rev,
        "file": "dataset/test.json",
        "file_sha256": sha(p),
        "rows": [
            dict(row_id=i, **json.loads(line))
            for i, line in enumerate(p.read_text().splitlines()[:16])
        ],
    }
    labels = download("tner/conll2003", rev, "dataset/label.json")
    label_copy = DATA / "ner-labels/dataset/label.json"
    label_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(labels, label_copy)
    records["ner"]["label_mapping_sha256"] = sha(labels)
    from benchmarks.vlm.prepare_images import prepare

    prepare(DATA / "images")
    records["images"] = json.loads((ROOT / "benchmarks/vlm/hf-images.json").read_text())
    from benchmarks.audio.fixtures import prepare as prepare_audio

    if not (DATA / "audio/manifest.json").exists():
        prepare_audio(DATA / "audio", WORK / "dataset-cache")
    records["audio"] = json.loads((DATA / "audio/manifest.json").read_text())
    p = download(
        "hf-internal-testing/sam2-fixtures", "ee201e1a2ba9f944f26ab4b3219a32cf6dd27252", "truck.jpg"
    )
    records["segmentation"] = {
        "dataset": "hf-internal-testing/sam2-fixtures",
        "revision": "ee201e1a2ba9f944f26ab4b3219a32cf6dd27252",
        "file": "truck.jpg",
        "path": str(p),
        "sha256": sha(p),
        "annotation_scope": "image fixture; concept/point outputs only, no human mask ground truth",
    }
    records["ocr"] = {
        "dataset": "huggingface/documentation-images",
        "revision": records["images"]["revision"],
        "file": "transformers/model_doc/ocr_receipt.jpeg",
        "path": str(DATA / "images/originals/transformers/model_doc/ocr_receipt.jpeg"),
        "evaluation": "saved transcription and manually reviewed receipt; not corpus accuracy",
    }
    from PIL import Image

    with Image.open(records["ocr"]["path"]) as receipt:
        receipt.crop((175, 180, 310, 211)).resize((810, 186)).save(DATA / "ocr-line.png")
    records["ocr"]["line_crop"] = {
        "box": [175, 180, 310, 211],
        "size": [810, 186],
        "reference": "CASH BILL",
        "sha256": sha(DATA / "ocr-line.png"),
    }
    sample = json.loads((ROOT / "docs/benchmarks/ocr/omni100-20261025/sample.json").read_text())
    row = sample["fixtures"][0]
    page = download(sample["dataset"], sample["revision"], row["path"])
    if sha(page) != row["sha256"]:
        raise ValueError("labeled OCR page checksum mismatch")
    page_record = dict(
        dataset=sample["dataset"],
        revision=sample["revision"],
        selection=(
            "first fixture in the existing seed-20261025 sample; "
            "selected before campaign OCR inference"
        ),
        fixture=row,
        image_path=str(page),
        max_tokens=1024,
        normalized_cer_limit=0.25,
    )
    (DATA / "ocr-sample.json").write_text(json.dumps(page_record, indent=2) + "\n")
    public_page = {
        **page_record,
        "fixture": {k: v for k, v in row.items() if k not in {"reference", "source_reference"}},
    }
    (args.evidence / "ocr-sample.json").write_text(json.dumps(public_page, indent=2) + "\n")
    cohort = ROOT / "docs/qualification/model-release-20261108"
    # Replay the exact already-verified public tokenizer pairing, including its
    # byte-identical publisher SentencePiece file. No gated file is requested.
    tokenizer = json.loads((cohort / "openelm-tokenizer.json").read_text())
    tokenizer_root = DATA / "openelm-tokenizer"
    expected = {**tokenizer["files"], "tokenizer.model": tokenizer["official_sentencepiece_sha256"]}
    for name, checksum in expected.items():
        path = hf_hub_download(
            tokenizer["repo"], name, revision=tokenizer["revision"], local_dir=tokenizer_root
        )
        if sha(path) != checksum:
            raise ValueError("OpenELM tokenizer asset checksum mismatch")
    (tokenizer_root / "provenance.json").write_text(json.dumps(tokenizer, indent=2) + "\n")
    records["openelm_tokenizer"] = tokenizer
    reference = json.loads((cohort / "openelm-reference-sources.json").read_text())
    reference_root = DATA / "openelm-reference-source"
    for name, checksum in reference["files"].items():
        path = hf_hub_download(
            reference["repo"], name, revision=reference["revision"], local_dir=reference_root
        )
        if sha(path) != checksum:
            raise ValueError("OpenELM reference source checksum mismatch")
    (reference_root / "provenance.json").write_text(json.dumps(reference, indent=2) + "\n")
    records["openelm_reference"] = reference
    (DATA / "inputs.json").write_text(json.dumps(records, indent=2) + "\n")
    # Persist IDs/hashes while retaining raw dataset samples only in the temporary work area.
    public = {
        k: {a: b for a, b in v.items() if a not in {"rows", "transcript"}}
        for k, v in records.items()
    }
    for k, v in records.items():
        if "rows" in v:
            public[k]["selected_row_ids"] = [r["row_id"] for r in v["rows"]]
    public["inputs_sha256"] = sha(DATA / "inputs.json")
    (args.evidence / "datasets.json").write_text(json.dumps(public, indent=2) + "\n")
    print("PREPARED", DATA, flush=True)


if __name__ == "__main__":
    main()
