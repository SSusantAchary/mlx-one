"""Resolve immutable Hugging Face checkpoint identities for all registered model types."""


def main():
    import argparse
    import concurrent.futures
    import json
    from pathlib import Path

    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(
        description="Freeze the exact 40-model checkpoint inventory without downloading weights."
    )
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    ROOT = Path(__file__).resolve().parents[2]
    OUT = args.evidence
    OUT.mkdir(parents=True, exist_ok=True)

    def read(p):
        return json.loads((ROOT / p).read_text())

    entries = []

    def add(mt, kind, repo, rev=None, **kw):
        entries.append(dict(model_type=mt, kind=kind, repo=repo, revision=rev, **kw))

    text = read("src/mlx_one/data/compatibility/alpha_release.json")["checkpoints"]
    for mid in ["M01", "M02", "M03", "M04", "M06", "M09", "M10", "M11", "M12"]:
        r = next(x for x in text if x["id"] == mid)
        add(
            r["model_type"],
            "text",
            r["model_id"],
            r["revision"],
            profile=r["primary_profile"],
            manifest_id=mid,
        )
    add("openelm", "text", "apple/OpenELM-270M")
    add(
        "qwen2_moe",
        "text",
        "mlx-community/Qwen1.5-MoE-A2.7B-4bit",
        profile="stored_affine_q4",
        selection_reason=(
            "Native Q4 representative fits the 32GB host; "
            "publisher BF16 is a separate untested artifact"
        ),
    )
    for r in read("docs/benchmarks/vlm/benchmark_plan.json")["checkpoints"]:
        if r["id"] == "V05":
            continue
        add(
            r["architecture"],
            "vlm",
            r["checkpoint"],
            r["revision"],
            manifest_id=r["id"],
            profile=r["precision"],
        )
    audio = read("benchmarks/audio/checkpoints.json")["entries"]
    for mt, aid in [
        ("whisper", "A01"),
        ("moonshine", "A04"),
        ("qwen3_asr", "A06"),
        ("parakeet_tdt", "A09"),
        ("qwen3_forced_aligner", "A11"),
        ("lfm2_audio", "A10"),
    ]:
        r = next(x for x in audio if x["key"] == aid)
        add(
            mt,
            "audio",
            r["checkpoint"],
            r["revision"],
            assets=r["assets"],
            manifest_id=aid,
            profile=r["precision"],
        )
    add(
        "silero_vad",
        "vad",
        "snakers4/silero-vad",
        "60b7ffa243625ebdc1070275a29f18c87843786a",
        source="github",
    )
    for r in read("docs/benchmarks/ocr/inventory.json")["checkpoints"]:
        if r["id"] == "O02":
            continue
        add(
            r["family"],
            "ocr",
            r["checkpoint"],
            r["revision"],
            manifest_id=r["id"],
            profile=r["precision"],
        )
    for mt, repo in [
        ("bert", "sentence-transformers/all-MiniLM-L6-v2"),
        ("mpnet", "sentence-transformers/all-mpnet-base-v2"),
        ("qwen3_embedding", "Qwen/Qwen3-Embedding-0.6B"),
        ("qwen3_reranker", "Qwen/Qwen3-Reranker-0.6B"),
        ("lfm2_colbert", "LiquidAI/LFM2-ColBERT-350M"),
    ]:
        add(mt, "retrieval", repo)
    for mt in ["sam3", "sam3_tracker", "sam3_video", "sam3_tracker_video"]:
        add(mt, "segmentation", "facebook/sam3", "3c879f39826c281e95690f02c7821c4de09afae7")
    for mt in ["distilbert", "bert_token_classification", "clip"]:
        r = next(
            x
            for x in read("src/mlx_one/data/native_tasks.json")["profiles"]
            if x["model_type"] == mt
        )
        add(mt, "classification", r["model_id"], r["revision"], profile=r["precision"])
    assert len(entries) == 40, (len(entries), [e["model_type"] for e in entries])
    api = HfApi()

    def pin(e):
        if e.get("source") == "github":
            return e
        try:
            info = api.model_info(e["repo"], revision=e["revision"], files_metadata=True)
            e["revision"] = info.sha
            e["gated"] = info.gated
            e["license"] = getattr(info.card_data, "license", None) if info.card_data else None
            e["files"] = [
                {"path": f.rfilename, "bytes": f.size, "sha256": f.lfs.sha256 if f.lfs else None}
                for f in info.siblings
            ]
            e["metadata_status"] = "PASS"
        except Exception as ex:
            e["metadata_status"] = "FAIL"
            e["metadata_error"] = str(ex)
        return e

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        entries = list(pool.map(pin, entries))
    (OUT / "checkpoint-plan.json").write_text(json.dumps(entries, indent=2) + "\n")
    for e in entries:
        weights = [
            f
            for f in e.get("files", [])
            if f["path"].endswith(".safetensors")
            and not f["path"].startswith(("onnx/", "openvino/"))
        ]
        print(
            e["model_type"],
            e.get("metadata_status", "github"),
            e.get("revision"),
            round(sum(f["bytes"] or 0 for f in weights) / 1e9, 2),
            e.get("metadata_error", "")[:180],
            flush=True,
        )


if __name__ == "__main__":
    main()
