"""Inventory the ten revamp checkpoints offline; never infer qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mlx_one.core.registry import registered_model_types

CHECKPOINTS = (
    ("openai-community/gpt2", "gpt2", ()),
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen2",
     ("mlx-community/Qwen2.5-0.5B-Instruct-8bit", "Qwen/Qwen2.5-0.5B")),
    ("Qwen/Qwen3-0.6B", "qwen3", ("mlx-community/Qwen3-0.6B-8bit",)),
    ("LiquidAI/LFM2-350M", "lfm2", ("mlx-community/LFM2-350M-4bit",)),
    ("Qwen/Qwen3.5-0.8B", "qwen3_5", ("mlx-community/Qwen3.5-0.8B-4bit",)),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "llama", ()),
    ("meta-llama/Llama-3.2-1B-Instruct", "llama", ()),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen2", ("Qwen/Qwen2.5-1.5B",)),
    ("google/gemma-3-1b-it", "gemma3_text", ()),
    ("microsoft/Phi-3-mini-4k-instruct", "phi3", ()),
)


def local_snapshots(hub, model):
    rows = []
    for config in sorted((hub / ("models--" + model.replace("/", "--"))).glob(
            "snapshots/*/config.json")):
        root = config.parent
        weights = sorted(root.glob("*.safetensors"))
        tokenizer = (root / "tokenizer.json").is_file() or all(
            (root / name).is_file() for name in ("vocab.json", "merges.txt"))
        metadata = json.loads(config.read_text())
        rows.append({"model_id": model, "revision": root.name, "path": str(root),
                     "safe_weight_files": [p.name for p in weights],
                     "weight_bytes": sum(p.stat().st_size for p in weights),
                     "tokenizer_assets": tokenizer,
                     "status": "LOCAL_ASSETS" if weights and tokenizer else "INCOMPLETE",
                     "quantization": metadata.get("quantization_config",
                                                  metadata.get("quantization")),
                     "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                     "qualification": "NOT_QUALIFIED"})
    return rows


def inventory(root, hub):
    native = registered_model_types()
    rows = []
    for index, (model, family, alternatives) in enumerate(CHECKPOINTS, 1):
        hf_family = "gemma3" if family == "gemma3_text" else family
        refs = [root / f"references_libs/mlx-lm-main/mlx_lm/models/{family}.py",
                root / (f"references_libs/transformers-main/src/transformers/models/{hf_family}"
                        f"/modeling_{hf_family}.py")]
        rows.append({"order": index, "model_id": model, "architecture": family,
                     "native_architecture_registered": family in native,
                     "references": [{"path": str(p.relative_to(root)), "exists": p.is_file(),
                                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()
                                     if p.is_file() else None} for p in refs],
                     "exact_checkpoint": local_snapshots(hub, model),
                     "local_alternatives": [row for alt in alternatives
                                            for row in local_snapshots(hub, alt)],
                     "qualification": "NOT_QUALIFIED"})
    return {"schema_version": "revamp-first-ten-inventory-v1", "offline": True,
            "checkpoint_count": len(rows), "checkpoints": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--hub", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new inventory path")
    report = inventory(args.root, args.hub)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checkpoint_count": report["checkpoint_count"], "offline": True}))


if __name__ == "__main__":
    main()
