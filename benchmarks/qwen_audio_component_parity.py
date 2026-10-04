"""Persist deterministic float32 Qwen components, then compare native MLX separately."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np

SOURCE = "7c6daf77a2421100f5fb066495372c00129d39ff"


def tokenizer_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            "Qwen/Qwen3-ASR-0.6B",
            revision="5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
            local_files_only=True,
        )
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(kind: str) -> dict:
    return {
        "model_type": "qwen3_asr",
        "timestamp_token_id": 6,
        "timestamp_segment_time": 80,
        "thinker_config": {
            "model_type": kind,
            "audio_token_id": 3,
            "audio_start_token_id": 4,
            "audio_end_token_id": 5,
            "classify_num": 12,
            "audio_config": {
                "d_model": 8,
                "encoder_layers": 2,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "output_dim": 8,
                "downsample_hidden_size": 2,
                "num_mel_bins": 128,
                "n_window": 50,
                "n_window_infer": 800,
                "max_source_positions": 1500,
                "conv_chunksize": 500,
            },
            "text_config": {
                "model_type": "qwen3",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "rope_theta": 1000000.0,
                "max_position_embeddings": 2048,
                "tie_word_embeddings": kind == "qwen3_asr",
                "rope_scaling": {
                    "rope_type": "default",
                    "mrope_section": [24, 20, 20],
                    "mrope_interleaved": True,
                },
            },
        },
    }


def reference(root: Path) -> dict:
    import qwen_asr
    import torch
    from qwen_asr.core.transformers_backend import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration
    from transformers import Qwen2TokenizerFast, WhisperFeatureExtractor

    source_root = Path(qwen_asr.__file__).resolve().parents[1]
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
        != SOURCE
    ):
        raise ValueError("component reference source does not match the frozen commit")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers")}
    if versions != {"torch": "2.7.1", "transformers": "4.57.6"}:
        raise ValueError("component reference environment differs from pinned versions")
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    samples = {"waveform": rng.normal(0, 0.02, 16001).astype(np.float32)}
    processor = WhisperFeatureExtractor(feature_size=128, return_attention_mask=True)
    samples["mel"] = processor(
        samples["waveform"],
        sampling_rate=16000,
        padding=True,
        truncation=False,
        return_tensors="np",
    )["input_features"][0]
    for kind in ("qwen3_asr", "qwen3_forced_aligner"):
        torch.manual_seed(123)
        payload = config(kind)
        reference_config = Qwen3ASRConfig.from_dict(payload)
        reference_config._attn_implementation = "sdpa"
        reference_config.thinker_config._attn_implementation = "sdpa"
        reference_config.thinker_config.audio_config._attn_implementation = "sdpa"
        reference_config.thinker_config.text_config._attn_implementation = "sdpa"
        model = Qwen3ASRForConditionalGeneration(reference_config).float().eval()
        np.savez(
            root / f"{kind}-weights.npz",
            **{name: value.detach().numpy() for name, value in model.state_dict().items()},
        )
        with torch.inference_mode():
            for frames in (1, 99, 100, 101, 257):
                features = rng.normal(0, 0.1, (128, frames)).astype(np.float32)
                encoded = model.thinker.audio_tower(
                    torch.from_numpy(features), feature_lens=torch.tensor([frames])
                ).last_hidden_state
                count = encoded.shape[0]
                ids = np.array([[1] + [3] * count + [2, 7, 9]], dtype=np.int64)
                logits = model.thinker(
                    input_ids=torch.from_numpy(ids),
                    input_features=torch.from_numpy(features[None]),
                    feature_attention_mask=torch.ones((1, frames), dtype=torch.long),
                    attention_mask=torch.ones_like(torch.from_numpy(ids)),
                    use_cache=False,
                ).logits
                key = f"{kind}-{frames}"
                samples[key + "-features"] = features
                samples[key + "-ids"] = ids
                samples[key + "-encoder"] = encoded.numpy()
                samples[key + "-logits"] = logits.numpy()
    np.savez(root / "fixtures.npz", **samples)
    # Freeze native tokenizer comparisons from the same checkpoint assets.
    snapshot = tokenizer_snapshot()
    tokenizer = Qwen2TokenizerFast.from_pretrained(snapshot, local_files_only=True)
    strings = ["Hello 北京 １２３ café é<asr_text>", "\nこんにちは 한국어", "<|audio_pad|>" * 13]
    metadata = {
        "source_commit": SOURCE,
        "versions": versions,
        "dtype": "float32",
        "seed": 123,
        "waveform_seed": 42,
        "configs": {kind: config(kind) for kind in ("qwen3_asr", "qwen3_forced_aligner")},
        "fixtures": {path.name: digest(path) for path in sorted(root.glob("*.npz"))},
        "tokenizer_assets": {
            name: digest(snapshot / name)
            for name in ("vocab.json", "merges.txt", "tokenizer_config.json")
        },
        "tokenizer": [
            {"text": text, "ids": tokenizer.encode(text, add_special_tokens=False)}
            for text in strings
        ],
    }
    with (root / "manifest.json").open("x") as stream:
        json.dump(metadata, stream, indent=2)
    return metadata


def native(root: Path) -> dict:
    import mlx.core as mx

    from mlx_one.models.audio.qwen3_asr.config import QwenASRConfig
    from mlx_one.models.audio.qwen3_asr.model import QwenASRModel
    from mlx_one.models.audio.qwen3_asr.processing import preprocess, tokenizer_from_directory
    from mlx_one.models.audio.qwen3_asr.weights import sanitize_weights

    manifest = json.loads((root / "manifest.json").read_text())
    expected_files = {"fixtures.npz", "qwen3_asr-weights.npz", "qwen3_forced_aligner-weights.npz"}
    if (
        manifest.get("source_commit") != SOURCE
        or set(manifest.get("fixtures", {})) != expected_files
    ):
        raise ValueError("component fixture manifest is incomplete or from another source")
    for filename, checksum in manifest["fixtures"].items():
        if digest(root / filename) != checksum:
            raise ValueError(f"component fixture checksum mismatch: {filename}")
    if manifest.get("configs") != {
        kind: config(kind) for kind in ("qwen3_asr", "qwen3_forced_aligner")
    }:
        raise ValueError("component configurations changed since reference capture")
    snapshot = tokenizer_snapshot()
    for filename, checksum in manifest["tokenizer_assets"].items():
        if filename not in {"vocab.json", "merges.txt", "tokenizer_config.json"}:
            raise ValueError("unknown component tokenizer asset")
        if digest(snapshot / filename) != checksum:
            raise ValueError(f"component tokenizer checksum mismatch: {filename}")
    fixtures = np.load(root / "fixtures.npz")
    metrics = {}

    def compare(name: str, result: object, expected: np.ndarray) -> None:
        actual = np.array(result)
        metrics[name] = {
            "pass": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
            "maximum_absolute_difference": float(np.max(np.abs(actual - expected))),
        }

    mel, _ = preprocess(mx.array(fixtures["waveform"]), offline_padding=False)
    compare("preprocessing", mel, fixtures["mel"])
    tokenizer = tokenizer_from_directory(
        snapshot,
        QwenASRConfig.from_dict(json.loads((snapshot / "config.json").read_text())),
    )
    for index, case in enumerate(manifest["tokenizer"]):
        ids = tokenizer.encode(case["text"], add_special_tokens=False).ids
        metrics[f"tokenizer-{index}"] = {"pass": ids == case["ids"]}
    for kind in ("qwen3_asr", "qwen3_forced_aligner"):
        c = QwenASRConfig.from_dict(config(kind))
        model = QwenASRModel(c)
        weights = np.load(root / f"{kind}-weights.npz")
        cleaned = sanitize_weights({name: mx.array(weights[name]) for name in weights.files}, c)
        model.load_weights(list(cleaned.items()), strict=True)
        model.eval()
        for frames in (1, 99, 100, 101, 257):
            key = f"{kind}-{frames}"
            features = mx.array(fixtures[key + "-features"])
            ids = mx.array(fixtures[key + "-ids"])
            compare(key + "-encoder", model.audio_tower(features), fixtures[key + "-encoder"])
            compare(key + "-logits", model(ids, features=features), fixtures[key + "-logits"])
            cache = model.make_cache()
            try:
                full = model(ids, features=features)
                prefill = model(ids[:, :-1], features=features, cache=cache)
                tail = model(ids[:, -1:], cache=cache)
                compare(
                    key + "-cached-logits", mx.concatenate((prefill, tail), axis=1), np.array(full)
                )
            finally:
                for state in cache:
                    state.reset()
    return {
        "status": "PASS" if all(v["pass"] for v in metrics.values()) else "FAIL",
        "source_commit": SOURCE,
        "fixture_manifest_sha256": digest(root / "manifest.json"),
        "fixtures": manifest["fixtures"],
        "reference_versions": manifest["versions"],
        "native_mlx_version": importlib.metadata.version("mlx"),
        "dtype": "float32",
        "atol": 1e-4,
        "rtol": 1e-4,
        "metrics": metrics,
        "scope": "synthetic deterministic components, not checkpoint qualification",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["reference", "native"], required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    if args.backend == "reference":
        args.directory.mkdir(parents=True, exist_ok=False)
    result = reference(args.directory) if args.backend == "reference" else native(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"output": str(args.output), "status": result.get("status", "RECORDED")}))
    if result.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
