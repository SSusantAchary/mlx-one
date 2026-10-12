"""Persist float32 Parakeet reference components before native Metal comparison."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
from audio_qualification import digest

SOURCE = "469230357aab0f2b303b0d638c1f8d06edb14184"
CONFIG = {
    "model_type": "parakeet_tdt",
    "vocab_size": 9,
    "blank_token_id": 8,
    "pad_token_id": 2,
    "decoder_hidden_size": 4,
    "num_decoder_layers": 2,
    "durations": [0, 1, 2, 3, 4],
    "max_symbols_per_step": 10,
    "encoder_config": {
        "model_type": "parakeet_encoder",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_mel_bins": 16,
        "subsampling_conv_channels": 2,
        "subsampling_factor": 8,
        "attention_bias": False,
        "convolution_bias": False,
        "scale_input": False,
    },
}


def reference(fixture: Path) -> dict:
    import torch
    import transformers
    from transformers import ParakeetForTDT, ParakeetTDTConfig
    from transformers.models.parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor

    root = Path(transformers.__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != SOURCE or importlib.metadata.version("torch") != "2.7.1":
        raise ValueError("component reference differs from frozen source/environment")
    torch.set_num_threads(4)
    torch.manual_seed(314)
    model = ParakeetForTDT(ParakeetTDTConfig(**CONFIG)).eval()
    values = {"source": np.array(SOURCE), "config": np.array(json.dumps(CONFIG))}
    values.update({"weight:" + key: value.numpy() for key, value in model.state_dict().items()})
    x = torch.randn(1, 25, 16)
    mask = torch.arange(25)[None] < 24
    with torch.inference_mode():
        encoded = model.get_audio_features(x, attention_mask=mask)
        first = model.decoder(torch.tensor([[2]]))
        from transformers.models.parakeet.generation_parakeet import ParakeetRNNTDecoderCache

        cache = ParakeetRNNTDecoderCache(model.config)
        model.decoder(torch.tensor([[2]]), cache=cache)
        second = model.decoder(torch.tensor([[4]]), cache=cache)
        values.update(
            {
                "input": x.numpy(),
                "mask": mask.numpy(),
                "encoder": encoded.pooler_output.numpy(),
                "decoder_first": first.numpy(),
                "decoder_second": second.numpy(),
                "hidden": cache.hidden_state.numpy(),
                "cell": cache.cell_state.numpy(),
            }
        )
        rng = np.random.default_rng(314)
        waveform = rng.normal(0, 0.01, 1600).astype(np.float32)
        processor = ParakeetFeatureExtractor(feature_size=128)
        features = processor(waveform, sampling_rate=16000, return_tensors="pt")
        silence = processor(
            np.zeros(1600, dtype=np.float32), sampling_rate=16000, return_tensors="pt"
        )
        values.update(
            {
                "waveform": waveform,
                "features": features.input_features.numpy(),
                "silence": silence.input_features.numpy(),
                "mel_filters": processor.mel_filters.numpy(),
            }
        )
    with fixture.open("xb") as stream:
        np.savez(stream, **values)
    return {
        "reference_source_commit": SOURCE,
        "fixture_sha256": digest(fixture),
        "dtype": "float32",
        "config": CONFIG,
        "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }


def native(fixture: Path) -> dict:
    values = np.load(fixture, allow_pickle=False)
    if str(values["source"]) != SOURCE or json.loads(str(values["config"])) != CONFIG:
        raise ValueError("component fixture differs from frozen reference contract")
    import mlx.core as mx

    from mlx_one.models.audio.parakeet.config import ParakeetConfig
    from mlx_one.models.audio.parakeet.model import ParakeetTDT
    from mlx_one.models.audio.parakeet.processing import ParakeetProcessor, mel_filters, preprocess
    from mlx_one.models.audio.parakeet.weights import sanitize_weights

    model = ParakeetTDT(ParakeetConfig.from_dict(CONFIG))
    raw = {
        key.removeprefix("weight:"): mx.array(values[key])
        for key in values.files
        if key.startswith("weight:")
    }
    model.load_weights(list(sanitize_weights(raw, model.config).items()), strict=True)
    encoded, _ = model.encode(mx.array(values["input"]), mx.array([24]))
    first, state = model.decoder(2)
    second, state = model.decoder(4, state)
    processor = ParakeetProcessor()
    features, _ = preprocess(mx.array(values["waveform"]), processor)
    silence, _ = preprocess(mx.zeros(1600), processor)
    actual = {
        "encoder": encoded,
        "decoder_first": first[:, None],
        "decoder_second": second[:, None],
        "hidden": state[0],
        "cell": state[1],
        "features": features,
        "silence": silence,
        "mel_filters": mel_filters(processor),
    }
    reports = {}
    for name, tensor in actual.items():
        mx.eval(tensor)
        left, right = values[name], np.array(tensor)
        reports[name] = {
            "pass": bool(np.allclose(left, right, atol=1e-4, rtol=1e-4)),
            "max_abs": float(np.max(np.abs(left - right))),
        }
    return {
        "status": "PASS" if all(r["pass"] for r in reports.values()) else "FAIL",
        "reference_source_commit": SOURCE,
        "fixture_sha256": digest(fixture),
        "atol": 1e-4,
        "rtol": 1e-4,
        "reports": reports,
        "scope": "float32 deterministic components; not checkpoint qualification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "native"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    result = reference(args.fixture) if args.backend == "reference" else native(args.fixture)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
    raise SystemExit(1 if result.get("status") == "FAIL" else 0)


if __name__ == "__main__":
    main()
