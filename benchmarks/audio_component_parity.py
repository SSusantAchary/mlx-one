"""Two-process float32 Whisper architecture/frontend parity on deterministic inputs.

Reference exports safetensors and goldens without MLX. Native consumes them
without importing PyTorch or Transformers. All output paths are exclusive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CONFIG = {
    "vocab_size": 64,
    "num_mel_bins": 80,
    "d_model": 8,
    "encoder_layers": 2,
    "decoder_layers": 1,
    "encoder_attention_heads": 2,
    "decoder_attention_heads": 2,
    "encoder_ffn_dim": 16,
    "decoder_ffn_dim": 16,
    "max_source_positions": 4,
    "max_target_positions": 12,
    "pad_token_id": 1,
    "bos_token_id": 1,
    "eos_token_id": 1,
    "decoder_start_token_id": 2,
    "begin_suppress_tokens": [],
    "suppress_tokens": [],
    "dropout": 0.0,
}


def reference(directory: Path) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration

    torch.set_num_threads(4)
    torch.manual_seed(7)
    model = WhisperForConditionalGeneration(WhisperConfig(**CONFIG)).eval()
    features = torch.randn(1, 80, 8)
    tokens = torch.tensor([[2, 5, 6]])
    waveform = (np.sin(np.arange(16000, dtype=np.float32) * 0.03125) * 0.1).astype(np.float32)
    with torch.inference_mode():
        output = model(input_features=features, decoder_input_ids=tokens)
    mel = WhisperFeatureExtractor(feature_size=80)(
        waveform, sampling_rate=16000, return_tensors="np"
    ).input_features[0]
    directory.mkdir()
    save_file(
        {name: tensor.detach().clone().contiguous() for name, tensor in model.state_dict().items()},
        str(directory / "weights.safetensors"),
    )
    with (directory / "goldens.npz").open("xb") as destination:
        np.savez(
            destination,
            features=features.numpy(),
            tokens=tokens.numpy(),
            waveform=waveform,
            encoder=output.encoder_last_hidden_state.numpy(),
            logits=output.logits.numpy(),
            mel=mel,
        )
    with (directory / "config.json").open("x") as destination:
        json.dump(CONFIG, destination)
    print(json.dumps({"reference": str(directory)}), flush=True)


def native(directory: Path) -> dict:
    import mlx.core as mx

    from mlx_one.models.audio.whisper.config import WhisperConfig
    from mlx_one.models.audio.whisper.model import WhisperForConditionalGeneration
    from mlx_one.models.audio.whisper.processing import log_mel_spectrogram, pad_or_trim
    from mlx_one.models.audio.whisper.weights import sanitize_weights, weight_contract

    config = WhisperConfig.from_dict(json.loads((directory / "config.json").read_text()))
    model = WhisperForConditionalGeneration(config)
    weights = sanitize_weights(mx.load(str(directory / "weights.safetensors")), config)
    weight_contract(config).validate(weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    with np.load(directory / "goldens.npz", allow_pickle=False) as arrays:
        output = model(mx.array(arrays["features"]), mx.array(arrays["tokens"]))
        waveform = pad_or_trim(mx.array(arrays["waveform"]))
        mel = log_mel_spectrogram(waveform, num_mel_bins=80)
        mx.eval(output.encoder_last_hidden_state, output.logits, mel)
        actual = {
            "encoder": np.array(output.encoder_last_hidden_state),
            "logits": np.array(output.logits),
            "mel": np.array(mel),
        }
        errors = {
            name: float(np.max(np.abs(value - arrays[name]))) for name, value in actual.items()
        }
        gates = {
            name: bool(np.allclose(value, arrays[name], atol=1e-4, rtol=1e-4))
            for name, value in actual.items()
        }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "max_absolute_errors": errors,
        "atol": 1e-4,
        "rtol": 1e-4,
        "dtype": "float32",
        "scope": "synthetic asymmetric Whisper and frontend",
        "weights_sha256": hashlib.sha256(
            (directory / "weights.safetensors").read_bytes()
        ).hexdigest(),
        "goldens_sha256": hashlib.sha256((directory / "goldens.npz").read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "native"), required=True)
    parser.add_argument("--reference-directory", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.backend == "reference":
        if args.reference_directory.exists():
            parser.error("reference directory must not exist")
        reference(args.reference_directory)
    else:
        if args.output is None or args.output.exists():
            parser.error("native output must be supplied and must not exist")
        result = native(args.reference_directory)
        with args.output.open("x") as destination:
            json.dump(result, destination, indent=2)
        print(json.dumps(result), flush=True)
        if result["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
