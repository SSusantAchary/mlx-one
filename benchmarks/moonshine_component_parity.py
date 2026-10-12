"""Two-process original Moonshine float32 component parity (validation only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def reference(directory: Path, variant: str) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import MoonshineConfig, MoonshineForConditionalGeneration

    torch.manual_seed(7)
    torch.set_num_threads(4)
    config = {
        "model_type": "moonshine",
        "hidden_size": 72 if variant == "tiny" else 104,
        "intermediate_size": 128,
        "encoder_num_hidden_layers": 2,
        "decoder_num_hidden_layers": 1,
        "encoder_num_attention_heads": 2,
        "decoder_num_attention_heads": 2,
        "vocab_size": 64,
        "partial_rotary_factor": 0.9 if variant == "tiny" else 0.62,
        "pad_head_dim_to_multiple_of": 8,
        "pad_token_id": 2,
    }
    model = MoonshineForConditionalGeneration(MoonshineConfig(**config)).eval()
    model.config._attn_implementation = "eager"
    waveform = torch.randn(2, 4000) * 0.1
    mask = torch.ones_like(waveform, dtype=torch.long)
    mask[1, -384:] = 0
    tokens = torch.tensor([[1, 4, 5], [1, 6, 7]])
    with torch.inference_mode():
        output = model(input_values=waveform, attention_mask=mask, decoder_input_ids=tokens)
        front = torch.nn.functional.gelu(
            model.model.encoder.conv3(
                torch.nn.functional.gelu(
                    model.model.encoder.conv2(
                        model.model.encoder.groupnorm(
                            torch.tanh(model.model.encoder.conv1(waveform[:, None, :]))
                        )
                    )
                )
            )
        )
    directory.mkdir()
    save_file(
        {key: value.detach().clone().contiguous() for key, value in model.state_dict().items()},
        str(directory / "weights.safetensors"),
    )
    with (directory / "config.json").open("x") as handle:
        json.dump(config, handle)
    with (directory / "goldens.npz").open("xb") as handle:
        np.savez(
            handle,
            waveform=waveform.numpy(),
            mask=mask.numpy(),
            tokens=tokens.numpy(),
            encoder=output.encoder_last_hidden_state.numpy(),
            logits=output.logits.numpy(),
            frontend=front.transpose(1, 2).numpy(),
        )


def native(directory: Path) -> dict:
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_one.models.audio.moonshine.config import MoonshineConfig
    from mlx_one.models.audio.moonshine.model import MoonshineForConditionalGeneration
    from mlx_one.models.audio.moonshine.weights import sanitize_weights, weight_contract

    config = MoonshineConfig.from_dict(json.loads((directory / "config.json").read_text()))
    weights = sanitize_weights(mx.load(str(directory / "weights.safetensors")), config)
    weight_contract(config).validate(weights)
    model = MoonshineForConditionalGeneration(config)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    with np.load(directory / "goldens.npz", allow_pickle=False) as arrays:
        waveform, mask, tokens = (mx.array(arrays[key]) for key in ("waveform", "mask", "tokens"))
        output = model(waveform, tokens, attention_mask=mask)
        enc = model.model.encoder
        front = nn.gelu(
            enc.conv3(nn.gelu(enc.conv2(enc.groupnorm(mx.tanh(enc.conv1(waveform[..., None]))))))
        )
        caches = model.make_cache()
        pieces = [
            model.decode(
                tokens[:, i : i + 1],
                output.encoder_last_hidden_state,
                cache=caches,
                encoder_mask=mask[:, ::384][:, : front.shape[1]],
            ).logits
            for i in range(tokens.shape[1])
        ]
        cached = mx.concatenate(pieces, axis=1)
        mx.eval(output.logits, output.encoder_last_hidden_state, front, cached)
        actual = {
            "encoder": np.array(output.encoder_last_hidden_state),
            "logits": np.array(output.logits),
            "frontend": np.array(front),
            "cached_logits": np.array(cached),
        }
        goldens = {
            **{key: arrays[key] for key in ("encoder", "logits", "frontend")},
            "cached_logits": arrays["logits"],
        }
        errors = {key: float(np.max(np.abs(value - goldens[key]))) for key, value in actual.items()}
        gates = {
            key: bool(np.allclose(value, goldens[key], atol=1e-4, rtol=1e-4))
            for key, value in actual.items()
        }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "max_absolute_errors": errors,
        "atol": 1e-4,
        "rtol": 1e-4,
        "dtype": "float32",
        "scope": "synthetic Moonshine frontend, padded heads, partial RoPE and cached decoder",
        "reference_source_commit": "cc832f9055ba11c8c55f918ab4bda9472b910d48",
        "weights_sha256": hashlib.sha256(
            (directory / "weights.safetensors").read_bytes()
        ).hexdigest(),
        "goldens_sha256": hashlib.sha256((directory / "goldens.npz").read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "native"), required=True)
    parser.add_argument("--variant", choices=("tiny", "base"), default="tiny")
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.backend == "reference":
        if args.reference_directory.exists():
            parser.error("reference directory must not exist")
        reference(args.reference_directory, args.variant)
    else:
        if args.output is None or args.output.exists():
            parser.error("supply a new native output path")
        result = native(args.reference_directory)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result), flush=True)
        if result["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
