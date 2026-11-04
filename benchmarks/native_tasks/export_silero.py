#!/usr/bin/env python3
"""Official pinned JIT vs native Silero chunk states, convs and probabilities."""

import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=("reference", "native"))
    p.add_argument("checkpoint", type=Path)
    p.add_argument("waveform", type=Path, help="mono 16-kHz PCM16 WAV or float32 NPY")
    p.add_argument("output", type=Path)
    p.add_argument("--max-chunks", type=int)
    args = p.parse_args()
    if args.waveform.suffix == ".npy":
        pcm = np.load(args.waveform, allow_pickle=False)
    else:
        with wave.open(str(args.waveform), "rb") as f:
            if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
                raise ValueError("oracle fixture must be mono 16-kHz PCM16 WAV")
            pcm = (
                np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768
            )
    if pcm.ndim != 1 or pcm.dtype != np.float32 or not np.isfinite(pcm).all():
        raise ValueError("expected finite mono FP32 PCM")
    if args.max_chunks is not None:
        pcm = pcm[: 512 * args.max_chunks]
    padded = np.pad(pcm, (0, -len(pcm) % 512))
    arrays = {"pcm": pcm}
    if args.backend == "reference":
        import torch
        from safetensors.torch import load_file

        torch.set_num_threads(4)
        model = torch.jit.load(str(args.checkpoint / "silero_vad.jit"), map_location="cpu").eval()
        source = model.state_dict()
        weights = load_file(args.checkpoint / "silero_vad_16k.safetensors")
        mapping = {"stft_conv.weight": "_model.stft.forward_basis_buffer"}
        for i in range(1, 5):
            for side in ("weight", "bias"):
                mapping[f"conv{i}.{side}"] = f"_model.encoder.{i - 1}.reparam_conv.{side}"
        for side in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            mapping["lstm_cell." + side] = "_model.decoder.rnn." + side
        for side in ("weight", "bias"):
            mapping["final_conv." + side] = "_model.decoder.decoder.2." + side
        assert set(mapping) == set(weights)
        # The pinned JIT ships different learned weights. Use its matching module
        # graph as an offline oracle for the *selected safetensors*, explicitly
        # rebinding every 16-kHz parameter and verifying the rebound state.
        for key, target in mapping.items():
            source[target] = weights[key]
        model.load_state_dict(source, strict=True)
        source = model.state_dict()
        assert all(torch.equal(weights[k], source[v]) for k, v in mapping.items())
        context, state = torch.zeros(1, 64), torch.zeros(2, 1, 128)
        probs, states, contexts, traces = [], [], [], {}
        with torch.inference_mode():
            for i in range(0, len(padded), 512):
                chunk = torch.from_numpy(padded[i : i + 512][None])
                x = torch.cat((context, chunk), dim=1)
                spectrum = torch.nn.functional.conv1d(
                    torch.nn.functional.pad(x, (0, 64), mode="reflect")[:, None],
                    source["_model.stft.forward_basis_buffer"],
                    stride=128,
                )
                trace = {"stft": spectrum, "magnitude": model._model.run_extractors(x)}
                x1 = trace["magnitude"]
                for j in range(4):
                    x1 = getattr(model._model.encoder, str(j))(x1)
                    trace[f"conv{j + 1}"] = x1
                prob = model(chunk, 16000)
                state, context = model._state.clone(), model._context.clone()
                probs.append(prob.numpy())
                states.append(state.numpy())
                contexts.append(context.numpy())
                for k, v in trace.items():
                    traces.setdefault(k, []).append(v.numpy())
    else:
        import mlx.core as mx

        from mlx_one.models.audio.silero.model import load_silero

        bundle = load_silero(args.checkpoint)
        context, state = mx.zeros((1, 64)), mx.zeros((2, 1, 128))
        probs, states, contexts, traces = [], [], [], {}
        for i in range(0, len(padded), 512):
            chunk = mx.array(padded[i : i + 512][None])
            prob, state, trace = bundle.model(
                mx.concatenate((context, chunk), axis=1), state, trace=True
            )
            context = chunk[:, -64:]
            mx.eval(prob, state, context, trace)
            probs.append(np.asarray(prob))
            states.append(np.asarray(state))
            contexts.append(np.asarray(context))
            for k, v in trace.items():
                traces.setdefault(k, []).append(np.asarray(v))
    if probs:
        arrays.update({k: np.stack(v) for k, v in traces.items()})
        arrays.update(
            probabilities=np.stack(probs), state=np.stack(states), context=np.stack(contexts)
        )
    with args.output.open("xb") as stream:
        np.savez(stream, **arrays)
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(
            {
                "backend": args.backend,
                "chunks": len(probs),
                "samples": len(pcm),
                "waveform_sha256": hashlib.file_digest(
                    args.waveform.open("rb"), "sha256"
                ).hexdigest(),
                "source_commit": "60b7ffa243625ebdc1070275a29f18c87843786a",
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
