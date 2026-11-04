#!/usr/bin/env python3
"""Bounded VAD clip-presence evaluation; no human segment-boundary claim."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=("reference", "native"))
    p.add_argument("checkpoint", type=Path)
    p.add_argument("fixtures", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    fixtures = json.loads(args.fixtures.read_text())["fixtures"]
    rows = []
    if args.backend == "reference":
        import ast

        import torch
        from safetensors.torch import load_file

        torch.set_num_threads(4)
        model = torch.jit.load(str(args.checkpoint / "silero_vad.jit"), map_location="cpu").eval()
        state = model.state_dict()
        weights = load_file(args.checkpoint / "silero_vad_16k.safetensors")
        for k, v in weights.items():
            if k == "stft_conv.weight":
                target = "_model.stft.forward_basis_buffer"
            elif k.startswith("conv"):
                target = f"_model.encoder.{int(k[4]) - 1}.reparam_conv.{k.split('.')[1]}"
            elif k.startswith("lstm_cell."):
                target = "_model.decoder.rnn." + k.split(".")[1]
            else:
                target = "_model.decoder.decoder.2." + k.split(".")[1]
            state[target] = v
        model.load_state_dict(state, strict=True)
        tree = ast.parse((args.checkpoint / "utils_vad.py").read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "get_speech_timestamps_from_probs"
        )
        namespace = {"List": list}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "pinned-utils.py", "exec"), namespace)
        timestamps = namespace[fn.name]
        for f in fixtures:
            pcm = np.load(f["path"], allow_pickle=False)
            padded = np.pad(pcm, (0, -len(pcm) % 512))
            model.reset_states()
            with torch.inference_mode():
                probs = [
                    float(model(torch.from_numpy(padded[i : i + 512]), 16000).item())
                    for i in range(0, len(padded), 512)
                ]
            rows.append(
                {
                    "id": f["id"],
                    "gold_speech_present": f["speech_present"],
                    "probabilities": probs,
                    "segments": timestamps(probs, audio_length_samples=len(pcm)),
                }
            )
    else:
        from mlx_one import VADEngine

        with VADEngine(args.checkpoint) as engine:
            for f in fixtures:
                result = engine.detect(np.load(f["path"], allow_pickle=False))
                rows.append(
                    {
                        "id": f["id"],
                        "gold_speech_present": f["speech_present"],
                        "probabilities": list(result.probabilities),
                        "segments": list(result.segments),
                    }
                )
    tp = sum(r["gold_speech_present"] and bool(r["segments"]) for r in rows)
    fp = sum(not r["gold_speech_present"] and bool(r["segments"]) for r in rows)
    fn = sum(r["gold_speech_present"] and not r["segments"] for r in rows)
    tn = sum(not r["gold_speech_present"] and not r["segments"] for r in rows)
    with args.output.open("x") as stream:
        json.dump(
            {
                "backend": args.backend,
                "metric": "clip speech presence, not segment F1",
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": tp / (tp + fp) if tp + fp else 0,
                "recall": tp / (tp + fn) if tp + fn else 0,
                "rows": rows,
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
