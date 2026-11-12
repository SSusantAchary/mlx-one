"""Run the audited publisher Kokoro model in the isolated Torch environment."""

import argparse
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    package = types.ModuleType("kokoro")
    package.__path__ = [str(args.source / "kokoro")]
    sys.modules["kokoro"] = package
    publisher = importlib.import_module("kokoro.model")
    config = json.loads((args.model / "config.json").read_text())
    model = publisher.KModel(
        repo_id="hexgrad/Kokoro-82M", config=config, model=str(args.model / "kokoro-v1_0.pth")
    ).eval()
    phonemes = "həlˈO wˈɜɹld."
    voice = torch.load(args.model / "voices/af_heart.pt", map_location="cpu", weights_only=True)[
        len(phonemes) - 1
    ]
    tokens = torch.tensor([[0, *[config["vocab"][char] for char in phonemes], 0]])
    captured = {}
    model.bert.register_forward_hook(lambda m, i, o: captured.update(albert=o))
    model.predictor.text_encoder.register_forward_hook(
        lambda m, i, o: captured.update(duration_features=o)
    )
    model.predictor.duration_proj.register_forward_hook(
        lambda m, i, o: captured.update(duration_logits=o)
    )
    model.predictor.F0_proj.register_forward_hook(lambda m, i, o: captured.update(f0=o))
    model.predictor.N_proj.register_forward_hook(lambda m, i, o: captured.update(noise=o))
    model.decoder.generator.m_source.l_sin_gen.register_forward_hook(
        lambda m, i, o: captured.update(forcing=o[0])
    )
    with torch.inference_mode():
        audio, durations = model.forward_with_tokens(tokens, voice)
    arrays = {name: value.detach().numpy() for name, value in captured.items()}
    arrays.update(waveform=audio.numpy(), durations=durations.numpy(), voice=voice.numpy())
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                phonemes=phonemes,
                voice="af_heart",
                sample_rate=24000,
                torch=torch.__version__,
                source_files={
                    str(path.relative_to(args.source)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in (args.source / "kokoro").glob("*.py")
                },
                parameter_count=sum(p.numel() for p in model.parameters()),
                waveform_samples=int(audio.numel()),
                source_revision="dfb907a02bba8152ca444717ca5d78747ccb4bec",
            ),
            indent=2,
        )
        + "\n"
    )
    print("Publisher waveform samples", audio.numel(), "durations", durations.tolist())


if __name__ == "__main__":
    main()
