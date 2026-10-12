"""Checkpoint BF16 rounding diagnostic, independent of acceptance thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from audio_qualification import digest
from qwen_audio_qualification import PINS, SOURCE

PREFIX = "Her eyes seemed to regard him with mild pity"


def validate_token_fixture(fixture):
    required = ("checkpoint_model", "checkpoint_revision", "reference_source_commit")
    if any(name not in fixture for name in required):
        raise ValueError("token fixture lacks the frozen checkpoint/source contract")
    model = str(fixture["checkpoint_model"])
    if (
        model not in PINS
        or model == "Qwen/Qwen3-ForcedAligner-0.6B"
        or str(fixture["checkpoint_revision"]) != PINS[model]
        or str(fixture["reference_source_commit"]) != SOURCE
    ):
        raise ValueError("token fixture differs from frozen checkpoint/source")
    return model


def checkpoint(model="Qwen/Qwen3-ASR-0.6B") -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model,
            revision=PINS[model],
            local_files_only=True,
        )
    )


def score_decision(scores):
    """Expose ties without relaxing or modifying any acceptance decision."""
    if scores.ndim != 1 or scores.size < 2 or not np.isfinite(scores).all():
        raise ValueError("score diagnostic requires a finite vocabulary vector")
    ordered = np.sort(scores)
    return {
        "argmax": int(scores.argmax()),
        "maximum": float(ordered[-1]),
        "runner_up": float(ordered[-2]),
        "margin": float(ordered[-1] - ordered[-2]),
        "tied_maximum_ids": np.flatnonzero(scores == ordered[-1]).tolist(),
    }


def reference(audio: Path, output: Path, model: str, prefix: str):
    import importlib.metadata
    import subprocess

    import qwen_asr
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(qwen_asr.__file__).resolve().parents[1],
        text=True,
    ).strip()
    if (
        commit != SOURCE
        or importlib.metadata.version("torch") != "2.7.1"
        or importlib.metadata.version("transformers") != "4.57.6"
    ):
        raise ValueError("reference source/environment differs from frozen contract")

    torch.set_num_threads(4)
    bundle = Qwen3ASRModel.from_pretrained(
        str(checkpoint(model)),
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="sdpa",
        max_new_tokens=512,
    )
    waveform, rate = sf.read(audio, dtype="float32")
    prompt = bundle._build_text_prompt("", "English") + prefix
    inputs = bundle.processor(
        text=prompt, audio=waveform, sampling_rate=rate, padding=True, return_tensors="pt"
    ).to(dtype=torch.bfloat16)
    values = {
        "waveform": waveform,
        "ids": inputs.input_ids.numpy(),
        "features": inputs.input_features.float().numpy(),
        "checkpoint_model": np.array(model),
        "checkpoint_revision": np.array(PINS[model]),
        "reference_source_commit": np.array(SOURCE),
    }
    handles = []

    def hook(name):
        def capture(module, inp, out):
            if hasattr(out, "last_hidden_state"):
                out = out.last_hidden_state
            out = out[0] if isinstance(out, tuple) else out
            values[name] = out.detach().float().numpy()
            if inp and isinstance(inp[0], torch.Tensor):
                values[name + ".input"] = inp[0].detach().float().numpy()

        return capture

    for name, module in bundle.model.thinker.named_modules():
        if (
            name == "audio_tower"
            or name.startswith("audio_tower.layers.")
            and name.count(".") == 2
            or name.startswith("model.layers.")
            and name.count(".") == 2
            or name.startswith("model.layers.0.")
            and not list(module.children())
        ):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():
        result = bundle.model.thinker(**inputs, use_cache=False)
    values["logits"] = result.logits[0, -1].float().numpy()
    for handle in handles:
        handle.remove()
    # The exact acceptance gate uses cached generation, not teacher-forced full
    # decoding; BF16 matrix shapes can change rounding even within the reference.
    base = bundle.processor(
        text=bundle._build_text_prompt("", "English"),
        audio=waveform,
        sampling_rate=rate,
        padding=True,
        return_tensors="pt",
    ).to(dtype=torch.bfloat16)
    with torch.inference_mode():
        generated = bundle.model.generate(**base, max_new_tokens=128, output_scores=True)
    new_ids = generated.sequences[0, base.input_ids.shape[1] :].tolist()
    target = next(
        i
        for i in range(len(new_ids))
        if bundle.processor.tokenizer.decode(new_ids[:i], skip_special_tokens=True) == prefix
    )
    values["prompt_ids"] = base.input_ids.numpy()
    values["teacher_ids"] = np.array(new_ids[:target], dtype=np.int64)
    values["cached_logits"] = generated.scores[target][0].float().numpy()
    with output.open("xb") as stream:
        np.savez(stream, **values)
    return {
        "scope": "teacher-forced numerical diagnostic, not qualification",
        "model": model,
        "revision": PINS[model],
        "reference_source_commit": SOURCE,
        "fixture_sha256": digest(output),
        "input_audio_sha256": digest(audio),
        "teacher_prefix": prefix,
        "cached_decision": score_decision(values["cached_logits"]),
    }


def native(reference: Path, candidate: str, reference_audio: bool):
    fixture = np.load(reference, allow_pickle=False)
    model = validate_token_fixture(fixture)
    import mlx.core as mx

    from mlx_one.models.audio.qwen3_asr.loading import load_qwen_audio
    from mlx_one.models.audio.qwen3_asr.processing import preprocess

    bundle = load_qwen_audio(checkpoint(model))
    if candidate in ("audio-rounding", "all-rounding", "reference-order"):
        from qwen_audio_rounding_candidate import install_candidate

        install_candidate(bundle.model)
    if reference_audio:
        # Explicit diagnostic intervention, NEVER an inference/qualification
        # fallback: distinguish encoder drift from decoder drift.
        base = type(bundle.model.audio_tower)

        class ReferenceAudio(base):
            def __call__(self, features):
                return mx.array(fixture["audio_tower"]).astype(mx.bfloat16)

        bundle.model.audio_tower.__class__ = ReferenceAudio
    if candidate in ("text-math", "all-rounding", "reference-order"):
        from qwen_audio_text_candidate import install_candidate

        install_candidate(bundle.model)
    if candidate in ("compensated", "reference-order"):
        from qwen_audio_text_candidate import install_compensated

        install_compensated(bundle.model)
    if candidate == "layernorm":
        from qwen_audio_text_candidate import install_layernorm

        install_layernorm(bundle.model)
    features, _ = preprocess(mx.array(fixture["waveform"]), offline_padding=True)
    expected_features = fixture["features"][0]
    rounded = np.array(features.astype(mx.bfloat16).astype(mx.float32))
    differences = {
        "features": {
            "different_values": int(np.count_nonzero(rounded != expected_features)),
            "maximum": float(np.max(np.abs(rounded - expected_features))),
        }
    }
    differences["isolated_first_text_layer"] = {}
    for name in fixture.files:
        if not name.startswith("model.layers.0.") or name.endswith(".input"):
            continue
        if name + ".input" not in fixture:
            continue
        module = bundle.model
        if not name.endswith(".act_fn"):
            for part in name.split("."):
                module = module[int(part)] if part.isdigit() else getattr(module, part)
        x = mx.array(fixture[name + ".input"]).astype(mx.bfloat16)
        expected = fixture[name]

        def diff(value, expected=expected):
            actual = np.array(value.astype(mx.float32))
            return {
                "different_values": int(np.count_nonzero(actual != expected)),
                "maximum": float(np.max(np.abs(actual - expected))),
            }

        if name.endswith(".act_fn"):
            import mlx.nn as nn

            row = {"native": diff(nn.silu(x.astype(mx.float32)).astype(mx.bfloat16))}
            value = x.astype(mx.float32)
            row["divide_exp"] = diff((value / (1 + mx.exp(-value))).astype(mx.bfloat16))
        else:
            row = {"native": diff(module(x))}
        if "proj" in name:
            value = x.astype(mx.float32) @ module.weight.astype(mx.float32).T
            if "bias" in module:
                value = value + module.bias.astype(mx.float32)
            row["float32_accumulation"] = diff(value.astype(mx.bfloat16))
        differences["isolated_first_text_layer"][name] = row
    originals = []

    def wrap(layer, name):
        # MLX modules call the class __call__, so observe at encoder/decoder loops
        # using read-only subclasses of the corresponding layer, not monkeypatch.
        base = type(layer)

        class Observed(base):
            def __call__(self, *args, **kwargs):
                result = super().__call__(*args, **kwargs)
                actual = np.array(result.astype(mx.float32))
                expected = fixture[name]
                if actual.ndim == expected.ndim + 1:
                    actual = actual[0]
                differences[name] = {
                    "different_values": int(np.count_nonzero(actual != expected)),
                    "maximum": float(np.max(np.abs(actual - expected))),
                }
                return result

        originals.append((layer, base))
        layer.__class__ = Observed

    for side in ("audio_tower", "model"):
        if reference_audio and side == "audio_tower":
            continue
        for i, layer in enumerate(getattr(bundle.model, side).layers):
            wrap(layer, f"{side}.layers.{i}")
    logits = bundle.model(
        mx.array(fixture["ids"]), features=features, output_positions=mx.array([-1])
    )[0, -1]
    actual = np.array(logits.astype(mx.float32))
    native_id, reference_id = int(actual.argmax()), int(fixture["logits"].argmax())
    differences["logits"] = {
        "native_id": native_id,
        "reference_id": reference_id,
        "native_text": bundle.tokenizer.decode([native_id]),
        "reference_text": bundle.tokenizer.decode([reference_id]),
        "native_top": [(int(i), float(actual[i])) for i in np.argsort(actual)[-5:]],
        "reference_top": [
            (int(i), float(fixture["logits"][i])) for i in np.argsort(fixture["logits"])[-5:]
        ],
    }
    for layer, base in originals:
        layer.__class__ = base
    caches = bundle.model.make_cache()
    logits = bundle.model(
        mx.array(fixture["prompt_ids"]),
        features=features,
        cache=caches,
        output_positions=mx.array([-1]),
    )
    for token in fixture["teacher_ids"]:
        logits = bundle.model(mx.array([[int(token)]]), cache=caches)
    values = np.array(logits[0, -1].astype(mx.float32))
    expected = fixture["cached_logits"]
    differences["cached_logits"] = {
        "native_top": [(int(i), float(values[i])) for i in np.argsort(values)[-5:]],
        "reference_top": [(int(i), float(expected[i])) for i in np.argsort(expected)[-5:]],
        "native_decision": score_decision(values),
        "reference_decision": score_decision(expected),
        "teacher_prefix_ids": fixture["teacher_ids"].tolist(),
    }
    for state in caches:
        state.reset()
    fixture.close()
    root = Path(__file__).resolve().parents[1]
    sources = {
        "model": root / "src/mlx_one/models/audio/qwen3_asr/model.py",
        "probe": Path(__file__),
    }
    if candidate in ("audio-rounding", "all-rounding", "reference-order", "layernorm"):
        sources["audio_candidate"] = root / "benchmarks/qwen_audio_rounding_candidate.py"
    if candidate in ("text-math", "all-rounding", "reference-order", "layernorm", "compensated"):
        sources["text_candidate"] = root / "benchmarks/qwen_audio_text_candidate.py"
    return {
        "scope": "teacher-forced numerical diagnostic, not qualification",
        "model": model,
        "revision": PINS[model],
        "candidate": candidate,
        "reference_source_commit": SOURCE,
        "reference_audio_intervention": reference_audio,
        "fixture_sha256": digest(reference),
        "native_code_sha256": {name: digest(path) for name, path in sources.items()},
        "reports": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["reference", "native"], required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=[m for m in PINS if "ForcedAligner" not in m],
        default="Qwen/Qwen3-ASR-0.6B",
    )
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument(
        "--candidate",
        choices=(
            "default",
            "text-math",
            "audio-rounding",
            "all-rounding",
            "compensated",
            "reference-order",
            "layernorm",
        ),
        default="default",
    )
    parser.add_argument(
        "--reference-audio",
        action="store_true",
        help="Diagnostic intervention only; never qualification.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output must not exist")
    if args.backend == "reference":
        if args.reference.exists():
            parser.error("reference output must not exist")
        if args.candidate != "default" or args.reference_audio:
            parser.error("candidate is native-only")
        if args.audio is None:
            parser.error("reference mode requires --audio")
        report = reference(args.audio, args.reference, args.model, args.prefix)
    else:
        report = native(args.reference, args.candidate, args.reference_audio)
    if args.output is not None:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
