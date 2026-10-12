"""Persist real-checkpoint BF16 intermediates in a separate reference process.

This is a diagnostic, not a qualification gate or production fallback. Inputs to
isolated native operations come from the pinned reference so upstream errors do
not get mistaken for differences in the operation under inspection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from audio_qualification import digest, fixtures
from qwen_audio_qualification import PINS, SOURCE

MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
NAMES = (
    "audio_tower.conv2d1",
    "audio_tower.conv2d2",
    "audio_tower.conv2d3",
    "audio_tower.conv_out",
    "audio_tower.layers.0.self_attn_layer_norm",
    "audio_tower.layers.0.self_attn.q_proj",
    "audio_tower.layers.0.self_attn.k_proj",
    "audio_tower.layers.0.self_attn.v_proj",
    "audio_tower.layers.0.self_attn.out_proj",
    "audio_tower.layers.0.final_layer_norm",
    "audio_tower.layers.0.fc1",
    "audio_tower.layers.0.fc2",
    "audio_tower.ln_post",
    "audio_tower.proj1",
    "audio_tower.proj2",
    "model.layers.0.input_layernorm",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.q_norm",
    "model.layers.0.self_attn.k_norm",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.post_attention_layernorm",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.norm",
)


def validate_reference(metadata: dict, fixture: Path) -> None:
    if (
        metadata.get("model") != MODEL
        or metadata.get("revision") != PINS[MODEL]
        or metadata.get("reference_source_commit") != SOURCE
        or metadata.get("fixture_sha256") != digest(fixture)
    ):
        raise ValueError("reference fixture does not match its frozen manifest")


def reference(args):
    import importlib.metadata
    import subprocess

    import qwen_asr
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ForcedAligner

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(qwen_asr.__file__).resolve().parents[1],
        text=True,
    ).strip()
    if (
        commit != SOURCE
        or importlib.metadata.version("torch") != "2.7.1"
        or (importlib.metadata.version("transformers") != "4.57.6")
    ):
        raise ValueError("reference source/environment differs from frozen contract")
    _, records = fixtures(args.manifest, 100)
    record = next(r for r in records if r["id"] == args.clip)
    waveform, rate = sf.read(args.manifest.parent / record["audio"], dtype="float32")
    if rate != 16000 or waveform.ndim != 1:
        raise ValueError("probe requires checksummed mono 16-kHz input")
    root = snapshot_download(MODEL, revision=PINS[MODEL], local_files_only=True)
    torch.set_num_threads(4)
    bundle = Qwen3ForcedAligner.from_pretrained(
        root,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    # Use exactly the public aligner's processor path, not a recreated template.
    values = {"waveform": waveform}
    handles = []
    thinker = bundle.model.thinker
    for name, module in thinker.named_modules():
        if name not in NAMES:
            continue

        def capture(module, inputs, output, name=name):
            values[name + ".input"] = inputs[0].detach().float().numpy()
            values[name + ".output"] = output.detach().float().numpy()

        handles.append(module.register_forward_hook(capture))

    def capture_inputs(module, inputs, kwargs):
        for name in ("input_ids", "input_features"):
            value = kwargs[name]
            values[name] = (
                value.detach().float().numpy() if value.is_floating_point() else value.numpy()
            )

    handles.append(thinker.register_forward_pre_hook(capture_inputs, with_kwargs=True))
    # Validation-only instrumentation: save the actual post-RoPE operands rather
    # than assume our reconstruction is identical to the reference.
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    attention_calls = {"audio": 0, "text": 0}

    def capture_sdpa(query, key, value, *args, **kwargs):
        side = "audio" if query.shape[1] == key.shape[1] else "text"
        if attention_calls[side] == 0:
            for name, tensor in (("q", query), ("k", key), ("v", value)):
                values[f"{side}_sdpa.{name}"] = tensor.detach().float().numpy()
            mask = kwargs.get("attn_mask", args[0] if args else None)
            if mask is not None:
                values[f"{side}_sdpa.mask"] = mask.detach().float().numpy()
        attention_calls[side] += 1
        return original_sdpa(query, key, value, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = capture_sdpa
    try:
        with torch.inference_mode():
            result = bundle.align((waveform, rate), record["reference"], language="English")
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa
        for handle in handles:
            handle.remove()
    missing = set(NAMES) - {key.removesuffix(".input") for key in values if key.endswith(".input")}
    if missing:
        raise ValueError(f"reference hooks missing: {sorted(missing)}")
    with args.fixture.open("xb") as stream:
        np.savez(stream, **values)
    return {
        "scope": "BF16 real-checkpoint layer diagnostic, not qualification",
        "model": MODEL,
        "revision": PINS[MODEL],
        "reference_source_commit": SOURCE,
        "manifest_sha256": digest(args.manifest),
        "fixture_sha256": digest(args.fixture),
        "clip": args.clip,
        "alignment": str(result),
    }


def native(args):
    metadata = json.loads(args.reference.read_text())
    validate_reference(metadata, args.fixture)
    import mlx.core as mx
    from qwen_audio_rounding_candidate import candidate_attention, install_candidate

    from mlx_one.audio import load_audio_model
    from mlx_one.models.audio.qwen3_asr.config import audio_token_length
    from mlx_one.models.audio.qwen3_asr.model import attention, gelu, rotary
    from mlx_one.models.audio.qwen3_asr.processing import preprocess

    bundle = load_audio_model(MODEL, task="align", revision=PINS[MODEL], offline=True)
    if args.numerical_candidate:
        install_candidate(bundle.model)
    reports = {}

    def difference(actual, expected):
        if actual.shape != expected.shape:
            raise ValueError(f"intermediate shape mismatch: {actual.shape} != {expected.shape}")
        return {
            "shape": list(actual.shape),
            "different_values": int(np.count_nonzero(actual != expected)),
            "maximum_absolute_difference": float(np.max(np.abs(actual - expected))),
            "mean_absolute_difference": float(np.mean(np.abs(actual - expected))),
        }

    with np.load(args.fixture, allow_pickle=False) as values:
        features, _ = preprocess(mx.array(values["waveform"]), offline_padding=False)
        expected_features = values["input_features"][0]
        reports["frontend"] = difference(
            np.array(features.astype(mx.bfloat16).astype(mx.float32)),
            expected_features,
        )
        for name in NAMES:
            module = bundle.model
            for part in name.split("."):
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            x = mx.array(values[name + ".input"]).astype(mx.bfloat16)
            conv = ".conv2d" in name
            if conv:
                x = x.transpose(0, 2, 3, 1)
            output = module(x)
            if conv:
                output = output.transpose(0, 3, 1, 2)
            reports[name] = difference(
                np.array(output.astype(mx.float32)), values[name + ".output"]
            )
            variants = {}
            if conv:
                value = mx.conv2d(
                    x.astype(mx.float32),
                    module.weight.astype(mx.float32),
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                )
                variants["float32_bias_then_round"] = (
                    (value + module.bias.astype(mx.float32))
                    .astype(mx.bfloat16)
                    .transpose(0, 3, 1, 2)
                )
            elif "layer_norm" in name or name == "audio_tower.ln_post":
                value = x.astype(mx.float32)
                variants["float32_affine_then_round"] = mx.fast.layer_norm(
                    value,
                    module.weight.astype(mx.float32),
                    module.bias.astype(mx.float32),
                    module.eps,
                ).astype(mx.bfloat16)
                normalized = mx.fast.layer_norm(value, None, None, module.eps)
                variants["round_normalized_before_affine"] = (
                    normalized.astype(mx.bfloat16) * module.weight + module.bias
                )
            if variants:
                reports[name]["diagnostic_variants"] = {
                    key: difference(np.array(value.astype(mx.float32)), values[name + ".output"])
                    for key, value in variants.items()
                }
        for source, target in (
            ("audio_tower.conv2d1", "audio_tower.conv2d2"),
            ("audio_tower.conv2d2", "audio_tower.conv2d3"),
            ("audio_tower.layers.0.fc1", "audio_tower.layers.0.fc2"),
            ("audio_tower.proj1", "audio_tower.proj2"),
        ):
            actual = gelu(mx.array(values[source + ".output"]).astype(mx.bfloat16))
            reports["gelu-" + source] = difference(
                np.array(actual.astype(mx.float32)), values[target + ".input"]
            )
        projected = mx.array(values["audio_tower.conv_out.output"]).astype(mx.bfloat16)
        half = projected.shape[-1] // 2
        inverse = mx.exp(-math.log(10000.0) / (half - 1) * mx.arange(half, dtype=mx.float32))
        phases = mx.arange(projected.shape[1], dtype=mx.float32)[:, None] * inverse
        positions = mx.concatenate((mx.sin(phases), mx.cos(phases)), axis=-1)
        positioned = projected + positions.astype(mx.bfloat16)[None]
        frame_count = expected_features.shape[1]
        lengths = [
            audio_token_length(min(100, frame_count - i)) for i in range(0, frame_count, 100)
        ]
        actual = mx.concatenate([positioned[i, :n] for i, n in enumerate(lengths)])
        reports["audio-position-addition"] = difference(
            np.array(actual.astype(mx.float32)),
            values["audio_tower.layers.0.self_attn_layer_norm.input"],
        )
        if "text_sdpa.q" in values:
            q, k, v = [
                mx.array(values["text_sdpa." + name]).astype(mx.bfloat16)
                for name in ("q", "k", "v")
            ]
            length = q.shape[2]
            mask = mx.where(
                mx.arange(length)[:, None] >= mx.arange(k.shape[2])[None, :],
                0.0,
                -float("inf"),
            )
            scale = q.shape[-1] ** -0.5
            q32 = q.astype(mx.float32)
            k32, v32 = [
                mx.repeat(x, q.shape[1] // x.shape[1], axis=1).astype(mx.float32) for x in (k, v)
            ]
            scores = (q32 * scale**0.5) @ (k32 * scale**0.5).swapaxes(-1, -2)
            variants = {
                "native_fast": attention(q, k, v, scale=scale, mask=mask),
                "split_scale_math": mx.softmax(scores + mask, axis=-1) @ v32,
            }
            expected = values["model.layers.0.self_attn.o_proj.input"]
            for name, value in variants.items():
                actual = value.astype(mx.bfloat16).transpose(0, 2, 1, 3).reshape(expected.shape)
                reports["text-attention-" + name] = difference(
                    np.array(actual.astype(mx.float32)), expected
                )
            for name in ("q", "k"):
                raw = mx.array(values[f"model.layers.0.self_attn.{name}_norm.output"])
                raw = raw.astype(mx.bfloat16).transpose(0, 2, 1, 3)
                actual = rotary(raw, 0, bundle.model.config.text_config.rope_theta)
                reports["text-rope-" + name] = difference(
                    np.array(actual.astype(mx.float32)), values["text_sdpa." + name]
                )
        # Compare the end-to-end encoder with native and reference frontend
        # values independently; neither is substituted into production inference.
        encoded = values["audio_tower.proj2.output"]
        for label, feature_values in (
            ("native_features", features),
            ("reference_features", mx.array(expected_features).astype(mx.bfloat16)),
        ):
            actual = bundle.model.audio_tower(feature_values)
            reports["encoder-" + label] = difference(
                np.array(actual.astype(mx.float32)),
                encoded,
            )
        layer = bundle.model.audio_tower.layers[0].self_attn
        heads, dims = layer._heads, layer._head_dim
        base = "audio_tower.layers.0.self_attn."
        q, k, v = [
            mx.array(values[base + projection + ".output"])
            .astype(mx.bfloat16)
            .reshape(-1, heads, dims)
            .transpose(1, 0, 2)[None]
            for projection in ("q_proj", "k_proj", "v_proj")
        ]
        expected = values[base + "out_proj.input"]
        q32, k32, v32 = [x.astype(mx.float32) for x in (q, k, v)]
        scores = (q32 @ k32.swapaxes(-1, -2)) * dims**-0.5
        probabilities = mx.softmax(scores, axis=-1)
        attention_variants = {
            "native_reference_rounding": candidate_attention(
                q, k, v, scale=dims**-0.5, cpu_flash=True
            ),
            "float32_fast": mx.fast.scaled_dot_product_attention(q32, k32, v32, scale=dims**-0.5),
            "float32_explicit": probabilities @ v32,
            "bf16_probabilities": probabilities.astype(mx.bfloat16).astype(mx.float32) @ v32,
        }
        for name, value in attention_variants.items():
            actual = value.astype(mx.bfloat16).transpose(0, 2, 1, 3).reshape(-1, heads * dims)
            reports["attention-" + name] = difference(np.array(actual.astype(mx.float32)), expected)
    return {
        "scope": metadata["scope"],
        "numerical_candidate": bool(args.numerical_candidate),
        "reference_sha256": digest(args.reference),
        "fixture_sha256": digest(args.fixture),
        "reports": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "native"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--clip", default="1089-134686-0005")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--numerical-candidate",
        action="store_true",
        help="Opt-in failed rounding candidate; never changes production defaults.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.backend == "reference" and args.fixture.exists()):
        parser.error("output targets must not exist")
    if args.backend == "reference" and args.numerical_candidate:
        parser.error("numerical candidate is native-only, never the reference")
    if args.backend == "reference" and args.manifest is None:
        parser.error("reference mode requires --manifest")
    if args.backend == "native" and args.reference is None:
        parser.error("native mode requires --reference")
    result = reference(args) if args.backend == "reference" else native(args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
