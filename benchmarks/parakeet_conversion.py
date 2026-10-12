"""Checksum-locked Parakeet v2 conversion; run only in an isolated reference env.

Archive extraction is deliberately separate from the upstream conversion helper.
Production loaders do not import this module or restore NeMo/pickle checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

MODEL = "nvidia/parakeet-tdt-0.6b-v2"
REVISION = "ae9ad07059c7c739ffaf932226a8fe64ae2620b0"
ARCHIVE_SHA256 = "d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c"
SOURCE = "469230357aab0f2b303b0d638c1f8d06edb14184"
MAX_BYTES = 4 * 1024**3


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inspect_archive(archive: tarfile.TarFile, *, max_bytes: int = MAX_BYTES) -> dict:
    """Validate the whole inventory before opening or extracting any member."""
    inventory = {}
    total = 0
    for member in archive:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in member.name
            or ":" in member.name
            or member.issym()
            or member.islnk()
            or not (member.isfile() or member.isdir())
            or member.sparse is not None
        ):
            raise ValueError(f"unsafe NeMo archive entry: {member.name}")
        normalized = path.as_posix()
        if normalized in inventory:
            raise ValueError(f"duplicate NeMo archive entry: {normalized}")
        if member.size < 0 or member.size > max_bytes:
            raise ValueError("NeMo member exceeds extraction limit")
        total += member.size
        if total > max_bytes or len(inventory) >= 4096:
            raise ValueError("NeMo archive exceeds extraction limits")
        inventory[normalized] = member
    return inventory


def convert(archive_path: Path, source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("conversion output must not exist")
    if digest(archive_path) != ARCHIVE_SHA256:
        raise ValueError("archive differs from the pinned Parakeet v2 checksum")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != SOURCE:
        raise ValueError("reference checkout differs from frozen conversion source")
    # Refuse modified source files; never trust checkpoint-supplied conversion code.
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD"], cwd=source, check=True, stdout=subprocess.DEVNULL
    )
    if importlib.metadata.version("torch") != "2.7.1":
        raise ValueError("conversion requires isolated PyTorch 2.7.1")

    import torch
    import yaml
    from safetensors.torch import save_file

    from mlx_one.models.audio.parakeet.config import ParakeetConfig
    from mlx_one.models.audio.parakeet.weights import weight_contract

    helper_path = source / "src/transformers/models/parakeet/convert_nemo_to_hf.py"
    spec = importlib.util.spec_from_file_location("pinned_parakeet_converter", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        inventory = inspect_archive(archive)
        model_files = {}
        targets = {
            "model_config.yaml": "model_config",
            "model_weights.ckpt": "model_weights",
            "tokenizer.model": "tokenizer_model_file",
        }
        with tempfile.TemporaryDirectory(prefix="parakeet-conversion-", dir=output.parent) as temp:
            root = Path(temp)
            for basename, key in targets.items():
                matches = [
                    member
                    for name, member in inventory.items()
                    if (
                        PurePosixPath(name).name == basename
                        or (basename == "tokenizer.model" and name.endswith("_tokenizer.model"))
                    )
                    and member.isfile()
                ]
                if len(matches) != 1:
                    raise ValueError(f"archive must contain exactly one {basename}")
                path = root / basename
                with archive.extractfile(matches[0]) as inp, path.open("xb") as out:
                    shutil.copyfileobj(inp, out)
                model_files[key] = str(path)
            with (root / "model_config.yaml").open() as stream:
                nemo = yaml.safe_load(stream)
            encoder = helper.convert_encoder_config(nemo)
            # Original v2 has no dedicated pad token, unlike v3. Preserve its
            # 1,024 lexical IDs and append only the model's blank at ID 1,024.
            prednet = nemo["decoder"]["prednet"]
            blank = len(nemo["labels"])
            config_data = {
                "model_type": "parakeet_tdt",
                "encoder_config": encoder.to_dict(),
                "vocab_size": blank + 1,
                "blank_token_id": blank,
                "pad_token_id": blank,
                "decoder_hidden_size": prednet["pred_hidden"],
                "num_decoder_layers": prednet["pred_rnn_layers"],
                "durations": nemo["decoding"]["durations"],
                "max_symbols_per_step": nemo["decoding"]["greedy"]["max_symbols"],
                "hidden_act": "relu",
                "dtype": "float32",
                "mlx_one_checkpoint_variant": "v2",
            }
            config = ParakeetConfig.from_dict(config_data)
            state = torch.load(model_files["model_weights"], map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or any(
                not isinstance(v, torch.Tensor) for v in state.values()
            ):
                raise ValueError("NeMo weights must be a tensor-only state dictionary")
            weights = {}
            frontend = {}
            mapping = {**helper.NEMO_TO_HF_WEIGHT_MAPPING, **helper.NEMO_TDT_WEIGHT_MAPPING}
            for name, tensor in state.items():
                if name.endswith(("featurizer.window", "featurizer.fb")):
                    frontend[name.rsplit(".", 1)[-1]] = tensor.contiguous()
                    continue
                translated = helper.convert_key(name, mapping)
                if translated in weights:
                    raise ValueError(f"duplicate converted tensor: {translated}")
                weights[translated] = tensor.contiguous()
            # The original predictor represents blank as zero input, not an embedding.
            name = "decoder.embedding.weight"
            if weights[name].shape[0] == config.blank_token_id:
                weights[name] = torch.cat(
                    (weights[name], torch.zeros_like(weights[name][:1])), dim=0
                )
            weight_contract(config, native=False).validate(weights)
            for name, tensor in weights.items():
                expected_dtype = (
                    torch.int64 if name.endswith("num_batches_tracked") else torch.float32
                )
                if tensor.dtype != expected_dtype:
                    raise ValueError(f"unsupported converted precision: {name}={tensor.dtype}")
            exported = root / "export"
            exported.mkdir()
            from tokenizers import AddedToken

            tokenizer = helper.ParakeetConverter(model_files["tokenizer_model_file"]).converted()
            tokenizer.add_special_tokens([AddedToken("<blank>", normalized=False, special=True)])
            if tokenizer.token_to_id("<blank>") != blank:
                raise ValueError("converted tokenizer blank ID differs from checkpoint")
            tokenizer.save(str(exported / "tokenizer.json"))
            (exported / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "ParakeetTokenizer",
                        "pad_token": "<blank>",
                        "unk_token": "<unk>",
                        "clean_up_tokenization_spaces": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            pre = nemo["preprocessor"]
            processor = {
                "processor_class": "ParakeetProcessor",
                "blank_token": "<blank>",
                "decoder_type": "tdt",
                "feature_extractor": {
                    "feature_extractor_type": "ParakeetFeatureExtractor",
                    "feature_size": pre["features"],
                    "sampling_rate": pre["sample_rate"],
                    "n_fft": pre["n_fft"],
                    "hop_length": int(pre["window_stride"] * pre["sample_rate"]),
                    "win_length": int(pre["window_size"] * pre["sample_rate"]),
                    "preemphasis": 0.97,
                    "padding_side": "right",
                    "padding_value": 0.0,
                    "return_attention_mask": True,
                },
            }
            (exported / "processor_config.json").write_text(json.dumps(processor, indent=2) + "\n")
            shutil.copyfile(root / "model_config.yaml", exported / "original-model-config.yaml")
            save_file(weights, str(exported / "model.safetensors"))
            if frontend:
                save_file(frontend, str(exported / "frontend.safetensors"))
            config_data["mlx_one_checkpoint_variant"] = "v2"
            (exported / "config.json").write_text(json.dumps(config_data, indent=2) + "\n")
            generation = {
                "decoder_start_token_id": config.blank_token_id,
                "pad_token_id": config.pad_token_id,
                "suppress_tokens": list(
                    range(config.vocab_size, config.vocab_size + len(config.durations))
                ),
            }
            (exported / "generation_config.json").write_text(
                json.dumps(generation, indent=2) + "\n"
            )
            provenance = {
                "model": MODEL,
                "revision": REVISION,
                "archive_sha256": ARCHIVE_SHA256,
                "reference_commit": SOURCE,
                "reference_source": "https://github.com/huggingface/transformers",
                "checkpoint_license": "cc-by-4.0",
                "weight_layout": "canonical",
                "tensor_count": len(weights),
                "assets": {p.name: digest(p) for p in sorted(exported.iterdir()) if p.is_file()},
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "safetensors",
                        "tokenizers",
                        "sentencepiece",
                        "pyyaml",
                    )
                },
                "neural_fallback": False,
            }
            (exported / "conversion-provenance.json").write_text(
                json.dumps(provenance, indent=2) + "\n"
            )
            # Exclusive mkdir also refuses a concurrently published result.
            output.mkdir()
            for file in sorted(
                exported.iterdir(), key=lambda p: p.name == "conversion-provenance.json"
            ):
                shutil.move(str(file), output / file.name)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = convert(args.archive, args.reference_source, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "revision": report["revision"],
                "tensors": report["tensor_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
