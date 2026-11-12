"""Offline, checksummed safe conversion of publisher-only PyTorch checkpoints.

Torch is required only in this development environment. Native inference reads
the resulting safetensors and never imports Torch or executes remote model code.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def convert_bge(root, output):
    tensors = {}
    for file, prefix in (
        ("pytorch_model.bin", ""),
        ("sparse_linear.pt", "sparse_linear."),
        ("colbert_linear.pt", "colbert_linear."),
    ):
        state = torch.load(root / file, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(
            not isinstance(v, torch.Tensor) for v in state.values()
        ):
            raise ValueError("BGE conversion requires a flat tensor state dict")
        for name, value in state.items():
            key = prefix + name
            if key in tensors:
                raise ValueError("duplicate converted tensor")
            tensors[key] = value.contiguous()
    config = json.loads((root / "config.json").read_text())
    config.update(multi_representation=True, colbert_dim=tensors["colbert_linear.weight"].shape[0])
    for path in root.iterdir():
        if path.is_file() and path.suffix in {".json", ".txt", ".model"}:
            shutil.copyfile(path, output / path.name)
    if (root / "1_Pooling").exists():
        shutil.copytree(root / "1_Pooling", output / "1_Pooling")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(tensors, output / "model.safetensors")
    return {
        name: dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        )
        for name, value in tensors.items()
    }


def convert_kokoro(root, output, shapes):
    """Fold declared weight norms, map recurrent gates, and transpose convolution layouts."""
    groups = torch.load(root / "kokoro-v1_0.pth", map_location="cpu", weights_only=True)
    flat = {}
    for group, state in groups.items():
        if not isinstance(state, dict):
            raise ValueError("Kokoro requires named tensor state dictionaries")
        for key, value in state.items():
            name = group + "." + key.removeprefix("module.")
            if not isinstance(value, torch.Tensor) or name in flat:
                raise ValueError("invalid or duplicated Kokoro source tensor")
            flat[name] = value
    folded = {}
    consumed = set()
    for name, value in flat.items():
        if name in consumed:
            continue
        if name.endswith("weight_v") or name.endswith("parametrizations.weight.original1"):
            continue
        if name.endswith("weight_g"):
            target = name.removesuffix("_g")
            vector = flat[target + "_v"]
            denominator = torch.linalg.vector_norm(
                vector, dim=tuple(range(1, vector.ndim)), keepdim=True
            )
            folded[target] = vector * (value / denominator)
            consumed.update((name, target + "_v"))
        elif name.endswith("parametrizations.weight.original0"):
            target = name.removesuffix("parametrizations.weight.original0") + "weight"
            vname = name.removesuffix("original0") + "original1"
            vector = flat[vname]
            folded[target] = vector * (
                value
                / torch.linalg.vector_norm(vector, dim=tuple(range(1, vector.ndim)), keepdim=True)
            )
            consumed.update((name, vname))
        else:
            folded[name] = value
            consumed.add(name)
    if consumed != set(flat):
        raise ValueError("unconsumed Kokoro source tensors")
    tensors = {}
    for name, value in folded.items():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in {"bias_hh_l0", "bias_hh_l0_reverse"}:
            continue
        if suffix.startswith(("weight_ih_l0", "weight_hh_l0", "bias_ih_l0")):
            base = name.rsplit(".", 1)[0]
            direction = "backward_lstm" if suffix.endswith("_reverse") else "forward_lstm"
            if suffix.startswith("bias_ih"):
                partner = name.replace("bias_ih_l0", "bias_hh_l0")
                value = value + folded[partner]
                tail = "bias"
            else:
                tail = "Wx" if suffix.startswith("weight_ih") else "Wh"
            name = f"{base}.{direction}.{tail}"
        elif name.endswith(".weight") and value.ndim == 3:
            value = value.permute(1, 2, 0) if ".ups." in name else value.permute(0, 2, 1)
        if name in tensors:
            raise ValueError("duplicate native Kokoro tensor")
        tensors[name] = value.contiguous()
    constants = []
    for name, shape in shapes.items():
        if name not in tensors and ".norm." in name and name.endswith((".weight", ".bias")):
            tensors[name] = torch.ones(shape) if name.endswith(".weight") else torch.zeros(shape)
            constants.append(name)
    if set(tensors) != set(shapes) or any(
        list(value.shape) != shapes[name] for name, value in tensors.items()
    ):
        missing = sorted(set(shapes) - set(tensors))
        extra = sorted(set(tensors) - set(shapes))
        mismatch = {
            k: [list(v.shape), shapes.get(k)]
            for k, v in tensors.items()
            if k in shapes and list(v.shape) != shapes[k]
        }
        raise ValueError(
            f"Kokoro contract mismatch: missing={missing}; extra={extra}; shapes={mismatch}"
        )
    save_file(tensors, output / "model.safetensors")
    voice = torch.load(root / "voices/af_heart.pt", map_location="cpu", weights_only=True)
    if not isinstance(voice, torch.Tensor) or tuple(voice.shape) != (510, 1, 256):
        raise ValueError("unexpected publisher af_heart voice shape")
    (output / "voices").mkdir()
    save_file({"voice": voice.contiguous()}, output / "voices/af_heart.safetensors")
    config = json.loads((root / "config.json").read_text())
    config.update(model_type="kokoro", native_conversion="mlx-one-kokoro-v1", voice="af_heart")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {
        name: dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            origin="fixed identity affine for publisher non-affine instance normalization"
            if name in constants
            else "publisher tensor/conversion",
        )
        for name, value in tensors.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["bge-m3", "kokoro"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shape-contract", type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    inputs = {}
    sources = (
        ("pytorch_model.bin", "sparse_linear.pt", "colbert_linear.pt")
        if args.kind == "bge-m3"
        else ("kokoro-v1_0.pth", "voices/af_heart.pt")
    )
    for name in sources:
        expected = inventory["files"][name]["lfs"]["sha256"]
        actual = sha(args.source / name)
        if actual != expected:
            raise ValueError("pinned publisher input checksum mismatch: " + name)
        inputs[name] = actual
    if args.output.exists() or args.report.exists():
        raise ValueError("refuse to overwrite converted assets or a prior conversion report")
    args.output.mkdir()
    if args.kind == "bge-m3":
        tensors = convert_bge(args.source, args.output)
    else:
        if args.shape_contract is None:
            raise ValueError("Kokoro conversion requires the independent native shape contract")
        tensors = convert_kokoro(
            args.source, args.output, json.loads(args.shape_contract.read_text())
        )
    report = dict(
        kind=args.kind,
        revision=inventory["revision"],
        source_sha256=inputs,
        torch=torch.__version__,
        tensors=tensors,
        converted_sha256=sha(args.output / "model.safetensors"),
        converted_tensor_elements=sum(
            int(torch.tensor(v["shape"]).prod()) for v in tensors.values()
        ),
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(converted_tensors=len(tensors), sha256=report["converted_sha256"])))


if __name__ == "__main__":
    main()
