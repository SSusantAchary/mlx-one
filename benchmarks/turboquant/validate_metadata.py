#!/usr/bin/env python3
"""Validate a local TQ3 checkpoint or saved metadata headers without loading weights."""

import argparse
import json
from pathlib import Path

from mlx_one.models.language.gpt_oss.config import GptOssConfig
from mlx_one.models.language.gpt_oss.weights import weight_contract
from mlx_one.quantization.turboquant.config import TQ3Config
from mlx_one.text.loading import _quantization_config


def validate(root):
    data = json.loads((root / "config.json").read_text())
    config = GptOssConfig.from_dict(data)
    TQ3Config.from_dict(_quantization_config(data))
    headers = {}
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else None
    files = sorted(set(index.values())) if index else ["model.safetensors"]
    for name in files:
        if Path(name).name != name:
            raise ValueError("unsafe safetensors shard name")
        saved = root / (name + ".header.json")
        if saved.exists():
            header = json.loads(saved.read_text())
        else:
            with (root / name).open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                if not 0 < size <= 16 * 1024**2:
                    raise ValueError("invalid safetensors header size")
                header = json.loads(stream.read(size))
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in headers or (index and index.get(key) != name):
                raise ValueError(f"duplicate or misindexed tensor: {key}")
            headers[key] = value
    if index and set(index) != set(headers):
        raise ValueError("header/index tensor names differ")
    expected = dict(weight_contract(config).expected)
    packed = []
    for key in headers:
        if key.endswith(".codebook"):
            prefix = key[:-9]
            shape = expected.get(prefix + ".weight")
            if shape is None or len(shape) not in (2, 3) or shape[-1] % 64:
                raise ValueError(f"unknown/incompatible TQ3 module: {prefix}")
            k = shape[-1]
            expected[prefix + ".weight"] = (*shape[:-1], (k + 9) // 10)
            expected[prefix + ".scales"] = (*shape[:-1], k // 64)
            expected[prefix + ".codebook"] = (8,)
            expected[prefix + ".signs"] = (k,)
            for suffix, dtype in [
                ("weight", "U32"),
                ("scales", "F16"),
                ("codebook", "F16"),
                ("signs", "F16"),
            ]:
                if headers.get(prefix + "." + suffix, {}).get("dtype") != dtype:
                    raise ValueError(f"{prefix}.{suffix}: wrong or missing dtype")
            packed.append(prefix)
    if not packed or set(expected) != set(headers):
        raise ValueError("missing/unexpected TQ3 or floating tensors")
    for key, shape in expected.items():
        if list(shape) != headers[key]["shape"]:
            raise ValueError(f"{key}: malformed shape {headers[key]['shape']}, expected {shape}")
        if headers[key]["dtype"] not in {"U32", "F16", "BF16", "F32"}:
            raise ValueError(f"{key}: unsupported dtype")
    return dict(
        status="METADATA_VALIDATED",
        format="turboquant_tq3",
        tensor_count=len(headers),
        packed_modules=len(packed),
        checkpoint_status="NOT_SMOKE_TESTED",
        weight_payload_read=False,
        model_type=config.model_type,
        layers=config.num_hidden_layers,
        experts=config.num_local_experts,
        active_experts=config.num_experts_per_tok,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.checkpoint)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
