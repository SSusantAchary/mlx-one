"""Check native OCR parameter contracts against saved immutable safetensors headers.

This is structural validation, not numerical qualification or a real checkpoint load.
No network access or model weight download/evaluation occurs.
"""

import hashlib
import json
from pathlib import Path


def audit():
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx_one.ocr.loading import make_config, make_model
    from mlx_one.text.loading import (
        _prepare_quantized_model,
        _quantization_config,
        _validate_quantized_tensors,
    )

    root = Path("docs/benchmarks/ocr")
    inventory = json.loads((root / "inventory.json").read_text())["checkpoints"]
    rows = []
    dtypes = {
        "F32": mx.float32,
        "F16": mx.float16,
        "BF16": mx.bfloat16,
        "U32": mx.uint32,
        "I64": mx.int64,
    }
    for item in inventory:
        folder = root / "metadata" / item["id"]
        provenance = json.loads((folder / "provenance.json").read_text())
        raw = (folder / "tensor-header.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != provenance["tensor_header_sha256"]:
            raise ValueError("saved tensor header hash mismatch")
        payload = json.loads((folder / "config.json").read_text())
        native = make_model(make_config(payload, item["family"]), item["family"])
        header = json.loads(raw)
        tensors = {
            name: mx.zeros(meta["shape"], dtype=dtypes[meta["dtype"]])
            for name, meta in header.items()
            if name != "__metadata__"
        }
        tensors = native.sanitize(tensors)
        expected = dict(tree_flatten(native.parameters()))
        for name, tensor in list(tensors.items()):
            target = expected.get(name)
            if tensor.ndim == 4 and target is not None and tensor.shape != target.shape:
                converted = tensor.transpose(0, 2, 3, 1)
                if converted.shape == target.shape:
                    tensors[name] = converted
        if item["family"] == "deepseek_ocr":
            tensors = native.language_model.sanitize(tensors)
        quant = _quantization_config(payload)
        if quant:
            _prepare_quantized_model(native, tensors, quant)
        _validate_quantized_tensors(native, tensors)
        rows.append(
            {
                "checkpoint": item["checkpoint"],
                "revision": item["revision"],
                "status": "NATIVE_PARAMETER_HEADER_MATCH",
                "full_checkpoint_executed": False,
                "tensor_count": len(tensors),
            }
        )
    return rows


if __name__ == "__main__":
    rows = audit()
    print(json.dumps(rows, indent=2))
    print(f"Validated native tensor names, shapes and formats for {len(rows)} checkpoint headers.")
