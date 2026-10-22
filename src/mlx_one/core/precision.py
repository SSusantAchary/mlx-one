"""Describe checkpoint precision without casting weights or granting qualification.

ModelSpec and its 1.0 serialization remain authoritative. This versioned view
separates storage metadata from explicitly observed compute/cache dtypes.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlx_one.schemas import ModelSpec

PRECISION_CONTRACT_VERSION = "checkpoint-precision-v1"
_FLOATS = {
    "fp32": "fp32",
    "float32": "fp32",
    "fp16": "fp16",
    "float16": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
}


def describe_precision(
    model: ModelSpec,
    *,
    activation_dtype: str | None = None,
    kv_cache_dtype: str | None = None,
    recurrent_state_dtype: str | None = None,
    accumulation_dtype: str | None = None,
) -> dict:
    """Return a non-authoritative, exact-format view of existing model metadata.

    INT4/INT8 planning labels alone do not establish MLX affine storage. Group
    sizes describe a checkpoint; they do not certify loader/hardware support.
    None means unobserved, never an inferred FP16 compute or cache claim.
    """
    observed = {}
    for name, dtype in (
        ("activation", activation_dtype),
        ("kv_cache", kv_cache_dtype),
        ("recurrent_state", recurrent_state_dtype),
        ("attention_accumulation", accumulation_dtype),
    ):
        if dtype is not None and dtype not in {*_FLOATS, "mixed"}:
            raise ValueError(f"unsupported {name} dtype: {dtype!r}")
        observed[name] = {"dtype": _FLOATS.get(dtype, dtype)}
    quantization = dict(model.quantization)
    weight = {
        "kind": _FLOATS.get(model.dtype, "mixed" if model.dtype == "mixed" else None),
        "bits": None,
        "group_size": None,
        "quantization_mode": None,
        "metadata": quantization,
    }
    if quantization.get("mode") == "turboquant":
        from mlx_one.quantization.turboquant.config import TQ3Config

        TQ3Config.from_dict(quantization)
        weight.update(
            kind="turboquant_tq3",
            bits=3,
            group_size=64,
            quantization_mode="turboquant",
            label="TQ3 / TurboQuant 3-bit",
        )
    elif quantization:
        bits = quantization.get("bits")
        group = quantization.get("group_size", 64)
        mode = quantization.get("mode", "affine")
        method = quantization.get("quant_method", quantization.get("method", "mlx"))
        if type(bits) is not int or bits not in {4, 8}:
            raise ValueError("MLX affine precision requires explicit 4 or 8 bits")
        if type(group) is not int or group < 1:
            raise ValueError("MLX affine group_size must be a positive integer")
        if mode != "affine" or method not in {"mlx", "affine"}:
            raise ValueError("alpha weight targets require MLX affine quantization")
        # Preserve exact per-layer exceptions rather than collapsing them into
        # a homogeneous Q4/Q8 support claim.
        mixed = any(
            key not in {"bits", "group_size", "mode", "quant_method", "method"}
            for key in quantization
        )
        weight.update(
            kind="mixed" if mixed else f"mlx_affine_q{bits}",
            bits=bits,
            group_size=group,
            quantization_mode="mlx_affine",
        )
    if weight["kind"] is None:
        raise ValueError("floating weight precision requires an explicit floating dtype")
    return {
        "schema_version": PRECISION_CONTRACT_VERSION,
        "weight": weight,
        **observed,
        "checkpoint": {
            "model_id": model.model_id,
            "revision": model.revision,
            "format": model.weight_format,
        },
        "qualification": "NOT_QUALIFIED",
    }


def validate_finite_safetensors(path: str | Path) -> None:
    """Reject non-finite floating payloads on CPU before native model allocation.

    Safetensors validates offsets/shapes first. Scan at most 8 MiB per read;
    packed integers are untouched and BF16 is checked in its stored bit format.
    No conversion, MLX import, pickle or checkpoint execution is involved.
    """
    import numpy as np
    from safetensors import SafetensorError, safe_open

    path = Path(path)
    try:
        with safe_open(path, framework="numpy"):
            pass
    except SafetensorError as exc:
        raise ValueError(f"invalid safetensors checkpoint: {path.name}") from exc
    formats = {"F16": ("<f2", 2), "F32": ("<f4", 4), "F64": ("<f8", 8), "BF16": ("<u2", 2)}
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        if header_size > 16 * 1024**2:
            raise ValueError("safetensors header exceeds bounded validation limit")
        header = json.loads(stream.read(header_size))
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            dtype = metadata["dtype"]
            if dtype not in formats:
                if dtype.startswith("F"):
                    raise ValueError(f"unsupported floating tensor dtype: {dtype}")
                continue
            numpy_dtype, width = formats[dtype]
            start, end = metadata["data_offsets"]
            if (end - start) % width:
                raise ValueError(f"invalid floating tensor byte count: {name}")
            stream.seek(8 + header_size + start)
            remaining = end - start
            while remaining:
                chunk = stream.read(min(remaining, 8 * 1024**2))
                if not chunk:
                    raise ValueError(f"truncated floating tensor: {name}")
                values = np.frombuffer(chunk, dtype=numpy_dtype)
                finite = (
                    np.all((values & 0x7F80) != 0x7F80)
                    if dtype == "BF16"
                    else np.isfinite(values).all()
                )
                if not finite:
                    raise ValueError(f"non-finite checkpoint tensor: {name}")
                remaining -= len(chunk)
