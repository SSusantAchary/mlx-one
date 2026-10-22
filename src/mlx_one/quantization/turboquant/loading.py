"""Strict conversion of native projection modules to original packed storage."""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_one.core.quantization import UnsupportedQuantization, quantization_spec
from mlx_one.models.shared.sparse import ExpertLinear

from .config import TQ3Config
from .experts import TQ3ExpertLinear
from .linear import TQ3Linear


def prepare_tq3_model(model, tensors, metadata):
    config = TQ3Config.from_dict(metadata)
    modules = dict(model.named_modules())
    prefixes = {k.rsplit(".", 1)[0] for k in tensors if k.endswith(".codebook")}
    scales = {k.rsplit(".", 1)[0] for k in tensors if k.endswith(".scales")}
    if not prefixes:
        raise ValueError("TQ3 checkpoint has no codebook tensors")
    overrides = dict(config.overrides)
    for path, spec in overrides.items():
        if path not in modules or not hasattr(modules[path], "to_quantized"):
            raise ValueError(f"unknown TQ3 override module: {path}")
        if spec is not False and path not in scales:
            raise ValueError(f"missing quantization scale tensors for override: {path}")
    updates = []
    checks = []
    for path in sorted(prefixes):
        source = modules.get(path)
        if not isinstance(source, (nn.Linear, ExpertLinear)):
            raise ValueError(f"TQ3 packed tensors on unsupported module: {path}")
        if path in overrides:
            raise ValueError(f"TQ3 codebook conflicts with override: {path}")
        dims = source.weight.shape
        inputs, outputs = dims[-1], dims[-2]
        if inputs % 64:
            raise ValueError(f"{path}: TQ3 input dimension must divide group_size=64")
        expected = {
            "weight": (*dims[:-1], (inputs + 9) // 10),
            "scales": (*dims[:-1], inputs // 64),
            "codebook": (8,),
            "signs": (inputs,),
        }
        if "bias" in source:
            expected["bias"] = dims[:-1]
        params = {}
        for name, shape in expected.items():
            key = f"{path}.{name}"
            value = tensors.get(key)
            if value is None or value.shape != shape:
                raise ValueError(f"{key}: missing or malformed TQ3 storage; expected {shape}")
            dtype = mx.uint32 if name == "weight" else mx.float16
            if value.dtype != dtype:
                raise ValueError(f"{key}: TQ3 expected {dtype}, got {value.dtype}")
            params[name] = value
        checks.extend(
            [
                mx.all(params["codebook"] == mx.array(config.codebook, mx.float16)),
                mx.all(mx.abs(params["signs"]) == 1),
                mx.all(mx.isfinite(params["scales"]) & (params["scales"] >= 0)),
            ]
        )
        cls = TQ3ExpertLinear if len(dims) == 3 else TQ3Linear
        updates.append((path, cls(inputs, outputs, params)))
    if not bool(mx.all(mx.stack(checks)).item()):
        raise ValueError("corrupt TQ3 codebook, signs or group scales")
    # Explicit ordinary-format exceptions only. Never pass turboquant to nn.quantize.
    other = scales - prefixes
    for path in other:
        spec = overrides.get(path)
        if not isinstance(spec, dict):
            raise UnsupportedQuantization(
                f"{path}: non-TQ3 packed module needs explicit format override"
            )
        source = modules.get(path)
        if source is None or not hasattr(source, "to_quantized"):
            raise ValueError(f"unknown packed module: {path}")
        updates.append((path, source.to_quantized(**quantization_spec(spec))))
    model.update_modules(tree_unflatten(updates))
    model._mlx_one_tq3 = True
