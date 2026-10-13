"""Native execution adapters: topology and precision, independent of scheduling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

ADAPTER_VERSION = "native-adapter-v1"
EXECUTION_POLICY_VERSION = "native-float-execution-v1"
DENSE_FAMILIES = {"qwen2", "qwen3"}


def dtype_name(value):
    return str(value.dtype).rsplit(".", 1)[-1]


@dataclass(frozen=True)
class QuantizedModuleSpec:
    path: str
    bits: int
    group_size: int
    mode: str
    weight_shape: tuple
    scale_shape: tuple
    scale_dtype: str


def affine_tensor_spec(path, weight, scales, biases, *, bits, group_size, mode):
    if bits not in {4, 8} or isinstance(bits, bool) or mode != "affine":
        raise ValueError("native affine module requires 4/8-bit integer packing")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 1:
        raise ValueError("invalid affine group size")
    if dtype_name(weight) != "uint32" or len(weight.shape) != 2:
        raise ValueError("affine weights must be rank-two packed uint32 tensors")
    rows, packed = weight.shape
    width = packed * 32 // bits
    expected = (rows, width // group_size)
    if width % group_size or tuple(scales.shape) != expected or tuple(biases.shape) != expected:
        raise ValueError("affine scale/bias grouping or packing mismatch")
    if dtype_name(scales) not in {"float16", "bfloat16", "float32"} or (
            dtype_name(scales) != dtype_name(biases)):
        raise ValueError("affine scale/bias tensors need matching floating dtype")
    return QuantizedModuleSpec(path, bits, group_size, mode, tuple(weight.shape),
                               tuple(scales.shape), dtype_name(scales))


def quantized_modules(bundle):
    config = bundle.quantization
    if not config:
        return ()
    if config.get("bits") != 8 or config.get("mode", "affine") != "affine":
        raise ValueError("optimized quantized execution requires affine INT8 weights")
    named = getattr(bundle.model, "named_modules", None)
    if not callable(named):
        raise ValueError("quantized execution requires observed native module metadata")
    result = []
    for path, module in named():
        if not hasattr(module, "bits"):
            continue
        if module.bits != 8 or module.group_size != config.get("group_size", 64):
            raise ValueError("module quantization differs from checkpoint metadata")
        result.append(affine_tensor_spec(path, module.weight, module.scales, module.biases,
            bits=module.bits, group_size=module.group_size, mode=module.mode))
    if not result:
        raise ValueError("checkpoint declares INT8 but no packed modules were observed")
    return tuple(result)


def quantization_identity(bundle):
    config = getattr(bundle, "quantization", {})
    modules = quantized_modules(bundle) if config.get("bits") == 8 else ()
    payload = {"declared": config, "modules": [asdict(s) for s in modules],
               "floating_policy": getattr(bundle.model, "_mlx_one_execution_policy", None)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class NativeExecutionSpec:
    architecture: str
    topology_digest: str
    query_heads: int
    kv_heads: int
    head_dim: int
    qk_normalization: bool
    state_kinds: tuple[str, ...]
    quantization_digest: str
    adapter_version: str = ADAPTER_VERSION

    @property
    def hybrid(self):
        return any(kind == "conv" for kind in self.state_kinds)


class NativeModelAdapter:
    def __init__(self, bundle, plan, *, allow_bfloat16=False):
        from mlx_one.engine.reporting import tensor_dtypes

        architecture = plan.architecture
        if architecture not in DENSE_FAMILIES:
            raise ValueError("no supported dense native adapter for " + architecture)
        observed = tensor_dtypes(bundle.model.parameters())
        if bundle.quantization:
            quantized_modules(bundle)
            if set(observed) - {"float16", "uint32"} or "float16" not in observed:
                raise ValueError("INT8 batching requires explicit FP16 floating execution")
        elif observed != ("float16",) and not (allow_bfloat16 and observed == ("bfloat16",)):
            raise ValueError("dense batching requires observed FP16 weights")
        config = bundle.model.config
        self.bundle = bundle
        self.spec = NativeExecutionSpec(architecture, plan.topology_digest,
            config.num_attention_heads, config.num_key_value_heads,
            getattr(config, "head_dim", 0) or config.hidden_size // config.num_attention_heads,
            architecture == "qwen3", tuple(s.kind for s in plan.state_specs),
            quantization_identity(bundle))
        if not plan.block_compatible:
            raise ValueError("adapter requires complete attention layer layouts")

    def forward(self, input_ids, *, cache, attention_mask=None, position_ids=None):
        return self.bundle.model(input_ids, cache=cache, attention_mask=attention_mask,
                                 position_ids=position_ids)


def native_adapter(bundle, plan=None, *, allow_bfloat16=False):
    from mlx_one.engine.cache_planner import build_cache_plan

    return NativeModelAdapter(bundle, plan or build_cache_plan(bundle),
                              allow_bfloat16=allow_bfloat16)
