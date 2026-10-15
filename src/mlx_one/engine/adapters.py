"""Native execution adapters: topology and precision, independent of scheduling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import SimpleNamespace

from mlx_one.engine.contracts import LayerExecutionSpec

ADAPTER_VERSION = "native-adapter-v2"
EXECUTION_POLICY_VERSION = "native-float-execution-v4"
DENSE_FAMILIES = {"qwen2", "qwen3", "llama"}


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


def quantized_modules(bundle, *, allowed_bits=(8,)):
    config = bundle.quantization
    if not config:
        return ()
    if config.get("bits") not in allowed_bits or config.get("mode", "affine") != "affine":
        raise ValueError("optimized quantized execution requires affine INT8 weights")
    named = getattr(bundle.model, "named_modules", None)
    if not callable(named):
        raise ValueError("quantized execution requires observed native module metadata")
    result = []
    for path, module in named():
        if not hasattr(module, "bits"):
            continue
        if module.bits != config["bits"] or module.group_size != config.get("group_size", 64):
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
    model_id: str | None = None
    revision: str | None = None
    layers: tuple[LayerExecutionSpec, ...] = ()

    @property
    def hybrid(self):
        return any(kind in {"conv","recurrent"} for kind in self.state_kinds)


class NativeModelAdapter:
    supported_architectures = frozenset(DENSE_FAMILIES)
    supports_paging = True
    supported_states = frozenset({"full_kv"})

    def __init__(self, bundle, plan, *, allow_bfloat16=False, allow_hybrid=False):
        from mlx_one.engine.reporting import tensor_dtypes

        architecture = plan.architecture
        if architecture not in self.supported_architectures:
            raise ValueError("no supported dense native adapter for " + architecture)
        observed = tensor_dtypes(bundle.model.parameters())
        legacy_hybrid = (allow_hybrid and architecture == "lfm2"
                         and bundle.quantization.get("bits") == 4)
        if legacy_hybrid:
            # Preserve the pre-P1 affine INT4 hybrid path without changing its floating policy.
            quantized_modules(bundle, allowed_bits=(4,))
            if set(observed) - {"float16", "bfloat16", "float32", "uint32"} or not (
                    set(observed) & {"float16", "bfloat16", "float32"}):
                raise ValueError("unsupported legacy hybrid floating precision")
        elif bundle.quantization:
            quantized_modules(bundle)
            if set(observed) - {"float16", "uint32"} or "float16" not in observed:
                raise ValueError("INT8 batching requires explicit FP16 floating execution")
        elif observed != ("float16",) and not (allow_bfloat16 and observed == ("bfloat16",)):
            raise ValueError("dense batching requires observed FP16 weights")
        config = getattr(bundle.model.config, "text_config", bundle.model.config)
        layers = []
        for layer, state in zip(plan.layer_specs, plan.state_specs, strict=True):
            attention = state.kind in {"full_kv", "sliding_kv", "cross_kv"}
            query_heads = layer.num_query_heads if attention else None
            if attention and query_heads is None:
                query_heads = getattr(config, "num_attention_heads",
                                      getattr(config, "n_head", None))
            layout = LayerExecutionSpec(layer.layer_id, state.kind, query_heads,
                layer.num_kv_heads, layer.key_head_dim, layer.value_head_dim or layer.key_head_dim,
                layer.position_policy, layer.max_window_tokens)
            if attention and layout.position_policy == "rope" and layout.key_dim % 2:
                raise ValueError("invalid native RoPE head dimension")
            layers.append(layout)
        attention_layers = [layer for layer in layers
                            if layer.state_kind in {"full_kv", "sliding_kv"}]
        if not attention_layers:
            raise ValueError("adapter requires supported attention layout")
        first = attention_layers[0]
        query_heads, kv_heads, head_dim = first.query_heads, first.kv_heads, first.key_dim
        if any(s.kind not in self.supported_states for s in plan.state_specs) or (
                not allow_hybrid and any(s.kind != "full_kv" for s in plan.state_specs)):
            raise ValueError("dense adapter requires full-attention components")
        hidden = getattr(config, "hidden_size", getattr(config, "n_embd",
                         getattr(config, "model_dim", 0)))
        self.activation_config = SimpleNamespace(hidden_size=hidden,
            num_attention_heads=max(layer.query_heads for layer in attention_layers),
            intermediate_size=getattr(config, "intermediate_size", getattr(config, "inner_dim",
                                       max(getattr(config, "ffn_dims", (hidden * 4,))))),
            vocab_size=getattr(config, "vocab_size", 0))
        self.bundle, self.plan = bundle, plan
        self.text_config = config
        self.activation_dtype = None if legacy_hybrid else next(
            dtype for dtype in observed if dtype in {"float16", "bfloat16"})
        self.spec = NativeExecutionSpec(architecture, plan.topology_digest,
            query_heads, kv_heads, head_dim, bool(getattr(self,"qk_normalization",
                getattr(config,"normalize_qk_projections",False))),
            tuple(s.kind for s in plan.state_specs), quantization_identity(bundle),
            model_id=getattr(bundle, "model_id", None), revision=getattr(bundle, "revision", None),
            layers=tuple(layers))
        if not plan.block_compatible and not (allow_hybrid and plan.state_specs
                and all(s.kind in self.supported_states for s in plan.state_specs)):
            raise ValueError("adapter requires complete supported layer layouts")

    def memory_estimate(self, tokens):
        from mlx_one.engine.cache import array_nbytes

        return {"model_storage_bytes": array_nbytes(self.bundle.model.parameters()),
            "fp16_kv_bytes": self.plan.bytes_per_token() * tokens,
            "carry_bytes": self.plan.carry_bytes()}

    def additional_transient_bytes(self, *, batch, chunk):
        config = getattr(self.bundle.model.config, "text_config", self.bundle.model.config)
        experts = int(getattr(config, "num_experts", 0))
        if not experts:
            return 0
        hidden = self.activation_config.hidden_size
        intermediate = int(getattr(config, "moe_intermediate_size", hidden * 4))
        layers = len(self.plan.layer_specs)
        # Native StackedExperts evaluates all expert outputs; active parameters
        # cannot be substituted for either resident weights or this scratch bound.
        return batch * chunk * experts * (2 * intermediate + hidden + layers * 2) * 4

    def forward(self, input_ids, *, cache, attention_mask=None, position_ids=None):
        # Cast RoPE/QK outputs within explicit optimized execution, without changing
        # weights or leaving a policy behind on a shared reference model.
        restored = []
        try:
            if self.activation_dtype:
                for _, module in getattr(self.bundle.model, "named_modules", lambda: ())():
                    if any(hasattr(module, name) for name in
                           ("q_proj", "c_attn", "qkv_proj", "in_proj_qkv")):
                        existed = hasattr(module, "_mlx_one_activation_dtype")
                        restored.append((module, existed,
                                         getattr(module, "_mlx_one_activation_dtype", None)))
                        module._mlx_one_activation_dtype = self.activation_dtype
            return self.bundle.model(input_ids, cache=cache, attention_mask=attention_mask,
                                     position_ids=position_ids)
        finally:
            for module, existed, previous in restored:
                if existed:
                    module._mlx_one_activation_dtype = previous
                else:
                    del module._mlx_one_activation_dtype


def native_adapter(bundle, plan=None, *, allow_bfloat16=False, allow_hybrid=False):
    from mlx_one.engine.adapter_registry import get_runtime_registration
    from mlx_one.engine.cache_planner import build_cache_plan

    plan = plan or build_cache_plan(bundle)
    binding = get_runtime_registration(plan.architecture)
    adapter = binding.adapter() if binding.adapter_path is not None else NativeModelAdapter
    return adapter(bundle, plan,
                              allow_bfloat16=allow_bfloat16, allow_hybrid=allow_hybrid)


def execution_contract(bundle, plan, mode):
    """Observed implementation identity, including mixed integer/floating precision."""
    from mlx_one.engine.reporting import tensor_dtypes

    quantization = getattr(bundle, "quantization", {})
    payload = {"adapter_version": ADAPTER_VERSION, "architecture": plan.architecture,
        "topology_digest": plan.topology_digest, "execution_mode": mode,
        "quantization": quantization, "quantization_digest": quantization_identity(bundle),
        "modules": [asdict(s) for s in quantized_modules(bundle)] if quantization.get(
            "bits") == 8 else [],
        "floating_policy": getattr(bundle.model, "_mlx_one_execution_policy", None),
        "actual_weight_dtypes": tensor_dtypes(bundle.model.parameters())
            if callable(getattr(bundle.model, "parameters", None)) else (),
        "components": [{**asdict(s), "storage": (
            "segmented-arena" if mode == "paged" and s.kind == "full_kv" else
            "request-local-carry" if s.kind in {"conv","recurrent"} else "dense")}
            for s in plan.state_specs]}
    return json.loads(json.dumps(payload, sort_keys=True))
