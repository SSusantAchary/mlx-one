"""Request-scoped GPT-OSS experiments; no loading, tokenization or generation engine.

The default context is empty. All weights remain owned by the native modules.
Unsupported shapes and failed candidates return control to the stock model.
The benchmark harness can select independent experiments before promotion.
"""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache

_MOE_FEATURES = frozenset(
    {
        "compiled_moe",
        "compiled_swiglu",
        "metal_swiglu",
        "mxfp4_projection",
        "fused_gate_up",
        "weighted_reduction",
    }
)
_PREFILL_FEATURES = frozenset({"prefill_sort", "prefill_grouped"})
_FEATURES = _MOE_FEATURES | _PREFILL_FEATURES | {"gqa_sinks", "mlx_lm_moe"}
_context = ContextVar("mlx_one_gptoss_experiment", default=None)


@dataclass
class GPTOSSExperiment:
    features: frozenset[str]
    words_per_lane: int = 1
    rows_per_group: int = 4
    dispatch: Counter = field(default_factory=Counter)
    failed: set = field(default_factory=set)
    warmed: set = field(default_factory=set)


@contextmanager
def _bind_gptoss(state):
    """Bind an existing request-local state for one forward, then restore context."""
    token = _context.set(state)
    try:
        yield state
    finally:
        _context.reset(token)


@contextmanager
def gptoss_execution(features, *, words_per_lane=1, rows_per_group=4):
    features = frozenset(features.split(",") if isinstance(features, str) else features)
    features = features - {"reference", ""}
    unknown = features - _FEATURES
    if (
        unknown
        or ("compiled_moe" in features and len(features & _MOE_FEATURES) != 1)
        or len(features & _PREFILL_FEATURES) > 1
        or ("mlx_lm_moe" in features and len(features) != 1)
    ):
        raise ValueError(f"Unsupported or conflicting GPT-OSS features: {sorted(features)}")
    if words_per_lane not in {1, 4} or rows_per_group not in {1, 2, 4, 8}:
        raise ValueError("Unsupported GPT-OSS kernel configuration")
    state = GPTOSSExperiment(features, words_per_lane, rows_per_group)
    with _bind_gptoss(state):
        yield state


@lru_cache(maxsize=1)
def compiled_swiglu():
    import mlx.core as mx

    def operation(gate, up):
        gate = mx.minimum(gate, 7)
        return (mx.clip(up, -7, 7) + 1) * (gate * mx.sigmoid(1.702 * gate))

    return mx.compile(operation, shapeless=True)


@lru_cache(maxsize=1)
def compiled_moe():
    import mlx.core as mx

    def operation(x, indices, scores, gw, gs, gb, uw, us, ub, dw, ds, db):
        gate = (
            mx.gather_qmm(
                x, gw, gs, rhs_indices=indices, transpose=True, group_size=32, bits=4, mode="mxfp4"
            )
            + gb[indices][..., None, :]
        )
        up = (
            mx.gather_qmm(
                x, uw, us, rhs_indices=indices, transpose=True, group_size=32, bits=4, mode="mxfp4"
            )
            + ub[indices][..., None, :]
        )
        gate = mx.minimum(gate, 7)
        activation = (mx.clip(up, -7, 7) + 1) * (gate * mx.sigmoid(1.702 * gate))
        down = (
            mx.gather_qmm(
                activation,
                dw,
                ds,
                rhs_indices=indices,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
            )
            + db[indices][..., None, :]
        )
        return (down.squeeze(-2) * scores[..., None]).sum(axis=-2)

    # Parameters are explicit inputs: no shared model/weight closure or global
    # mutable model state can accidentally leak across concurrent configurations.
    return mx.compile(operation, shapeless=True)


def moe_forward(experts, x, indices, scores, state):
    """Pure candidate operation, also used by the real-weight operator benchmark."""
    import mlx.core as mx

    from mlx_one.mlxboost import gptoss_kernels as kernels

    expanded = mx.expand_dims(x, (-2, -3))
    gate, up, down = experts.gate_proj, experts.up_proj, experts.down_proj
    if not all(kernels.supported(module, expanded, indices) for module in (gate, up, down)):
        return None
    features = state.features
    settings = {"words_per_lane": state.words_per_lane, "rows_per_group": state.rows_per_group}
    if "compiled_moe" in features:
        return compiled_moe()(
            expanded,
            indices,
            scores,
            gate.weight,
            gate.scales,
            gate.bias,
            up.weight,
            up.scales,
            up.bias,
            down.weight,
            down.scales,
            down.bias,
        )
    if "fused_gate_up" in features:
        activation = kernels.gate_up(gate, up, expanded, indices, **settings)
        if activation is None:
            return None
    else:
        if "mxfp4_projection" in features:
            g, u = (
                kernels.project(gate, expanded, indices, **settings),
                kernels.project(up, expanded, indices, **settings),
            )
            if g is None or u is None:
                return None
        else:
            g, u = gate(expanded, indices), up(expanded, indices)
        if "compiled_swiglu" in features:
            activation = compiled_swiglu()(g, u)
        elif "metal_swiglu" in features:
            activation = kernels.swiglu(g, u)
        else:
            g = mx.minimum(g, 7)
            activation = (mx.clip(u, -7, 7) + 1) * (g * mx.sigmoid(1.702 * g))
    if "mxfp4_projection" in features:
        values = kernels.project(down, activation, indices, **settings)
        if values is None:
            return None
        values = values.squeeze(-2)
    else:
        values = down(activation, indices).squeeze(-2)
    if "weighted_reduction" in features:
        return kernels.weighted_sum(values, scores)
    return (values * scores[..., None]).sum(axis=-2)


def try_moe(experts, x, indices, scores):
    state = _context.get()
    if state is None:
        return None
    if "mlx_lm_moe" in state.features:
        return try_mlx_lm_moe(experts, x, indices, scores, state)
    if state.features & _PREFILL_FEATURES and len(x.shape) == 3 and x.shape[1] >= 16:
        return try_prefill(experts, x, indices, scores, state)
    if not state.features & _MOE_FEATURES:
        return None
    if (
        x.shape != (1, 1, 2880)
        or indices.shape != (1, 1, 4)
        or experts.swiglu_limit != 7
        or experts.gate_proj.weight.shape[:2] != (32, 2880)
    ):
        state.dispatch["stock: unsupported shape or architecture"] += 1
        return None
    key = (x.dtype, x.shape, state.features)
    if key in state.failed:
        state.dispatch["stock: previously unavailable candidate"] += 1
        return None
    try:
        result = moe_forward(experts, x, indices, scores, state)
        if result is None:
            state.dispatch["stock: unsupported packed layout or dtype"] += 1
            return None
        if key not in state.warmed:
            # Resolve initial Metal compilation errors while fallback is still
            # safe: MoE is pure and has not committed any recurrent/cache state.
            import mlx.core as mx

            mx.eval(result)
            state.warmed.add(key)
        state.dispatch["candidate: " + ",".join(sorted(state.features))] += 1
        return result
    except RuntimeError:
        state.failed.add(key)
        state.dispatch["stock: candidate compile/runtime failure"] += 1
        return None


def try_mlx_lm_moe(experts, x, indices, scores, state):
    """Use the complete pinned upstream MoE sequence for prefill and decode."""
    if (
        x.ndim != 3
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 2048
        or x.shape[-1] != 2880
        or indices.shape != (*x.shape[:2], 4)
        or experts.gate_proj.weight.shape[:2] != (32, 2880)
    ):
        state.dispatch["mlx_lm stock: unsupported shape or architecture"] += 1
        return None
    key = ("mlx_lm_moe", x.dtype, x.shape)
    if key in state.failed:
        state.dispatch["mlx_lm stock: previously unavailable candidate"] += 1
        return None
    try:
        from mlx_one.mlxboost.gptoss_mlx_lm import mlx_lm_moe

        result = mlx_lm_moe(experts, x, indices, scores)
        if result is None:
            state.dispatch["mlx_lm stock: unsupported packed layout or dtype"] += 1
            return None
        if key not in state.warmed:
            import mlx.core as mx

            mx.eval(result)
            state.warmed.add(key)
        state.dispatch[
            "mlx_lm candidate: sorted" if x.shape[1] >= 16 else "mlx_lm candidate: unsorted"
        ] += 1
        return result
    except RuntimeError:
        state.failed.add(key)
        state.dispatch["mlx_lm stock: candidate compile/runtime failure"] += 1
        return None


def try_prefill(experts, x, indices, scores, state):
    """Schedule only the target B1, BF16, top-four 20B prefill geometry."""
    if (
        x.ndim != 3
        or x.shape[0] != 1
        or x.shape[1] > 1024
        or x.shape[-1] != 2880
        or indices.shape != (*x.shape[:2], 4)
        or experts.gate_proj.weight.shape[:2] != (32, 2880)
    ):
        state.dispatch["prefill stock: unsupported shape or architecture"] += 1
        return None
    feature = next(iter(state.features & _PREFILL_FEATURES))
    key = ("prefill", x.dtype, x.shape, feature)
    if key in state.failed:
        state.dispatch["prefill stock: previously unavailable candidate"] += 1
        return None
    try:
        from mlx_one.mlxboost.gptoss_prefill import prefill_moe

        result = prefill_moe(experts, x, indices, scores, grouped=feature == "prefill_grouped")
        if result is None:
            state.dispatch["prefill stock: unsupported packed layout or dtype"] += 1
            return None
        if key not in state.warmed:
            import mlx.core as mx

            mx.eval(result)
            state.warmed.add(key)
        state.dispatch["prefill candidate: " + feature] += 1
        return result
    except RuntimeError:
        state.failed.add(key)
        state.dispatch["prefill stock: candidate compile/runtime failure"] += 1
        return None


def try_attention(queries, keys, values, sinks, *, full_causal_single_query):
    state = _context.get()
    if state is None or "gqa_sinks" not in state.features:
        return None
    if not full_causal_single_query:
        state.dispatch["attention stock: unsupported attention scope"] += 1
        return None
    key = ("attention", queries.dtype)
    if key in state.failed:
        state.dispatch["attention stock: previously unavailable candidate"] += 1
        return None
    try:
        from mlx_one.mlxboost.gptoss_attention import attention

        result = attention(queries, keys, values, sinks.astype(queries.dtype))
        if result is None:
            state.dispatch["attention stock: unsupported layout or dtype"] += 1
            return None
        if key not in state.warmed:
            import mlx.core as mx

            mx.eval(result)
            state.warmed.add(key)
        state.dispatch["attention candidate: gqa_sinks"] += 1
        return result
    except RuntimeError:
        state.failed.add(key)
        state.dispatch["attention stock: candidate compile/runtime failure"] += 1
        return None
