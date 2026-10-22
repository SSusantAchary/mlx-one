"""Process-local module hooks for bounded diagnostics; no production instrumentation."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager


def canonical_name(name, family):
    if family == "gpt2" and name.startswith("model."):
        return "transformer." + name[6:]
    if family == "qwen3_5":
        return name.replace("language_model.model.", "language_model.")
    return name


@contextmanager
def capture_modules(model, family, *, max_layers=64, projection_inputs=False):
    """Capture module outputs in execution order, restoring every patched method."""
    import mlx.core as mx

    selected = {}
    for name, module in model.named_modules():
        name = canonical_name(name, family)
        if not name or name.startswith(("visual", "vision")):
            continue
        parts = name.split(".")
        indices = [int(p) for p in parts if p.isdigit()]
        if indices and indices[0] >= max_layers:
            continue
        selected[id(module)] = name
    events, counts, originals = [], defaultdict(int), {}
    classes = {type(m) for _, m in model.named_modules() if id(m) in selected}

    def wrapper(original):
        def call(module, *args, **kwargs):
            name = selected.get(id(module))
            projections = (
                ".q_proj",
                ".k_proj",
                ".v_proj",
                ".gate_proj",
                ".up_proj",
                ".down_proj",
                ".qkv_proj",
                ".gate_up_proj",
                ".in_proj",
                ".in_proj_qkv",
                ".in_proj_z",
                ".in_proj_b",
                ".in_proj_a",
                ".c_fc",
                ".c_attn",
                ".w1",
                ".w2",
                ".w3",
            )
            capture_input = name and (
                name.endswith((".o_proj", ".c_proj", ".out_proj"))
                or (projection_inputs and name.endswith(projections))
            )
            if capture_input and args:
                value = args[0]
                if isinstance(value, mx.array):
                    snapshot = mx.zeros_like(value)
                    snapshot[...] = value
                    mx.eval(snapshot)
                    key = name + ".input"
                    counts[key] += 1
                    events.append((f"{key}#{counts[key]}", snapshot))
            output = original(module, *args, **kwargs)
            name = selected.get(id(module))
            value = output[0] if isinstance(output, tuple) else output
            if name is not None and isinstance(value, mx.array):
                # Retain arrays only inside the bounded diagnostic forward.
                snapshot = mx.zeros_like(value)
                snapshot[...] = value
                mx.eval(snapshot)
                counts[name] += 1
                if name.endswith(".wpe") and snapshot.ndim == 2:
                    snapshot = snapshot[None]
                events.append((f"{name}#{counts[name]}", snapshot))
            return output

        return call

    try:
        for cls in classes:
            originals[cls] = ("__call__" in cls.__dict__, cls.__call__)
            cls.__call__ = wrapper(cls.__call__)
        yield events
    finally:
        for cls, (owned, original) in originals.items():
            if owned:
                cls.__call__ = original
            else:
                del cls.__call__


def compare_events(reference, candidate, dtype):
    import mlx.core as mx
    import numpy as np

    from mlx_one.ops.numerical_trace import compare_tensors, tensor_metadata

    candidate = dict(candidate)
    results = []
    for name, left in reference:
        right = candidate.get(name)
        if right is None:
            continue
        result = compare_tensors(
            np.asarray(left.astype(mx.float32)),
            np.asarray(right.astype(mx.float32)),
            dtype=dtype,
            reference_metadata=tensor_metadata(left),
            candidate_metadata=tensor_metadata(right),
        )
        left_values, right_values = (
            np.asarray(left.astype(mx.float32)).ravel(),
            np.asarray(right.astype(mx.float32)).ravel(),
        )
        if left_values.shape == right_values.shape:
            result["rms_error"] = float(
                np.sqrt(np.mean((left_values.astype(np.float64) - right_values) ** 2))
            )
            denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
            result["cosine_similarity"] = (
                float(np.dot(left_values, right_values) / denominator) if denominator else None
            )
        results.append({"operation": name, **result})
    return {
        "operations": results,
        "unmatched_reference_operations": [name for name, _ in reference if name not in candidate],
        "first_nonzero": next((r["operation"] for r in results if r["changed_elements"]), None),
        "first_failure": next((r["operation"] for r in results if r["status"] != "PASS"), None),
    }
