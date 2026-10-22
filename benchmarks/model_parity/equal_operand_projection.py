"""Isolate packed projections only after checking actual operands and payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from four_path_oracle import compare, load_reference, snapshot


def run(checkpoint, precision, target):
    import mlx.core as mx
    import numpy as np
    from layerwise import canonical_name

    from mlx_one.models.shared import execution
    from mlx_one.server.model_manager import ModelManager

    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    manager = ModelManager()
    bundle = manager.load(checkpoint, offline=True)

    def compute():
        nonlocal bundle
        bundle, reference, reference_module = load_reference(bundle, precision)
        selected = {}
        for label, model in (("native", bundle.model), ("reference", reference)):
            selected[label] = next(
                m
                for n, m in model.named_modules()
                if canonical_name(n, bundle.architecture) == target
            )
        seed = bundle.tokenizer.encode("The capital of France is")
        tokens = mx.array([(seed * ((16 + len(seed) - 1) // len(seed)))[:16]])
        inputs, outputs = {}, {}
        original_project = execution._project
        originals = {}

        def save(label, value, output):
            mx.eval(value, output)
            # Independent evaluated arrays, not aliases into later mutable state.
            inputs[label] = mx.array(snapshot(value), dtype=value.dtype)
            outputs[label] = snapshot(output)

        def call_wrapper(original):
            def call(module, value, *args, **kwargs):
                result = original(module, value, *args, **kwargs)
                for label, chosen in selected.items():
                    if module is chosen:
                        save(label, value, result)
                return result

            return call

        def project(module, value, **kwargs):
            result = original_project(module, value, **kwargs)
            if module is selected["native"]:
                save("native", value, result)
            return result

        try:
            for cls in {type(m) for m in selected.values()}:
                originals[cls] = ("__call__" in cls.__dict__, cls.__call__)
                cls.__call__ = call_wrapper(cls.__call__)
            execution._project = project
            mx.eval(reference(tokens))
            mx.eval(bundle.model(tokens).logits)
        finally:
            execution._project = original_project
            for cls, (owned, original) in originals.items():
                if owned:
                    cls.__call__ = original
                else:
                    del cls.__call__
        dtype = str(inputs["reference"].dtype).rsplit(".", 1)[-1]
        operand_metrics = compare(snapshot(inputs["reference"]), snapshot(inputs["native"]), dtype)
        payload = {}
        for key in ("weight", "scales", "biases"):
            tensors = [getattr(selected[label], key) for label in ("reference", "native")]
            mx.eval(*tensors)
            arrays = [np.array(x, copy=True) for x in tensors]
            payload[key] = {
                "exact_equal": bool(np.array_equal(*arrays)),
                "shape": list(tensors[0].shape),
                "dtypes": [str(x.dtype) for x in tensors],
                "sha256": [hashlib.sha256(x.tobytes()).hexdigest() for x in arrays],
            }
        module = selected["native"]
        value = inputs["reference"]
        expected = selected["reference"](value)
        gathered = original_project(module, value, execution_dtype="float16")
        ordinary = mx.quantized_matmul(
            value,
            module.weight,
            scales=module.scales,
            biases=module.biases,
            transpose=True,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
        )
        dequantized = mx.dequantize(
            module.weight,
            module.scales.astype(mx.float32),
            module.biases.astype(mx.float32),
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
        )
        ref_module = selected["reference"]
        ref_dequantized = mx.dequantize(
            ref_module.weight,
            ref_module.scales.astype(mx.float32),
            ref_module.biases.astype(mx.float32),
            group_size=ref_module.group_size,
            bits=ref_module.bits,
            mode=ref_module.mode,
        )
        dense = (value.astype(mx.float32) @ dequantized.T).astype(value.dtype)
        if "bias" in module:
            ordinary, dense = ordinary + module.bias, dense + module.bias
        # Only one row for the independent CPU FP64 reference; bounded diagnostic allocation.
        cpu = snapshot(value[:, :1]).astype(np.float64) @ snapshot(dequantized).astype(np.float64).T
        if "bias" in module:
            cpu += snapshot(module.bias)
        # Decode the packed integer payload independently, without an MLX FP32
        # dequantization rounding becoming part of the supposed FP64 oracle.
        words = np.array(module.weight).astype(np.uint32)
        shifts = np.arange(32 // module.bits, dtype=np.uint32) * module.bits
        codes = ((words[..., None] >> shifts) & ((1 << module.bits) - 1)).reshape(
            words.shape[0], -1
        )
        groups = codes.reshape(codes.shape[0], -1, module.group_size)
        exact_weight = (
            groups * snapshot(module.scales).astype(np.float64)[..., None]
            + snapshot(module.biases).astype(np.float64)[..., None]
        ).reshape(codes.shape)
        exact_cpu = snapshot(value[:, :1]).astype(np.float64) @ exact_weight.T
        if "bias" in module:
            exact_cpu += snapshot(module.bias).astype(np.float64)
        result = {
            "checkpoint": bundle.model_id,
            "reference": {
                "module": reference_module.__file__,
                "sha256": hashlib.sha256(Path(reference_module.__file__).read_bytes()).hexdigest(),
            },
            "revision": bundle.revision,
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "precision": precision,
            "input_ids": snapshot(tokens)[0].tolist(),
            "context": 16,
            "parts": [16],
            "target": target,
            "input_dtype": dtype,
            "output_dtype": str(expected.dtype),
            "input_shape": list(value.shape),
            "first_row_activation_sha256": hashlib.sha256(
                snapshot(value[:, :1]).tobytes()
            ).hexdigest(),
            "projection_inputs": operand_metrics,
            "packed_payload": payload,
            "bits": module.bits,
            "group_size": module.group_size,
            "quantization_metadata": {
                label: {
                    "bits": m.bits,
                    "group_size": m.group_size,
                    "mode": m.mode,
                    "group_axis": -1,
                }
                for label, m in selected.items()
            },
            "dequantized_weight_equivalence": compare(
                snapshot(ref_dequantized), snapshot(dequantized), "float32"
            ),
            "orientation": "out_features x packed_in_features; transpose=True",
            "native_vs_reference_actual_forward": compare(
                outputs["reference"], outputs["native"], dtype
            ),
            "equal_operands": {
                "reference_vs_native_gather": compare(
                    snapshot(expected), snapshot(gathered), dtype
                ),
                "reference_vs_ordinary_qmm": compare(snapshot(expected), snapshot(ordinary), dtype),
                "reference_vs_fp32_dequantized_dense": compare(
                    snapshot(expected), snapshot(dense), dtype
                ),
                "cpu_float64_vs_reference_first_row": compare(
                    cpu, snapshot(expected[:, :1]), dtype
                ),
                "cpu_float64_vs_gather_first_row": compare(cpu, snapshot(gathered[:, :1]), dtype),
                "cpu_float64_vs_dense_first_row": compare(cpu, snapshot(dense[:, :1]), dtype),
                "qmm_full_vs_single_row": compare(
                    snapshot(ordinary[:, :1]), snapshot(selected["reference"](value[:, :1])), dtype
                ),
                "gather_full_vs_single_row": compare(
                    snapshot(gathered[:, :1]),
                    snapshot(original_project(module, value[:, :1], execution_dtype="float16")),
                    dtype,
                ),
            },
            "diagnostic_only": True,
            "exact_affine_cpu_fp64": {
                "equation": "integer codes * stored scale + stored bias in NumPy FP64",
                "input_dtype": dtype,
                "weight_storage_dtype": str(module.weight.dtype),
                "scale_dtype": str(module.scales.dtype),
                "accumulator_dtype": None,
                "kernel_accumulator_contract": "UNKNOWN; no assumed FP32 error bound",
                "packed": compare(exact_cpu, snapshot(ordinary[:, :1]), dtype),
                "gather": compare(exact_cpu, snapshot(gathered[:, :1]), dtype),
                "dense_fp32": compare(exact_cpu, snapshot(dense[:, :1]), dtype),
                "dequantized_weight_max_fp32_rounding": float(
                    np.max(np.abs(exact_weight - snapshot(dequantized)))
                ),
                "criterion": "Unchanged frozen component tolerance; bounded first-row control",
            },
        }
        return result

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = run(args.checkpoint, args.precision, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        result["checkpoint"],
        "inputs",
        result["projection_inputs"]["max_abs_error"],
        "payload",
        all(r["exact_equal"] for r in result["packed_payload"].values()),
        flush=True,
    )


if __name__ == "__main__":
    main()
