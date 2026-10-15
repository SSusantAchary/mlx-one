"""Explicit simulated provenance for registry unit tests; never qualification results."""

import hashlib
import json


def attach_provenance(proof, directory):
    from mlx_one.engine.adapters import ADAPTER_VERSION

    scope = proof["scope"]
    if not proof.get("execution_contract"):
        proof["execution_contract"] = {"adapter_version": ADAPTER_VERSION,
        "topology_digest": scope["topology_digest"], "execution_mode": scope["execution_path"],
        "quantization_digest": "unit-test-only", "components": [{"kind": "full_kv"}]}
    cache_type = {"float16": "f16", "bfloat16": "bf16",
                  "float16,uint8": "q4_0"}[scope["cache_precision"]]
    payloads = {
        "measurement.json": {"backend": "mlx-metal", "metal_available": True},
        "checkpoint.json": {"revision": scope["revision"],
                            "files": {"model.safetensors": "a" * 64}},
        "manifest.json": {"source": {"git_revision": "b" * 40},
                          "hardware": {"chip": "unit-test-only", "memory_bytes": 1024},
                          "profile": {"model_id": scope["model_id"],
                                      "revision": scope["revision"],
                                      "inference_mode": scope["execution_path"],
                                      "cache_type": cache_type}},
    }
    if proof.get("performance_qualified"):
        payloads["gate-d.json"] = {"status": "passed", "checks": {"unit_test": True}}
    if proof.get("feature") in {"quantized_paged_kv", "long_context"}:
        from mlx_one.quality_dataset import DEPTH_BANDS

        context = proof.get("workload_envelope", {}).get("context_tokens", 2048)
        payloads["manifest.json"]["profile"]["context_tokens"] = context
        payloads["held-out-quality.json"] = {"document_count": 100, "scored_tokens": 100000,
                                             "dataset_identity": {"fixture": True}}
        payloads["retrieval.json"] = {"fixtures": [
            {"depth_band": band, "context_tokens": context, "rendered_tokens": context}
            for band in DEPTH_BANDS for _ in range(20)]}
    for name, payload in payloads.items():
        path = directory / name
        path.write_text(json.dumps(payload))
        proof.setdefault("artifacts", []).append({"uri": str(path), "kind": "cpu-test-fixture",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    proof["validation_kind"] = "native-hardware"
    return proof
