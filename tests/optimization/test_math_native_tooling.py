"""Evidence/observer tests without importing MLX or constructing native models."""
import copy
import importlib
import json

import numpy as np
import pytest

from mlx_one.inference_qualification import write_integrity
from qualification.p01_experiments import (
    VERSION,
    policy_settings,
    unavailable_operator,
    validate_controlled_result,
    validate_parity_receipt,
    validate_result,
)
from qualification.p01_math import canonical_digest
from qualification.p01_native_cases import MathObserver, compare_components


def fixture():
    metadata = {"model_id": "fixture", "revision": "a" * 40, "assets": {"weights": "digest"}}
    source = {"source_sha256": "source"}
    task = {"route": "optimized", "chunk": 512, "profile": False, "workload": "parity",
            "prompt_tokens": 2, "operators": {"attention": "mlx_sdpa", "projection": "mlx_qmm"}}
    metrics = {"within_tolerance": True, "greedy_agreement": True}
    result = {"schema_version": VERSION, "source": source, "policy": policy_settings(task),
        "status": "PASS", "identity": {"checkpoint": metadata["model_id"],
        "revision": metadata["revision"], "assets_sha256": canonical_digest(metadata["assets"]),
        "prompt_tokens": 2, "prompt_ids": [1, 2], "source_sha256": "source",
        "execution_dtype": "float16", "kv_dtype": "float16", "batch": 1,
        "prefix_reuse": False, "profile": False},
        "native_parity": {"status": "PASS", "logits": [metrics], "components": [metrics],
            "full_prefill_components": [metrics], "independent_full_prefill": metrics,
            "forced_tokens": [3], "token_ids": [4], "target_token_ids": [4],
            "dispatch": [{"operation": "attention", "selected": "mlx_sdpa"},
                         {"operation": "projection", "selected": "mlx_qmm"}],
            "observed_outputs": {name: [{"logits": [1, 1 if name == "target" else 2, 16],
                "logits_dtype": "float16", "hidden": [1, 2, 8], "input_tokens": 2}]
                for name in ("reference", "target", "full_reference")}},
        "ownership": {"active_handles": 0, "reserved_bytes": 0, "allocated_bytes": 0}}
    return task, metadata, source, result


@pytest.mark.parametrize("mutation", (
    lambda r: r.update(schema_version="old"),
    lambda r: r["policy"].update(attention="reference"),
    lambda r: r["source"].update(source_sha256="stale"),
    lambda r: r["identity"].update(revision="wrong"),
    lambda r: r["identity"].update(execution_dtype="float32"),
    lambda r: r["identity"].update(prompt_ids=[1]),
    lambda r: r["native_parity"].update(logits=[]),
    lambda r: r["native_parity"].update(components=[]),
    lambda r: r["native_parity"]["logits"][0].update(within_tolerance=False),
    lambda r: r["native_parity"]["logits"][0].update(greedy_agreement=False),
    lambda r: r["native_parity"].update(target_token_ids=[5]),
    lambda r: r["ownership"].update(reserved_bytes=8),
    lambda r: r["native_parity"].update(observed_outputs={}),
    lambda r: r["native_parity"].update(dispatch=[]),
    lambda r: r["native_parity"]["observed_outputs"]["target"][0].update(logits=[1, 2, 16]),
))
def test_native_receipt_cannot_substitute_missing_or_mismatched_evidence(mutation):
    task, metadata, source, result = fixture()
    validate_result(result, task, metadata, source)
    changed = copy.deepcopy(result)
    mutation(changed)
    with pytest.raises(ValueError):
        validate_result(changed, task, metadata, source)


def test_checksums_supervision_and_hardware_required_for_parity_receipt(tmp_path):
    task, metadata, source, result = fixture()
    root = tmp_path / "worker"
    data = root / "data"
    data.mkdir(parents=True)
    hardware = {"chip": "native fixture"}
    manifest = {"task": task, "native": {"environment": {"preflight": {"hardware": hardware}}}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "supervision.json").write_text(json.dumps({"status": "completed"}))
    path = data / "experiment.json"
    path.write_text(json.dumps(result))
    write_integrity(data)
    write_integrity(root)
    assert validate_parity_receipt(path, task, metadata, source, hardware)["path"] == str(path)
    with pytest.raises(ValueError, match="hardware"):
        validate_parity_receipt(path, task, metadata, source, {"chip": "different"})
    path.write_text(json.dumps({**result, "policy": {}}))
    with pytest.raises(ValueError):
        validate_parity_receipt(path, task, metadata, source, hardware)


def test_component_comparison_retains_required_fp32_tolerance_and_alignment():
    reference = [{"class": "HybridState", "position": 128, "fields": {
        "recurrent_state": {"shape": [1, 2], "dtype": "float32",
                            "array": np.array([[1.0, 2.0]])}}}]
    target = copy.deepcopy(reference)
    target[0]["fields"]["recurrent_state"]["array"][0, 0] += .001
    metrics = compare_components(reference, target)
    assert metrics[0]["atol"] == 1e-4 and metrics[0]["within_tolerance"] is False
    assert "greedy_agreement" not in metrics[0]
    target[0]["position"] = 127
    with pytest.raises(ValueError, match="position"):
        compare_components(reference, target)


def test_phase_trace_bounded_and_nested_phases_are_marked_inclusive(monkeypatch):
    module = importlib.import_module("qualification.p01_native_cases")
    monkeypatch.setattr(module, "TRACE_LIMIT", 2)
    observer = MathObserver(128, profile=True)
    for _ in range(3):
        observer({"kind": "math_phase", "phase": "nested", "duration_ms": 1.0})
    assert len(observer.events) == 2 and observer.truncated == 1
    assert all(event["inclusive"] for event in observer.events)
    assert not observer.timing.token_ids


def test_prepared_history_timing_does_not_count_prefill_or_claim_generated_answers():
    task = {"forced_tokens": [3, 4], "prompt_tokens": 128, "repetitions": 1}
    row = {"forced_tokens": [3, 4], "output_kind": "forced_not_generated",
        "prepared_history_position": 128, "committed_position": 130,
        "actual_batch_sizes": [1], "actual_prefill_tokens": 128,
        "synchronized_decode_forward_seconds": [.01, .02]}
    result = {"identity": {"measurement_kind": "synchronized_prepared_history_forwards",
                           "output_fixture": [3, 4]}, "measurements": [row],
        "ownership": {"active_handles": 0, "reserved_bytes": 0, "allocated_bytes": 0}}
    validate_controlled_result(result, task)
    for key, value in (("prepared_history_position", 0), ("committed_position", 128),
                       ("forced_tokens", [5, 6]), ("output_kind", "generated")):
        changed = copy.deepcopy(result)
        changed["measurements"][0][key] = value
        with pytest.raises(ValueError):
            validate_controlled_result(changed, task)


def test_known_failed_kernel_is_reported_unavailable_before_native_worker_launch():
    task = {"operators": {"attention": "mlx_sdpa", "projection": "reference"}}
    reason = unavailable_operator(task, {"architecture": "qwen3", "quantization": {"bits": 8}})
    assert "state parity failed" in reason
    assert unavailable_operator({"operators": {"attention": "reference"}},
                                 {"architecture": "qwen3"}) is None


def test_external_comparison_requires_complete_identical_continuation():
    from qualification.p01_external import validate_peer_pair

    _, _, _, parity = fixture()
    parity["native_parity"]["forced_tokens"] = list(range(128))
    parity["correctness_passed"] = True
    parity["identity"].update(environment={}, tokenizer_identity="tokenizer", context_limit=4096)
    peer = copy.deepcopy(parity)
    peer["identity"].update(measurement_kind="synchronized_prepared_history_forwards",
                            output_fixture=list(range(128)))
    assert validate_peer_pair(peer, parity) is peer["identity"]
    for change in (lambda p: p["native_parity"].update(forced_tokens=[1, 2, 3, 4]),
                   lambda p: p["identity"].update(tokenizer_identity="different"),
                   lambda p: p["identity"].update(context_limit=8192),
                   lambda p: p.update(correctness_passed=False)):
        altered = copy.deepcopy(parity)
        change(altered)
        with pytest.raises(ValueError):
            validate_peer_pair(peer, altered)


def test_numerical_payload_is_checksummed_and_tampering_fails_closed(tmp_path):
    from mlx_one.inference_qualification import validate_integrity
    from qualification.p01_math import write_integrity as write_math_integrity

    payload = tmp_path / "logits.npz"
    payload.write_bytes(b"fixed numerical fixture")
    (tmp_path / "experiment.json").write_text('{"status":"PASS"}')
    write_math_integrity(tmp_path)
    assert "logits.npz" in validate_integrity(tmp_path, required={"logits.npz"})
    payload.write_bytes(b"altered numerical fixture")
    with pytest.raises(ValueError, match="stale"):
        validate_integrity(tmp_path, required={"logits.npz"})
