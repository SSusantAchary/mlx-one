"""Qualification must distinguish missing evidence, exact dtype and profile roles."""

import importlib.util
import os
from pathlib import Path

import pytest


def selector():
    path = Path(__file__).resolve().parents[2] / "scripts/qualification/operation_selection.py"
    spec = importlib.util.spec_from_file_location("correction_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("selected", ["float16", "bfloat16", "float32"])
def test_exact_dtype_buckets_never_overlap(selected):
    module = selector()
    records = [
        {"id": f"dot-{d}-False-None-gemm", "input_dtype": d, "status": "PASS"}
        for d in ["float16", "bfloat16", "float32"]
    ]
    assert [r["input_dtype"] for r in module.select_operations(records, selected, None)] == [
        selected
    ]


def test_missing_metadata_cannot_match():
    with pytest.raises(ValueError, match="missing"):
        selector().select_operations([{"id": "unknown-operation"}], "float16", None)


def test_empty_probe_selection_is_not_pass():
    assert selector().operation_verdict([]) == "NOT_RUN"


def test_legacy_dtype_is_parsed_as_whole_field():
    records = [
        {"id": "dot-bfloat16-False-8-gemm", "bits": 8, "status": "FAIL"},
        {"id": "dot-float16-False-8-gemm", "bits": 8, "status": "PASS"},
    ]
    assert selector().select_operations(records, "float16", 8) == [records[1]]


def test_conflicting_metadata_and_duplicate_probes_rejected():
    with pytest.raises(ValueError, match="conflict"):
        selector().operation_dtype({"id": "dot-bfloat16-False-8-gemm", "input_dtype": "float16"})
    with pytest.raises(ValueError, match="duplicate"):
        selector().select_operations([{"id": "rms-float16-False"}] * 2, "float16", None)


def contract():
    path = Path(__file__).resolve().parents[2] / "scripts/qualification/numerical_contract_v2.py"
    spec = importlib.util.spec_from_file_location("v2_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cross_profile_or_kv_comparisons_cannot_masquerade_as_parity():
    identity = {
        "checkpoint_revision": "a" * 40,
        "weight_profile": "affine_q8",
        "kv_policy": "natural",
        "input_ids_sha256": "f" * 64,
    }
    contract().require_profile_alignment(identity, dict(identity))
    for changed in [
        {"kv_policy": "f16"},
        {"weight_profile": "stored"},
        {"checkpoint_revision": "b" * 40},
    ]:
        with pytest.raises(ValueError, match="mismatch"):
            contract().require_profile_alignment(identity, {**identity, **changed})


def test_missing_dimensions_and_empty_independent_controls_are_blocking():
    assert contract().evidence_status([]) == "NOT_RUN"
    with pytest.raises(ValueError, match="six"):
        contract().evaluate({"Q1": "PASS"})
    dims = {f"Q{i}": "PASS" for i in range(1, 7)}
    dims["Q3"] = "NOT_RUN"
    assert contract().evaluate(dims)["outcome"] == "BLOCKED_INSUFFICIENT_EVIDENCE"


def test_proposed_contract_never_grants_alpha_support_or_waives_p1():
    dims = {f"Q{i}": "PASS" for i in range(1, 7)}
    result = contract().evaluate(dims)
    assert result["outcome"] == "PROVISIONAL_NUMERICAL_PASS"
    assert result["alpha_supported"] is False and result["historical_frozen_overridden"] is False
    dims["Q4"] = "NEEDS_DISPOSITION"
    assert contract().evaluate(dims)["outcome"] == "PENDING_NUMERICAL_DISPOSITION"
    dims["Q3"] = "FAIL"
    assert (
        contract().evaluate(dims, reliability_disposition="APPROVED")["outcome"]
        == "FAIL_MANDATORY_EVIDENCE"
    )


def test_probability_metrics_are_shift_invariant_and_reject_bad_alignment(monkeypatch):
    import numpy as np

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "benchmarks/model_parity")
    )
    from numerical_metrics import distribution_metrics

    left = np.array([1.0, 2.0, 3.0])
    right = left + 9
    m = distribution_metrics(left, right, 2, "float32")
    assert m["max_centered_logit_error"] == 0 and m["kl_nats"] == 0 and m["js_nats"] == 0
    assert m["top1_agreement"] and m["nll_difference"] == 0
    assert (
        abs(
            m["mass_accounting"]["head_reference_mass"]
            + m["mass_accounting"]["tail_reference_mass"]
            - 1
        )
        < 1e-12
    )
    with pytest.raises(ValueError):
        distribution_metrics(left, right[:2], 2, "float32")
    with pytest.raises(ValueError):
        distribution_metrics(left, np.array([1.0, np.nan, 3.0]), 2, "float32")


def test_proposed_logit_pass_cannot_replace_required_frozen_reference_pass():
    assert (
        contract().frozen_reference_verdict([{"status": "FAIL", "proposed_status": "PASS"}])
        == "FAIL"
    )
    assert (
        contract().frozen_reference_verdict([{"status": "PASS"}], complete_scope=False)
        == "NOT_ESTABLISHED"
    )


def test_reference_guard_resolves_the_pinned_local_snapshot():
    source = Path(__file__).resolve().parents[2] / "benchmarks/model_parity/four_path_oracle.py"
    spec = importlib.util.spec_from_file_location("reference_guard_configuration", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A checkout is optional developer data, not a prerequisite for a clean clone.
    assert module.load_reference.__globals__["ROOT"] == Path(__file__).resolve().parents[2]


def test_exact_logit_ties_use_actual_argmax_token_identity(monkeypatch):
    import numpy as np

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "benchmarks/model_parity")
    )
    from numerical_metrics import distribution_metrics

    m = distribution_metrics(np.array([3.0, 3.0, 1.0]), np.array([3.0, 3.0, 2.0]), 0, "float32")
    assert m["top1_ids"] == [0, 0] and m["top1_agreement"]


@pytest.mark.skipif(os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1", reason="real Metal")
def test_independent_attention_accepts_actual_bf16_additive_mask(monkeypatch):
    import mlx.core as mx

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "benchmarks/model_parity")
    )
    from precision_operations import attention_control

    assert mx.metal.is_available()
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        mask = mx.array(
            [
                [0, -float("inf"), -float("inf"), -float("inf")],
                [0, 0, -float("inf"), -float("inf")],
                [0, 0, 0, -float("inf")],
                [-float("inf"), 0, 0, 0],
            ],
            dtype=mx.bfloat16,
        )
        q = mx.zeros((1, 2, 4, 16), dtype=mx.bfloat16)
        k = mx.zeros((1, 1, 4, 16), dtype=mx.bfloat16)
        v = mx.broadcast_to(
            mx.arange(4, dtype=mx.float32)[None, None, :, None], (1, 1, 4, 16)
        ).astype(mx.bfloat16)
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.25, mask=mask)
        result = attention_control(q, k, v, output, mask, 0.25, label="bf16-additive-mask")
        assert all(row["status"] == "PASS" for row in result["comparisons"].values())
    finally:
        mx.set_default_device(previous)
