"""Incremental memory calibration and complete execution-envelope matching."""
from dataclasses import replace

import pytest

from mlx_one.engine.envelope import ENVELOPE_VERSION, QualificationEnvelope, profile_envelope
from mlx_one.inference_baseline import CHECKPOINTS, BaselineProfile
from mlx_one.peak_calibration import summarize_peaks


def test_incremental_peak_subtracts_persistent_growth_and_uses_max_plus_margin():
    samples = [{"temperature": phase, "active_before": 1000, "active_after": 1300,
                "peak_active": peak, "cached_before": 20, "cached_after": 30,
                "persistent_before": 100, "persistent_after": 300}
               for phase in ("cold", "warm") for peak in (1400, 1600, 1500)]
    result = summarize_peaks(samples)
    assert result["observed_max_incremental_bytes"] == 400
    assert result["incremental_bytes"] == 480
    assert all(s["persistent_growth"] == 200 for s in result["samples"])
    samples[0]["peak_active"] = None
    with pytest.raises(ValueError, match="incomplete"):
        summarize_peaks(samples)
    with pytest.raises(ValueError, match="three"):
        summarize_peaks(samples[1:])


def test_envelope_preserves_dictionary_api_but_legacy_is_diagnostic():
    item = BaselineProfile(*CHECKPOINTS["qwen"], "paged", workload="mixed", slots=2,
                           context_tokens=8192)
    envelope = profile_envelope(item)
    assert envelope.schema_version == ENVELOPE_VERSION
    assert QualificationEnvelope.from_mapping(envelope.to_dict()) == envelope
    with pytest.raises(ValueError, match="current"):
        QualificationEnvelope.from_mapping({"context_tokens": 8192, "slots": 2})
    changed = profile_envelope(replace(item, max_scheduled_tokens=4096))
    assert changed.execution_bounds() != envelope.execution_bounds()
    different_traffic = profile_envelope(replace(item, workload="turnover"))
    assert different_traffic.execution_bounds() == envelope.execution_bounds()
    assert different_traffic.workload != envelope.workload


def test_legacy_evidence_is_readable_but_cannot_be_promoted(tmp_path):
    import json
    from dataclasses import asdict

    from mlx_one.engine.selection import load_evidence
    from mlx_one.registry import (
        InferenceFeatureRegistry,
        InferenceFeatureStatus,
        InferenceScope,
        RegistryError,
    )

    scope = InferenceScope("test", "a" * 40, "paged", "topology", "float16", "float16",
                           "0.32", "test-hardware", "source")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"schema_version": "native-inference-qualification-v1",
        "backend": "mlx-one-native", "feature": "paged_attention", "scope": asdict(scope),
        "correctness_passed": True, "performance_qualified": True, "status": "passed"}))
    registry = load_evidence([path])
    assert registry.report("paged_attention", scope)["status"] == "experimental"
    with pytest.raises(RegistryError):
        InferenceFeatureRegistry().promote("paged_attention", scope, path,
                                          InferenceFeatureStatus.PERFORMANCE_QUALIFIED)
