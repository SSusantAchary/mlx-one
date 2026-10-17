import copy

import pytest
from test_math_experiments import identity

from qualification.p01_comparison import baseline_repeatability, promotion_gates


def measured(prefill=10, decode=20, peak=100):
    return {"identity": identity(), "correctness_passed": True,
            "measurements": [{"prefill_tps": prefill, "decode_tps": decode,
                              "peak_mlx_bytes": peak, "rss_peak_bytes": 200}] * 10,
            "independent_processes": 2, "execution_order": "ABBA", "thermal_noise_checked": True,
            "required_protected_profiles": ["b1_decode"]}


def test_no_universal_win_or_promotion_from_isolated_gain():
    reference, candidate = measured(), measured(prefill=12, decode=21)
    engineering = promotion_gates(reference, candidate, protected={"b1_decode": 1})
    assert engineering["gate_status"] == "PASS"
    assert engineering["qualification_promoted"] is False
    competitor = promotion_gates(reference, candidate, competitor=True, protected={"b1_decode": 1})
    assert not competitor["competitor_win"]
    assert competitor["gate_status"] == "FAIL"


def test_both_throughputs_memory_and_protected_routes_required_for_competitor_win():
    reference, candidate = measured(), measured(prefill=12, decode=23, peak=104)
    assert promotion_gates(reference, candidate, competitor=True,
                           protected={"b1_decode": .97})["competitor_win"]
    assert not promotion_gates(reference, candidate, competitor=True,
                               protected={"b1_decode": .96})["competitor_win"]
    candidate["measurements"] = [{**row, "peak_mlx_bytes": 106}
                                  for row in candidate["measurements"]]
    assert not promotion_gates(reference, candidate, competitor=True,
                               protected={"b1_decode": 1})["competitor_win"]


def test_patch_installation_without_actual_engagement_is_insufficient():
    candidate = measured(prefill=12, decode=23, peak=100)
    candidate.update(optimized_competitor=True, engaged_patch_calls=0)
    report = promotion_gates(measured(), candidate, competitor=True, protected={"b1_decode": 1})
    assert report["gate_status"] == "PENDING"
    assert any("patch" in reason for reason in report["reasons"])


def test_missing_matrix_numerics_noise_or_telemetry_stay_pending():
    candidate = measured(prefill=12, decode=23)
    assert promotion_gates(measured(), candidate)["gate_status"] == "PENDING"
    candidate["correctness_passed"] = False
    assert promotion_gates(measured(), candidate, protected={})["gate_status"] == "PENDING"
    candidate["correctness_passed"] = True
    candidate["thermal_noise_checked"] = False
    assert promotion_gates(measured(), candidate, protected={})["gate_status"] == "PENDING"


def test_repeatability_rejects_mismatched_tokens_and_undefined_intervals():
    left, right = measured(), measured(prefill=10.4)
    assert baseline_repeatability([left, right])["status"] == "PASS"
    right = measured(prefill=11)
    assert baseline_repeatability([left, right])["status"] == "PENDING"


def test_repeatability_validates_tokens_and_missing_decode_samples():
    left = measured()
    right = copy.deepcopy(left)
    right["identity"]["prompt_ids"] = [2, 3]
    with pytest.raises(ValueError, match="unmatched"):
        baseline_repeatability([left, right])
    right = measured()
    right["measurements"][0]["decode_tps"] = None
    assert baseline_repeatability([left, right])["status"] == "PENDING"


@pytest.mark.parametrize("matrix", ({}, {"unrelated": 1}, {"b1_decode": float("nan")},
                                    {"b1_decode": True}, {"b1_decode": 0}))
def test_empty_mismatched_and_invalid_protection_cannot_pass(matrix):
    assert promotion_gates(measured(), measured(prefill=12, decode=23),
                           protected=matrix)["gate_status"] == "PENDING"
