"""Replay identity and progress-based fairness use deterministic event clocks."""
import json

import pytest

from mlx_one.inference_baseline import CHECKPOINTS, BaselineProfile
from mlx_one.traffic_replay import (
    generated_traffic,
    load_traffic_manifest,
    progress_fairness,
    traffic_identity,
    validate_traffic,
)


def profile(**kwargs):
    return BaselineProfile(*CHECKPOINTS["qwen"], "paged", context_tokens=8192,
                           slots=4, workload="mixed", **kwargs)


def test_mixed_replay_is_identical_across_policy_and_has_actual_mixed_lengths(tmp_path):
    a = generated_traffic(profile(admission_policy="deadline"))
    b = generated_traffic(profile(admission_policy="cache_aware"))
    assert a == b
    assert {r["prompt_tokens"] for r in a["requests"]} == {2048, 8192}
    path = tmp_path / "traffic.json"
    path.write_text(json.dumps(a))
    assert load_traffic_manifest(path)["sha256"] == traffic_identity(a)
    a["requests"][0]["api_key"] = "credentials are forbidden"
    with pytest.raises(ValueError, match="invalid traffic"):
        validate_traffic(a)


def test_progress_detects_starvation_without_request_failure():
    events = [{"request_id": "a", "event": kind, "time": at} for kind, at in
              (("submitted", 0), ("admitted", 3), ("forward", 3.1), ("terminal", 3.2))]
    result = progress_fairness(events, max_queue_wait_ms=2000, max_service_gap_ms=1000)
    assert result["starved_requests"] == 1 and result["status"] == "failed"
    assert progress_fairness(events, max_queue_wait_ms=None,
                             max_service_gap_ms=None)["status"] == "diagnostic"


def test_backpressure_and_pressure_are_reported_separately():
    sequence = (("submitted", 0, None), ("blocked", .1, "pressure"),
                ("unblocked", 5, "pressure"), ("admitted", 5.1, None),
                ("forward", 5.2, None), ("blocked", 5.3, "backpressure"),
                ("unblocked", 40, "backpressure"), ("forward", 40.1, None),
                ("terminal", 40.2, None))
    events = [{"request_id": "a", "event": kind, "time": at, "reason": reason}
              for kind, at, reason in sequence]
    result = progress_fairness(events, max_queue_wait_ms=1000, max_service_gap_ms=1000)
    assert result["status"] == "passed"
    assert result["exclusions"]["pressure"] == result["exclusions"]["backpressure"] == 1


def test_sustained_feasible_arrivals_have_bounded_progress():
    events = []
    for index in range(200):
        start = index * .1
        events.extend({"request_id": str(index), "event": kind, "time": start + at}
                      for kind, at in (("submitted", 0), ("admitted", .02),
                                       ("forward", .04), ("terminal", .06)))
    assert progress_fairness(events, max_queue_wait_ms=50,
                             max_service_gap_ms=50)["status"] == "passed"


def test_policy_comparison_rejects_different_arrivals_and_exports_deltas(tmp_path):
    from mlx_one.inference_qualification import write_integrity
    from mlx_one.traffic_replay import compare_policy_runs

    paths = [tmp_path / "default", tmp_path / "locality"]
    for path, policy, wait in zip(paths, ("deadline", "cache_aware"), (10, 5), strict=True):
        path.mkdir()
        item = profile(admission_policy=policy)
        files = {"manifest.json": {"schema_version": "native-baseline-v1"},
                 "summary.json": {"profile": item.to_dict(), "identity": {"hardware": "fake"},
                                  "queue_wait_p95_ms": wait}, "runtime.json": {},
                 "traffic.json": generated_traffic(item), "tokenization.json": []}
        for name, value in files.items():
            (path / name).write_text(json.dumps(value))
        (path / "requests.jsonl").write_text("")
        write_integrity(path)
    result = compare_policy_runs(*paths)
    assert result["metrics"]["queue_wait_p95_ms"]["delta"] == -5
    file = paths[1] / "traffic.json"
    value = json.loads(file.read_text())
    value["requests"][0]["arrival_ms"] = 100
    file.write_text(json.dumps(value))
    write_integrity(paths[1])
    with pytest.raises(ValueError, match="identical arrivals"):
        compare_policy_runs(*paths)
