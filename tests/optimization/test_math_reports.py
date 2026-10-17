import json

from qualification.p01_reports import collect, coverage_catalog, record_failure


def test_complete_declared_matrix_starts_pending_without_invented_scores():
    rows = coverage_catalog()
    assert len(rows) == 4 * 5 * 5 * 4
    assert {row["prompt_tokens"] for row in rows} == {128, 512, 2048, 4096, 8192}
    assert {row["chunk"] for row in rows} == {None, 256, 512, 1024, 2048}
    assert all(row["native_parity"] == "NOT_RUN" and row["performance"] == "PENDING"
               and row["release"] == "NOT_RUN" and not row["artifacts"] for row in rows)


def test_missing_native_evidence_never_promotes_cpu_readiness(tmp_path):
    report = collect(tmp_path, {"source_sha256": "current"})
    assert report["qualification_promoted"] is False
    assert all(row["native_parity"] == "NOT_RUN" for row in report["rows"])


def test_historical_source_is_readable_without_qualifying_current_source(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"source": {"source_sha256": "old"},
        "status": "completed", "tasks": [{"status": "PASS"}]}))
    report = collect(tmp_path, {"source_sha256": "current"})
    assert len(report["historical_cohorts"]) == 1
    assert all(row["native_parity"] == "NOT_RUN" for row in report["rows"])


def test_unverified_pass_cannot_enter_native_matrix(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"source": {"source_sha256": "current"},
        "status": "completed", "tasks": [{"checkpoint": "mlx-community/Qwen3-0.6B-8bit",
        "prompt_tokens": 128, "chunk": None, "operators": {"attention": "reference",
        "projection": "reference"}, "route": "optimized", "workload": "parity",
        "status": "PASS", "artifact": str(tmp_path / "missing") }]}))
    report = collect(tmp_path, {"source_sha256": "current"})
    assert report["rejected_artifacts"]
    assert all(row["native_parity"] == "NOT_RUN" for row in report["rows"])


def test_timing_and_release_failures_do_not_get_mislabeled_as_numerical_failures():
    row = coverage_catalog()[0]
    row["native_parity"] = "PASS"
    record_failure(row, {"status": "FAIL", "workload": "engine", "reason": "memory leak"})
    assert row["performance"] == "FAIL" and row["native_parity"] == "PASS"
    record_failure(row, {"status": "BLOCKED", "workload": "release", "suites": ["streaming"]})
    assert row["release"] == "BLOCKED" and row["release_suites"]["streaming"] == "BLOCKED"
    assert row["release_suites"]["lifecycle"] == "NOT_RUN" and row["native_parity"] == "PASS"
    assert len(row["failures"]) == 2


def test_unavailable_operator_is_visible_without_a_fabricated_pass():
    row = coverage_catalog()[0]
    record_failure(row, {"status": "NOT_SUPPORTED", "workload": "parity", "reason": "gate"})
    assert row["native_parity"] == "NOT_SUPPORTED" and row["performance"] == "PENDING"
