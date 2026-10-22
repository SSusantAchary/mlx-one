import copy

import pytest

from qualification.p01_campaign import MODELS, inventory, run_steps


def test_shared_campaign_preserves_numeric_order_and_qualified_repetition_floor():
    steps = inventory(MODELS)
    assert [s["workload"] for s in steps] == ["parity"] * 2 + ["release"] * 2 + ["profile"] * 2 + [
        "engine"] * 2
    assert all(s["independent_runs"] == 2 for s in steps)
    assert all(s["repetitions"] >= 10 and s["routes"] == ["reference", "optimized"]
               for s in steps if s["workload"] == "engine")
    for step in steps:
        if step["workload"] in {"profile", "engine"}:
            prereqs = [s for s in steps if s["id"] in step["depends"]]
            assert {s["workload"] for s in prereqs} == {"parity", "release"}
            assert all(s["checkpoint"] == step["checkpoint"] for s in prereqs)


@pytest.mark.parametrize("models,widths,repeats", (
    ([], [128], 10), (["lfm2"], [128], 10), ([MODELS[0]], [128], 5),
    ([MODELS[0]], [8193], 10), ([MODELS[0]] * 2, [128], 10),
    ([MODELS[0]], [128, 128], 10)))
def test_invalid_shared_campaign_cannot_expand_or_relax_gates(models, widths, repeats):
    with pytest.raises(ValueError):
        inventory(models, widths, repetitions=repeats)


def test_model_failure_skips_dependent_benchmarks_without_preventing_peer_checks():
    steps, executed, snapshots = inventory(MODELS), [], []
    def execute(step):
        executed.append((step["checkpoint"], step["workload"]))
        return {"status": "FAIL" if step["checkpoint"] == MODELS[0] else "PASS"}
    status = run_steps(steps, execute, validate=lambda: None,
                       persist=lambda: snapshots.append(copy.deepcopy(steps)))
    assert status == "FAIL" and snapshots
    assert (MODELS[0], "engine") not in executed and (MODELS[1], "engine") in executed
    assert all(s["status"] == "SKIPPED" for s in steps
               if s["checkpoint"] == MODELS[0] and s["workload"] != "parity")


@pytest.mark.parametrize("failure", ("telemetry", "source changed", "worker watchdog"))
def test_environment_failure_stops_native_expansion(failure):
    steps, executed = inventory(MODELS), []
    def validate():
        raise RuntimeError(failure)
    status = run_steps(steps, lambda s: executed.append(s), validate=validate, persist=lambda: None)
    assert status == "BLOCKED" and not executed
    assert steps[0]["status"] == "BLOCKED" and steps[0]["reason"] == failure
    assert all(s["status"] == "NOT_RUN" for s in steps[1:])
