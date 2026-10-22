"""Simulated evidence tests exercise admission only; they never qualify native execution."""
import hashlib
import json
import xml.etree.ElementTree as ET
from copy import deepcopy

import pytest

from mlx_one import native_correctness as contract
from mlx_one import native_qualification as driver
from mlx_one.inference_qualification import write_integrity


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    source = {"source_sha256": "source", "native_test_sha256": "tests",
              "git_revision": "baseline", "packages": {"mlx": "0.32.2"}}
    hardware = {"chip": "simulated", "memory_bytes": 32 * 1024**3}
    asset = {"checkpoint": "qwen3-small", "model_id": "test-only", "revision": "pinned",
             "files": {"model.safetensors": "checksum"}}
    monkeypatch.setattr(driver, "checkpoint_proof", lambda *args: deepcopy(asset))
    outcomes = {}
    for paged in (False, True):
        directory = tmp_path / str(paged)
        directory.mkdir()
        log = b"6 passed\n"
        (directory / "worker.log").write_bytes(log)
        driver.write(directory / "supervision.json", {"status": "completed",
            "worker_log_sha256": hashlib.sha256(log).hexdigest()})
        driver.write(directory / "command.json", {
            "argv": contract.correctness_command("qwen3-small", paged, directory, "python"),
            "source": {k: v for k, v in source.items() if k != "native_test_sha256"}})
        suite = ET.Element("testsuite", tests="6", skipped="0", errors="0", failures="0")
        for node in contract.test_nodes("qwen3-small", paged):
            ET.SubElement(suite, "testcase", name=node.split("::")[-1],
                          classname="tests.test_p1_checkpoint_parity", time="0.1")
        ET.ElementTree(suite).write(directory / "junit.xml")
        mode = contract.execution_mode("qwen3-small", paged)
        row = {"within_tolerance": True, "greedy_agreement": True,
               "max_error": 0.0, "mean_error": 0.0}
        for page in (16, 32):
            for size in (1, 2, 4):
                parity = {"status": "passed", "execution_mode": mode, "batch_size": size,
                    "runtime": {"scope": {"execution_path": mode, "revision": "pinned",
                                            "mlx_version": "0.32.2"}},
                    "forced_steps": 32, "atol": 1e-2, "rtol": 1e-2,
                    "test_identity": contract.parity_identity(
                        "qwen3-small", paged, page, size, asset),
                    "greedy_agreement_fraction": 1, "prefix_isolation_passed": True,
                    "seeded_rng_isolation": True,
                    "cleanup": {"active_handles": 0, "reserved_bytes": 0},
                    "comparisons": [row], "component_comparisons": [row],
                    "prefix_restoration": [row]}
                driver.write(directory / f"parity-qwen3-small-{paged}-{page}-{size}.json", parity)
        write_integrity(directory)
        outcomes[f"qwen3-small:{paged}"] = {"status": "PASS", "path": str(directory)}
    proof = {"schema_version": driver.CORRECTNESS_VERSION, "source": source,
             "hardware": hardware, "outcomes": outcomes,
             "assets": {"qwen3-small": deepcopy(asset)}}
    path = tmp_path / "correctness.json"
    driver.write(path, proof)
    write_integrity(tmp_path)
    return path, proof, source, hardware


def validate(evidence, **kwargs):
    path, _, source, hardware = evidence
    return driver.validate_correctness(path, ["qwen3-small"], source=source, hardware=hardware,
                                       **kwargs)


def save(evidence):
    path, proof, _, _ = evidence
    driver.write(path, proof)
    write_integrity(path.parent)


@pytest.mark.parametrize("status", ["FAIL", "DEFERRED", "HARDWARE_PENDING", "NOT_RUN",
                                    "UNSUPPORTED", "BLOCKED"])
def test_dense_qualification_is_independent_but_status_never_becomes_pass(evidence, status):
    evidence[1]["outcomes"]["qwen3-small:True"] = {"status": status}
    save(evidence)
    assert validate(evidence, backends=["reference_dense", "experimental_dense_batch"])
    with pytest.raises(ValueError, match="incomplete"):
        validate(evidence, backends=["paged"])
    with pytest.raises(ValueError, match="incomplete"):
        validate(evidence)  # Conservative legacy omission still requires both.


def test_paged_qualification_does_not_require_dense_pass(evidence):
    evidence[1]["outcomes"].pop("qwen3-small:False")
    save(evidence)
    assert validate(evidence, backends=["paged"])


@pytest.mark.parametrize("field", ["git_revision", "packages", "native_test_sha256"])
def test_source_software_and_test_revision_cannot_be_borrowed(evidence, field):
    evidence[1]["source"] = {**evidence[2], field: "changed"}
    save(evidence)
    with pytest.raises(ValueError, match="source"):
        validate(evidence)


def test_changed_checkpoint_is_rejected_even_after_resealing(evidence):
    evidence[1]["assets"]["qwen3-small"]["files"]["model.safetensors"] = "changed"
    save(evidence)
    with pytest.raises(ValueError, match="checkpoint"):
        validate(evidence)


@pytest.mark.parametrize("change", ["command", "skip", "duplicate", "backend", "tolerance",
                                    "numerical", "nan", "cleanup", "identity", "missing"])
def test_forged_pass_rejected_after_resealing_artifacts(evidence, change):
    directory = evidence[0].parent / "False"
    if change == "command":
        path = directory / "command.json"
        data = json.loads(path.read_text())
        data["argv"] = ["python", "-c", "print('6 passed')"]
        driver.write(path, data)
    elif change in {"skip", "duplicate"}:
        path = directory / "junit.xml"
        tree = ET.parse(path)
        tests = list(tree.getroot().iter("testcase"))
        if change == "skip":
            ET.SubElement(tests[0], "skipped")
        else:
            tests[0].set("name", tests[1].get("name"))
        tree.write(path)
    else:
        path = directory / "parity-qwen3-small-False-16-1.json"
        data = json.loads(path.read_text())
        if change == "backend":
            data["execution_mode"] = "paged"
        elif change == "tolerance":
            data["atol"] = 100
        elif change == "numerical":
            data["component_comparisons"][0]["within_tolerance"] = False
        elif change == "nan":
            data["comparisons"][0]["max_error"] = float("nan")
        elif change == "cleanup":
            data["cleanup"]["reserved_bytes"] = 1
        elif change == "identity":
            data["test_identity"]["attention_backend"] = "mlx_fused"
        else:
            data["comparisons"] = []
        driver.write(path, data)
    write_integrity(directory)
    with pytest.raises(ValueError):
        validate(evidence, backends=["experimental_dense_batch"])


def test_log_mutation_is_detected(evidence):
    (evidence[0].parent / "False/worker.log").write_text("6 passed, altered\n")
    with pytest.raises(ValueError, match="stale"):
        validate(evidence)


@pytest.mark.parametrize("backends", [[], ["mlx_fused"], ["auto"]])
def test_unknown_attention_or_execution_backend_cannot_borrow_dense_evidence(backends):
    with pytest.raises(ValueError):
        contract.backend_requirements(backends)


def test_selected_manifest_has_no_paged_tasks_or_dependencies():
    manifest = driver.command_manifest(["qwen3-small"], backends=["experimental_dense_batch"])
    assert len(manifest["tasks"]) > 0
    assert all(t["case"]["mode"] != "paged" for t in manifest["tasks"])
    assert "BLOCKED" in manifest["status_schema"]
    reference = driver.command_manifest(["qwen3-small"], backends=["reference_dense"])
    assert reference["tasks"]
    assert all(t["case"]["mode"] == "reference_dense" for t in reference["tasks"])


def test_phase0_baseline_contains_all_five_required_workloads():
    manifest = driver.baseline_manifest(["qwen3-small"])
    assert {(t["case"]["prompt_tokens"], t["case"]["slots"]) for t in manifest["tasks"]} == {
        (512, 1), (2048, 1), (4096, 1), (2048, 2), (2048, 4)}
    assert all(t["case"]["output_tokens"] == 128 for t in manifest["tasks"])
    assert all(t["settings"]["exact_prompt_tokens"] for t in manifest["tasks"])
    assert manifest["repetitions"] == 5 and manifest["warmup_requests"] == 1


def test_checkpoint_manifest_hashes_actual_files_and_does_not_download(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(contract, "profile", lambda *a, **k: SimpleNamespace(revision="pinned"))
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (tmp_path / name).write_text("fixture")
    first = contract.checkpoint_proof("qwen3-small", tmp_path)
    (tmp_path / "model.safetensors").write_text("mutated")
    second = contract.checkpoint_proof("qwen3-small", tmp_path)
    assert first["files"] != second["files"]


def test_cli_retains_failure_and_continues_independent_dense_checkpoint(tmp_path, monkeypatch):
    from mlx_one import p1_profiles

    source = {"source_sha256": "source", "native_test_sha256": "tests"}
    monkeypatch.setattr(driver, "preflight", lambda *a, **k: {
        "source": source, "hardware": {"chip": "test"}, "process_ceiling_bytes": 1024})
    monkeypatch.setattr(p1_profiles, "asset_inventory", lambda: {})
    monkeypatch.setattr(driver, "checkpoint_proof", lambda *a: {"fixture": True})
    calls = []
    def worker(command, directory, **kwargs):
        calls.append(command)
        return {"status": "completed"}
    monkeypatch.setattr(driver, "supervised_command", worker)
    def validate_row(directory, checkpoint, paged, **kwargs):
        if paged:
            raise ValueError("inherited paged mismatch")
    monkeypatch.setattr(driver, "validate_native_row", validate_row)
    assert driver.main(["correctness", "--checkpoints", "qwen3-small", "qwen3-large",
                        "--output", str(tmp_path)]) == 1
    assert len(calls) == 4
    proof = json.loads(next(tmp_path.rglob("correctness.json")).read_text())
    assert proof["outcomes"]["qwen3-large:False"]["status"] == "PASS"
    assert proof["outcomes"]["qwen3-small:True"]["status"] == "FAIL"


def test_exact_benchmark_prompts_include_template_and_suffix_in_the_token_count():
    from types import SimpleNamespace

    from mlx_one.inference_baseline_worker import build_exact_messages

    bundle = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text: [ord(c) for c in text]),
        chat_template=SimpleNamespace(render=lambda messages, **kwargs:
                                      "<user>" + messages[0]["content"] + "</user>"))
    for count in (512, 2048, 4096):
        messages, rendered = build_exact_messages(bundle, count, suffix="\nRequest 2.")
        assert len(bundle.tokenizer.encode(rendered)) == count
        assert messages[0]["content"].endswith("\nRequest 2.")


def test_unrepresentable_exact_prompt_fails_instead_of_truncating():
    from types import SimpleNamespace

    from mlx_one.inference_baseline import BaselineError
    from mlx_one.inference_baseline_worker import build_exact_messages

    bundle = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text: [0] * (2 * len(text))),
        chat_template=SimpleNamespace(render=lambda messages, **kwargs: messages[0]["content"]))
    with pytest.raises(BaselineError, match="cannot construct exact"):
        build_exact_messages(bundle, 513)


@pytest.mark.parametrize("actual_output", (127, 128))
def test_exact_baseline_rejects_early_eos_as_a_128_token_measurement(monkeypatch, actual_output):
    import asyncio
    from types import SimpleNamespace

    from mlx_one import inference_baseline_worker as worker
    from mlx_one.inference_baseline import BaselineError, BaselineProfile

    monkeypatch.setattr(worker, "build_exact_messages", lambda *a, **k: ([], "exact"))
    async def stream(*args, **kwargs):
        return {"status": "completed", "generated_tokens": actual_output}
    monkeypatch.setattr(worker, "stream_request", stream)
    bundle = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text: list(range(512))),
                             context_length=40960)
    item = BaselineProfile("test", "a" * 40, "reference_dense", context_tokens=512,
                           output_tokens=128, exact_prompt_tokens=True)
    replay = {"requests": [{"request_id": "a", "arrival_ms": 0,
                            "prompt_tokens": 512, "output_tokens": 128}]}
    def execute():
        return asyncio.run(worker.replay_group(None, bundle, SimpleNamespace(), item, replay,
                                               group="test", phase="warm", token_metadata=[]))
    if actual_output == 127:
        with pytest.raises(BaselineError, match="output length"):
            execute()
    else:
        assert execute()[0]["generated_tokens"] == 128
