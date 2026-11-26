"""Backend-free negative evidence, strictness and request isolation tests."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pytest

from mlx_one.mlxboost.native.dispatcher import choose
from mlx_one.mlxboost.native.registry import Registry, write_immutable
from mlx_one.mlxboost.native.schemas import (
    KERNEL_ID,
    KernelCandidate,
    KernelRequest,
    NativeConfig,
    NativeKernelUnavailable,
)

REQUEST = KernelRequest(
    chip="Apple M4",
    gpu_architecture="applegpu_g16g",
    os_version="test",
    mlx_build_sha256="a" * 64,
    shape=(16,),
    strides=(1,),
)
STATUS = dict(
    loadable=True, implementation_sha256="b" * 64, binary_sha256="c" * 64, mlx_build_sha256="a" * 64
)
ENABLED = NativeConfig(True)


def fixture(tmp_path):
    profile = dict(
        status="measured",
        samples_ms=[1.0, 1.0, 1.0],
        sample_count=3,
        median_ms=1.0,
        timing_scope="synchronized_wall",
        methodology=dict(synchronized=True),
    )
    proof = dict(
        schema_version=1,
        request=json.loads(json.dumps(asdict(REQUEST))),
        implementation_sha256="b" * 64,
        binary_sha256="c" * 64,
        reference_profile=copy.deepcopy(profile),
        results=[
            dict(
                threadgroup_size=128,
                correct=True,
                stable=True,
                no_regression=True,
                profile=profile,
                max_abs_error=0.0,
            )
        ],
    )
    path = write_immutable(tmp_path, proof)
    entry = dict(
        request=proof["request"],
        implementation_sha256="b" * 64,
        binary_sha256="c" * 64,
        numerical_status="pass",
        performance_status="qualified",
        approved_for_runtime=True,
        reviewed_by="test fixture; no real qualification",
        candidate=asdict(
            KernelCandidate("axpby-128", KERNEL_ID, "1.0.0", "b" * 64, "c" * 64, 128, path.stem)
        ),
    )
    manifest = tmp_path / "qualified.json"
    manifest.write_text(json.dumps(dict(schema_version=1, candidates=[entry])))
    return manifest, entry, path


def decide(registry=None, request=REQUEST, config=ENABLED, status=STATUS):
    return choose(request, config, registry, status_provider=lambda: status)


def test_disabled_never_probes_or_reads_evidence():
    def forbidden():
        raise AssertionError("backend probe on disabled path")

    assert choose(REQUEST, NativeConfig(), status_provider=forbidden).reason == "native_disabled"
    assert (
        choose(REQUEST, NativeConfig(False, True), status_provider=forbidden).reason
        == "native_disabled"
    )


def test_exact_qualified_selection_and_no_runtime_writes(tmp_path):
    manifest, _, _ = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    selected = decide(Registry(manifest))
    assert selected.backend == "mlxboost_native" and selected.threadgroup_size == 128
    assert selected.scope == "demo_operator_only"
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("approved_for_runtime", False, "unapproved_evidence"),
        ("approved_for_runtime", 1, "unapproved_evidence"),
        ("reviewed_by", "", "unapproved_evidence"),
        ("performance_status", "experimental", "regression_blocked"),
        ("numerical_status", "fail", "failed_parity"),
        ("binary_sha256", "d" * 64, "stale_evidence"),
        ("implementation_sha256", "d" * 64, "stale_evidence"),
    ],
)
def test_bad_entry_falls_back(tmp_path, key, value, reason):
    manifest, entry, _ = fixture(tmp_path)
    entry[key] = value
    manifest.write_text(json.dumps(dict(schema_version=1, candidates=[entry])))
    assert decide(Registry(manifest)).fallback_reason == reason


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"dtype": "bfloat16"}, "unsupported_dtype"),
        ({"shape": (-1,)}, "unsupported_shape"),
        ({"shape": (2**32,)}, "unsupported_shape"),
        ({"strides": (2,)}, "unsupported_layout"),
        ({"layout": "strided"}, "unsupported_layout"),
        ({"quantization": "q4"}, "unsupported_quantization"),
        ({"group_size": 64}, "unsupported_quantization"),
        ({"chip": "unknown"}, "unsupported_chip"),
        ({"mlx_version": "0.33.0"}, "mlx_version_mismatch"),
        ({"mlx_metal_version": "0.32.1"}, "mlx_version_mismatch"),
        ({"mlx_build_sha256": "other"}, "mlx_version_mismatch"),
        ({"stream_device": "cpu"}, "unsupported_stream"),
        ({"alpha": float("nan")}, "unsupported_parameters"),
        ({"execution_mode": "grad"}, "reference_selected"),
        ({"model_architecture": "qwen3"}, "reference_selected"),
        ({"chip": "Apple M3"}, "scope_mismatch"),
        ({"os_version": "changed"}, "scope_mismatch"),
        ({"beta": 0.5}, "scope_mismatch"),
    ],
)
def test_request_scope_rejected_and_strict(tmp_path, change, reason):
    manifest, _, _ = fixture(tmp_path)
    request = replace(REQUEST, **change)
    assert decide(Registry(manifest), request).fallback_reason == reason
    with pytest.raises(NativeKernelUnavailable, match=reason):
        decide(Registry(manifest), request, NativeConfig(True, True))


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "null",
        "[]",
        '{"schema_version":9}',
        '{"schema_version":1,"candidates":{}}',
        '{"schema_version":1,"candidates":[null]}',
    ],
)
def test_malformed_json_fails_closed(tmp_path, payload):
    manifest = tmp_path / "bad.json"
    manifest.write_text(payload)
    assert decide(Registry(manifest)).reason == "invalid_manifest"


def test_corrupt_or_missing_proof_and_path_traversal(tmp_path):
    manifest, entry, proof = fixture(tmp_path)
    original = proof.read_bytes()
    proof.write_text("{}")
    assert decide(Registry(manifest)).reason == "stale_evidence"
    proof.unlink()
    assert decide(Registry(manifest)).reason == "invalid_manifest"
    proof.write_bytes(original)
    entry["candidate"]["evidence_sha256"] = "../private"
    manifest.write_text(json.dumps(dict(schema_version=1, candidates=[entry])))
    assert decide(Registry(manifest)).reason == "invalid_manifest"


def test_symlink_evidence_outside_manifest_directory_rejected(tmp_path):
    manifest, _, proof = fixture(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    moved = outside / proof.name
    proof.rename(moved)
    proof.symlink_to(moved)
    assert decide(Registry(manifest)).reason == "invalid_manifest"


def test_empty_registry_missing_extension_and_abi_strict():
    assert decide().reason == "missing_evidence"
    for reason in ("extension_unavailable", "unsupported_platform", "mlx_version_mismatch"):
        assert decide(status=dict(loadable=False, reason=reason)).reason == reason
        with pytest.raises(NativeKernelUnavailable, match=reason):
            decide(config=NativeConfig(True, True), status=dict(loadable=False, reason=reason))


def test_concurrent_decisions_do_not_share_configuration(tmp_path):
    manifest, _, _ = fixture(tmp_path)
    registry = Registry(manifest)
    with ThreadPoolExecutor(8) as pool:
        results = list(
            pool.map(lambda n: decide(registry, config=NativeConfig(bool(n % 2))), range(80))
        )
    assert [r.backend for r in results] == [
        "mlxboost_native" if n % 2 else "mlx_reference" for n in range(80)
    ]


def test_config_and_immutable_records(tmp_path):
    assert NativeConfig.from_env({}) == NativeConfig()
    assert NativeConfig.from_env({"MLX_ONE_NATIVE": "1"}).enabled
    with pytest.raises(ValueError):
        NativeConfig.from_env({"MLX_ONE_NATIVE": "yes"})
    with pytest.raises(TypeError):
        NativeConfig(enabled=1)
    path = write_immutable(tmp_path, {"b": 2, "a": 1})
    assert write_immutable(tmp_path, {"a": 1, "b": 2}) == path
    before = path.read_bytes()
    write_immutable(tmp_path, {"a": 2})
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        write_immutable(tmp_path, {"a": float("nan")})


def test_failed_candidate_is_never_ranked_even_if_faster():
    from mlx_one.mlxboost.native.autotune import rank_results

    good = dict(
        correct=True,
        stable=True,
        no_regression=True,
        threadgroup_size=128,
        profile={"median_ms": 2},
    )
    bad = copy.deepcopy(good)
    bad.update(correct=False, threadgroup_size=64, profile={"median_ms": 0.001})
    assert rank_results([bad, good]) == [good]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda p: p["results"][0].update(correct=False), "regression_blocked"),
        (lambda p: p["results"][0].update(max_abs_error=-1), "missing_evidence"),
        (lambda p: p["results"][0]["profile"].update(samples_ms=[0, 0, 0]), "missing_evidence"),
        (lambda p: p["results"][0]["profile"].update(median_ms=0.001), "missing_evidence"),
        (lambda p: p["results"][0]["profile"].update(timing_scope="enqueue"), "missing_evidence"),
        (
            lambda p: p["reference_profile"].update(median_ms=0.5, samples_ms=[0.5] * 3),
            "regression_blocked",
        ),
        (
            lambda p: p["results"][0]["profile"].update(samples_ms=[0.1, 1.0, 9.0]),
            "regression_blocked",
        ),
    ],
)
def test_recomputed_hash_does_not_bypass_numerical_or_timing_gates(tmp_path, mutation, reason):
    manifest, entry, proof_path = fixture(tmp_path)
    proof = json.loads(proof_path.read_text())
    mutation(proof)
    new_path = write_immutable(tmp_path, proof)
    entry["candidate"]["evidence_sha256"] = new_path.stem
    manifest.write_text(json.dumps(dict(schema_version=1, candidates=[entry])))
    assert decide(Registry(manifest)).reason == reason


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":1,"schema_version":1,"candidates":[]}',
        '{"schema_version":true,"candidates":[]}',
        '{"schema_version":1,"candidates":[],"value":NaN}',
    ],
)
def test_ambiguous_json_is_rejected(tmp_path, text):
    path = tmp_path / "qualified.json"
    path.write_text(text)
    assert decide(Registry(path)).reason == "invalid_manifest"
