"""Identity regressions use tiny files; none of these tests is hardware evidence."""

import hashlib
from dataclasses import replace

import pytest
from test_inference_baseline import identity, profile

from mlx_one.engine.kernel_dispatch import sdpa_capability
from mlx_one.inference_baseline import BaselineError, validate_identity
from mlx_one.text.loading import TextModelLoadError, checkpoint_identity


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "models--org--model" / "snapshots" / ("a" * 40)
    root.mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"test weights, not a model")
    files = {
        "model.safetensors": hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()
    }
    return root, files


def test_canonical_identity_local_path_revision_and_verified_hashes_are_separate(snapshot):
    root, files = snapshot
    assert checkpoint_identity(
        root, "org/model", revision=root.name, explicit_path=True, expected_files=files
    ) == ("org/model", root.name, files)
    assert checkpoint_identity(root, str(root)) == ("org/model", root.name, {})


@pytest.mark.parametrize("mismatch", ("model", "revision", "weight", "extra_file"))
def test_checkpoint_identity_rejects_mismatch_before_model_execution(snapshot, mismatch):
    root, files = snapshot
    model, revision = "org/model", root.name
    if mismatch == "model":
        model = "org/other"
    elif mismatch == "revision":
        revision = "b" * 40
    elif mismatch == "weight":
        (root / "model.safetensors").write_bytes(b"modified")
    else:
        (root / "config.json").write_text("{}")
    with pytest.raises(TextModelLoadError, match="match|verification"):
        checkpoint_identity(
            root, model, revision=revision, explicit_path=True, expected_files=files
        )


def test_arbitrary_local_copy_requires_content_proof(tmp_path):
    with pytest.raises(TextModelLoadError, match="requires file hashes"):
        checkpoint_identity(tmp_path, "org/model", explicit_path=True)


def test_remote_symbolic_revision_resolves_to_snapshot_without_relabeling(snapshot):
    root, _ = snapshot
    assert checkpoint_identity(root, "org/model", revision="main") == ("org/model", root.name, {})
    with pytest.raises(TextModelLoadError, match="revision mismatch"):
        checkpoint_identity(root, "org/model", revision="main", explicit_path=True)


def test_conversion_identity_is_not_replaced_by_its_hosting_repository(snapshot):
    root, files = snapshot
    converted = {"source": "original/model", "artifact_digest": "b" * 64}
    assert checkpoint_identity(root, "original/model", revision=converted["artifact_digest"],
        converted=converted, explicit_path=True, expected_files=files) == (
            "original/model", converted["artifact_digest"], files)


def test_fixed_output_policy_is_private_to_the_qualification_harness():
    from types import SimpleNamespace

    from mlx_one.text.generation import _qualification_eos_ids

    assert _qualification_eos_ids({2, 3}) == {2, 3}
    with pytest.raises(ValueError, match="qualification harness"):
        _qualification_eos_ids({2, 3}, ignore_eos=True)
    assert not _qualification_eos_ids({2, 3}, ignore_eos=True,
                                     runtime=SimpleNamespace(harness_only=True))
    with pytest.raises(BaselineError, match="exact P1"):
        replace(profile(), eos_policy="ignore_for_fixed_workload")


@pytest.mark.parametrize("field", (None, "canonical_model_id", "path", "revision", "weight_hashes"))
def test_benchmark_identity_checks_canonical_path_revision_and_weights(snapshot, field):
    root, files = snapshot
    item = replace(profile(), checkpoint_path=str(root), checkpoint_files=files)
    runtime = identity(item)
    proof = {
        "canonical_model_id": item.model_id,
        "path": str(root),
        "revision": item.revision,
        "weight_hashes": files,
    }
    if field:
        proof[field] = "mismatched"
    runtime["inference"]["checkpoint_identity"] = proof
    if field:
        with pytest.raises(BaselineError, match="checkpoint path|hashes"):
            validate_identity(item, runtime)
    else:
        validate_identity(item, runtime)


@pytest.mark.parametrize(
    "queries,heads,kv,expected",
    (
        (7, 8, 1, "reference"),
        (8, 8, 2, "mlx_sdpa"),
        (9, 8, 1, "mlx_sdpa"),
        (32, 8, 1, "mlx_sdpa"),
        (4096, 8, 2, "mlx_sdpa"),
    ),
)
def test_vector_gqa_limit_is_not_generalized_to_full_attention(queries, heads, kv, expected):
    decision = sdpa_capability(
        q_shape=(1, heads, queries, 64),
        k_shape=(1, kv, queries + 19, 64),
        v_shape=(1, kv, queries + 19, 64),
        mlx_version=(0, 32, 2),
    )
    assert decision.selected == expected
    assert ("vector" if queries <= 8 else "full") in decision.reason
