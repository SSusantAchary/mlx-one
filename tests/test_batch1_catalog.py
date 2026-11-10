"""Current checkpoint discovery must preserve scope, pins and old diagnostics."""

from mlx_one.core.release_support import checkpoint_release_status
from mlx_one.engine.adapter_registry import checkpoint_task_profiles


def test_new_checkpoint_status_requires_exact_revision_and_architecture():
    revision = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
    exact = checkpoint_release_status("google-t5/t5-small", revision, architecture="t5")
    assert exact["status"] == "SUPPORTED"
    assert exact["primary_task"] == "text-to-text-generation"
    assert exact["release_approval"] == "NOT_REQUESTED"
    for pin, architecture in (("unverified", "t5"), (revision, "bert")):
        wrong = checkpoint_release_status("google-t5/t5-small", pin, architecture=architecture)
        assert wrong["status"] != "SUPPORTED"


def test_vit_current_task_replaces_old_plan_without_erasing_diagnostic():
    profiles = checkpoint_task_profiles("vit")
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile["functional_tested"] and profile["wheel_verified"]
    assert not profile["source_parity"]
    assert profile["numerical_status"] == "CLASSIFICATION_PASS_WITH_RETAINED_HIDDEN_DIAGNOSTIC_FAIL"
    assert not profile["release_qualified"]


def test_existing_bert_type_exposes_both_admitted_tasks():
    profiles = checkpoint_task_profiles("bert")
    assert {row["task"] for row in profiles} == {"embedding", "reranking"}
    assert len({(row["model_id"], row["revision"]) for row in profiles}) == 4
