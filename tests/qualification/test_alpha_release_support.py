"""Release metadata cannot promote architecture declarations into certification."""

import sys

from mlx_one.core.release_support import checkpoint_release_status


def test_unknown_revision_and_registered_architecture_remain_experimental():
    before = set(sys.modules)
    report = checkpoint_release_status("Qwen/Qwen3-0.6B", "unqualified", architecture="qwen3")
    assert report["architecture_registered"]
    assert report["status"] == "EXPERIMENTAL_NOT_QUALIFIED"
    assert report["alpha_supported"] is False
    assert report["release_approval"] == "PENDING"
    assert not any(n == "mlx" or n.startswith("mlx.") for n in set(sys.modules) - before)


def test_qwen35_is_retained_and_never_certified():
    report = checkpoint_release_status(
        "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", architecture="qwen3_5"
    )
    assert report["candidate_profile"] == "affine_q4"
    assert report["alpha_supported"] is False


def test_gemma_is_admitted_by_functional_evidence_and_large_models_are_architecture_only():
    report = checkpoint_release_status(
        "google/gemma-3-1b-it", "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        architecture="gemma3_text")
    assert report["status"] == "EXPERIMENTAL_SUPPORTED"
    assert report["numerical_contract"] == "OPTIONAL_RESEARCH_NOT_AN_ADMISSION_GATE"
    report = checkpoint_release_status(
        "mlx-community/gpt-oss-20b-MXFP4-Q4", "f356f2747216d7e98fee755df25987459fc19089",
        architecture="gpt_oss")
    assert report["status"] == report["architecture_status"] == "ARCHITECTURE_IMPLEMENTED"
    assert report["checkpoint_status"] == "NOT_DOWNLOADED_NOT_SMOKE_TESTED"
    wrong = checkpoint_release_status(
        "mlx-community/gpt-oss-20b-MXFP4-Q4", "f356f2747216d7e98fee755df25987459fc19089",
        architecture="lfm2_moe")
    assert wrong["status"] == "EXPERIMENTAL_NOT_QUALIFIED"
