"""Deletion ordering and ownership boundaries for real-checkpoint qualification."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/qualification"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "release_campaign", SCRIPTS / "model_release_campaign.py"
)
campaign = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(campaign)


def setup_run(tmp_path, monkeypatch, *, model_type="gpt2"):
    work, evidence = tmp_path / "work", tmp_path / "evidence"
    work.mkdir()
    payload = b"bounded-test-artifact"
    entry = dict(
        model_type=model_type,
        kind="text",
        repo="test/checkpoint",
        revision="a" * 40,
        metadata_status="PASS",
        files=[
            dict(
                path="model.safetensors",
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        ],
    )
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([entry]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--plan",
            str(plan),
            "--work",
            str(work),
            "--evidence",
            str(evidence),
            "--python",
            sys.executable,
        ],
    )
    monkeypatch.setattr(campaign.os, "environ", dict(campaign.os.environ))
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "model.safetensors").write_bytes(payload)
    monkeypatch.setattr(campaign, "PROTECTED", protected)
    import huggingface_hub

    def fake_download(repo, filename, *, revision, local_dir):
        assert repo == "test/checkpoint" and revision == "a" * 40
        (local_dir / filename).write_bytes(payload)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return work, evidence, protected, payload


def test_results_are_durable_before_owned_weights_are_removed(tmp_path, monkeypatch):
    work, evidence, protected, payload = setup_run(tmp_path, monkeypatch)

    def worker(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        campaign.atomic(
            output, dict(model_type="gpt2", status="PASS_FUNCTIONAL", release_qualified=False)
        )
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(campaign.subprocess, "run", worker)
    original = campaign.shutil.rmtree
    removed = []

    def remove(path):
        assert json.loads((evidence / "gpt2.json").read_text())["status"] == "PASS_FUNCTIONAL"
        assert json.loads((evidence / "gpt2-cleanup.json").read_text())[
            "results_recorded_before_cleanup"
        ]
        assert path.parent == work / "models" and (path / ".campaign-owned").is_file()
        removed.append(path)
        original(path)

    monkeypatch.setattr(campaign.shutil, "rmtree", remove)
    campaign.main()
    assert len(removed) == 1 and not removed[0].exists()
    assert (protected / "model.safetensors").read_bytes() == payload
    assert json.loads((evidence / "gpt2-cleanup.json").read_text())["cleanup_complete"]


def test_failed_verification_is_recorded_before_partial_download_cleanup(tmp_path, monkeypatch):
    work, evidence, _, _ = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        campaign, "verify", lambda *a: (_ for _ in ()).throw(ValueError("bad digest"))
    )
    campaign.main()
    result = json.loads((evidence / "gpt2.json").read_text())
    assert result["stage"] == "verify" and result["error"] == "bad digest"
    assert not (work / "models/gpt2").exists()


def test_existing_unowned_download_directory_is_never_adopted(tmp_path, monkeypatch):
    work, _, _, payload = setup_run(tmp_path, monkeypatch)
    existing = work / "models"
    existing.mkdir()
    (existing / "keep").write_bytes(payload)
    with pytest.raises(RuntimeError, match="refuse to adopt"):
        campaign.main()
    assert (existing / "keep").read_bytes() == payload


def test_preserved_gpt_oss_weights_are_reused_without_mutation(tmp_path, monkeypatch):
    work, evidence, protected, payload = setup_run(tmp_path, monkeypatch, model_type="gpt_oss")
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *a, **k: pytest.fail("downloaded preserved weights"),
    )

    def worker(command, **kwargs):
        assert Path(command[command.index("--model") + 1]) == protected
        campaign.atomic(
            Path(command[command.index("--output") + 1]),
            dict(
                model_type="gpt_oss", status="FAIL", release_qualified=False, error="test failure"
            ),
        )
        return type("Completed", (), {"returncode": 1})()

    monkeypatch.setattr(campaign.subprocess, "run", worker)
    campaign.main()
    assert (protected / "model.safetensors").read_bytes() == payload
    assert not (work / "models/gpt_oss").exists()
    assert json.loads((evidence / "gpt_oss-assets.json").read_text())["preserved_existing"]
