"""Release validation must reject changed proof and missing approval."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def validator(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[2] / "scripts/release/validate.py"
    spec = importlib.util.spec_from_file_location("alpha_release_validator_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    payload = tmp_path / "raw.json"
    payload.write_text('{"status":"FAIL"}')
    (tmp_path / "evidence-index.json").write_text(
        json.dumps({"files": {"raw.json": hashlib.sha256(payload.read_bytes()).hexdigest()}})
    )
    gates = {
        "source_commit": "a" * 40,
        "production_source_sha256": {},
        "decision": "NO-GO",
        "numerical_contract_approval": "PENDING",
        "release_owner_approval": "PENDING",
    }
    (tmp_path / "release-gate-summary.json").write_text(json.dumps(gates))
    return module, tmp_path, gates


def test_changed_evidence_is_rejected(validator):
    module, root, _ = validator
    (root / "raw.json").write_text('{"status":"PASS"}')
    with pytest.raises(ValueError, match="changed evidence"):
        module.evidence(root, root)


def test_changed_packaged_ui_is_rejected(validator):
    module, root, gates = validator
    ui = root / "index.html"
    ui.write_text("audited UI")
    gates["package_source_sha256"] = {"index.html": hashlib.sha256(ui.read_bytes()).hexdigest()}
    (root / "release-gate-summary.json").write_text(json.dumps(gates))
    ui.write_text("different build")
    with pytest.raises(ValueError, match="index.html"):
        module.evidence(root, root)


def test_evidence_path_cannot_escape_cohort(validator):
    module, root, _ = validator
    (root / "evidence-index.json").write_text(json.dumps({"files": {"../outside": "0" * 64}}))
    with pytest.raises(ValueError, match="unsafe evidence path"):
        module.evidence(root, root)


def test_go_label_still_requires_release_owner_approval(validator):
    module, root, gates = validator
    gates["decision"] = "GO — Restricted alpha"
    (root / "release-gate-summary.json").write_text(json.dumps(gates))
    with pytest.raises(ValueError, match="release approval missing"):
        module.evidence(root, root, publication=True)


def test_missing_gate_coverage_rejects_publication(validator):
    module, root, gates = validator
    gates.update(
        decision="GO — Restricted alpha",
        numerical_contract_approval="APPROVED",
        release_owner_approval="APPROVED",
        mandatory_gates=[],
    )
    (root / "release-gate-summary.json").write_text(json.dumps(gates))
    with pytest.raises(ValueError, match="gate coverage"):
        module.evidence(root, root, publication=True)
