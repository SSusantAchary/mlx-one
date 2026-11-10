"""Support claims require unchanged native execution receipts, not accuracy relabeling."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import CHECKPOINT_CATALOG, CHECKPOINT_RESULTS, ROOT, verified_checkpoint, verified_result


class SupportEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = dict(
            model_type="example",
            checkpoint="org/model",
            revision="a" * 40,
            status="PASS_FUNCTIONAL",
            quality_status="FAIL_DIAGNOSTIC",
        )
        self.path = self.root / "result.json"
        self.path.write_text(json.dumps(self.result))
        self.claim = dict(
            model_type="example",
            checkpoint="org/model",
            revision="a" * 40,
            evidence="result.json",
            evidence_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
        )

    def test_failed_quality_diagnostic_is_preserved_for_functional_support(self):
        result = verified_result(self.root, self.claim, lambda p: None)
        self.assertEqual(result["quality_status"], "FAIL_DIAGNOSTIC")

    def test_changed_receipt_is_rejected(self):
        self.path.write_text(json.dumps({**self.result, "status": "FAIL"}))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verified_result(self.root, self.claim, lambda p: None)

    def test_a_hash_does_not_promote_a_failed_probe(self):
        self.path.write_text(json.dumps({**self.result, "status": "FAIL"}))
        self.claim["evidence_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "identity or execution mismatch"):
            verified_result(self.root, self.claim, lambda p: None)

    def test_a_different_checkpoint_cannot_reuse_a_passing_receipt(self):
        self.claim["checkpoint"] = "other/checkpoint"
        with self.assertRaisesRegex(ValueError, "identity or execution mismatch"):
            verified_result(self.root, self.claim, lambda p: None)

    def test_evidence_path_cannot_escape_repository(self):
        self.claim["evidence"] = "../result.json"
        with self.assertRaisesRegex(ValueError, "leaves the repository"):
            verified_result(self.root, self.claim, lambda p: None)


class CurrentCheckpointEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def profile(self, index=0):
        claim = json.loads((ROOT / CHECKPOINT_CATALOG).read_text())["models"][index]
        saved = json.loads((ROOT / CHECKPOINT_RESULTS).read_text())["models"][index]
        for key in ("evidence", "functional_decision"):
            if key in claim:
                path = self.root / claim[key]["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / claim[key]["path"]).read_bytes())
        return claim, saved

    def test_failed_functional_gate_is_rejected(self):
        claim, saved = self.profile()
        saved["gates"]["G3"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "incomplete functional gate"):
            verified_checkpoint(self.root, claim, saved, lambda p: None)

    def test_changed_checkpoint_revision_is_rejected(self):
        claim, saved = self.profile()
        claim["revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "differs from its saved result"):
            verified_checkpoint(self.root, claim, saved, lambda p: None)

    def test_changed_native_receipt_is_rejected(self):
        claim, saved = self.profile()
        (self.root / claim["evidence"]["path"]).write_text('{"status": "PASS"}')
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verified_checkpoint(self.root, claim, saved, lambda p: None)

    def test_vit_needs_its_explicit_functional_decision(self):
        claim, saved = self.profile(5)
        del claim["functional_decision"]
        del saved["functional_decision"]
        with self.assertRaisesRegex(ValueError, "needs its recorded functional decision"):
            verified_checkpoint(self.root, claim, saved, lambda p: None)

    def test_vit_decision_cannot_accept_a_failed_classification_check(self):
        claim, saved = self.profile(5)
        path = self.root / claim["evidence"]["path"]
        native = json.loads(path.read_text())
        native["checks"]["logits"]["status"] = "FAIL"
        path.write_text(json.dumps(native))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        claim["evidence"]["sha256"] = saved["evidence"]["sha256"] = digest
        with self.assertRaisesRegex(ValueError, "unaccepted failure"):
            verified_checkpoint(self.root, claim, saved, lambda p: None)
