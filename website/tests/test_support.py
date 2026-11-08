"""Support claims require unchanged native execution receipts, not accuracy relabeling."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import verified_result


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
