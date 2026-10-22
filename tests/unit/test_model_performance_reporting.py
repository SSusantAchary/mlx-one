"""Public benchmark integrity checks; no MLX, models or network required."""

import copy
import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "performance_report", ROOT / "benchmarks/model_performance/report.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class PerformanceReportingTests(unittest.TestCase):
    def setUp(self):
        self.inventory, self.records = report.load_records()
        self.samples = copy.deepcopy(self.records[0]["samples"])

    def validate(self, samples=None):
        report.validate_samples(self.samples if samples is None else samples, 128, 16, 3)

    def test_all_primary_historical_rows_are_published(self):
        historical = [
            r for r in self.records if r["environment"]["protocol"] == "historical-128-16"
        ]
        self.assertEqual(len(historical), 16)
        primary = {r["model"]["id"] for r in historical if r["primary"]}
        self.assertEqual(primary, {"M01", "M02", "M03", "M04", "M06", "M07", "M08", "M09", "M10"})
        markdown = report.render(self.inventory, self.records)
        for record in self.records:
            self.assertIn(f"]({record['path']})", markdown)
        self.assertIn("Standardized 128/128 coverage:", markdown)

    def test_planned_models_have_no_invented_rows(self):
        self.assertEqual(len(self.inventory), 13)
        self.assertFalse(
            any(
                r["model"]["id"] in {"M05", "M11", "M12", "M13"}
                for r in self.records
                if r["environment"]["protocol"] == "historical-128-16"
            )
        )

    def test_readme_block_is_current_and_after_supported_table(self):
        text = (ROOT / "README.md").read_text()
        table = report.render(self.inventory, self.records)
        self.assertIn(table, text)
        preceding = text[: text.index(report.START)].rstrip().splitlines()[-1]
        self.assertTrue(preceding.startswith("| [microsoft/Phi-3"))

    def test_actual_count_must_equal_token_ids(self):
        self.samples[0]["generated_tokens"] = 15
        with self.assertRaisesRegex(ValueError, "token count"):
            self.validate()

    def test_requested_budget_cannot_replace_actual_outputs(self):
        self.samples[0]["token_ids"] = self.samples[0]["token_ids"][:8]
        with self.assertRaises(ValueError):
            self.validate()

    def test_actual_prefill_count_is_checked(self):
        self.samples[0]["input_tokens"] = 127
        with self.assertRaisesRegex(ValueError, "input count"):
            self.validate()

    def test_missing_repetitions_rejected(self):
        with self.assertRaisesRegex(ValueError, "repetitions"):
            self.validate(self.samples[:2])

    def test_nonfinite_and_negative_metrics_rejected(self):
        for value in (math.nan, math.inf, -1, True):
            with self.subTest(value=value):
                samples = copy.deepcopy(self.samples)
                samples[0]["decode_tokens_per_second"] = value
                with self.assertRaisesRegex(ValueError, "invalid metric"):
                    self.validate(samples)

    def test_prefill_must_use_measured_duration(self):
        self.samples[0]["prefill_tokens_per_second"] *= 2
        with self.assertRaisesRegex(ValueError, "prefill rate"):
            self.validate()

    def test_decode_must_use_actual_steps_excluding_first_token(self):
        s = self.samples[0]
        s["decode_seconds"] = s["generated_tokens"] / s["decode_tokens_per_second"]
        with self.assertRaisesRegex(ValueError, "actual decode steps"):
            self.validate()
        s["decode_seconds"] = (s["generated_tokens"] - 1) / s["decode_tokens_per_second"]
        self.validate()

    def test_first_token_eos_has_no_decode_throughput(self):
        for s in self.samples:
            s.update(
                generated_tokens=1, token_ids=[1], decode_seconds=0.001, decode_tokens_per_second=0
            )
        self.validate()
        r = dict(self.records[0], samples=self.samples)
        self.assertIn("| — |", report.render(self.inventory, [r]))

    def test_failed_cleanup_rejected(self):
        self.samples[0]["cleanup"] = False
        with self.assertRaisesRegex(ValueError, "cleanup"):
            self.validate()

    def test_historical_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / "historical").mkdir()
            (evidence / "inventory.json").write_bytes(
                (report.EVIDENCE / "inventory.json").read_bytes()
            )
            cohort = json.loads((report.EVIDENCE / "historical/cohort.json").read_text())
            (evidence / "historical/cohort.json").write_text(json.dumps(cohort))
            (evidence / "historical/runner.py.txt").write_text("changed driver")
            with self.assertRaisesRegex(ValueError, "runner changed"):
                report.load_records(evidence)

    def test_standard_records_reject_interface_revision_hardware_and_count_tampering(self):
        cases = [
            ("interface", lambda d: d["environment"].update(measurement_type="python-engine")),
            ("revision", lambda d: d.update(revision="0" * 40)),
            ("hardware", lambda d: d["environment"]["hardware"].update(chip_type="")),
            ("count", lambda d: d["samples"][0].update(generated_tokens=127)),
            ("speed", lambda d: d["samples"][0].update(decode_tokens_per_second=99999)),
        ]
        for label, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                evidence = root / "docs/benchmarks"
                shutil.copytree(report.EVIDENCE, evidence)
                support = root / "src/mlx_one/data/compatibility/alpha_release.json"
                support.parent.mkdir(parents=True)
                shutil.copyfile(ROOT / "src/mlx_one/data/compatibility/alpha_release.json", support)
                record = evidence / "runs/M01-direct-128-128.json"
                data = json.loads(record.read_text())
                mutate(data)
                record.write_text(json.dumps(data))
                with patch.object(report, "ROOT", root), self.assertRaises(ValueError):
                    report.load_records(evidence)

    def test_current_standard_record_preserves_its_exact_runner(self):
        data = json.loads((report.EVIDENCE / "runs/M01-direct-128-128.json").read_text())
        self.assertEqual(
            report.sha(report.EVIDENCE / "runs/M01-runner.py.txt"), data["runner_sha256"]
        )

    def test_inventory_revision_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            data = json.loads((report.EVIDENCE / "inventory.json").read_text())
            data["checkpoints"][0]["revision"] = "0" * 40
            (evidence / "inventory.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "inventory drifted"):
                report.load_records(evidence)


if __name__ == "__main__":
    unittest.main()
