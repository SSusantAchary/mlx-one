"""Check evidence boundaries and reproducible dashboard generation without MLX."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

WEBSITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEBSITE))
from insights import CATALOGS, ROOT, classify, make_data, profiles


class StatusTests(unittest.TestCase):
    def test_blocked_diagnostics_never_become_supported(self):
        status, smoke, raw = classify('native tasks', {'decision': 'BLOCKED', 'functional_tested': True,
                                                     'source_parity': True, 'checkpoint_loaded': True})
        self.assertEqual((status, smoke, raw), ('blocked', False, 'BLOCKED'))

    def test_architecture_and_quantization_do_not_establish_checkpoint_smoke(self):
        for raw in ['ARCHITECTURE_IMPLEMENTED', 'QUANTIZATION_BACKEND_TESTED']:
            status, smoke, _ = classify('text', {'status': raw})
            self.assertEqual((status, smoke), ('architecture-only', False))

    def test_wip_keeps_scope_despite_smoke(self):
        status, smoke, _ = classify('audio', {'status': 'WIP', 'functional_smoke': 'PASS_SCOPED'})
        self.assertEqual((status, smoke), ('work-in-progress', True))

    def test_unknown_status_is_not_promoted(self):
        self.assertEqual(classify('text', {'status': 'NEW_UNREVIEWED_STATE'})[:2], ('unqualified', False))

    def test_absent_release_field_remains_unknown(self):
        row = profiles('audio', {'entries': [{'checkpoint': 'org/model', 'revision': 'abc', 'task': 'transcribe'}]})[0]
        self.assertIsNone(row['release_qualified'])

    def test_task_profiles_remain_distinct(self):
        rows = profiles('audio', {'entries': [{'checkpoint': 'org/model', 'revision': 'abc', 'task': task}
                                             for task in ['transcribe', 'synthesize']]})
        self.assertNotEqual(rows[0]['key'], rows[1]['key'])


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {}
        def record(path):
            cls.sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        cls.data = make_data(record)

    def test_counts_match_sources_and_history(self):
        data = self.data
        expected = sum(len(json.loads((ROOT / path).read_text())[key]) for path, key in CATALOGS.values())
        self.assertEqual(data['summary']['profiles'], expected)
        self.assertEqual(data['history']['points'][-1]['catalog_profiles'], expected)
        self.assertEqual(data['history']['points'][-1]['scoped_smoke_profiles'], data['summary']['scoped_smoke'])
        self.assertEqual(len({r['key'] for r in data['models']}), expected)

    def test_dates_are_bound_to_real_source_commits(self):
        for row in self.data['models']:
            self.assertIsNotNone(row['first_recorded'], row['model_id'])
            self.assertEqual(len(row['first_recorded']['commit']), 40)
            if row['first_smoke_recorded']:
                self.assertLessEqual(row['first_recorded']['date'], row['first_smoke_recorded']['date'])

    def test_missing_performance_is_not_zero(self):
        text = next(c for c in self.data['performance'] if c['id'] == 'text-historical')
        missing = [r for r in text['rows'] if r['value'] is None]
        self.assertEqual(len(missing), 4)
        self.assertFalse(any(r['value'] == 0 for r in missing))
        self.assertEqual(len([r for r in text['rows'] if r['value'] is not None]), 9)

    def test_cohorts_and_original_evidence_remain_separate(self):
        self.assertEqual(len({c['id'] for c in self.data['performance']}), 8)
        standard = next(c for c in self.data['performance'] if c['id'] == 'text-standard')
        self.assertEqual(len(standard['rows']), 1)
        for cohort in self.data['performance']:
            for row in cohort['rows']:
                if row['value'] is not None:
                    self.assertIn(row['evidence'], self.sources)
                    self.assertEqual(row['evidence_sha256'], self.sources[row['evidence']])

    def test_mlxboost_reuses_flag_cohorts_without_deduplicating_requests(self):
        first, second, third = self.data['mlxboost']['rows']
        self.assertEqual([r['samples_per_flag'] for r in [first, second, third]], [50, 50, 6])
        self.assertEqual(first['paired']['exact_output_matches'], 50)
        self.assertAlmostEqual(first['off'], 22.93530117487263)
        self.assertAlmostEqual(first['on'], 109.17109486891428)

    def test_generated_json_has_no_nonfinite_numbers(self):
        json.dumps(self.data, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
