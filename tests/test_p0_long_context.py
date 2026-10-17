"""Backend-free long-context profiles, measured scores, and failure handling."""
import json

import pytest

from qualification import p0_long_context as long


def metadata(context=32768):
    return {'model_id': 'test/model', 'revision': 'a' * 40, 'configured_context': context}


def test_exact_lengths_never_silently_truncate_and_maximum_legal_is_separate():
    rows = long.inventory([metadata()], maximum_legal=True)
    assert [(r['prompt_tokens'], r['status']) for r in rows] == [
        (16384, 'QUEUED'), (32768, 'NOT_SUPPORTED'), (32640, 'QUEUED')]
    assert rows[2]['context_class'] == 'maximum_legal_32k'
    assert rows[2]['requested_prompt_tokens'] == 32768
    assert all(r['output_token_budget'] == 128 for r in rows)


def test_gpt2_has_no_long_context_substitute():
    rows = long.inventory([metadata(1024)], maximum_legal=True)
    assert len(rows) == 2 and all(r['status'] == 'NOT_SUPPORTED' for r in rows)


@pytest.mark.parametrize('context', [None, 0, True])
def test_missing_or_invalid_context_metadata_rejected(context):
    with pytest.raises(ValueError, match='context limit'):
        long.inventory([metadata(context)])


def result(count=5, status='PASS', **fields):
    return {'cases': [{'case_id': 'H16384', 'status': status, **fields}],
            'benchmarks': [{'time_to_first_token_ms': (i + 1) * 1000,
                'decode_tokens_per_second': (i + 1) * 10, 'mlx_peak_bytes': (i + 1) * 100,
                'generated_tokens': 128} for i in range(count)]}


def test_scores_require_five_complete_measurements_and_keep_null_metrics():
    case = long.inventory([metadata()])[0]
    row = long.performance(case, result(), {'status': 'completed'})
    assert row['status'] == 'PASS' and row['measured_repetitions'] == 5
    assert row['decode_tps'] == 30 and row['ttft_p50_seconds'] == 3
    assert row['ttft_p95_seconds'] == 5 and row['peak_mlx_bytes'] == 500
    assert row['prefill_tps'] is row['peak_rss_bytes'] is None
    assert not row['qualification_promoted']
    incomplete = long.performance(case, result(count=3), {'status': 'completed'})
    assert incomplete['status'] == 'BLOCKED' and incomplete['decode_tps'] is None


@pytest.mark.parametrize('status', ['worker_failed', 'request_timeout',
                                   'memory_watchdog_unavailable', 'memory_watchdog_terminated'])
def test_supervision_failure_never_becomes_a_performance_pass(status):
    row = long.performance(long.inventory([metadata()])[0], result(), {'status': status})
    assert row['status'] == 'BLOCKED' and row['decode_tps'] is None


def test_capacity_rejection_has_no_inference_score():
    row = long.performance(long.inventory([metadata()])[0], result(count=0, status='FAIL',
        error_type='CacheCapacityError', reason='conservative headroom'), {'status': 'completed'})
    assert row['status'] == 'REJECTED_ADMISSION'
    assert row['reason'] == 'conservative headroom' and row['peak_mlx_bytes'] is None


def test_rss_peaks_belong_to_their_own_requests():
    rows = [{'started_monotonic': 101, 'finished_monotonic': 103},
            {'started_monotonic': 104, 'finished_monotonic': 105}]
    samples = [{'elapsed_seconds': 0, 'rss_bytes': 999},
               {'elapsed_seconds': 2, 'rss_bytes': 123}]
    long.attribute_rss(rows, samples, 100)
    assert rows[0]['rss_peak_bytes'] == 123
    assert rows[1]['rss_peak_bytes'] is None


def test_asset_mutation_rejected_before_native_initialization(monkeypatch):
    original = {**metadata(), 'snapshot_path': '/unused', 'assets': {'weight': 'old'}}
    monkeypatch.setattr(long, 'validate_assets', lambda *_args: {
        'assets': {'weight': 'different'}, 'configured_context': 32768})
    with pytest.raises(ValueError, match='assets changed'):
        long.verify_assets(original)


def test_readable_and_machine_scores_retain_rejection(tmp_path):
    rows = long.inventory([metadata(1024)])
    long.export(tmp_path, {'schema_version': long.VERSION}, rows)
    assert 'NOT_SUPPORTED' in (tmp_path / 'scores.md').read_text()
    assert (tmp_path / 'scores.json').exists() and (tmp_path / 'scores.csv').exists()


def test_incomplete_baseline_blocks_before_gpu_preflight(tmp_path, monkeypatch):
    source = {'source_sha256': 'a' * 64}
    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    for name, value in {
            'environment.json': {'source': source, 'policy': {}},
            'download_manifest.json': {'model': {'status': 'PASS', 'metadata': metadata(40960)}},
            'summary.json': {'decision': 'NO_GO'}}.items():
        (baseline / name).write_text(json.dumps(value))
    monkeypatch.setattr(long, 'source_provenance', lambda *_args: source)
    monkeypatch.setattr(long, 'validate_integrity', lambda *_args: None)
    monkeypatch.setattr(long, 'preflight', lambda *_args: pytest.fail('GPU started before gates'))
    output = tmp_path / 'results'
    assert long.main(['--baseline', str(baseline), '--output', str(output)]) == 1
    run = next((output / 'runs').iterdir())
    assert {r['status'] for r in json.loads((run / 'scores.json').read_text())} == {'BLOCKED'}
