"""Opt-in actual cold/warm/cancel/cleanup/repeat/independent native 8K regressions."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

if os.environ.get('MLX_ONE_RUN_H8192_CHECKPOINTS') != '1':
    pytest.skip('set MLX_ONE_RUN_H8192_CHECKPOINTS=1 on native Metal', allow_module_level=True)

from mlx_one.inference_baseline import source_provenance
from mlx_one.p0_engine_qualification import MODELS


@pytest.mark.parametrize('model_id', [m for m in MODELS if m != 'openai-community/gpt2'])
def test_native_cold_warm_cancel_cleanup_repeat_and_independent(model_id):
    recorded = os.environ.get('MLX_ONE_H8192_COHORT')
    if recorded:
        # A completed supervised cohort can supply the expensive executed cases.
        root = Path(recorded).resolve()
    else:
        command = [sys.executable, 'benchmarks/p0_engine_qualification.py', '--offline',
                   '--model', model_id, '--suite', 'admission', '--output',
                   'qualification/results/p0_engine/h8192-checkpoint-regressions']
        process = subprocess.run(command, text=True, capture_output=True, timeout=14400)
        assert process.returncode == 0, process.stdout + process.stderr
        root = Path(process.stdout.strip().splitlines()[-1]).resolve()
    environment = json.loads((root / 'environment.json').read_text())
    assert environment['source']['source_sha256'] == source_provenance(Path.cwd())['source_sha256']
    assert environment['preflight']['status'] == 'PASS'
    rows = json.loads((root / 'qualification_results.json').read_text())
    cases = [r for r in rows if r['model_id'] == model_id and r['suite'] == 'admission']
    assert {r['case_id'] for r in cases} == {f'I{i:02}' for i in range(1, 7)}
    assert all(r['status'] == 'PASS' for r in cases), cases
    by_id = {r['case_id']: r for r in cases}
    assert by_id['I01']['artifact'] != by_id['I06']['artifact']
    workers = json.loads((root / 'worker_evidence.json').read_text())
    selected = [w for w in workers if w['model_id'] == model_id
                and Path(w['path']).parent.name in {'admission', 'admission_independent'}]
    assert len(selected) == 2
    for worker in selected:
        path = Path(worker['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == worker['sha256']
        for name, checksum in json.loads(path.read_text()).items():
            assert hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == checksum
        supervision = json.loads((path.parent / 'supervision.json').read_text())
        assert supervision['status'] == 'completed' and supervision['rss_samples']
