"""Opt-in real-weight P0 state regressions; every model subprocess remains supervised."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

if os.environ.get('MLX_ONE_RUN_P0_CHECKPOINTS') != '1':
    pytest.skip('set MLX_ONE_RUN_P0_CHECKPOINTS=1 on native Metal', allow_module_level=True)

from mlx_one.p0_engine_qualification import MODELS


@pytest.mark.parametrize('model_id', MODELS)
def test_pinned_reference_logits_complete_state_and_cleanup(model_id):
    output = Path(os.environ.get('MLX_ONE_P0_RESULTS_DIR',
                                 'qualification/results/p0_engine/checkpoint-regressions'))
    command = [sys.executable, 'benchmarks/p0_engine_qualification.py', '--offline',
               '--suite', 'state', '--model', model_id, '--output', str(output)]
    process = subprocess.run(command, text=True, capture_output=True, timeout=4800)
    assert process.returncode == 0, process.stdout + process.stderr
    run = Path(process.stdout.strip().splitlines()[-1])
    rows = json.loads((run / 'qualification_results.json').read_text())
    applicable = [r for r in rows if r['model_id'] == model_id and r['suite'] in {
        'startup', 'state'} and r['mandatory'] and r.get('applicable', True)]
    assert len(applicable) == 34
    assert all(r['status'] == 'PASS' for r in applicable), applicable
    assert json.loads((run / 'summary.json').read_text())['qualification_promoted'] is False
