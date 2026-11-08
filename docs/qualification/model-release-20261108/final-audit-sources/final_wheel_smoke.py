import hashlib
import importlib.abc
import importlib.metadata
import json
import sys
from importlib.resources import files
from pathlib import Path

forbidden = {'torch', 'transformers', 'mlx_lm', 'mlx_vlm', 'mlx_audio', 'mlx_embeddings'}

class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise AssertionError('Native wheel imported reference library: ' + fullname)

sys.meta_path.insert(0, Guard())
absent = {}
for package in forbidden:
    try:
        importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        absent[package] = True
    else:
        raise AssertionError('Reference package installed: ' + package)

import mlx_one
from click.testing import CliRunner
from mlx_one.cli import main
from mlx_one.core.registry import registered_model_types

root = Path('/Users/susant-achary/dev_work/git_codes/c_base/mlx-one-release-qualification')
e = root / 'docs/qualification/model-release-20261108'
assert Path(mlx_one.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
catalog = json.loads(files('mlx_one').joinpath('data/model_support_release_20261108.json').read_text())
expected = json.loads((root / 'scripts/release/expected_model_types.json').read_text())
assert list(registered_model_types()) == expected
assert sorted(item['model_type'] for item in catalog['models']) == expected
for item in catalog['models']:
    evidence = root / item['evidence']
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == item['evidence_sha256']
    result = json.loads(evidence.read_text())
    assert result['status'] == 'PASS_FUNCTIONAL'
    assert result['checkpoint'] == item['checkpoint'] and result['revision'] == item['revision']
    assert item['status'] == 'SUPPORTED'
help_checks = {}
for command in ([], ['segment'], ['track']):
    output = CliRunner().invoke(main, [*command, '--help'])
    assert output.exit_code == 0, output.output
    help_checks[' '.join(command) or 'root'] = 'PASS'
assert not forbidden.intersection(sys.modules)
report = dict(status='PASS', model_types=40, support_standard=catalog['support_standard'],
              installed_source=mlx_one.__file__, reference_packages_absent=absent,
              cli_help=help_checks, all_support_claims_bound_to_passing_results=True,
              installed_record_sha256=hashlib.sha256(importlib.metadata.distribution('mlx-one').read_text('RECORD').encode()).hexdigest(),
              scope='Final installed wheel registry/catalog/import/help smoke; real-checkpoint results retain their recorded historical wheel identities.')
(e / 'clean-wheel-resolution-final.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
