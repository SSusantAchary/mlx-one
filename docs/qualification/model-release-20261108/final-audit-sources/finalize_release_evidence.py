import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path('/Users/susant-achary/dev_work/git_codes/c_base/mlx-one-release-qualification')
WORK = Path('/private/tmp/mlx-one-all40-20261108')
E = ROOT / 'docs/qualification/model-release-20261108'
PYTHON = WORK / 'wheel-env/bin/python'
FINAL = WORK / 'dist-resolution-final/mlx_one-0.1.1-py3-none-any.whl'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(name, data):
    (E / name).write_text(json.dumps(data, indent=2) + '\n')

records = []
with (WORK / '.qualification-gpu.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with (E / 'candidate-install-lineage-resolution.log').open('w') as log:
        def install(wheel):
            subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--no-index', '--no-deps',
                            '--force-reinstall', str(wheel)], check=True, stdout=log,
                           stderr=subprocess.STDOUT,
                           env={**os.environ, 'PIP_DISABLE_PIP_VERSION_CHECK': '1'})
        try:
            for folder in [*(WORK / f'dist-resolution{i}' for i in range(1, 9)), FINAL.parent]:
                wheel = next(folder.glob('*.whl'))
                install(wheel)
                record = next((WORK / 'wheel-env/lib').glob('python*/site-packages/mlx_one-0.1.1.dist-info/RECORD'))
                digest = hashlib.sha256(record.read_text().encode()).hexdigest()
                saved = E / 'installed-records' / f'resolution-{digest}.csv'
                saved.write_bytes(record.read_bytes())
                records.append(dict(build=folder.name, installed_record_sha256=digest,
                                    raw_record_sha256=sha(saved),
                                    installed_record_file=str(saved.relative_to(E)),
                                    artifacts={p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                                               for p in folder.iterdir() if p.is_file()}))
        finally:
            install(FINAL)

used = {}
for path in E.rglob('*.json'):
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        continue
    if isinstance(data, dict) and data.get('installed_record_sha256'):
        used.setdefault(data['installed_record_sha256'], []).append(str(path.relative_to(E)))
old = json.loads((E / 'candidate-lineage.json').read_text())
identities = {key: value['build'] for key, value in old['probe_identity_matches'].items() if value.get('build')}
identities.update({r['installed_record_sha256']: r['build'] for r in records})
unmatched = sorted(set(used) - identities.keys())
write('candidate-lineage-resolution.json', dict(
    builds=records,
    probe_identity_matches={key: dict(build=identities.get(key), results=paths)
                            for key, paths in used.items()},
    unmatched_installed_records=unmatched,
    method='Immutable wheels reinstalled locally using the original candidate interpreter and pip. RECORD text normalized as importlib.metadata.read_text() used by probe workers. Raw bytes and their checksum archived independently; final wheel restored.'
))
if unmatched:
    raise RuntimeError(f'Unmapped installed identities: {unmatched}')

packaged = {}
with zipfile.ZipFile(FINAL) as wheel:
    for member in wheel.namelist():
        if member.startswith('mlx_one/') and not member.endswith('/'):
            digest = hashlib.sha256(wheel.read(member)).hexdigest()
            relative = 'src/' + member
            if not (ROOT / relative).is_file() or sha(ROOT / relative) != digest:
                raise RuntimeError('Final wheel/source mismatch: ' + relative)
            packaged[relative] = digest
with zipfile.ZipFile(WORK / 'dist-resolution8' / FINAL.name) as previous:
    old_files = {n: hashlib.sha256(previous.read(n)).hexdigest()
                 for n in previous.namelist() if n.startswith('mlx_one/') and not n.endswith('/')}
changes = sorted(name for name, digest in packaged.items()
                 if old_files.get(name.removeprefix('src/')) != digest)
write('candidate-build-resolution-final.json', dict(
    base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    version='0.1.1', build=FINAL.parent.name, artifacts=records[-1]['artifacts'],
    packaged_files=packaged, exact_packaged_source_match=True,
    files_changed_since_last_inference_wheel=changes,
    scope='Per-model task results retain their own historical installed wheel identity. Final source regression and exact wheel/sdist payload checks pass separately; models were not redownloaded for a blanket final-wheel replay.'
))

scan_roots = [WORK / n for n in ('models', 'hf', 'xet', 'reference-sources')]
remaining = []
for folder in scan_roots:
    for path in folder.rglob('*'):
        if path.is_file() and (path.suffix in {'.safetensors', '.gguf'} or path.name.startswith('pytorch_model') and path.suffix == '.bin'):
            remaining.append(str(path))
protected = Path('/Users/susant-achary/mlx-one-models/gpt-oss-20b-MXFP4-Q4')
assets = json.loads((E / 'gpt_oss-assets.json').read_text())
checked = [dict(path=item['path'], unchanged=(protected / item['path']).is_file()
                and sha(protected / item['path']) == item['sha256'], bytes=item['bytes'])
           for item in assets['files']]
gguf = Path('/Users/susant-achary/models/gpt-oss-20b-MXFP4')
owned = sorted(str(path) for path in (WORK / 'models').iterdir() if path.is_dir())
cleanups = [json.loads(path.read_text()) for path in E.rglob('*-cleanup.json')]
usage = shutil.disk_usage(WORK)
ok = not remaining and not owned and all(item['unchanged'] for item in checked) and gguf.is_dir()
write('cleanup-resolution-audit.json', dict(
    timestamp=time.time(), status='PASS' if ok else 'FAIL', cleanup_records=len(cleanups),
    all_owned_containers_removed=not owned, remaining_owned_containers=owned,
    remaining_task_checkpoint_files=remaining, scanned_roots=list(map(str, scan_roots)),
    protected_native_gpt_oss=dict(path=str(protected), all_assets_hash_unchanged=all(item['unchanged'] for item in checked), files=checked),
    protected_gguf_folder_exists=gguf.is_dir(),
    disk_bytes=dict(total=usage.total, used=usage.used, free=usage.free),
    scope='Task-owned downloaded checkpoints removed after recording outputs. Existing user GPT-OSS models retained. Fixtures, outputs, software environments and candidate artifacts retained.'
))
if not ok:
    raise RuntimeError('Cleanup/protected model audit failed')
print(json.dumps(dict(installed_builds=len(records), installed_probe_identities=len(used),
                      unmatched=unmatched, packaged_files=len(packaged),
                      changes_since_build8=changes, cleanup='PASS'), indent=2))
