"""Offline long-context diagnostics using the frozen P0 Engine and native supervisor.

Example: PYTHONPATH=src python qualification/p0_long_context.py --baseline RUN
Generated scores are diagnostics, never public context/quality qualification.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from mlx_one.inference_baseline import process_rss, source_provenance, supervise
from mlx_one.inference_qualification import validate_integrity, write_integrity
from mlx_one.native_qualification import preflight
from mlx_one.p0_engine_qualification import atomic_json, csv_write, digest, utc, validate_assets

VERSION = 'native-p0-long-context-v1'
OUTPUT_TOKENS = 128
REPEATS = 5
REQUEST_TIMEOUT = 600
WORKER_TIMEOUT = (REPEATS + 1) * REQUEST_TIMEOUT + 120


def inventory(metadata, lengths=(16384, 32768), *, maximum_legal=False):
    rows = []
    for model in metadata:
        limit = model['configured_context']
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError('validated checkpoint context limit required')
        for length in lengths:
            legal = length + OUTPUT_TOKENS <= limit
            rows.append({'model_id': model['model_id'], 'revision': model['revision'],
                'prompt_tokens': length, 'requested_prompt_tokens': length,
                'output_token_budget': OUTPUT_TOKENS, 'context_limit': limit,
                'context_class': 'literal', 'status': 'QUEUED' if legal else 'NOT_SUPPORTED',
                'reason': None if legal else 'prompt plus output exceeds checkpoint context'})
            if maximum_legal and length == 32768 and limit == length:
                rows.append({**rows[-1], 'prompt_tokens': limit - OUTPUT_TOKENS,
                    'context_class': 'maximum_legal_32k', 'status': 'QUEUED', 'reason': None})
    return rows


def performance(case, result, supervision):
    measurements = result.get('benchmarks', [])
    observed = next((r for r in result.get('cases', [])
                     if r['case_id'] == f"H{case['prompt_tokens']}"), {})
    status, reason = observed.get('status', 'BLOCKED'), observed.get('reason')
    if supervision['status'] != 'completed':
        status, reason = 'BLOCKED', supervision['status']
    elif observed.get('error_type') == 'CacheCapacityError':
        status = 'REJECTED_ADMISSION'
    elif status == 'PASS' and len(measurements) != REPEATS:
        status, reason = 'BLOCKED', 'incomplete measured repetitions'
    output = {**case, 'status': status, 'reason': reason,
              'measured_repetitions': len(measurements), 'qualification_promoted': False}
    for field in ('ttft_p50_seconds', 'ttft_p95_seconds', 'prefill_tps', 'decode_tps',
                  'end_to_end_tps', 'total_seconds', 'peak_mlx_bytes', 'peak_rss_bytes'):
        output[field] = None
    if status != 'PASS':
        return output  # Partial timings stay in raw evidence, never passed aggregate scores.
    for target, key, scale in (
            ('ttft_p50_seconds', 'time_to_first_token_ms', 1000),
            ('prefill_tps', 'prefill_tokens_per_second', 1),
            ('decode_tps', 'decode_tokens_per_second', 1),
            ('end_to_end_tps', 'end_to_end_tokens_per_second', 1),
            ('total_seconds', 'end_to_end_seconds', 1)):
        values = [r[key] / scale for r in measurements if r.get(key) is not None]
        output[target] = statistics.median(values) if values else None
    values = sorted(r['time_to_first_token_ms'] / 1000 for r in measurements
                    if r.get('time_to_first_token_ms') is not None)
    output['ttft_p95_seconds'] = values[math.ceil(.95 * len(values)) - 1] if values else None
    for target, key in (('peak_mlx_bytes', 'mlx_peak_bytes'), ('peak_rss_bytes', 'rss_peak_bytes')):
        values = [r[key] for r in measurements if r.get(key) is not None]
        output[target] = max(values) if values else None
    output['generated_tokens'] = [r['generated_tokens'] for r in measurements]
    return output


def attribute_rss(measurements, samples, origin):
    for row in measurements:
        start, end = row.get('started_monotonic'), row.get('finished_monotonic')
        selected = [s['rss_bytes'] for s in samples
                    if origin is not None and start is not None and end is not None
                    and start <= origin + s['elapsed_seconds'] <= end
                    and s.get('rss_bytes') is not None]
        row['rss_peak_bytes'] = max(selected) if selected else None
        row['rss_peak_scope'] = 'supervised-request-samples' if selected else 'unavailable'
        row['rss_sampling_interval_seconds'] = .1


def verify_assets(metadata):
    actual = validate_assets(metadata['model_id'], metadata['snapshot_path'], metadata['revision'])
    if (actual['assets'] != metadata['assets']
            or actual['configured_context'] != metadata['configured_context']):
        raise ValueError('checkpoint assets changed since the pinned baseline')


def export(root, manifest, rows):
    atomic_json(root / 'manifest.json', manifest)
    atomic_json(root / 'scores.json', rows)
    csv_write(root / 'scores.csv', rows)
    lines = ['# P0 16K / 32K diagnostics', '',
        'Reference dense B1; explicit FP16 floating components and attention KV; native packed '
        'weights; full-width prefill; prefix reuse disabled; output budget 128; one warm-up and '
        'five measured repetitions. Rejected/unexecuted profiles have no performance score.', '',
        '| Model | Prompt | Profile | Result | Runs | TTFT p50 s | Prefill TPS | Decode TPS | '
        'Total s | MLX peak GiB | Sampled RSS peak GiB |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def value(row, key, scale=1):
        return f"{row[key] / scale:.2f}" if row.get(key) is not None else '—'
    for row in rows:
        lines.append('| ' + ' | '.join([row['model_id'], str(row['prompt_tokens']),
            row['context_class'], row['status'], str(row.get('measured_repetitions', 0)),
            value(row, 'ttft_p50_seconds'), value(row, 'prefill_tps'), value(row, 'decode_tps'),
            value(row, 'total_seconds'), value(row, 'peak_mlx_bytes', 1024**3),
            value(row, 'peak_rss_bytes', 1024**3)]) + ' |')
    for row in rows:
        if row.get('reason'):
            lines.extend(['', f"- {row['model_id']} / {row['prompt_tokens']}: {row['reason']}"])
    lines.extend(['', 'These are diagnostic measurements, not long-context quality or public '
                  'capability qualification. MLX and RSS overlap and are never summed.'])
    (root / 'scores.md').write_text('\n'.join(lines) + '\n')
    atomic_json(root / 'scores-markdown.json', {'sha256': digest(root / 'scores.md')})
    write_integrity(root)


def worker(manifest_path, destination):
    # All MLX imports/model initialization remain isolated in the supervised child.
    from mlx_one.p0_engine_worker import Cases

    manifest = json.loads(Path(manifest_path).read_text())
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    if source_provenance(Path.cwd())['source_sha256'] != manifest['source']['source_sha256']:
        raise RuntimeError('frozen runtime source mismatch')
    if digest(__file__) != manifest['tool_sha256']:
        raise RuntimeError('long-context tooling mismatch')
    case = manifest['case']
    verify_assets(manifest['native_manifest']['metadata'])
    cases = Cases(manifest['native_manifest'], directory)
    original_generate, original_check = cases.generate, cases.check
    request_number = 0
    def generate(prompt, **options):
        nonlocal request_number
        request_number += 1
        heartbeat = {'request': request_number, 'status': 'running',
                     'started_monotonic': time.monotonic(), 'timeout_seconds': REQUEST_TIMEOUT}
        atomic_json(directory / 'current-request.json', heartbeat)
        try:
            return original_generate(prompt, **options)
        finally:
            stats = cases.engine._cache_manager.stats()
            heartbeat.update(status='finished', ended_monotonic=time.monotonic(),
                             ownership=stats.to_dict())
            atomic_json(directory / 'current-request.json', heartbeat)
            assert stats.active_handles == stats.reserved_bytes == stats.allocated_bytes == 0
    def check(identity, function, **settings):
        settings['seconds'] = WORKER_TIMEOUT - 60
        return original_check(identity, function, **settings)
    cases.generate, cases.check = generate, check
    try:
        assert cases.bundle.context_length == case['context_limit']
        cases.benchmark([case['prompt_tokens']])
    finally:
        cases.close()
        cases.save()


def run_case(root, case, metadata, baseline, source, observed):
    name = case['model_id'].replace('/', '--') + f"--{case['prompt_tokens']}"
    directory = root / name
    directory.mkdir()
    native = {'metadata': metadata, 'environment': {**baseline['environment'],
        'source': source, 'preflight': observed, 'policy': {**baseline['environment']['policy'],
        'diagnostic': True, 'benchmark_prompt_tokens': [case['prompt_tokens']]}}}
    manifest = {'schema_version': VERSION, 'case': case, 'source': source,
                'tool_sha256': digest(__file__), 'native_manifest': native}
    path = directory / 'manifest.json'
    atomic_json(path, manifest)
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(path),
               str(directory / 'data')]
    atomic_json(directory / 'command.json', {'argv': command, 'timestamp': utc()})
    origin, timed_out = [], []
    def clock():
        value = time.monotonic()
        if not origin:
            origin.append(value)
        return value
    with (directory / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
        def rss(pid):
            value = process_rss(pid)
            heartbeat = directory / 'data/current-request.json'
            if heartbeat.exists():
                active = json.loads(heartbeat.read_text())
                if active['status'] == 'running' and (
                        time.monotonic() - active['started_monotonic'] > REQUEST_TIMEOUT):
                    timed_out.append(active)
                    process.terminate()
            return value
        try:
            supervision = supervise(process, timeout=WORKER_TIMEOUT,
                memory_limit=observed['process_ceiling_bytes'], clock=clock, rss=rss)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if timed_out:
        supervision.update(status='request_timeout', timed_out_request=timed_out[0])
    atomic_json(directory / 'supervision.json', supervision)
    result_path = directory / 'data/results.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {'cases': []}
    attribute_rss(result.get('benchmarks', []), supervision['rss_samples'],
                  origin[0] if origin else None)
    for row in result.get('cases', []):
        if isinstance(row.get('details'), list):
            row['details'] = result.get('benchmarks', [])
    atomic_json(result_path, result)
    write_integrity(result_path.parent)
    index = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != 'artifact-index.json'}
    atomic_json(directory / 'artifact-index.json', index)
    write_integrity(directory)
    row = performance(case, result, supervision)
    row.update(artifact=str(directory), source=source, tool_sha256=manifest['tool_sha256'])
    return row, supervision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--after-driver', type=Path)
    parser.add_argument('--after-pid', type=int)
    parser.add_argument('--maximum-legal-32k', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=Path('qualification/results/p0_engine/long_context'))
    parser.add_argument('--worker', nargs=2, metavar=('MANIFEST', 'DESTINATION'))
    args = parser.parse_args(argv)
    if args.worker:
        worker(*args.worker)
        return 0
    if args.baseline is None:
        parser.error('--baseline is required')
    root = args.output.resolve() / 'runs' / str(time.time_ns())
    root.mkdir(parents=True)
    baseline = {'environment': json.loads((args.baseline / 'environment.json').read_text())}
    downloads = json.loads((args.baseline / 'download_manifest.json').read_text())
    metadata = [value['metadata'] for value in downloads.values() if value['status'] == 'PASS']
    source = source_provenance(Path.cwd())
    if source['source_sha256'] != baseline['environment']['source']['source_sha256']:
        raise RuntimeError('baseline runtime source mismatch')
    rows = inventory(metadata, maximum_legal=args.maximum_legal_32k)
    manifest = {'schema_version': VERSION, 'source': source, 'tool_sha256': digest(__file__),
        'baseline': str(args.baseline.resolve()), 'timestamp': utc(), 'status': 'QUEUED',
        'after_driver': str(args.after_driver.resolve()) if args.after_driver else None,
        'after_pid': args.after_pid, 'prompt_tokens': [16384, 32768],
        'maximum_legal_32k': args.maximum_legal_32k,
        'mode': 'reference_dense', 'batch': 1, 'prefill_chunk_tokens': None,
        'output_token_budget': OUTPUT_TOKENS, 'execution_dtype': 'float16', 'kv_dtype': 'float16',
        'warmup_requests': 1, 'measured_repetitions': REPEATS, 'request_timeout_s': REQUEST_TIMEOUT,
        'worker_timeout_s': WORKER_TIMEOUT, 'workers_at_once': 1, 'qualification_promoted': False}
    (root / 'runner-source.txt').write_text(Path(__file__).read_text())
    export(root, manifest, rows)
    print(root, flush=True)
    try:
        if args.after_driver:
            until = time.monotonic() + 12 * 3600
            while True:
                state = json.loads(args.after_driver.read_text())
                if state['status'] != 'running':
                    if state['status'] != 'PASS':
                        raise RuntimeError('prerequisite qualification driver: ' + state['status'])
                    break
                if args.after_pid is not None:
                    os.kill(args.after_pid, 0)
                if time.monotonic() >= until:
                    raise TimeoutError('prerequisite qualification driver deadline')
                time.sleep(10)
        validate_integrity(args.baseline)
        if json.loads((args.baseline / 'summary.json').read_text())['decision'] != 'GO':
            raise RuntimeError('baseline qualification is not complete; no native worker started')
        for entry in json.loads((args.baseline / 'worker_evidence.json').read_text()):
            path = Path(entry['path'])
            if digest(path) != entry['sha256']:
                raise ValueError('baseline worker evidence checksum mismatch')
            validate_integrity(path.parent)
            for name, checksum in json.loads(path.read_text()).items():
                if Path(name).name != name or digest(path.parent / name) != checksum:
                    raise ValueError('baseline worker artifact checksum mismatch')
        baseline_rows = json.loads((args.baseline / 'qualification_results.json').read_text())
        baseline_cases = {(r['model_id'], r['case_id']): r for r in baseline_rows}
        manifest['status'] = 'running'
        observed = preflight(root / 'preflight')
        manifest['preflight'] = observed
        export(root, manifest, rows)
        for index, case in enumerate(rows):
            if case['status'] != 'QUEUED':
                continue
            if source_provenance(Path.cwd())['source_sha256'] != source['source_sha256']:
                raise RuntimeError('frozen runtime source changed')
            if digest(__file__) != manifest['tool_sha256']:
                raise RuntimeError('long-context tooling changed')
            prerequisites = [r for (model, _), r in baseline_cases.items()
                             if model == case['model_id'] and r['mandatory']
                             and r.get('applicable', True)]
            if not prerequisites or any(r['status'] != 'PASS' for r in prerequisites):
                rows[index] = {**case, 'status': 'BLOCKED', 'reason': 'baseline gates incomplete'}
                export(root, manifest, rows)
                continue
            model = next(m for m in metadata if m['model_id'] == case['model_id'])
            rows[index], supervision = run_case(root, case, model, baseline, source, observed)
            export(root, manifest, rows)
            print(json.dumps(rows[index]), flush=True)
            if supervision['status'] not in {'completed', 'request_timeout'}:
                raise RuntimeError('native supervision: ' + supervision['status'])
            if supervision['status'] == 'request_timeout' or rows[index]['status'] == 'FAIL':
                observed = preflight(root / f"recovery-{index}")
        manifest['status'] = 'completed'
    except Exception as error:
        manifest.update(status='BLOCKED', reason=str(error), error_type=type(error).__name__)
        for row in rows:
            if row['status'] == 'QUEUED':
                row.update(status='BLOCKED', reason=str(error))
    export(root, manifest, rows)
    print(root, flush=True)
    return int(manifest['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
