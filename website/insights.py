"""Build public model/benchmark views from versioned evidence, without importing MLX."""
from __future__ import annotations

from collections import Counter, defaultdict
import ast
from datetime import date, timedelta
import hashlib
import html
import json
import math
from pathlib import Path
import runpy
import subprocess

from support import current_support, registration_history

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'https://github.com/SSusantAchary/mlx-one'
CATALOGS = {
    'text': ('src/mlx_one/data/compatibility/alpha_release.json', 'checkpoints'),
    'vision': ('docs/benchmarks/vlm/benchmark_plan.json', 'checkpoints'),
    'audio': ('benchmarks/audio/checkpoints.json', 'entries'),
    'ocr': ('docs/benchmarks/ocr/inventory.json', 'checkpoints'),
    'native tasks': ('src/mlx_one/data/native_tasks.json', 'profiles'),
}
STATUSES = {
    'scoped-smoke': ('Scoped smoke tested', 'The pinned checkpoint passed a bounded smoke test. Broader quality, hardware, context and release qualification are separate.'),
    'experimental': ('Experimental', 'Implemented or measured experimental scope; this label does not establish full checkpoint or release qualification.'),
    'architecture-only': ('Architecture only', 'Architecture or quantization tests exist; the pinned full checkpoint has no qualifying smoke result.'),
    'work-in-progress': ('Work in progress', 'Partial evidence exists, with unresolved scope or qualification work.'),
    'blocked': ('Blocked', 'A required gate failed. Diagnostic results do not admit the blocked task.'),
    'not-implemented': ('Not implemented', 'The manifest explicitly records that this task is not implemented.'),
    'unqualified': ('Unqualified / unrecorded', 'No recognized qualifying status is recorded. Unknown statuses are never promoted to supported.'),
}

MODALITIES = (
    ('text', 'Text generation', {'text'}, 'models/index.md#text-generation-guides',
     'GPT-OSS, Llama, Qwen, Gemma and more',
     '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z"/><path d="m10 7-3 3 3 3m4-6 3 3-3 3"/>'),
    ('vision', 'Vision-language', {'vision-language'}, 'models/index.md#vision-language-guides',
     'Qwen2-VL, Qwen2.5-VL, Qwen3.5 and LFM2-VL',
     '<path d="M3 7V5a2 2 0 0 1 2-2h2m10 0h2a2 2 0 0 1 2 2v2M3 17v2a2 2 0 0 0 2 2h2m10 0h2a2 2 0 0 0 2-2v-2"/><path d="M2 12s4-6 10-6 10 6 10 6-4 6-10 6S2 12 2 12Z"/><circle cx="12" cy="12" r="2"/>'),
    ('audio', 'Audio & speech', {'asr', 'audio', 'audio-language'}, 'models/index.md#audio-guides',
     'Whisper, Moonshine, Qwen ASR, Parakeet, LFM2 Audio and Silero',
     '<path d="M3 10v4m4-7v10m5-15v20m5-15v10m4-7v4"/>'),
    ('ocr', 'OCR & documents', {'ocr'}, 'models/index.md#ocr-and-documents',
     'TrOCR, GOT-OCR, Florence, PaddleOCR, LightOnOCR and DeepSeek',
     '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h5M14 2v6h6M14 2l6 6"/><path d="M15 13h-1a1 1 0 0 0-1 1v1m6-2h1a1 1 0 0 1 1 1v1m-8 4v1a1 1 0 0 0 1 1h1m4 0h1a1 1 0 0 0 1-1v-1"/>'),
    ('retrieval', 'Embeddings & reranking', {'embedding', 'reranking'}, 'models/index.md#retrieval-guides',
     'BERT, MPNet, Qwen3 Embedding, Qwen3 Reranker and ColBERT',
     '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>'),
    ('segmentation', 'Segmentation', {'segmentation'}, 'models/index.md#segmentation-guides',
     'SAM3 image, interactive prompts and video tracking',
     '<path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m8 0h3a2 2 0 0 0 2-2v-3"/>'),
    ('classification', 'Classification', {'classification'}, 'models/index.md#classification-guides',
     'DistilBERT sentiment, BERT NER and CLIP zero-shot images',
     '<path d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zm11 3 2 2 4-5"/>'),
)


def modality_coverage(record):
    """Read the real lazy registry without importing the MLX-One package/runtime."""
    paths = {key: ROOT / path for key, path in {
        'models': 'src/mlx_one/models/registry.py',
        'core': 'src/mlx_one/core/registry.py',
        'ocr': 'src/mlx_one/ocr/registry.py',
    }.items()}
    for path in paths.values():
        record(path)
    core = runpy.run_path(str(paths['core']))
    ocr = runpy.run_path(str(paths['ocr']))
    source = ast.parse(paths['models'].read_text())
    functions = [item for item in source.body if isinstance(item, ast.FunctionDef)
                 and item.name in {'builtin_registrations', 'ocr_registrations'}]

    class RegistryFunctions(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            # OCR's backend-free metadata is supplied directly, avoiding package __init__.
            if node.module == 'mlx_one.ocr.registry':
                return None
            return node

    module = RegistryFunctions().visit(ast.Module(body=functions, type_ignores=[]))
    namespace = {'ModelRegistration': core['ModelRegistration'], 'CAPABILITIES': ocr['CAPABILITIES']}
    exec(compile(ast.fix_missing_locations(module), str(paths['models']), 'exec'), namespace)
    registrations = namespace['builtin_registrations']()
    if len({r.model_type for r in registrations}) != len(registrations):
        raise ValueError('Duplicate built-in model type in modality coverage')
    groups = []
    assigned = set()
    for key, title, kinds, target, description, _ in MODALITIES:
        members = sorted(r.model_type for r in registrations if r.modality in kinds)
        assigned.update(members)
        groups.append({'id': key, 'title': title, 'count': len(members), 'model_types': members,
                       'target': target, 'description': description})
    additional = sorted(r.model_type for r in registrations if r.modality == 'classification')
    assigned.update(additional)
    if assigned != {r.model_type for r in registrations}:
        raise ValueError('A new registry modality needs a homepage coverage group')
    return {'total_model_types': len(registrations), 'groups': groups,
            'classification': {'count': len(additional), 'model_types': additional},
            'count_basis': 'Distinct registered native model types, counted once by primary modality. Checkpoint sizes, revisions and quantizations do not add to this count. Implementation scope and checkpoint qualification vary.'}


def modality_cards_html(data):
    coverage = data['modality_coverage']
    output = [f'<p class="modality-intro"><strong>{coverage["total_model_types"]} native model types</strong> '
              'for generation, perception, speech and retrieval on Apple Silicon.</p>',
              '<div class="modality-cards" aria-label="Native model types by modality">']
    for group, spec in zip(coverage['groups'], MODALITIES):
        icon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + spec[-1] + '</svg>'
        output.append(f'<a class="modality-card" href="{group["target"]}">{icon}'
                      f'<span class="modality-title">{html.escape(group["title"])}</span>'
                      f'<span class="modality-count"><strong>{group["supported"]}/{group["count"]}</strong> model types supported</span>'
                      f'<span class="modality-description">{html.escape(group["description"])}</span></a>')
    output.append('</div>')
    return '\n'.join(output)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def classify(scope, row):
    """Preserve explicit blockers and qualification boundaries before smoke signals."""
    raw = str(row.get('decision') or row.get('status') or row.get('checkpoint_status')
              or row.get('execution_status') or row.get('source_status') or 'UNRECORDED')
    if raw == 'NOT IMPLEMENTED':
        return 'not-implemented', False, raw
    if raw == 'BLOCKED':
        return 'blocked', False, raw
    smoke = (row.get('checkpoint_status') in {'MODEL_SMOKE_TESTED', 'CHECKPOINT_SMOKE_PASSED'}
             or row.get('functional_smoke') == 'PASS_SCOPED'
             or (row.get('functional_tested') is True and row.get('source_parity') is True))
    if raw == 'WIP':
        return 'work-in-progress', smoke, raw
    if raw in {'ARCHITECTURE_IMPLEMENTED', 'QUANTIZATION_BACKEND_TESTED'}:
        return 'architecture-only', False, raw
    if raw in {'EXPERIMENTAL_NOT_QUALIFIED', 'EXPERIMENTAL'}:
        return 'experimental', smoke, raw
    if smoke:
        return 'scoped-smoke', True, raw
    if raw in {'MEASURED', 'NATIVE_IMPLEMENTED'} or row.get('experimental') is True:
        return 'experimental', False, raw
    return 'unqualified', False, raw


def profiles(scope, document):
    source, key = CATALOGS[scope]
    result = []
    for row in document.get(key, []):
        model_id = row.get('model_id') or row.get('checkpoint')
        if not model_id:
            continue
        task = row.get('task') or row.get('primary_task') or {
            'text': 'text-generation', 'vision': 'image-text-to-text',
        }.get(scope, scope)
        precision = str(row.get('precision') or row.get('primary_profile') or 'Unrecorded')
        status, smoke, raw = classify(scope, row)
        revision = row.get('revision')
        identity = '|'.join([scope, model_id, str(revision), task, precision])
        limitations = row.get('known_limitations', [])
        notes = [str(row[k]) for k in ('reason', 'notes', 'scope', 'known_limitation') if row.get(k)]
        if isinstance(limitations, list):
            notes.extend(str(v) for v in limitations)
        result.append({
            'key': identity, 'id': row.get('id') or row.get('key') or row.get('model_type'),
            'model_id': model_id, 'revision': revision, 'modality': scope, 'task': task,
            'precision': precision, 'status': status, 'raw_status': raw, 'smoke_tested': smoke,
            'release_qualified': row.get('release_qualified', row.get('alpha_supported')),
            'notes': notes, 'source': source,
            'interfaces': {k: row[k] for k in ('python_api', 'cli', 'http') if k in row},
        })
    return result


def build_history(current):
    """First-parent snapshots date when evidence reached this branch, not when a test ran."""
    paths = [v[0] for v in CATALOGS.values()]
    logs = git('log', '--first-parent', '--reverse', '--format=%H\t%cs', '--', *paths)
    points, first_seen, first_smoke = [], {}, {}
    for line in logs.splitlines():
        commit, day = line.split('\t')
        snapshot = []
        for scope, (path, _) in CATALOGS.items():
            response = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=ROOT,
                                      text=True, capture_output=True)
            if response.returncode:
                continue
            document = json.loads(response.stdout)
            snapshot.extend(profiles(scope, document))
        for row in snapshot:
            first_seen.setdefault(row['key'], {'date': day, 'commit': commit})
            if row['smoke_tested']:
                first_smoke.setdefault(row['key'], {'date': day, 'commit': commit})
        points.append({'date': day, 'commit': commit, 'catalog_profiles': len(snapshot),
                       'scoped_smoke_profiles': sum(r['smoke_tested'] for r in snapshot)})
    for row in current:
        row['first_recorded'] = first_seen.get(row['key'])
        row['first_smoke_recorded'] = first_smoke.get(row['key'])
    return {'points': points,
            'method': 'First-parent Git snapshots of the five versioned manifests. Dates mark when records reached the branch, not original test execution, release approval, or first implementation. Counts are checkpoint/task/precision profiles, including candidates. Earlier support outside these manifests is not reconstructed.',
            'shallow': git('rev-parse', '--is-shallow-repository') == 'true'}


def weekly_milestones(changelog):
    weeks = defaultdict(lambda: {'commits': [], 'milestones': []})
    for line in git('log', '--first-parent', '--format=%H\t%cs\t%s').splitlines():
        commit, day, subject = line.split('\t', 2)
        d = date.fromisoformat(day)
        monday = (d - timedelta(days=d.weekday())).isoformat()
        weeks[monday]['commits'].append({'commit': commit, 'date': day, 'subject': subject})
    import re
    for title, day in re.findall(r'^### (.+?)\s+[—–]\s+(\d{4}-\d{2}-\d{2})\s*$', changelog, re.M):
        d = date.fromisoformat(day)
        monday = (d - timedelta(days=d.weekday())).isoformat()
        weeks[monday]['milestones'].append(title)
    return [dict(week=week, end=(date.fromisoformat(week) + timedelta(days=6)).isoformat(),
                 commit_count=len(values['commits']), commits=values['commits'], milestones=values['milestones'])
            for week, values in sorted(weeks.items(), reverse=True)]


def make_data(record):
    revision = git('rev-parse', 'HEAD')

    def load(path):
        source = ROOT / path
        record(source)
        return json.loads(source.read_text())

    models = []
    for scope, (path, _) in CATALOGS.items():
        models.extend(profiles(scope, load(path)))
    history = build_history(models)
    cohorts = []

    def cohort(key, title, unit, note, rows):
        for row in rows:
            value = row.get('value')
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f'Invalid benchmark value: {key}: {row}')
            if row.get('evidence'):
                path = ROOT / row['evidence']
                record(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if row.get('evidence_sha256') and row['evidence_sha256'] != digest:
                    raise ValueError(f'Evidence hash mismatch: {path}')
                row['evidence_sha256'] = digest
                row['evidence_url'] = f'{REPOSITORY}/blob/{revision}/{row["evidence"]}'
        cohorts.append({'id': key, 'title': title, 'unit': unit, 'note': note, 'rows': rows})

    text = load('docs/benchmarks/figures/text-model-performance-data.json')
    for protocol, key, title in [
        ('historical-128-16', 'text-historical', 'Text · historical 128-input / 16-output'),
        ('standard-128-128', 'text-standard', 'Text · standardized 128-input / 128-output'),
    ]:
        rows = []
        for point in text['direct_model_points']:
            if point['protocol'] == protocol and point['primary']:
                rows.append(dict(point, model_id=point['checkpoint'], value=point['decode_tokens_per_second'],
                                 status='Measured', context=point.get('environment', {})))
        if key == 'text-historical':
            rows.extend(dict(point, model_id=point['checkpoint'], value=None) for point in text['missing_direct_model'])
        cohort(key, title, 'decode tokens/s',
               'Apple M4 / 32 GiB, MLX 0.32.2; direct-model, batch 1, 1 warmup, 3 measured repetitions, median. Historical and standardized protocols remain separate; precision and architecture vary. Missing measurements stay unavailable.', rows)

    multimodal = load('docs/benchmarks/figures/multimodal-performance-data.json')
    for source_key, key, title, metric, unit, note in [
        ('vlm', 'vision', 'Vision-language · VLM-HF-S1', 'decode_tps', 'decode tokens/s', 'One 512×512 image, fixed prompt, greedy decoding, up to 128 emitted tokens; actual counts retained.'),
        ('asr', 'asr', 'Speech recognition · ASR-S1', 'audio_seconds_per_second', 'audio seconds/s', 'One 10-second English fixture, prepared PCM; single-clip WER is not corpus accuracy.'),
        ('speech', 'tts', 'Speech generation · TTS-S1', 'audio_seconds_per_second', 'generated audio seconds/s', 'Fixed text and voice; waveform codec included. Speech quality is not established by speed.'),
        ('speech', 'sts', 'Speech-to-speech · short question', 'audio_seconds_per_second', 'generated audio seconds/s', 'Short-question and failed 10-second cases remain distinct rows. This is not a shared TTS workload.'),
    ]:
        rows = []
        for point in multimodal[source_key]:
            if source_key == 'speech' and (point.get('task') == 'synthesize') != (key == 'tts'):
                continue
            rows.append(dict(point, model_id=point['checkpoint'], value=point['metrics'].get(metric),
                             context={k: point.get(k) for k in ('hardware', 'software', 'mlx_version', 'protocol', 'measurement_type', 'actual_input_tokens', 'actual_output_tokens')}))
        cohort(key, title, unit, 'Apple M4 / 32 GiB, MLX 0.32.2; batch 1, 1 warmup, 5 repetitions, per-column median. ' + note, rows)

    for path, key, title, directory, unit, note in [
        ('docs/benchmarks/figures/ocr-omni100-performance.json', 'ocr-pages', 'OCR · Omni100 pages', 'omni100-20261025', 'pages/s', '100 pages, 1 warmup, one timed request/page, greedy output cap 1,023 tokens. Failures and truncations remain in the evidence.'),
        ('docs/benchmarks/figures/ocr-hand10-performance.json', 'ocr-lines', 'OCR · handwritten lines', 'hand10-20261025', 'images/s', '10 handwritten images, 1 warmup, one timed request/image, output cap 128 tokens. These line crops are not a page workload.'),
    ]:
        rows = []
        for point in load(path)['values']:
            entry = next(r for r in load(CATALOGS['ocr'][0])['checkpoints'] if r['checkpoint'] == point['checkpoint'])
            evidence = f'docs/benchmarks/ocr/{directory}/{entry["id"]}.json'
            raw = load(evidence)
            if raw['revision'] != point['revision'] or not math.isclose(raw['metrics']['completed_pages_per_second'], point['throughput']):
                raise ValueError(f'OCR summary differs from evidence: {evidence}')
            rows.append(dict(point, model_id=point['checkpoint'], value=point['throughput'],
                             precision=raw['precision_label'], status=raw['status'], evidence=evidence,
                             metrics=raw['metrics'], context={k: raw[k] for k in ('hardware', 'software', 'protocol')}))
        cohort(key, title, unit, 'Apple M4 / 32 GiB, MLX 0.32.2; direct-model, batch 1. ' + note, rows)

    boost_source = 'docs/mlxboost/performance-summary.json'
    boost = load(boost_source)
    boost_rows = []
    for model, label, off_key, on_key in [
        ('qwen2', 'Qwen2.5-0.5B-Instruct · Q8', 'expanded-qwen2-off', 'expanded-qwen2-on'),
        ('qwen3', 'Qwen3-0.6B · Q8', 'expanded-qwen3-off', 'expanded-qwen3-on'),
        ('smollm2', 'SmolLM2-1.7B-Instruct · BF16', 'original-protocol-off', 'original-protocol-fast'),
    ]:
        def selection(key):
            matches = [r for r in boost[key] if r['model'] == model and r['backend'] == 'mlx-one' and r['view'] == 'unfiltered']
            if len(matches) != 1:
                raise ValueError(f'Ambiguous MLXBoost cohort: {key}/{model}')
            return matches[0]
        off, on = selection(off_key), selection(on_key)
        boost_rows.append({'model': label, 'off': off['output_tokens_per_second']['median'],
                           'on': on['output_tokens_per_second']['median'], 'samples_per_flag': off['samples'],
                           'processes_per_flag': off['processes'], 'paired': boost.get('paired-' + model),
                           'off_cohort': off_key, 'on_cohort': on_key})
    changelog = ROOT / 'CHANGELOG.md'
    record(changelog)
    for path in ['website/insights.py', 'src/mlx_one/models/registry.py']:
        record(ROOT / path)
    coverage = modality_coverage(record)
    support = current_support(record, coverage)
    support['registration_history'] = registration_history()
    return {
        'schema_version': 1, 'source_commit': revision, 'source_url': f'{REPOSITORY}/tree/{revision}',
        'modality_coverage': coverage, 'support': support,
        'source_date': git('show', '-s', '--format=%cI', 'HEAD'),
        'scope': 'Five versioned catalogs: text, VLM, audio, OCR and native task candidates. Counts are checkpoint/task/precision profiles, not all architecture families or a blanket support claim. Refreshed at each documentation build; no model download or inference.',
        'summary': {'profiles': len(models), 'unique_revisions': len({(r['model_id'], r['revision']) for r in models}),
                    'scoped_smoke': sum(r['smoke_tested'] for r in models),
                    'release_qualified_true': sum(r['release_qualified'] is True for r in models),
                    'release_qualification_unrecorded': sum(r['release_qualified'] is None for r in models),
                    'by_status': dict(Counter(r['status'] for r in models)),
                    'by_modality': dict(Counter(r['modality'] for r in models))},
        'status_definitions': {key: {'label': label, 'description': description} for key, (label, description) in STATUSES.items()},
        'models': models, 'history': history, 'weeks': weekly_milestones(changelog.read_text()),
        'performance': cohorts,
        'mlxboost': {'rows': boost_rows, 'evidence_url': f'{REPOSITORY}/blob/{revision}/{boost_source}',
                    'note': 'Historical Apple M4 / 32 GiB, MLX 0.32.2. Output tokens divided by complete HTTP SSE wall time, not direct-model decode. Qwen: 50 requests/flag across five processes; SmolLM2: six requests/flag across three processes. One excluded warmup/process. Host load and thermals uncontrolled; cohorts and run order differ. Qwen has 100/100 paired exact-output matches across both models. The measured host-overhead change adds no custom Metal kernel. Existing numerical failures remain failed. EXPERIMENTAL, opt-in, no production rollout approval.'},
    }


def summary_html(data):
    support = data['support']
    release = support['release']
    cards = [(support['model_types'], 'Native model types'), (support['modalities'], 'Modality categories'),
             (f'{support["supported"]}/{support["model_types"]}', 'Primary workflows passed'),
             (release['version'],
              'Approved for release · publication pending' if not release['published']
              and release['release_approval'] == 'APPROVED' else 'Published release'
              if release['published'] else 'Release candidate')]
    return '<div class="insights-cards">' + ''.join(
        f'<div class="insights-card"><strong>{value}</strong><span>{html.escape(label)}</span></div>' for value, label in cards) + '</div>'


def fallback_markdown(data):
    lines = ['<details markdown="1" class="insights-fallback">', '<summary>Full model catalog (also available without JavaScript)</summary>', '',
             '| Checkpoint / task | Precision | Status |', '| --- | --- | --- |']
    for row in data['models']:
        label = STATUSES[row['status']][0]
        lines.append(f'| `{row["model_id"]}` — {row["task"]} | {row["precision"]} | {label} |')
    lines.extend(['', '</details>'])
    return '\n'.join(lines)
