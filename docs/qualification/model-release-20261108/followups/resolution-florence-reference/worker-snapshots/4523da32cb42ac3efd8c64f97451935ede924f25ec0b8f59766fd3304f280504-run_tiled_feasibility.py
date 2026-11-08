#!/usr/bin/env python3
"""Offline M4 feasibility for native gathered tiled attention; explicit diagnostics."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(Path(__file__).resolve().parent)]
DEFAULT_CAPTURE = Path('qualification/results/inference_optimization_qwen_causal') / (
    '1792344698348447000/causal')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['captured', 'model', 'operators', 'model-performance'])
    parser.add_argument('--tile', type=int, choices=[32, 64, 128, 256], default=128)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--integrated', action='store_true',
                        help='Qualify/time the opt-in production route with actual dispatch')
    parser.add_argument('--capture', type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument('--checkpoint-paths', type=Path,
                        default=Path('scripts/qualification/checkpoint-paths.local.json'))
    parser.add_argument('--correctness-report', type=Path)
    parser.add_argument('--output', type=Path,
                        default=Path('qualification/results/inference_optimization_tiled_feasibility'))
    args = parser.parse_args()
    if args.samples < 3:
        parser.error('at least three timing samples required')
    if args.integrated and (args.stage not in {'model', 'model-performance'} or args.tile != 128):
        parser.error('integrated route is model/model-performance only, with fixed tile 128')
    directory = args.output.resolve() / str(time.time_ns()) / args.stage
    directory.mkdir(parents=True)
    report = {'schema_version': 'native-tiled-feasibility-v1', 'stage': args.stage,
              'status': 'RUNNING', 'query_tile': args.tile, 'command': sys.argv,
              'backend': 'native_gather_tiled', 'stock_sdpa': False,
              'production_integrated': args.integrated, 'qualification_promoted': False,
              'memory_limit_bytes': 20 * 1024**3,
              'latency_scope': 'synchronized Python graph construction + GPU evaluation wall time; '
                               'compiled shaders warm, allocator cache cleared before each sample',
              'strict_boundary': 'finite outputs; zero numeric AND bitwise FP16 mismatches',
              'frozen_atol_rtol': {'float16': [0.01, 0.01], 'float32': [0.0001, 0.0001]}}
    (directory / 'command.json').write_text(json.dumps(sys.argv, indent=2) + '\n')
    try:
        execute(args, directory, report)
    except BaseException as exc:
        report.update(status='FAIL', exception={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        (directory / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        from mlx_one.inference_qualification import write_integrity
        write_integrity(directory)
        print(directory, flush=True)
    return 0 if report['status'] == 'PASS' else 1


def load_bundle(args, report):
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.native_correctness import checkpoint_proof
    from mlx_one.text import load_text_model

    path = json.loads(args.checkpoint_paths.read_text())['qwen3-small']
    asset = checkpoint_proof('qwen3-small', path)
    pinned = REPO / DEFAULT_CAPTURE / 'causal-results.json'
    assert digest(pinned) == '9f1ba7297a578d33b813d9bae54544b9f88c10a5a16b498f3c8b2733062ffb87'
    original = json.loads(pinned.read_text())['checkpoint']
    assert {key: value for key, value in asset.items() if key != 'path'} == {
        key: value for key, value in original.items() if key != 'path'}, (
            'checkpoint files/identity differ from original captured operands')
    report['original_checkpoint_evidence'] = {'path': str(pinned), 'sha256': digest(pinned)}
    assert asset['model_id'] == 'mlx-community/Qwen3-0.6B-8bit'
    assert asset['source_revision'] == asset['revision'] == (
        '11de96878523501bcaa86104e3c186de07ff9068')
    assert asset['files']['model.safetensors'] == (
        '3ad5c96e0a476d4eee48e9f525d0ee0a3f1830b3e5b8e17b441de0661e921397')
    bundle = load_text_model(asset['model_id'], checkpoint_path=path,
                             expected_files=asset['files'], revision=asset['revision'],
                             offline=True)
    configure_execution_dtype(bundle, 'float16')
    report['checkpoint'] = asset
    return bundle, path, asset


def execute(args, directory, report):
    import mlx.core as mx
    from reference_tiled_attention import reference_compatible_attention as candidate

    from mlx_one.native_correctness import checkpoint_proof
    from mlx_one.native_qualification import qualification_source

    assert mx.metal.is_available(), 'Metal required; CPU substitution forbidden'
    mx.set_default_device(mx.gpu)
    from mlx_one.ops.attention import mlx_version
    assert mlx_version() == (0, 32, 2), 'This experiment is pinned to MLX 0.32.2'
    mx.set_memory_limit(report['memory_limit_bytes'])
    report['source'] = qualification_source()
    report['hardware'] = mx.device_info()
    report['candidate_sha256'] = digest(Path(__file__).with_name('reference_tiled_attention.py'))
    (directory / 'source').mkdir()
    for path in [Path(__file__), Path(__file__).with_name('reference_tiled_attention.py'),
                 Path(__file__).with_name('tiled_model_qualification.py')]:
        (directory / 'source' / path.name).write_bytes(path.read_bytes())
    if args.stage != 'captured':
        from mlx_one.inference_qualification import validate_integrity
        assert args.correctness_report, 'source-bound numerical report required'
        assert args.correctness_report.name == 'results.json'
        validate_integrity(args.correctness_report.parent, required={'results.json'})
        proof = json.loads(args.correctness_report.read_text())
        assert proof['status'] == 'PASS', 'numerical prerequisite failed'
        assert proof['source'] == report['source'], 'stale numerical source/test identity'
        assert proof['candidate_sha256'] == report['candidate_sha256']
        assert proof['query_tile'] == args.tile
        assert proof['hardware'] == report['hardware'], 'different numerical hardware'
        if args.stage == 'model-performance':
            assert proof['stage'] == 'model', 'complete checkpoint numerical parity required'
            assert proof['production_integrated'] == args.integrated, 'different execution route'
            assert proof['comparison_count'] == 3818 and proof['failing_comparisons'] == 0
            assert proof['core_comparison_count'] == 2100
            assert proof['fp16_boundary_comparison_count'] == 2100
            assert proof['failing_core_comparisons'] == proof['failing_fp16_boundaries'] == 0
            assert proof['greedy_comparisons'] == proof['greedy_passed'] == 66
            assert proof['cleanup_passed'] and proof['total_fp16_bit_unequal'] == 0
            if args.integrated:
                assert proof['actual_candidate_backend'] == 'native_gather_tiled'
                assert proof['actual_candidate_calls'] == 2100
                assert proof['actual_fallback_calls'] == proof['actual_native_sdpa_calls'] == 0
                assert proof['unverified_candidate_dispatch_calls'] == 0
                uninstrumented = proof['uninstrumented_model']
                assert uninstrumented['status'] == 'PASS'
                assert uninstrumented['comparison_count'] == 3818
                assert uninstrumented['failing_comparisons'] == 0
                assert uninstrumented['actual_candidate_calls'] == 2100
                assert uninstrumented['actual_fallback_calls'] == 0
                assert uninstrumented['cleanup_passed']
        elif args.stage == 'operators':
            assert proof['stage'] in {'model', 'captured'}
        else:
            assert proof['stage'] == 'captured'
        report['numerical_prerequisite'] = {'path': str(args.correctness_report.resolve()),
                                           'sha256': digest(args.correctness_report)}
    bundle, path, asset = load_bundle(args, report)
    try:
        if args.stage != 'captured':
            assert proof['checkpoint'] == asset, 'different numerical checkpoint identity'
        if args.stage == 'captured':
            captured(args, report, bundle, candidate)
        elif args.stage == 'model':
            from tiled_model_qualification import qualify_model
            qualify_model(bundle, candidate, args.tile, report, integrated=args.integrated)
            if args.integrated:
                report['uninstrumented_model'] = {'status': 'NOT_RUN',
                    'reason': 'strict per-core numerical prerequisite must pass first'}
                if report['status'] == 'PASS':
                    uninstrumented = {}
                    qualify_model(bundle, candidate, args.tile, uninstrumented,
                                  integrated=True, core_snapshots=False)
                    report['uninstrumented_model'] = uninstrumented
                    if uninstrumented['status'] != 'PASS':
                        report['status'] = 'FAIL'
        elif args.stage == 'operators':
            operators(args, report, bundle, candidate)
        else:
            model_performance(args, report, bundle, candidate)
    finally:
        report['checkpoint_unchanged'] = checkpoint_proof('qwen3-small', path) == asset
        report['source_unchanged'] = qualification_source() == report['source']
        bundle = None
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
    assert report['checkpoint_unchanged'] and report['source_unchanged']


def host(value):
    import mlx.core as mx
    import numpy as np
    mx.eval(value)
    return np.asarray(value.astype(mx.float32)).copy()


def compare(left, right):
    from tiled_model_qualification import boundary_metrics
    return boundary_metrics(host(left), host(right))


def projection_error(bundle, left, right, index=0):
    import mlx.core as mx

    from mlx_one.models.shared.execution import project

    layer = bundle.model.model.layers[index]
    def apply(value):
        value = value.transpose(0, 2, 1, 3).reshape(value.shape[0], value.shape[2], -1)
        return project(layer.self_attn.o_proj, value.astype(mx.float16), execution_dtype='float16')
    return compare(apply(left), apply(right))


def captured(args, report, bundle, candidate):
    import mlx.core as mx
    import numpy as np

    from mlx_one.ops.attention import attention_reference

    report['captured_inputs'] = {'directory': str(args.capture.resolve()),
        'npz_sha256': digest(args.capture / 'first-three-layers.npz'),
        'reference_trace_sha256': digest(args.capture / 'reference-trace.json')}
    assert report['captured_inputs']['npz_sha256'] == (
        'cdccca2626202161576e000cd8e4cc07ca45068257b016e72120727508b28667')
    assert report['captured_inputs']['reference_trace_sha256'] == (
        'f15243eecbc12331409daa864c388ff8c7403d52a54ae4024606c17ea03be470')
    events = json.loads((args.capture / 'reference-trace.json').read_text())
    rows = []
    with np.load(args.capture / 'first-three-layers.npz', allow_pickle=False) as values:
        for index in range(3):
            event = {e['name']: e for e in events if e['layer'] == index}
            q, k, v = [mx.array(values[event[name]['array_key']]) for name in
                       ['q_operand', 'k_committed', 'v_committed']]
            mask = mx.array(values[event['attention_mask']['array_key'] + '_raw_mask'])
            scale = event['attention_mask']['metadata']['scale']
            reference = attention_reference(q, k, v, scale=scale, mask=mask,
                                            stable=True, promote=True)
            actual = candidate(q, k, v, scale, mask=mask, query_tile=args.tile)
            recorded = mx.array(values[event['attention_output']['array_key']])
            row = {'layer': index, 'q_shape': list(q.shape), 'k_shape': list(k.shape),
                   'scale': scale, 'original_reference_replay': compare(recorded, reference),
                   'candidate': compare(reference, actual),
                   'output_projection': projection_error(bundle, reference, actual, index),
                   'actual_backend': 'native_gather_tiled'}
            # Alter forbidden V positions independently for one query. Equality
            # proves the additive mask, rather than a close average, was honored.
            masked_q = q[:, :, :1]
            altered_v = mx.concatenate([v[:, :, :1], v[:, :, 1:] + 100], axis=2)
            masked_original = candidate(masked_q, k, v, scale, mask=mask[:1], query_tile=args.tile)
            masked_changed = candidate(masked_q, k, altered_v, scale,
                                        mask=mask[:1], query_tile=args.tile)
            row['masked_value_invariance'] = compare(masked_original, masked_changed)
            rows.append(row)
            print(f'captured layer {index}: {row["candidate"]}', flush=True)
            if not all(row[key]['status'] == 'PASS' for key in
                       ['candidate', 'original_reference_replay', 'output_projection',
                        'masked_value_invariance']):
                break
    report['cases'] = rows
    report['status'] = 'PASS' if len(rows) == 3 and all(
        row[key]['status'] == 'PASS' for row in rows for key in
        ['candidate', 'original_reference_replay', 'output_projection',
         'masked_value_invariance']) else 'FAIL'


def timed_pair(reference, candidate, samples):
    import mlx.core as mx

    functions = {'reference': reference, 'candidate': candidate}
    for function in functions.values():
        for _ in range(2):
            output = function()
            mx.eval(output)
            del output
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    rows = {'reference': [], 'candidate': []}
    for index in range(samples):
        order = ['reference', 'candidate'] if index % 2 == 0 else ['candidate', 'reference']
        for name in order:
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
            base = mx.get_active_memory()
            mx.reset_peak_memory()
            start = time.perf_counter()
            output = functions[name]()
            mx.eval(output)
            mx.synchronize()
            elapsed = time.perf_counter() - start
            peak = mx.get_peak_memory()
            rows[name].append({'seconds': elapsed, 'baseline_active_bytes': base,
                'peak_mlx_bytes': peak, 'peak_increment_bytes': peak-base,
                'output_bytes': output.nbytes,
                'peak_increment_minus_output_bytes_proxy': max(0, peak-base-output.nbytes)})
            del output
    for name in rows:
        values = rows[name]
        rows[name] = {'samples': values,
                      'median_seconds': statistics.median(r['seconds'] for r in values),
                      'peak_mlx_bytes': max(r['peak_mlx_bytes'] for r in values),
                      'max_peak_increment_bytes': max(r['peak_increment_bytes'] for r in values),
                      'max_increment_minus_output_bytes_proxy': max(
                          r['peak_increment_minus_output_bytes_proxy'] for r in values)}
    rows['speedup_percent'] = (rows['reference']['median_seconds'] /
                               rows['candidate']['median_seconds'] - 1) * 100
    rows['latency_reduction_percent'] = (1 - rows['candidate']['median_seconds'] /
                                       rows['reference']['median_seconds']) * 100
    return rows


def layer0_operands(bundle, length):
    """Actual native checkpoint projections/RoPE, stopping before attention."""
    import mlx.core as mx

    from mlx_one.models.shared import transformer

    text = 'An engineer measures a quiet river, then records seven distinct observations. '
    ids = bundle.tokenizer.encode(text * (length // 8 + 8))[:length]
    assert len(ids) == length
    layer = bundle.model.model.layers[0]
    hidden = layer.input_layernorm(bundle.model.model.embed_tokens(mx.array([ids])))
    captured_values = []
    class Captured(Exception):
        pass
    def capture(q, k, v, **kwargs):
        captured_values.extend([q, k, v, kwargs['mask'], kwargs['scale']])
        raise Captured()
    try:
        with patch.object(transformer, 'attention', capture):
            # Real native KV append preserves the padded live-cache layout.
            # The private scratch cache is not shared with another workload.
            cache = bundle.model.make_cache()[0]
            layer.self_attn(hidden, mask=transformer.causal_mask(length), cache=cache)
    except Captured:
        pass
    assert len(captured_values) == 5
    mx.eval(captured_values[:4])
    return captured_values, ids


def operators(args, report, bundle, candidate):
    import mlx.core as mx

    from mlx_one.ops.attention import attention_reference

    rows = []
    for queries in [1, 7, 32, 64, 512, 2048, 4096]:
        length = max(64, queries)
        (full_q, k, v, full_mask, scale), ids = layer0_operands(bundle, length)
        q, mask = full_q[:, :, -queries:, :], full_mask[-queries:]
        # Conservative check before any full-score allocation.
        scratch_estimate = 8 * q.shape[0] * q.shape[1] * queries * length * 4
        if scratch_estimate + mx.get_active_memory() > report['memory_limit_bytes']:
            rows.append({'queries': queries, 'status': 'UNSUPPORTED',
                         'reason': 'conservative full-reference scratch exceeds configured limit'})
            del q, k, v, full_q, full_mask, mask
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
            continue
        reference = partial(attention_reference, q, k, v, scale=scale, mask=mask,
                            stable=True, promote=True)
        proposed = partial(candidate, q, k, v, scale, mask=mask, query_tile=args.tile)
        expected, actual = reference(), proposed()
        result = compare(expected, actual)
        projection = projection_error(bundle, expected, actual)
        row = {'queries': queries, 'keys': length, 'q_shape': list(q.shape),
               'k_shape': list(k.shape), 'status': 'PASS' if (
                   result['status'] == projection['status'] == 'PASS') else 'FAIL',
               'numerical': result, 'output_projection': projection,
               'prompt_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
               'operand_origin': 'actual native checkpoint layer0 projections/norm/RoPE and '
                                 'native cache append; last queries of full length',
               'positions': [length-queries, length-1],
               'reference_repeat_output_bytes': (k.nbytes+v.nbytes) * (q.shape[1]//k.shape[1]),
               'candidate_repeat_output_bytes': 0,
               'native_kv_input_bytes': k.nbytes+v.nbytes,
               'candidate_rhs_heads': k.shape[1], 'reference_rhs_heads': q.shape[1]}
        del expected, actual
        gc.collect()
        mx.clear_cache()
        if result['status'] == 'PASS' and projection['status'] == 'PASS':
            row['timing'] = timed_pair(reference, proposed, args.samples)
            for name in ['reference', 'candidate']:
                row['timing'][name]['attention_query_tokens_per_second'] = (
                    queries / row['timing'][name]['median_seconds'])
        rows.append(row)
        print(f'operator Q={queries}: {row}', flush=True)
        del reference, proposed, q, k, v, full_q, full_mask, mask
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
    report['cases'] = rows
    report['status'] = ('PASS' if len(rows) == 7 and all(r['status'] == 'PASS' for r in rows)
                        else 'FAIL')


def model_performance(args, report, bundle, candidate):
    import mlx.core as mx

    from mlx_one.engine.optimization_policy import NativeMathPolicy, math_execution
    from mlx_one.models.shared import transformer

    ids = bundle.tokenizer.encode(
        'An engineer measures a quiet river, then records seven distinct observations. ' * 8)[:64]
    forced = bundle.tokenizer.encode(' The measured result remains reproducible.' * 8)[:32]
    original = transformer.attention

    def route(proposed):
        cache, calls, hypothetical_repeat_bytes = bundle.model.make_cache(), 0, 0
        selected_backends = {}
        class DispatchCounter:
            capture_math_dispatch = True
            profile_math = False

            def __init__(self):
                self.backends = {}
                self.repeat_bytes = 0

            def __call__(self, event):
                if event.get('operation') == 'attention':
                    name = event['selected']
                    self.backends[name] = self.backends.get(name, 0) + 1
                elif event.get('operation') == 'kv_head_repeat':
                    self.repeat_bytes += event['shape']['bytes']
        observer = DispatchCounter() if args.integrated else None

        def core(q, k, v, *, scale, mask, stable, promote, decision):
            nonlocal calls, hypothetical_repeat_bytes
            calls += 1
            # Matched counter overhead on both routes; candidate does not repeat.
            hypothetical_repeat_bytes += (k.nbytes+v.nbytes) * (q.shape[1]//k.shape[1])
            selected = (decision.selected if args.integrated or not proposed
                        else 'native_gather_tiled')
            selected_backends[selected] = selected_backends.get(selected, 0) + 1
            if proposed:
                if not args.integrated:
                    return candidate(q, k, v, scale, mask=mask, query_tile=args.tile)
            return original(q, k, v, scale=scale, mask=mask, stable=stable,
                            promote=promote, decision=decision)

        mx.synchronize()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        result = {}
        try:
            with patch.object(transformer, 'attention', core), math_execution(
                    NativeMathPolicy(attention='native_gather_tiled' if proposed and
                                     args.integrated else 'reference'),
                    family=bundle.architecture, observer=observer):
                start = time.perf_counter()
                logits = bundle.model(mx.array([ids]), cache=cache).logits[:, -1:, :]
                mx.eval(logits)
                mx.synchronize()
                prefill_seconds = time.perf_counter() - start
                first_token = mx.argmax(logits, axis=-1)
                mx.eval(first_token)
                ttft_seconds = time.perf_counter() - start
                decode_start = time.perf_counter()
                for token in forced:
                    logits = bundle.model(mx.array([[token]]), cache=cache).logits[:, -1:, :]
                    mx.eval(logits)
                mx.synchronize()
                decode_seconds = time.perf_counter() - decode_start
                assert calls == 924 and all(c.offset == 96 for c in cache)
                expected_backend = 'native_gather_tiled' if proposed else 'reference'
                assert selected_backends == {expected_backend: 924}, selected_backends
                if observer is not None:
                    assert observer.backends == {expected_backend: 924}, observer.backends
                    assert observer.repeat_bytes == (0 if proposed else hypothetical_repeat_bytes)
                result.update(ttft_seconds=ttft_seconds, prefill_seconds=prefill_seconds,
                    prefill_tokens_per_second=64/prefill_seconds,
                    decode_seconds=decode_seconds, decode_tokens_per_second=32/decode_seconds,
                    baseline_active_bytes=base, peak_mlx_bytes=mx.get_peak_memory(),
                    peak_increment_bytes=mx.get_peak_memory()-base,
                    first_greedy_token=int(first_token.item()), completed_forced_steps=32,
                    selected_attention_calls=calls,
                    selected_backends=selected_backends,
                    actual_dispatch_counts=(observer.backends if observer is not None else None),
                    observed_gqa_repeat_output_bytes=(observer.repeat_bytes
                                                       if observer is not None else None),
                    dispatch_observation='production observer' if args.integrated else
                                         'explicit diagnostic function override',
                    actual_backend='native_gather_tiled' if proposed else 'reference',
                    materialized_gqa_repeat_output_bytes=(
                        0 if proposed else hypothetical_repeat_bytes),
                    hypothetical_reference_repeat_output_bytes=hypothetical_repeat_bytes)
        finally:
            for layer in cache:
                layer.reset()
            result['cleanup_offsets'] = [layer.offset for layer in cache]
            cache = None
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
        return result

    route(False)
    route(True)
    rows = {'reference': [], 'candidate': []}
    for index in range(args.samples):
        order = ['reference', 'candidate'] if index % 2 == 0 else ['candidate', 'reference']
        for name in order:
            result = route(name == 'candidate')
            rows[name].append(result)
            print(f'model sample {index}/{name}: {result}', flush=True)
    fields = ['ttft_seconds', 'prefill_tokens_per_second', 'decode_tokens_per_second',
              'peak_mlx_bytes', 'peak_increment_bytes']
    report['measurements'] = {
        name: {'samples': values, 'medians': {
            field: statistics.median(r[field] for r in values) for field in fields}}
        for name, values in rows.items()}
    left = report['measurements']['reference']['medians']
    right = report['measurements']['candidate']['medians']
    report['paired_speedup_percent'] = {
        'ttft_latency_reduction': (1-right['ttft_seconds']/left['ttft_seconds'])*100,
        'prefill_tps_increase': (right['prefill_tokens_per_second']/
                                left['prefill_tokens_per_second']-1)*100,
        'decode_tps_increase': (right['decode_tokens_per_second']/
                               left['decode_tokens_per_second']-1)*100,
        'peak_mlx_reduction': (1-right['peak_mlx_bytes']/left['peak_mlx_bytes'])*100}
    report['scope'] = ('B1 direct native model,64 prompt/32 forced continuation; no HTTP, '
                       'batching, tokenization, sampling or network timing. TTFT includes '
                       'prefill and first greedy argmax. Decode excludes argmax/sampling.')
    good = all(
        row['completed_forced_steps'] == 32 and row['selected_attention_calls'] == 924
        and not any(row['cleanup_offsets']) for values in rows.values() for row in values)
    report['status'] = 'PASS' if good else 'FAIL'


if __name__ == '__main__':
    sys.exit(main())
