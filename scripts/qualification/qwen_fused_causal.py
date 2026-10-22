#!/usr/bin/env python3
"""Bounded, opt-in Qwen3 causal diagnostics; never grants backend qualification."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

# Permit execution from the checked-out repository without installation changes.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-paths', type=Path,
                        default=Path('scripts/qualification/checkpoint-paths.local.json'))
    parser.add_argument('--output', type=Path,
                        default=Path('qualification/results/inference_optimization_qwen_causal'))
    parser.add_argument('--snapshot-mib', type=int, default=128)
    parser.add_argument('--runtime-validation', action='store_true',
                        help='validate guarded candidate policy; '
                             'reference fallback is not fused parity')
    parser.add_argument('--vector-validation', action='store_true',
                        help='force Qlen=1 FP32 vector replay through the complete workload')
    args = parser.parse_args()
    if args.runtime_validation and args.vector_validation:
        parser.error('choose one validation mode')
    if not 32 <= args.snapshot_mib <= 256:
        parser.error('--snapshot-mib must be between 32 and 256')
    directory = args.output.resolve() / str(time.time_ns()) / 'causal'
    directory.mkdir(parents=True)
    report = {'schema_version': 'qwen-fused-causal-v1', 'status': 'RUNNING',
              'qualification_promoted': False, 'performance': 'BLOCKED_NUMERICAL_GATE',
              'command': sys.argv, 'criteria': {'float32': [0.0001, 0.0001],
                                               'float16': [0.01, 0.01]},
              'snapshot_limit_bytes': args.snapshot_mib * 1024**2,
              'scope': 'B1 full prefill; first three layers; all 28 single-layer controls'}
    (directory / 'command.json').write_text(json.dumps(sys.argv, indent=2) + '\n')
    try:
        execute(args, directory, report)
    except BaseException as exc:
        report.update(status='FAIL', exception={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        (directory / 'causal-results.json').write_text(json.dumps(report, indent=2) + '\n')
        from mlx_one.inference_qualification import write_integrity
        write_integrity(directory)
        print(directory, flush=True)
    return 0 if report['status'] == 'PASS' else 1


def execute(args, directory, report):
    import mlx.core as mx
    import numpy as np

    from mlx_one.engine.kernel_dispatch import KernelDecision, sdpa_capability
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.engine.optimization_policy import NativeMathPolicy, math_execution
    from mlx_one.models.shared import transformer
    from mlx_one.native_correctness import checkpoint_proof
    from mlx_one.native_qualification import qualification_source
    from mlx_one.ops.attention import attention_fused, attention_reference, mlx_version
    from mlx_one.ops.numerical_trace import compare_tensors, compare_traces, numerical_trace
    from mlx_one.text import load_text_model

    assert mx.metal.is_available(), 'Metal required; no CPU substitution'
    mx.set_default_device(mx.gpu)
    report['source'] = qualification_source()
    report['diagnostic_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_files = ['src/mlx_one/ops/attention.py', 'src/mlx_one/models/shared/transformer.py',
                    'src/mlx_one/ops/numerical_trace.py', 'src/mlx_one/models/shared/execution.py',
                    'src/mlx_one/engine/kernel_dispatch.py',
                    'tests/optimization/test_fused_attention_native.py', __file__]
    for item in source_files:
        path = Path(item)
        target = directory / 'source' / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    path = json.loads(args.checkpoint_paths.read_text())['qwen3-small']
    asset = checkpoint_proof('qwen3-small', path)
    report['checkpoint'] = asset
    bundle = load_text_model(asset['model_id'], checkpoint_path=path,
                             expected_files=asset['files'], revision=asset['revision'],
                             offline=True)
    configure_execution_dtype(bundle, 'float16')
    layers = bundle.model.model.layers
    for index, layer in enumerate(layers):
        layer._mlx_one_trace_layer = index
        layer.self_attn._mlx_one_trace_layer = index
    ids = bundle.tokenizer.encode(
        'An engineer measures a quiet river, then records seven distinct observations. ' * 8)[:64]
    forced = bundle.tokenizer.encode(' The measured result remains reproducible.' * 8)[:32]
    assert len(ids) == 64 and len(forced) == 32
    report.update(prompt_ids=ids, forced_ids=forced, chunk_size=7, continuation_steps=32)
    if args.runtime_validation or args.vector_validation:
        runtime_validation(bundle, ids, forced, report, force_vector=args.vector_validation)
        report['checkpoint_unchanged'] = checkpoint_proof('qwen3-small', path) == asset
        report['source_unchanged'] = qualification_source() == report['source']
        assert report['checkpoint_unchanged'] and report['source_unchanged']
        return
    arrays, capture_bytes = {}, 0
    dispatches = []
    original_attention = transformer.attention

    def snapshot(value):
        if value is None or isinstance(value, str):
            return value, {'shape': None, 'dtype': type(value).__name__, 'strides': None,
                           'strides_available': False}
        mx.eval(value)
        # Exported strides are read before snapshot copying. The native Python
        # API does not expose device strides; explicitly distinguish that fact.
        view = np.asarray(value)
        metadata = {'shape': list(value.shape), 'dtype': str(value.dtype).rsplit('.', 1)[-1],
                    'strides': None, 'strides_available': False,
                    'exported_host_strides': [s // view.itemsize for s in view.strides],
                    'exported_host_strides_unit': 'elements',
                    'native_strides_reason': 'MLX 0.32.2 Python array has no strides property'}
        return view.copy(), metadata

    def metric(left, right, dtype='float16'):
        left_value, lm = snapshot(left)
        r, rm = snapshot(right)
        return compare_tensors(left_value, r, dtype=dtype, reference_metadata=lm,
                               candidate_metadata=rm)

    def forward(fused_layers=(), *, capture=False, rounding_oracle=False):
        nonlocal capture_bytes
        events, hidden, cache, call = [], {}, bundle.model.make_cache(), 0
        selected = frozenset(fused_layers)

        def observe(event):
            nonlocal capture_bytes
            index, name = event['layer'], event['name']
            if name == 'hidden_output':
                hidden[index] = event['tensor']
            if not capture or index >= 3:
                return
            value, metadata = snapshot(event['tensor'])
            event = {**event, 'tensor': value, 'tensor_metadata': metadata}
            if name == 'attention_mask' and isinstance(value, np.ndarray):
                # Exact support semantics include legal -inf. Preserve the raw
                # additive mask separately for identical-operand replay.
                event['raw_mask'] = value
                event['tensor'] = np.isfinite(value)
                event['tensor_metadata'] = {'shape': list(value.shape), 'dtype': 'bool',
                                            'strides': None, 'strides_available': False}
            events.append(event)
            if isinstance(value, np.ndarray):
                capture_bytes += value.nbytes
                if capture_bytes > report['snapshot_limit_bytes']:
                    raise RuntimeError('snapshot byte limit exceeded; diagnostic incomplete')

        def select(q, k, v, *, mask=None, history=0, requires_stable_rounding=False):
            nonlocal call
            index = call
            call += 1
            ks, vs = list(k.shape), list(v.shape)
            ks[2] += history
            vs[2] += history
            decision = (sdpa_capability(q_shape=q.shape, k_shape=ks, v_shape=vs,
                                       mlx_version=mlx_version()) if index in selected
                        else KernelDecision('reference', 'single-layer causal control'))
            dispatches.append({'run': list(selected), 'rounding_oracle': rounding_oracle,
                               'layer': index, 'q': list(q.shape), 'k': ks,
                               'selected': decision.selected, 'reason': decision.reason})
            return decision

        def attention_control(q, k, v, **kwargs):
            # Explicit forced diagnostic, separate from the guarded runtime API.
            output = (attention_fused(q, k, v, scale=kwargs['scale'], mask=kwargs['mask'],
                                      accumulation='float32')
                      if kwargs['decision'].selected == 'mlx_sdpa'
                      else original_attention(q, k, v, **kwargs))
            if rounding_oracle and kwargs['decision'].selected == 'mlx_sdpa':
                mx.eval(output)  # Ensure the fused diagnostic really executes.
                reference = attention_reference(q, k, v, scale=kwargs['scale'],
                    mask=kwargs['mask'], stable=kwargs['stable'], promote=kwargs['promote'])
                # Counterfactual only: replace precisely the FP16 cast boundary.
                # Never used as a runtime correction or qualification result.
                return reference.astype(mx.float16).astype(mx.float32)
            return output

        with patch.object(transformer, 'select_attention', select), \
             patch.object(transformer, 'attention', attention_control), \
             numerical_trace(observe, max_layers=len(layers), max_events=1024), \
             math_execution(NativeMathPolicy(attention='reference'), family=bundle.architecture):
            value = bundle.model(mx.array([ids]), cache=cache).logits[:, -1:, :]
            mx.eval(value, list(hidden.values()))
        assert call == len(layers)
        return value, cache, hidden, events

    baseline = forward(capture=True)
    candidate = forward(range(len(layers)), capture=True)
    report['full_prefill_logits'] = metric(baseline[0], candidate[0])
    report['full_prefill_states'] = [
        {'layer': index, 'component': component, **metric(a, b)}
        for index, (left, right) in enumerate(zip(baseline[1], candidate[1], strict=True))
        for component, (a, b) in enumerate(zip(left.state(), right.state(), strict=True))]
    trace_comparison = compare_traces(baseline[3], candidate[3])
    report['trace_comparison'] = trace_comparison
    report['earliest_changed_tensor'] = next((r for r in trace_comparison['comparisons']
                                              if r.get('changed_elements', 0)), None)
    # Offsets, scale and positions are exact metadata invariants; requested/selected
    # backend labels are expected to differ and are not equality gates.
    report['trace_metadata_checks'] = [
        {'layer': left['layer'], 'name': left['name'], 'key': key,
         'reference': left['metadata'][key],
         'candidate': r['metadata'].get(key),
         'status': 'PASS' if left['metadata'][key] == r['metadata'].get(key) else 'FAIL'}
        for left, r in zip(baseline[3], candidate[3], strict=True)
        for key in ('offset', 'scale', 'length', 'effective_positions') if key in left['metadata']]
    for execution, events in [('reference', baseline[3]), ('fused', candidate[3])]:
        metadata = []
        for event in events:
            key = f'{execution}_{event["sequence"]}_{event["layer"]}_{event["name"]}'
            value = event['tensor']
            if isinstance(value, np.ndarray):
                arrays[key] = value
            raw = event.get('raw_mask')
            if raw is not None:
                arrays[key + '_raw_mask'] = raw
            metadata.append({k: v for k, v in event.items() if k not in {'tensor', 'raw_mask'}}
                            | {'array_key': key if isinstance(value, np.ndarray) else None,
                               'scalar': value if not isinstance(value, np.ndarray) else None})
        (directory / f'{execution}-trace.json').write_text(json.dumps(metadata, indent=2) + '\n')
    np.savez(directory / 'first-three-layers.npz', **arrays)
    report['snapshot_bytes'] = capture_bytes

    replay = []
    arithmetic = []
    for index in range(3):
        events = {e['name']: e for e in baseline[3] if e['layer'] == index}
        q, k, v = [mx.array(events[name]['tensor']) for name in
                   ('q_operand', 'k_committed', 'v_committed')]
        mask_event = events['attention_mask']
        mask = mx.array(mask_event['raw_mask'])
        scale = mask_event['metadata']['scale']
        reference = attention_reference(q, k, v, scale=scale, mask=mask, stable=True, promote=True)
        from mlx_one.models.shared.execution import attention_matmul
        repeated_k = mx.repeat(k, q.shape[1] // k.shape[1], axis=1).astype(mx.float32)
        repeated_v = mx.repeat(v, q.shape[1] // v.shape[1], axis=1).astype(mx.float32)
        scores = attention_matmul(q, repeated_k.transpose(0, 1, 3, 2)) * scale
        probability = mx.softmax(scores + mask, axis=-1, precise=True)
        batched_scores = (q.astype(mx.float32) @ repeated_k.transpose(0, 1, 3, 2)) * scale
        batched_probability = mx.softmax(batched_scores + mask, axis=-1, precise=True)
        # Source-guided counterfactuals isolate operation order, not inaccessible
        # private Metal intermediates. None is used as a production correction.
        exp2_scores = (scores + mask) * mx.array(1.4426950408889634, dtype=mx.float32)
        # Python MLX 0.32.2 exposes power, not Metal's private fast::exp2.
        exponent = mx.power(mx.array(2.0, dtype=mx.float32),
                            exp2_scores - mx.max(exp2_scores, axis=-1, keepdims=True))
        exp2_probability = exponent / mx.sum(exponent, axis=-1, keepdims=True)
        arithmetic.append({'layer': index, 'scope': 'counterfactual existing primitives',
            'private_fused_intermediates_available': False,
            'base2_control': 'mx.power(2,x); not a measurement of Metal fast::exp2',
            'qk_batched_vs_rowwise': metric(scores, batched_scores, 'float32'),
            'probabilities_batched_qk': metric(probability, batched_probability, 'float32'),
            'output_batched_qk': metric(reference,
                attention_matmul(batched_probability, repeated_v), 'float32'),
            'output_batched_pv': metric(reference, probability @ repeated_v, 'float32'),
            'probabilities_exp2': metric(probability, exp2_probability, 'float32'),
            'output_exp2': metric(reference,
                attention_matmul(exp2_probability, repeated_v), 'float32')})
        variants = {
            'native_fp16': attention_fused(q, k, v, scale=scale,
                                           mask=mask.astype(q.dtype), accumulation='native'),
            'fp32_full': attention_fused(q, k, v, scale=scale, mask=mask, accumulation='float32'),
            'fp32_vector': mx.concatenate([
                attention_fused(q[:, :, j:j+1], k, v, scale=scale, mask=mask[j:j+1],
                                accumulation='float32') for j in range(q.shape[2])], axis=2)}
        for name, actual in variants.items():
            row = {'layer': index, 'variant': name, 'force_fused': True,
                   'mask': 'identical lower-right additive mask; FP16 mask values exactly 0/-inf',
                   'output': metric(reference, actual, 'float32'),
                   'cast': metric(reference.astype(mx.float16), actual.astype(mx.float16)),
                   'operand_precision_output': metric(reference, actual,
                                    'float16' if name == 'native_fp16' else 'float32')}
            ref_input = reference.transpose(0, 2, 1, 3).reshape(1, 64, -1).astype(mx.float16)
            got_input = actual.transpose(0, 2, 1, 3).reshape(1, 64, -1).astype(mx.float16)
            from mlx_one.models.shared.execution import project
            row['output_projection'] = metric(
                project(layers[index].self_attn.o_proj, ref_input, execution_dtype='float16'),
                project(layers[index].self_attn.o_proj, got_input, execution_dtype='float16'))
            replay.append(row)
        print(f'replayed layer {index}', flush=True)
    report['identical_operand_replay'] = replay
    report['arithmetic_controls'] = arithmetic

    controls = []
    for index in range(len(layers)):
        run = forward([index])
        states = []
        for layer_index, (left, right) in enumerate(zip(baseline[1], run[1], strict=True)):
            for component, (a, b) in enumerate(zip(left.state(), right.state(), strict=True)):
                row = metric(a, b)
                states.append({'layer': layer_index, 'component': component, **row})
        output = metric(baseline[2][index], run[2][index])
        logit = metric(baseline[0], run[0])
        controls.append({'fused_layer': index, 'layer_output': output, 'logits': logit,
                         'state_failures': sum(r['status'] != 'PASS' for r in states),
                         'first_state_failure': next(
                             (r for r in states if r['status'] != 'PASS'), None),
                         'max_state_error': max(r['max_abs_error'] for r in states),
                         'states': states})
        print(f'isolated layer {index}: '
              f'state failures={controls[-1]["state_failures"]}', flush=True)
        del run
    report['single_layer_controls'] = controls
    oracle = forward([0], rounding_oracle=True)
    report['rounding_boundary_counterfactual'] = {
        'scope': 'diagnostic oracle only; fused executes then reference FP16 boundary substituted',
        'qualified': False, 'logits': metric(baseline[0], oracle[0]),
        'layer_outputs': [metric(baseline[2][i], oracle[2][i]) for i in range(len(layers))],
        'states': [metric(a, b) for left, r in zip(baseline[1], oracle[1], strict=True)
                   for a, b in zip(left.state(), r.state(), strict=True)]}
    report['dispatches'] = dispatches
    report['checkpoint_unchanged'] = checkpoint_proof('qwen3-small', path) == asset
    report['source_unchanged'] = qualification_source() == report['source']
    checks = [trace_comparison['status'] == 'PASS',
              report['full_prefill_logits']['status'] == 'PASS',
              all(r['status'] == 'PASS' for r in report['full_prefill_states']),
              all(r['status'] == 'PASS' for r in report['trace_metadata_checks'])]
    report['status'] = 'PASS' if all(checks) else 'FAIL'
    report['status_scope'] = 'numerical diagnostic outcome; no qualification/promotion'
    for run in (baseline, candidate, oracle):
        for cache in run[1]:
            cache.reset()
    report['cleanup_offsets'] = [[cache.offset for cache in run[1]] for run in
                                 (baseline, candidate, oracle)]
    del baseline, candidate, oracle
    bundle = None
    gc.collect()
    mx.clear_cache()
    assert report['checkpoint_unchanged'] and report['source_unchanged']


def runtime_validation(bundle, ids, forced, report, *, force_vector=False):
    """The exact 3818-comparison workload, truthfully labelled reference fallback."""
    import mlx.core as mx
    import numpy as np

    from mlx_one.engine.optimization_policy import NativeMathPolicy, math_execution
    from mlx_one.models.shared import transformer
    from mlx_one.ops.attention import attention_fused, mlx_version, validate_attention_inputs
    from mlx_one.ops.numerical_trace import compare_tensors
    from mlx_one.ops.profiling import dispatch

    caches = [bundle.model.make_cache() for _ in range(3)]
    comparisons, dispatches = [], []
    vector_calls = 0

    def observe(event):
        if event.get('kind') == 'math_dispatch' and event.get('operation') == 'attention':
            dispatches.append(event)

    observe.capture_math_dispatch = True

    def vector_select(q, k, v, *, mask=None, history=0, requires_stable_rounding=False):
        from mlx_one.engine.kernel_dispatch import sdpa_capability

        validate_attention_inputs(q, k, v, mask=mask, history=history)
        qs, ks, vs = list(q.shape), list(k.shape), list(v.shape)
        qs[2] = 1
        ks[2] += history
        vs[2] += history
        return sdpa_capability(q_shape=qs, k_shape=ks, v_shape=vs, mlx_version=mlx_version())

    def vector_attention(q, k, v, *, scale, mask, stable, promote, decision):
        nonlocal vector_calls
        if decision.selected != 'mlx_sdpa':
            raise RuntimeError(f'unsupported forced vector diagnostic: {decision.reason}')
        dispatch('attention', 'mlx_sdpa', 'mlx_sdpa', 'forced vector diagnostic, unqualified',
                 q=list(q.shape), k=list(k.shape), accumulation_operands='float32')
        if isinstance(mask, str):
            offset = k.shape[2] - q.shape[2]
            mask = mx.triu(mx.full((q.shape[2], k.shape[2]), -float('inf')), k=offset+1)
        outputs = []
        for index in range(q.shape[2]):
            sliced_mask = None if mask is None else mask[..., index:index+1, :]
            outputs.append(attention_fused(q[:, :, index:index+1], k, v, scale=scale,
                                          mask=sliced_mask, accumulation='float32'))
            vector_calls += 1
        return mx.concatenate(outputs, axis=2)

    def forward(cache, tokens, candidate=False):
        with ExitStack() as stack, math_execution(
                NativeMathPolicy(attention='mlx_sdpa' if candidate else 'reference'),
                family=bundle.architecture, observer=observe):
            if candidate and force_vector:
                stack.enter_context(patch.object(transformer, 'select_attention', vector_select))
                stack.enter_context(patch.object(transformer, 'attention', vector_attention))
            value = bundle.model(mx.array([tokens]), cache=cache).logits[:, -1:, :]
            mx.eval(value)
        return value

    def compare(left, right, stage, state=False):
        left_value = np.asarray(left.astype(mx.float32))
        right_value = np.asarray(right.astype(mx.float32))
        row = compare_tensors(left_value, right_value, dtype='float16')
        row.update(stage=stage, kind='state' if state else 'logits')
        if not state:
            row['greedy_agreement'] = bool(np.all(left_value.argmax(-1) == right_value.argmax(-1)))
        comparisons.append(row)

    def states(others, stage):
        for lane, other in enumerate(others, 1):
            for index, (left, right) in enumerate(zip(caches[0], other, strict=True)):
                assert left.offset == right.offset
                for component, (a, b) in enumerate(zip(left.state(), right.state(), strict=True)):
                    compare(a, b, f'{stage}/lane={lane}/layer={index}/component={component}', True)

    try:
        reference = forward(caches[0], ids)
        compare(reference, forward(caches[1], ids, True), 'full_prefill')
        states(caches[1:2], 'full_prefill_kv')
        for start in range(0, len(ids), 7):
            chunked = forward(caches[2], ids[start:start+7], True)
        compare(reference, chunked, 'chunked_prefill')
        states(caches[1:], 'chunked_prefill_kv')
        for step, token in enumerate(forced):
            reference = forward(caches[0], [token])
            for cache in caches[1:]:
                compare(reference, forward(cache, [token], True), f'continuation/{step}')
            states(caches[1:], f'continuation_kv/{step}')
        assert len(comparisons) == 3818
        candidate_dispatch = [r for r in dispatches if r['requested'] == 'mlx_sdpa']
        expected_backend = 'mlx_sdpa' if force_vector else 'reference'
        good = (len(candidate_dispatch) == 2100 and
                all(r['selected'] == expected_backend for r in candidate_dispatch) and
                all(r['status'] == 'PASS' and r.get('greedy_agreement', True)
                    for r in comparisons))
        report.update(status='PASS' if good else 'FAIL', comparisons=comparisons,
                      dispatches=dispatches, comparison_count=len(comparisons),
                      failing_comparisons=sum(r['status'] != 'PASS' for r in comparisons),
                      actual_candidate_backend=expected_backend, actual_fused_calls=vector_calls,
                      candidate_reference_calls=0 if force_vector else len(candidate_dispatch),
                      max_error=max(r['max_abs_error'] for r in comparisons),
                      scope=('forced FP32 Qlen=1 vector; diagnostic only' if force_vector else
                             'stable INT8 safety fallback; NOT fused qualification'),
                      performance='BLOCKED_DIRECT_FUSED_NUMERICAL_GATE')
        if not force_vector:
            assert good, 'guarded reference fallback validation failed'
    finally:
        for cache in caches:
            for layer in cache:
                layer.reset()
        report['cleanup_offsets'] = [[layer.offset for layer in cache] for cache in caches]
        caches.clear()
        gc.collect()
        mx.clear_cache()


if __name__ == '__main__':
    sys.exit(main())
