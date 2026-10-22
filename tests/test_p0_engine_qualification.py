"""Backend-free evidence, asset, isolation, and release-decision regressions."""
import json
import struct
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_one import p0_engine_qualification as p0


def fixture_assets(tmp_path, *, bits=8):
    config = {"model_type": "qwen3", "quantization": {"bits": bits, "group_size": 64}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text('{}')
    (tmp_path / "tokenizer_config.json").write_text('{}')
    packed = 64 * bits // 32
    header = {"linear.weight": {"dtype": "U32", "shape": [2, packed],
                                "data_offsets": [0, packed * 8]},
              "linear.scales": {"dtype": "F16", "shape": [2, 1],
                                "data_offsets": [packed * 8, packed * 8 + 4]},
              "linear.biases": {"dtype": "F16", "shape": [2, 1],
                                "data_offsets": [packed * 8 + 4, packed * 8 + 8]}}
    encoded = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack('<Q', len(encoded)) + encoded + bytes(packed * 8 + 8))
    return tmp_path


def test_complete_inventory_and_scoped_model_decision():
    assert len(p0.MODELS) == 5
    inventory = p0.case_inventory()
    assert len({r['case_id'] for r in inventory}) == len(inventory)
    assert all(any(r['case_id'] == f'{letter}{i:02}' for r in inventory)
               for letter, (_, count) in p0.GROUPS.items() for i in range(1, count + 1))
    rows = [{**r, 'status': 'PASS'} for r in inventory]
    assert p0.decide_model(rows) == 'QUALIFIED'
    rows[0]['status'] = 'NOT_RUN'
    assert p0.decide_model(rows) == 'NOT_TESTED'
    rows[0]['status'] = 'FAIL'
    assert p0.decide_model(rows) == 'FAILED'
    rows[0]['status'] = 'BLOCKED'
    assert p0.decide_model(rows) == 'BLOCKED'
    rows[0].update(status='NOT_SUPPORTED', applicable=False)
    assert p0.decide_model(rows) == 'QUALIFIED'


def test_admission_inventory_includes_independent_worker():
    cases = [r for r in p0.case_inventory() if r['suite'] == 'admission']
    assert [r['case_id'] for r in cases] == [f'I{i:02}' for i in range(1, 7)]
    assert all(r['mandatory'] for r in cases)


def test_diagnostic_run_cannot_qualify_even_complete_fixtures(tmp_path):
    model = next(iter(p0.MODELS))
    rows = [{**case, 'status': 'PASS', 'details': {}} for case in p0.case_inventory()]
    summary = p0.aggregate(tmp_path, {'source': {}, 'policy': {'diagnostic': True}}, {},
                           {model: [{'cases': rows}]})
    assert summary['models'][model] == 'NOT_TESTED'
    assert not summary['qualification_promoted']


def test_request_rss_peak_uses_only_in_range_samples(tmp_path, monkeypatch):
    directory = tmp_path / 'worker'
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    monkeypatch.setattr(p0.subprocess, 'Popen', lambda *_args, **_kwargs: object())
    monkeypatch.setattr(p0.time, 'monotonic', lambda: 100.)

    def supervise(*_args, **kwargs):
        kwargs['clock']()
        p0.atomic_json(directory / 'results.json', {
            'cases': [{'case_id': 'H8192', 'status': 'PASS'}],
            'benchmarks': [{'case_id': 'H8192', 'started_monotonic': 102.,
                            'finished_monotonic': 104.}, {'case_id': 'H512'}]})
        return {'status': 'completed', 'rss_samples': [
            {'elapsed_seconds': 1., 'rss_bytes': 999},
            {'elapsed_seconds': 2., 'rss_bytes': 400},
            {'elapsed_seconds': 3., 'rss_bytes': 500},
            {'elapsed_seconds': 5., 'rss_bytes': 999}]}

    monkeypatch.setattr(p0, 'supervise', supervise)
    result = p0.run_worker(manifest, 'benchmark', directory, 1024)
    assert result['benchmarks'][0]['rss_peak_bytes'] == 500
    assert result['benchmarks'][1]['rss_peak_bytes'] is None


def test_watchdog_failure_cannot_preserve_passed_qualification(tmp_path, monkeypatch):
    directory = tmp_path / 'worker'
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    monkeypatch.setattr(p0.subprocess, 'Popen', lambda *_args, **_kwargs: object())

    def supervise(*_args, **_kwargs):
        p0.atomic_json(directory / 'results.json', {
            'cases': [{'case_id': 'I06', 'status': 'PASS'}]})
        return {'status': 'memory_watchdog_terminated', 'rss_samples': []}

    monkeypatch.setattr(p0, 'supervise', supervise)
    result = p0.run_worker(manifest, 'admission_independent', directory, 1024)
    assert len(result['cases']) == 1
    assert result['cases'][0]['status'] == 'BLOCKED'
    assert result['cases'][0]['observed_status'] == 'PASS'
    assert result['cases'][0]['shared_blocker']


@pytest.mark.parametrize(('count', 'expected'), [(0, 'NO_GO'), (3, 'NO_GO'),
                                               (4, 'CONDITIONAL_GO'), (5, 'GO')])
def test_release_minimum_and_shared_blocker(count, expected):
    models = {str(i): 'QUALIFIED' if i < count else 'FAILED' for i in range(5)}
    assert p0.release_decision(models) == expected
    assert p0.release_decision(models, shared_blocker=True) == 'NO_GO'


def test_quantized_headers_count_logical_parameters_not_packed_elements(tmp_path):
    path = fixture_assets(tmp_path)
    data = p0.validate_assets('mlx-community/Qwen3-0.6B-8bit', path, 'a' * 40)
    assert data['logical_parameters_before_tie_deduplication'] == 128
    assert data['packed_modules'] == 1
    assert data['stored_tensor_elements'] == 36
    assert all(len(v['sha256']) == 64 for v in data['assets'].values())


def test_wrong_precision_and_missing_tokenizer_fail_closed(tmp_path):
    fixture_assets(tmp_path, bits=4)
    with pytest.raises(ValueError, match='precision'):
        p0.validate_assets('mlx-community/Qwen3-0.6B-8bit', tmp_path, 'a' * 40)
    fixture_assets(tmp_path)
    (tmp_path / 'tokenizer.json').unlink()
    with pytest.raises(ValueError, match='tokenizer'):
        p0.validate_assets('mlx-community/Qwen3-0.6B-8bit', tmp_path, 'a' * 40)


def test_missing_indexed_shard_fails(tmp_path):
    fixture_assets(tmp_path)
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map': {'linear.weight': 'missing.safetensors'}}))
    with pytest.raises(ValueError, match='indexed tensor'):
        p0.validate_assets('mlx-community/Qwen3-0.6B-8bit', tmp_path, 'a' * 40)


def test_atomic_json_preserves_previous_result_on_failure(tmp_path, monkeypatch):
    path = tmp_path / 'result.json'
    p0.atomic_json(path, {'status': 'PASS'})
    monkeypatch.setattr(p0.json, 'dump', lambda *_args, **_kwargs: (_ for _ in ()).throw(
        OSError('injected write failure')))
    with pytest.raises(OSError):
        p0.atomic_json(path, {'status': 'FAIL'})
    assert json.loads(path.read_text()) == {'status': 'PASS'}


def test_aggregation_keeps_filtered_models_and_cases_not_run(tmp_path):
    result = p0.aggregate(tmp_path, {'source': {'git_revision': 'a' * 40}}, {}, {})
    assert result['decision'] == 'NO_GO'
    assert set(result['models'].values()) == {'NOT_TESTED'}
    rows = json.loads((tmp_path / 'qualification_results.json').read_text())
    assert len(rows) == 5 * len(p0.case_inventory())
    assert {r['status'] for r in rows} == {'NOT_RUN'}


def test_failed_worker_retains_completed_cases_and_blocks_remainder(tmp_path, monkeypatch):
    class Process:
        pass

    monkeypatch.setattr(p0.subprocess, 'Popen', lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(p0, 'supervise', lambda *_args, **_kwargs: {
        'status': 'timeout', 'exit_code': -15, 'rss_samples': []})
    (tmp_path / 'manifest.json').write_text('{}')
    result = p0.run_worker(tmp_path / 'manifest.json', 'state', tmp_path / 'worker', 1024)
    assert result['supervision']['status'] == 'timeout'
    assert {r['status'] for r in result['cases']} == {'BLOCKED'}
    assert {r['case_id'] for r in result['cases']} == {
        r['case_id'] for r in p0.case_inventory() if r['suite'] == 'state'}


@pytest.mark.parametrize('bits', [4, 8])
def test_fp16_conversion_preserves_native_affine_payload(bits, monkeypatch):
    from mlx_one.engine.adapters import quantized_modules
    from mlx_one.engine.modes import configure_execution_dtype

    payload = np.arange(2 * (64 * bits // 32), dtype=np.uint32).reshape(2, -1)
    module = SimpleNamespace(weight=payload, scales=np.ones((2, 1), np.float32),
        biases=np.zeros((2, 1), np.float32), bits=bits, group_size=64, mode='affine')

    class Model:
        def parameters(self):
            return {'linear': {n: getattr(module, n) for n in ('weight', 'scales', 'biases')}}

        def named_modules(self):
            return [('linear', module)]

        def set_dtype(self, dtype, predicate):
            for name in ('weight', 'scales', 'biases'):
                value = getattr(module, name)
                if predicate(value.dtype):
                    setattr(module, name, value.astype(dtype))

    core = ModuleType('mlx.core')
    core.float16, core.eval = np.float16, lambda *_args: None
    package = ModuleType('mlx')
    package.core = core
    monkeypatch.setitem(sys.modules, 'mlx', package)
    monkeypatch.setitem(sys.modules, 'mlx.core', core)
    bundle = SimpleNamespace(model=Model(), architecture='lfm2',
                             quantization={'bits': bits, 'group_size': 64})
    configure_execution_dtype(bundle, 'float16')
    assert module.weight is payload
    assert module.scales.dtype == module.biases.dtype == np.float16
    if bits == 4:
        with pytest.raises(ValueError, match='INT8'):
            quantized_modules(bundle)  # Optimized INT8 eligibility remains strict.


def test_error_metrics_and_undefined_token_intervals():
    from mlx_one.engine.timing import RequestTiming
    from mlx_one.p0_engine_worker import error_metrics

    metric = error_metrics(np.array([1., 2.]), np.array([1.001, 2.]))
    assert metric['within_tolerance'] and metric['greedy_agreement']
    assert metric['max_absolute_error'] == pytest.approx(.001)
    timing = RequestTiming(0)
    assert timing.summary()['decode_tokens_per_second'] is None
    timing.observe({'kind': 'token', 'at': 5., 'token_id': 1})
    assert timing.summary()['decode_tokens_per_second'] is None
    timing.observe({'kind': 'token', 'at': 5.5, 'token_id': 2})
    assert timing.summary()['decode_tokens_per_second'] == 2


def test_truncated_weight_payload_is_not_a_complete_snapshot(tmp_path):
    fixture_assets(tmp_path)
    path = tmp_path / 'model.safetensors'
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match='payload'):
        p0.validate_assets('mlx-community/Qwen3-0.6B-8bit', tmp_path, 'a' * 40)


def test_exact_prompt_roundtrip_and_invalid_lengths():
    class Characters:
        def encode(self, text):
            return list(text)

        def decode(self, ids, **_kwargs):
            return ''.join(ids)

    for length in (1, 16, 128):
        prompt = p0.exact_prompt(Characters(), length)
        assert len(Characters().encode(prompt)) == length
    with pytest.raises(ValueError, match='positive'):
        p0.exact_prompt(Characters(), 0)


def test_download_reuses_exact_complete_pin_and_never_calls_online(tmp_path, monkeypatch):
    import huggingface_hub

    fixture_assets(tmp_path)
    calls = []

    def snapshot(model, **settings):
        calls.append((model, settings))
        assert settings['local_files_only'] is True
        assert settings['revision'] == 'a' * 40
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    outcome = p0.acquire('mlx-community/Qwen3-0.6B-8bit', offline=True, revision='a' * 40)
    assert outcome['status'] == 'PASS'
    assert len(calls) == 1


def test_download_failure_is_blocked_not_model_failure(monkeypatch):
    import huggingface_hub

    def unavailable(*_args, **_kwargs):
        raise OSError('fixture: network unavailable')

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', unavailable)
    result = p0.acquire('mlx-community/Qwen3-0.6B-8bit', offline=True,
                        revision='a' * 40, retries=3)
    assert result['status'] == 'BLOCKED'
    assert result['reason'] == 'BLOCKED_DOWNLOAD'
    assert len(result['attempts']) == 3
    assert all(r['revision'] == 'a' * 40 for r in result['attempts'])


def test_unknown_worker_case_or_status_rejected(tmp_path):
    with pytest.raises(ValueError, match='mapping'):
        p0.aggregate(tmp_path, {'source': {}}, {}, {'openai-community/gpt2': [
            {'cases': [{'case_id': 'INVENTED', 'status': 'PASS'}]}]})
    with pytest.raises(ValueError, match='mapping'):
        p0.aggregate(tmp_path, {'source': {}}, {}, {'openai-community/gpt2': [
            {'cases': [{'case_id': 'A01', 'status': 'maybe'}]}]})


def test_lazy_clone_is_created_on_device_owner():
    from mlx_one.engine.device_executor import DeviceExecutor
    from mlx_one.p0_engine_worker import owned_clone

    executor = DeviceExecutor()

    class Cache:
        def clone(self):
            assert executor.is_owner, 'lazy clone created on a different MLX stream owner'
            return object()

    try:
        with pytest.raises(AssertionError, match='stream owner'):
            Cache().clone()
        assert owned_clone(executor, 'model', Cache()) is not None
    finally:
        executor.close()


def test_conditional_release_does_not_hide_failed_mandatory_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(p0, 'preflight', lambda *_args: {
        'process_ceiling_bytes': 1024, 'status': 'PASS'})
    monkeypatch.setattr(p0, 'source_provenance', lambda *_args: {
        'source_sha256': 'a' * 64, 'git_revision': 'b' * 40})
    monkeypatch.setattr(p0, 'acquire', lambda model, **_kwargs: {
        'status': 'PASS', 'metadata': {'model_id': model, 'revision': 'c' * 40}})

    def worker(manifest, suite, *_args):
        model = json.loads(manifest.read_text())['metadata']['model_id']
        cases = []
        for case in p0.case_inventory():
            if case['suite'] != suite:
                continue
            status = 'FAIL' if model.endswith('Qwen3.5-0.8B-4bit') and (
                case['case_id'] == 'B01') else 'PASS'
            details = {'observed_parameters': {'logical_parameters': 1}} if (
                case['case_id'] == 'A03') else {}
            cases.append({'case_id': case['case_id'], 'status': status, 'details': details})
        return {'cases': cases, 'supervision': {'status': 'completed'}}

    monkeypatch.setattr(p0, 'run_worker', worker)
    code = p0.main(['--all', '--offline', '--output', str(tmp_path)])
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['decision'] == 'CONDITIONAL_GO'
    assert code != 0, 'a failing mandatory gate must produce nonzero qualification exit'


def test_filtered_suite_success_keeps_other_gates_unqualified():
    model = 'openai-community/gpt2'
    rows = [{**case, 'model_id': model,
             'status': 'PASS' if case['suite'] in {'startup', 'streaming'} else 'NOT_RUN'}
            for case in p0.case_inventory()]
    assert p0.qualification_exit(rows, [model], ['streaming']) == 0
    assert p0.decide_model(rows) == 'NOT_TESTED'
    assert p0.qualification_exit(rows, [model], list(p0.SUITES)) == 1


def test_selected_benchmark_exit_keeps_unexecuted_lengths_not_run():
    model = 'mlx-community/Qwen3-0.6B-8bit'
    rows = [{**case, 'model_id': model,
             'status': 'PASS' if case['suite'] == 'startup' or case['case_id'] == 'H8192'
             else 'NOT_RUN'} for case in p0.case_inventory()]
    assert p0.qualification_exit(rows, [model], ['benchmark'], prompt_tokens=[8192]) == 0
    assert p0.qualification_exit(rows, [model], ['benchmark']) == 1
    assert p0.decide_model(rows) == 'NOT_TESTED'


def test_lifecycle_includes_one_hundred_generations_before_reload(monkeypatch):
    from mlx_one.engine import pressure
    from mlx_one.p0_engine_worker import Cases

    events = []
    stats = SimpleNamespace(active_handles=0, reserved_bytes=0, to_dict=lambda: {})
    fake = SimpleNamespace(metadata={'model_id': 'fixture'}, rows={'memory': []},
        reload=lambda: events.append('reload'),
        generate=lambda *_args, **_kwargs: events.append('generate'), prompt=lambda x:x,
        save=lambda: None,
        mx=SimpleNamespace(synchronize=lambda: None, clear_cache=lambda: None,
            get_active_memory=lambda: 0, get_cache_memory=lambda: 0, get_peak_memory=lambda: 0),
        engine=SimpleNamespace(_cache_manager=SimpleNamespace(stats=lambda: stats),
            _device=SimpleNamespace(stats=lambda: {'queued':0}),
            _executor=SimpleNamespace(_work_queue=SimpleNamespace(qsize=lambda:0))))
    monkeypatch.setattr(pressure, 'sample_telemetry', lambda: SimpleNamespace(rss_bytes=100))
    Cases.cycles(fake)
    assert events[:100] == ['generate'] * 100
    assert events.count('reload') == 20


def test_explicit_four_bit_projection_uses_existing_rowwise_primitive(monkeypatch):
    from mlx_one.models.shared.execution import project

    class Affine:
        bits, group_size, mode = 4, 64, 'affine'
        weight = np.zeros((5,8), np.uint32)
        scales, biases = np.ones((5,1), np.float16), np.zeros((5,1), np.float16)

        def as_linear(self, _value):
            raise AssertionError('native row-count-dependent dispatch')

    calls = []
    core = ModuleType('mlx.core')
    core.arange, core.zeros, core.int32 = np.arange, np.zeros, np.int32

    def gather(value, weight, **kwargs):
        calls.append((value.shape, weight, kwargs))
        return np.zeros((value.shape[0],1,5), np.float16)

    core.gather_qmm = gather
    package = ModuleType('mlx')
    package.core = core
    monkeypatch.setitem(sys.modules, 'mlx', package)
    monkeypatch.setitem(sys.modules, 'mlx.core', core)
    module = Affine()
    output = project(module, np.zeros((1,3,64), np.float16), execution_dtype='float16',
                     as_linear=True)
    assert output.shape == (1,3,5)
    assert calls[0][0] == (3,1,64)
    assert calls[0][1] is module.weight and calls[0][2]['bits'] == 4
    with pytest.raises(AssertionError, match='row-count'):
        project(module, np.zeros((1,3,64), np.float16), as_linear=True)


def test_performance_observer_records_state_bytes_without_logits():
    from mlx_one.p0_engine_worker import Observer

    observer = Observer(capture=False)
    observer({'kind': 'cache_bytes', 'bytes': 1024})
    observer({'kind': 'cache_bytes', 'bytes': 2048})
    observer({'kind': 'cache_bytes', 'bytes': 512})
    assert observer.summary()['peak_request_state_bytes'] == 2048
    assert observer.capture_bytes and not observer.capture_state and observer.states == []


def test_streaming_eos_gate_requires_evaluated_zero_token_termination(tmp_path):
    from mlx_one.p0_engine_worker import Cases
    from mlx_one.text import generation

    checked = {}
    def check(case, function, **kwargs):
        if case == 'C08':
            checked[case] = function()
        return {'status': 'PASS'}

    calls = []
    def generate(_prompt, *, stream=False, **kwargs):
        calls.append(stream)
        assert generation.select_token(None, None) == 99
        return {'text': '', 'generation_tokens': 0, 'finish_reason': 'stop',
                'observer': {'token_ids': [], 'batch_sizes': [1]}}, None

    fake = SimpleNamespace(check=check, generate=generate, directory=tmp_path,
        prompt=lambda text:text,
        bundle=SimpleNamespace(eos_token_ids=(99,), tokenizer=SimpleNamespace(eos_token_id=99)))
    original = generation.select_token
    Cases.streaming(fake)
    assert checked['C08']['controlled_eos_id'] == 99 and calls == [False, True]
    assert generation.select_token is original
