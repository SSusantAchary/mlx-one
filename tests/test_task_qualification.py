"""Task discovery/identity, pending gates, integrity and scoped matrix coverage."""
import json
from dataclasses import asdict, replace

import pytest

from mlx_one.inference_qualification import write_integrity
from mlx_one.runtime_baseline import file_digest
from mlx_one.task_qualification import (
    TaskProfile,
    export_task_matrix,
    read_inputs,
    task_gates,
    task_inventory,
)
from mlx_one.task_qualification_worker import compare_result, task_quality


def profile(tmp_path):
    inputs=tmp_path/'inputs.json'
    inputs.write_text(json.dumps({'schema_version':'native-task-inputs-v1','cases':[
        {'request_id':'a','payload':{'texts':['hello']}}]}))
    return TaskProfile('bert','embedding','org/checkpoint','a'*40,str(tmp_path/'model'),
                       str(inputs),file_digest(inputs))


def test_registry_derived_inventory_is_complete_and_lazy():
    inventory=task_inventory()
    assert len(inventory['entries'])==30
    assert inventory['qualification_promoted'] is False
    assert sum(r['component_only'] for r in inventory['entries'])==1
    assert next(r for r in inventory['entries'] if r['model_type']=='lfm2_audio')[
        'integration_status']=='architecture_only'


def test_missing_service_quality_telemetry_never_passes(tmp_path):
    p=profile(tmp_path)
    measured={'backend':'mlx-metal','metal_available':True,'rss_bytes':10,
        'mlx_active_bytes':10,'mlx_cached_bytes':10,'mlx_peak_bytes':10,
        'active_handles':0,'reservations_bytes':0,'active_leases':0}
    gates=task_gates(p,{'passed':True},measured,{'passed':None})
    assert gates['status']=='pending'
    assert {'quality','task_lifecycle','task_quality_coverage'}<=set(gates['pending'])
    assert task_gates(replace(p,validation_kind='synthetic-smoke'),{'passed':True},measured,
        {'passed':True,'ground_truth_sha256':'a'*64})['status']=='failed'
    missing=task_gates(p,{'passed':True},{**measured,'rss_bytes':None},{})
    assert missing['checks']['telemetry'] is None


def test_pairing_rejects_changed_inputs_precision_and_budgets(tmp_path):
    p=profile(tmp_path)
    reference={'profile':asdict(p),'throughput_per_second':1,'source':{},'hardware':'same'}
    measured={'throughput_per_second':2,'source':{},'hardware':'same'}
    for key,value in (('inputs_sha256','b'*64),('execution_dtype','float16'),
                      ('settings',{'queue_size':99})):
        mismatch={**reference,'profile':{**asdict(p),key:value}}
        gate=task_gates(p,{},measured,{},reference=mismatch)
        assert gate['checks']['matched_reference'] is False


def test_manifest_assets_integrity_and_authority(tmp_path):
    p=profile(tmp_path)
    read_inputs(p)
    path=tmp_path/'inputs.json'
    data=json.loads(path.read_text())
    data['cases'][0]['cache_namespace']='other-user'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='checksum'):
        read_inputs(p)
    with pytest.raises(ValueError,match='authority'):
        read_inputs(replace(p,inputs_sha256=file_digest(path)))


def test_matrix_includes_unrun_architectures_and_retains_failed_worker(tmp_path):
    p=profile(tmp_path)
    run=tmp_path/'run'
    run.mkdir()
    (run/'manifest.json').write_text(json.dumps({'profile':asdict(p),'source':{'git_revision':'a'}}))
    (run/'failure.json').write_text(json.dumps({'status':'FAIL','reason':'watchdog timeout'}))
    write_integrity(run)
    rows=export_task_matrix([run],tmp_path/'matrix')
    assert len({r['model_type'] for r in rows})==30
    assert next(r for r in rows if r['model_type']=='bert')['status']=='FAIL'
    assert next(r for r in rows if r['model_type']=='sam3_tracker_video')['status']=='SKIPPED'
    assert all(r['qualification_promoted'] is False for r in rows)
    (run/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='stale'):
        export_task_matrix([run],tmp_path/'matrix')


def test_parity_ignores_no_numeric_error_and_preserves_integer_ids():
    assert compare_result({'x':[.5,2]},{'x':[.50001,2]},atol=1e-4,rtol=1e-4)
    assert not compare_result([2],[3],atol=99,rtol=99)
    assert not compare_result([float('nan')],[float('nan')],atol=.1,rtol=.1)


def test_quality_requires_anchors_and_never_relaxes_alignment_80ms():
    assert task_quality('transcription',[{'text':'x'}],[{}],{},'a')['passed'] is None
    assert task_quality('transcription',[{'text':'x'}],[{'expected':{'text':'x'}}],
        {'max_wer':0},'a')['passed'] is True
    result=task_quality('alignment',[{'end':.2}],[{'expected':{'end':.1}}],
        {'quality_atol':1,'quality_rtol':1},'a')
    assert result['passed'] is False and result['coverage_passed'] is None


def test_observed_cases_do_not_manufacture_quality_coverage():
    result=task_quality('text-generation',[{'text':'Paris'}],
                        [{'expected':{'text':'Paris'}}],
                        {'quality_atol':0,'quality_rtol':0},'a'*64)
    assert result['passed'] is True
    assert result['coverage']=={'declared_cases':1,'observed_outputs':1,
        'cases_with_ground_truth':1,'status':'NOT_RUN',
        'reason':'no independently validated coverage protocol supplied'}
    assert result['coverage_passed'] is None
    missing=task_quality('text-generation',[{'text':'Paris'}],[{}],{},'a'*64)
    assert missing['passed'] is None and missing['coverage_passed'] is None
    assert missing['coverage']['cases_with_ground_truth']==0


@pytest.mark.parametrize('ignore_cancel,leak,change_recovery',[
    (False,False,False),(True,False,False),(False,True,False),(False,False,True)])
def test_text_lifecycle_observes_cancel_leaks_and_recovery(ignore_cancel,leak,change_recovery):
    from types import SimpleNamespace

    from mlx_one.task_qualification_worker import probe_text_cancellation

    class Target:
        _cache_manager=SimpleNamespace(stats=lambda:SimpleNamespace(
            active_handles=int(leak),reserved_bytes=int(leak)))

        def stream(self,prompt,*,options,cancel):
            yield SimpleNamespace(token_id=1)
            if ignore_cancel or not cancel.is_set():
                yield SimpleNamespace(token_id=2)

    result=probe_text_cancellation(Target(),{'prompt':'hi'}, {'tokens':[1]},
        lambda case:({'tokens':[2 if change_recovery else 1]},{}),{},atol=0,rtol=0)
    assert result['cancellation'] is (not ignore_cancel)
    assert result['cancel_cleanup'] is (not leak)
    assert result['recovery'] is (not change_recovery)


def test_task_registry_rejects_incomplete_or_synthetic_native_artifacts(tmp_path):
    from mlx_one.engine.adapters import ADAPTER_VERSION
    from mlx_one.registry import (
        InferenceFeatureRegistry,
        InferenceFeatureStatus,
        InferenceScope,
        RegistryError,
    )

    scope=InferenceScope('org/checkpoint','a'*40,'native-task-v1','topology',
                         'float32','not_applicable','0.32.2','Apple M4','b'*40)
    evidence=tmp_path/'evidence.json'
    result={'schema_version':'native-inference-qualification-v4','backend':'mlx-one-native',
        'scope':asdict(scope),'feature':'task:embedding','status':'passed','correctness_passed':True,
        'execution_contract':{'adapter_version':ADAPTER_VERSION,'topology_digest':'topology',
            'execution_mode':'native-task-v1','quantization_digest':'digest','components':['output']}}
    evidence.write_text(json.dumps(result))
    registry=InferenceFeatureRegistry()
    with pytest.raises(RegistryError,match='complete checksummed'):
        registry.promote('task:embedding',scope,evidence,InferenceFeatureStatus.CORRECTNESS_PASSED)
    assert registry.records()==()
