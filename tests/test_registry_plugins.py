"""A new architecture uses existing task scheduling without scheduler modifications."""
import sys
from dataclasses import replace

import pytest
from test_dense_batching import native as native

from mlx_one.core.registry import ModelRegistration, get_registration
from mlx_one.engine.adapter_registry import RUNTIME_CONTRACT_VERSION, RuntimeRegistration
from mlx_one.engine.plugins import qualified_task_profile, register_plugin, validate_plugin


def descriptors():
    module='registry_plugin_fixture:'
    architecture=ModelRegistration('example_encoder',module+'Config',module+'Model',
        'embedding',frozenset({'forward'}),loader_path=module+'load_model',
        tasks=frozenset({'embedding'}))
    runtime=RuntimeRegistration('example_encoder',module+'Adapter',module+'task_specs',
        module+'state_specs',RUNTIME_CONTRACT_VERSION,integration_status='integrated',
        reason='Explicit local synthetic adapter; not checkpoint qualified')
    return architecture,runtime


def test_isolated_metadata_validation_does_not_load_backend_in_parent():
    before=set(sys.modules)
    architecture,runtime=descriptors()
    report=validate_plugin(architecture,runtime,config={'model_type':'example_encoder'})
    assert report['metadata_validated'] and not report['model_executed']
    assert not report['hardware_qualified']
    assert 'registry_plugin_fixture' not in set(sys.modules)-before
    assert 'mlx.core' not in set(sys.modules)-before


def test_atomic_plugin_registration_and_existing_actor_task_execution(native,monkeypatch):
    from mlx_one import BatchingConfig
    from mlx_one.core import registry
    from mlx_one.engine import adapter_registry
    from mlx_one.engine.service import InferenceService

    registry.registered_model_types()
    monkeypatch.setattr(registry,'_REGISTRY',dict(registry._REGISTRY))
    monkeypatch.setattr(adapter_registry,'_RUNTIME_REGISTRY',dict(adapter_registry._RUNTIME_REGISTRY))
    architecture,runtime=descriptors()
    register_plugin(architecture,runtime,config={'model_type':'example_encoder'})
    assert get_registration('example_encoder')==architecture
    service=InferenceService(batching_config=BatchingConfig(max_sequences=2))
    service.register_task('new',architecture.loader()('/fixture'))
    try:
        result=service.embed('new',['one','two'])
        assert result.dimensions==2 and result.texts==('one','two')
        capabilities=service.task_capabilities('new','embedding')
        assert capabilities['availability']['implemented']
        assert not capabilities['qualification']['performance_qualified']
        assert service.pool.stats()['active_leases']==0
    finally:
        service.close()
    register_plugin(architecture,runtime,config={'model_type':'example_encoder'})


def test_invalid_plugin_never_leaves_partial_architecture(monkeypatch):
    from mlx_one.core import registry
    from mlx_one.engine import adapter_registry

    old_models=dict(registry._REGISTRY)
    old_runtime=dict(adapter_registry._RUNTIME_REGISTRY)
    architecture,runtime=descriptors()
    with pytest.raises(ValueError,match='identity'):
        register_plugin(architecture,replace(runtime,contract_version='old'),
                        config={'model_type':'example_encoder'})
    assert registry._REGISTRY==old_models and adapter_registry._RUNTIME_REGISTRY==old_runtime
    with pytest.raises(ValueError,match='statically'):
        validate_plugin(replace(architecture,model_path='registry_plugin_fixture:MissingModel'),
                        runtime,config={'model_type':'example_encoder'})


def test_canonical_aliases_cannot_duplicate_task_authority():
    _,runtime=descriptors()
    with pytest.raises(ValueError,match='duplicate'):
        replace(runtime,task_adapters=(('asr','registry_plugin_fixture:Adapter'),
                                       ('transcribe','registry_plugin_fixture:Adapter')))
    with pytest.raises(ValueError,match='not declared'):
        replace(runtime,task_adapters=(('alignment','registry_plugin_fixture:Adapter'),)).specifications(
            descriptors()[0])


def test_task_rollout_rejects_unqualified_or_absent_evidence(tmp_path):
    from test_task_qualification import profile

    from mlx_one.registry import InferenceFeatureRegistry, InferenceScope

    requested=profile(tmp_path)
    scope=InferenceScope('org/checkpoint','a'*40,'native-task-v1','topology',
                         'float32','not_applicable','0.32.2','M4','b'*40)
    registry=InferenceFeatureRegistry()
    registry.register_experimental('task:embedding',scope)
    assert qualified_task_profile(registry,scope,requested)['resolved_profile'] is None
    assert qualified_task_profile(None,scope,requested)['resolved_profile'] is None


def test_metadata_that_imports_backend_is_rejected_in_isolated_child(tmp_path,monkeypatch):
    module=tmp_path/'eager_plugin_metadata.py'
    module.write_text('import mlx.core\nclass Config: pass\nclass Model: pass\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    architecture,runtime=descriptors()
    architecture=replace(architecture,config_path='eager_plugin_metadata:Config',
                         model_path='eager_plugin_metadata:Model')
    before=set(sys.modules)
    with pytest.raises(ValueError,match='execution backend'):
        validate_plugin(architecture,runtime,config={'model_type':'example_encoder'})
    assert 'eager_plugin_metadata' not in set(sys.modules)-before
    assert 'mlx.core' not in set(sys.modules)-before
