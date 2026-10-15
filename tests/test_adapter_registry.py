"""Explicit lazy bindings and version checks for all built-ins and local extensions."""
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from mlx_one.core.registry import get_registration, registered_model_types
from mlx_one.engine.adapter_registry import (
    RUNTIME_CONTRACT_VERSION,
    get_runtime_registration,
    register_runtime,
    registered_runtime_types,
    runtime_coverage,
)


def test_all_architectures_have_explicit_truthful_runtime_records():
    assert registered_runtime_types() == registered_model_types()
    rows = {row['model_type']: row for row in runtime_coverage()}
    assert len(rows) == 27
    assert {name for name, row in rows.items()
            if row['integration_status']=='architecture_only'} == {
        'lfm2_audio'}
    assert rows['sam3_tracker_video']['component_only']
    assert rows['sam3_tracker_video']['parent_model_type'] == 'sam3_video'
    assert set(rows['qwen3_5']['tasks']) == {'text-generation','vision-language-generation'}
    assert 'vision-language-generation' in rows['qwen3_5']['declared_tasks']
    assert rows['whisper']['tasks'] == rows['qwen3_asr']['tasks'] == ('transcription',)
    assert rows['qwen3_forced_aligner']['tasks'] == ('alignment',)
    from mlx_one.engine.runtime_specs import BUILTIN_ADAPTERS

    assert {name for name,row in rows.items() if row['shared_service_integrated']} == set(
        BUILTIN_ADAPTERS)
    assert all(not row['correctness_qualified'] and not row['performance_qualified']
               for row in rows.values())
    with pytest.raises(ValueError, match='unavailable'):
        get_runtime_registration('lfm2_audio').adapter()


def test_coverage_is_lazy_in_fresh_interpreter():
    script = """
import sys
from mlx_one.engine.adapter_registry import runtime_coverage
assert len(runtime_coverage())==27
assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
assert not any(n.startswith('mlx_one.models.') and n.endswith('.model') for n in sys.modules)
"""
    subprocess.run([sys.executable, '-c', script], check=True,
                   env={**os.environ, 'PYTHONPATH':'src'}, capture_output=True, text=True)


def test_contract_version_is_checked_before_selected_adapter_import():
    runtime=get_runtime_registration('gpt2', contract_version=RUNTIME_CONTRACT_VERSION)
    assert runtime.adapter_path.endswith(':GPT2Adapter')
    with pytest.raises(ValueError, match='version'):
        get_runtime_registration('gpt2', contract_version='registry-runtime-v0')
    with pytest.raises(ValueError, match='already registered'):
        register_runtime(replace(runtime, contract_version='registry-runtime-v2'))
    register_runtime(runtime)


@pytest.mark.parametrize('path', ['bad', ':Model', 'pkg:Model:Other', '../pkg:Model',
                                  'pkg..model:Model', 'pkg:model.method'])
def test_invalid_lazy_paths_rejected_without_import(path):
    with pytest.raises(ValueError, match='import path'):
        replace(get_registration('gpt2'), model_path=path)
    with pytest.raises(ValueError, match='import path'):
        replace(get_runtime_registration('gpt2'), adapter_path=path)


def test_component_and_integration_claims_require_complete_contract():
    runtime=get_runtime_registration('gpt2')
    with pytest.raises(ValueError, match='adapter'):
        replace(runtime, integration_status='integrated', adapter_path=None)
    with pytest.raises(ValueError, match='component'):
        replace(runtime, component_only=True)
    with pytest.raises(ValueError, match='version'):
        replace(runtime, contract_version='')


def test_new_architecture_cannot_silently_inherit_runtime_binding(monkeypatch):
    from mlx_one.core import registry
    from mlx_one.engine import adapter_registry

    entry=replace(get_registration('gpt2'), model_type='local_extension')
    monkeypatch.setattr(registry, '_REGISTRY', {**registry._REGISTRY, entry.model_type:entry})
    monkeypatch.setattr(adapter_registry, '_RUNTIME_REGISTRY',
                        dict(adapter_registry._RUNTIME_REGISTRY))
    with pytest.raises(KeyError, match='no explicit runtime'):
        get_runtime_registration(entry.model_type)
    # A selected local adapter remains a lazy import until requested explicitly.
    binding=replace(get_runtime_registration('gpt2'), model_type=entry.model_type,
                    adapter_path='local_unselected_extension:Adapter')
    register_runtime(binding)
    assert get_runtime_registration(entry.model_type) == binding
    assert 'local_unselected_extension' not in sys.modules
    tasks,states=binding.specifications()
    assert len(tasks)==1 and not states
    with pytest.raises(ModuleNotFoundError):
        binding.adapter()


def test_spec_provider_validation_rejects_undeclared_or_duplicate_tasks(monkeypatch):
    from mlx_one.engine import runtime_specs
    from mlx_one.engine.contracts import TaskKind

    binding=get_runtime_registration('qwen3')
    task=binding.specifications()[0][0]
    monkeypatch.setattr(runtime_specs,'task_specs',lambda entry:(task,task))
    with pytest.raises(ValueError, match='duplicate'):
        binding.specifications()
    monkeypatch.setattr(runtime_specs, 'task_specs',
                        lambda entry: (replace(task, task=TaskKind.EMBEDDING),))
    with pytest.raises(ValueError, match='not declared'):
        binding.specifications()
