"""Task-neutral metadata does not load model math or grant runtime eligibility."""
from dataclasses import replace

import pytest

from mlx_one.engine.contracts import (
    BatchCompatibilityKey,
    LayerExecutionSpec,
    RequestPhase,
    StageKind,
    TaskKind,
    TaskSpec,
    WorkItem,
    canonical_task,
    request_phase,
)


def key(**overrides):
    values = dict(model_instance='m1', model_epoch=1, adapter_version='a1',
                  task=TaskKind.TEXT_GENERATION, stage=StageKind.DECODE,
                  precision_policy='p1', state_schema='s1', position_policy='rope',
                  computation_identity='options1')
    values.update(overrides)
    return BatchCompatibilityKey(**values)


def test_task_aliases_preserve_distinct_tasks():
    assert canonical_task('asr') is canonical_task('transcribe') is TaskKind.TRANSCRIPTION
    assert canonical_task('align') is TaskKind.ALIGNMENT
    assert canonical_task('late-interaction') is TaskKind.LATE_INTERACTION
    with pytest.raises(ValueError):
        canonical_task('unknown-task')


@pytest.mark.parametrize('change', [dict(model_instance='m2'), dict(model_epoch=2),
    dict(adapter_version='a2'), dict(task=TaskKind.EMBEDDING), dict(stage=StageKind.PREFILL),
    dict(precision_policy='p2'), dict(state_schema='s2'), dict(position_policy='learned'),
    dict(computation_identity='options2')])
def test_compatibility_key_separates_every_execution_dimension(change):
    assert key() != key(**change)
    assert key() == key()


def test_task_spec_validates_stages_and_immutable_metadata():
    spec = TaskSpec(TaskKind.TEXT_GENERATION, 'causal', (StageKind.PREFILL, StageKind.DECODE),
                    'token-budget', 'text-v1')
    assert spec.speculation_modes == ()
    with pytest.raises(ValueError, match='stage'):
        replace(spec, stage_kinds=())
    with pytest.raises(ValueError, match='duplicate'):
        replace(spec, stage_kinds=(StageKind.PREFILL, StageKind.PREFILL))


def test_per_layer_layout_supports_varying_heads_and_non_rope_dimensions():
    layers = (LayerExecutionSpec(0, 'full_kv', 6, 2, 64, 64, 'rope'),
              LayerExecutionSpec(1, 'full_kv', 8, 4, 65, 65, 'learned'))
    assert layers[0].query_heads != layers[1].query_heads
    with pytest.raises(ValueError, match='head'):
        replace(layers[0], kv_heads=4)
    with pytest.raises(ValueError, match='dimension'):
        replace(layers[0], key_dim=0)
    with pytest.raises(ValueError, match='window'):
        replace(layers[0], state_kind='sliding_kv')


def test_work_items_keep_cost_units_and_result_ownership_separate():
    work = WorkItem('r1', key(), real_input_cost=1, output_bound=4,
                    estimated_transient_bytes=4096, cost_unit='tokens')
    assert work.request_id == 'r1' and work.compatibility.model_epoch == 1
    with pytest.raises(ValueError, match='cost'):
        replace(work, real_input_cost=-1)
    with pytest.raises(ValueError, match='request'):
        replace(work, request_id='')
    audio = replace(work, compatibility=key(task=TaskKind.TRANSCRIPTION,
                                           stage=StageKind.ENCODE), cost_unit='samples')
    assert audio.cost_unit != work.cost_unit


def test_request_lifecycle_preserves_legacy_text_phase_mapping():
    from mlx_one.engine.contracts import Phase

    assert request_phase(Phase.WAITING) is RequestPhase.QUEUED
    assert request_phase(Phase.PREFILL) is request_phase(Phase.DECODE) is RequestPhase.RUNNING
    assert request_phase(Phase.PAUSED) is RequestPhase.PAUSED
    for phase in Phase:
        assert request_phase(phase).terminal == phase.terminal
    RequestPhase.QUEUED.validate_transition(RequestPhase.PREPARING)
    RequestPhase.RUNNING.validate_transition(RequestPhase.PAUSED)
    RequestPhase.PAUSED.validate_transition(RequestPhase.RUNNING)
    RequestPhase.CANCELLED.validate_transition(RequestPhase.CANCELLED)
    with pytest.raises(ValueError, match='transition'):
        RequestPhase.FINISHED.validate_transition(RequestPhase.RUNNING)
