from dataclasses import replace

import numpy as np
import pytest
from test_dense_batching import native as native
from test_paged_hybrid import advanced as advanced

from mlx_one.engine.batching import BatchingConfig, BatchScheduler, ForwardRequest
from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.cache_specs import LayerStateSpec
from mlx_one.engine.components import load_snapshot, save_snapshot
from mlx_one.engine.compound import COMPOUND_VERSION, CompoundLinearComponent


def component():
    value=CompoundLinearComponent(3,((1,2,4),(1,2,3,3)),('float16','float32'))
    value.conv_state=np.ones((1,2,4),dtype=np.float16)
    value.recurrent_state=np.ones((1,2,3,3),dtype=np.float32)
    value._offset=8
    return value


def spec():
    value=component()
    return LayerStateSpec(0,'recurrent',state_shapes=value.state_shapes,
                          state_dtypes=value.state_dtypes,checkpoint_semantics=COMPOUND_VERSION)


def test_compound_clone_and_rollback_constraints(advanced):
    advanced()
    original=component()
    copy=original.clone()
    original.conv_state[0,0,0]=9
    original.recurrent_state[0,0,0,0]=9
    assert copy.conv_state[0,0,0]==copy.recurrent_state[0,0,0,0]==1
    copy.validate(spec())
    with pytest.raises(ValueError,match='replay'):
        copy.trim(1)
    with pytest.raises(ValueError,match='dtype'):
        copy.recurrent_state=copy.recurrent_state.astype(np.float16)
        copy.validate(spec())


def test_compound_snapshots_require_complete_typed_state(advanced,tmp_path):
    advanced()
    original=CacheBundle((component(),))
    identity={'fixture':'qwen35-v1'}
    manifest=save_snapshot(tmp_path,original,identity=identity)
    assert manifest['schema_version']=='hybrid-state-v4'
    restored=load_snapshot(tmp_path,identity=identity,expected_specs=(spec(),))
    np.testing.assert_array_equal(restored.entries[0].recurrent_state,
                                  original.entries[0].recurrent_state)
    with pytest.raises(ValueError,match='identity'):
        load_snapshot(tmp_path,identity={'fixture':'other'},expected_specs=(spec(),))
    with pytest.raises(ValueError,match='transition identity'):
        load_snapshot(tmp_path,identity=identity,
                      expected_specs=(replace(spec(),state_dtypes=('float32','float32')),))
    original.release()
    restored.release()


def test_scheduler_groups_compound_rows_without_padding_state(native):
    rows=[]
    for i,(offset,width) in enumerate(((0,2),(3,1),(3,1))):
        cache=component()
        cache._offset=offset
        rows.append(ForwardRequest(native,tuple([1]*width),(cache,),
            'prefill',(1,)*8,request_id=str(i)))
    scheduler=BatchScheduler(BatchingConfig(max_sequences=3))
    groups=scheduler.select(rows)
    assert [len(group) for group in groups]==[1,2]
    assert sum(len(r.token_ids) for g in groups for r in g)==4
    assert all(len({(r.cache[0].offset,len(r.token_ids)) for r in group})==1 for group in groups)


def test_compound_missing_state_and_generic_recurrent_rejected(advanced):
    from mlx_one.engine.components import RecurrentComponent

    advanced()
    incomplete=component()
    incomplete.recurrent_state=None
    with pytest.raises(ValueError,match='missing'):
        incomplete.validate(spec())
    with pytest.raises(ValueError,match='identity'):
        RecurrentComponent().validate(spec())


def test_compound_failed_publication_preserves_snapshot(advanced,tmp_path,monkeypatch):
    import os

    advanced()
    original=CacheBundle((component(),))
    identity={'fixture':'atomic'}
    first=save_snapshot(tmp_path,original,identity=identity)
    replace_file=os.replace
    def fail_manifest(source,target):
        if str(target).endswith('state.json'):
            raise OSError('manifest failure')
        return replace_file(source,target)
    monkeypatch.setattr(os,'replace',fail_manifest)
    with pytest.raises(OSError,match='manifest failure'):
        save_snapshot(tmp_path,original,identity=identity)
    assert [p.name for p in tmp_path.glob('tensors-*.npz')]==[first['payload']]
    restored=load_snapshot(tmp_path,identity=identity,expected_specs=(spec(),))
    assert restored.entries[0].offset==8
    original.release()
    restored.release()


def test_bucketed_execution_restores_original_result_order(native):
    from mlx_one.engine.batching import execute_compatible
    from mlx_one.engine.contracts import RunnerOutput

    requests=[]
    for i,offset in enumerate((0,3,0)):
        c=component()
        c._offset=offset
        requests.append(ForwardRequest(native,(i+1,),(c,),'decode',(),request_id=str(i)))
    class Runner:
        def execute(self,group):
            return RunnerOutput(tuple(r.request_id for r in group),
                                tuple(r.token_ids[0] for r in group),tuple(1 for _ in group))
    result=execute_compatible(Runner(),tuple(requests))
    assert result.request_ids==('0','1','2') and result.logits==(1,2,3)
