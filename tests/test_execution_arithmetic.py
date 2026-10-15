"""Independent CPU arithmetic/mapping tests for existing native gathered ops."""
import sys
from types import ModuleType

import numpy as np
import pytest

from mlx_one.models.shared.execution import attention_matmul, project


@pytest.fixture
def core(monkeypatch):
    module = ModuleType('mlx.core')
    module.arange = np.arange
    module.zeros = np.zeros
    module.repeat = np.repeat
    module.float32, module.int32 = np.float32, np.int32
    calls = []
    def gather_mm(left, right, *, lhs_indices, rhs_indices):
        calls.append((left.copy(), right.copy(), lhs_indices.copy(), rhs_indices.copy()))
        return np.stack([left[i] @ right[j] for i, j in zip(lhs_indices,rhs_indices,strict=True)])
    module.gather_mm = gather_mm
    module.calls = calls
    package=ModuleType('mlx')
    package.core=module
    monkeypatch.setitem(sys.modules,'mlx',package)
    monkeypatch.setitem(sys.modules,'mlx.core',module)
    return module


@pytest.mark.parametrize('shape', [(1,2,1,4,7),(2,4,3,8,5),(1,3,17,4,6)])
def test_per_query_attention_preserves_batch_head_and_row_mapping(core,shape):
    batch,heads,queries,width,keys=shape
    rng=np.random.default_rng(31)
    left=rng.normal(size=(batch,heads,queries,width)).astype(np.float16)
    right=rng.normal(size=(batch,heads,width,keys)).astype(np.float16)
    actual=attention_matmul(left,right)
    expected=np.einsum('bhqd,bhdk->bhqk',left.astype(np.float32),right.astype(np.float32))
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
    assert actual.dtype == np.float32
    assert core.calls[0][0].shape == (batch*heads*queries,1,width)
    assert core.calls[0][1].shape == (batch*heads,width,keys)
    for row in range(queries):
        np.testing.assert_allclose(attention_matmul(left[:,:,row:row+1],right),
                                   actual[:,:,row:row+1],atol=1e-6,rtol=1e-6)


def test_attention_layout_mismatch_fails_before_dispatch(core):
    with pytest.raises(ValueError, match='aligned'):
        attention_matmul(np.ones((1,2,3,4)),np.ones((1,1,4,5)))
    assert core.calls == []


class Projection(dict):
    bits,group_size,mode = 8,64,'affine'
    weight=np.zeros((3,2),dtype=np.uint32)
    scales=np.ones((3,1),dtype=np.float16)
    biases=np.zeros((3,1),dtype=np.float16)
    def __call__(self,value):
        return value+1
    def as_linear(self,value):
        return value+2


def test_gathered_projection_preserves_packed_payload_and_optional_bias(core):
    calls=[]
    def qmm(rows,weight,**options):
        calls.append((rows.copy(),weight,options))
        # Independent affine fixture result; gathered row identities are verified below.
        return np.repeat(rows.sum(axis=-1,keepdims=True),3,axis=-1)
    core.gather_qmm=qmm
    module=Projection()
    module.bias=np.array([1,2,3],dtype=np.float16)
    module['bias']=module.bias
    value=np.arange(24,dtype=np.float16).reshape(2,3,4)
    output=project(module,value,execution_dtype='float16')
    np.testing.assert_array_equal(output,np.repeat(value.sum(-1,keepdims=True),3,-1)+module.bias)
    rows,weight,options=calls[0]
    assert rows.shape == (6,1,4) and weight is module.weight
    assert options['scales'] is module.scales and options['biases'] is module.biases
    np.testing.assert_array_equal(options['lhs_indices'],np.arange(6))
    np.testing.assert_array_equal(options['rhs_indices'],np.zeros(6))
    assert output.dtype == np.float16 and module.weight.dtype == np.uint32
    tied=project(module,value,execution_dtype='float16',as_linear=True)
    np.testing.assert_array_equal(tied,output-module.bias)


@pytest.mark.parametrize('dtype,bits',[(None,8),('bfloat16',8),('float16',4)])
def test_omitted_policy_and_other_formats_keep_module_dispatch(core,dtype,bits):
    module=Projection()
    module.bits=bits
    value=np.zeros((1,2,4),dtype=np.float16)
    np.testing.assert_array_equal(project(module,value,execution_dtype=dtype),value+1)
    np.testing.assert_array_equal(project(module,value,execution_dtype=dtype,as_linear=True),value+2)
    assert core.calls == []


def test_parity_gate_requires_evaluated_success_and_cleanup():
    from mlx_one.runtime_baseline import diagnostic_outcome

    valid={'status':'passed','cleanup_passed':True,'failure_reproduced':False}
    report=diagnostic_outcome({'status':'completed'},valid)
    assert report['readiness']=='parity_passed' and not report['qualification_promoted']
    assert diagnostic_outcome({'status':'completed'},{**valid,'cleanup_passed':False})[
        'readiness'] != 'parity_passed'
    assert diagnostic_outcome({'status':'completed'},{**valid,'cleanup_passed':False})[
        'qualification'] == 'failed'
    assert diagnostic_outcome({'status':'memory_watchdog_terminated'},valid)[
        'readiness'] != 'parity_passed'


def test_scratch_bound_covers_casts_and_index_overlap():
    from mlx_one.models.shared.execution import attention_scratch_bound

    query=np.zeros((2,4,3,8),dtype=np.float32)
    history=np.zeros((2,4,11,8),dtype=np.float32)
    scores=np.zeros((2,4,3,11),dtype=np.float32)
    query_indices=np.zeros((2*4*3,2),dtype=np.int32)
    projection_indices=np.zeros((2*3,2),dtype=np.int32)
    observed=query.nbytes+history.nbytes*2+scores.nbytes
    observed+=query_indices.nbytes+projection_indices.nbytes
    bound=attention_scratch_bound(batch=2,heads=4,head_dim=8,chunk=3,history=8)
    assert bound>=observed
    with pytest.raises(ValueError,match='dimensions'):
        attention_scratch_bound(batch=1,heads=1,head_dim=8,chunk=0,history=0)


def test_compact_ragged_attention_excludes_padding_from_reduction_width(core):
    from mlx_one.engine.batching import _PackedCache

    core.float16=np.float16
    core.where=np.where
    def softmax(x,axis=-1,precise=True):
        y=np.exp(x-x.max(axis=axis,keepdims=True))
        return y/y.sum(axis=axis,keepdims=True)
    core.softmax=softmax
    rng=np.random.default_rng(71)
    keys=rng.normal(size=(2,1,10,4)).astype(np.float16)
    values=rng.normal(size=keys.shape).astype(np.float16)
    # The short row's five left-padding slots must never enter a reduction.
    keys[1,:,:5]=values[1,:,:5]=1000
    q=rng.normal(size=(2,2,3,4)).astype(np.float16)
    k=rng.normal(size=(2,1,3,4)).astype(np.float16)
    v=rng.normal(size=k.shape).astype(np.float16)
    cache=_PackedCache(keys,values,7,lengths=(7,2),widths=(3,1),compact_attention=True)
    result=cache.attend(q,k,v,scale=.5,position_ids=None)
    for row,(old,width) in enumerate(((7,3),(2,1))):
        rk,rv=(np.repeat(t[row:row+1,:,7-old:7+width],2,axis=1).astype(np.float32)
               for t in (cache.keys,cache.values))
        scores=q[row:row+1,:,:width].astype(np.float32)@rk.transpose(0,1,3,2)*.5
        valid=np.arange(old+width)[None,:] <= old+np.arange(width)[:,None]
        expected=softmax(np.where(valid,scores,-np.inf))@rv
        np.testing.assert_array_equal(result[row:row+1,:,:width],expected.astype(np.float16))
    assert [call[1].shape[-1] for call in core.calls[::2]]==[10,3]
    assert not np.any(result[1,:,1:])
