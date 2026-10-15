"""Independent task batching and retrieval result/ownership regressions."""
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_dense_batching import native as native

from mlx_one import BatchingConfig
from mlx_one.engine.cache_errors import CacheCapacityError
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.device_executor import DeviceExecutor
from mlx_one.engine.service import InferenceService
from mlx_one.engine.task_runners.retrieval import (
    BertEmbeddingAdapter,
    LFM2ColBERTAdapter,
    MPNetEmbeddingAdapter,
    Qwen3EmbeddingAdapter,
    Qwen3RerankAdapter,
    maxsim,
)
from mlx_one.retrieval.loading import LoadedRetrievalModel

ADAPTERS={'bert':BertEmbeddingAdapter,'mpnet':MPNetEmbeddingAdapter,
          'qwen3_embedding':Qwen3EmbeddingAdapter,'qwen3_reranker':Qwen3RerankAdapter,
          'lfm2_colbert':LFM2ColBERTAdapter}


class Tokenizer:
    pad_token_id=0
    def encode(self,text,*,add_special_tokens=False,max_length=None):
        from mlx_one.retrieval.service import RERANK_PREFIX, RERANK_SUFFIX

        special={'no':[1],'yes':[2],RERANK_PREFIX:[30],RERANK_SUFFIX:[31,32]}
        ids=special.get(text,[3+ord(c)%23 for c in text])
        return ids[:max_length] if max_length else ids
    def pad(self,sequences,*,padding_side):
        width=max(map(len,sequences))
        ids,masks=[],[]
        for seq in sequences:
            n=width-len(seq)
            ids.append([self.pad_token_id]*n+list(seq) if padding_side=='left'
                       else list(seq)+[self.pad_token_id]*n)
            masks.append([False]*n+[True]*len(seq) if padding_side=='left'
                         else [True]*len(seq)+[False]*n)
        return ids,masks


def bundle(family,tmp_path):
    class Model:
        def __init__(self):
            self.config=SimpleNamespace(hidden_size=2,max_seq_length=256,sentence_max_length=256,
                projection_dim=2,document_max_length=256,query_max_length=32)
            if family=='qwen3_embedding':
                self.config.validate_dimensions=lambda d:2 if d is None else d
            self.calls=[]
        def parameters(self):
            return {'weight':np.ones(2,dtype=np.float32)}
        def __call__(self,ids,attention_mask,**kwargs):
            self.calls.append((ids.shape,DeviceExecutor.shared().is_owner))
            if family=='qwen3_reranker':
                assert kwargs['false_token_id']==1 and kwargs['true_token_id']==2
                assert np.all(ids[:,-2:]==[31,32])
                return SimpleNamespace(scores=np.ones(len(ids)),probabilities=np.full(len(ids),.5))
            values=np.stack((ids,ids.cumsum(axis=1)),axis=-1).astype(np.float32)
            if family=='lfm2_colbert':
                return SimpleNamespace(embeddings=values)
            if family=='bert':
                values=np.stack((ids,np.broadcast_to(np.arange(ids.shape[1]),ids.shape)),axis=-1)
            output=(values*attention_mask[...,None]).sum(axis=1)/attention_mask.sum(axis=1)[:,None]
            return SimpleNamespace(embeddings=output)
    return LoadedRetrievalModel(Model(),Tokenizer(),
        'reranking' if family=='qwen3_reranker' else 'embedding',tmp_path,'tiny/'+family,'a'*40,
        architecture=family)


@pytest.mark.parametrize('family',list(ADAPTERS))
def test_retrieval_batch_matches_independent_and_restores_order(native,tmp_path,family):
    b=bundle(family,tmp_path)
    adapter=ADAPTERS[family](b)
    task=(TaskKind.RERANKING if family=='qwen3_reranker' else
          TaskKind.LATE_INTERACTION if family=='lfm2_colbert' else TaskKind.EMBEDDING)
    payloads=[{'query':'query','documents':[text]} if task==TaskKind.RERANKING else
              {'texts':[text]} for text in ('abc','wxyz','rst')]
    prepared=tuple(adapter.prepare(task,p) for p in payloads)
    together=adapter.run_batch(prepared)
    independent=tuple(adapter.run_batch((p,))[0] for p in prepared)
    assert together==independent
    assert any(shape[0]>1 for shape,_ in b.model.calls)
    if task==TaskKind.LATE_INTERACTION:
        assert len(together[0].embeddings[0])==3
        assert maxsim(together[0],together[1])[0]>0
        with pytest.raises(ValueError,match='identities'):
            maxsim(together[0],replace(together[1],revision='different'))


def test_service_task_cancellation_lease_and_capacity(native,tmp_path):
    b=bundle('bert',tmp_path)
    service=InferenceService(batching_config=BatchingConfig(max_sequences=4,
        max_scheduled_tokens=128,queue_size=4))
    service.register_task('encoder',b)
    try:
        result=service.embed('encoder',['abc','xyz','abcd'])
        assert len(result.embeddings)==3 and all(owner for _,owner in b.model.calls)
        event=threading.Event()
        event.set()
        handle=service.submit_task('encoder','embedding',{'texts':['cancel']},cancel=event)
        from concurrent.futures import CancelledError

        with pytest.raises(CancelledError):
            handle.result(2)
        with pytest.raises(CacheCapacityError):
            service.submit_task('encoder','embedding',{'texts':['x'*200]})
        assert service.embed('encoder',['ok']).embeddings
        assert service.pool.stats()['active_leases']==0
    finally:
        service.close()
    assert service.pool.stats()['resident_models']==0


def test_rerank_stable_ties_topk_and_suffix(native,tmp_path):
    b=bundle('qwen3_reranker',tmp_path)
    adapter=Qwen3RerankAdapter(b)
    p=adapter.prepare('reranking',{'query':'q','documents':['first','second'],'top_k':1})
    result=adapter.run_batch((p,))[0]
    assert result.items[0].index==0 and result.items[0].probability==.5
    with pytest.raises(ValueError,match='top_k'):
        adapter.prepare('reranking',{'query':'q','documents':['d'],'top_k':False})


def test_all_masked_colbert_features_fail(native,tmp_path):
    adapter=LFM2ColBERTAdapter(bundle('lfm2_colbert',tmp_path))
    result=adapter.run_batch((adapter.prepare('late-interaction',{'texts':['abc']}),))[0]
    with pytest.raises(ValueError,match='masked'):
        maxsim(replace(result,masks=((False,False,False),)),result)


def test_existing_bert_embedding_api_matches_singletons(native,tmp_path):
    from mlx_one.retrieval.service import embed

    b=bundle('bert',tmp_path)
    result=embed(b,['abc','wxyz'],max_length=256,batch_size=2)
    expected=tuple(embed(b,text,max_length=256,batch_size=1).embeddings[0]
                   for text in ('abc','wxyz'))
    assert result.embeddings==expected


def test_dynamic_task_batch_and_cancellation_isolation(native,tmp_path,monkeypatch):
    from concurrent.futures import CancelledError

    b=bundle('bert',tmp_path)
    service=InferenceService(batching_config=BatchingConfig(max_sequences=2,
        queue_size=4,max_scheduled_tokens=128))
    service.register_task('encoder',b)
    try:
        service.embed('encoder','warm')
        work=service._task_actor._task_work
        cancel=threading.Event()
        original=type(b.model).__call__
        def cancel_during_forward(self,*args,**kwargs):
            output=original(self,*args,**kwargs)
            cancel.set()
            return output
        monkeypatch.setattr(type(b.model),'__call__',cancel_during_forward)
        with work.condition:
            first=service.submit_task('encoder','embedding',{'texts':['abc']},cancel=cancel)
            peer=service.submit_task('encoder','embedding',{'texts':['xyz']})
        with pytest.raises(CancelledError):
            first.result(2)
        assert peer.result(2).output.texts==('xyz',)
        assert b.model.calls[-1][0][0]==2
        assert service.pool.stats()['active_leases']==0
    finally:
        service.close()


def test_typed_retrieval_adapter_contract_and_pack_validation(native,tmp_path):
    from mlx_one.engine.contracts import (
        BatchCompatibilityKey,
        ExecutionAdapter,
        StageKind,
        WorkItem,
    )

    b=bundle('bert',tmp_path)
    adapter=BertEmbeddingAdapter(b)
    assert isinstance(adapter,ExecutionAdapter)
    prepared=adapter.prepare('embedding',{'texts':['abc']})
    key=BatchCompatibilityKey(str(id(b)),1,'native-task-adapter-v1',TaskKind.EMBEDDING,
        StageKind.ENCODE,adapter.identity,'encoder-v1','absolute','plain-text-v1')
    work=WorkItem('r',key,3,1,prepared.transient_bytes,'tokens',prepared)
    packed=adapter.pack((work,))
    assert adapter.unpack(adapter.execute(packed))[0].texts==('abc',)
    assert adapter.estimate(prepared)['persistent_cache_bytes']==0
    with pytest.raises(ValueError,match='identity'):
        adapter.pack((replace(work,compatibility=replace(key,model_instance='wrong')),))
