"""Opt-in tiny native task parity, never checkpoint or performance qualification."""
import os
from dataclasses import asdict

import numpy as np
import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS')!='1':
    pytest.skip('requires supervised Metal host',allow_module_level=True)

import mlx.core as mx
from test_task_retrieval_runtime import ADAPTERS, Tokenizer

from mlx_one.engine.contracts import TaskKind
from mlx_one.retrieval.loading import LoadedRetrievalModel


def make_bundle(family,path):
    from mlx_one.models.embeddings.bert.config import BertEmbeddingConfig
    from mlx_one.models.embeddings.bert.model import BertForSentenceEmbedding
    from mlx_one.models.embeddings.lfm2_colbert.config import Lfm2ColBERTConfig
    from mlx_one.models.embeddings.lfm2_colbert.model import Lfm2ColBERTModel
    from mlx_one.models.embeddings.mpnet.config import MPNetEmbeddingConfig
    from mlx_one.models.embeddings.mpnet.model import MPNetForSentenceEmbedding
    from mlx_one.models.embeddings.qwen3_embedding.config import Qwen3EmbeddingConfig
    from mlx_one.models.embeddings.qwen3_embedding.model import Qwen3ForEmbedding
    from mlx_one.models.embeddings.qwen3_reranker.config import Qwen3RerankerConfig
    from mlx_one.models.embeddings.qwen3_reranker.model import Qwen3ForReranking

    config=dict(vocab_size=64,hidden_size=8,num_hidden_layers=2,num_attention_heads=2,
        num_key_value_heads=1,head_dim=4,tie_word_embeddings=True,
        bos_token_id=1,eos_token_id=2,intermediate_size=16,max_position_embeddings=256,
        hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
    if family=='bert':
        model=BertForSentenceEmbedding(BertEmbeddingConfig.from_dict(
            {**config,'model_type':family,'sentence_max_length':256}))
    elif family=='mpnet':
        model=MPNetForSentenceEmbedding(MPNetEmbeddingConfig.from_dict(
            {**config,'model_type':family,'sentence_max_length':128}))
    elif family=='lfm2_colbert':
        model=Lfm2ColBERTModel(Lfm2ColBERTConfig.from_dict({'model_type':family,
            'projection_dim':4,'document_max_length':128,'query_max_length':32,
            'text_config':{**config,'layer_types':['conv','full_attention'],
                           'conv_L_cache':3,'block_multiple_of':4}}))
    else:
        cls=Qwen3EmbeddingConfig if family=='qwen3_embedding' else Qwen3RerankerConfig
        payload={'model_type':family,'text_config':config,'max_seq_length':256}
        if family=='qwen3_embedding':
            payload.update(min_dimension=2,max_dimension=8)
        model=(Qwen3ForEmbedding if family=='qwen3_embedding' else Qwen3ForReranking)(
            cls.from_dict(payload))
    model.eval()
    tokenizer=Tokenizer()
    if family=='mpnet':
        tokenizer.pad_token_id=1
    return LoadedRetrievalModel(model,tokenizer,'reranking'
        if family=='qwen3_reranker' else 'embedding',path,'tiny/'+family,'tiny-v1',family)


@pytest.mark.parametrize('family',list(ADAPTERS))
def test_native_task_ragged_matches_independent(family,tmp_path,monkeypatch):
    mx.random.seed(73)
    b=make_bundle(family,tmp_path)
    adapter=ADAPTERS[family](b)
    task=(TaskKind.RERANKING if family=='qwen3_reranker' else
          TaskKind.LATE_INTERACTION if family=='lfm2_colbert' else TaskKind.EMBEDDING)
    payloads=[{'query':'query','documents':[text]} if task==TaskKind.RERANKING else
              {'texts':[text]} for text in ('abc','wxyz','rst','more')]
    prepared=tuple(adapter.prepare(task,p) for p in payloads)
    calls=[]
    original=type(b.model).__call__
    def observed(self,ids,**kwargs):
        calls.append(ids.shape[0])
        return original(self,ids,**kwargs)
    monkeypatch.setattr(type(b.model),'__call__',observed)
    actual=adapter.run_batch(prepared)
    assert max(calls)>1
    expected=tuple(adapter.run_batch((p,))[0] for p in prepared)
    for a,e in zip(actual,expected,strict=True):
        if task==TaskKind.RERANKING:
            np.testing.assert_allclose([i.score for i in a.items],[i.score for i in e.items],
                                       atol=1e-4,rtol=1e-4)
            np.testing.assert_allclose([i.probability for i in a.items],
                                       [i.probability for i in e.items],atol=1e-4,rtol=1e-4)
        else:
            np.testing.assert_allclose(asdict(a)['embeddings'],asdict(e)['embeddings'],
                                       atol=1e-4,rtol=1e-4)


def test_reranker_loader_returns_registered_architecture(tmp_path):
    import json

    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file
    from test_native_qwen3_retrieval import _write_tokenizer

    from mlx_one.models.embeddings.qwen3_reranker.config import Qwen3RerankerConfig
    from mlx_one.models.embeddings.qwen3_reranker.model import Qwen3ForReranking
    from mlx_one.retrieval.loading import load_retrieval_model

    text=dict(model_type='qwen3',vocab_size=300,hidden_size=8,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=4,intermediate_size=16,
        max_position_embeddings=256,tie_word_embeddings=True,bos_token_id=1,eos_token_id=2)
    model=Qwen3ForReranking(Qwen3RerankerConfig.from_dict({'text_config':text,
        'model_type':'qwen3_reranker','max_seq_length':256}))
    _write_tokenizer(tmp_path)
    (tmp_path/'config.json').write_text(json.dumps(text))
    (tmp_path/'config_sentence_transformers.json').write_text(json.dumps({'model_type':'CrossEncoder'}))
    save_file({name:np.asarray(value) for name,value in tree_flatten(model.parameters())},
              str(tmp_path/'model.safetensors'))
    loaded=load_retrieval_model(tmp_path,task='reranking',offline=True)
    assert loaded.architecture=='qwen3_reranker'
    assert ADAPTERS[loaded.architecture](loaded).family=='qwen3_reranker'
