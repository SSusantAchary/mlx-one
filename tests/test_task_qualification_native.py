"""Runnable isolated task worker smoke; synthetic assets cannot qualify anything."""
import os

import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS')!='1':
    pytest.skip('requires supervised native Metal execution',allow_module_level=True)


def test_isolated_native_task_worker_and_pending_matrix(tmp_path):
    import json

    import numpy as np
    from mlx.utils import tree_flatten
    from test_native_qwen3_retrieval import _write_tokenizer

    from mlx_one.inference_qualification import validate_integrity
    from mlx_one.models.embeddings.qwen3_reranker.config import Qwen3RerankerConfig
    from mlx_one.models.embeddings.qwen3_reranker.model import Qwen3ForReranking
    from mlx_one.runtime_baseline import file_digest
    from mlx_one.task_qualification import TaskProfile, export_task_matrix, run_task_profile

    modeldir=tmp_path/'model'
    modeldir.mkdir()
    text=dict(model_type='qwen3',vocab_size=300,hidden_size=8,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=4,intermediate_size=16,
        max_position_embeddings=256,tie_word_embeddings=True,bos_token_id=1,eos_token_id=2)
    model=Qwen3ForReranking(Qwen3RerankerConfig.from_dict({'text_config':text,
        'model_type':'qwen3_reranker','max_seq_length':256}))
    _write_tokenizer(modeldir)
    (modeldir/'config.json').write_text(json.dumps(text))
    (modeldir/'config_sentence_transformers.json').write_text(
        json.dumps({'model_type':'CrossEncoder'}))
    from safetensors.numpy import save_file

    save_file({name:np.asarray(value) for name,value in tree_flatten(model.parameters())},
              str(modeldir/'model.safetensors'))
    inputs=tmp_path/'inputs.json'
    inputs.write_text(json.dumps({'schema_version':'native-task-inputs-v1','cases':[
        {'request_id':str(i),'payload':{'query':'hello','documents':[word]}}
        for i,word in enumerate(('hello','world'))]}))
    profile=TaskProfile('qwen3_reranker','reranking','synthetic/smoke','a'*40,str(modeldir),
        str(inputs),file_digest(inputs),slots=2,validation_kind='synthetic-smoke')
    from pathlib import Path

    result_root=Path(os.environ.get('MLX_ONE_NATIVE_RESULTS_DIR',str(tmp_path)))
    output=run_task_profile(profile,result_root/'task-smoke')
    validate_integrity(output,required=('measurement.json','task-parity.json','task-evidence.json'))
    assert json.loads((output/'task-parity.json').read_text())['passed']
    assert json.loads((output/'task-evidence.json').read_text())['correctness_passed'] is False
    rows=export_task_matrix([output],tmp_path/'matrix')
    from mlx_one.runtime_baseline import registry_inventory

    expected = {row['model_type'] for row in registry_inventory()['registrations']}
    assert len(expected) == 30
    assert {row['model_type'] for row in rows} == expected
    assert all(not row['qualification_promoted'] for row in rows)
