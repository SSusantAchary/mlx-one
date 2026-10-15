"""Native fixed-vector, pair-score and ragged token-feature execution."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.task_runners.base import NativeTaskAdapter
from mlx_one.engine.tasks import PreparedTask
from mlx_one.retrieval.schemas import EmbeddingResult, RerankItem, RerankResult
from mlx_one.retrieval.service import (
    DEFAULT_RETRIEVAL_INSTRUCTION,
    RERANK_PREFIX,
    RERANK_SUFFIX,
    _single_token,
    _validate_batch_options,
    _validate_texts,
)


@dataclass(frozen=True)
class TokenFeatureResult:
    texts: tuple[str, ...]
    embeddings: tuple
    token_ids: tuple[tuple[int, ...], ...]
    masks: tuple[tuple[bool, ...], ...]
    dimensions: int
    model: str
    revision: str | None
    execution_identity: str = ""

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(),sort_keys=True)


class RetrievalAdapter(NativeTaskAdapter):
    family=None

    def __init__(self,bundle):
        if bundle.architecture!=self.family:
            raise ValueError('retrieval adapter/checkpoint mismatch')
        import hashlib

        from mlx_one.engine.reporting import tensor_dtypes

        self.bundle=bundle
        self.identity=hashlib.sha256(repr((id(bundle.model),bundle.model.config,
            tensor_dtypes(bundle.model.parameters()),id(bundle.tokenizer))).encode()).hexdigest()

    def prepare(self,task,payload):
        task=TaskKind(task)
        config=self.bundle.model.config
        late=self.family=='lfm2_colbert'
        if task==TaskKind.RERANKING:
            if self.family!='qwen3_reranker' or self.bundle.task!='reranking':
                raise ValueError('checkpoint has no pair scoring task')
            query=payload['query']
            texts=tuple(payload['documents'])
            if not isinstance(query,str) or not query:
                raise ValueError('query must be nonempty')
        else:
            if self.family=='qwen3_reranker' or self.bundle.task!='embedding':
                raise ValueError('checkpoint has no embedding task')
            if task not in ({TaskKind.LATE_INTERACTION} if late else {TaskKind.EMBEDDING}):
                raise ValueError('ColBERT requires ragged late-interaction output')
            value=payload['texts']
            texts=(value,) if isinstance(value,str) else tuple(value)
        _validate_texts(texts,'retrieval inputs')
        capacity=(getattr(config,'document_max_length',0) if late else
                  getattr(config,'max_seq_length',getattr(config,'sentence_max_length',0)))
        limit=payload.get('max_length',capacity)
        _validate_batch_options(limit,1,capacity)
        input_type=payload.get('input_type','document')
        if input_type not in {'query','document'}:
            raise ValueError('input_type must be query or document')
        instruction=payload.get('instruction',DEFAULT_RETRIEVAL_INSTRUCTION)
        if not isinstance(instruction,str) or not instruction:
            raise ValueError('instruction must be nonempty')
        size=(config.projection_dim if late else
              config.validate_dimensions(payload.get('dimensions'))
              if hasattr(config,'validate_dimensions') else getattr(config,'hidden_size',0))
        if not late and not hasattr(config,'validate_dimensions') and payload.get(
                'dimensions') not in {None,size}:
            raise ValueError('dimensions differ from native encoder width')
        tokenizer=self.bundle.tokenizer
        if task==TaskKind.RERANKING:
            prefix,suffix=tokenizer.encode(RERANK_PREFIX),tokenizer.encode(RERANK_SUFFIX)
            remaining=limit-len(prefix)-len(suffix)
            if remaining<1:
                raise ValueError('max_length leaves no pair capacity')
            top_k=payload.get('top_k')
            if top_k is not None and (isinstance(top_k,bool)
                                      or not isinstance(top_k,int) or top_k<1):
                raise ValueError('top_k must be positive')
            sequences=tuple(tuple(prefix+tokenizer.encode(
                f'<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {t}')[:remaining]+suffix)
                for t in texts)
            key=(_single_token(self.bundle,'no'),_single_token(self.bundle,'yes'))
            if key[0]==key[1]:
                raise ValueError('reranker yes/no IDs must be distinct')
        else:
            if input_type=='document' and 'instruction' in payload:
                raise ValueError('instruction is only valid for queries')
            if late and input_type=='query':
                limit=min(limit,config.query_max_length)
            sequences=tuple(tuple(tokenizer.encode(
                f'Instruct: {instruction}\nQuery:{t}'
                if input_type=='query' and self.family=='qwen3_embedding' else t,
                add_special_tokens=True,max_length=limit)) for t in texts)
            key=(size,)
        if any(not seq for seq in sequences):
            raise ValueError('empty tokenized input is unsupported')
        data={'texts':texts,'sequences':sequences,'options':dict(payload),
              'input_type':input_type,'size':size}
        max_width=max(map(len,sequences))
        hidden=getattr(getattr(config,'text_config',config),'hidden_size',size)
        transient=len(sequences)*max_width*(hidden*32+max_width*16+size*4)
        return PreparedTask(task,data,tuple(map(len,sequences)),key,transient)

    def run_batch(self,prepared):
        import mlx.core as mx

        sequences=[seq for request in prepared for seq in request.payload['sequences']]
        # ColBERT's convolution backbone has no general padded-carry contract:
        # execute exact-length groups, with no persistent cross-document carries.
        groups={}
        for index,seq in enumerate(sequences):
            key=len(seq) if self.family=='lfm2_colbert' else (len(seq)-1).bit_length()
            groups.setdefault(key,[]).append(index)
        outputs={}
        for indices in groups.values():
            ids,masks=self.bundle.tokenizer.pad([sequences[i] for i in indices],
                padding_side='right' if self.family=='bert' else 'left')
            if any(sum(mask)!=len(sequences[index]) for index,mask in
                   zip(indices,masks,strict=True)):
                raise ValueError('tokenizer padding mask does not preserve real inputs')
            arguments={'attention_mask':mx.array(masks)}
            if prepared[0].task==TaskKind.RERANKING:
                arguments.update(false_token_id=prepared[0].semantic_key[0],
                                 true_token_id=prepared[0].semantic_key[1])
            elif hasattr(self.bundle.model.config,'validate_dimensions'):
                arguments['dimensions']=prepared[0].payload['size']
            result=self.bundle.model(mx.array(ids),**arguments)
            if prepared[0].task==TaskKind.RERANKING:
                mx.eval(result.scores,result.probabilities)
                rows=list(zip(result.scores.tolist(),result.probabilities.tolist(),strict=True))
            else:
                mx.eval(result.embeddings)
                expected=3 if prepared[0].task==TaskKind.LATE_INTERACTION else 2
                if (result.embeddings.ndim!=expected or result.embeddings.shape[0]!=len(indices)
                        or result.embeddings.shape[-1]!=prepared[0].payload['size']):
                    raise ValueError('native embedding output shape differs from task contract')
                rows=result.embeddings.tolist()
            for index,row in zip(indices,rows,strict=True):
                outputs[index]=row
        cursor=0
        results=[]
        for request in prepared:
            data=request.payload
            count=len(data['texts'])
            rows=[outputs[i] for i in range(cursor,cursor+count)]
            cursor+=count
            if request.task==TaskKind.RERANKING:
                items=[RerankItem(i,text,float(score),float(prob))
                    for i,(text,(score,prob)) in enumerate(zip(data['texts'],rows,strict=True))]
                items.sort(key=lambda item:(-item.score,item.index))
                result=RerankResult(data['options']['query'],tuple(items[:data['options'].get(
                    'top_k')]),self.bundle.model_id,self.bundle.revision)
            elif request.task==TaskKind.LATE_INTERACTION:
                result=TokenFeatureResult(data['texts'],tuple(tuple(tuple(v) for v in row)
                    for row in rows),data['sequences'],tuple(tuple(True for _ in seq)
                    for seq in data['sequences']),data['size'],self.bundle.model_id,
                    self.bundle.revision,self.identity)
            else:
                result=EmbeddingResult(data['texts'],tuple(tuple(float(v) for v in row)
                    for row in rows),data['size'],data['input_type'],self.bundle.model_id,
                    self.bundle.revision)
            results.append(result)
        return tuple(results)


class BertEmbeddingAdapter(RetrievalAdapter):
    family='bert'
    supported_tasks=(TaskKind.EMBEDDING,)


class MPNetEmbeddingAdapter(RetrievalAdapter):
    family='mpnet'
    supported_tasks=(TaskKind.EMBEDDING,)


class Qwen3EmbeddingAdapter(RetrievalAdapter):
    family='qwen3_embedding'
    supported_tasks=(TaskKind.EMBEDDING,)


class Qwen3RerankAdapter(RetrievalAdapter):
    family='qwen3_reranker'
    supported_tasks=(TaskKind.RERANKING,)


class LFM2ColBERTAdapter(RetrievalAdapter):
    family='lfm2_colbert'
    supported_tasks=(TaskKind.LATE_INTERACTION,)


def maxsim(query,document):
    """Ragged CPU result scoring; all-masked inputs fail explicitly."""
    import numpy as np

    if query.model!=document.model or query.revision!=document.revision or (
            query.dimensions!=document.dimensions or not query.execution_identity
            or query.execution_identity!=document.execution_identity):
        raise ValueError('late-interaction feature identities differ')
    if len(query.embeddings)!=1 or not document.embeddings:
        raise ValueError('MaxSim requires one query and nonempty documents')
    scores=[]
    for vectors,mask in zip(document.embeddings,document.masks,strict=True):
        q=np.asarray(query.embeddings[0])[np.asarray(query.masks[0],dtype=bool)]
        d=np.asarray(vectors)[np.asarray(mask,dtype=bool)]
        if not len(q) or not len(d):
            raise ValueError('all-masked late-interaction inputs are unsupported')
        scores.append(float((q@d.T).max(axis=1).sum()))
    return tuple(scores)
