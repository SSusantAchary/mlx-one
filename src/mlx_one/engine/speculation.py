"""Greedy native MTP proposals with evaluated, reversible target-state ownership."""
from __future__ import annotations

import time
from contextlib import ExitStack
from types import SimpleNamespace

from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.modes import ModelCapabilityUnsupported

SPECULATION_VERSION='native-speculation-v1'


def validate_mtp(bundle,*,temperature=0):
    if (bundle.architecture!='qwen3_5' or not hasattr(bundle.model,'mtp')
            or not callable(getattr(bundle.model,'make_mtp_cache',None))):
        raise ModelCapabilityUnsupported('native MTP requires complete Qwen3.5 predictor weights')
    if temperature!=0:
        raise ModelCapabilityUnsupported('native MTP currently supports greedy decoding only')


class SpeculativeProposal:
    def __init__(self,bundle,cache,*,runtime=None,handle=None,cancelled=lambda:False,observer=None):
        self.bundle,self.cache=bundle,cache
        self.runtime,self.handle=runtime,handle
        self.cancelled,self.observer=cancelled,observer
        self.resources=ExitStack()
        self.candidates=[]
        self.verified=None
        self.verify_ids=()
        self.accepted=self.drafted=0
        self.closed=False

    def _forward(self,ids,cache,kind):
        import mlx.core as mx

        started=time.perf_counter()
        output=self.bundle.model(mx.array([ids]),cache=cache)
        mx.eval(output.logits)
        if self.observer is not None:
            self.observer({'kind':'forward','phase':'decode','duration_ms':
                (time.perf_counter()-started)*1000,'at':time.perf_counter(),
                'batch_size':1,'input_tokens':len(ids),'speculative_stage':kind,
                'logits_dtype':str(output.logits.dtype)})
        return output

    def propose_mtp(self,first,hidden,maximum):
        import mlx.core as mx

        validate_mtp(self.bundle)
        model=self.bundle.model
        try:
            if self.runtime is not None:
                from mlx_one.engine.pressure import execution_forecast

                amount=execution_forecast(self.runtime,model.config,batch=1,chunk=maximum+1,
                    history=self.cache[0].offset,explicit_bytes=CacheBundle(self.cache).
                    allocated_nbytes*3,phase='decode')['bytes']
                self.resources.enter_context(self.runtime.transient_reservation(amount))
            self.before=CacheBundle(self.cache).clone()
            self.resources.callback(self.before.release)
            self.verified=self.before.clone()
            self.resources.callback(self.verified.release)
            mtp=CacheBundle(tuple(model.make_mtp_cache()))
            self.resources.callback(mtp.release)
            previous=first
            drafts=[]
            for _ in range(maximum):
                if self.cancelled():
                    return self
                hidden=model.mtp(model.language_model.embed_tokens(mx.array([[previous]])),hidden,
                    cache=mtp.entries,position_offset=self.cache[0].offset)
                logits=(model.language_model.embed_tokens.as_linear(hidden)
                        if model.config.tie_word_embeddings else model.lm_head(hidden))
                mx.eval(logits)
                previous=int(mx.argmax(logits[0,-1]).item())
                drafts.append(previous)
            self.drafted=len(drafts)
            self.verify_ids=tuple([first,*drafts])
            verified=self._forward(self.verify_ids,self.verified.entries,'verify')
            if verified.logits.shape[:2]!=(1,len(self.verify_ids)):
                raise ValueError('MTP verification requires logits at every proposed position')
            predictions=[int(t) for t in mx.argmax(verified.logits[0],axis=-1).tolist()]
            for index,draft in enumerate(drafts):
                if draft!=predictions[index]:
                    self.candidates=[first,*drafts[:index],predictions[index]]
                    self.accepted=index
                    return self
            self.accepted=len(drafts)
            self.candidates=[first,*drafts,predictions[-1]]
            return self
        except BaseException:
            self.close()
            raise

    def commit(self,consumed):
        """Leave the final sampled token uncomputed, including stop/limit boundaries."""
        if self.cancelled() or consumed<1:
            return
        if consumed>len(self.candidates):
            raise ValueError('speculative consumed count exceeds verified candidates')
        ids=tuple(self.candidates[:consumed-1])
        if not ids:
            return
        staged=self.verified
        if ids!=self.verify_ids:
            staged=self.before.clone()
            self.resources.callback(staged.release)
            self._forward(ids,staged.entries,'rollback-replay')
        from mlx_one.engine.batching import ForwardRequest
        from mlx_one.engine.transactions import commit_evaluated

        request=ForwardRequest(self.bundle,ids,self.cache,'decode',(),self.runtime,self.handle,
            request_id='mtp',cancel=SimpleNamespace(is_set=self.cancelled))
        commit_evaluated((request,),(staged,))

    def close(self):
        if not self.closed:
            self.closed=True
            self.resources.close()


def tokenizer_semantics(bundle):
    import hashlib
    import json

    tokenizer=bundle.tokenizer
    serialize=getattr(getattr(tokenizer,'_tokenizer',None),'to_str',None)
    if not callable(serialize):
        raise ModelCapabilityUnsupported('external draft requires serialized tokenizer semantics')
    payload={'tokenizer':json.loads(serialize()),'special_tokens':tokenizer.special_tokens,
        'ids':[tokenizer.bos_token_id,tokenizer.eos_token_id,tokenizer.pad_token_id],
        'eos':sorted(bundle.eos_token_ids)}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def validate_external(target,draft,*,temperature=0):
    from mlx_one.engine.adapters import native_adapter

    if draft is None or temperature!=0:
        raise ModelCapabilityUnsupported(
            'external draft requires a native bundle and greedy decoding')
    if tokenizer_semantics(target)!=tokenizer_semantics(draft):
        raise ModelCapabilityUnsupported('external draft tokenizer/token/EOS semantics differ')
    for bundle in (target,draft):
        adapter=native_adapter(bundle)
        if adapter.spec.hybrid or bundle.architecture not in {'qwen2','qwen3','llama'}:
            raise ModelCapabilityUnsupported(
                'external drafting requires supported full-attention models')


def prepare_external(proposal,draft,prefix,first,maximum):
    """Request-private draft state; target verification is scheduled separately."""
    import mlx.core as mx

    from mlx_one.core.cache import configure_kv_caches

    try:
        if proposal.runtime is not None:
            from mlx_one.engine.pressure import execution_forecast

            amount=execution_forecast(proposal.runtime,proposal.bundle.model.config,
                batch=1,chunk=maximum+1,history=proposal.cache[0].offset,
                explicit_bytes=CacheBundle(proposal.cache).allocated_nbytes*3,phase='decode')['bytes']
            # Draft caches + full-prefix activations are distinct from target replacements.
            from mlx_one.engine.cache_planner import build_cache_plan

            plan=build_cache_plan(draft)
            amount+=plan.bytes_per_token()*((len(prefix)+maximum+255)//256*256)
            hidden=getattr(draft.model.config,'hidden_size',1)
            heads=getattr(draft.model.config,'num_attention_heads',1)
            amount+=len(prefix)**2*heads*20+len(prefix)*hidden*32
            proposal.resources.enter_context(proposal.runtime.transient_reservation(amount))
        proposal.before=CacheBundle(proposal.cache).clone()
        proposal.resources.callback(proposal.before.release)
        proposal.verified=proposal.before.clone()
        proposal.resources.callback(proposal.verified.release)
        state=CacheBundle(configure_kv_caches(draft.model.make_cache(),'f16','f16'))
        proposal.resources.callback(state.release)
        current=tuple(prefix)+(first,)
        drafts=[]
        for _ in range(maximum):
            if proposal.cancelled():
                break
            output=draft.model(mx.array([current]),cache=state.entries)
            mx.eval(output.logits)
            token=int(mx.argmax(output.logits[0,-1]).item())
            drafts.append(token)
            current=(token,)
        proposal.drafted=len(drafts)
        proposal.verify_ids=tuple([first,*drafts])
        return proposal.verify_ids
    except BaseException:
        proposal.close()
        raise


def accept_verification(proposal,logits):
    import mlx.core as mx

    if logits.shape[:2]!=(1,len(proposal.verify_ids)):
        raise ValueError('external verification requires logits for every proposed position')
    predicted=[int(x) for x in mx.argmax(logits[0],axis=-1).tolist()]
    first,*drafts=proposal.verify_ids
    for index,token in enumerate(drafts):
        if token!=predicted[index]:
            proposal.accepted=index
            proposal.candidates=[first,*drafts[:index],predicted[index]]
            return
    proposal.accepted=len(drafts)
    proposal.candidates=[first,*drafts,predicted[-1]]
