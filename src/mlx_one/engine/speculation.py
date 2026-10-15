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
