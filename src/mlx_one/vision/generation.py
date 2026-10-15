"""Native media prefill with transactional, position-aware reference decoding."""
from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from types import SimpleNamespace

from mlx_one.engine.cache import CacheBundle
from mlx_one.text.schemas import GenerationChunk, TextGenerationOptions
from mlx_one.vision.loading import LoadedVLM


def _media_kwargs(prepared):
    if hasattr(prepared,'spatial_shapes'):
        return {'pixel_values':prepared.pixel_values,'spatial_shapes':prepared.spatial_shapes}
    return {'pixel_values':prepared.pixel_values,'image_grid_thw':prepared.image_grid_thw,
            'mm_token_type_ids':prepared.mm_token_type_ids}


class VisionForward:
    """One request owns media identity, multimodal positions and complete model state."""
    def __init__(self,bundle,prepared,cancelled):
        self.bundle,self.prepared,self.cancelled=bundle,prepared,cancelled
        self.delta=None
        self.request_id='vision-'+uuid.uuid4().hex
        self.text_bundle=None

    def __getattr__(self,name):
        return getattr(self.bundle.model,name)

    def __call__(self,ids,*,cache):
        import mlx.core as mx

        from mlx_one.engine.batching import ForwardRequest
        from mlx_one.engine.transactions import commit_evaluated

        state=CacheBundle(cache)
        draft=state.clone()
        initial=cache[0].offset==0
        try:
            kwargs=_media_kwargs(self.prepared) if initial else {}
            if not initial and self.delta is not None:
                base=mx.arange(cache[0].offset,cache[0].offset+ids.shape[1])[None,:]+self.delta
                kwargs['position_ids']=mx.stack((base,base,base),axis=0)
            output=self.bundle.model(ids,cache=draft.entries,**kwargs)
            mx.eval(output.logits)
            request=ForwardRequest(self.text_bundle,tuple(int(t) for t in ids[0].tolist()),
                cache,'prefill' if initial else 'decode',(),request_id=self.request_id,
                cancel=SimpleNamespace(is_set=self.cancelled))
            commit_evaluated((request,),(draft,))
            if initial:
                self.delta=getattr(output,'rope_deltas',None)
            return output
        finally:
            draft.release()


def _stream_vlm(bundle,prompt,images,options,cancelled,*,yield_steps=False,observer=None):
    from mlx_one.text.generation import _stream_bundle
    from mlx_one.text.loading import LoadedTextModel

    if cancelled():
        return
    prepared=bundle.processor.prepare(prompt,tuple(images))
    if cancelled():
        return
    config=bundle.model.config.text_config
    capacity=config.max_position_embeddings
    tokens=tuple(int(t) for t in prepared.input_ids[0].tolist())
    if observer is not None:
        observer({'kind':'prepared','tokens':tokens})
    if len(tokens)+options.max_tokens>capacity:
        raise ValueError('multimodal prompt plus output exceeds the checkpoint context')
    from mlx_one.engine.cache import array_nbytes
    from mlx_one.engine.media import FeatureStore

    features=FeatureStore()
    with ExitStack() as resources:
        resources.callback(features.close)
        lease=features.acquire('private:'+uuid.uuid4().hex,
            getattr(bundle,'processor_identity',None) or 'private-native-media',
            lambda:prepared,array_nbytes(vars(prepared)))
        resources.callback(lease.close)
        wrapper=VisionForward(bundle,lease.value,cancelled)
        text=LoadedTextModel(wrapper,bundle.tokenizer,bundle.path,bundle.model_id,bundle.revision,
                            bundle.architecture,capacity)
        wrapper.text_bundle=text
        state=CacheBundle(bundle.model.make_cache())
        resources.callback(state.release)
        yield from _stream_bundle(text,prompt,options,cancelled,cache=state.entries,
            yield_steps=yield_steps,_prepared_input_ids=tokens)


def stream_vlm(bundle:LoadedVLM,prompt:str,images:Sequence[object],*,
               options:TextGenerationOptions|None=None)->Iterator[GenerationChunk]:
    """Preserve native generation events; no shared media prefix or backend substitution."""
    yield from _stream_vlm(bundle,prompt,images,options or TextGenerationOptions(),lambda:False)
