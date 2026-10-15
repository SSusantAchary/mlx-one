"""Static-image VLM integration, with explicitly singleton decoder forwards."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from mlx_one.engine.cache_errors import CacheCapacityError
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.task_runners.base import NativeTaskAdapter
from mlx_one.engine.tasks import PreparedTask
from mlx_one.text.schemas import GenerationResult, TextGenerationOptions


class VisionTaskAdapter(NativeTaskAdapter):
    supported_tasks=(TaskKind.VISION_LANGUAGE,)

    def __init__(self,bundle):
        if bundle.architecture!=self.family:
            raise ValueError('vision adapter/checkpoint mismatch')
        self.bundle=bundle

    def prepare(self,task,payload):
        from PIL import Image

        if TaskKind(task)!=TaskKind.VISION_LANGUAGE or any(k in payload for k in ('video','audio')):
            raise ValueError('this VLM task currently supports static images only')
        prompt=payload['prompt']
        images=tuple(payload['images'])
        if not isinstance(prompt,str) or not prompt or not images or len(images)>8:
            raise ValueError('vision inputs require text and one to eight images')
        if prompt.count('<image>')!=len(images):
            raise ValueError('image placeholders must match images')
        settings=payload.get('options') or TextGenerationOptions(
            **dict(payload.get('generation',{})))
        if not isinstance(settings,TextGenerationOptions):
            raise ValueError('vision generation requires typed options')
        copies=[]
        digest=hashlib.sha256(str(getattr(self.bundle,'processor_identity',None)).encode())
        total=0
        try:
            for image in images:
                if isinstance(image,(str,Path)):
                    if Path(image).stat().st_size>64*1024**2:
                        raise CacheCapacityError(
                            'encoded media exceeds preparation quota',retryable=False)
                    source=Image.open(image)
                elif isinstance(image,Image.Image):
                    source=image
                else:
                    raise ValueError('image must be local path or decoded PIL image')
                try:
                    total+=source.width*source.height*3
                    if total>64*1024**2:
                        raise CacheCapacityError(
                            'decoded media exceeds preparation quota',retryable=False)
                    copy=source.convert('RGB').copy()
                    copies.append(copy)
                    digest.update(repr(copy.size).encode())
                    digest.update(copy.tobytes())
                finally:
                    if source is not image:
                        source.close()
        except BaseException:
            for copy in copies:
                copy.close()
            raise
        try:
            config=self.bundle.model.config
            vision=config.vision_config
            patch=vision.patch_size
            if self.family=='lfm2_vl':
                patches=len(copies)*(vision.image_size//patch)**2
            else:
                from mlx_one.vision.processing import _smart_resize

                processor=self.bundle.processor
                patches=sum((w//patch)*(h//patch) for w,h in (_smart_resize(
                    im.width,im.height,factor=patch*vision.spatial_merge_size,
                    min_pixels=processor.min_pixels,max_pixels=processor.max_pixels)
                    for im in copies))
            cost=len(self.bundle.tokenizer.encode(prompt))+patches+2*len(copies)
            hidden=config.text_config.hidden_size
            heads=getattr(vision,'num_attention_heads',getattr(vision,'num_heads',16))
            bound=total*8+patches*patches*heads*4+(cost+settings.max_tokens)*hidden*32
            return PreparedTask(TaskKind.VISION_LANGUAGE,{'prompt':prompt,'images':tuple(copies),
                'options':settings,'media_identity':digest.hexdigest()},(cost,),
                (self.family,getattr(self.bundle,'processor_identity',None)),bound)
        except BaseException:
            for copy in copies:
                copy.close()
            raise

    def dispose_prepared(self,prepared):
        for image in prepared.payload['images']:
            image.close()

    def stream(self,prepared,cancel):
        from mlx_one.vision.generation import _stream_vlm

        data=prepared.payload
        try:
            yield from _stream_vlm(self.bundle,data['prompt'],data['images'],data['options'],
                cancel.is_set,yield_steps=True)
        finally:
            for image in data['images']:
                image.close()

    def run_steps(self,prepared,controls):
        from mlx_one.vision.generation import _stream_vlm

        streams=[]
        chunks=[[] for _ in prepared]
        prompt_counts=[None for _ in prepared]
        def observe(index,event):
            if event["kind"]=="prepared":
                prompt_counts[index]=len(event["tokens"])
        started=time.perf_counter()
        try:
            for i,(p,control) in enumerate(zip(prepared,controls,strict=True)):
                data=p.payload
                streams.append(_stream_vlm(self.bundle,data['prompt'],data['images'],
                    data['options'],control.cancelled,yield_steps=True,
                    observer=lambda event,index=i:observe(index,event)))
            active=set(range(len(streams)))
            while active:
                for i in tuple(sorted(active)):
                    if controls[i].cancelled():
                        streams[i].close()
                        active.remove(i)
                    else:
                        try:
                            chunks[i].append(next(streams[i]))
                        except StopIteration:
                            active.remove(i)
                    yield None
            return tuple(GenerationResult(p.payload['prompt'],''.join(c.text for c in rows),
                prompt_counts[i] or 0,rows[-1].generated_tokens if rows else 0,
                rows[-1].finish_reason if rows and rows[-1].finish_reason else 'stop',
                self.bundle.model_id,self.bundle.revision,
                {'total_seconds':time.perf_counter()-started})
                for i,(p,rows) in enumerate(zip(prepared,chunks,strict=True)))
        finally:
            for stream in streams:
                stream.close()
            for p in prepared:
                for image in p.payload['images']:
                    image.close()

    def run_batch(self,prepared):
        from threading import Event

        from mlx_one.engine.tasks import TaskControl

        iterator=self.run_steps(prepared,tuple(TaskControl(Event(),None) for _ in prepared))
        while True:
            try:
                next(iterator)
            except StopIteration as stopped:
                return stopped.value


class Qwen2VLAdapter(VisionTaskAdapter):
    family='qwen2_vl'


class Qwen25VLAdapter(VisionTaskAdapter):
    family='qwen2_5_vl'


class LFM2VLAdapter(VisionTaskAdapter):
    family='lfm2_vl'


class Qwen35VisionAdapter(VisionTaskAdapter):
    family='qwen3_5'
