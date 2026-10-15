"""Owned native SAM3 workflows; image features are distinct from decoder KV."""
from __future__ import annotations

import hashlib
from concurrent.futures import CancelledError

from mlx_one.engine.cache_errors import CacheCapacityError
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.task_runners.base import NativeTaskAdapter
from mlx_one.engine.tasks import PreparedTask, TaskFailure


def _float_arrays(tree):
    if isinstance(tree,dict):
        for value in tree.values():
            yield from _float_arrays(value)
    elif isinstance(tree,(list,tuple)):
        for value in tree:
            yield from _float_arrays(value)
    elif hasattr(tree,'dtype'):
        yield tree


class SAM3ConceptAdapter(NativeTaskAdapter):
    family='sam3'
    supported_tasks=(TaskKind.IMAGE_SEGMENTATION,)

    def __init__(self,bundle):
        if bundle.architecture!=self.family:
            raise ValueError('segmentation adapter/checkpoint mismatch')
        for tensor in _float_arrays(bundle.model.parameters()):
            if str(tensor.dtype).rsplit('.',1)[-1] != 'float32':
                raise ValueError('SAM3 native task execution requires float32 weights')
        self.bundle=bundle

    def prepare(self,task,payload):
        from mlx_one.segmentation.processing import (
            read_image,
            validate_coordinates,
            validate_labels,
        )

        task=TaskKind(task)
        if task not in self.supported_tasks:
            raise ValueError('segmentation task unavailable for this bundle')
        source=read_image(payload['image'])
        if source.width*source.height*3>64*1024**2:
            source.close()
            raise CacheCapacityError('segmentation image exceeds preparation quota',retryable=False)
        image=source.copy()
        source.close()
        try:
            data=dict(payload,image=image)
            w,h=image.size
            for name,boxes in (('boxes',True),('points',False)):
                if data.get(name) is not None:
                    data[name]=validate_coordinates(data[name],w,h,boxes=boxes).copy()
            for label,coordinates,default in (('point_labels','points',None),
                                               ('box_labels','boxes',1)):
                if data.get(coordinates) is not None and (label in data or default is not None):
                    data[label]=validate_labels(data.get(label),len(data[coordinates]),
                                                default=default).copy()
            digest=hashlib.sha256(image.tobytes()+repr(image.size).encode()).hexdigest()
            data['_image_identity']=digest
            size=self.bundle.processor.image_size
            # Conservative image encoder/feature forecast; no decoder KV is charged.
            forecast=size*size*256+image.width*image.height*24
            return PreparedTask(task,data,(1,),
                (self.family,self.bundle.task,size),forecast)
        except BaseException:
            image.close()
            raise

    def dispose_prepared(self,prepared):
        prepared.payload['image'].close()

    def run_steps(self,prepared,controls):
        from mlx_one.segmentation.inference import generate_masks, predict_masks, segment_image

        results=[]
        for p,control in zip(prepared,controls,strict=True):
            try:
                if control.cancelled():
                    raise CancelledError('segmentation cancelled')
                data={k:v for k,v in p.payload.items() if not k.startswith('_')}
                function=(generate_masks if p.task==TaskKind.MASK_GENERATION else
                          segment_image if self.family=='sam3' else predict_masks)
                result=function(self.bundle,**data)
                if control.cancelled():
                    raise CancelledError('segmentation cancelled after evaluation')
                results.append(result)
            except Exception as error:
                results.append(TaskFailure(error))
            yield None
        return tuple(results)

    def run_batch(self,prepared):
        from threading import Event

        from mlx_one.engine.tasks import TaskControl

        iterator=self.run_steps(prepared,tuple(TaskControl(Event(),None) for _ in prepared))
        while True:
            try:
                next(iterator)
            except StopIteration as result:
                return result.value


class SAM3InteractiveAdapter(SAM3ConceptAdapter):
    family='sam3_tracker'
    supported_tasks=(TaskKind.IMAGE_SEGMENTATION,TaskKind.MASK_GENERATION)


class SAM3VideoAdapter(NativeTaskAdapter):
    family='sam3_video'
    supported_tasks=(TaskKind.VIDEO_SEGMENTATION,)

    def __init__(self,bundle):
        if bundle.architecture!=self.family or bundle.task!='video':
            raise ValueError('video task requires complete video bundle')
        self.bundle=bundle

    def prepare(self,task,payload):
        if TaskKind(task)!=TaskKind.VIDEO_SEGMENTATION:
            raise ValueError('video task unavailable')
        # Stateful streaming admission uses open_video_session. Do not eagerly
        # decode or collect an unbounded input iterator in the task queue.
        raise ValueError('use open_video_session for ordered, bounded frame execution')
