"""SAM3 task identity, float32 policy, cancellation and ordered service ownership."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_dense_batching import native as native

from mlx_one.engine.adapter_registry import get_runtime_registration
from mlx_one.engine.service import InferenceService
from mlx_one.engine.task_runners.segmentation import SAM3ConceptAdapter, SAM3VideoAdapter


def bundle(dtype=np.float32):
    model=SimpleNamespace(parameters=lambda:{'w':np.ones(1,dtype=dtype)})
    return SimpleNamespace(architecture='sam3',task='concept',model=model,
                           processor=SimpleNamespace(image_size=12))


def test_segmentation_image_copy_and_strict_float32():
    with pytest.raises(ValueError,match='float32'):
        SAM3ConceptAdapter(bundle(np.float16))
    adapter=SAM3ConceptAdapter(bundle())
    image=Image.new('RGB',(8,6),'red')
    prepared=adapter.prepare('image-segmentation',{'image':image,'text':'truck'})
    image.putpixel((0,0),(0,0,0))
    assert prepared.payload['image'].getpixel((0,0))==(255,0,0)
    assert prepared.payload['_image_identity']
    adapter.dispose_prepared(prepared)
    with pytest.raises(ValueError):
        adapter.prepare('image-segmentation',{'image':image,'boxes':[[0,0,9,2]]})
    image.close()


def test_native_task_service_routes_without_generation_or_kv(native,monkeypatch):
    import mlx_one.segmentation.inference

    service=InferenceService()
    b=bundle()
    service.register_task('sam',b)
    monkeypatch.setattr(mlx_one.segmentation.inference,'segment_image',
        lambda bundle,**kw:('mask',kw['image'].size,kw['text']))
    try:
        output=service.run_task('sam','image-segmentation',
            {'image':Image.new('RGB',(8,6)),'text':'truck'}).output
        assert output==('mask',(8,6),'truck')
        assert service.pool.stats()['active_leases']==0
    finally:
        service.close()


def test_nested_video_tracker_has_no_independent_task_authority():
    binding=get_runtime_registration('sam3_tracker_video')
    assert binding.component_only and not binding.adapter_path
    assert get_runtime_registration('sam3_video').adapter() is SAM3VideoAdapter
