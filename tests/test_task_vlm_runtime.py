"""CPU media ownership, dispatch and processor policy regressions."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from test_dense_batching import native as native

from mlx_one.engine.adapter_registry import get_runtime_registration
from mlx_one.engine.cache_errors import CacheCapacityError
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.media import FeatureStore
from mlx_one.engine.task_runners.multimodal import Qwen25VLAdapter
from mlx_one.vision.processing import QwenImageProcessor


def test_media_leases_pin_eviction_and_private_namespaces():
    store=FeatureStore(4)
    first=store.acquire('alice','same',lambda:object(),4)
    with pytest.raises(CacheCapacityError):
        store.acquire('bob','same',lambda:object(),4)
    with pytest.raises(RuntimeError,match='live'):
        store.close()
    first.close()
    second=store.acquire('bob','same',lambda:object(),4)
    assert store.stats()=={'bytes':4,'leases':1,'entries':1}
    second.close()
    second.close()
    store.close()
    assert store.stats()['bytes']==0


def test_qwen35_keeps_separate_text_and_media_adapters():
    binding=get_runtime_registration('qwen3_5')
    assert binding.adapter().__name__=='Qwen35Adapter'
    assert binding.adapter(TaskKind.VISION_LANGUAGE).__name__=='Qwen35VisionAdapter'
    assert {t.task for t in binding.specifications()[0]}=={
        TaskKind.TEXT_GENERATION,TaskKind.VISION_LANGUAGE}


def processor_config():
    return SimpleNamespace(vision_config=SimpleNamespace(patch_size=2,
        temporal_patch_size=1,spatial_merge_size=2),vision_start_token_id=61,
        image_token_id=60,vision_end_token_id=62)


def test_processor_rejects_layout_and_normalization_substitution():
    data={'image_mean':[.5,.5,.5],'image_std':[.5,.5,.5]}
    QwenImageProcessor(None,processor_config(),metadata=data)
    for changed in ({'patch_size':14},{'image_std':[0,1,1]},
                    {'resample':0},{'rescale_factor':1.},{'do_normalize':False}):
        with pytest.raises(ValueError):
            QwenImageProcessor(None,processor_config(),metadata={**data,**changed})


def test_image_preparation_copies_content_and_rejects_video(native,tmp_path):
    from PIL import Image

    config=processor_config()
    config.text_config=SimpleNamespace(hidden_size=8,max_position_embeddings=256)
    model=SimpleNamespace(config=config)
    tokenizer=SimpleNamespace(encode=lambda text:[1,2])
    bundle=SimpleNamespace(model=model,tokenizer=tokenizer,architecture='qwen2_5_vl',
        processor=QwenImageProcessor(tokenizer,config,min_pixels=16,max_pixels=16),
        processor_identity='processor')
    adapter=Qwen25VLAdapter(bundle)
    image=Image.new('RGB',(4,4),'red')
    prepared=adapter.prepare(TaskKind.VISION_LANGUAGE,{'prompt':'<image> describe','images':[image],
                                                      'generation':{'max_tokens':2}})
    image.putpixel((0,0),(0,0,0))
    assert prepared.payload['images'][0].getpixel((0,0))==(255,0,0)
    with pytest.raises(ValueError,match='static'):
        adapter.prepare(TaskKind.VISION_LANGUAGE,{'video':[]})
    adapter.dispose_prepared(prepared)
    image.close()


def test_cancel_during_vision_forward_does_not_adopt(native):
    from mlx_one.engine.cache import CacheBundle
    from mlx_one.vision.generation import VisionForward

    cancelled=threading.Event()
    prepared=SimpleNamespace(pixel_values=np.ones(1),image_grid_thw=((1,1,1),),
                             mm_token_type_ids=np.ones((1,1)))
    class Model:
        def __call__(self,ids,*,cache,**kwargs):
            data=np.ones((1,1,len(ids[0]),2),dtype=np.float16)
            for entry in cache:
                entry.update(data,data)
            cancelled.set()
            return SimpleNamespace(logits=np.ones((1,len(ids[0]),4)),rope_deltas=None)
    wrapped=VisionForward(SimpleNamespace(model=Model()),prepared,cancelled.is_set)
    wrapped.text_bundle=native
    state=CacheBundle(native.model.make_cache())
    wrapped(np.array([[1,2]]),cache=state.entries)
    assert all(c.offset==0 for c in state.entries)
    state.release()
