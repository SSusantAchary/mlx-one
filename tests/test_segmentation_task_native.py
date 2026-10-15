"""Opt-in native video state/service ownership; no checkpoint quality claim."""
import os
from types import MethodType

import numpy as np
import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS')!='1':
    pytest.skip('requires supervised Metal execution',allow_module_level=True)

from test_sam3_workflows import fake_bundle as fake_bundle

from mlx_one.engine.service import InferenceService
from mlx_one.segmentation.video import Sam3VideoSession


def test_video_service_model_epoch_session_order_and_peer_isolation(fake_bundle):
    bundle=fake_bundle
    bundle.architecture='sam3_video'
    bundle.model.parameters=lambda:{'weight':np.ones(1,dtype=np.float32)}
    bundle.video_session=MethodType(lambda self,**kw:Sam3VideoSession(self,**kw),bundle)
    service=InferenceService()
    service.register_task('video',bundle)
    try:
        first=service.open_video_session('video')
        second=service.open_video_session('video')
        first.call('add_prompt','truck',event_number=0)
        second.call('add_prompt','truck',event_number=0)
        frame=np.zeros((4,6,3),dtype=np.uint8)
        assert first.call('process_frame',frame,frame_index=0,event_number=1).object_ids==(0,)
        with pytest.raises(ValueError,match='reordered'):
            first.call('process_frame',frame,event_number=1)
        with pytest.raises(TimeoutError):
            service.unload('video',timeout=.01)
        first.close()
        first.close()
        assert second.call('process_frame',frame,event_number=1).frame_index==0
        second.call('reset',event_number=2)
        second.close()
        assert service.pool.stats()['active_leases']==0 and not service._sessions
        service.unload('video')
    finally:
        service.close()
