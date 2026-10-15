from dataclasses import replace

import pytest
from test_dense_batching import native as native
from test_paged_hybrid import _requests
from test_paged_hybrid import advanced as advanced

from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.hybrid import HybridRunner


def test_moe_hybrid_reuses_components_and_balances_failure(advanced,monkeypatch):
    bundle=replace(advanced(True),architecture='lfm2_moe')
    bundle.model.config.model_type='lfm2_moe'
    bundle.model.config.num_experts=4
    bundle.model.config.moe_intermediate_size=12
    runtime=CacheManager(bundle,CacheConfig(backend='hybrid'))
    requests=tuple(_requests(bundle,runtime,[(1,2,3),(4,)]))
    runner=HybridRunner()
    runner.execute(requests)
    assert bundle.model.calls[-1]==(2,3)
    assert [r.cache[0].offset for r in requests]==[3,1]
    assert runner.capabilities().family=='lfm2_moe'
    assert runner.last_trace['adapter_transient_bound_bytes']>0
    before=[r.cache[0].values.copy() for r in requests]
    def fail(handle):
        raise ValueError('component commit injection')
    monkeypatch.setattr(runtime,'reconcile',fail)
    with pytest.raises(ValueError,match='commit injection'):
        runner.execute(tuple(replace(r,phase='decode',token_ids=(5,)) for r in requests))
    import numpy as np

    for r,old in zip(requests,before,strict=True):
        np.testing.assert_array_equal(r.cache[0].values,old)
    assert [r.cache[0].offset for r in requests]==[3,1]
    runtime.close()
    assert runtime.stats().active_handles==runtime.stats().reserved_bytes==0
