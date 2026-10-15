"""Crash, corruption, private namespaces and owner adoption for the SSD tier."""
from dataclasses import replace

import numpy as np
import pytest
from test_dense_batching import native as native

from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig, LayerStateSpec
from mlx_one.engine.cold_prefix import ColdPrefixStore
from mlx_one.engine.components import FullKVComponent


def snapshot():
    component=FullKVComponent()
    component.update(np.ones((1,1,4,2),np.float16),np.ones((1,1,4,2),np.float16))
    return CacheBundle((component,))


def test_cold_generations_identity_trim_corruption_and_atomic_failure(native,tmp_path,monkeypatch):
    import mlx_one.engine.cold_prefix as cold

    manager=CacheManager(native)
    fp=manager.fingerprint(namespace='alice')
    store=ColdPrefixStore(tmp_path/'ssd',budget_bytes=1000000,staging_bytes=1000000)
    specs=(LayerStateSpec(0,'full_kv',1,2,2,'float16'),)
    donor=snapshot()
    try:
        assert store.put(fp,(1,2,3,4),donor,specs)
        assert store.preview(replace(fp,namespace='bob'),(1,2,3,5))==0
        assert store.preview(fp,(1,2,3,4))==3
        count,restored=store.lookup(fp,(1,2,3,5),specs)
        assert count==3 and restored.offset==3 and donor.offset==4
        restored.release()
        old=(store.root/'index.json').read_bytes()
        original=cold._atomic_json
        monkeypatch.setattr(cold,'_atomic_json',lambda *a,**kw:(_ for _ in ()).throw(
            OSError('commit injection')))
        with pytest.raises(OSError,match='injection'):
            store.put(fp,(5,6,7,8),donor,specs)
        assert (store.root/'index.json').read_bytes()==old
        assert store.preview(fp,(1,2,3,4))==3
        monkeypatch.setattr(cold,'_atomic_json',original)
        assert len([p for p in store.root.iterdir() if p.is_dir()])==1
        payload=next(store.root.glob('*/*.safetensors'))
        data=payload.read_bytes()
        payload.write_bytes(data[:-1]+bytes([data[-1]^1]))
        with pytest.raises(ValueError,match='checksum'):
            store.lookup(fp,(1,2,3,5),specs)
        assert donor.offset==4
    finally:
        donor.release()
        manager.close()


def test_cold_capacity_cancellation_and_full_state_boundary(native,tmp_path):
    manager=CacheManager(native)
    fp=manager.fingerprint(namespace='principal')
    donor=snapshot()
    store=ColdPrefixStore(tmp_path/'ssd',budget_bytes=1000000,staging_bytes=10)
    try:
        specs=(LayerStateSpec(0,'full_kv',1,2,2,'float16'),)
        assert not store.put(fp,(1,2,3,4),donor,specs)
        assert not store.put(fp,(1,2,3,4),donor,specs,cancelled=lambda:True)
        assert store.stats()['entries']==0
        assert not store.put(replace(fp,multimodal_identity='image'),(1,2,3,4),donor,specs)
    finally:
        donor.release()
        manager.close()


def test_opt_in_manager_restores_from_disk_after_ram_clear(native,tmp_path):
    manager=CacheManager(native,CacheConfig(ssd_cache_directory=str(tmp_path/'ssd'),
        ssd_cache_budget_bytes=1000000,ssd_staging_budget_bytes=1000000))
    fp=manager.fingerprint(namespace='alice')
    donor=manager.create_handle('donor')
    borrower=manager.create_handle('borrower')
    try:
        for cache in donor.bundle.entries:
            cache.update(np.ones((1,1,4,2),np.float16),np.ones((1,1,4,2),np.float16))
        assert manager.publish_prefix(donor,fp,(1,2,3,4))
        manager.clear_prefixes()
        assert manager.preview_prefix(fp,(1,2,3,5))==3
        assert manager.adopt_prefix(borrower,fp,(1,2,3,5))==3
        assert borrower.metadata['prefix_storage_tier']=='ssd'
        assert donor.bundle.offset==4 and borrower.bundle.offset==3
        assert manager.stats().reserved_bytes==0
        manager.release(donor)
        assert borrower.bundle.offset==3
    finally:
        manager.close()
    assert manager.stats().active_handles==0


def test_quota_eviction_and_no_unrelated_directory_adoption(native,tmp_path):
    store=ColdPrefixStore(tmp_path/'ssd',budget_bytes=1000000,max_entries=1,staging_bytes=1000000)
    manager=CacheManager(native)
    donor=snapshot()
    fp=manager.fingerprint(namespace='alice')
    specs=(LayerStateSpec(0,'full_kv',1,2,2,'float16'),)
    try:
        assert store.put(fp,(1,2,3,4),donor,specs)
        assert store.put(fp,(5,6,7,8),donor,specs)
        assert store.stats()['entries']==1
        assert store.preview(fp,(1,2,3,4))==0
        assert len(list(store.root.glob('*/*.safetensors')))==1
        unrelated=tmp_path/'assets'
        unrelated.mkdir()
        (unrelated/'weights.txt').write_text('keep')
        with pytest.raises(ValueError,match='dedicated'):
            ColdPrefixStore(unrelated,budget_bytes=1000).preview(fp,(1,2))
        assert (unrelated/'weights.txt').read_text()=='keep'
    finally:
        donor.release()
        manager.close()
