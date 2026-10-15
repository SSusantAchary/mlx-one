"""Opt-in byte-bounded SSD prefix tier with immutable safetensors generations.

The existing RAM indexes remain owners of live GPU state. This tier holds no GPU
references; restoration is staged and charged before atomic owner adoption.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.components import (
    FullKVComponent,
    _snapshot_lock,
    load_snapshot,
    save_snapshot,
)

COLD_PREFIX_VERSION='native-cold-prefix-v1'


def _digest(value):
    encoded=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path,data):
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as file:
            temporary=Path(file.name)
            file.write(json.dumps(data,sort_keys=True))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def canonical_snapshot(bundle,specs):
    """Explicit cold-export copies; no production paged history API is called."""
    import mlx.core as mx

    staged=[]
    try:
        if len(bundle.entries)!=len(specs) or len({c.offset for c in bundle.entries})!=1:
            raise ValueError('cold snapshot topology or committed boundary mismatch')
        for cache,spec in zip(bundle.entries,specs,strict=True):
            if spec.kind=='full_kv':
                component=FullKVComponent(spec.dtype or 'float16')
                staged.append(component)
                if hasattr(cache,'_blocks'):
                    for reference in cache._blocks:
                        page=reference.block
                        keys,values=page.state
                        component.update(keys[:,:,:page.token_count],values[:,:,:page.token_count])
                elif hasattr(cache,'_keys'):
                    if cache.key_bits is not None or cache.value_bits is not None:
                        raise ValueError('quantized KV persistence is unavailable')
                    if cache.offset:
                        component.update(cache._keys[:,:,:cache.offset],
                                         cache._values[:,:,:cache.offset])
                elif cache.offset:
                    component.update(cache.keys[:,:,:cache.offset],cache.values[:,:,:cache.offset])
            elif spec.kind in {'conv','sliding_kv','recurrent'}:
                if not callable(getattr(cache,'validate',None)):
                    raise ValueError('cold hybrid state requires complete typed components')
                component=cache.clone()
                staged.append(component)
            else:
                raise ValueError('cold state capability unavailable')
            component.validate(spec)
        from mlx_one.engine.transactions import component_tensors

        mx.eval(*(t for c in staged for t in component_tensors(c)))
        return CacheBundle(tuple(staged))
    except BaseException:
        CacheBundle(tuple(staged)).release()
        raise


class ColdPrefixStore:
    def __init__(self,directory,*,budget_bytes,max_entries=128,staging_bytes=256*1024**2):
        if min(budget_bytes,max_entries,staging_bytes)<1:
            raise ValueError('cold prefix tier requires positive byte and entry limits')
        self.root=Path(directory).expanduser().resolve()
        self.budget=budget_bytes
        self.max_entries=max_entries
        self.staging=staging_bytes
        self.hits=self.misses=self.rejections=0

    def _index(self):
        path=self.root/'index.json'
        if not path.exists():
            if any(p.name!=".state.lock" for p in self.root.iterdir()):
                raise ValueError("SSD cache directory must be dedicated to prefix storage")
            return []
        if path.stat().st_size>16*1024**2:
            raise ValueError('cold index metadata exceeds limit')
        data=json.loads(path.read_text())
        if data.get('schema')!=COLD_PREFIX_VERSION or data.get('sha256')!=_digest(data['entries']):
            raise ValueError('cold index checksum/schema mismatch')
        records=data['entries']
        if not isinstance(records,list) or len(records)>self.max_entries:
            raise ValueError('cold index entry limit mismatch')
        for r in records:
            if (not isinstance(r.get('generation'),str) or len(r['generation'])!=32
                    or any(c not in '0123456789abcdef' for c in r['generation'])
                    or not isinstance(r.get('bytes'),int) or r['bytes']<0
                    or not isinstance(r.get('tokens'),list)
                    or any(type(t) is not int or t<0 for t in r['tokens'])):
                raise ValueError('invalid cold prefix record')
        return records

    @staticmethod
    def _match(record,identity,tokens,minimum):
        if record['identity']!=identity:
            return 0
        count=0
        for a,b in zip(record['tokens'],tokens[:-1],strict=False):
            if a!=b:
                break
            count+=1
        if record['complete_state'] and count!=len(record['tokens']):
            return 0
        page=record['page_tokens']
        if page:
            count=count//page*page
        return count if count>=minimum else 0

    def preview(self,fingerprint,tokens,*,minimum=1):
        with _snapshot_lock(self.root,exclusive=False):
            return max((self._match(r,fingerprint.digest,tokens,minimum)
                        for r in self._index()),default=0)

    def put(self,fingerprint,tokens,bundle,specs,*,page_tokens=0,cancelled=lambda:False):
        if (not tokens or fingerprint.multimodal_identity is not None
                or bundle.offset!=len(tokens) or cancelled()):
            return False
        estimate=bundle.allocated_nbytes*4+len(tokens)*16+65536
        if estimate>self.staging:
            self.rejections+=1
            return False
        canonical=None
        generation=uuid.uuid4().hex
        directory=self.root/generation
        published=False
        with _snapshot_lock(self.root,exclusive=True):
            records=self._index()
            try:
                if shutil.disk_usage(self.root).free<estimate:
                    self.rejections+=1
                    return False
                canonical=canonical_snapshot(bundle,specs)
                save_snapshot(directory,canonical,identity={'fingerprint':fingerprint.digest,
                    'prefix_version':COLD_PREFIX_VERSION},tensor_format='safetensors')
                size=sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
                if size>self.budget or cancelled():
                    return False
                record={'identity':fingerprint.digest,'generation':generation,'tokens':list(tokens),
                    'bytes':size,'complete_state':any(s.kind!='full_kv' for s in specs),
                    'page_tokens':page_tokens,'state_sha256':hashlib.sha256(
                        (directory/'state.json').read_bytes()).hexdigest()}
                retained=[r for r in records if (r['identity'],r['tokens'])!=(record['identity'],
                                                                                          record['tokens'])]
                retained.append(record)
                while (len(retained)>self.max_entries
                       or sum(r['bytes'] for r in retained)>self.budget):
                    retained.pop(0)
                _atomic_json(self.root/'index.json',{'schema':COLD_PREFIX_VERSION,
                    'entries':retained,'sha256':_digest(retained)})
                published=True
                live={r['generation'] for r in retained}
                # Only exclusively locked writers collect obsolete/orphan generations.
                for path in self.root.iterdir():
                    if path.is_dir() and len(path.name)==32 and path.name not in live:
                        shutil.rmtree(path)
                return True
            finally:
                if canonical is not None:
                    canonical.release()
                if not published:
                    shutil.rmtree(directory,ignore_errors=True)

    def lookup(self,fingerprint,tokens,specs,*,minimum=1):
        with _snapshot_lock(self.root,exclusive=False):
            candidates=[(self._match(r,fingerprint.digest,tokens,minimum),r) for r in self._index()]
            count,record=max(candidates,key=lambda pair:pair[0],default=(0,None))
            if not count:
                self.misses+=1
                return None
            directory=self.root/record['generation']
            if directory.is_symlink() or any(p.is_symlink() for p in directory.iterdir()):
                raise ValueError('cold generation cannot contain symbolic links')
            if record['bytes']*4>self.staging or record['bytes']>self.budget:
                raise ValueError('cold restore exceeds staging/disk policy')
            if (hashlib.sha256((directory/'state.json').read_bytes()).hexdigest()
                    !=record['state_sha256']):
                raise ValueError('cold state manifest checksum mismatch')
            # Reject unexpected files/oversized payload before loading any tensor.
            if sum(p.stat().st_size for p in directory.iterdir() if p.is_file())!=record['bytes']:
                raise ValueError('cold generation byte accounting mismatch')
            restored=load_snapshot(directory,identity={'fingerprint':fingerprint.digest,
                'prefix_version':COLD_PREFIX_VERSION},expected_specs=specs)
            try:
                if restored.offset!=len(record['tokens']):
                    raise ValueError('cold committed token identity mismatch')
                if restored.offset!=count and not restored.trim(restored.offset-count):
                    raise ValueError('cold snapshot cannot trim to matching boundary')
                self.hits+=1
                return count,restored
            except BaseException:
                restored.release()
                raise

    def stats(self):
        with _snapshot_lock(self.root,exclusive=False):
            records=self._index()
            return {'entries':len(records),'disk_bytes':sum(r['bytes'] for r in records),
                    'budget_bytes':self.budget,'staging_budget_bytes':self.staging,
                    'hits':self.hits,'misses':self.misses,'rejections':self.rejections}

    def clear(self):
        with _snapshot_lock(self.root,exclusive=True):
            _atomic_json(self.root/'index.json',{'schema':COLD_PREFIX_VERSION,
                'entries':[],'sha256':_digest([])})
            for path in self.root.iterdir():
                if path.is_dir() and len(path.name)==32:
                    shutil.rmtree(path)
