"""Byte-bounded immutable feature ownership; live leases prevent LRU eviction."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

from mlx_one.engine.cache_errors import CacheCapacityError


@dataclass
class _Feature:
    value: object
    nbytes: int
    leases: int = 0


class FeatureLease:
    def __init__(self,store,key,entry):
        self.store,self.key,self.entry=store,key,entry

    @property
    def value(self):
        if self.entry is None:
            raise RuntimeError('feature lease released')
        return self.entry.value

    def close(self):
        with self.store.lock:
            if self.entry is not None:
                self.entry.leases-=1
                self.entry=None

    def __enter__(self):
        return self.value

    def __exit__(self,*args):
        self.close()


class FeatureStore:
    def __init__(self,budget_bytes=64*1024**2):
        if budget_bytes<1:
            raise ValueError('feature quota must be positive')
        self.budget=budget_bytes
        self.entries=OrderedDict()
        self.lock=threading.RLock()
        self.closed=False

    def acquire(self,namespace,identity,factory,nbytes):
        if not namespace or not identity or nbytes<0:
            raise ValueError('features require namespace, content identity and byte size')
        key=(namespace,identity)
        with self.lock:
            if self.closed:
                raise RuntimeError('feature store closed')
            if key not in self.entries:
                used=sum(e.nbytes for e in self.entries.values())
                for candidate in list(self.entries):
                    if used+nbytes<=self.budget:
                        break
                    entry=self.entries[candidate]
                    if not entry.leases:
                        used-=self.entries.pop(candidate).nbytes
                if used+nbytes>self.budget:
                    raise CacheCapacityError('media features exceed admitted quota')
                self.entries[key]=_Feature(factory(),nbytes)
            entry=self.entries[key]
            if entry.nbytes!=nbytes:
                raise ValueError('feature identity size mismatch')
            entry.leases+=1
            self.entries.move_to_end(key)
            return FeatureLease(self,key,entry)

    def stats(self):
        with self.lock:
            return {'bytes':sum(e.nbytes for e in self.entries.values()),
                    'leases':sum(e.leases for e in self.entries.values()),
                    'entries':len(self.entries)}

    def close(self):
        with self.lock:
            if any(e.leases for e in self.entries.values()):
                raise RuntimeError('cannot dispose live feature leases')
            self.entries.clear()
            self.closed=True
