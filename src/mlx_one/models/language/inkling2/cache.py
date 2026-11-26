"""One request-owned KV and convolution state per executed loop visit."""

from mlx_one.core.cache import KVCache, _tree_nbytes
from mlx_one.core.sliding_cache import SlidingKVCache


class InklingKVCache:
    supports_batching = False
    supports_quantization = False
    supports_prefix_sharing = False

    def __init__(self, window=0):
        self.max_size = window
        self.kv = SlidingKVCache(window) if window else KVCache()
        self.conv = [None] * 4
        self.draft = None
        self.replay = None

    @property
    def offset(self):
        return self.kv.offset

    @property
    def nbytes(self):
        return self.kv.nbytes + _tree_nbytes(self.conv) + (self.draft.nbytes if self.draft else 0)

    @property
    def allocated_nbytes(self):
        draft = self.draft.allocated_nbytes if self.draft else 0
        workspace = self.replay.workspace_bytes if self.replay else 0
        return self.kv.allocated_nbytes + _tree_nbytes(self.conv) + draft + workspace

    def reset(self):
        self.kv.reset()
        self.conv = [None] * 4
        self.draft = None
        self.replay = None

    release = reset

    def tensors(self):
        return [x for x in (self.kv.keys, self.kv.values, *self.conv) if x is not None]

    def boundary(self):
        """Global KV has an append-only tail. Only window writes overwrite logical history."""
        window = (self.kv.keys, self.kv.values) if self.max_size else None
        return self.offset, window, list(self.conv)

    def restore(self, boundary):
        offset, window, conv = boundary
        if window is not None:
            self.kv.keys, self.kv.values = window
        elif not offset:
            self.kv.reset()
        self.kv._offset = offset
        self.conv = list(conv)
