"""Explicit hybrid state boundaries and safe, versioned tensor checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from mlx_one.core.cache import ConvCache, KVCache
from mlx_one.engine.cache import CacheBundle, array_nbytes


@dataclass(frozen=True)
class ComponentCapabilities:
    can_batch: bool = False
    can_page: bool = False
    can_snapshot: bool = False
    can_quantize: bool = False
    can_share_prefix: bool = False
    can_trim: bool = False
    transition_math: bool = False


@runtime_checkable
class StateComponent(Protocol):
    kind: str

    @property
    def committed_position(self) -> int: ...
    def capabilities(self) -> ComponentCapabilities: ...
    def validate(self, spec) -> None: ...
    def clone(self): ...
    def snapshot(self): ...
    def dispose(self) -> None: ...


class ComponentContract:
    @property
    def committed_position(self):
        return self.offset

    def capabilities(self):
        implemented = self.kind in {"full_kv", "sliding_kv", "conv"}
        return ComponentCapabilities(can_batch=implemented, can_snapshot=True,
            can_share_prefix=implemented, can_trim=self.kind == "full_kv",
            transition_math=implemented)

    def dispose(self):
        self.reset()

    def validate(self, spec):
        if (self.kind != spec.kind or not isinstance(self.offset, int)
                or isinstance(self.offset, bool) or self.offset < 0):
            raise ValueError("component kind or committed position mismatch")
        if self.kind in {"full_kv", "sliding_kv"}:
            length = min(self.offset, getattr(self, "max_size", self.offset))
            if self.kind == "sliding_kv" and self.max_size != spec.window_tokens:
                raise ValueError("component sliding-window layout mismatch")
            for value in (self.keys, self.values):
                if value is None:
                    if self.offset:
                        raise ValueError("component is missing committed KV")
                    continue
                dtype = str(value.dtype).rsplit(".", 1)[-1]
                if (value.shape[:2] != (1, spec.num_kv_heads)
                        or value.shape[2] < length or value.shape[3] != spec.key_head_dim
                        or dtype != self.storage_dtype or spec.dtype and dtype != spec.dtype):
                    raise ValueError("component attention layout or dtype mismatch")
        elif self.values is not None:
            if spec.state_shapes and tuple(self.values.shape) != spec.state_shapes[0]:
                raise ValueError("component carry shape mismatch")
            if spec.dtype and str(self.values.dtype).rsplit(".", 1)[-1] != spec.dtype:
                raise ValueError("component carry dtype mismatch")
        elif self.offset:
            raise ValueError("component is missing committed carry")


class FullKVComponent(ComponentContract, KVCache):
    kind = "full_kv"

    def __init__(self, dtype="float16"):
        super().__init__()
        self.storage_dtype = dtype

    def update(self, keys, values):
        import mlx.core as mx

        dtype = getattr(mx, self.storage_dtype)
        return super().update(keys.astype(dtype), values.astype(dtype))

    def clone(self):
        result = super().clone()
        result.__class__ = FullKVComponent
        result.storage_dtype = self.storage_dtype
        return result

    snapshot = clone


class SlidingKVComponent(ComponentContract, KVCache):
    kind = "sliding_kv"

    def __init__(self, window, dtype="float16"):
        super().__init__()
        if window < 1:
            raise ValueError("sliding window must be positive")
        self.max_size = window
        self.storage_dtype = dtype

    @property
    def window_start(self):
        return max(self.offset - self.max_size, 0)

    def update(self, keys, values):
        import mlx.core as mx

        if keys.shape != values.shape:
            raise ValueError("sliding K/V shapes differ")
        keys, values = keys.astype(getattr(mx, self.storage_dtype)), values.astype(
            getattr(mx, self.storage_dtype))
        old = self.offset
        complete = (keys, values) if self.keys is None else (
            mx.concatenate((self.keys, keys), axis=2),
            mx.concatenate((self.values, values), axis=2))
        self.keys, self.values = tuple(v[:, :, -self.max_size:, :] for v in complete)
        self._offset = old + keys.shape[2]
        return complete

    def clone(self):
        result = SlidingKVComponent(self.max_size, self.storage_dtype)
        result._offset = self.offset
        if self.keys is not None:
            import mlx.core as mx

            result.keys = mx.zeros(self.keys.shape, dtype=self.keys.dtype)
            result.values = mx.zeros(self.values.shape, dtype=self.values.dtype)
            result.keys[...] = self.keys
            result.values[...] = self.values
        return result

    snapshot = clone

    def trim(self, count):
        if count:
            raise ValueError("sliding state requires an exact checkpoint boundary")
        return True


class ConvCarryComponent(ComponentContract, ConvCache):
    kind = "conv"

    def clone(self):
        import mlx.core as mx

        result = ConvCarryComponent(self.kernel_size)
        result._offset = self.offset
        if self.values is not None:
            result.values = mx.zeros(self.values.shape, dtype=self.values.dtype)
            result.values[...] = self.values
        return result

    snapshot = clone


@dataclass
class RecurrentComponent(ComponentContract):
    """Lifecycle contract only; no native SSM/DeltaNet math is registered."""

    values: object = None
    offset: int = 0
    kind: str = "recurrent"

    @property
    def nbytes(self):
        return array_nbytes(self.values)

    def clone(self):
        import mlx.core as mx

        copied = None
        if self.values is not None:
            copied = mx.zeros(self.values.shape, dtype=self.values.dtype)
            copied[...] = self.values
        return RecurrentComponent(copied, self.offset)

    snapshot = clone

    def reset(self):
        self.values, self.offset = None, 0

    release = reset

    def update(self, *_):
        raise NotImplementedError("native recurrent transition math is not implemented")


def hybrid_components(bundle, key_type="f16"):
    config = bundle.model.config
    dtype = "bfloat16" if key_type == "bf16" else "float16"
    result = []
    for kind in config.layer_types:
        if kind == "conv":
            result.append(ConvCarryComponent(config.conv_L_cache))
        elif kind == "sliding_attention":
            result.append(SlidingKVComponent(config.sliding_window, dtype))
        elif kind == "full_attention":
            result.append(FullKVComponent(dtype))
        else:
            raise ValueError("unsupported hybrid component")
    return tuple(result)


class HybridPrefixIndex:
    """Bounded exact-boundary snapshots; carries can never be trimmed backwards."""

    def __init__(self, budget, page_tokens):
        self.budget, self.page_tokens = budget, page_tokens
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = self.tokens_reused = 0

    def put(self, fingerprint, tokens, bundle):
        if (not tokens or len(tokens) % self.page_tokens
                or any(c.offset != len(tokens) for c in bundle.entries)):
            raise ValueError("hybrid snapshot must contain one complete committed boundary")
        clone = bundle.clone()
        try:
            import mlx.core as mx

            arrays = [v for c in clone.entries
                      for v in (getattr(c, "keys", None), getattr(c, "values", None))
                      if v is not None]
            mx.eval(*arrays)
        except BaseException:
            clone.release()
            raise
        size = clone.allocated_nbytes + len(tokens) * 8 + 256
        if size > self.budget:
            clone.release()
            return False
        key = (fingerprint, tuple(tokens))
        if key in self.entries:
            self.entries.pop(key)[0].release()
        self.entries[key] = (clone, size)
        while sum(v[1] for v in self.entries.values()) > self.budget:
            self.entries.popitem(last=False)[1][0].release()
            self.evictions += 1
        return True

    def preview(self, fingerprint, tokens):
        return max((len(ids) for fp, ids in self.entries
                    if fp == fingerprint and len(ids) < len(tokens)
                    and tokens[:len(ids)] == ids), default=0)

    def lookup(self, fingerprint, tokens):
        length = self.preview(fingerprint, tokens)
        if not length:
            self.misses += 1
            return None
        key = (fingerprint, tokens[:length])
        clone = self.entries[key][0].clone()
        self.entries.move_to_end(key)
        self.hits += 1
        self.tokens_reused += length
        return length, clone

    def stats(self):
        return {"bytes": sum(v[0].allocated_nbytes for v in self.entries.values()),
                "metadata_bytes": sum(v[1] - v[0].allocated_nbytes
                                      for v in self.entries.values()),
                "entries": len(self.entries), "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions, "tokens_reused": self.tokens_reused}

    def clear(self):
        for bundle, _ in self.entries.values():
            bundle.release()
        self.entries.clear()


def _snapshot_contents(bundle):
    """Local JSON+NPZ, SHA256 checked. BF16 is losslessly encoded as FP32."""
    import numpy as np

    if len({cache.offset for cache in bundle.entries}) > 1:
        raise ValueError("snapshot components are not at one committed boundary")
    tensors, layers = {}, []
    for index, cache in enumerate(bundle.entries):
        layer = {"kind": getattr(cache, "kind", "unknown"), "offset": cache.offset,
                 "window": getattr(cache, "max_size", None),
                 "kernel_size": getattr(cache, "kernel_size", None),
                 "window_start": getattr(cache, "window_start", None),
                 "capabilities": asdict(cache.capabilities()), "tensors": {}}
        for name in ("keys", "values"):
            value = getattr(cache, name, None)
            if value is not None and layer["kind"] == "full_kv":
                value = value[:, :, :cache.offset, :]
            if value is not None:
                dtype = str(value.dtype).rsplit(".", 1)[-1]
                if dtype == "bfloat16":
                    import mlx.core as mx

                    value = value.astype(mx.float32)
                key = f"layer_{index}_{name}"
                tensors[key] = np.array(value)
                layer["tensors"][name] = {"key": key, "shape": list(value.shape), "dtype": dtype}
        layers.append(layer)
    return tensors, layers


@contextmanager
def _snapshot_lock(root, *, exclusive):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".state.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def save_snapshot(directory, bundle, *, identity):
    """Immutable payload generation; atomic manifest is the commit point."""
    import numpy as np

    root = Path(directory)
    with _snapshot_lock(root, exclusive=True):
        tensors, layers = _snapshot_contents(bundle)
        payload = root / f"tensors-{uuid.uuid4().hex}.npz"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(handle, **tensors)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, payload)
            manifest = {"schema_version": "hybrid-state-v2", "identity": identity,
                        "layers": layers, "payload": payload.name,
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
            with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, root / "state.json")
            directory_fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            for old in root.glob("tensors-*.npz"):
                if old != payload:
                    old.unlink()
            return manifest
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def load_snapshot(directory, *, identity, expected_specs):
    with _snapshot_lock(Path(directory), exclusive=False):
        return _load_snapshot(directory, identity=identity, expected_specs=expected_specs)


def _load_snapshot(directory, *, identity, expected_specs):
    import mlx.core as mx
    import numpy as np

    root = Path(directory)
    manifest = json.loads((root / "state.json").read_text())
    name = manifest.get("payload", "tensors.npz")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("invalid snapshot payload path")
    payload = root / name
    current = manifest.get("schema_version") == "hybrid-state-v2"
    if (manifest.get("schema_version") not in {"hybrid-state-v1", "hybrid-state-v2"}
            or manifest.get("identity") != identity
            or manifest.get("sha256") != hashlib.sha256(payload.read_bytes()).hexdigest()
            or len(manifest["layers"]) != len(expected_specs)):
        raise ValueError("hybrid snapshot identity, checksum or schema mismatch")
    staged = []
    try:
        with np.load(payload, allow_pickle=False) as tensors:
            for layer, spec in zip(manifest["layers"], expected_specs, strict=True):
                if layer["kind"] != spec.kind or layer["offset"] < 0:
                    raise ValueError("hybrid snapshot layer mismatch")
                if spec.kind == "conv":
                    if (spec.state_shapes and layer["kernel_size"] != spec.state_shapes[0][1] + 1):
                        raise ValueError("convolution kernel mismatch")
                    cache = ConvCarryComponent(layer["kernel_size"])
                elif spec.kind == "sliding_kv":
                    if layer["window"] != spec.window_tokens:
                        raise ValueError("sliding window mismatch")
                    cache = SlidingKVComponent(layer["window"])
                elif spec.kind == "full_kv":
                    cache = FullKVComponent()
                elif spec.kind == "recurrent":
                    cache = RecurrentComponent()
                else:
                    raise ValueError("unsupported checkpoint state")
                if current and (layer.get("capabilities") != asdict(cache.capabilities())
                        or layer.get("window_start") != (
                            max(layer["offset"] - spec.window_tokens, 0)
                            if spec.kind == "sliding_kv" else None)):
                    raise ValueError("snapshot component capabilities or window position mismatch")
                staged.append(cache)
                required = ({"keys", "values"} if spec.kind in {"full_kv", "sliding_kv"}
                            else {"values"})
                if layer["offset"] and set(layer["tensors"]) != required:
                    raise ValueError("hybrid snapshot is missing required state tensors")
                for name, meta in layer["tensors"].items():
                    array = tensors[meta["key"]]
                    if (name not in {"keys", "values"} or list(array.shape) != meta["shape"]
                            or meta["dtype"] not in {"float16", "float32", "bfloat16"}):
                        raise ValueError("hybrid snapshot tensor mismatch")
                    storage_dtype = "float32" if meta["dtype"] == "bfloat16" else meta["dtype"]
                    if str(array.dtype) != storage_dtype:
                        raise ValueError("hybrid snapshot storage dtype mismatch")
                    if spec.kind in {"full_kv", "sliding_kv"}:
                        length = (min(layer["offset"], spec.window_tokens)
                                  if spec.kind == "sliding_kv" else layer["offset"])
                        expected = (1, spec.num_kv_heads, length, spec.key_head_dim)
                        if (array.shape != expected or meta["dtype"] not in {"float16", "bfloat16"}
                                or spec.dtype and meta["dtype"] != spec.dtype):
                            raise ValueError("hybrid snapshot attention shape mismatch")
                    elif spec.state_shapes and array.shape != spec.state_shapes[0]:
                        raise ValueError("hybrid snapshot carry shape mismatch")
                    setattr(cache, name, mx.array(array).astype(getattr(mx, meta["dtype"])))
                    if spec.kind in {"full_kv", "sliding_kv"}:
                        cache.storage_dtype = meta["dtype"]
                if isinstance(cache, RecurrentComponent):
                    cache.offset = layer["offset"]
                else:
                    cache._offset = layer["offset"]
            for cache, spec in zip(staged, expected_specs, strict=True):
                cache.validate(spec)
            if len({c.offset for c in staged}) != 1:
                raise ValueError("hybrid snapshot component boundaries differ")
            mx.eval(*(v for c in staged for v in (getattr(c, "keys", None),
                                                  getattr(c, "values", None)) if v is not None))
        return CacheBundle(tuple(staged))
    except BaseException:
        for cache in staged:
            cache.reset()
        raise
