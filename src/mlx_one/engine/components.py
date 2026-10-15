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
            for value, dimension in zip((self.keys, self.values),
                    (spec.key_head_dim, spec.value_head_dim), strict=True):
                if value is None:
                    if self.offset:
                        raise ValueError("component is missing committed KV")
                    continue
                dtype = str(value.dtype).rsplit(".", 1)[-1]
                if (value.shape[:2] != (1, spec.num_kv_heads)
                        or value.shape[2] < length or value.shape[3] != dimension
                        or dtype != self.storage_dtype or spec.dtype and dtype != spec.dtype):
                    raise ValueError("component attention layout or dtype mismatch")
        elif (self.kind == "recurrent" and spec.checkpoint_semantics is not None
              and getattr(self,"checkpoint_semantics",None) != spec.checkpoint_semantics):
            raise ValueError("recurrent transition identity mismatch")
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


def hybrid_components(bundle, key_type="f16", *, pool=None):
    provider = getattr(bundle.model,"make_state_components",None)
    if callable(provider):
        return provider(key_type,pool=pool)
    config = bundle.model.config
    dtype = "bfloat16" if key_type == "bf16" else "float16"
    result = []
    for layer_id, kind in enumerate(config.layer_types):
        if kind == "conv":
            result.append(ConvCarryComponent(config.conv_L_cache))
        elif kind == "sliding_attention":
            result.append(SlidingKVComponent(config.sliding_window, dtype))
        elif kind == "full_attention":
            if pool is None:
                result.append(FullKVComponent(dtype))
            else:
                from mlx_one.engine.paged import PagedKVComponent

                result.append(PagedKVComponent(pool, layer_id=layer_id, dtype=dtype))
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

            from mlx_one.engine.transactions import component_tensors

            arrays = [v for c in clone.entries for v in component_tensors(c)]
            mx.eval(*arrays)
        except BaseException:
            clone.release()
            raise
        size = clone.allocated_nbytes + len(tokens) * 8 + 256
        if size > self.budget:
            clone.release()
            return False
        pinned = []
        try:
            for component in clone.entries:
                for ref in getattr(component, "_blocks", ()):
                    if not ref.block.immutable:
                        raise ValueError("hybrid prefix contains a mutable page")
                    ref.pin()
                    pinned.append(ref)
        except BaseException:
            for ref in pinned:
                ref.unpin()
            clone.release()
            raise
        key = (fingerprint, tuple(tokens))
        if key in self.entries:
            self._release(self.entries.pop(key)[0])
        self.entries[key] = (clone, size)
        while sum(v[1] for v in self.entries.values()) > self.budget:
            self._release(self.entries.popitem(last=False)[1][0])
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

    def _release(self, bundle):
        for component in bundle.entries:
            for ref in getattr(component, "_blocks", ()):
                ref.unpin()
        bundle.release()

    def clear(self):
        for bundle, _ in self.entries.values():
            self._release(bundle)
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
        compound = getattr(cache,"checkpoint_semantics",None) == "qwen35-gated-delta-v1"
        names = ("conv_state","recurrent_state") if compound else ("keys","values")
        if compound:
            layer["compound"] = {"version":cache.checkpoint_semantics,
                "state_shapes":cache.state_shapes,"state_dtypes":cache.state_dtypes}
        paged = hasattr(cache, "page_table")
        if paged:
            # Explicit serialization diagnostic only; generation never calls this export.
            import mlx.core as mx

            layer["storage"] = "paged-fp16-v1"
            layer["page_tokens"] = cache.block_tokens
            exported = {name: mx.concatenate([ref.block.state[i][:, :, :ref.block.token_count]
                for ref in cache._blocks], axis=2) if cache._blocks else None
                for i, name in enumerate(("keys", "values"))}
        else:
            exported = {}
        for name in names:
            value = exported.get(name) if paged else getattr(cache, name, None)
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


def save_snapshot(directory, bundle, *, identity, tensor_format="npz"):
    if tensor_format not in {"npz","safetensors"}:
        raise ValueError("unsupported snapshot tensor format")
    """Immutable payload generation; atomic manifest is the commit point."""
    import numpy as np

    root = Path(directory)
    with _snapshot_lock(root, exclusive=True):
        tensors, layers = _snapshot_contents(bundle)
        payload = root / f"tensors-{uuid.uuid4().hex}.{tensor_format}"
        temporary = None
        published = False
        try:
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
                temporary = Path(handle.name)
                if tensor_format == "npz":
                    np.savez(handle, **tensors)
                else:
                    from safetensors.numpy import save

                    handle.write(save({k:np.ascontiguousarray(v) for k,v in tensors.items()}))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, payload)
            manifest = {"schema_version": ("hybrid-state-v4"
                if any("compound" in layer for layer in layers)
                else "hybrid-state-v3" if any(
                "storage" in layer for layer in layers) else "hybrid-state-v2"),
                        "identity": identity,
                        "layers": layers, "payload": payload.name, "tensor_format": tensor_format,
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
            with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, root / "state.json")
            published = True
            directory_fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            for old in (*root.glob("tensors-*.npz"),*root.glob("tensors-*.safetensors")):
                if old != payload:
                    old.unlink()
            return manifest
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if not published:
                payload.unlink(missing_ok=True)


@contextmanager
def _snapshot_tensors(payload,tensor_format):
    if tensor_format == "npz":
        import numpy as np

        with np.load(payload,allow_pickle=False) as tensors:
            yield tensors
    elif tensor_format == "safetensors":
        from safetensors.numpy import load_file

        yield load_file(payload)
    else:
        raise ValueError("unsupported snapshot tensor format")


def load_snapshot(directory, *, identity, expected_specs, page_pool=None):
    with _snapshot_lock(Path(directory), exclusive=False):
        return _load_snapshot(directory, identity=identity, expected_specs=expected_specs,
                              page_pool=page_pool)


def _load_snapshot(directory, *, identity, expected_specs, page_pool=None):
    import mlx.core as mx
    root = Path(directory)
    manifest = json.loads((root / "state.json").read_text())
    name = manifest.get("payload", "tensors.npz")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("invalid snapshot payload path")
    payload = root / name
    current = manifest.get("schema_version") in {
        "hybrid-state-v2", "hybrid-state-v3", "hybrid-state-v4"}
    supported = {"hybrid-state-v1", "hybrid-state-v2", "hybrid-state-v3", "hybrid-state-v4"}
    if (manifest.get("schema_version") not in supported
            or manifest.get("identity") != identity
            or manifest.get("sha256") != hashlib.sha256(payload.read_bytes()).hexdigest()
            or len(manifest["layers"]) != len(expected_specs)):
        raise ValueError("hybrid snapshot identity, checksum or schema mismatch")
    staged = []
    try:
        with _snapshot_tensors(payload,manifest.get("tensor_format","npz")) as tensors:
            for layer, spec in zip(manifest["layers"], expected_specs, strict=True):
                if layer["kind"] != spec.kind or layer["offset"] < 0:
                    raise ValueError("hybrid snapshot layer mismatch")
                compound = layer.get("compound")
                if spec.checkpoint_semantics == "qwen35-gated-delta-v1" and compound is None:
                    raise ValueError("snapshot lacks complete native recurrent state")
                paged = layer.get("storage") == "paged-fp16-v1"
                restored_arrays = {}
                if layer.get("storage") is not None and not paged:
                    raise ValueError("unsupported hybrid snapshot storage")
                if compound is not None:
                    from mlx_one.engine.compound import COMPOUND_VERSION, CompoundLinearComponent

                    if (manifest["schema_version"] != "hybrid-state-v4"
                            or compound.get("version") != COMPOUND_VERSION
                            or spec.checkpoint_semantics != COMPOUND_VERSION
                            or compound.get("state_shapes") != [list(s) for s in spec.state_shapes]
                            or compound.get("state_dtypes") != list(spec.state_dtypes)):
                        raise ValueError("compound snapshot layout or transition identity mismatch")
                    cache = CompoundLinearComponent(layer["kernel_size"],spec.state_shapes,
                                                    spec.state_dtypes)
                elif paged:
                    from mlx_one.engine.paged import PagedKVComponent

                    if (manifest["schema_version"] not in {"hybrid-state-v3","hybrid-state-v4"}
                            or page_pool is None
                            or spec.kind != "full_kv"
                            or layer["page_tokens"] != page_pool.block_tokens):
                        raise ValueError("paged snapshot requires a matching page arena")
                    cache = PagedKVComponent(page_pool, layer_id=spec.layer_id)
                elif spec.kind == "conv":
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
                required = ({"conv_state","recurrent_state"} if compound is not None else
                            {"keys", "values"} if spec.kind in {"full_kv", "sliding_kv"}
                            else {"values"})
                if layer["offset"] and set(layer["tensors"]) != required:
                    raise ValueError("hybrid snapshot is missing required state tensors")
                for name, meta in layer["tensors"].items():
                    array = tensors[meta["key"]]
                    if (name not in required or list(array.shape) != meta["shape"]
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
                    elif compound is not None:
                        index = ("conv_state","recurrent_state").index(name)
                        if (array.shape != spec.state_shapes[index]
                                or meta["dtype"] != spec.state_dtypes[index]):
                            raise ValueError("compound tensor shape or dtype mismatch")
                    elif spec.state_shapes and array.shape != spec.state_shapes[0]:
                        raise ValueError("hybrid snapshot carry shape mismatch")
                    value = mx.array(array).astype(getattr(mx, meta["dtype"]))
                    if paged:
                        if meta["dtype"] != "float16":
                            raise ValueError("hybrid paged snapshots require FP16")
                        restored_arrays[name] = value
                    else:
                        setattr(cache, name, value)
                    if spec.kind in {"full_kv", "sliding_kv"}:
                        cache.storage_dtype = meta["dtype"]
                if paged:
                    if layer["offset"]:
                        cache.append(restored_arrays["keys"], restored_arrays["values"])
                elif isinstance(cache, RecurrentComponent):
                    cache.offset = layer["offset"]
                else:
                    cache._offset = layer["offset"]
            for cache, spec in zip(staged, expected_specs, strict=True):
                cache.validate(spec)
            if len({c.offset for c in staged}) != 1:
                raise ValueError("hybrid snapshot component boundaries differ")
            from mlx_one.engine.transactions import component_tensors

            mx.eval(*(v for c in staged for v in component_tensors(c)))
        return CacheBundle(tuple(staged))
    except BaseException:
        for cache in staged:
            cache.reset()
        raise


class CrossKVComponent(FullKVComponent):
    """Immutable encoder KV plus a separate committed decoder position."""
    kind = 'cross_kv'

    def __init__(self, encoder_identity, dtype='float32'):
        super().__init__(dtype)
        if not encoder_identity:
            raise ValueError('cross attention requires encoder identity')
        self.encoder_identity, self.encoder_tokens = encoder_identity, 0

    def capabilities(self):
        return ComponentCapabilities(can_snapshot=True)

    def bind(self, keys, values):
        import mlx.core as mx

        if self.keys is not None or keys.shape[:3] != values.shape[:3]:
            raise ValueError('cross KV is immutable and requires aligned encoder shapes')
        owned = []
        for value in (keys, values):
            copy = mx.zeros(value.shape, dtype=getattr(mx, self.storage_dtype))
            copy[...] = value
            owned.append(copy)
        mx.eval(*owned)
        self.keys, self.values = owned
        self.encoder_tokens = keys.shape[2]

    def advance_decoder(self, count):
        if count < 1 or self.keys is None:
            raise ValueError('decoder advance requires bound encoder KV and real tokens')
        self._offset += count

    def update(self, *_):
        raise ValueError('cross KV uses immutable encoder binding, not decoder append')

    def state(self):
        return self.keys, self.values

    def validate(self, spec):
        if spec.kind != self.kind or self.offset < 0:
            raise ValueError('cross KV component or decoder position mismatch')
        for value, width in zip(self.state(),(spec.key_head_dim,spec.value_head_dim),strict=True):
            if value is None:
                if self.encoder_tokens or self.offset:
                    raise ValueError('missing encoder KV')
            elif (value.shape != (1,spec.num_kv_heads,self.encoder_tokens,width)
                  or str(value.dtype).rsplit('.',1)[-1] != self.storage_dtype
                  or spec.dtype and spec.dtype != self.storage_dtype):
                raise ValueError('cross KV encoder layout or dtype mismatch')

    def clone(self):
        result = CrossKVComponent(self.encoder_identity,self.storage_dtype)
        if self.keys is not None:
            result.bind(self.keys,self.values)
        result._offset = self.offset
        return result

    snapshot = clone

    def reset(self):
        super().reset()
        self.encoder_tokens = 0

    release = reset
