"""Explicit hybrid state boundaries and safe, versioned tensor checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from mlx_one.core.cache import ConvCache, KVCache
from mlx_one.engine.cache import CacheBundle, array_nbytes


class FullKVComponent(KVCache):
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


class SlidingKVComponent(KVCache):
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


class ConvCarryComponent(ConvCache):
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
class RecurrentComponent:
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


def save_snapshot(directory, bundle, *, identity):
    """Local JSON+NPZ, SHA256 checked. BF16 is losslessly encoded as FP32."""
    import numpy as np

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    tensors, layers = {}, []
    for index, cache in enumerate(bundle.entries):
        layer = {"kind": getattr(cache, "kind", "unknown"), "offset": cache.offset,
                 "window": getattr(cache, "max_size", None),
                 "kernel_size": getattr(cache, "kernel_size", None), "tensors": {}}
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
    payload = root / "tensors.npz"
    np.savez(payload, **tensors)
    manifest = {"schema_version": "hybrid-state-v1", "identity": identity, "layers": layers,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
    (root / "state.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def load_snapshot(directory, *, identity, expected_specs):
    import mlx.core as mx
    import numpy as np

    root = Path(directory)
    manifest = json.loads((root / "state.json").read_text())
    payload = root / "tensors.npz"
    if (manifest.get("schema_version") != "hybrid-state-v1"
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
                        if array.shape != expected:
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
            if len({c.offset for c in staged}) != 1:
                raise ValueError("hybrid snapshot component boundaries differ")
            mx.eval(*(v for c in staged for v in (getattr(c, "keys", None),
                                                  getattr(c, "values", None)) if v is not None))
        return CacheBundle(tuple(staged))
    except BaseException:
        for cache in staged:
            cache.reset()
        raise
