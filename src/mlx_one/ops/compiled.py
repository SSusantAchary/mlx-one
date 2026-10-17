"""Bounded experiments for pure operations; never captures mutable model/cache state."""
from __future__ import annotations

from collections import OrderedDict


class CompiledMathCache:
    def __init__(self, limit=16):
        if type(limit) is not int or limit < 1:
            raise ValueError("positive compiled operation cache limit required")
        self.limit = limit
        self._functions = OrderedDict()
        self.evictions = 0

    def rms_norm(self, value, weight, *, eps):
        import mlx.core as mx

        if weight.ndim != 1 or weight.shape[0] != value.shape[-1] or eps <= 0:
            raise ValueError("unsupported pure RMSNorm signature")
        key = ("rms_norm", tuple(value.shape), str(value.dtype), tuple(weight.shape),
               str(weight.dtype), float(eps), str(mx.default_device()))
        if key not in self._functions:
            # Only a scalar epsilon is captured. Arrays remain dynamic arguments.
            self._functions[key] = mx.compile(
                lambda x, w: mx.fast.rms_norm(x, w, eps), shapeless=False)
            if len(self._functions) > self.limit:
                self._functions.popitem(last=False)
                self.evictions += 1
        self._functions.move_to_end(key)
        return self._functions[key](value, weight)

    def clear(self):
        self._functions.clear()

    def stats(self):
        return {"entries": len(self._functions), "limit": self.limit,
                "evictions": self.evictions, "scope": "python-pure-operation-cache",
                "native_graph_memory_reconciled": False, "qualified": False}
