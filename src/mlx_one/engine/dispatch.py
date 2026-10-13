"""Runner selection shared by Python and server actors."""

from __future__ import annotations

import time
from contextlib import ExitStack

from mlx_one.engine.batching import BatchedDenseRunner, request_cancelled
from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.contracts import RunnerOutput


class ReferenceRunner:
    """Explicit auto fallback: independent B=1 forwards, never claimed as batching."""

    def execute(self, requests):
        import mlx.core as mx

        logits = []
        started = time.perf_counter()
        for request in requests:
            with ExitStack() as stack:
                runtime = request.runtime
                if runtime is not None and runtime.pressure is not None:
                    from mlx_one.engine.pressure import execution_forecast

                    amount = execution_forecast(
                        runtime, request.bundle.model.config, batch=1,
                        chunk=len(request.token_ids), history=request.cache[0].offset,
                        explicit_bytes=CacheBundle(request.cache).allocated_nbytes,
                        phase=request.phase)["bytes"]
                    stack.enter_context(runtime.transient_reservation(amount))
                draft = CacheBundle(request.cache).clone()
                try:
                    output = request.bundle.model(mx.array([request.token_ids]),
                                                  cache=draft.entries)
                    mx.eval(output.logits)
                    compact = mx.zeros((1, 1, output.logits.shape[-1]), dtype=output.logits.dtype)
                    compact[...] = output.logits[:, -1:]
                    mx.eval(compact)
                    if not request_cancelled(request):
                        for original, replacement in zip(request.cache, draft.entries, strict=True):
                            original.__dict__.update(replacement.__dict__)
                        if request.handle is not None:
                            request.handle.computed_tokens = request.cache[0].offset
                            runtime.reconcile(request.handle)
                    logits.append(compact)
                finally:
                    draft.release()
        self.last_trace = {"phase": requests[0].phase, "batch_size": 1,
                           "model_forward_count": len(requests),
                           "request_ids": [r.request_id for r in requests],
                           "real_tokens": sum(len(r.token_ids) for r in requests),
                           "padded_tokens": sum(len(r.token_ids) for r in requests),
                           "padding_efficiency": 1.0,
                           "history_lengths": [], "transient_reserved_bytes": 0,
                           "duration_ms": (time.perf_counter() - started) * 1000}
        return RunnerOutput(tuple(r.request_id for r in requests), tuple(logits),
                            tuple(len(r.token_ids) for r in requests))


class DispatchRunner:
    def __init__(self):
        from mlx_one.engine.hybrid import HybridRunner
        from mlx_one.engine.paged import PagedRunner

        self.runners = {"paged": PagedRunner(), "hybrid": HybridRunner(),
                        "dense": BatchedDenseRunner(), "reference": ReferenceRunner()}

    def execute(self, requests):
        def path(r):
            if r.runtime is None:
                return "dense"
            if getattr(r.runtime, "mode", None) == "reference_dense":
                return "reference"
            return r.runtime.config.backend
        paths = {path(r) for r in requests}
        if len(paths) != 1:
            raise ValueError("microbatch contains incompatible execution paths")
        runner = self.runners[next(iter(paths))]
        result = runner.execute(requests)
        self.last_trace = runner.last_trace
        return result
