"""Bounded, request-local model timing with explicitly separated time windows."""

from __future__ import annotations

from collections import deque

from mlx_one.inference_baseline import percentile


class RequestTiming:
    def __init__(self, started, *, trace=False, limit=4096):
        self.started = started
        self.first = self.last = None
        self.count = 0
        self.tokens = deque(maxlen=limit) if trace else None
        self.token_ids = deque(maxlen=limit) if trace else None
        self.gaps = deque(maxlen=limit)
        self.batches = deque(maxlen=limit) if trace else None
        self.dtypes = set()
        self.forward_ms = {"prefill": 0.0, "decode": 0.0}
        self.prefill_tokens = self.forwards = 0
        self.tokenization_ms = 0.0

    def observe(self, event):
        kind = event["kind"]
        if kind == "token":
            at = event["at"]
            if self.last is not None:
                if at < self.last:
                    raise ValueError("model token timestamps must be monotonic")
                self.gaps.append((at - self.last) * 1000)
            self.first = at if self.first is None else self.first
            self.last = at
            self.count += 1
            if self.tokens is not None:
                self.tokens.append((at - self.started) * 1000)
                self.token_ids.append(event.get("token_id"))
        elif kind == "forward":
            self.forwards += 1
            self.forward_ms[event["phase"]] += event["duration_ms"]
            self.prefill_tokens += event["input_tokens"] if event["phase"] == "prefill" else 0
            self.dtypes.add(event["logits_dtype"])
            if self.batches is not None:
                self.batches.append(event["batch_size"])
        elif kind == "tokenization":
            self.tokenization_ms += event["duration_ms"]

    def summary(self):
        interval = (self.last - self.first) if self.count > 1 else None
        return {"model_token_count": self.count,
                "time_to_first_token_ms": ((self.first - self.started) * 1000
                                            if self.first is not None else None),
                "decode_tokens_per_second": ((self.count - 1) / interval
                                              if interval and interval > 0 else None),
                "inter_token_latency_ms": (interval * 1000 / (self.count - 1)
                                            if interval is not None else None),
                "model_token_gap_p50_ms": percentile(self.gaps, .5),
                "model_token_gap_p95_ms": percentile(self.gaps, .95),
                "timing_samples_truncated": self.count - 1 > self.gaps.maxlen}


METRIC_DEFINITIONS = {
    "tokens_per_second": "output tokens / whole engine request wall time",
    "decode_tokens_per_second": "output intervals / first-to-last model output-token interval",
    "inter_token_latency_ms": "mean model output-token gap; excludes prefill and first token",
    "time_to_first_token_ms": "engine execution start to first accepted model output token",
    "queue_ms": "enqueue to execution admission",
    "prefill_ms": "sum of synchronized model prefill forwards",
    "model_token_gap_percentiles": "bounded last 4096 gaps; truncation is explicit",
    "sse_delivery_timing": "client wire observations, separate from model-token timing",
}
