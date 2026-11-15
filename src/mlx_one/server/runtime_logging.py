"""Bounded terminal lifecycle/timing logs using existing request observations."""

from __future__ import annotations

import itertools
import logging
import math
import os
import sys
import time

logger = logging.getLogger("mlx_one.runtime")
_tasks = itertools.count(1)
_progress_interval = 3.0


class RuntimeFormatter(logging.Formatter):
    def __init__(self, *, color=False, clock=time.perf_counter):
        super().__init__()
        self.clock, self.started, self.color = clock, clock(), color

    def format(self, record):
        elapsed = max(int((self.clock() - self.started) * 1_000_000), 0)
        seconds, fraction = divmod(elapsed, 1_000_000)
        minutes, seconds = divmod(seconds, 60)
        stamp = f"{minutes}.{seconds:02d}.{fraction // 1000:03d}.{fraction % 1000:03d}"
        level = record.levelname[0]
        message = record.getMessage()
        # Keep each event on one terminal line, including untrusted aliases.
        message = "".join(c if c.isprintable() else repr(c)[1:-1] for c in message)
        if self.color:
            code = (
                31
                if record.levelno >= logging.ERROR
                else 33
                if record.levelno >= logging.WARNING
                else 32
            )
            stamp = f"\033[35m{stamp}\033[0m"
            level = f"\033[{code}m{level}\033[0m"
        return f"{stamp} {level} {getattr(record, 'component', 'srv'):4} {message}"


def configure_runtime_logging(level="info", *, interval=3.0, stream=None):
    global _progress_interval
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("log progress interval must be finite and nonnegative")
    _progress_interval = interval
    stream = sys.stderr if stream is None else stream
    for handler in list(logger.handlers):
        if getattr(handler, "_mlx_runtime", False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(stream)
    handler._mlx_runtime = True
    handler.setFormatter(
        RuntimeFormatter(
            color=bool(getattr(stream, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False


def event(action, message, *args, component="srv", level=logging.INFO):
    logger.log(level, action + ": " + message, *args, extra={"component": component})


class RequestLog:
    def __init__(self, slot=0):
        self.enabled = logger.isEnabledFor(logging.INFO)
        self.task, self.slot = next(_tasks), slot
        self.finished = False
        self.first = None
        self.count = self.last_count = 0
        self.last_at = None

    def emit(self, action, message, *args, level=logging.INFO):
        event(
            action,
            f"slot {self.slot} | task {self.task} | " + message,
            *args,
            component="slot",
            level=level,
        )

    def observe(self, observation):
        if not self.enabled or observation["kind"] != "token":
            return
        at = observation["at"]
        self.count += 1
        if self.first is None:
            self.first = self.last_at = at
            self.last_count = self.count
        if not _progress_interval or at - self.last_at < _progress_interval:
            return
        self.emit(
            "progress",
            "n_gen = %d | decode = %.2f tok/s | recent = %.2f tok/s",
            self.count,
            (self.count - 1) / (at - self.first),
            (self.count - self.last_count) / (at - self.last_at),
        )
        self.last_at, self.last_count = at, self.count

    def complete(self, metrics, finish_reason):
        if self.finished:
            return
        self.finished = True
        if not self.enabled:
            return
        prompt, generated = metrics.get("prompt_tokens") or 0, metrics.get("generated_tokens") or 0
        prefill = metrics.get("actual_prefill_tokens")
        if prefill is None:
            prefill = max(prompt - (metrics.get("reused_prompt_tokens") or 0), 0)
        ms = metrics.get("prefill_ms")
        if ms is not None:
            self.emit(
                "print_timing",
                "prompt eval = %.2f ms / %d tokens (%.2f tok/s)",
                ms,
                prefill,
                prefill * 1000 / ms if ms > 0 else 0,
            )
        decode = metrics.get("decode_tokens_per_second")
        if decode is not None:
            self.emit(
                "print_timing",
                "decode = %.2f tok/s | n_gen = %d%s",
                decode,
                generated,
                (
                    f" | forward = {metrics['decode_ms']:.2f} ms"
                    if metrics.get("decode_ms") is not None
                    else ""
                ),
            )
        total = metrics.get("generation_latency_ms")
        if total is not None:
            self.emit(
                "print_timing",
                "total time = %.2f ms / %d tokens | overall = %.2f tok/s",
                total,
                prompt + generated,
                metrics.get("tokens_per_second") or 0,
            )
        for key, label in [
            ("time_to_first_token_ms", "model TTFT"),
            ("server_text_ttft_ms", "visible text TTFT"),
        ]:
            if metrics.get(key) is not None:
                self.emit("print_timing", "%s = %.2f ms", label, metrics[key])
        self.emit(
            "release",
            "finish = %s | n_ctx = %d | prefix reused = %d%s",
            finish_reason,
            metrics.get("context_used") or 0,
            metrics.get("reused_prompt_tokens") or 0,
            (
                f" | MLX active = {metrics['memory_used_bytes'] / 1e9:.2f} GB"
                if metrics.get("memory_used_bytes") is not None
                else ""
            ),
        )

    def stop(self, outcome, error=None):
        if self.finished:
            return
        self.finished = True
        self.emit(
            "release",
            "status = %s | n_gen observed = %d%s",
            outcome,
            self.count,
            f" | error = {type(error).__name__}" if error else "",
            level=logging.WARNING if error else logging.INFO,
        )


def logged_events(iterator, request_log):
    try:
        item = next(iterator)
        send = getattr(iterator, "send", None)
        while True:
            metrics = getattr(item, "metrics", None)
            if metrics is not None:
                request_log.complete(metrics, item.finish_reason)
            # Actor execution yields ForwardRequest objects and receives the
            # scheduler's result through send(). Preserve that protocol while
            # observing only terminal generation metrics.
            response = yield item
            item = send(response) if callable(send) else next(iterator)
    except StopIteration:
        return
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
