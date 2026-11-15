import io
import logging
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from mlx_one.cli import main
from mlx_one.server import runtime_logging as logs


@pytest.fixture
def output():
    previous = (
        logs.logger.handlers[:],
        logs.logger.level,
        logs.logger.propagate,
        logs._progress_interval,
    )
    logs.logger.handlers = []
    stream = io.StringIO()
    logs.configure_runtime_logging(stream=stream)
    try:
        yield stream
    finally:
        for handler in logs.logger.handlers:
            handler.close()
        logs.logger.handlers, level, logs.logger.propagate, logs._progress_interval = previous
        logs.logger.setLevel(level)


def test_elapsed_format_and_single_line_output():
    clock = iter([10, 72.123456])
    formatter = logs.RuntimeFormatter(clock=lambda: next(clock))
    record = logging.LogRecord("x", logging.INFO, "", 0, "hello\n%s", ("world\x1b[31m",), None)
    text = formatter.format(record)
    assert text.startswith("1.02.123.456 I srv ")
    assert "\n" not in text and "\x1b" not in text
    assert "\\n" in text and "\\x1b" in text


def test_progress_is_rate_limited_and_slot_scoped(output):
    a, b = logs.RequestLog(0), logs.RequestLog(1)
    assert a.task != b.task
    a.observe({"kind": "token", "at": 10})
    a.observe({"kind": "forward"})
    a.observe({"kind": "token", "at": 11})
    a.observe({"kind": "token", "at": 13})
    a.observe({"kind": "token", "at": 13.5})
    assert output.getvalue().count("progress:") == 1
    assert f"slot 0 | task {a.task}" in output.getvalue()
    assert "n_gen = 3 | decode = 0.67 tok/s" in output.getvalue()
    logs._progress_interval = 0
    a.observe({"kind": "token", "at": 20})
    assert output.getvalue().count("progress:") == 1


def test_summary_uses_actual_prefill_and_distinguishes_ttft(output):
    request = logs.RequestLog()
    request.complete(
        {
            "prompt_tokens": 100,
            "actual_prefill_tokens": 20,
            "generated_tokens": 11,
            "prefill_ms": 50,
            "decode_tokens_per_second": 30,
            "decode_ms": 320,
            "generation_latency_ms": 500,
            "tokens_per_second": 22,
            "time_to_first_token_ms": 80,
            "server_text_ttft_ms": 200,
            "context_used": 111,
            "reused_prompt_tokens": 80,
            "memory_used_bytes": 11e9,
        },
        "stop",
    )
    request.stop("cancelled")
    text = output.getvalue()
    assert "prompt eval = 50.00 ms / 20 tokens (400.00 tok/s)" in text
    assert "decode = 30.00 tok/s" in text and "overall = 22.00 tok/s" in text
    assert "model TTFT = 80.00 ms" in text and "visible text TTFT = 200.00 ms" in text
    assert text.count("release:") == 1
    assert "MLX active = 11.00 GB" in text
    assert "graphs" not in text


def test_unknown_timings_and_memory_are_omitted(output):
    request = logs.RequestLog()
    request.complete({"prompt_tokens": 2, "generated_tokens": 1}, "stop")
    assert "print_timing" not in output.getvalue()
    assert "MLX active" not in output.getvalue()


def test_failure_does_not_log_exception_content_or_duplicate_terminal(output):
    request = logs.RequestLog()
    request.stop("failed", RuntimeError("private prompt and api-key"))
    request.stop("interrupted")
    assert "RuntimeError" in output.getvalue()
    assert "private" not in output.getvalue()
    assert output.getvalue().count("release:") == 1


def test_handler_setup_is_idempotent_and_no_color_is_respected(output, monkeypatch):
    output.isatty = lambda: True
    monkeypatch.setenv("NO_COLOR", "1")
    logs.configure_runtime_logging(stream=output)
    logs.configure_runtime_logging(stream=output)
    logs.event("ready", "ok")
    assert output.getvalue().count("ready:") == 1
    assert "\x1b" not in output.getvalue()


def test_warning_level_avoids_progress_and_summary(output):
    logs.configure_runtime_logging("warning", stream=output)
    request = logs.RequestLog()
    assert not request.enabled
    request.observe({"kind": "token", "at": 1})
    request.complete({"generated_tokens": 1}, "stop")
    assert not output.getvalue()


def test_wrapper_closes_iterator_on_consumer_disconnect(output):
    closed = []

    def stream():
        try:
            yield SimpleNamespace(metrics=None)
        finally:
            closed.append(True)

    wrapped = logs.logged_events(stream(), logs.RequestLog())
    next(wrapped)
    wrapped.close()
    assert closed == [True]


def test_wrapper_preserves_actor_forward_request_responses(output):
    forward, result = object(), object()
    received, closed = [], []

    def stream():
        try:
            received.append((yield forward))
            yield SimpleNamespace(metrics={"generated_tokens": 1}, finish_reason="stop")
        finally:
            closed.append(True)

    wrapped = logs.logged_events(stream(), logs.RequestLog())
    assert next(wrapped) is forward
    assert wrapped.send(result).finish_reason == "stop"
    with pytest.raises(StopIteration):
        next(wrapped)
    assert received == [result] and closed == [True]
    assert output.getvalue().count("release:") == 1


@pytest.mark.parametrize("interval", [-1, float("nan"), float("inf")])
def test_invalid_progress_intervals_rejected(interval):
    with pytest.raises(ValueError):
        logs.configure_runtime_logging(interval=interval)


def test_cli_documents_logging_controls():
    result = CliRunner().invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--log-progress-interval" in result.output and "--log-level" in result.output


@pytest.mark.parametrize("finish", ["stop", "length", "tool_calls"])
def test_engine_emits_one_summary_and_releases_lease(output, monkeypatch, finish):
    import threading

    from mlx_one.server.generation_engine import GenerationEngine, GenerationEvent
    from mlx_one.text.schemas import TextGenerationOptions

    released = []
    lease = SimpleNamespace(release=lambda: released.append(True))
    manager = SimpleNamespace(current_model=lambda: object(), model_lease=lambda: lease)
    engine = GenerationEngine(manager)
    monkeypatch.setattr(
        engine,
        "_stream_request",
        lambda **kw: iter(
            [
                GenerationEvent(text="ok"),
                GenerationEvent(
                    finish_reason=finish,
                    metrics={
                        "prompt_tokens": 2,
                        "generated_tokens": 1,
                        "generation_latency_ms": 10,
                        "tokens_per_second": 100,
                    },
                ),
            ]
        ),
    )
    events = list(
        engine.stream(
            model="example", messages=[], options=TextGenerationOptions(), cancel=threading.Event()
        )
    )
    assert events[-1].finish_reason == finish
    assert output.getvalue().count("launch:") == 1
    assert output.getvalue().count("release:") == 1
    assert f"finish = {finish}" in output.getvalue()
    assert released == [True]
