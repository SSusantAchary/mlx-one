"""Deterministic Kilo/Harmony contracts; no weights or Metal required."""

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mlx_one.engine.prompts import prepare_chat
from mlx_one.server.app import _until_disconnected, create_app
from mlx_one.server.config import ServerConfig
from mlx_one.server.generation_engine import GenerationEvent, RuntimeStats
from mlx_one.server.model_manager import ModelMetadata
from mlx_one.server.schemas import ChatCompletionRequest
from mlx_one.text.chat import ChatTemplate
from mlx_one.text.harmony_tools import ALIASES, HarmonyToolParser, load_arguments

FIXTURE = Path(__file__).parent / "fixtures/kilo/request.json"
TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}
USER = {"role": "user", "content": "Write hello.py using the tool."}
ARGS = json.dumps({"path": "hello.py", "content": 'print("héllo")\n'}, ensure_ascii=False)
CALL = "<|channel|>commentary to=functions.write_file<|message|>" + ARGS + "<|call|>"
FINAL = (
    "<|channel|>analysis<|message|>private analysis<|end|><|start|>assistant"
    "<|channel|>final<|message|>Done.<|return|>"
)


class Tokenizer:
    """Character IDs plus real control IDs; literal spellings can be ordinary text."""

    bos_token_id = None
    special_tokens = {}
    eos_token = "<|return|>"
    eos_token_id = 200002
    bos_token = None

    def __init__(self, modern=False):
        self.special = {
            values[int(modern and len(values) > 1)]: 200000 + index
            for index, values in enumerate(ALIASES.values())
        }
        self.inverse = {value: key for key, value in self.special.items()}

    def encode(self, text, **kwargs):
        parts = re.split("(" + "|".join(re.escape(x) for x in self.special) + ")", text)
        return [
            token
            for part in parts
            for token in ([self.special[part]] if part in self.special else [ord(c) for c in part])
        ]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(
            self.inverse.get(i, "")
            if i in self.inverse and skip_special_tokens
            else self.inverse[i]
            if i in self.inverse
            else chr(i)
            for i in ids
        )


def request(**kwargs):
    return {"model": "test-alias", "messages": [USER], "tools": [TOOL], **kwargs}


def parser(tokenizer=None, **kwargs):
    return HarmonyToolParser(tokenizer or Tokenizer(), tools=[TOOL], **kwargs)


def feed(p, text):
    return "".join(p.feed(token) for token in p.tokenizer.encode(text))


@pytest.mark.parametrize("choice", [None, "none", "auto"])
def test_request_schema(choice):
    data = ChatCompletionRequest.model_validate(
        request(
            tool_choice=choice, parallel_tool_calls=False, stream_options={"include_usage": True}
        )
    )
    assert data.tools[0].function.parameters == TOOL["function"]["parameters"]


def test_large_sanitized_kilo_schema():
    data = ChatCompletionRequest.model_validate_json(FIXTURE.read_text())
    assert len(data.tools) == 29
    assert data.stream and data.stream_options.include_usage
    assert data.tools[0].function.parameters["properties"]["options"]["anyOf"][1]


@pytest.mark.parametrize(
    "change",
    [
        {"tool_choice": {"type": "function", "function": {"name": "absent"}}},
        {"tools": [TOOL, TOOL]},
        {"stream_options": {"unknown": True}},
        {"unexpected": True},
        {"temperature": float("inf")},
        {"messages": [{"role": "tool", "tool_call_id": "missing", "content": "data"}]},
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "write_file", "arguments": "{}"},
                        }
                    ],
                }
            ]
        },
    ],
)
def test_invalid_requests(change):
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(request(**change))


@pytest.mark.parametrize("arguments", ["{", "[]", '{"a":NaN}', '{"a":1,"a":2}'])
def test_arguments_validation(arguments):
    with pytest.raises(ValueError):
        load_arguments(arguments)


@pytest.mark.parametrize("modern", [False, True])
def test_tool_prompt_and_round_trip(modern):
    tokenizer = Tokenizer(modern)
    template = ChatTemplate("gpt_oss", None, tokenizer)
    history = [
        USER,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "one",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": ARGS},
                },
                {
                    "id": "two",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "two", "content": "second result"},
        {"role": "tool", "tool_call_id": "one", "content": "first result"},
    ]
    ChatCompletionRequest.model_validate(request(messages=history))
    prompt = template.render(history, tools=[TOOL])
    assert prompt.count("type write_file =") == 1
    assert "functions.write_file to=assistant" in prompt
    assert prompt.index("second result") < prompt.index("first result")
    assert ARGS in prompt and json.dumps(ARGS) not in prompt
    assert "commentary json" in prompt
    assert json.dumps("second result") in prompt
    assert "path: string" in prompt
    assert prompt.endswith(("<|start|>assistant", "<|im_start|>assistant"))
    assert "type write_file =" not in template.render([USER], tools=[TOOL], tool_choice="none")


def test_large_catalog_prompt_tokens_include_tools_once():
    data = ChatCompletionRequest.model_validate_json(FIXTURE.read_text())
    bundle = Bundle()
    tools = [tool.model_dump(exclude_none=True) for tool in data.tools]
    prepared = prepare_chat(
        bundle, [USER], 10, bundle.context_length, rendering_options={"tools": tools}
    )
    assert prepared.text.count("namespace functions {") == 1
    assert prepared.text.count("type coding_tool_") == 29
    assert len(prepared.token_ids) > 20000
    with pytest.raises(ValueError, match="context"):
        prepare_chat(bundle, [USER], 10, 100, rendering_options={"tools": tools})


def test_partial_call_kept_private_until_validated_handoff():
    p = parser()
    text = CALL[: -len("<|call|>")]
    for token in p.tokenizer.encode(text):
        assert p.feed(token) == ""
        assert not p.calls
    feed(p, "<|call|>")
    assert p.finish("stop") == "tool_calls"
    assert p.calls[0]["function"]["arguments"] == ARGS
    deltas = list(p.deltas())
    assert "".join(d["function"]["arguments"] for d in deltas) == ARGS
    assert deltas[0]["index"] == 0 and deltas[0]["id"] == p.calls[0]["id"]


@pytest.mark.parametrize("suffix", ["<|return|>", "<|call|>"])
def test_distinct_control_tokens(suffix):
    p = parser()
    text = CALL.replace("<|call|>", suffix)
    if suffix == "<|return|>":
        with pytest.raises(ValueError, match="handoff"):
            feed(p, text)
    else:
        feed(p, text)
        assert p.finish("stop") == "tool_calls"


def test_final_content_reasoning_and_ordinary_prose():
    p = parser()
    assert feed(p, FINAL) == "Done."
    assert p.reasoning_tokens > 0 and p.finish("stop") == "stop"
    assert not p.calls
    p = parser()
    prose = 'Please call write_file({"path":"x"})'
    assert feed(p, "<|channel|>final<|message|>" + prose + "<|return|>") == prose
    assert not p.calls


def test_literal_marker_spelling_is_not_a_protocol_token():
    p = parser()
    feed(p, "<|channel|>final<|message|>")
    literal = "<|call|>"
    assert "".join(p.feed(ord(c)) for c in literal) == literal
    feed(p, "<|return|>")
    assert p.finish("stop") == "stop" and not p.calls


@pytest.mark.parametrize(
    "text",
    [
        CALL.replace(ARGS, '{"path":'),
        CALL.replace("write_file", "undeclared"),
        CALL.replace("to=functions.write_file", ""),
        CALL.replace("<|call|>", "<|return|>"),
        "<|channel|>final<|message|>bad<|start|>assistant",
    ],
)
def test_malformed_output_is_controlled(text):
    p = parser()
    with pytest.raises(ValueError):
        feed(p, text)
    assert not list(p.deltas())
    p.close()
    assert not p.body_ids and not p.calls


def test_tool_choice_none_blocks_actions():
    p = parser(tool_choice="none")
    with pytest.raises(ValueError, match="disabled"):
        feed(p, CALL)
    assert not p.calls


def test_truncation_does_not_publish_partial_action():
    p = parser()
    feed(p, CALL[: -len("<|call|>")])
    assert p.finish("length") == "length"
    assert not list(p.deltas())


def test_multiple_calls_and_parallel_control():
    text = CALL.replace("<|call|>", "<|end|>") + "<|start|>assistant" + CALL
    p = parser()
    feed(p, text)
    assert p.finish("stop") == "tool_calls"
    assert len(p.calls) == 2 and p.calls[0]["id"] != p.calls[1]["id"]
    assert {d["index"] for d in p.deltas()} == {0, 1}
    p = parser(parallel_tool_calls=False)
    with pytest.raises(ValueError, match="at most one"):
        feed(p, text)


def test_bounded_state_and_request_isolation():
    p = parser(max_buffer=2)
    with pytest.raises(ValueError, match="limit"):
        feed(p, CALL)
    p.close()
    for _ in range(25):
        first, second = parser(), parser()
        feed(first, CALL)
        feed(second, CALL)
        assert first.calls[0]["id"] != second.calls[0]["id"]
        first.close()
        second.close()
        assert not first.body_ids and not first.calls


class Bundle:
    architecture = "gpt_oss"
    model_id = "test-alias"
    context_length = 131072

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.chat_template = ChatTemplate("gpt_oss", None, self.tokenizer)


class Manager:
    def current_model(self):
        return Bundle()

    def model_info(self):
        return ModelMetadata("test-alias", "gpt_oss", 131072, 1, {}, None)

    def list_models(self):
        return (self.model_info(),)

    def unload(self):
        pass


class Engine:
    stats = RuntimeStats()
    closed = 0
    seen = []

    def stream(self, **settings):
        self.seen.append(settings)
        p = HarmonyToolParser(
            Tokenizer(),
            tools=settings.get("tools") or [],
            tool_choice=settings.get("tool_choice"),
            parallel_tool_calls=settings.get("parallel_tool_calls"),
        )
        try:
            history = settings["messages"]
            output = (
                FINAL
                if history[-1]["role"] == "tool" or settings.get("tool_choice") == "none"
                else CALL
            )
            for token in p.tokenizer.encode(output):
                text = p.feed(token)
                if text:
                    yield GenerationEvent(text=text)
            reason = p.finish("stop")
            if reason == "tool_calls":
                for delta in p.deltas():
                    yield GenerationEvent(tool_calls=[delta])
            yield GenerationEvent(
                finish_reason=reason, metrics={"prompt_tokens": 50, "generated_tokens": 20}
            )
        finally:
            p.close()
            self.closed += 1


def chunks(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize("streaming", [False, True])
def test_http_tool_round_trip_usage_and_alias(tmp_path, streaming):
    engine = Engine()
    with TestClient(create_app(Manager(), engine, ui_dir=tmp_path)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=request(stream=streaming, stream_options={"include_usage": True}),
        )
        assert response.status_code == 200, response.text
        if streaming:
            events = chunks(response)
            calls = {}
            for event in events:
                assert event["model"] == "test-alias"
                for choice in event["choices"]:
                    for delta in choice["delta"].get("tool_calls", []):
                        call = calls.setdefault(
                            delta["index"],
                            {
                                "id": delta.get("id"),
                                "type": "function",
                                "function": {
                                    "name": delta["function"].get("name"),
                                    "arguments": "",
                                },
                            },
                        )
                        call["function"]["arguments"] += delta["function"].get("arguments", "")
            call = calls[0]
            assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"
            assert events[-1]["choices"] == []
            assert events[-1]["usage"]["total_tokens"] == 70
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            data = response.json()
            call = data["choices"][0]["message"]["tool_calls"][0]
            assert data["choices"][0]["finish_reason"] == "tool_calls"
            assert data["choices"][0]["message"]["content"] is None
        assert call["function"]["arguments"] == ARGS
        history = [
            USER,
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "File written."},
        ]
        continuation = client.post("/v1/chat/completions", json=request(messages=history)).json()
        assert continuation["choices"][0]["message"] == {"role": "assistant", "content": "Done."}
    assert engine.closed == 2


def test_http_kilo_fixture_reaches_generation(tmp_path):
    engine = Engine()
    # Sanitized tool names: a final-only fake exercises HTTP request validation.
    engine.stream = lambda **kw: iter(
        [GenerationEvent(text="accepted"), GenerationEvent(finish_reason="stop")]
    )
    with TestClient(create_app(Manager(), engine, ui_dir=tmp_path)) as client:
        response = client.post(
            "/v1/chat/completions", json=json.loads(FIXTURE.read_text()) | {"model": "test-alias"}
        )
        assert response.status_code == 200 and "accepted" in response.text


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("tools", [None, [TOOL]])
@pytest.mark.parametrize("server_budget,request_budget", [(-1, None), (-1, -1), (128, -1)])
def test_http_gpt_oss_accepts_automatic_reasoning_budget(
    tmp_path, streaming, tools, server_budget, request_budget
):
    """The UI's explicit -1 sentinel has the same meaning as the default."""
    engine = Engine()
    received = []

    def stream(**settings):
        received.append(settings)
        yield GenerationEvent(text="Hello.")
        yield GenerationEvent(finish_reason="stop")

    engine.stream = stream
    config = ServerConfig(reasoning_budget=server_budget, warmup=False, cache_prompt=False)
    payload = {
        "model": "test-alias", "messages": [USER], "stream": streaming,
        "temperature": 0.7, "top_p": 0.9, "max_tokens": 512, "reasoning": "auto",
    }
    if request_budget is not None:
        payload["reasoning_budget"] = request_budget
    if tools is not None:
        payload["tools"] = tools
    with TestClient(create_app(Manager(), engine, config=config, ui_dir=tmp_path)) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200, response.text
        if streaming:
            events = chunks(response)
            assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == "Hello."
            assert events[-1]["choices"][0]["finish_reason"] == "stop"
        else:
            assert response.json()["choices"][0]["message"]["content"] == "Hello."
    assert len(received) == 1
    assert received[0]["reasoning"] == "auto"
    assert received[0]["reasoning_budget"] == -1


@pytest.mark.parametrize(
    "server_budget,overrides",
    [(-1, {"reasoning_budget": 0}), (-1, {"reasoning_budget": 128}),
     (128, {}), (-1, {"reasoning": "on"}), (-1, {"reasoning": "off"})],
)
def test_http_gpt_oss_rejects_explicit_reasoning_controls(tmp_path, server_budget, overrides):
    engine = Engine()

    def unexpected_generation(**settings):
        pytest.fail("unsupported reasoning controls must be rejected before generation")

    engine.stream = unexpected_generation
    config = ServerConfig(reasoning_budget=server_budget, warmup=False, cache_prompt=False)
    with TestClient(create_app(Manager(), engine, config=config, ui_dir=tmp_path)) as client:
        response = client.post(
            "/v1/chat/completions", json={"model": "test-alias", "messages": [USER], **overrides}
        )
        assert response.status_code == 400, response.text
        assert "reasoning" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "write_file"}}]
)
def test_unsupported_choice_returns_capability_error(tmp_path, choice):
    with TestClient(create_app(Manager(), Engine(), ui_dir=tmp_path)) as client:
        response = client.post("/v1/chat/completions", json=request(tool_choice=choice))
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "capability_error"


def test_http_none_auth_overflow_and_bounded_validation(tmp_path):
    config = ServerConfig(api_keys=("test-secret",), warmup=False, cache_prompt=False)
    with TestClient(create_app(Manager(), Engine(), config=config, ui_dir=tmp_path)) as client:
        assert client.post("/v1/chat/completions", json=request()).status_code == 401
        client.headers["Authorization"] = "Bearer test-secret"
        assert client.get("/v1/models").json()["data"][0]["id"] == "test-alias"
        response = client.post("/v1/chat/completions", json=request(tool_choice="none"))
        assert response.json()["choices"][0]["message"]["content"] == "Done."
        assert "tool_calls" not in response.json()["choices"][0]["message"]
        assert client.post("/v1/chat/completions", json=request(model="wrong")).status_code == 404
        assert (
            client.post(
                "/v1/chat/completions",
                json=request(messages=[{"role": "user", "content": "x" * 131072}]),
            ).status_code
            == 400
        )
        response = client.post("/v1/chat/completions", json=request(private="DO-NOT-ECHO" * 1000))
        assert response.status_code == 422 and "DO-NOT-ECHO" not in response.text
        assert len(response.text) < 4096
        assert client.post("/v1/chat/completions", content=b"x" * 1048577).status_code == 413


def test_disconnect_cancels_while_parser_has_no_visible_output():
    closed = []

    async def source():
        try:
            await asyncio.sleep(10)
            yield GenerationEvent()
        finally:
            closed.append(True)

    async def exercise():
        request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=True))
        stream = _until_disconnected(request, source())
        with pytest.raises(Exception, match="disconnected"):
            await anext(stream)
        await stream.aclose()

    asyncio.run(exercise())
    assert closed == [True]


def test_http_text_content_parts_are_text_not_vision(tmp_path):
    engine = Engine()
    with TestClient(create_app(Manager(), engine, ui_dir=tmp_path)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=request(
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": "Use tools."}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Write"},
                            {"type": "text", "text": "hello.py"},
                        ],
                    },
                ]
            ),
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
        assert engine.seen[-1]["messages"][-1]["content"] == "Write\nhello.py"


def test_adaptive_tool_prefill_keeps_native_admission_guards(monkeypatch):
    from contextlib import contextmanager

    from mlx_one.engine.cache_errors import CacheCapacityError
    from mlx_one.server.generation_engine import _fit_tool_prefill

    attempts = []

    class Runtime:
        capacity_diagnostics = {}

        @contextmanager
        def transient_reservation(self, amount):
            attempts.append(amount)
            if amount > 128:
                self.capacity_diagnostics = {
                    "rejection_reason": "request exceeds conservative host memory headroom"
                }
                raise CacheCapacityError("headroom")
            yield

    monkeypatch.setattr(
        "mlx_one.engine.pressure.execution_forecast", lambda *a, chunk, **kw: {"bytes": chunk}
    )
    assert _fit_tool_prefill(Runtime(), None, 6000, 512) == 128
    assert attempts == [512, 256, 128]


def test_adaptive_prefill_does_not_bypass_critical_pressure(monkeypatch):
    from contextlib import contextmanager

    from mlx_one.engine.cache_errors import CacheCapacityError
    from mlx_one.server.generation_engine import _fit_tool_prefill

    class Runtime:
        capacity_diagnostics = {
            "rejection_reason": "request rejected by critical host memory pressure"
        }

        @contextmanager
        def transient_reservation(self, amount):
            raise CacheCapacityError("critical pressure")
            yield

    monkeypatch.setattr("mlx_one.engine.pressure.execution_forecast", lambda *a, **kw: {"bytes": 1})
    with pytest.raises(CacheCapacityError, match="critical pressure"):
        _fit_tool_prefill(Runtime(), None, 6000, 512)


def test_context_shift_removes_whole_tool_turn():
    bundle = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=list),
        chat_template=SimpleNamespace(
            render=lambda messages, **kw: "|".join(
                message.get("content") or "" for message in messages
            )
        ),
    )
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "call"},
        {"role": "tool", "content": "data"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "new"},
    ]
    prepared = prepare_chat(bundle, messages, 2, 8, context_shift=True)
    assert [message["content"] for message in prepared.messages] == ["S", "new"]


@pytest.mark.parametrize(
    "schema",
    [
        {"properties": []},
        {"required": "path"},
        {"type": {}},
        {"anyOf": []},
        {"enum": 12},
        {"properties": {"x": {"description": 1}}},
    ],
)
def test_invalid_schema_shapes_return_validation_errors(schema):
    tool = {"type": "function", "function": {"name": "f", "parameters": schema}}
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(request(tools=[tool]))


def test_unknown_schema_constraints_preserved_without_duplicate_definitions():
    data = ChatCompletionRequest.model_validate_json(FIXTURE.read_text())
    prompt = Bundle().chat_template.render(
        [USER], tools=[tool.model_dump(exclude_none=True) for tool in data.tools]
    )
    assert prompt.count("type coding_tool_00 =") == 1
    assert "https://json-schema.org/draft/2020-12/schema" in prompt
    assert "path: string" in prompt
    assert "options?" in prompt
    assert '"properties":{"path":{"type":"string"}' not in prompt


def test_sdk_heartbeat_is_an_empty_text_delta(monkeypatch):
    from mlx_one.server.app import _sse

    async def source():
        yield GenerationEvent()
        yield GenerationEvent()
        yield GenerationEvent(text="ok")
        yield GenerationEvent(finish_reason="stop")

    from itertools import count

    ticks = count(step=20)
    monkeypatch.setattr("mlx_one.server.app.time", SimpleNamespace(monotonic=lambda: next(ticks)))

    async def collect():
        request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))
        return [part async for part in _sse(request, source(), "id", 1, "model")]

    parts = asyncio.run(collect())
    events = [json.loads(part[6:]) for part in parts if part.startswith("data: {")]
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(event["choices"][0]["delta"] == {"content": ""} for event in events)
    assert parts[-1] == "data: [DONE]\n\n"


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_reasoning_effort_reaches_harmony_prompt(tmp_path, effort):
    assert (
        ChatCompletionRequest.model_validate(request(reasoning_effort=effort)).reasoning_effort
        == effort
    )
    prompt = Bundle().chat_template.render([USER], tools=[TOOL], reasoning_effort=effort)
    assert "Reasoning: " + effort in prompt
    engine = Engine()
    with TestClient(create_app(Manager(), engine, ui_dir=tmp_path)) as client:
        response = client.post("/v1/chat/completions", json=request(reasoning_effort=effort))
        assert response.status_code == 200, response.text
        assert engine.seen[-1]["reasoning_effort"] == effort


@pytest.mark.parametrize("effort", ["none", "minimal", "xhigh"])
def test_unsupported_reasoning_effort_is_explicit(tmp_path, effort):
    with TestClient(create_app(Manager(), Engine(), ui_dir=tmp_path)) as client:
        response = client.post("/v1/chat/completions", json=request(reasoning_effort=effort))
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "capability_error"


def test_checkpoint_layout_places_instructions_before_tools_in_one_developer_message():
    prompt = Bundle().chat_template.render(
        [{"role": "system", "content": "Use coding tools carefully."}, USER], tools=[TOOL]
    )
    assert prompt.count("<|start|>developer") == 1
    assert prompt.index("# Instructions") < prompt.index("Use coding tools carefully.")
    assert prompt.index("Use coding tools carefully.") < prompt.index("# Tools")
    assert prompt.index("# Tools") < prompt.index("<|start|>user")
