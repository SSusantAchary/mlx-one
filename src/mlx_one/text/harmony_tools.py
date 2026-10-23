"""Native function-call adaptation for GPT-OSS Harmony.

Protocol reference: https://developers.openai.com/cookbook/articles/openai-harmony
Implemented here without a serving-framework or openai-harmony dependency.
Only checkpoint-resolved single token IDs can change the output parser's state.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime

ALIASES = {
    "start": ("<|start|>", "<|im_start|>"),
    "message": ("<|message|>", "<|im_sep|>"),
    "channel": ("<|channel|>", "<|meta_sep|>"),
    "format": ("<|constrain|>", "<|meta_start|>", "<|format|>"),
    "end": ("<|end|>", "<|im_end|>"),
    "return": ("<|return|>", "<|fim_suffix|>"),
    "call": ("<|call|>", "<|ghissue|>"),
    "eot": ("<|endoftext|>",),
}


def control_tokens(tokenizer):
    result = {}
    for kind, spellings in ALIASES.items():
        for spelling in spellings:
            ids = tokenizer.encode(spelling)
            if len(ids) == 1 and tokenizer.decode(ids, skip_special_tokens=False) == spelling:
                result.setdefault(kind, (ids[0], spelling))
    if any(kind not in result for kind in ("start", "message", "channel", "end", "return", "call")):
        raise ValueError("checkpoint tokenizer lacks required Harmony special tokens")
    return result


def load_arguments(text):
    def invalid_constant(_):
        raise ValueError("function arguments must be finite JSON")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("function arguments contain duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(text, parse_constant=invalid_constant, object_pairs_hook=unique_object)
    except (ValueError, RecursionError) as exc:
        raise ValueError("function arguments must be a complete JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("function arguments must be a JSON object")
    return value


def validate_schema(schema, depth=0):
    """Check renderer-relevant JSON Schema shapes, retaining all vocabulary keys."""
    if depth > 48:
        raise ValueError("tool schema nesting exceeds 48 levels")
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise ValueError("JSON Schema nodes must be objects or booleans")
    for key in ("properties", "patternProperties", "$defs", "definitions"):
        if key in schema:
            if not isinstance(schema[key], dict):
                raise ValueError("JSON Schema property definitions must be objects")
            for child in schema[key].values():
                validate_schema(child, depth + 1)
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if key in schema:
            if not isinstance(schema[key], list) or not schema[key]:
                raise ValueError("JSON Schema alternatives must be non-empty arrays")
            for child in schema[key]:
                validate_schema(child, depth + 1)
    for key in ("items", "additionalProperties", "not", "if", "then", "else", "contains"):
        if key in schema:
            children = schema[key] if isinstance(schema[key], list) else [schema[key]]
            for child in children:
                validate_schema(child, depth + 1)
    if "required" in schema and (
        not isinstance(schema["required"], list)
        or any(not isinstance(name, str) for name in schema["required"])
    ):
        raise ValueError("JSON Schema required must be an array of property names")
    kind = schema.get("type")
    if kind is not None:
        kinds = kind if isinstance(kind, list) else [kind]
        if not kinds or any(
            value not in ("object", "array", "string", "number", "integer", "boolean", "null")
            for value in kinds
        ):
            raise ValueError("invalid JSON Schema type")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("JSON Schema enum must be a non-empty array")
    if "description" in schema and not isinstance(schema["description"], str):
        raise ValueError("JSON Schema description must be text")


def _typescript(schema, depth=0):
    if depth > 48:
        raise ValueError("tool schema nesting exceeds 48 levels")
    if not isinstance(schema, dict):
        return "any" if schema else "never"

    def render(item):
        return _typescript(item, depth + 1)

    if "const" in schema:
        return json.dumps(schema["const"], ensure_ascii=False)
    if "enum" in schema:
        return " | ".join(json.dumps(item, ensure_ascii=False) for item in schema["enum"])
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            return (" & " if key == "allOf" else " | ").join(
                "(" + render(item) + ")" for item in schema[key]
            )
    kind = schema.get("type")
    if isinstance(kind, list):
        return " | ".join(render({**schema, "type": item}) for item in kind)
    if kind == "object" or "properties" in schema:
        required = schema.get("required", [])
        fields = []
        for name, spec in schema.get("properties", {}).items():
            description = spec.get("description", "") if isinstance(spec, dict) else ""
            comment = "".join("// " + line + "\n" for line in description.splitlines())
            fields.append(
                comment
                + (name if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name) else json.dumps(name))
                + ("" if name in required else "?")
                + ": "
                + render(spec)
                + ","
            )
        return "{\n" + "\n".join(fields) + "\n}" if fields else "object"
    if kind == "array":
        return "(" + render(schema.get("items", {})) + ")[]"
    return {
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "string": "string",
        "null": "null",
    }.get(kind, "any")


def _schema_constraints(schema, *, description_rendered=False):
    """Keep vocabulary that signatures cannot express without repeating the whole schema."""
    if not isinstance(schema, dict):
        return {}
    represented = {
        "type",
        "properties",
        "required",
        "enum",
        "const",
        "description",
        "items",
        "anyOf",
        "oneOf",
        "allOf",
    }
    result = {key: value for key, value in schema.items() if key not in represented}
    if "description" in schema and not description_rendered:
        result["description"] = schema["description"]
    properties = {
        name: extra
        for name, child in schema.get("properties", {}).items()
        if (extra := _schema_constraints(child, description_rendered=True))
    }
    if properties:
        result["properties"] = properties
    if "items" in schema:
        extra = _schema_constraints(schema["items"])
        if extra:
            result["items"] = extra
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            children = [_schema_constraints(child) for child in schema[key]]
            if any(children):
                result[key] = children
    return result


def render_harmony(
    messages,
    tokenizer,
    *,
    tools=None,
    tool_choice=None,
    parallel_tool_calls=None,
    add_generation_prompt=True,
    reasoning_effort="medium",
    **_,
):
    """Render tool conversations, resolving every control spelling from the checkpoint."""
    markers = control_tokens(tokenizer)

    def token(kind):
        return markers[kind][1]

    active_tools = tools if tool_choice != "none" else []
    system = (
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\nCurrent date: "
        + datetime.now().strftime("%Y-%m-%d")
        + "\n\nReasoning: "
        + reasoning_effort
        + "\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    )
    if active_tools:
        system += "\nCalls to these tools must go to the commentary channel: 'functions'."
    parts = [token("start") + "system" + token("message") + system + token("end")]
    remaining = list(messages)
    leading = []
    while remaining and remaining[0]["role"] in {"system", "developer"}:
        content = remaining.pop(0).get("content") or ""
        if not isinstance(content, str):
            raise ValueError("Harmony developer instructions must be text")
        leading.append(content)
    instructions = ("# Instructions\n\n" + "\n\n".join(leading) + "\n\n") if leading else ""
    if active_tools:
        declarations = []
        for tool in active_tools:
            function = tool["function"]
            schema = function.get("parameters") or {"type": "object"}
            description = "".join(
                "// " + line + "\n" for line in (function.get("description") or "").splitlines()
            )
            constraints = _schema_constraints(schema)
            schema_comment = (
                (
                    "// JSON Schema constraints: "
                    + json.dumps(constraints, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                if constraints
                else ""
            )
            signature = "(_: " + _typescript(schema) + ")"
            declarations.append(
                description
                + schema_comment
                + "type "
                + function["name"]
                + " = "
                + signature
                + " => any;\n"
            )
        instructions += "# Tools\n\n## functions\n\nnamespace functions {\n\n"
        instructions += "\n".join(declarations) + "\n} // namespace functions"
        if parallel_tool_calls is False:
            instructions += "\nCall at most one function before waiting for its result."
    if instructions:
        parts.append(token("start") + "developer" + token("message") + instructions + token("end"))
    calls = {}
    for message in remaining:
        role = message["role"]
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("Harmony function calling requires text messages")
        if role in {"system", "developer", "user"}:
            role = "developer" if role == "system" else role
            parts.append(token("start") + role + token("message") + content + token("end"))
        elif role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if content:
                channel = "commentary" if tool_calls else "final"
                parts.append(
                    token("start")
                    + "assistant"
                    + token("channel")
                    + channel
                    + token("message")
                    + content
                    + token("end")
                )
            for call in tool_calls:
                function = call["function"]
                load_arguments(function["arguments"])
                calls[call["id"]] = function["name"]
                parts.append(
                    token("start")
                    + "assistant to=functions."
                    + function["name"]
                    + token("channel")
                    + "commentary json"
                    + token("message")
                    + function["arguments"]
                    + token("call")
                )
        elif role == "tool":
            name = calls.pop(message.get("tool_call_id"), None)
            if name is None:
                raise ValueError("tool result has no preceding matching tool call")
            parts.append(
                token("start")
                + "functions."
                + name
                + " to=assistant"
                + token("channel")
                + "commentary"
                + token("message")
                + json.dumps(content, ensure_ascii=False)
                + token("end")
            )
        else:
            raise ValueError("unsupported Harmony message role")
    if add_generation_prompt:
        parts.append(token("start") + "assistant")
    return "".join(parts)


class HarmonyToolParser:
    """Request-local bounded token state. Publish actions only after a valid handoff.

    Analysis is counted but never returned. Tool JSON is decoded once at the end,
    then exposed as ordered deltas; truncated or malformed calls expose no action.
    """

    def __init__(
        self, tokenizer, *, tools=(), tool_choice=None, parallel_tool_calls=None, max_buffer=1048576
    ):
        self.tokenizer = tokenizer
        self.controls = {value[0]: kind for kind, value in control_tokens(tokenizer).items()}
        self.allowed = {tool["function"]["name"] for tool in tools}
        self.enabled = tool_choice != "none" and bool(self.allowed)
        self.parallel = parallel_tool_calls is not False
        self.max_buffer = max_buffer
        self.header = "assistant"
        self.in_header = True
        self.body_ids = []
        self.committed = ""
        self.recipient = None
        self.channel = None
        self.reasoning_tokens = 0
        self.calls = []
        self.done = False
        self.handoff = False
        self.prefix = uuid.uuid4().hex

    def feed(self, token_id):
        try:
            return self._feed(token_id)
        except ValueError as exc:
            # All parser diagnostics are fixed messages, never arguments, prompts or analysis.
            logging.getLogger(__name__).warning("Harmony output rejected: %s", str(exc)[:200])
            raise

    def _feed(self, token_id):
        if self.done:
            raise ValueError("Harmony output continues after termination")
        kind = self.controls.get(token_id)
        if kind is None:
            if self.in_header:
                self.body_ids.append(token_id)
                if len(self.body_ids) > 1024:
                    raise ValueError("Harmony header exceeds limit")
            elif self.recipient or self.channel != "analysis":
                self.body_ids.append(token_id)
                if len(self.body_ids) > 32768:
                    raise ValueError("Harmony message exceeds token limit")
                if self.recipient:
                    return ""
                return self._visible()
            else:
                self.reasoning_tokens += 1
            return ""
        if kind in {"channel", "format", "message"}:
            if not self.in_header:
                raise ValueError("unexpected Harmony header marker in body")
            self.header += self.tokenizer.decode(self.body_ids, skip_special_tokens=False)
            self.body_ids.clear()
            if kind == "channel":
                self.header += " channel="
            elif kind == "format":
                self.header += " format="
            else:
                fields = self.header.split()
                if not fields or fields[0] != "assistant":
                    raise ValueError("model generated a non-assistant message")
                recipient = re.search(r"(?:^|\s)to=([^\s]+)", self.header)
                self.recipient = recipient.group(1) if recipient else None
                channel = re.search(r"channel=\s*(\w+)", self.header)
                self.channel = channel.group(1) if channel else "final"
                if self.channel not in {"analysis", "commentary", "final"}:
                    raise ValueError("unsupported Harmony channel")
                self.in_header = False
            return ""
        if kind == "start":
            if self.body_ids or (not self.in_header):
                raise ValueError("Harmony message missing termination")
            self.header = ""
            return ""
        if kind in {"end", "return", "call", "eot"}:
            if self.in_header:
                raise ValueError("incomplete Harmony header at termination")
            visible = ""
            if self.recipient:
                if kind not in {"call", "end"}:
                    raise ValueError("tool output missing Harmony handoff")
                if not self.enabled:
                    raise ValueError("model requested a tool when tool calls are disabled")
                name = self.recipient.removeprefix("functions.")
                if not self.recipient.startswith("functions.") or name not in self.allowed:
                    raise ValueError("model requested an undeclared function")
                arguments = self.tokenizer.decode(self.body_ids, skip_special_tokens=False)
                if len(arguments.encode("utf-8")) > self.max_buffer:
                    raise ValueError("tool arguments exceed limit")
                load_arguments(arguments)
                if self.calls and not self.parallel:
                    raise ValueError("parallel_tool_calls=false permits at most one call")
                self.calls.append(
                    {
                        "id": "call_" + self.prefix + "_" + str(len(self.calls)),
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            elif kind == "call":
                raise ValueError("Harmony handoff has no function recipient")
            elif self.channel != "analysis":
                visible = self._visible(final=True)
            if kind in {"return", "eot"} and self.calls:
                raise ValueError("function calls lack a final Harmony handoff")
            self.done = kind in {"return", "call", "eot"}
            self.handoff = kind == "call"
            self.in_header = True
            self.body_ids.clear()
            self.committed = ""
            self.header = ""
            return visible
        return ""

    def _visible(self, final=False):
        text = self.tokenizer.decode(self.body_ids, skip_special_tokens=False)
        if not final:
            text = text.rstrip("\ufffd")
        if len(text.encode("utf-8")) > self.max_buffer:
            raise ValueError("Harmony content exceeds limit")
        if not text.startswith(self.committed):
            raise ValueError("tokenizer revised committed Harmony text")
        result, self.committed = text[len(self.committed) :], text
        return result

    def finish(self, reason, *, external_stop=False):
        if reason == "length":
            self.calls.clear()
            return "length"
        if external_stop and not self.recipient and not self.calls:
            return "stop"
        if not self.done:
            raise ValueError("Harmony generation stopped without a terminal control token")
        return "tool_calls" if self.handoff else "stop"

    def deltas(self):
        if not self.handoff or not self.done:
            return
        for index, call in enumerate(self.calls):
            yield {
                "index": index,
                "id": call["id"],
                "type": "function",
                "function": {"name": call["function"]["name"], "arguments": ""},
            }
            arguments = call["function"]["arguments"]
            for start in range(0, len(arguments), 256):
                yield {"index": index, "function": {"arguments": arguments[start : start + 256]}}

    def close(self):
        self.body_ids.clear()
        self.calls.clear()
        self.header = self.committed = ""


class VisibleStops:
    """Apply requested text stops after channel parsing, never to function arguments."""

    def __init__(self, stops):
        self.stops = stops
        self.buffer = ""
        self.matched = False

    def feed(self, text):
        self.buffer += text
        matches = [self.buffer.find(stop) for stop in self.stops if stop in self.buffer]
        if matches:
            ready = self.buffer[: min(matches)]
            self.buffer = ""
            self.matched = True
            return ready
        hold = max(
            (
                length
                for stop in self.stops
                for length in range(1, len(stop))
                if self.buffer.endswith(stop[:length])
            ),
            default=0,
        )
        ready = self.buffer[:-hold] if hold else self.buffer
        self.buffer = self.buffer[-hold:] if hold else ""
        return ready

    def flush(self):
        ready, self.buffer = self.buffer, ""
        return ready
