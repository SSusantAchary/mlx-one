"""Fake native engine used by the Playwright packaged-UI smoke test."""

from __future__ import annotations

import threading
import time

import uvicorn

from mlx_one.server import ModelMetadata, RuntimeStats, create_app
from mlx_one.server.generation_engine import GenerationEvent


class FakeBundle:
    model_id = "test/native"


class FakeManager:
    def current_model(self) -> FakeBundle:
        return FakeBundle()

    def list_models(self) -> tuple[ModelMetadata, ...]:
        return (self.model_info(),)

    def model_info(self) -> ModelMetadata:
        return ModelMetadata("test/native", "qwen3", 4096, 1, {}, None)

    def unload(self) -> None:
        pass


class FakeEngine:
    stats = RuntimeStats()

    def stream(self, **kwargs: object):
        cancel = kwargs["cancel"]
        messages = kwargs["messages"]
        assert isinstance(cancel, threading.Event)
        assert isinstance(messages, list)
        if "long" in messages[-1]["content"]:
            yield GenerationEvent(text="\n".join(f"Response line {line}" for line in range(100)))
        elif "slow" not in messages[-1]["content"]:
            yield GenerationEvent(text="Hello from native MLX")
        else:
            for _ in range(100):
                if cancel.is_set():
                    break
                yield GenerationEvent(text="tick ")
                time.sleep(0.03)
        yield GenerationEvent(
            finish_reason="stop",
            metrics={"prompt_tokens": 2, "generated_tokens": 1, "context_used": 3},
        )


if __name__ == "__main__":
    uvicorn.run(create_app(FakeManager(), FakeEngine()), host="127.0.0.1", port=8080)
