"""Managed VLM inference using the existing task actor and model pool."""

from __future__ import annotations

from mlx_one.engine.batching import BatchingConfig
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.service import InferenceService
from mlx_one.vision.messages import prepare_messages


class VisionLanguageEngine:
    """Own one native bundle; close streams and the engine to release request state."""

    def __init__(self, model, **load_options):
        self.service = InferenceService(
            batching_config=BatchingConfig(
                max_sequences=1, max_scheduled_tokens=32768, task_transient_budget_bytes=8 * 1024**3
            )
        )
        self.service.register_task("vlm", model, **load_options)

    def _payload(self, prompt, images, messages, options):
        if messages is not None:
            if prompt is not None or images:
                raise ValueError("provide messages or prompt/images")
            actor, lease = self.service._task_lease("vlm", TaskKind.VISION_LANGUAGE)
            try:
                prompt, images = prepare_messages(lease.runtime.value, messages)
            finally:
                lease.release()
        elif images and "<image>" not in (prompt or ""):
            prompt = "<image>" * len(images) + "\n" + (prompt or "")
        if messages is None:
            _, lease = self.service._task_lease("vlm", TaskKind.VISION_LANGUAGE)
            try:
                bundle = lease.runtime.value
                prompt = bundle.chat_template.render(
                    [{"role": "user", "content": prompt}],
                    enable_thinking=False,
                    add_vision_id=False,
                )
            finally:
                lease.release()
        return {"prompt": prompt, "images": tuple(images), "options": options}

    def generate(self, prompt=None, images=(), *, messages=None, options=None, cancel=None):
        return self.service.run_task(
            "vlm",
            TaskKind.VISION_LANGUAGE,
            self._payload(prompt, images, messages, options),
            cancel=cancel,
        ).output

    def stream(self, prompt=None, images=(), *, messages=None, options=None, cancel=None):
        yield from self.service.stream_task(
            "vlm",
            TaskKind.VISION_LANGUAGE,
            self._payload(prompt, images, messages, options),
            cancel=cancel,
        )

    def close(self):
        self.service.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def generate_vlm(model, prompt=None, images=(), *, messages=None, options=None, **load_options):
    with VisionLanguageEngine(model, **load_options) as engine:
        return engine.generate(prompt, images, messages=messages, options=options)
