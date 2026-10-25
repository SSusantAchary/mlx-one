"""OCR task adapter on the existing device actor, without a new scheduler."""

from __future__ import annotations

from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.task_runners.base import NativeTaskAdapter
from mlx_one.engine.tasks import PreparedTask


class OCRTaskAdapter(NativeTaskAdapter):
    def __init__(self, bundle):
        self.bundle = bundle
        self.family = bundle.architecture
        self.supported_tasks = tuple(TaskKind(task) for task in bundle.capabilities.tasks)

    def prepare(self, task, payload):
        from mlx_one.ocr.schemas import OCRInput

        request = payload["request"]
        if not isinstance(request, OCRInput) or str(TaskKind(task).value) not in (
            self.bundle.capabilities.tasks
        ):
            raise ValueError("selected checkpoint does not implement this OCR task")
        image = payload["image"]
        limit = request.limits
        # Includes page buffers, float preprocessing, vision attention and decoder KV.
        if image.width * image.height > limit.max_pixels:
            raise ValueError("OCR input exceeds pixel budget")
        cfg = self.bundle.config
        v = getattr(cfg, "vision_config", getattr(cfg, "encoder", None))
        hidden = getattr(v, "embed_dim", getattr(v, "hidden_size", 1024))
        # Bound the supported visual modes, including DeepSeek's 9 local tiles.
        visual_bound = {
            "trocr": (cfg.encoder.image_size // cfg.encoder.patch_size) ** 2 + 1
            if self.family == "trocr"
            else 0,
            "got_ocr2": 4096,
            "florence2": 144 * 144,
            "paddleocr_vl": 5200,
            "lightonocr": 12100,
            "deepseek_ocr": 4096 + limit.max_tiles * 1600,
        }[self.family]
        # SDPA does not materialize all attention scores; allow one worst-case
        # global SAM block (4096^2 * 12 heads) as a conservative fixed allowance.
        budget = image.width * image.height * 32 + visual_bound * hidden * 32 + 1024**3
        if budget > limit.memory_budget_bytes:
            raise ValueError("OCR request exceeds transient memory budget")
        copy = image.copy()
        return PreparedTask(
            TaskKind(task),
            {**payload, "image": copy},
            (min(32768, visual_bound + request.max_tokens),),
            (self.family, request.task),
            budget,
        )

    def dispose_prepared(self, prepared):
        prepared.payload["image"].close()

    def run_steps(self, prepared, controls):
        from mlx_one.ocr.generation import stream_page

        results = []
        for item, control in zip(prepared, controls, strict=True):
            data = item.payload
            iterator = stream_page(
                self.bundle,
                data["image"],
                data["request"],
                page=data.get("page", 0),
                page_metadata=data.get("metadata"),
                cancelled=control.cancelled,
            )
            try:
                while True:
                    try:
                        next(iterator)
                    except StopIteration as stopped:
                        results.append(stopped.value)
                        break
                    yield None
            finally:
                iterator.close()
        return tuple(results)
