"""Built-in declaration bridge. No new service integration or tensor capability."""
from mlx_one.engine.adapter_registry import RUNTIME_CONTRACT_VERSION, RuntimeRegistration
from mlx_one.engine.contracts import StageKind, TaskKind, TaskSpec, canonical_task


def builtin_runtime(entry):
    component = entry.model_type == "sam3_tracker_video"
    status = ("component_only" if component else "architecture_only" if not entry.loader_path
              and not entry.tasks else "not_integrated" if not entry.loader_path
              else "loader_declared")
    return RuntimeRegistration(entry.model_type, None,
        "mlx_one.engine.runtime_specs:task_specs", "mlx_one.engine.runtime_specs:state_specs",
        RUNTIME_CONTRACT_VERSION, component_only=component, integration_status=status,
        parent_model_type="sam3_video" if component else None,
        reason="Nested video tracker; no standalone task loader" if component else
            "Architecture only; loader and task integration required"
            if status == "architecture_only"
            else "Native loader declaration does not establish shared task-service integration")


def task_specs(entry):
    tasks = {canonical_task(task) for task in entry.tasks}
    # The text loader cannot establish multimodal preparation or outputs.
    if entry.model_type == "qwen3_5":
        tasks.discard(TaskKind.VISION_LANGUAGE)
    recipes = {
        TaskKind.TEXT_GENERATION: ("causal", (StageKind.PREFILL, StageKind.DECODE), "token-budget"),
        TaskKind.VISION_LANGUAGE: ("multimodal", (StageKind.ENCODE, StageKind.PREFILL,
                                                StageKind.DECODE), "media-buckets"),
        TaskKind.EMBEDDING: ("encoder", (StageKind.ENCODE,), "length-buckets"),
        TaskKind.LATE_INTERACTION: ("late-interaction", (StageKind.ENCODE, StageKind.SCORE),
                                   "length-buckets"),
        TaskKind.RERANKING: ("rerank", (StageKind.SCORE,), "length-buckets"),
        TaskKind.TRANSCRIPTION: ("transcription", (StageKind.ENCODE, StageKind.DECODE),
                                "audio-buckets"),
        TaskKind.ALIGNMENT: ("alignment", (StageKind.ENCODE, StageKind.ALIGN),
                             "audio-text-buckets"),
        TaskKind.IMAGE_SEGMENTATION: ("segmentation", (StageKind.ENCODE, StageKind.SCORE),
                                    "image-prompt-buckets"),
        TaskKind.MASK_GENERATION: ("masks", (StageKind.ENCODE, StageKind.SCORE), "image-buckets"),
        TaskKind.VIDEO_SEGMENTATION: ("tracking", (StageKind.FRAME,), "ordered-sessions"),
    }
    if entry.model_type == "parakeet_tdt":
        recipes[TaskKind.TRANSCRIPTION] = ("transducer",
            (StageKind.ENCODE, StageKind.PREDICT, StageKind.JOINT), "audio-ready-steps")
    return tuple(TaskSpec(task, *recipes[task], output_schema=task.value + "-v1")
                 for task in sorted(tasks, key=lambda t: t.value))


def state_specs(entry):
    # Architecture metadata cannot infer checkpoint layer dimensions or state capabilities.
    # Adapters supply validated per-checkpoint layouts in subsequent PRs.
    return ()
