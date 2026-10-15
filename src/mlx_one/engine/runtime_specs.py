"""Lazy built-in bindings; integration and checkpoint qualification remain distinct."""
from mlx_one.engine.adapter_registry import RUNTIME_CONTRACT_VERSION, RuntimeRegistration
from mlx_one.engine.contracts import StageKind, TaskKind, TaskSpec, canonical_task

BUILTIN_ADAPTERS = {name: "mlx_one.engine.native_adapters:" + adapter for name, adapter in (
    ("gpt2", "GPT2Adapter"), ("llama", "LlamaAdapter"),
    ("qwen2", "Qwen2Adapter"), ("qwen3", "Qwen3Adapter"),
    ("openelm", "OpenELMAdapter"), ("qwen2_moe", "Qwen2MoeAdapter"),
    ("lfm2", "LFM2Adapter"), ("lfm2_moe", "LFM2MoeAdapter"),
    ("qwen3_5", "Qwen35Adapter"))}

BUILTIN_ADAPTERS.update({name:"mlx_one.engine.task_runners.retrieval:"+adapter for name,adapter in (
    ("bert","BertEmbeddingAdapter"),("mpnet","MPNetEmbeddingAdapter"),
    ("qwen3_embedding","Qwen3EmbeddingAdapter"),("qwen3_reranker","Qwen3RerankAdapter"),
    ("lfm2_colbert","LFM2ColBERTAdapter"))})

VISION_ADAPTERS={name:"mlx_one.engine.task_runners.multimodal:"+adapter for name,adapter in (
    ("qwen2_vl","Qwen2VLAdapter"),("qwen2_5_vl","Qwen25VLAdapter"),
    ("lfm2_vl","LFM2VLAdapter"),("qwen3_5","Qwen35VisionAdapter"))}
BUILTIN_ADAPTERS.update({k:v for k,v in VISION_ADAPTERS.items() if k!="qwen3_5"})

BUILTIN_ADAPTERS.update({name:"mlx_one.engine.task_runners.speech:"+adapter
    for name,adapter in (("whisper","WhisperAdapter"),("moonshine","MoonshineAdapter"),
    ("qwen3_asr","Qwen3ASRAdapter"),("qwen3_forced_aligner","Qwen3AlignmentAdapter"))})

BUILTIN_ADAPTERS["parakeet_tdt"]=(
    "mlx_one.engine.task_runners.transducer:ParakeetTDTAdapter")

BUILTIN_ADAPTERS.update({name:"mlx_one.engine.task_runners.segmentation:"+adapter
    for name,adapter in (("sam3","SAM3ConceptAdapter"),
        ("sam3_tracker","SAM3InteractiveAdapter"),("sam3_video","SAM3VideoAdapter"))})


def builtin_runtime(entry):
    component = entry.model_type == "sam3_tracker_video"
    status = ("component_only" if component else "architecture_only" if not entry.loader_path
              and not entry.tasks else "not_integrated" if not entry.loader_path
              else "loader_declared")
    adapter = BUILTIN_ADAPTERS.get(entry.model_type)
    if adapter is not None:
        status = "integrated"
    reason = ("Native task service adapter; checkpoint qualification is separate" if adapter
              else "Nested video tracker; no standalone task loader" if component
              else "Architecture only; loader and task integration required"
              if status == "architecture_only" else "Shared task-service integration unavailable")
    return RuntimeRegistration(entry.model_type, adapter,
        "mlx_one.engine.runtime_specs:task_specs", "mlx_one.engine.runtime_specs:state_specs",
        RUNTIME_CONTRACT_VERSION, component_only=component, integration_status=status,
        parent_model_type="sam3_video" if component else None,
        reason=reason,task_adapters=((TaskKind.VISION_LANGUAGE,VISION_ADAPTERS[entry.model_type]),)
        if entry.model_type in VISION_ADAPTERS else ())


def task_specs(entry):
    tasks = {canonical_task(task) for task in entry.tasks}
    # The text loader cannot establish multimodal preparation or outputs.
    if entry.model_type == "lfm2_colbert":
        tasks.discard(TaskKind.EMBEDDING)
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
