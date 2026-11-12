"""Bounded T5 and ViT adapters on the existing native task actor."""

from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.task_runners.base import NativeTaskAdapter
from mlx_one.engine.tasks import PreparedTask


class T5Adapter(NativeTaskAdapter):
    family = "t5"
    supported_tasks = (TaskKind.TEXT_TO_TEXT,)

    def __init__(self, bundle):
        if bundle.architecture != self.family:
            raise ValueError("T5 adapter requires a native T5 bundle")
        self.bundle = bundle

    def prepare(self, task, payload):
        if TaskKind(task) != TaskKind.TEXT_TO_TEXT:
            raise ValueError("T5 exposes text-to-text-generation, not causal chat")
        if set(payload) - {"texts", "max_length", "max_new_tokens"}:
            raise ValueError("unsupported T5 generation setting")
        texts = payload.get("texts")
        values = (texts,) if isinstance(texts, str) else tuple(texts or ())
        if not 1 <= len(values) <= 8 or any(
            not isinstance(v, str) or not v.strip() for v in values
        ):
            raise ValueError("T5 requires one to eight nonempty texts")
        cap = self.bundle.model.config.n_positions
        limit, new = payload.get("max_length", cap), payload.get("max_new_tokens", 32)
        if (
            type(limit) is not int
            or not 2 <= limit <= cap
            or type(new) is not int
            or not 1 <= new < cap
        ):
            raise ValueError("T5 token bounds exceed configured capacities")
        lengths = tuple(
            len(self.bundle.tokenizer.encode(v, add_special_tokens=True, max_length=limit))
            for v in values
        )
        transient = len(values) * (limit + new) ** 2 * self.bundle.model.config.num_heads * 16
        return PreparedTask(
            TaskKind.TEXT_TO_TEXT,
            dict(texts=values, max_length=limit, max_new_tokens=new),
            lengths,
            (new,),
            transient,
        )

    def run_batch(self, prepared):
        from mlx_one.seq2seq import generate

        return tuple(generate(self.bundle, **request.payload) for request in prepared)


class ViTAdapter(NativeTaskAdapter):
    family = "vit"
    supported_tasks = (TaskKind.IMAGE_CLASSIFICATION,)

    def __init__(self, bundle):
        if bundle.architecture != self.family:
            raise ValueError("ViT adapter requires a supervised native image classifier")
        self.bundle = bundle

    def prepare(self, task, payload):
        if TaskKind(task) != TaskKind.IMAGE_CLASSIFICATION:
            raise ValueError(
                "ViT exposes supervised image classification, not CLIP candidate scoring"
            )
        if set(payload) - {"images", "top_k"}:
            raise ValueError("unsupported ViT request option")
        images = payload.get("images")
        values = images if isinstance(images, (list, tuple)) else [images]
        top = payload.get("top_k", 5)
        if (
            not 1 <= len(values) <= 8
            or type(top) is not int
            or not 1 <= top <= self.bundle.model.config.num_labels
        ):
            raise ValueError("invalid ViT batch size or top_k")
        size = self.bundle.model.config.image_size
        patches = (size // self.bundle.model.config.patch_size) ** 2 + 1
        return PreparedTask(
            TaskKind.IMAGE_CLASSIFICATION,
            dict(images=values, top_k=top),
            tuple(patches for _ in values),
            (size,),
            len(values) * 64 * 1024**2,
        )

    def run_batch(self, prepared):
        from mlx_one.classification.image import classify_image

        return tuple(classify_image(self.bundle, **request.payload) for request in prepared)


class KokoroAdapter(NativeTaskAdapter):
    family = "kokoro"
    supported_tasks = (TaskKind.SPEECH_GENERATION,)

    def __init__(self, bundle):
        if bundle.architecture != self.family:
            raise ValueError("Kokoro adapter requires a native Kokoro bundle")
        self.bundle = bundle

    def prepare(self, task, payload):
        if TaskKind(task) != TaskKind.SPEECH_GENERATION:
            raise ValueError("Kokoro supports speech-generation only")
        if set(payload) - {"text", "voice", "language", "phonemes", "speed"}:
            raise ValueError("unsupported Kokoro generation setting")
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 8192:
            raise ValueError("Kokoro needs a bounded nonempty utterance")
        return PreparedTask(
            TaskKind.SPEECH_GENERATION,
            dict(payload),
            (len(text),),
            (payload.get("voice", "af_heart"),),
            # The bounded 30-second profile peaked near 3.1 GB including weights.
            # Reserve vocoder working memory separately from resident parameters.
            3 * 1024**3,
        )

    def run_batch(self, prepared):
        from mlx_one.audio.kokoro import synthesize

        return tuple(synthesize(self.bundle, **request.payload) for request in prepared)
