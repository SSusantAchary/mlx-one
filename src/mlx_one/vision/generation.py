"""Native media prefill with transactional, position-aware reference decoding."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from types import SimpleNamespace

from mlx_one.engine.cache import CacheBundle
from mlx_one.text.schemas import GenerationChunk, TextGenerationOptions
from mlx_one.vision.loading import LoadedVLM


def _media_kwargs(prepared):
    if prepared.pixel_values is None:
        return {}
    if hasattr(prepared, "spatial_shapes"):
        return {"pixel_values": prepared.pixel_values, "spatial_shapes": prepared.spatial_shapes}
    return {
        "pixel_values": prepared.pixel_values,
        "image_grid_thw": prepared.image_grid_thw,
        "mm_token_type_ids": prepared.mm_token_type_ids,
    }


class VisionForward:
    """One request owns media identity, multimodal positions and complete model state."""

    def __init__(self, bundle, prepared, cancelled):
        self.bundle, self.prepared, self.cancelled = bundle, prepared, cancelled
        self.delta = None
        self.request_id = "vision-" + uuid.uuid4().hex
        self.text_bundle = None

    def __getattr__(self, name):
        return getattr(self.bundle.model, name)

    def __call__(self, ids, *, cache):
        import mlx.core as mx

        from mlx_one.engine.batching import ForwardRequest
        from mlx_one.engine.transactions import commit_evaluated

        state = CacheBundle(cache)
        draft = state.clone()
        initial = cache[0].offset == 0
        try:
            kwargs = _media_kwargs(self.prepared) if initial else {}
            if not initial and self.delta is not None:
                base = (
                    mx.arange(cache[0].offset, cache[0].offset + ids.shape[1])[None, :] + self.delta
                )
                kwargs["position_ids"] = mx.stack((base, base, base), axis=0)
            output = self.bundle.model(ids, cache=draft.entries, logits_mode="last_only", **kwargs)
            mx.eval(output.logits)
            request = ForwardRequest(
                self.text_bundle,
                tuple(int(t) for t in ids[0].tolist()),
                cache,
                "prefill" if initial else "decode",
                (),
                request_id=self.request_id,
                cancel=SimpleNamespace(is_set=self.cancelled),
            )
            commit_evaluated((request,), (draft,))
            if initial:
                self.delta = getattr(output, "rope_deltas", None)
            return output
        finally:
            draft.release()


def _stream_vlm(
    bundle,
    prompt,
    images,
    options,
    cancelled,
    *,
    yield_steps=False,
    observer=None,
    context_length=None,
    cache_type_k="f16",
    cache_type_v="f16",
):
    from mlx_one.text.generation import _stream_bundle
    from mlx_one.text.loading import LoadedTextModel

    if cancelled():
        return
    started = time.perf_counter()
    prepared = bundle.processor.prepare(prompt, tuple(images))
    import mlx.core as mx

    mx.eval(
        prepared.input_ids, prepared.pixel_values
    ) if prepared.pixel_values is not None else mx.eval(prepared.input_ids)
    preprocessing_ms = (time.perf_counter() - started) * 1000
    if cancelled():
        return
    capacity = min(context_length or bundle.context_length, bundle.context_length)
    tokens = tuple(int(t) for t in prepared.input_ids[0].tolist())
    if observer is not None:
        observer(
            {
                "kind": "prepared",
                "tokens": tokens,
                "image_metadata": prepared.image_metadata,
                "preprocessing_ms": preprocessing_ms,
            }
        )
    if len(tokens) + options.max_tokens > capacity:
        raise ValueError("multimodal prompt plus output exceeds the checkpoint context")
    from mlx_one.engine.cache import array_nbytes
    from mlx_one.engine.media import FeatureStore

    features = FeatureStore()
    with ExitStack() as resources:
        resources.callback(features.close)
        lease = features.acquire(
            "private:" + uuid.uuid4().hex,
            getattr(bundle, "processor_identity", None) or "private-native-media",
            lambda: prepared,
            array_nbytes(vars(prepared)),
        )
        resources.callback(lease.close)
        wrapper = VisionForward(bundle, lease.value, cancelled)
        text = LoadedTextModel(
            wrapper,
            bundle.tokenizer,
            bundle.path,
            bundle.model_id,
            bundle.revision,
            bundle.architecture,
            capacity,
            eos_token_ids=bundle.eos_token_ids,
        )
        wrapper.text_bundle = text

        def release_wrapper():
            # LoadedTextModel refers back to this wrapper; break that request cycle.
            wrapper.text_bundle = None
            wrapper.prepared = None
            wrapper.delta = None

        resources.callback(release_wrapper)
        from mlx_one.core.cache import configure_kv_caches

        state = CacheBundle(
            configure_kv_caches(bundle.model.make_cache(), cache_type_k, cache_type_v)
        )

        def release_state():
            state.release()
            if observer is not None:
                observer({"kind": "cleanup", "closed": state.closed, "cache_bytes": state.nbytes})

        resources.callback(release_state)
        yield from _stream_bundle(
            text,
            prompt,
            options,
            cancelled,
            cache=state.entries,
            yield_steps=yield_steps,
            _prepared_input_ids=tokens,
            on_execution=observer,
        )


def stream_vlm(
    bundle: LoadedVLM,
    prompt: str,
    images: Sequence[object],
    *,
    options: TextGenerationOptions | None = None,
) -> Iterator[GenerationChunk]:
    """Preserve native generation events; no shared media prefix or backend substitution."""
    yield from _stream_vlm(
        bundle, prompt, images, options or TextGenerationOptions(), lambda: False
    )
