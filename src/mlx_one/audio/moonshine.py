"""Original Moonshine offline inference; no implicit waveform sample rate."""

from __future__ import annotations

import resource
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlx_one.audio.schemas import ASRDecodeOptions, SpeechTranscript
from mlx_one.models.audio.moonshine.config import encoded_length
from mlx_one.models.audio.moonshine.loading import LoadedMoonshine
from mlx_one.models.audio.whisper.processing import decode_audio

BUDGET = 24 * 1024**3


class AudioResourceLimitError(RuntimeError):
    """An audio request exceeds its memory or explicit decoding budget."""


def transcribe_moonshine(
    bundle: LoadedMoonshine,
    audio: Any,
    *,
    model_name: str,
    sample_rate: int | None,
    language: str | None,
    task: str,
    word_timestamps: bool,
    options: ASRDecodeOptions | None,
    cancelled: Callable[[], bool] | None,
    started: float,
    loaded_at: float,
) -> SpeechTranscript:
    if task != "transcribe" or word_timestamps or language not in {None, "en", "english"}:
        raise ValueError(
            "original Moonshine supports English transcription without timestamps only"
        )
    if options is not None and not isinstance(options, ASRDecodeOptions):
        raise ValueError(
            "Moonshine requires ASRDecodeOptions, not Whisper decoding/fallback options"
        )
    is_file = isinstance(audio, (str, Path))
    if sample_rate is not None and (isinstance(sample_rate, bool) or sample_rate != 16000):
        raise ValueError("Moonshine waveform sample_rate must be 16000; resample explicitly")
    if not is_file and sample_rate is None:
        raise ValueError("Moonshine waveform inputs require an explicit sample_rate=16000")

    def check() -> None:
        if cancelled is not None and cancelled():
            raise RuntimeError("transcription was cancelled")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET:
            raise AudioResourceLimitError("24 GiB audio process-memory budget exceeded")

    check()
    import mlx.core as mx

    waveform = (
        decode_audio(audio, cancelled=cancelled) if is_file else mx.array(audio, dtype=mx.float32)
    )
    if waveform.ndim != 1 or not mx.all(mx.isfinite(waveform)).item():
        raise ValueError("Moonshine waveform must contain finite mono samples")
    duration = waveform.shape[0] / 16000
    tokens: list[int] = []
    processed_at = time.perf_counter()
    caches = ()
    try:
        if waveform.shape[0]:
            length = encoded_length(waveform.shape[0])
            if length < 1:
                raise ValueError(
                    "nonempty Moonshine audio requires at least 895 samples; no implicit padding"
                )
            # Conservative dense-attention working-set bound, not input truncation.
            working = 5 * 4 * bundle.model.config.encoder_num_attention_heads * length**2
            if working + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET:
                raise AudioResourceLimitError(
                    "Moonshine dense encoder would exceed 24 GiB process budget"
                )
            encoded = bundle.model.encode(waveform[None, :])
            mx.eval(encoded)
            processed_at = time.perf_counter()
            opts = options or ASRDecodeOptions()
            limit = opts.max_tokens or (bundle.generation_config.get("max_length", 194) - 1)
            caches = bundle.model.make_cache()
            input_ids = mx.array([[bundle.model.config.decoder_start_token_id]])
            for _ in range(limit):
                check()
                output = bundle.model.decode(input_ids, encoded, cache=caches)
                token_id = int(mx.argmax(output.logits[:, -1, :], axis=-1).item())
                if token_id == bundle.model.config.eos_token_id:
                    break
                tokens.append(token_id)
                input_ids = mx.array([[token_id]])
            else:
                raise AudioResourceLimitError(
                    f"Moonshine did not reach EOS within {limit} tokens; "
                    "refusing a truncated transcript"
                )
            check()
    finally:
        for cache in caches:
            cache.reset()
        # Request-owned caches only: never clear the shared MLX allocator/model.
    completed = time.perf_counter()
    return SpeechTranscript(
        bundle.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        duration,
        model_name,
        bundle.revision,
        16000,
        tokens=tuple(tokens),
        language="en" if language is not None else None,
        timings={
            "load_seconds": loaded_at - started,
            "preprocess_seconds": processed_at - loaded_at,
            "decode_seconds": completed - processed_at,
            "wall_seconds": completed - started,
        },
        provenance={
            "model_type": "moonshine",
            "checkpoint_path": str(bundle.path),
            "checkpoint_revision": bundle.revision,
            "decoding": "greedy",
            "weight_dtype": str(bundle.model.model.encoder.conv1.weight.dtype),
        },
    )
