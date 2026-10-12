"""Validation-only upstream accumulated-audio wrapper around pinned Torch SDPA.

Qwen publishes the wrapper through vLLM only. This reproduces its rollback and
buffer algorithm with the pinned Transformers decoder, not vLLM-kernel parity.
Adapted from QwenLM/Qwen3-ASR, Apache-2.0, commit
7c6daf77a2421100f5fb066495372c00129d39ff. Copyright 2026 The Alibaba Qwen team.
"""

from __future__ import annotations

import time


def reference_stream(model, waveform, language: str, chunk_samples: int) -> list[dict]:
    import torch
    from qwen_asr.inference.utils import parse_asr_output

    if len(waveform) > 480000:
        raise ValueError("qualification utterance exceeds 30 seconds; no segmentation")
    received, consumed, chunk_id, raw = 0, 0, 0, ""
    events = []
    started = time.perf_counter()

    def decode(end: int, tail: bool):
        nonlocal chunk_id, raw
        prefix = ""
        if chunk_id >= 2:
            ids = model.processor.tokenizer.encode(raw)
            if tail:
                prefix = model.processor.tokenizer.decode(ids[: max(1, len(ids) - 5)])
            else:
                rollback = 5
                while True:
                    end_id = max(0, len(ids) - rollback)
                    prefix = model.processor.tokenizer.decode(ids[:end_id]) if end_id else ""
                    if "\ufffd" not in prefix or not end_id:
                        break
                    rollback += 1
        prompt = model._build_text_prompt("", language) + prefix
        inputs = model.processor(
            text=prompt, audio=waveform[:end], padding=True, return_tensors="pt"
        ).to(dtype=torch.bfloat16)
        with torch.inference_mode():
            output = model.model.generate(**inputs, max_new_tokens=512)
        text = model.processor.tokenizer.decode(
            output.sequences[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw = prefix + text
        chunk_id += 1

    for start in range(0, len(waveform), chunk_samples):
        received += min(chunk_samples, len(waveform) - start)
        while received - consumed >= 32000:
            consumed += 32000
            decode(consumed, False)
            detected, text = parse_asr_output(raw, user_language=language)
            events.append(
                {
                    "kind": "partial",
                    "text": text,
                    "language": detected or None,
                    "audio_seconds": consumed / 16000,
                    "sequence": len(events),
                    "latency_seconds": time.perf_counter() - started,
                }
            )
    if consumed < len(waveform):
        decode(len(waveform), True)
    detected, text = parse_asr_output(raw, user_language=language)
    events.append(
        {
            "kind": "final",
            "text": text,
            "language": detected or None,
            "audio_seconds": len(waveform) / 16000,
            "sequence": len(events),
            "latency_seconds": time.perf_counter() - started,
        }
    )
    return events
