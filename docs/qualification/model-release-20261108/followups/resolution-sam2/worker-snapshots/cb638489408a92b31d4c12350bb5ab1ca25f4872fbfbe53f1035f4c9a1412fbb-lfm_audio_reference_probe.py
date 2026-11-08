"""Isolated LFM audio diagnosis: independently prepared inputs and forced history.

The reference branch imports only the pinned MLX-Audio implementation. Forced
history isolates numerical/model/codec differences from stochastic branching.
This diagnostic does not itself certify speech intelligibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    import mlx.core as mx
    import numpy as np
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "reference"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mx.set_cache_limit(256 * 1024**2)
    waveform, rate = sf.read(args.audio, dtype="float32")
    assert rate == 16000 and waveform.ndim == 1
    native = args.backend == "native"
    if native:
        from mlx_one.audio.lfm import _prompt
        from mlx_one.models.audio.lfm2_audio.dsp import preprocess
        from mlx_one.models.audio.lfm2_audio.loading import load_lfm_audio

        bundle = load_lfm_audio(args.model, offline=True)
        model = bundle.model
        features = preprocess(mx.array(waveform), model.config.preprocessor)
        embeddings = _prompt(bundle, "speech-to-speech", waveform=waveform)
        events = []
        for kind, value in model.generate(
            embeddings,
            mode="interleaved",
            max_tokens=1024,
            audio_temperature=1.0,
            audio_top_k=4,
            seed=0,
        ):
            events.append([kind, int(value) if kind in {"text", "end"} else value.tolist()])
        args.trace.write_text(json.dumps(events) + "\n")
        cache = model.make_cache()
        hidden, _, _ = model.lfm(None, cache=cache, input_embeddings=embeddings)
        codec = bundle.codec
    else:
        from mlx_audio.sts.models.lfm_audio.model import LFM2AudioModel
        from mlx_audio.sts.models.lfm_audio.processor import ChatState, LFM2AudioProcessor
        from tokenizers import Tokenizer

        model = LFM2AudioModel.from_pretrained(str(args.model))

        class Tokens:
            bos_token_id = 1

            def __init__(self):
                self.tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))

            def encode(self, value, *, add_special_tokens=False):
                return self.tokenizer.encode(value, add_special_tokens=add_special_tokens).ids

        processor = LFM2AudioProcessor(model.config, tokenizer=Tokens())
        processor.model_path = args.model
        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text("Respond with interleaved text and audio.")
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(mx.array(waveform), sample_rate=16000)
        chat.end_turn()
        chat.new_turn("assistant")
        features = chat.get_audio_features()
        embeddings = model._build_interleaved_embeddings(
            chat.get_text_tokens(), features, None, chat.get_modalities()
        )
        hidden, cache = model._prefill(
            text_tokens=chat.get_text_tokens(),
            audio_features=features,
            modalities=chat.get_modalities(),
        )
        events = json.loads(args.trace.read_text())
        codec = processor.detokenizer
    arrays = dict(
        features=np.asarray(features.astype(mx.float32)),
        input_embeddings=np.asarray(embeddings.astype(mx.float32)),
    )
    hiddens, text_logits, audio_logits, spans, current = [], [], [], [], []
    for step, (kind, value) in enumerate(events):
        last = hidden[:, -1:]
        hiddens.append(np.asarray(last.astype(mx.float32)))
        if kind in {"text", "end"}:
            logits = model.lfm.embed_tokens.as_linear(last)
            text_logits.append(np.asarray(logits.astype(mx.float32)))
            if kind == "end":
                break
            next_embedding = model.lfm.embed_tokens(mx.array([[value]]))
        else:
            frame = mx.array(value)
            projected = model.depth_linear(last).reshape(1, 1, model.config.codebooks, -1)
            previous = mx.zeros(projected.shape[:2] + (projected.shape[-1],), dtype=projected.dtype)
            depth_cache, frame_logits = None, []
            for index, embedding in enumerate(model.depth_embeddings):
                depth, depth_cache = model.audio_head.depthformer(
                    projected[:, :, index] + previous, cache=depth_cache, use_cache=True
                )
                frame_logits.append(
                    np.asarray(embedding.get_logits(depth[:, -1]).astype(mx.float32))
                )
                previous = embedding.embed(frame[:, index])[:, None]
            audio_logits.append(np.stack(frame_logits))
            next_embedding = (
                model.audio_embedding(frame) if native else model._embed_audio_out(frame)
            )[:, None]
            if kind == "audio":
                current.append(frame)
            elif current:
                spans.append(current)
                current = []
        if native:
            hidden, _, _ = model.lfm(None, cache=cache, input_embeddings=next_embedding)
        else:
            hidden = model.lfm(None, cache=cache, input_embeddings=next_embedding)
        if step % 32 == 0:
            mx.eval(hidden)
            print(
                json.dumps(dict(stage="forced_history", backend=args.backend, step=step)),
                flush=True,
            )
    if current:
        spans.append(current)
    waves = []
    for span in spans:
        waves.append(np.asarray(codec(mx.stack(span, axis=-1)).reshape(-1).astype(mx.float32)))
    arrays.update(
        hidden_steps=np.stack(hiddens),
        text_logits=np.stack(text_logits),
        audio_logits=np.stack(audio_logits),
        codec_waveform=np.concatenate(waves),
    )
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(json.dumps(dict(stage="saved", backend=args.backend, events=len(events))), flush=True)


if __name__ == "__main__":
    main()
