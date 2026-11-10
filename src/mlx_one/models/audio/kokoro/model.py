"""Native single-utterance Kokoro phoneme-to-waveform execution."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.audio.kokoro.blocks import KokoroAlbert, ProsodyPredictor, TextEncoder
from mlx_one.models.audio.kokoro.vocoder import Decoder


@dataclass
class KokoroOutput:
    waveform: mx.array
    durations: mx.array
    f0: mx.array
    noise: mx.array
    albert_hidden: mx.array
    sample_rate: int = 24000


class KokoroForSpeech(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = KokoroAlbert()
        self.bert_encoder = nn.Linear(768, 512)
        self.predictor = ProsodyPredictor()
        self.text_encoder = TextEncoder()
        self.decoder = Decoder()

    def __call__(self, phonemes, voice, *, speed=1.0, forcing=None):
        if not isinstance(phonemes, str) or not phonemes.strip():
            raise ValueError("Kokoro phonemes must be nonempty text")
        unknown = set(phonemes) - set(self.config.vocab)
        if unknown:
            raise ValueError("unsupported Kokoro phoneme characters: " + repr(sorted(unknown)))
        if len(phonemes) + 2 > self.config.context_length:
            raise ValueError("Kokoro utterance exceeds its 510-phoneme bound")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0.5 <= speed <= 2:
            raise ValueError("Kokoro speed must be between 0.5 and 2")
        if voice.shape != (1, 256):
            raise ValueError("Kokoro voice style must have shape [1,256]")
        ids = mx.array([[0, *[self.config.vocab[p] for p in phonemes], 0]], mx.int32)
        albert = self.bert(ids)
        encoding = self.bert_encoder(albert)
        style = voice[:, 128:]
        duration_features = self.predictor.text_encoder(encoding, style)
        durations = mx.maximum(
            mx.round(
                mx.sum(
                    mx.sigmoid(
                        self.predictor.duration_proj(self.predictor.lstm(duration_features))
                    ),
                    axis=-1,
                )
                / speed
            ),
            1,
        ).astype(mx.int32)
        mx.eval(durations)
        counts = durations[0].tolist()
        if sum(counts) * 600 > self.config.sample_rate * self.config.max_audio_seconds:
            raise ValueError("Kokoro predicted audio exceeds its 30-second bound")
        indices = mx.array([i for i, count in enumerate(counts) for _ in range(count)], mx.int32)
        aligned = duration_features[:, indices, :]
        f0, noise = self.predictor.f0_noise(aligned, style)
        text = self.text_encoder(ids)[:, indices, :]
        waveform = self.decoder(text, f0, noise, voice[:, :128], forcing)
        return KokoroOutput(waveform, durations, f0, noise, albert)
