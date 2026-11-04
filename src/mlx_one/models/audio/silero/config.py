"""Backend-free configuration for the pinned Silero 16-kHz checkpoint."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SileroConfig:
    model_type: str = "silero_vad"
    sample_rate: int = 16000
    chunk_samples: int = 512
    context_samples: int = 64
    decoder_hidden_size: int = 128
    num_decoder_layers: int = 1

    @classmethod
    def from_dict(cls, payload):
        if payload and any(payload.get(k, v) != v for k, v in cls().__dict__.items()):
            raise ValueError("only the pinned 16-kHz Silero configuration is admitted")
        return cls()
