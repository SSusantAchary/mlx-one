"""Strict configuration for the converted publisher Kokoro-82M v1 checkpoint."""

from dataclasses import dataclass

from mlx_one.core.config import ConfigError


@dataclass(frozen=True)
class KokoroConfig:
    vocab: dict[str, int]
    raw: dict
    model_type: str = "kokoro"
    sample_rate: int = 24000
    context_length: int = 512
    max_audio_seconds: int = 30

    @classmethod
    def from_dict(cls, data):
        if data.get("model_type", "kokoro") != "kokoro":
            raise ConfigError("Kokoro requires its own model type")
        expected = dict(
            hidden_dim=512,
            style_dim=128,
            n_token=178,
            n_layer=3,
            max_dur=50,
            n_mels=80,
            dim_in=64,
            max_conv_dim=512,
            text_encoder_kernel_size=5,
            multispeaker=True,
        )
        if any(data.get(key) != value for key, value in expected.items()):
            raise ConfigError(
                "native Kokoro currently accepts the publisher v1 82M architecture only"
            )
        plbert = data.get("plbert", {})
        if any(
            plbert.get(k) != v
            for k, v in dict(
                hidden_size=768,
                num_attention_heads=12,
                intermediate_size=2048,
                num_hidden_layers=12,
                max_position_embeddings=512,
            ).items()
        ):
            raise ConfigError("unsupported Kokoro ALBERT architecture")
        dsp = data.get("istftnet", {})
        if (
            dsp.get("upsample_rates") != [10, 6]
            or dsp.get("gen_istft_n_fft") != 20
            or dsp.get("gen_istft_hop_size") != 5
        ):
            raise ConfigError("unsupported Kokoro vocoder/STFT layout")
        if any(
            dsp.get(key) != value
            for key, value in {
                "upsample_kernel_sizes": [20, 12],
                "upsample_initial_channel": 512,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5]] * 3,
            }.items()
        ):
            raise ConfigError("unsupported Kokoro vocoder residual architecture")
        vocab = data.get("vocab")
        if (
            not isinstance(vocab, dict)
            or not vocab
            or any(
                not isinstance(k, str) or len(k) != 1 or type(v) is not int or not 0 < v < 178
                for k, v in vocab.items()
            )
        ):
            raise ConfigError("invalid Kokoro phoneme vocabulary")
        return cls(vocab, dict(data))
