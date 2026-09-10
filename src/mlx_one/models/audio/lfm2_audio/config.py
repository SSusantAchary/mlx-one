"""Nested configuration for LFM2.5-Audio."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_divisible, require_positive
from mlx_one.models.language.lfm2.config import Lfm2Config


@dataclass(frozen=True)
class AudioPreprocessorConfig:
    sampling_rate: int = 16000
    n_fft: int = 512
    window_length: int = 400
    hop_length: int = 160
    num_mel_bins: int = 128
    normalization: str = "per_feature"

    def __post_init__(self) -> None:
        for name in ("sampling_rate", "n_fft", "window_length", "hop_length", "num_mel_bins"):
            require_positive(name, getattr(self, name))
        if self.window_length > self.n_fft:
            raise ConfigError("window_length cannot exceed n_fft")
        if self.normalization not in {"none", "per_feature"}:
            raise ConfigError(f"unsupported audio normalization: {self.normalization}")


@dataclass(frozen=True)
class ConformerConfig:
    input_dim: int = 128
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 17
    num_attention_heads: int = 8
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            require_positive(name, getattr(self, name))
        require_divisible(
            "hidden_size", self.hidden_size, "num_attention_heads", self.num_attention_heads
        )
        if self.conv_kernel_size % 2 == 0:
            raise ConfigError("Conformer conv_kernel_size must be odd")


@dataclass(frozen=True)
class DepthformerConfig:
    num_hidden_layers: int = 6
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "intermediate_size",
        ):
            require_positive(name, getattr(self, name))
        require_divisible(
            "hidden_size", self.hidden_size, "num_attention_heads", self.num_attention_heads
        )


@dataclass(frozen=True)
class AudioDetokenizerConfig:
    codebook_size: int = 2049
    num_codebooks: int = 8
    hidden_size: int = 1024
    num_hidden_layers: int = 6
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    upsample_factor: int = 6
    n_fft: int = 1280
    hop_length: int = 320
    sliding_window: int = 30

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            require_positive(name, getattr(self, name))
        require_divisible(
            "hidden_size", self.hidden_size, "num_attention_heads", self.num_attention_heads
        )


def _nested(cls: type[Any], value: Any) -> Any:
    if isinstance(value, cls):
        return value
    if value is None:
        return cls()
    if not isinstance(value, Mapping):
        raise ConfigError(f"{cls.__name__} must be an object")
    known = set(cls.__dataclass_fields__)
    return cls(**{key: item for key, item in value.items() if key in known})


@dataclass(frozen=True)
class Lfm2AudioConfig:
    model_type: str
    text_config: Lfm2Config
    audio_config: ConformerConfig = field(default_factory=ConformerConfig)
    depthformer_config: DepthformerConfig = field(default_factory=DepthformerConfig)
    detokenizer_config: AudioDetokenizerConfig = field(default_factory=AudioDetokenizerConfig)
    preprocessor_config: AudioPreprocessorConfig = field(default_factory=AudioPreprocessorConfig)
    audio_vocab_size: int = 2049
    num_codebooks: int = 8
    audio_eos_token_id: int = 2048
    modality_flag_ids: Sequence[int] = (1, 2)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type not in {"lfm2_audio", "lfm2-audio"}:
            raise ConfigError(f"Lfm2AudioConfig cannot represent model_type={self.model_type!r}")
        require_positive("audio_vocab_size", self.audio_vocab_size)
        require_positive("num_codebooks", self.num_codebooks)
        if not 0 <= self.audio_eos_token_id < self.audio_vocab_size:
            raise ConfigError("audio_eos_token_id must be within audio vocabulary")
        if self.audio_config.input_dim != self.preprocessor_config.num_mel_bins:
            raise ConfigError("Conformer input_dim must match num_mel_bins")
        if self.depthformer_config.hidden_size != self.text_config.hidden_size:
            raise ConfigError("Depthformer hidden_size must match text hidden_size")
        if self.detokenizer_config.num_codebooks != self.num_codebooks:
            raise ConfigError("detokenizer num_codebooks must match num_codebooks")
        object.__setattr__(self, "modality_flag_ids", tuple(self.modality_flag_ids))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Lfm2AudioConfig:
        text = data.get("text_config")
        if not isinstance(text, Mapping):
            raise ConfigError("LFM2-Audio requires text_config")
        text = dict(text)
        text["model_type"] = "lfm2"
        known = set(cls.__dataclass_fields__) - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["model_type"] = str(data.get("model_type", "lfm2_audio")).replace("-", "_")
        payload["text_config"] = Lfm2Config.from_dict(text)
        payload["audio_config"] = _nested(ConformerConfig, data.get("audio_config"))
        payload["depthformer_config"] = _nested(DepthformerConfig, data.get("depthformer_config"))
        payload["detokenizer_config"] = _nested(
            AudioDetokenizerConfig, data.get("detokenizer_config")
        )
        payload["preprocessor_config"] = _nested(
            AudioPreprocessorConfig, data.get("preprocessor_config")
        )
        payload["extra"] = extras(data, known)
        return cls(**payload)
