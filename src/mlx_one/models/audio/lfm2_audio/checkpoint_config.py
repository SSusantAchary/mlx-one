# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass, field, fields
from typing import Any

from mlx_one.models.language.lfm2.config import Lfm2Config as LFM2Config


@dataclass
class PreprocessorConfig:
    """Audio preprocessor configuration for mel spectrogram extraction."""

    sample_rate: int = 16000
    normalize: str = "per_feature"
    window_size: float = 0.025
    window_stride: float = 0.01
    window: str = "hann"
    features: int = 128
    n_fft: int = 512
    log: bool = True
    frame_splicing: int = 1
    dither: float = 1e-05
    pad_to: int = 0
    pad_value: float = 0.0
    preemph: float = 0.97  # Pre-emphasis high-pass filter coefficient

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate * self.window_stride)

    @property
    def win_length(self) -> int:
        return int(self.sample_rate * self.window_size)


@dataclass
class ConformerEncoderConfig:
    """FastConformer audio encoder configuration."""

    feat_in: int = 128
    feat_out: int = -1
    n_layers: int = 17
    d_model: int = 512
    subsampling: str = "dw_striding"
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    causal_downsampling: bool = False
    reduction: str | None = None
    reduction_position: int | None = None
    reduction_factor: int = 1
    ff_expansion_factor: int = 4
    self_attention_model: str = "rel_pos"
    n_heads: int = 8
    att_context_size: list[int] = field(default_factory=lambda: [-1, -1])
    xscaling: bool = False
    untie_biases: bool = True
    pos_emb_max_len: int = 5000
    conv_kernel_size: int = 9
    conv_norm_type: str = "batch_norm"
    conv_context_size: int | None = None
    dropout: float = 0.1
    dropout_pre_encoder: float = 0.1
    dropout_emb: float = 0.0
    dropout_att: float = 0.1


@dataclass
class DepthformerConfig:
    """Depthformer configuration for audio frame generation."""

    layers: int = 6
    dim: int = 1024
    num_heads: int = 32  # Q attention heads
    num_kv_heads: int = 8  # K/V attention heads (GQA)
    rope_theta: float = 1000000.0  # reference MHA default; not stored in config.json
    tie: bool = True


@dataclass
class LFM2AudioConfig:
    """Complete LFM2.5-Audio model configuration."""

    # Model architecture
    model_type: str = "lfm_audio"
    sample_rate: int = 24000
    codebooks: int = 8
    tie_audio_embeddings: bool = False
    semantic_codebook_factor: int = 100
    codebook_weight: str = "log"
    audio_vocab_size: int = 2049  # 2048 + 1 for padding

    # Interleaved generation config
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12

    # Sub-module configs
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    encoder: ConformerEncoderConfig = field(default_factory=ConformerEncoderConfig)
    lfm: LFM2Config = field(default_factory=LFM2Config)
    depthformer: DepthformerConfig = field(default_factory=DepthformerConfig)

    # MLP adapter config (conformer output -> LFM input)
    # Checkpoint uses: LayerNorm(512) -> Linear(512, 2048) -> GELU -> Linear(2048, 2048)
    adapter_hidden_dims: list[int] = field(default_factory=lambda: [2048])
    adapter_dropout: float = 0.0
    adapter_use_layer_norm: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "LFM2AudioConfig":
        """Create config from dictionary."""
        # Parse sub-configs
        preprocessor = PreprocessorConfig(**config_dict.get("preprocessor", {}))
        encoder = ConformerEncoderConfig(**config_dict.get("encoder", {}))
        lfm = LFM2Config.from_dict(config_dict.get("lfm", {}))
        depthformer = DepthformerConfig(**config_dict.get("depthformer", {}))

        # Remove nested dicts
        config_dict = {
            k: v
            for k, v in config_dict.items()
            if k
            not in (
                "preprocessor",
                "encoder",
                "lfm",
                "depthformer",
                "architectures",
                "quantization",
                "quantization_config",
            )
        }

        known = {f.name for f in fields(cls)}
        return cls(
            preprocessor=preprocessor,
            encoder=encoder,
            lfm=lfm,
            depthformer=depthformer,
            **{k: v for k, v in config_dict.items() if k in known},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict

        return asdict(self)
