# Native MLX port of mlx-vlm/deepseekocr/config.py
# Reference revision: e9af376626b04ba0cdad23b1476fd7a4a583d721
# See ../UPSTREAM.md and ../UPSTREAM_LICENSE for attribution and local changes.
import inspect
from dataclasses import dataclass

from mlx_one.models.ocr._layers import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "deepseek_v2"
    vocab_size: int = 102400
    hidden_size: int = 1280
    intermediate_size: int = 6848
    moe_intermediate_size: int = 896
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: int | None = 2
    n_routed_experts: int | None = 64
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 0
    v_head_dim: int = 128
    qk_nope_head_dim: int = 0
    topk_method: str = "greedy"
    n_group: int | None = 1
    topk_group: int | None = 1
    num_experts_per_tok: int | None = 6
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: dict = None
    attention_bias: bool = False
    scoring_func: str = "softmax"
    attn_type: str = "DeepseekV2Attention"

    def __post_init__(self):
        if self.qk_nope_head_dim == 0:
            self.attn_type = "LlamaAttention"

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str
    layers: int = 24
    width: int = 1152
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    mlp_ratio: float = 3.7362
    cls: str = None
    params: dict = None


@dataclass
class MLPConfig(BaseModelConfig):
    hidden_size: int
    intermediate_size: int
    hidden_act: str = "gelu"


@dataclass
class ProjectorConfig(BaseModelConfig):
    projector_type: str = "linear"
    input_dim: int = 2048
    n_embed: int = 1280
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False


@dataclass
class SAMViTConfig(BaseModelConfig):
    image_size: tuple[int, int] | int = 1024
    width: int = 768
    layers: int = 12
    heads: int = 12
    patch_size: int = 16
    window_size: int = 14
    prompt_embed_dim: int = 256
    global_attn_indexes: list[int] | tuple[int] = (2, 5, 8, 11)
    downsample_channels: list[int] | tuple[int] = (512, 1024)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    projector_config: ProjectorConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 128815
    vision_feature_select_strategy: str = "default"
    select_layer: int = -1
    pad_id: int = 100001
    num_image_tokens: int = 576
    vocab_size: int = 32000
    tile_tag: str = "2D"
    global_view_pos: str = "head"
    eos_token_id: list[int] | None = None
    quantization: dict | None = None

    @classmethod
    def from_dict(cls, params):
        if "language_config" in params:
            params["text_config"] = params["language_config"]
            del params["language_config"]

        return cls(
            text_config=TextConfig.from_dict(params["text_config"]),
            vision_config=VisionConfig.from_dict(params["vision_config"]),
            projector_config=ProjectorConfig.from_dict(params["projector_config"]),
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
                and k not in ["text_config", "vision_config", "projector_config"]
            },
        )


@dataclass
class Conversation:
    """A class that represents a conversation."""

    system: str
    roles: list[str]
    messages: list[list[str]]
    offset: int
    sep_style: int
    sep: str
    sep2: str
    version: str = "Unknown"


@dataclass
class VLChatProcessorOutput:
    """
    Output of the VL chat processor.
    """

    sft_format: str
    input_ids: list[int]
    pixel_values: list
    num_image_tokens: list[int]
    image_grid_thw: list[list[int]]
    image_sizes: list[list[int]] | None = None
    videos: list | None = None
    aspect_ratio_ids: list[int] | None = None
    aspect_ratio_mask: list[list[int]] | None = None
    cross_attention_mask: list[list[list[int]]] | None = None
    attention_mask: list[int] | None = None
    labels: list[int] | None = None


@dataclass
class BatchCollateOutput:
    input_ids: list
    labels: list
    attention_mask: list
    pixel_values: list
