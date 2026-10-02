# Copyright 2025 Meta AI and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.md.
# Native MLX port of Transformers revision 6133195dcb; no PyTorch runtime.
"""Native MLX inference port; see NOTICE.md for reference provenance."""

from __future__ import annotations
from ._runtime import (
    ACT2FN,
    ALL_ATTENTION_FUNCTIONS,
    AutoModel,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CLIPTextModelWithProjection,
    Callable,
    GradientCheckpointingLayer,
    Iterable,
    ModelOutput,
    PreTrainedModel,
    Tensor,
    TransformersKwargs,
    Unpack,
    auto_docstring,
    can_return_tuple,
    capture_outputs,
    compile_compatible_method_lru_cache,
    create_bidirectional_mask,
    dataclass,
    functional,
    init,
    is_flash_attention_requested,
    layers,
    logging,
    math,
    merge_with_config_defaults,
    np,
    ops,
    torchvision,
)
from .config import (
    Sam3Config,
    Sam3DETRDecoderConfig,
    Sam3DETREncoderConfig,
    Sam3GeometryEncoderConfig,
    Sam3MaskDecoderConfig,
    Sam3ViTConfig,
    Sam3VisionConfig,
)

logger = logging.get_logger(__name__)


@dataclass
@auto_docstring
class Sam3VisionEncoderOutput(BaseModelOutputWithPooling):
    fpn_hidden_states: tuple[ops.FloatTensor, ...] = None
    fpn_position_encoding: tuple[ops.FloatTensor, ...] = None


@dataclass
@auto_docstring
class Sam3GeometryEncoderOutput(ModelOutput):
    last_hidden_state: ops.FloatTensor = None
    attention_mask: ops.BoolTensor | None = None


@dataclass
@auto_docstring
class Sam3DETREncoderOutput(ModelOutput):
    last_hidden_state: ops.FloatTensor = None
    pos_embeds_flattened: ops.FloatTensor | None = None
    text_features: ops.FloatTensor | None = None
    spatial_shapes: ops.LongTensor | None = None
    hidden_states: tuple[ops.FloatTensor] | None = None
    attentions: tuple[ops.FloatTensor] | None = None


@dataclass
@auto_docstring
class Sam3DETRDecoderOutput(ModelOutput):
    intermediate_hidden_states: ops.FloatTensor = None
    reference_boxes: ops.FloatTensor = None
    presence_logits: ops.FloatTensor = None
    hidden_states: tuple[ops.FloatTensor] | None = None
    attentions: tuple[ops.FloatTensor] | None = None


@dataclass
@auto_docstring
class Sam3MaskDecoderOutput(ModelOutput):
    pred_masks: ops.FloatTensor = None
    semantic_seg: ops.FloatTensor | None = None
    attentions: tuple[ops.FloatTensor] | None = None


@dataclass
@auto_docstring
class Sam3ImageSegmentationOutput(ModelOutput):
    pred_masks: ops.FloatTensor = None
    pred_boxes: ops.FloatTensor = None
    pred_logits: ops.FloatTensor | None = None
    presence_logits: ops.FloatTensor | None = None
    semantic_seg: ops.FloatTensor | None = None
    decoder_hidden_states: tuple[ops.FloatTensor] | None = None
    decoder_reference_boxes: ops.FloatTensor | None = None
    encoder_hidden_states: tuple[ops.FloatTensor] | None = None
    vision_hidden_states: tuple[ops.FloatTensor] | None = None
    vision_attentions: tuple[ops.FloatTensor] | None = None
    detr_encoder_attentions: tuple[ops.FloatTensor] | None = None
    detr_decoder_attentions: tuple[ops.FloatTensor] | None = None
    mask_decoder_attentions: tuple[ops.FloatTensor] | None = None


def inverse_sigmoid(x: ops.Tensor, eps: float = 0.001) -> ops.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return ops.log(x1 / x2)


def concat_padded_sequences(seq1, mask1, seq2, mask2, return_index: bool = False):
    batch_size, seq1_length, hidden_size = seq1.shape
    batch_size2, seq2_length, hidden_size2 = seq2.shape
    assert batch_size == batch_size2 == mask1.size(0) == mask2.size(0)
    assert hidden_size == hidden_size2
    assert seq1_length == mask1.size(1)
    assert seq2_length == mask2.size(1)
    actual_seq1_lengths = mask1.sum(dim=-1)
    actual_seq2_lengths = mask2.sum(dim=-1)
    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length
    concatenated_mask = (
        ops.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1)
        < final_lengths[:, None]
    )
    concatenated_sequence = ops.zeros(
        (batch_size, max_length, hidden_size), device=seq2.device, dtype=seq2.dtype
    )
    concatenated_sequence[:, :seq1_length, :] = seq1
    index = ops.arange(seq2_length, device=seq2.device)[None].repeat(batch_size, 1)
    index = index + actual_seq1_lengths[:, None]
    concatenated_sequence = concatenated_sequence.scatter(
        1, index[:, :, None].expand(-1, -1, hidden_size), seq2
    )
    if return_index:
        return (concatenated_sequence, concatenated_mask, index)
    return (concatenated_sequence, concatenated_mask)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return ops.stack(b, dim=-1)


class Sam3MLP(layers.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = layers.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = layers.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = layers.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: ops.Tensor) -> ops.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


def eager_attention_forward(
    module: layers.Module,
    query: ops.Tensor,
    key: ops.Tensor,
    value: ops.Tensor,
    attention_mask: ops.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** (-0.5)
    attn_weights = ops.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = layers.functional.softmax(attn_weights, dim=-1)
    attn_weights = layers.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = ops.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return (attn_output, attn_weights)


class Sam3Attention(layers.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** (-0.5)
        self.is_causal = False
        self.q_proj = layers.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = layers.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = layers.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = layers.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        query: ops.Tensor,
        key: ops.Tensor,
        value: ops.Tensor,
        attention_mask: ops.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[ops.Tensor, ops.Tensor]:
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]
        query = (
            self.q_proj(query)
            .view(batch_size, query_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(key)
            .view(batch_size, key_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(value)
            .view(batch_size, key_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        if (
            is_flash_attention_requested(self.config)
            and attention_mask is not None
            and (attention_mask.dtype != ops.bool)
        ):
            attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
            logger.warning_once(
                "Sam3Attention: falling back to SDPA for relative-position cross-attention because Flash Attention does not support additive bias masks."
            )
        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = attn_output.reshape(
            batch_size, query_len, self.num_attention_heads * self.head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights)


class Sam3ViTRotaryEmbedding(layers.Module):
    def __init__(self, config: Sam3ViTConfig, end_x: int, end_y: int, scale: float = 1.0):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        if dim % 4 != 0:
            raise ValueError("Dimension must be divisible by 4 for axial RoPE")
        self.end_x, self.end_y = (end_x, end_y)
        self.dim = dim
        self.rope_theta = config.rope_theta
        self.scale = scale
        freqs = 1.0 / config.rope_theta ** (ops.arange(0, dim, 4)[: dim // 4].float() / dim)
        flattened_indices = ops.arange(end_x * end_y, dtype=ops.long)
        x_positions = flattened_indices % end_x * scale
        y_positions = ops.div(flattened_indices, end_x, rounding_mode="floor") * scale
        freqs_x = ops.outer(x_positions, freqs).float()
        freqs_y = ops.outer(y_positions, freqs).float()
        inv_freq = ops.cat([freqs_x, freqs_y], dim=-1)
        inv_freq = inv_freq.repeat_interleave(2, dim=-1)
        self.register_buffer("rope_embeddings_cos", inv_freq.cos(), persistent=False)
        self.register_buffer("rope_embeddings_sin", inv_freq.sin(), persistent=False)

    @ops.no_grad()
    def forward(self) -> tuple[ops.Tensor, ops.Tensor]:
        return (self.rope_embeddings_cos, self.rope_embeddings_sin)


def rotate_pairwise(x):
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = ops.stack((-x2, x1), dim=-1)
    return x.flatten(start_dim=-2)


def apply_rotary_pos_emb_2d(
    q: ops.Tensor, k: ops.Tensor, cos: ops.Tensor, sin: ops.Tensor
) -> tuple[ops.Tensor, ops.Tensor]:
    q_embed = q.float()
    q_embed = q_embed * cos + rotate_pairwise(q_embed) * sin
    k_embed = k.float()
    k_embed = k_embed * cos + rotate_pairwise(k_embed) * sin
    return (q_embed.type_as(q), k_embed.type_as(k))


class Sam3ViTRoPEAttention(layers.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = layers.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = layers.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = layers.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = layers.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        hidden_states: ops.Tensor,
        position_embeddings: tuple[ops.Tensor, ops.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor:
        batch_size, height, width, _ = hidden_states.shape
        seq_len = height * width
        new_shape = (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(*new_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(*new_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(*new_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_2d(query, key, cos=cos, sin=sin)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, height, width, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights)


class Sam3ViTPatchEmbeddings(layers.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        image_size, patch_size = (config.pretrain_image_size, config.patch_size)
        num_channels, hidden_size = (config.num_channels, config.hidden_size)
        image_size = image_size if isinstance(image_size, Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = layers.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, pixel_values: ops.Tensor) -> ops.Tensor:
        embeddings = (
            self.projection(pixel_values.to(self.projection.weight.dtype))
            .flatten(2)
            .transpose(1, 2)
        )
        return embeddings


class Sam3ViTEmbeddings(layers.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.patch_embeddings = Sam3ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = layers.Parameter(ops.randn(1, num_patches, config.hidden_size))
        self.dropout = layers.Dropout(config.hidden_dropout)
        self.patch_size = config.patch_size

    def _tile_position_embeddings(
        self, position_embeddings: ops.Tensor, height: int, width: int
    ) -> ops.Tensor:
        pretrain_size = int(position_embeddings.shape[1] ** 0.5)
        if not ops.jit.is_tracing() and pretrain_size == height and (pretrain_size == width):
            return position_embeddings.reshape(1, height * width, -1)
        hidden_size = position_embeddings.shape[-1]
        pos_embed = position_embeddings.reshape(
            1, pretrain_size, pretrain_size, hidden_size
        ).permute(0, 3, 1, 2)
        repeat_h = height // pretrain_size + 1
        repeat_w = width // pretrain_size + 1
        pos_embed = pos_embed.tile([1, 1, repeat_h, repeat_w])[:, :, :height, :width]
        return pos_embed.permute(0, 2, 3, 1).reshape(1, height * width, hidden_size)

    def forward(
        self, pixel_values: ops.Tensor, interpolate_pos_encoding: bool = False
    ) -> ops.Tensor:
        height, width = pixel_values.shape[-2:]
        embeddings = self.patch_embeddings(pixel_values)
        height_patches = height // self.patch_size
        width_patches = width // self.patch_size
        position_embeddings = self._tile_position_embeddings(
            self.position_embeddings, height_patches, width_patches
        )
        embeddings = embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


def window_partition(hidden_state, window_size):
    batch_size, height, width, num_channels = hidden_state.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    hidden_state = layers.functional.pad(hidden_state, (0, 0, 0, pad_width, 0, pad_height))
    padded_height, padded_width = (height + pad_height, width + pad_width)
    hidden_state = hidden_state.view(
        batch_size,
        padded_height // window_size,
        window_size,
        padded_width // window_size,
        window_size,
        num_channels,
    )
    windows = (
        hidden_state.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, num_channels)
    )
    return (windows, (padded_height, padded_width))


def window_unpartition(windows, window_size, pad_height_width, height_width):
    padded_height, padded_width = pad_height_width
    height, width = height_width
    batch_size = windows.shape[0] // (padded_height * padded_width // window_size // window_size)
    hidden_state = windows.view(
        batch_size,
        padded_height // window_size,
        padded_width // window_size,
        window_size,
        window_size,
        -1,
    )
    hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous()
    hidden_state = hidden_state.view(batch_size, padded_height, padded_width, -1)
    hidden_state = hidden_state[:, :height, :width, :].contiguous()
    return hidden_state


class Sam3ViTLayerScale(layers.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lambda1 = layers.Parameter(
            config.layer_scale_init_value * ops.ones(config.hidden_size)
        )

    def forward(self, hidden_state: ops.Tensor) -> ops.Tensor:
        return hidden_state * self.lambda1


class Sam3ViTLayer(GradientCheckpointingLayer):
    def __init__(self, config: Sam3ViTConfig, window_size: int = 0) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        image_size = config.image_size
        image_size = (
            image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        )
        patch_size = config.patch_size
        patch_size = (
            patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        )
        input_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.layer_norm1 = layers.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        rotary_input_size = input_size if window_size == 0 else (window_size, window_size)
        rotary_scale = config.window_size / rotary_input_size[0]
        self.rotary_emb = Sam3ViTRotaryEmbedding(
            config, end_x=rotary_input_size[0], end_y=rotary_input_size[1], scale=rotary_scale
        )
        self.attention = Sam3ViTRoPEAttention(config)
        self.layer_norm2 = layers.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = Sam3MLP(config)
        self.dropout = layers.Dropout(config.hidden_dropout)
        self.window_size = window_size

    def forward(
        self, hidden_states: ops.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> ops.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if self.window_size > 0:
            height, width = (hidden_states.shape[1], hidden_states.shape[2])
            hidden_states, pad_height_width = window_partition(hidden_states, self.window_size)
        position_embeddings = self.rotary_emb()
        hidden_states, _ = self.attention(hidden_states, position_embeddings, **kwargs)
        if self.window_size > 0:
            hidden_states = window_unpartition(
                hidden_states, self.window_size, pad_height_width, (height, width)
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


@auto_docstring
class Sam3PreTrainedModel(PreTrainedModel):
    config_class = Sam3Config
    base_model_prefix = "sam3"
    main_input_name = "pixel_values"
    input_modalities = ["image", "text"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Sam3ViTEmbeddings):
            init.normal_(module.position_embeddings, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Sam3ViTRotaryEmbedding):
            end_x, end_y = (module.end_x, module.end_y)
            dim = module.dim
            freqs = 1.0 / module.rope_theta ** (ops.arange(0, dim, 4)[: dim // 4].float() / dim)
            flattened_indices = ops.arange(end_x * end_y, dtype=ops.long)
            x_positions = flattened_indices % end_x * module.scale
            y_positions = ops.div(flattened_indices, end_x, rounding_mode="floor") * module.scale
            freqs_x = ops.outer(x_positions, freqs).float()
            freqs_y = ops.outer(y_positions, freqs).float()
            inv_freq = ops.cat([freqs_x, freqs_y], dim=-1)
            inv_freq = inv_freq.repeat_interleave(2, dim=-1)
            init.copy_(module.rope_embeddings_cos, inv_freq.cos())
            init.copy_(module.rope_embeddings_sin, inv_freq.sin())


@auto_docstring
class Sam3ViTModel(Sam3PreTrainedModel):
    _can_record_outputs = {"hidden_states": Sam3ViTLayer, "attentions": Sam3ViTRoPEAttention}

    def __init__(self, config: Sam3ViTConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = Sam3ViTEmbeddings(config)
        self.layer_norm = layers.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = layers.ModuleList(
            [
                Sam3ViTLayer(
                    config,
                    window_size=config.window_size if i not in config.global_attn_indexes else 0,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.post_init()

    def get_input_embeddings(self) -> Sam3ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self, pixel_values: ops.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutput:
        hidden_states = self.embeddings(pixel_values)
        batch_size = hidden_states.shape[0]
        height = pixel_values.shape[-2] // self.config.patch_size
        width = pixel_values.shape[-1] // self.config.patch_size
        hidden_size = hidden_states.shape[-1]
        hidden_states = hidden_states.view(batch_size, height, width, hidden_size)
        hidden_states = self.layer_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        hidden_states = hidden_states.view(batch_size, height * width, hidden_size)
        return BaseModelOutput(last_hidden_state=hidden_states)


class Sam3SinePositionEmbedding(layers.Module):
    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def encode_1d_positions(self, x: ops.Tensor, y: ops.Tensor) -> tuple[ops.Tensor, ops.Tensor]:
        x_embed = x * self.scale
        y_embed = y * self.scale
        dim_t = ops.arange(self.num_pos_feats, dtype=ops.int64, device=x.device).to(x.dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = ops.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = ops.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return (pos_x, pos_y)

    def encode_boxes(self, boxes: ops.Tensor) -> ops.Tensor:
        assert boxes.size(-1) == 4, (
            f"Expected 4D box coordinates (x, y, w, h), got shape {boxes.shape}"
        )
        dim_t = ops.arange(self.num_pos_feats, dtype=ops.int64, device=boxes.device).to(boxes.dtype)
        dim_t = self.temperature ** (
            2 * ops.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )
        x_embed = boxes[:, :, 0] * self.scale
        y_embed = boxes[:, :, 1] * self.scale
        w_embed = boxes[:, :, 2] * self.scale
        h_embed = boxes[:, :, 3] * self.scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_w = w_embed[:, :, None] / dim_t
        pos_h = h_embed[:, :, None] / dim_t
        pos_x = ops.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = ops.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_w = ops.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_h = ops.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = ops.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        return pos

    @compile_compatible_method_lru_cache(maxsize=4)
    def forward(
        self,
        shape: ops.Size,
        device: ops.device | str,
        dtype: ops.dtype,
        mask: Tensor | None = None,
    ) -> Tensor:
        if mask is None:
            mask = ops.zeros((shape[0], shape[2], shape[3]), device=device, dtype=ops.bool)
        not_mask = (~mask).to(dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = ops.arange(self.num_pos_feats, dtype=ops.int64, device=device).to(dtype)
        dim_t = self.temperature ** (
            2 * ops.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = ops.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(
            3
        )
        pos_y = ops.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(
            3
        )
        pos = ops.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Sam3FPNLayer(layers.Module):
    def __init__(self, in_channels: int, fpn_dim: int, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        self.scale_layers = layers.ModuleList()
        if scale_factor == 4.0:
            self.scale_layers.append(
                layers.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            )
            self.scale_layers.append(layers.GELU())
            self.scale_layers.append(
                layers.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
            )
            intermediate_channels = in_channels // 4
        elif scale_factor == 2.0:
            self.scale_layers.append(
                layers.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            )
            intermediate_channels = in_channels // 2
        elif scale_factor == 1.0:
            intermediate_channels = in_channels
        elif scale_factor == 0.5:
            self.scale_layers.append(layers.MaxPool2d(kernel_size=2, stride=2))
            intermediate_channels = in_channels
        else:
            raise NotImplementedError(f"scale_factor={scale_factor} is not supported yet.")
        self.proj1 = layers.Conv2d(
            in_channels=intermediate_channels, out_channels=fpn_dim, kernel_size=1
        )
        self.proj2 = layers.Conv2d(
            in_channels=fpn_dim, out_channels=fpn_dim, kernel_size=3, padding=1
        )

    def forward(self, hidden_states: ops.Tensor) -> ops.Tensor:
        hidden_states = hidden_states.to(self.proj1.weight.dtype)
        for layer in self.scale_layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


class Sam3VisionNeck(layers.Module):
    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        self.config = config
        self.position_encoding = Sam3SinePositionEmbedding(
            num_pos_feats=config.fpn_hidden_size // 2, normalize=True
        )
        self.fpn_layers = layers.ModuleList(
            [
                Sam3FPNLayer(
                    in_channels=config.backbone_config.hidden_size,
                    fpn_dim=config.fpn_hidden_size,
                    scale_factor=scale,
                )
                for scale in config.scale_factors
            ]
        )

    def forward(
        self, hidden_states: ops.Tensor
    ) -> tuple[tuple[ops.Tensor, ...], tuple[ops.Tensor, ...]]:
        fpn_hidden_states = ()
        fpn_position_encoding = ()
        for fpn_layer in self.fpn_layers:
            fpn_output = fpn_layer(hidden_states)
            fpn_hidden_states += (fpn_output,)
            pos_enc = self.position_encoding(fpn_output.shape, fpn_output.device, fpn_output.dtype)
            fpn_position_encoding += (pos_enc,)
        return (fpn_hidden_states, fpn_position_encoding)


@auto_docstring(
    custom_intro="\n    The vision model from Sam without any head or projection on top.\n    "
)
class Sam3VisionModel(Sam3PreTrainedModel):
    config_class = Sam3VisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Sam3VisionConfig):
        super().__init__(config)
        self.config = config
        self.backbone = AutoModel.from_config(config.backbone_config)
        self.neck = Sam3VisionNeck(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @can_return_tuple
    def forward(
        self, pixel_values: ops.FloatTensor | None = None, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | Sam3VisionEncoderOutput:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        backbone_output = self.backbone(pixel_values, **kwargs)
        hidden_states = backbone_output.last_hidden_state
        batch_size = hidden_states.shape[0]
        height = pixel_values.shape[-2] // self.config.backbone_config.patch_size
        width = pixel_values.shape[-1] // self.config.backbone_config.patch_size
        hidden_states_spatial = hidden_states.view(batch_size, height, width, -1).permute(
            0, 3, 1, 2
        )
        fpn_hidden_states, fpn_position_encoding = self.neck(hidden_states_spatial)
        return Sam3VisionEncoderOutput(
            last_hidden_state=hidden_states,
            fpn_hidden_states=fpn_hidden_states,
            fpn_position_encoding=fpn_position_encoding,
            hidden_states=backbone_output.hidden_states,
            attentions=backbone_output.attentions,
        )


class Sam3GeometryEncoderLayer(layers.Module):
    def __init__(self, config: Sam3GeometryEncoderConfig):
        super().__init__()
        self.layer_norm1 = layers.LayerNorm(config.hidden_size)
        self.self_attn = Sam3Attention(config)
        self.dropout = layers.Dropout(config.dropout)
        self.cross_attn = Sam3Attention(config)
        self.layer_norm2 = layers.LayerNorm(config.hidden_size)
        self.mlp = Sam3MLP(config)
        self.layer_norm3 = layers.LayerNorm(config.hidden_size)

    def forward(
        self,
        prompt_feats: Tensor,
        vision_feats: Tensor,
        vision_pos_encoding: Tensor,
        prompt_mask: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ):
        residual = prompt_feats
        hidden_states = self.layer_norm1(prompt_feats)
        hidden_states, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attention_mask=prompt_mask,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        key = vision_feats + vision_pos_encoding
        hidden_states, _ = self.cross_attn(
            query=hidden_states, key=key, value=vision_feats, **kwargs
        )
        hidden_states = self.dropout(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual
        return hidden_states


class Sam3GeometryEncoder(layers.Module):
    def __init__(self, config: Sam3GeometryEncoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.roi_size = config.roi_size
        self.position_encoding = Sam3SinePositionEmbedding(
            num_pos_feats=config.hidden_size // 2, normalize=True
        )
        self.label_embed = layers.Embedding(2, self.hidden_size)
        self.cls_embed = layers.Embedding(1, self.hidden_size)
        self.boxes_direct_project = layers.Linear(4, self.hidden_size)
        self.boxes_pool_project = layers.Conv2d(self.hidden_size, self.hidden_size, self.roi_size)
        self.boxes_pos_enc_project = layers.Linear(self.hidden_size + 2, self.hidden_size)
        self.vision_layer_norm = layers.LayerNorm(self.hidden_size)
        self.final_proj = layers.Linear(self.hidden_size, self.hidden_size)
        self.prompt_layer_norm = layers.LayerNorm(self.hidden_size)
        self.layers = layers.ModuleList(
            [Sam3GeometryEncoderLayer(config) for _ in range(config.num_layers)]
        )
        self.output_layer_norm = layers.LayerNorm(self.hidden_size)

    def _encode_box_coordinates(
        self, center_x: ops.Tensor, center_y: ops.Tensor, width: ops.Tensor, height: ops.Tensor
    ) -> ops.Tensor:
        pos_x, pos_y = self.position_encoding.encode_1d_positions(center_x, center_y)
        pos = ops.cat((pos_y, pos_x, height[:, None], width[:, None]), dim=1)
        return pos

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, vision_features):
        batch_size, num_boxes = boxes.shape[:2]
        height, width = vision_features.shape[-2:]
        boxes_embed = self.boxes_direct_project(boxes)
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = ops.tensor(
            [width, height, width, height], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device
        )
        scale = scale.view(1, 1, 4)
        boxes_xyxy = boxes_xyxy * scale
        dtype = ops.float16 if vision_features.dtype == ops.bfloat16 else vision_features.dtype
        sampled_features = torchvision.ops.roi_align(
            vision_features.to(dtype), boxes_xyxy.to(dtype).unbind(0), self.roi_size
        ).to(vision_features.dtype)
        pooled_projection = self.boxes_pool_project(sampled_features)
        pooled_projection = pooled_projection.view(batch_size, num_boxes, self.hidden_size)
        boxes_embed = boxes_embed + pooled_projection
        center_x, center_y, box_width, box_height = boxes.unbind(-1)
        pos_enc = self._encode_box_coordinates(
            center_x.flatten(), center_y.flatten(), box_width.flatten(), box_height.flatten()
        )
        pos_enc = pos_enc.view(batch_size, num_boxes, pos_enc.shape[-1])
        pos_projection = self.boxes_pos_enc_project(pos_enc)
        boxes_embed = boxes_embed + pos_projection
        label_embed = self.label_embed(boxes_labels.long())
        return (label_embed + boxes_embed, boxes_mask)

    def forward(
        self,
        box_embeddings: ops.Tensor,
        box_mask: ops.Tensor,
        box_labels: ops.Tensor,
        img_feats: tuple[ops.Tensor, ...],
        img_pos_embeds: tuple[ops.Tensor, ...] | None = None,
    ):
        batch_size = box_embeddings.shape[0]
        vision_feats = img_feats[-1]
        vision_pos_embeds = (
            img_pos_embeds[-1] if img_pos_embeds is not None else ops.zeros_like(vision_feats)
        )
        vision_feats_flat = vision_feats.flatten(2).transpose(1, 2)
        vision_pos_embeds_flat = vision_pos_embeds.flatten(2).transpose(1, 2)
        img_feats_last = img_feats[-1]
        img_feats_last = img_feats_last.permute(0, 2, 3, 1)
        normalized_img_feats = self.vision_layer_norm(img_feats_last)
        normalized_img_feats = normalized_img_feats.permute(0, 3, 1, 2)
        prompt_embeds, prompt_mask = self._encode_boxes(
            box_embeddings, box_mask, box_labels, normalized_img_feats
        )
        cls_embed = (
            self.cls_embed.weight.view(1, self.hidden_size).unsqueeze(0).expand(batch_size, -1, -1)
        )
        cls_mask = ops.ones(batch_size, 1, dtype=prompt_mask.dtype, device=prompt_mask.device)
        prompt_embeds, prompt_mask = concat_padded_sequences(
            prompt_embeds, prompt_mask, cls_embed, cls_mask
        )
        prompt_embeds = self.prompt_layer_norm(self.final_proj(prompt_embeds))
        prompt_attention_mask = None
        if prompt_mask is not None:
            prompt_attention_mask = create_bidirectional_mask(
                config=self.config, inputs_embeds=prompt_embeds, attention_mask=prompt_mask
            )
        for layer in self.layers:
            prompt_embeds = layer(
                prompt_feats=prompt_embeds,
                vision_feats=vision_feats_flat,
                vision_pos_encoding=vision_pos_embeds_flat,
                prompt_mask=prompt_attention_mask,
            )
        prompt_embeds = self.output_layer_norm(prompt_embeds)
        return Sam3GeometryEncoderOutput(
            last_hidden_state=prompt_embeds, attention_mask=prompt_mask
        )


class Sam3DetrEncoderLayer(layers.Module):
    def __init__(self, config: Sam3DETREncoderConfig):
        super().__init__()
        self.config = config
        self.layer_norm1 = layers.LayerNorm(config.hidden_size)
        self.self_attn = Sam3Attention(config)
        self.dropout = layers.Dropout(config.dropout)
        self.cross_attn = Sam3Attention(config)
        self.layer_norm2 = layers.LayerNorm(config.hidden_size)
        self.mlp = Sam3MLP(config)
        self.layer_norm3 = layers.LayerNorm(config.hidden_size)

    def forward(
        self,
        vision_feats: Tensor,
        prompt_feats: Tensor,
        vision_pos_encoding: Tensor,
        prompt_cross_attn_mask: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        residual = vision_feats
        hidden_states = self.layer_norm1(vision_feats)
        hidden_states_with_pos = hidden_states + vision_pos_encoding
        hidden_states, _ = self.self_attn(
            query=hidden_states_with_pos, key=hidden_states_with_pos, value=hidden_states, **kwargs
        )
        hidden_states = self.dropout(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states, _ = self.cross_attn(
            query=hidden_states,
            key=prompt_feats,
            value=prompt_feats,
            attention_mask=prompt_cross_attn_mask,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual
        return hidden_states


class Sam3DetrEncoder(Sam3PreTrainedModel):
    _can_record_outputs = {"hidden_states": Sam3DetrEncoderLayer, "attentions": Sam3Attention}

    def __init__(self, config: Sam3DETREncoderConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layers = layers.ModuleList(
            [Sam3DetrEncoderLayer(config) for _ in range(config.num_layers)]
        )
        self.post_init()

    def _prepare_multilevel_features(
        self, vision_features: list[ops.Tensor], vision_pos_embeds: list[ops.Tensor]
    ):
        features_flattened = []
        pos_embeds_flattened = []
        spatial_shapes = []
        for features, pos_embed in zip(vision_features, vision_pos_embeds):
            height, width = features.shape[-2:]
            spatial_shapes.append((height, width))
            features = features.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            features_flattened.append(features)
            pos_embeds_flattened.append(pos_embed)
        features_flattened = ops.cat(features_flattened, dim=1)
        pos_embeds_flattened = ops.cat(pos_embeds_flattened, dim=1)
        spatial_shapes = ops.tensor(
            spatial_shapes, dtype=ops.long, device=features_flattened.device
        )
        return (features_flattened, pos_embeds_flattened, spatial_shapes)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        vision_features: list[ops.Tensor],
        text_features: ops.Tensor,
        vision_pos_embeds: list[ops.Tensor] | None = None,
        text_mask: ops.Tensor | None = None,
        spatial_sizes: list[tuple[int, int]] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Sam3DETREncoderOutput:
        batch_size = (
            vision_features[0].shape[0]
            if vision_features[0].dim() == 4
            else vision_features[0].shape[1]
        )
        if spatial_sizes is not None:
            for i, (height, width) in enumerate(spatial_sizes):
                vision_features[i] = (
                    vision_features[i].reshape(height, width, batch_size, -1).permute(2, 3, 0, 1)
                )
                vision_pos_embeds[i] = (
                    vision_pos_embeds[i].reshape(height, width, batch_size, -1).permute(2, 3, 0, 1)
                )
        features_flattened, pos_embeds_flattened, spatial_shapes = (
            self._prepare_multilevel_features(vision_features, vision_pos_embeds)
        )
        prompt_cross_attn_mask = None
        if text_mask is not None:
            prompt_cross_attn_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=features_flattened,
                attention_mask=text_mask,
                encoder_hidden_states=text_features,
            )
        hidden_states = features_flattened
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                prompt_feats=text_features,
                vision_pos_encoding=pos_embeds_flattened,
                prompt_cross_attn_mask=prompt_cross_attn_mask,
                **kwargs,
            )
        return Sam3DETREncoderOutput(
            last_hidden_state=hidden_states,
            pos_embeds_flattened=pos_embeds_flattened,
            text_features=text_features,
            spatial_shapes=spatial_shapes,
        )


class Sam3DecoderMLP(layers.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        if num_layers == 2:
            self.layer1 = layers.Linear(input_dim, hidden_dim)
            self.layer2 = layers.Linear(hidden_dim, output_dim)
            self.layer3 = None
        elif num_layers == 3:
            self.layer1 = layers.Linear(input_dim, hidden_dim)
            self.layer2 = layers.Linear(hidden_dim, hidden_dim)
            self.layer3 = layers.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Only 2 or 3 layers supported, got {num_layers}")

    def forward(self, x: ops.Tensor) -> ops.Tensor:
        x = functional.relu(self.layer1(x))
        if self.layer3 is not None:
            x = functional.relu(self.layer2(x))
            x = self.layer3(x)
        else:
            x = self.layer2(x)
        return x


class Sam3DetrDecoderLayer(layers.Module):
    def __init__(self, config: Sam3DETRDecoderConfig):
        super().__init__()
        self.config = config
        self.self_attn = Sam3Attention(config)
        self.self_attn_dropout = layers.Dropout(config.dropout)
        self.self_attn_layer_norm = layers.LayerNorm(config.hidden_size)
        self.text_cross_attn = Sam3Attention(config)
        self.text_cross_attn_dropout = layers.Dropout(config.dropout)
        self.text_cross_attn_layer_norm = layers.LayerNorm(config.hidden_size)
        self.vision_cross_attn = Sam3Attention(config)
        self.vision_cross_attn_dropout = layers.Dropout(config.dropout)
        self.vision_cross_attn_layer_norm = layers.LayerNorm(config.hidden_size)
        self.mlp = Sam3MLP(config)
        self.mlp_layer_norm = layers.LayerNorm(config.hidden_size)
        self.mlp_dropout = layers.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: ops.Tensor,
        query_pos: ops.Tensor,
        text_features: ops.Tensor,
        vision_features: ops.Tensor,
        vision_pos_encoding: ops.Tensor,
        text_cross_attn_mask: ops.Tensor | None = None,
        vision_cross_attn_mask: ops.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ops.Tensor:
        query_pos = functional.pad(query_pos, (0, 0, 1, 0), mode="constant", value=0)
        residual = hidden_states
        query_with_pos = hidden_states + query_pos
        attn_output, _ = self.self_attn(
            query=query_with_pos,
            key=query_with_pos,
            value=hidden_states,
            attention_mask=None,
            **kwargs,
        )
        hidden_states = residual + self.self_attn_dropout(attn_output)
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        query_with_pos = hidden_states + query_pos
        attn_output, _ = self.text_cross_attn(
            query=query_with_pos,
            key=text_features,
            value=text_features,
            attention_mask=text_cross_attn_mask,
            **kwargs,
        )
        hidden_states = residual + self.text_cross_attn_dropout(attn_output)
        hidden_states = self.text_cross_attn_layer_norm(hidden_states)
        residual = hidden_states
        query_with_pos = hidden_states + query_pos
        key_with_pos = vision_features + vision_pos_encoding
        attn_output, _ = self.vision_cross_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=vision_features,
            attention_mask=vision_cross_attn_mask,
            **kwargs,
        )
        hidden_states = residual + self.vision_cross_attn_dropout(attn_output)
        hidden_states = self.vision_cross_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_dropout(hidden_states)
        hidden_states = self.mlp_layer_norm(hidden_states)
        return hidden_states


class Sam3DetrDecoder(Sam3PreTrainedModel):
    _can_record_outputs = {"hidden_states": Sam3DetrDecoderLayer, "attentions": Sam3Attention}

    def __init__(self, config: Sam3DETRDecoderConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layers = layers.ModuleList(
            [Sam3DetrDecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.output_layer_norm = layers.LayerNorm(config.hidden_size)
        self.box_head = Sam3DecoderMLP(config.hidden_size, config.hidden_size, 4, 3)
        self.query_embed = layers.Embedding(config.num_queries, config.hidden_size)
        self.reference_points = layers.Embedding(config.num_queries, 4)
        self.presence_token = layers.Embedding(1, config.hidden_size)
        self.presence_head = Sam3DecoderMLP(config.hidden_size, config.hidden_size, 1, 3)
        self.presence_layer_norm = layers.LayerNorm(config.hidden_size)
        self.clamp_presence_logit_max_val = 10.0
        self.ref_point_head = Sam3DecoderMLP(
            2 * config.hidden_size, config.hidden_size, config.hidden_size, 2
        )
        self.box_rpb_embed_x = Sam3DecoderMLP(2, config.hidden_size, config.num_attention_heads, 2)
        self.box_rpb_embed_y = Sam3DecoderMLP(2, config.hidden_size, config.num_attention_heads, 2)
        self.position_encoding = Sam3SinePositionEmbedding(
            num_pos_feats=config.hidden_size // 2, normalize=False
        )
        self.post_init()

    @compile_compatible_method_lru_cache(maxsize=1)
    def _get_coords(
        self, height: ops.Tensor, width: ops.Tensor, dtype: ops.dtype, device: ops.device
    ) -> tuple[ops.Tensor, ops.Tensor]:
        coords_h = ops.arange(0, height, device=device, dtype=dtype) / height
        coords_w = ops.arange(0, width, device=device, dtype=dtype) / width
        return (coords_h, coords_w)

    def _get_rpb_matrix(
        self, reference_boxes: ops.Tensor, spatial_shape: tuple[ops.Tensor, ops.Tensor]
    ) -> ops.Tensor:
        height, width = spatial_shape
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        batch_size, num_queries, _ = boxes_xyxy.shape
        coords_h, coords_w = self._get_coords(
            height, width, dtype=reference_boxes.dtype, device=reference_boxes.device
        )
        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(batch_size, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(batch_size, num_queries, -1, 2)
        deltas_x_log = deltas_x * 8
        deltas_x_log = ops.sign(deltas_x_log) * ops.log2(ops.abs(deltas_x_log) + 1.0) / math.log2(8)
        deltas_y_log = deltas_y * 8
        deltas_y_log = ops.sign(deltas_y_log) * ops.log2(ops.abs(deltas_y_log) + 1.0) / math.log2(8)
        deltas_x = self.box_rpb_embed_x(deltas_x_log)
        deltas_y = self.box_rpb_embed_y(deltas_y_log)
        rpb_matrix = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
        rpb_matrix = rpb_matrix.flatten(2, 3)
        rpb_matrix = rpb_matrix.permute(0, 3, 1, 2).contiguous()
        return rpb_matrix

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        vision_features: ops.Tensor,
        text_features: ops.Tensor,
        vision_pos_encoding: ops.Tensor,
        text_mask: ops.Tensor | None = None,
        spatial_shapes: ops.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Sam3DETRDecoderOutput:
        batch_size = vision_features.shape[0]
        query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = self.reference_points.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = reference_boxes.sigmoid()
        presence_token = self.presence_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = ops.cat([presence_token, query_embeds], dim=1)
        text_cross_attn_mask = None
        if text_mask is not None:
            text_cross_attn_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=text_mask,
                encoder_hidden_states=text_features,
            )
        intermediate_outputs = []
        intermediate_boxes = [reference_boxes]
        intermediate_presence_logits = []
        for layer in self.layers:
            reference_points_input = reference_boxes.unsqueeze(2)
            query_sine_embed = self.position_encoding.encode_boxes(
                reference_points_input[:, :, 0, :]
            )
            query_pos = self.ref_point_head(query_sine_embed)
            vision_cross_attn_mask = None
            if spatial_shapes is not None and spatial_shapes.shape[0] == 1:
                spatial_shape = (spatial_shapes[0, 0], spatial_shapes[0, 1])
                rpb_matrix = self._get_rpb_matrix(reference_boxes, spatial_shape)
                vision_cross_attn_mask = functional.pad(
                    rpb_matrix, (0, 0, 1, 0), mode="constant", value=0
                )
            hidden_states = layer(
                hidden_states,
                query_pos=query_pos,
                text_features=text_features,
                vision_features=vision_features,
                vision_pos_encoding=vision_pos_encoding,
                text_cross_attn_mask=text_cross_attn_mask,
                vision_cross_attn_mask=vision_cross_attn_mask,
                **kwargs,
            )
            query_hidden_states = hidden_states[:, 1:]
            reference_boxes_before_sigmoid = inverse_sigmoid(reference_boxes)
            delta_boxes = self.box_head(self.output_layer_norm(query_hidden_states))
            new_reference_boxes = (delta_boxes + reference_boxes_before_sigmoid).sigmoid()
            reference_boxes = new_reference_boxes.detach()
            intermediate_outputs.append(self.output_layer_norm(query_hidden_states))
            intermediate_boxes.append(new_reference_boxes)
            presence_hidden = hidden_states[:, :1]
            presence_logits = self.presence_head(self.presence_layer_norm(presence_hidden)).squeeze(
                -1
            )
            presence_logits = presence_logits.clamp(
                min=-self.clamp_presence_logit_max_val, max=self.clamp_presence_logit_max_val
            )
            intermediate_presence_logits.append(presence_logits)
        intermediate_outputs = ops.stack(intermediate_outputs)
        intermediate_boxes = ops.stack(intermediate_boxes[:-1])
        intermediate_presence_logits = ops.stack(intermediate_presence_logits)
        return Sam3DETRDecoderOutput(
            intermediate_hidden_states=intermediate_outputs,
            reference_boxes=intermediate_boxes,
            presence_logits=intermediate_presence_logits,
        )


class Sam3DotProductScoring(layers.Module):
    def __init__(self, config: Sam3Config):
        super().__init__()
        self.config = config
        hidden_size = config.detr_decoder_config.hidden_size
        projection_dim = config.detr_decoder_config.hidden_size
        self.text_mlp = Sam3DecoderMLP(
            input_dim=hidden_size,
            hidden_dim=config.detr_decoder_config.intermediate_size,
            output_dim=hidden_size,
            num_layers=2,
        )
        self.text_mlp_dropout = layers.Dropout(config.detr_decoder_config.dropout)
        self.text_mlp_out_norm = layers.LayerNorm(hidden_size)
        self.text_proj = layers.Linear(hidden_size, projection_dim)
        self.query_proj = layers.Linear(hidden_size, projection_dim)
        self.scale = float(1.0 / np.sqrt(projection_dim))
        self.clamp_logits = True
        self.clamp_max_val = 12.0

    def _pool_text_features(
        self, text_features: ops.Tensor, text_mask: ops.Tensor | None
    ) -> ops.Tensor:
        if text_mask is None:
            return text_features.mean(dim=1)
        is_valid = text_mask.to(text_features.dtype).unsqueeze(-1)
        num_valid = is_valid.sum(dim=1).clamp(min=1.0)
        pooled_text = (text_features * is_valid).sum(dim=1) / num_valid
        return pooled_text

    def forward(
        self,
        decoder_hidden_states: ops.Tensor,
        text_features: ops.Tensor,
        text_mask: ops.Tensor | None = None,
    ) -> ops.Tensor:
        orig_text_features = text_features
        text_features = self.text_mlp(text_features)
        text_features = self.text_mlp_dropout(text_features)
        text_features = text_features + orig_text_features
        text_features = self.text_mlp_out_norm(text_features)
        pooled_text = self._pool_text_features(text_features, text_mask)
        proj_text = self.text_proj(pooled_text)
        proj_queries = self.query_proj(decoder_hidden_states)
        proj_text = proj_text.unsqueeze(-1)
        scores = ops.matmul(proj_queries, proj_text.unsqueeze(0))
        scores = scores * self.scale
        if self.clamp_logits:
            scores = scores.clamp(min=-self.clamp_max_val, max=self.clamp_max_val)
        return scores


class Sam3MaskEmbedder(layers.Module):
    def __init__(self, config: Sam3MaskDecoderConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.layers = layers.ModuleList(
            [
                layers.Linear(hidden_size, hidden_size),
                layers.Linear(hidden_size, hidden_size),
                layers.Linear(hidden_size, hidden_size),
            ]
        )
        self.activation = layers.ReLU()

    def forward(self, queries: ops.Tensor) -> ops.Tensor:
        hidden_states = queries
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if i < len(self.layers) - 1:
                hidden_states = self.activation(hidden_states)
        return hidden_states


class Sam3PixelDecoder(layers.Module):
    def __init__(self, config: Sam3MaskDecoderConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        num_upsampling_stages = config.num_upsampling_stages
        self.conv_layers = layers.ModuleList(
            [
                layers.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
                for _ in range(num_upsampling_stages)
            ]
        )
        self.norms = layers.ModuleList(
            [layers.GroupNorm(8, hidden_size) for _ in range(num_upsampling_stages)]
        )
        self.out_channels = hidden_size

    def forward(self, backbone_features: list[ops.Tensor]) -> ops.Tensor:
        prev_fpn = backbone_features[-1]
        for layer_idx, backbone_feat in enumerate(reversed(backbone_features[:-1])):
            prev_fpn = functional.interpolate(
                prev_fpn, size=backbone_feat.shape[-2:], mode="nearest"
            )
            prev_fpn = prev_fpn + backbone_feat
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = self.norms[layer_idx](prev_fpn)
            prev_fpn = functional.relu(prev_fpn)
        return prev_fpn


class Sam3MaskDecoder(Sam3PreTrainedModel):
    _can_record_outputs = {"attentions": Sam3Attention}

    def __init__(self, config: Sam3MaskDecoderConfig):
        super().__init__(config)
        self.config = config
        hidden_size = config.hidden_size
        self.pixel_decoder = Sam3PixelDecoder(config)
        self.mask_embedder = Sam3MaskEmbedder(config)
        self.instance_projection = layers.Conv2d(
            self.pixel_decoder.out_channels, hidden_size, kernel_size=1
        )
        self.semantic_projection = layers.Conv2d(self.pixel_decoder.out_channels, 1, kernel_size=1)
        self.prompt_cross_attn = Sam3Attention(config)
        self.prompt_cross_attn_norm = layers.LayerNorm(hidden_size)
        self.prompt_cross_attn_dropout = layers.Dropout(config.dropout)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        decoder_queries: ops.Tensor,
        backbone_features: list[ops.Tensor],
        encoder_hidden_states: ops.Tensor,
        prompt_features: ops.Tensor | None = None,
        prompt_mask: ops.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Sam3MaskDecoderOutput:
        if prompt_features is not None:
            residual = encoder_hidden_states
            normed_hidden_states = self.prompt_cross_attn_norm(encoder_hidden_states)
            cross_attn_mask = None
            if prompt_mask is not None:
                cross_attn_mask = create_bidirectional_mask(
                    config=self.config,
                    inputs_embeds=normed_hidden_states,
                    encoder_hidden_states=prompt_features,
                    attention_mask=prompt_mask,
                )
            attn_output, _ = self.prompt_cross_attn(
                query=normed_hidden_states,
                key=prompt_features,
                value=prompt_features,
                attention_mask=cross_attn_mask,
                **kwargs,
            )
            encoder_hidden_states = residual + self.prompt_cross_attn_dropout(attn_output)
        pixel_embed = self._embed_pixels(
            backbone_features=backbone_features, encoder_hidden_states=encoder_hidden_states
        )
        instance_embeds = self.instance_projection(pixel_embed)
        mask_embeddings = self.mask_embedder(decoder_queries)
        pred_masks = ops.einsum("bqc,bchw->bqhw", mask_embeddings, instance_embeds)
        semantic_seg = self.semantic_projection(pixel_embed)
        return Sam3MaskDecoderOutput(pred_masks=pred_masks, semantic_seg=semantic_seg)

    def _embed_pixels(
        self, backbone_features: list[ops.Tensor], encoder_hidden_states: ops.Tensor
    ) -> ops.Tensor:
        backbone_visual_feats = [feat.clone() for feat in backbone_features]
        spatial_dim = backbone_features[-1].shape[-2] * backbone_features[-1].shape[-1]
        encoder_visual_embed = encoder_hidden_states[:, :spatial_dim, :]
        batch_size, _, hidden_size = encoder_visual_embed.shape
        height, width = backbone_features[-1].shape[-2:]
        encoder_visual_embed = encoder_visual_embed.transpose(1, 2).reshape(
            batch_size, hidden_size, height, width
        )
        backbone_visual_feats[-1] = encoder_visual_embed
        pixel_embed = self.pixel_decoder(backbone_visual_feats)
        return pixel_embed


class Sam3Model(Sam3PreTrainedModel):
    input_modalities = ["image", "text"]
    _checkpoint_conversion_mapping = {"detector_model.(.+)": "\\1"}
    _keys_to_ignore_on_load_unexpected = ["^tracker_model.", "^tracker_neck."]

    def __init__(self, config: Sam3Config):
        if hasattr(config, "detector_config") and config.detector_config is not None:
            detector_config = config.detector_config
            if isinstance(detector_config, dict):
                detector_config = Sam3Config(**detector_config)
            config = detector_config
        super().__init__(config)
        self.vision_encoder = Sam3VisionModel(config.vision_config)
        self.text_encoder = CLIPTextModelWithProjection(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.text_projection = layers.Linear(
            config.text_config.hidden_size, config.detr_encoder_config.hidden_size
        )
        config.geometry_encoder_config._attn_implementation = config._attn_implementation
        config.detr_encoder_config._attn_implementation = config._attn_implementation
        config.detr_decoder_config._attn_implementation = config._attn_implementation
        config.mask_decoder_config._attn_implementation = config._attn_implementation
        self.geometry_encoder = Sam3GeometryEncoder(config.geometry_encoder_config)
        self.detr_encoder = Sam3DetrEncoder(config.detr_encoder_config)
        self.detr_decoder = Sam3DetrDecoder(config.detr_decoder_config)
        self.mask_decoder = Sam3MaskDecoder(config.mask_decoder_config)
        self.dot_product_scoring = Sam3DotProductScoring(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_text_features(
        self,
        input_ids: ops.LongTensor,
        attention_mask: ops.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        )
        last_hidden_state = text_outputs.last_hidden_state
        text_outputs.pooler_output = self.text_projection(last_hidden_state)
        return text_outputs

    @auto_docstring
    def get_vision_features(
        self, pixel_values: ops.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> Sam3VisionEncoderOutput:
        vision_outputs = self.vision_encoder(pixel_values, **kwargs)
        return vision_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: ops.FloatTensor | None = None,
        vision_embeds: Sam3VisionEncoderOutput | None = None,
        input_ids: ops.LongTensor | None = None,
        attention_mask: ops.Tensor | None = None,
        text_embeds: ops.FloatTensor | None = None,
        input_boxes: ops.FloatTensor | None = None,
        input_boxes_labels: ops.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sam3ImageSegmentationOutput:
        if (pixel_values is None) == (vision_embeds is None):
            raise ValueError("You must specify exactly one of pixel_values or vision_embeds")
        if (input_ids is None) == (text_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or text_embeds")
        if pixel_values is not None:
            batch_size = pixel_values.shape[0]
            device = pixel_values.device
        else:
            batch_size = vision_embeds.fpn_hidden_states[0].shape[0]
            device = vision_embeds.fpn_hidden_states[0].device
        if vision_embeds is None:
            vision_outputs = self.vision_encoder(pixel_values, **kwargs)
        else:
            vision_outputs = vision_embeds
        fpn_hidden_states = vision_outputs.fpn_hidden_states[:-1]
        fpn_position_encoding = vision_outputs.fpn_position_encoding[:-1]
        if text_embeds is None:
            text_features = self.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).pooler_output
        else:
            text_features = text_embeds
        text_mask = attention_mask.bool() if attention_mask is not None else None
        has_geometry_prompts = input_boxes is not None and input_boxes.numel() > 0
        geometry_prompt_features = None
        geometry_prompt_mask = None
        if has_geometry_prompts:
            if input_boxes is not None and input_boxes.numel() > 0:
                box_embeddings = input_boxes
                box_labels = (
                    input_boxes_labels
                    if input_boxes_labels is not None
                    else ops.ones_like(box_embeddings[..., 0], dtype=ops.long)
                )
                box_mask = (
                    input_boxes_labels != -10
                    if input_boxes_labels is not None
                    else ops.ones(batch_size, input_boxes.shape[1], dtype=ops.bool, device=device)
                )
                box_labels = ops.where(box_labels == -10, 0, box_labels)
            else:
                box_embeddings = ops.zeros(
                    batch_size, 0, 4, dtype=text_features.dtype, device=device
                )
                box_labels = ops.zeros(batch_size, 0, dtype=ops.long, device=device)
                box_mask = ops.zeros(batch_size, 0, dtype=ops.bool, device=device)
            geometry_outputs = self.geometry_encoder(
                box_embeddings=box_embeddings,
                box_mask=box_mask,
                box_labels=box_labels,
                img_feats=fpn_hidden_states,
                img_pos_embeds=fpn_position_encoding,
            )
            geometry_prompt_features = geometry_outputs.last_hidden_state
            geometry_prompt_mask = geometry_outputs.attention_mask
        if geometry_prompt_features is not None:
            if text_features.shape[0] == 1 and geometry_prompt_features.shape[0] > 1:
                text_features = text_features.repeat(geometry_prompt_features.shape[0], 1, 1)
            combined_prompt_features = ops.cat([text_features, geometry_prompt_features], dim=1)
            if (
                text_mask is not None
                and text_mask.shape[0] == 1
                and (geometry_prompt_mask.shape[0] > 1)
            ):
                text_mask = text_mask.repeat(geometry_prompt_mask.shape[0], 1)
            if text_mask is not None and geometry_prompt_mask is not None:
                combined_prompt_mask = ops.cat([text_mask, geometry_prompt_mask], dim=1)
            elif text_mask is not None:
                geo_valid_mask = ops.ones(
                    batch_size, geometry_prompt_features.shape[1], dtype=ops.bool, device=device
                )
                combined_prompt_mask = ops.cat([text_mask, geo_valid_mask], dim=1)
            elif geometry_prompt_mask is not None:
                text_valid_mask = ops.ones(
                    batch_size, text_features.shape[1], dtype=ops.bool, device=device
                )
                combined_prompt_mask = ops.cat([text_valid_mask, geometry_prompt_mask], dim=1)
            else:
                combined_prompt_mask = None
        else:
            combined_prompt_features = text_features
            combined_prompt_mask = text_mask
        encoder_outputs = self.detr_encoder(
            vision_features=[fpn_hidden_states[-1]],
            text_features=combined_prompt_features,
            vision_pos_embeds=[fpn_position_encoding[-1]],
            text_mask=combined_prompt_mask,
            **kwargs,
        )
        decoder_outputs = self.detr_decoder(
            vision_features=encoder_outputs.last_hidden_state,
            text_features=encoder_outputs.text_features,
            vision_pos_encoding=encoder_outputs.pos_embeds_flattened,
            text_mask=combined_prompt_mask,
            spatial_shapes=encoder_outputs.spatial_shapes,
            **kwargs,
        )
        all_box_offsets = self.detr_decoder.box_head(decoder_outputs.intermediate_hidden_states)
        reference_boxes_inv_sig = inverse_sigmoid(decoder_outputs.reference_boxes)
        all_pred_boxes_cxcywh = (reference_boxes_inv_sig + all_box_offsets).sigmoid()
        all_pred_boxes = box_cxcywh_to_xyxy(all_pred_boxes_cxcywh)
        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_outputs.intermediate_hidden_states,
            text_features=encoder_outputs.text_features,
            text_mask=combined_prompt_mask,
        ).squeeze(-1)
        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hidden_states = decoder_outputs.intermediate_hidden_states[-1]
        presence_logits = decoder_outputs.presence_logits[-1]
        mask_outputs = self.mask_decoder(
            decoder_queries=decoder_hidden_states,
            backbone_features=list(fpn_hidden_states),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            prompt_features=combined_prompt_features,
            prompt_mask=combined_prompt_mask,
            **kwargs,
        )
        return Sam3ImageSegmentationOutput(
            pred_masks=mask_outputs.pred_masks,
            pred_boxes=pred_boxes,
            pred_logits=pred_logits,
            presence_logits=presence_logits,
            semantic_seg=mask_outputs.semantic_seg,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_reference_boxes=decoder_outputs.reference_boxes,
            encoder_hidden_states=encoder_outputs.hidden_states,
            vision_hidden_states=vision_outputs.hidden_states,
            vision_attentions=vision_outputs.attentions,
            detr_encoder_attentions=encoder_outputs.attentions,
            detr_decoder_attentions=decoder_outputs.attentions,
            mask_decoder_attentions=mask_outputs.attentions,
        )
