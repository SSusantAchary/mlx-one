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
    Callable,
    GradientCheckpointingLayer,
    ModelOutput,
    OutputRecorder,
    PreTrainedModel,
    Tensor,
    TransformersKwargs,
    Unpack,
    auto_docstring,
    can_return_tuple,
    capture_outputs,
    dataclass,
    functional,
    init,
    is_flash_attention_requested,
    layers,
    logging,
    merge_with_config_defaults,
    np,
    ops,
)
from .config import (
    Sam3TrackerConfig,
    Sam3TrackerMaskDecoderConfig,
    Sam3TrackerPromptEncoderConfig,
)

logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(custom_intro="Base class for the Sam3Tracker model's output.")
class Sam3TrackerImageSegmentationOutput(ModelOutput):
    iou_scores: ops.FloatTensor | None = None
    pred_masks: ops.FloatTensor | None = None
    object_score_logits: ops.FloatTensor | None = None
    image_embeddings: tuple[ops.FloatTensor, ...] = None
    vision_hidden_states: tuple[ops.FloatTensor, ...] | None = None
    vision_attentions: tuple[ops.FloatTensor, ...] | None = None
    mask_decoder_attentions: tuple[ops.FloatTensor, ...] | None = None


class Sam3TrackerFeedForward(layers.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: str = "relu",
        sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.activation = ACT2FN[activation]
        self.proj_in = layers.Linear(input_dim, hidden_dim)
        self.proj_out = layers.Linear(hidden_dim, output_dim)
        self.layers = layers.ModuleList(
            [layers.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))
        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = functional.sigmoid(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="\n    Segment Anything Model 3 (SAM 3) for generating segmentation masks, given an input image and\n    input points and labels, boxes, or masks.\n    "
)
class Sam3TrackerPreTrainedModel(PreTrainedModel):
    config_class = Sam3TrackerConfig
    base_model_prefix = "sam3_tracker"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    _keys_to_ignore_on_load_unexpected = [
        "^memory_.*",
        "^mask_downsample.*",
        "^object_pointer_proj.*",
        "^temporal_positional_encoding_projection_layer.*",
        "no_memory_positional_encoding",
        "no_object_pointer",
        "occlusion_spatial_embedding_parameter",
    ]

    @ops.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Sam3TrackerModel):
            if module.no_memory_embedding is not None:
                init.zeros_(module.no_memory_embedding)
        elif isinstance(module, Sam3TrackerPositionalEmbedding):
            init.normal_(module.positional_embedding, std=module.scale)


class Sam3TrackerPositionalEmbedding(layers.Module):
    def __init__(self, config: Sam3TrackerPromptEncoderConfig):
        super().__init__()
        self.scale = config.scale
        positional_embedding = self.scale * ops.randn((2, config.hidden_size // 2))
        self.register_buffer("positional_embedding", positional_embedding)

    def forward(self, input_coords, input_shape=None):
        coordinates = input_coords.clone()
        if input_shape is not None:
            coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / input_shape[1]
            coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / input_shape[0]
        coordinates.to(ops.float32)
        coordinates = 2 * coordinates - 1
        coordinates = coordinates.to(self.positional_embedding.dtype)
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * np.pi * coordinates
        return ops.cat([ops.sin(coordinates), ops.cos(coordinates)], dim=-1)


class Sam3TrackerMaskEmbedding(layers.Module):
    def __init__(self, config: Sam3TrackerPromptEncoderConfig):
        super().__init__()
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = layers.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = layers.Conv2d(
            self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2
        )
        self.conv3 = layers.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = Sam3TrackerLayerNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.layer_norm2 = Sam3TrackerLayerNorm(
            self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format="channels_first"
        )

    def forward(self, masks):
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings


class Sam3TrackerPromptEncoder(layers.Module):
    def __init__(self, config: Sam3TrackerPromptEncoderConfig):
        super().__init__()
        self.shared_embedding = Sam3TrackerPositionalEmbedding(config)
        self.mask_embed = Sam3TrackerMaskEmbedding(config)
        self.no_mask_embed = layers.Embedding(1, config.hidden_size)
        self.image_embedding_size = (
            config.image_size // config.patch_size,
            config.image_size // config.patch_size,
        )
        self.mask_input_size = (
            4 * config.image_size // config.patch_size,
            4 * config.image_size // config.patch_size,
        )
        self.input_image_size = config.image_size
        self.point_embed = layers.Embedding(config.num_point_embeddings, config.hidden_size)
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = layers.Embedding(1, config.hidden_size)

    def _embed_points(self, points: ops.Tensor, labels: ops.Tensor, pad: bool) -> ops.Tensor:
        points = points + 0.5
        if pad:
            points = ops.nn.functional.pad(points, (0, 0, 0, 1), mode="constant", value=0)
            labels = ops.nn.functional.pad(labels, (0, 1), mode="constant", value=-1)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)
        point_embedding = ops.where(
            labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding
        )
        point_embedding = ops.where(
            labels[..., None] != -10, point_embedding, ops.zeros_like(point_embedding)
        )
        point_embedding = point_embedding + self.point_embed(labels.clamp(min=0)) * (
            labels >= 0
        ).unsqueeze(-1)
        return point_embedding

    def _embed_boxes(self, boxes: ops.Tensor) -> ops.Tensor:
        boxes = boxes + 0.5
        coords = boxes.view(*boxes.shape[:2], 2, 2)
        coords = ops.nn.functional.pad(coords, (0, 0, 0, 1), mode="constant", value=0)
        corner_embedding = self.shared_embedding(
            coords, (self.input_image_size, self.input_image_size)
        )
        corner_embedding[:, :, 0, :] += self.point_embed.weight[2]
        corner_embedding[:, :, 1, :] += self.point_embed.weight[3]
        corner_embedding[:, :, 2, :] = self.not_a_point_embed.weight.expand_as(
            corner_embedding[:, :, 2, :]
        )
        return corner_embedding

    def forward(
        self,
        input_points: tuple[ops.Tensor, ops.Tensor] | None,
        input_labels: ops.Tensor | None,
        input_boxes: ops.Tensor | None,
        input_masks: ops.Tensor | None,
    ) -> tuple[ops.Tensor, ops.Tensor]:
        sparse_embeddings = None
        batch_size = 1
        if input_points is not None:
            batch_size = input_points.shape[0]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            point_embeddings = self._embed_points(
                input_points, input_labels, pad=input_boxes is None
            )
            sparse_embeddings = point_embeddings
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = ops.cat([sparse_embeddings, box_embeddings], dim=2)
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return (sparse_embeddings, dense_embeddings)


def eager_attention_forward(
    module: layers.Module,
    query: ops.Tensor,
    key: ops.Tensor,
    value: ops.Tensor,
    attention_mask: ops.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = ops.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = layers.functional.softmax(attn_weights, dim=-1, dtype=ops.float32).to(
        query.dtype
    )
    attn_weights = layers.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = ops.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return (attn_output, attn_weights)


class Sam3TrackerAttention(layers.Module):
    def __init__(self, config, downsample_rate=None):
        super().__init__()
        downsample_rate = (
            config.attention_downsample_rate if downsample_rate is None else downsample_rate
        )
        self.config = config
        self.hidden_size = config.hidden_size
        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.internal_dim // config.num_attention_heads
        self.scaling = self.head_dim ** (-0.5)
        self.is_causal = False
        self.q_proj = layers.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = layers.Linear(self.hidden_size, self.internal_dim)
        self.v_proj = layers.Linear(self.hidden_size, self.internal_dim)
        self.o_proj = layers.Linear(self.internal_dim, self.hidden_size)

    def forward(
        self,
        query: ops.Tensor,
        key: ops.Tensor,
        value: ops.Tensor,
        attention_similarity: ops.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[ops.Tensor, ops.Tensor]:
        batch_size, point_batch_size = query.shape[:2]
        new_shape = (batch_size * point_batch_size, -1, self.num_attention_heads, self.head_dim)
        query = self.q_proj(query).view(*new_shape).transpose(1, 2)
        key = self.k_proj(key).view(*new_shape).transpose(1, 2)
        value = self.v_proj(value).view(*new_shape).transpose(1, 2)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        if is_flash_attention_requested(self.config) and attention_similarity is not None:
            attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
            logger.warning_once(
                "Falling back to SDPA for target-guided attention because Flash Attention does not support additive bias masks."
            )
        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=attention_similarity,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = attn_output.reshape(
            batch_size, point_batch_size, -1, self.num_attention_heads * self.head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights)


class Sam3TrackerTwoWayAttentionBlock(GradientCheckpointingLayer):
    def __init__(self, config: Sam3TrackerMaskDecoderConfig, skip_first_layer_pe: bool = False):
        super().__init__()
        self.self_attn = Sam3TrackerAttention(config, downsample_rate=1)
        self.layer_norm1 = layers.LayerNorm(config.hidden_size)
        self.cross_attn_token_to_image = Sam3TrackerAttention(config)
        self.layer_norm2 = layers.LayerNorm(config.hidden_size)
        self.mlp = Sam3TrackerFeedForward(
            config.hidden_size,
            config.mlp_dim,
            config.hidden_size,
            num_layers=config.num_hidden_layers,
        )
        self.layer_norm3 = layers.LayerNorm(config.hidden_size)
        self.layer_norm4 = layers.LayerNorm(config.hidden_size)
        self.cross_attn_image_to_token = Sam3TrackerAttention(config)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
        attention_similarity: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if self.skip_first_layer_pe:
            queries, _ = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out, _ = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)
        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out, _ = self.cross_attn_token_to_image(
            query=query, key=key, value=keys, attention_similarity=attention_similarity
        )
        queries = queries + attn_out
        queries = self.layer_norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)
        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out, _ = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out
        keys = self.layer_norm4(keys)
        return (queries, keys, attn_out)


class Sam3TrackerTwoWayTransformer(layers.Module):
    def __init__(self, config: Sam3TrackerMaskDecoderConfig):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.layers = layers.ModuleList()
        for i in range(self.num_hidden_layers):
            self.layers.append(Sam3TrackerTwoWayAttentionBlock(config, skip_first_layer_pe=i == 0))
        self.final_attn_token_to_image = Sam3TrackerAttention(config)
        self.layer_norm_final_attn = layers.LayerNorm(config.hidden_size)

    def forward(
        self,
        point_embeddings: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        attention_similarity: Tensor,
        target_embedding=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput:
        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")
        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = (
            image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        )
        queries = point_embeddings
        keys = image_embeddings
        for layer in self.layers:
            if target_embedding is not None:
                queries += target_embedding
            queries, keys, _ = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_positional_embeddings,
                attention_similarity=attention_similarity,
                **kwargs,
            )
        query = queries + point_embeddings
        key = keys + image_positional_embeddings
        attn_out, _ = self.final_attn_token_to_image(query=query, key=key, value=keys)
        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return (queries, keys)


class Sam3TrackerLayerNorm(layers.LayerNorm):
    def __init__(self, normalized_shape, *, eps=1e-06, data_format="channels_last", **kwargs):
        super().__init__(normalized_shape, eps=eps, **kwargs)
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {data_format}")
        self.data_format = data_format

    def forward(self, features: ops.Tensor) -> ops.Tensor:
        if self.data_format == "channels_first":
            features = features.permute(0, 2, 3, 1)
            features = super().forward(features)
            features = features.permute(0, 3, 1, 2)
        else:
            features = super().forward(features)
        return features


class Sam3TrackerMaskDecoder(layers.Module):
    def __init__(self, config: Sam3TrackerMaskDecoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1
        self.iou_token = layers.Embedding(1, self.hidden_size)
        self.mask_tokens = layers.Embedding(self.num_mask_tokens, self.hidden_size)
        self.transformer = Sam3TrackerTwoWayTransformer(config)
        self.upscale_conv1 = layers.ConvTranspose2d(
            self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2
        )
        self.upscale_conv2 = layers.ConvTranspose2d(
            self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2
        )
        self.upscale_layer_norm = Sam3TrackerLayerNorm(
            self.hidden_size // 4, data_format="channels_first"
        )
        self.activation = layers.GELU()
        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [
                Sam3TrackerFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)
            ]
        self.output_hypernetworks_mlps = layers.ModuleList(mlps_list)
        self.iou_prediction_head = Sam3TrackerFeedForward(
            self.hidden_size,
            config.iou_head_hidden_dim,
            self.num_mask_tokens,
            config.iou_head_depth,
            sigmoid_output=True,
        )
        self.conv_s0 = layers.Conv2d(
            config.hidden_size, config.hidden_size // 8, kernel_size=1, stride=1
        )
        self.conv_s1 = layers.Conv2d(
            config.hidden_size, config.hidden_size // 4, kernel_size=1, stride=1
        )
        self.obj_score_token = layers.Embedding(1, self.hidden_size)
        self.pred_obj_score_head = Sam3TrackerFeedForward(self.hidden_size, self.hidden_size, 1, 3)
        self.dynamic_multimask_via_stability = config.dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = config.dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = config.dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: ops.Tensor,
        image_positional_embeddings: ops.Tensor,
        sparse_prompt_embeddings: ops.Tensor,
        dense_prompt_embeddings: ops.Tensor,
        multimask_output: bool,
        high_resolution_features: list[ops.Tensor],
        attention_similarity: ops.Tensor | None = None,
        target_embedding: ops.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[ops.Tensor, ops.Tensor, ops.Tensor, ops.Tensor]:
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        output_tokens = ops.cat(
            [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)
        if sparse_prompt_embeddings.shape[0] != 0:
            tokens = ops.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(
            point_batch_size, 0
        )
        point_embeddings, image_embeddings = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            **kwargs,
        )
        iou_token_out = point_embeddings[:, :, 1, :]
        mask_tokens_out = point_embeddings[:, :, 2 : 2 + self.num_mask_tokens, :]
        image_embeddings = image_embeddings.transpose(2, 3).view(
            batch_size * point_batch_size, num_channels, height, width
        )
        feat_s0, feat_s1 = high_resolution_features
        feat_s0 = feat_s0.repeat_interleave(point_batch_size, dim=0)
        feat_s1 = feat_s1.repeat_interleave(point_batch_size, dim=0)
        upscaled_embedding = self.upscale_conv1(image_embeddings) + feat_s1
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding) + feat_s0)
        hyper_in_list: list[ops.Tensor] = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = ops.stack(hyper_in_list, dim=2)
        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.view(
            batch_size, point_batch_size, num_channels, height * width
        )
        masks = (hyper_in @ upscaled_embedding).view(
            batch_size, point_batch_size, -1, height, width
        )
        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(point_embeddings[:, :, 0, :])
        if multimask_output:
            mask_slice = slice(1, None)
            masks = masks[:, :, mask_slice, :, :]
            iou_pred = iou_pred[:, :, mask_slice]
        elif self.dynamic_multimask_via_stability and (not self.training):
            mask_slice = slice(0, 1)
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            mask_slice = slice(0, 1)
            masks = masks[:, :, mask_slice, :, :]
            iou_pred = iou_pred[:, :, mask_slice]
        sam_tokens_out = mask_tokens_out[:, :, mask_slice]
        return (masks, iou_pred, sam_tokens_out, object_score_logits)

    def _get_stability_scores(self, mask_logits):
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = ops.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = ops.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = ops.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        multimask_logits = all_mask_logits[:, :, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, :, 1:]
        best_scores_inds = ops.argmax(multimask_iou_scores, dim=-1)
        best_scores_inds_expanded = best_scores_inds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        best_scores_inds_expanded = best_scores_inds_expanded.expand(
            -1, -1, 1, multimask_logits.size(-2), multimask_logits.size(-1)
        )
        best_multimask_logits = ops.gather(multimask_logits, 2, best_scores_inds_expanded)
        best_multimask_iou_scores = ops.gather(
            multimask_iou_scores, 2, best_scores_inds.unsqueeze(-1)
        )
        singlemask_logits = all_mask_logits[:, :, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, :, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh
        mask_logits_out = ops.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = ops.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return (mask_logits_out, iou_scores_out)


@dataclass
@auto_docstring(custom_intro="Base class for the vision encoder's outputs.")
class Sam3TrackerVisionEncoderOutput(BaseModelOutputWithPooling):
    fpn_hidden_states: ops.FloatTensor | None = None
    fpn_position_encoding: ops.FloatTensor | None = None


@auto_docstring(
    custom_intro="\n    Segment Anything Model 2 (SAM 2) for generating segmentation masks, given an input image and\n    input points and labels, boxes, or masks.\n    "
)
class Sam3TrackerModel(Sam3TrackerPreTrainedModel):
    input_modalities = ("image", "text")
    _can_record_outputs = {
        "mask_decoder_attentions": OutputRecorder(Sam3TrackerTwoWayAttentionBlock, index=2)
    }
    _tied_weights_keys = {}
    _checkpoint_conversion_mapping = {
        "tracker_model.(.+)": "\\1",
        "detector_model.vision_encoder.backbone.": "vision_encoder.backbone.",
        "tracker_neck.": "vision_encoder.neck.",
    }
    _keys_to_ignore_on_load_unexpected = [
        "^detector_model.",
        "^memory_.*",
        "^mask_downsample.*",
        "^object_pointer_proj.*",
        "^temporal_positional_encoding_projection_layer.*",
        "no_memory_positional_encoding",
        "no_object_pointer",
        "occlusion_spatial_embedding_parameter",
    ]

    def __init__(self, config: Sam3TrackerConfig):
        if hasattr(config, "tracker_config") and config.tracker_config is not None:
            if isinstance(config.tracker_config, dict):
                config.tracker_config = Sam3TrackerConfig(**config.tracker_config)
            config = config.tracker_config
        super().__init__(config)
        self.shared_image_embedding = Sam3TrackerPositionalEmbedding(config.prompt_encoder_config)
        self.vision_encoder = AutoModel.from_config(config.vision_config)
        self.prompt_encoder = Sam3TrackerPromptEncoder(config.prompt_encoder_config)
        config.mask_decoder_config._attn_implementation = config._attn_implementation
        self.mask_decoder = Sam3TrackerMaskDecoder(config.mask_decoder_config)
        self.backbone_feature_sizes = config.vision_config.backbone_feature_sizes
        self.hidden_dim = config.vision_config.fpn_hidden_size
        self.no_memory_embedding = ops.nn.Parameter(ops.zeros(1, 1, self.hidden_dim))
        self.post_init()

    def get_input_embeddings(self):
        return self.vision_encoder.get_input_embeddings()

    def get_image_wide_positional_embeddings(self) -> ops.Tensor:
        size = self.prompt_encoder.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = ops.ones(size, device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size[0]
        x_embed = x_embed / size[1]
        positional_embedding = self.shared_image_embedding(ops.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)

    @ops.no_grad()
    def get_image_embeddings(
        self, pixel_values: ops.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> list[ops.Tensor]:
        batch_size = pixel_values.shape[0]
        image_outputs = self.get_image_features(pixel_values, return_dict=True, **kwargs)
        feature_maps = image_outputs.fpn_hidden_states
        feature_maps[-1] = feature_maps[-1] + self.no_memory_embedding
        image_embeddings = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(feature_maps, self.backbone_feature_sizes)
        ]
        return image_embeddings

    @ops.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: ops.FloatTensor | None = None,
        input_labels: ops.LongTensor | None = None,
        input_boxes: ops.FloatTensor | None = None,
        input_masks: ops.LongTensor | None = None,
    ):
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: ops.FloatTensor | None = None,
        input_points: ops.FloatTensor | None = None,
        input_labels: ops.LongTensor | None = None,
        input_boxes: ops.FloatTensor | None = None,
        input_masks: ops.LongTensor | None = None,
        image_embeddings: ops.FloatTensor | None = None,
        multimask_output: bool = True,
        attention_similarity: ops.FloatTensor | None = None,
        target_embedding: ops.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sam3TrackerImageSegmentationOutput:
        if not (pixel_values is None) ^ (image_embeddings is None):
            raise ValueError("Exactly one of pixel_values or image_embeddings must be provided.")
        if input_points is not None and input_boxes is not None:
            if input_points.shape[1] != input_boxes.shape[1]:
                raise ValueError(
                    f"You should provide as many bounding boxes as input points per box. Got {input_points.shape[1]} and {input_boxes.shape[1]}."
                )
        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        batch_size = (
            pixel_values.shape[0] if pixel_values is not None else image_embeddings[-1].shape[0]
        )
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
        vision_attentions = None
        vision_hidden_states = None
        if pixel_values is not None:
            image_outputs: Sam3TrackerVisionEncoderOutput = self.get_image_features(
                pixel_values, return_dict=True, **kwargs
            )
            feature_maps = image_outputs.fpn_hidden_states
            vision_hidden_states = image_outputs.hidden_states
            vision_attentions = image_outputs.attentions
            feature_maps[-1] = feature_maps[-1] + self.no_memory_embedding
            image_embeddings = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(feature_maps, self.backbone_feature_sizes)
            ]
        if input_points is not None and input_labels is None:
            input_labels = ops.ones_like(
                input_points[:, :, :, 0], dtype=ops.int, device=input_points.device
            )
        if input_points is None and input_boxes is None:
            input_points = ops.zeros(
                batch_size,
                1,
                1,
                2,
                dtype=image_embeddings[-1].dtype,
                device=image_embeddings[-1].device,
            )
            input_labels = -ops.ones(
                batch_size, 1, 1, dtype=ops.int32, device=image_embeddings[-1].device
            )
        if input_masks is not None:
            if input_masks.shape[-2:] != self.prompt_encoder.mask_input_size:
                input_masks = functional.interpolate(
                    input_masks.float(),
                    size=self.prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                ).to(input_masks.dtype)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        low_res_multimasks, iou_scores, _, object_score_logits = self.mask_decoder(
            image_embeddings=image_embeddings[-1],
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            high_resolution_features=image_embeddings[:-1],
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            **kwargs,
        )
        return Sam3TrackerImageSegmentationOutput(
            iou_scores=iou_scores,
            pred_masks=low_res_multimasks,
            object_score_logits=object_score_logits,
            image_embeddings=image_embeddings,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
        )

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self, pixel_values: ops.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | Sam3TrackerVisionEncoderOutput:
        vision_outputs: Sam3TrackerVisionEncoderOutput = self.vision_encoder(
            pixel_values, return_dict=True, **kwargs
        )
        feature_maps = vision_outputs.fpn_hidden_states
        feature_maps_position_embeddings = vision_outputs.fpn_position_encoding
        feature_maps = list(feature_maps)
        feature_maps[0] = self.mask_decoder.conv_s0(feature_maps[0])
        feature_maps[1] = self.mask_decoder.conv_s1(feature_maps[1])
        feature_maps = [feature_map.flatten(2).permute(2, 0, 1) for feature_map in feature_maps]
        feature_maps_position_embeddings = [
            feature_map_position_embedding.flatten(2).permute(2, 0, 1)
            for feature_map_position_embedding in feature_maps_position_embeddings
        ]
        vision_outputs.fpn_hidden_states = feature_maps
        vision_outputs.fpn_position_encoding = feature_maps_position_embeddings
        return vision_outputs
