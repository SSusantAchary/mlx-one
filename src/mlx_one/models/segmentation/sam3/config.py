# Copyright 2025 Meta AI and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.md.
# Native MLX port of Transformers revision 6133195dcb; no PyTorch runtime.
"""Backend-free SAM3 configuration schemas."""

from __future__ import annotations
from ._configuration import (
    CONFIG_MAPPING,
    AutoConfig,
    CLIPTextConfig,
    PreTrainedConfig,
    auto_docstring,
    logging,
)

"""SAM3 model configuration"""


@auto_docstring(checkpoint="facebook/sam3")
class Sam3ViTConfig(PreTrainedConfig):
    base_config_key = "backbone_config"
    model_type = "sam3_vit_model"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4736,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_channels=3,
        image_size=1008,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-06,
        attention_dropout=0.0,
        rope_theta=10000.0,
        window_size=24,
        global_attn_indexes=None,
        layer_scale_init_value=None,
        pretrain_image_size=336,
        hidden_dropout=0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if global_attn_indexes is None:
            global_attn_indexes = [7, 15, 23, 31]
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.layer_scale_init_value = layer_scale_init_value
        self.pretrain_image_size = pretrain_image_size
        self.hidden_dropout = hidden_dropout
        self.initializer_range = initializer_range


@auto_docstring(checkpoint="facebook/sam3")
class Sam3VisionConfig(PreTrainedConfig):
    base_config_key = "vision_config"
    model_type = "sam3_vision_model"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config=None,
        fpn_hidden_size=256,
        backbone_feature_sizes=None,
        scale_factors=None,
        hidden_act="gelu",
        layer_norm_eps=1e-06,
        initializer_range=0.02,
        **kwargs,
    ):
        scale_factors = [4.0, 2.0, 1.0, 0.5] if scale_factors is None else scale_factors
        if backbone_feature_sizes is None:
            backbone_feature_sizes = [[288, 288], [144, 144], [72, 72]]
        if isinstance(backbone_config, dict):
            backbone_config["model_type"] = backbone_config.get("model_type", "sam3_vit_model")
            backbone_config = CONFIG_MAPPING[backbone_config["model_type"]](**backbone_config)
        elif backbone_config is None:
            backbone_config = CONFIG_MAPPING["sam3_vit_model"]()
        self.backbone_config = backbone_config
        self.fpn_hidden_size = fpn_hidden_size
        self.scale_factors = scale_factors
        self.backbone_feature_sizes = backbone_feature_sizes
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @property
    def image_size(self):
        return self.backbone_config.image_size

    @image_size.setter
    def image_size(self, value):
        self.backbone_config.image_size = value


@auto_docstring(checkpoint="facebook/sam3")
class Sam3GeometryEncoderConfig(PreTrainedConfig):
    model_type = "sam3_geometry_encoder"

    def __init__(
        self,
        hidden_size=256,
        num_layers=3,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        hidden_act="relu",
        hidden_dropout=0.0,
        layer_norm_eps=1e-06,
        roi_size=7,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.layer_norm_eps = layer_norm_eps
        self.roi_size = roi_size
        self.initializer_range = initializer_range


@auto_docstring(checkpoint="facebook/sam3")
class Sam3DETREncoderConfig(PreTrainedConfig):
    model_type = "sam3_detr_encoder"

    def __init__(
        self,
        hidden_size=256,
        num_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        hidden_act="relu",
        hidden_dropout=0.0,
        layer_norm_eps=1e-06,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range


@auto_docstring(checkpoint="facebook/sam3")
class Sam3DETRDecoderConfig(PreTrainedConfig):
    model_type = "sam3_detr_decoder"

    def __init__(
        self,
        hidden_size=256,
        num_layers=6,
        num_queries=200,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        hidden_act="relu",
        hidden_dropout=0.0,
        layer_norm_eps=1e-06,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range


@auto_docstring(checkpoint="facebook/sam3")
class Sam3MaskDecoderConfig(PreTrainedConfig):
    model_type = "sam3_mask_decoder"

    def __init__(
        self,
        hidden_size=256,
        num_upsampling_stages=3,
        layer_norm_eps=1e-06,
        dropout=0.0,
        num_attention_heads=8,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_upsampling_stages = num_upsampling_stages
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range


@auto_docstring(checkpoint="facebook/sam3")
class Sam3Config(PreTrainedConfig):
    model_type = "sam3"
    is_composition = True
    sub_configs = {
        "vision_config": Sam3VisionConfig,
        "text_config": CLIPTextConfig,
        "geometry_encoder_config": Sam3GeometryEncoderConfig,
        "detr_encoder_config": Sam3DETREncoderConfig,
        "detr_decoder_config": Sam3DETRDecoderConfig,
        "mask_decoder_config": Sam3MaskDecoderConfig,
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        geometry_encoder_config=None,
        detr_encoder_config=None,
        detr_decoder_config=None,
        mask_decoder_config=None,
        initializer_range=0.02,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = {}
        if isinstance(vision_config, dict):
            self.vision_config = Sam3VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config
        if text_config is None:
            text_config = {
                "vocab_size": 49408,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "projection_dim": 512,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "max_position_embeddings": 32,
                "hidden_act": "gelu",
            }
        if isinstance(text_config, dict):
            self.text_config = CLIPTextConfig(**text_config)
        else:
            self.text_config = text_config
        if geometry_encoder_config is None:
            geometry_encoder_config = {}
        if isinstance(geometry_encoder_config, dict):
            self.geometry_encoder_config = Sam3GeometryEncoderConfig(**geometry_encoder_config)
        else:
            self.geometry_encoder_config = geometry_encoder_config
        if detr_encoder_config is None:
            detr_encoder_config = {}
        if isinstance(detr_encoder_config, dict):
            self.detr_encoder_config = Sam3DETREncoderConfig(**detr_encoder_config)
        else:
            self.detr_encoder_config = detr_encoder_config
        if detr_decoder_config is None:
            detr_decoder_config = {}
        if isinstance(detr_decoder_config, dict):
            self.detr_decoder_config = Sam3DETRDecoderConfig(**detr_decoder_config)
        else:
            self.detr_decoder_config = detr_decoder_config
        if mask_decoder_config is None:
            mask_decoder_config = {}
        if isinstance(mask_decoder_config, dict):
            self.mask_decoder_config = Sam3MaskDecoderConfig(**mask_decoder_config)
        else:
            self.mask_decoder_config = mask_decoder_config
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @property
    def image_size(self):
        return self.vision_config.image_size

    @image_size.setter
    def image_size(self, value):
        self.vision_config.image_size = value


@auto_docstring(checkpoint="facebook/sam3")
class Sam3TrackerPromptEncoderConfig(PreTrainedConfig):
    base_config_key = "prompt_encoder_config"

    def __init__(
        self,
        hidden_size=256,
        image_size=1008,
        patch_size=14,
        mask_input_channels=16,
        num_point_embeddings=4,
        hidden_act="gelu",
        layer_norm_eps=1e-06,
        scale=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_input_channels = mask_input_channels
        self.num_point_embeddings = num_point_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.scale = scale


@auto_docstring(checkpoint="facebook/sam3")
class Sam3TrackerMaskDecoderConfig(PreTrainedConfig):
    base_config_key = "mask_decoder_config"

    def __init__(
        self,
        hidden_size=256,
        hidden_act="gelu",
        mlp_dim=2048,
        num_hidden_layers=2,
        num_attention_heads=8,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_multimask_outputs = num_multimask_outputs
        self.hidden_act = hidden_act
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.mlp_dim = mlp_dim
        self.attention_downsample_rate = attention_downsample_rate


@auto_docstring(checkpoint="facebook/sam3")
class Sam3TrackerConfig(PreTrainedConfig):
    model_type = "sam3_tracker"
    sub_configs = {
        "vision_config": AutoConfig,
        "prompt_encoder_config": Sam3TrackerPromptEncoderConfig,
        "mask_decoder_config": Sam3TrackerMaskDecoderConfig,
    }

    def __init__(
        self,
        vision_config=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        initializer_range=0.02,
        **kwargs,
    ):
        vision_config = (
            vision_config
            if vision_config is not None
            else {"backbone_feature_sizes": [[288, 288], [144, 144], [72, 72]]}
        )
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}
        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "sam3_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        if isinstance(prompt_encoder_config, Sam3TrackerPromptEncoderConfig):
            prompt_encoder_config = prompt_encoder_config.to_dict()
        if isinstance(mask_decoder_config, Sam3TrackerMaskDecoderConfig):
            mask_decoder_config = mask_decoder_config.to_dict()
        self.vision_config = vision_config
        self.prompt_encoder_config = Sam3TrackerPromptEncoderConfig(**prompt_encoder_config)
        self.mask_decoder_config = Sam3TrackerMaskDecoderConfig(**mask_decoder_config)
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


@auto_docstring(checkpoint="facebook/sam3")
class Sam3TrackerVideoPromptEncoderConfig(PreTrainedConfig):
    base_config_key = "prompt_encoder_config"

    def __init__(
        self,
        hidden_size=256,
        image_size=1008,
        patch_size=14,
        mask_input_channels=16,
        num_point_embeddings=4,
        hidden_act="gelu",
        layer_norm_eps=1e-06,
        scale=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_input_channels = mask_input_channels
        self.num_point_embeddings = num_point_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.scale = scale


@auto_docstring(checkpoint="facebook/sam3")
class Sam3TrackerVideoMaskDecoderConfig(PreTrainedConfig):
    base_config_key = "mask_decoder_config"

    def __init__(
        self,
        hidden_size=256,
        hidden_act="gelu",
        mlp_dim=2048,
        num_hidden_layers=2,
        num_attention_heads=8,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_multimask_outputs = num_multimask_outputs
        self.hidden_act = hidden_act
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.mlp_dim = mlp_dim
        self.attention_downsample_rate = attention_downsample_rate


@auto_docstring(checkpoint="facebook/sam3")
class Sam3TrackerVideoConfig(PreTrainedConfig):
    model_type = "sam3_tracker_video"
    sub_configs = {
        "vision_config": AutoConfig,
        "prompt_encoder_config": Sam3TrackerVideoPromptEncoderConfig,
        "mask_decoder_config": Sam3TrackerVideoMaskDecoderConfig,
    }

    def __init__(
        self,
        vision_config=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        initializer_range=0.02,
        num_maskmem=7,
        image_size=1008,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        enable_occlusion_spatial_embedding=True,
        multimask_output_in_sam=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=True,
        max_object_pointers_in_encoder=16,
        max_cond_frame_num=4,
        enable_temporal_pos_encoding_for_object_pointers=True,
        memory_attention_hidden_size=256,
        memory_attention_num_layers=4,
        memory_attention_num_attention_heads=1,
        memory_attention_downsample_rate=1,
        memory_attention_feed_forward_hidden_size=2048,
        memory_attention_feed_forward_hidden_act="relu",
        memory_attention_dropout=0.1,
        memory_attention_rope_theta=10000,
        memory_attention_rope_feat_sizes=None,
        memory_attention_rope_dropout=0.1,
        memory_encoder_hidden_size=256,
        memory_encoder_output_channels=64,
        mask_downsampler_embed_dim=256,
        mask_downsampler_kernel_size=3,
        mask_downsampler_stride=2,
        mask_downsampler_padding=1,
        mask_downsampler_total_stride=16,
        mask_downsampler_hidden_act="gelu",
        memory_fuser_num_layers=2,
        memory_fuser_embed_dim=256,
        memory_fuser_intermediate_dim=1024,
        memory_fuser_kernel_size=7,
        memory_fuser_padding=3,
        memory_fuser_layer_scale_init_value=1e-06,
        memory_fuser_hidden_act="gelu",
        **kwargs,
    ):
        vision_config = (
            vision_config
            if vision_config is not None
            else {"backbone_feature_sizes": [[288, 288], [144, 144], [72, 72]]}
        )
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}
        memory_attention_rope_feat_sizes = (
            [72, 72]
            if memory_attention_rope_feat_sizes is None
            else memory_attention_rope_feat_sizes
        )
        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "sam3_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        if isinstance(prompt_encoder_config, Sam3TrackerVideoPromptEncoderConfig):
            prompt_encoder_config = prompt_encoder_config.to_dict()
        if isinstance(mask_decoder_config, Sam3TrackerVideoMaskDecoderConfig):
            mask_decoder_config = mask_decoder_config.to_dict()
        self.vision_config = vision_config
        self.prompt_encoder_config = Sam3TrackerVideoPromptEncoderConfig(**prompt_encoder_config)
        self.mask_decoder_config = Sam3TrackerVideoMaskDecoderConfig(**mask_decoder_config)
        self.initializer_range = initializer_range
        self.num_maskmem = num_maskmem
        self.image_size = image_size
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.max_object_pointers_in_encoder = max_object_pointers_in_encoder
        self.max_cond_frame_num = max_cond_frame_num
        self.enable_occlusion_spatial_embedding = enable_occlusion_spatial_embedding
        self.enable_temporal_pos_encoding_for_object_pointers = (
            enable_temporal_pos_encoding_for_object_pointers
        )
        self.memory_attention_hidden_size = memory_attention_hidden_size
        self.memory_attention_num_layers = memory_attention_num_layers
        self.memory_attention_num_attention_heads = memory_attention_num_attention_heads
        self.memory_attention_downsample_rate = memory_attention_downsample_rate
        self.memory_attention_feed_forward_hidden_size = memory_attention_feed_forward_hidden_size
        self.memory_attention_feed_forward_hidden_act = memory_attention_feed_forward_hidden_act
        self.memory_attention_dropout = memory_attention_dropout
        self.memory_attention_rope_theta = memory_attention_rope_theta
        self.memory_attention_rope_feat_sizes = memory_attention_rope_feat_sizes
        self.memory_attention_rope_dropout = memory_attention_rope_dropout
        self.memory_encoder_hidden_size = memory_encoder_hidden_size
        self.memory_encoder_output_channels = memory_encoder_output_channels
        self.mask_downsampler_embed_dim = mask_downsampler_embed_dim
        self.mask_downsampler_kernel_size = mask_downsampler_kernel_size
        self.mask_downsampler_stride = mask_downsampler_stride
        self.mask_downsampler_padding = mask_downsampler_padding
        self.mask_downsampler_total_stride = mask_downsampler_total_stride
        self.mask_downsampler_hidden_act = mask_downsampler_hidden_act
        self.memory_fuser_num_layers = memory_fuser_num_layers
        self.memory_fuser_embed_dim = memory_fuser_embed_dim
        self.memory_fuser_intermediate_dim = memory_fuser_intermediate_dim
        self.memory_fuser_kernel_size = memory_fuser_kernel_size
        self.memory_fuser_padding = memory_fuser_padding
        self.memory_fuser_layer_scale_init_value = memory_fuser_layer_scale_init_value
        self.memory_fuser_hidden_act = memory_fuser_hidden_act
        super().__init__(**kwargs)

    @property
    def image_size(self):
        return self.vision_config.image_size

    @image_size.setter
    def image_size(self, value):
        self.prompt_encoder_config.image_size = value
        self.vision_config.image_size = value
        patch_size = self.vision_config.backbone_config.patch_size
        self.vision_config.backbone_feature_sizes = [
            [4 * value // patch_size, 4 * value // patch_size],
            [2 * value // patch_size, 2 * value // patch_size],
            [value // patch_size, value // patch_size],
        ]
        self.memory_attention_rope_feat_sizes = [value // patch_size, value // patch_size]
        self.__dict__["image_size"] = value


"""SAM3 Video model configuration"""
logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/sam3")
class Sam3VideoConfig(PreTrainedConfig):
    model_type = "sam3_video"
    is_composition = True
    sub_configs = {"detector_config": AutoConfig, "tracker_config": AutoConfig}

    def __init__(
        self,
        detector_config=None,
        tracker_config=None,
        initializer_range=0.02,
        low_res_mask_size=288,
        score_threshold_detection=0.5,
        det_nms_thresh=0.1,
        assoc_iou_thresh=0.1,
        trk_assoc_iou_thresh=0.5,
        new_det_thresh=0.7,
        recondition_on_trk_masks=True,
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=True,
        init_trk_keep_alive=30,
        max_trk_keep_alive=30,
        min_trk_keep_alive=-1,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        decrease_trk_keep_alive_for_empty_masklets=False,
        fill_hole_area=16,
        max_num_objects=10000,
        recondition_every_nth_frame=16,
        high_conf_thresh=0.8,
        high_iou_thresh=0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if detector_config is None:
            detector_config = {}
            logger.info("detector_config is None. Initializing the Sam3Config with default values.")
        if isinstance(detector_config, dict):
            detector_config["model_type"] = detector_config.get("model_type", "sam3")
            self.detector_config = CONFIG_MAPPING[detector_config["model_type"]](**detector_config)
        elif isinstance(detector_config, PreTrainedConfig):
            self.detector_config = detector_config
        else:
            raise ValueError(
                f"detector_config must be a dict or Sam3Config, got {type(detector_config)}"
            )
        if tracker_config is None:
            tracker_config = {}
            logger.info(
                "tracker_config is None. Initializing the Sam3TrackerVideoConfig with default values."
            )
        if isinstance(tracker_config, dict):
            tracker_config["model_type"] = tracker_config.get("model_type", "sam3_tracker_video")
            self.tracker_config = CONFIG_MAPPING[tracker_config["model_type"]](**tracker_config)
        elif isinstance(tracker_config, PreTrainedConfig):
            self.tracker_config = tracker_config
        else:
            raise ValueError(
                f"tracker_config must be a dict or Sam3TrackerVideoConfig, got {type(tracker_config)}"
            )
        self.initializer_range = initializer_range
        self.low_res_mask_size = low_res_mask_size
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh
        self.recondition_on_trk_masks = recondition_on_trk_masks
        if hotstart_delay > 0:
            if hotstart_unmatch_thresh > hotstart_delay:
                raise ValueError(
                    f"hotstart_unmatch_thresh ({hotstart_unmatch_thresh}) must be <= hotstart_delay ({hotstart_delay})"
                )
            if hotstart_dup_thresh > hotstart_delay:
                raise ValueError(
                    f"hotstart_dup_thresh ({hotstart_dup_thresh}) must be <= hotstart_delay ({hotstart_delay})"
                )
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.suppress_unmatched_only_within_hotstart = suppress_unmatched_only_within_hotstart
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.decrease_trk_keep_alive_for_empty_masklets = decrease_trk_keep_alive_for_empty_masklets
        self.fill_hole_area = fill_hole_area
        self.max_num_objects = max_num_objects
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.high_conf_thresh = high_conf_thresh
        self.high_iou_thresh = high_iou_thresh

    @property
    def image_size(self):
        return self.detector_config.image_size

    @image_size.setter
    def image_size(self, value):
        self.detector_config.image_size = value
        self.tracker_config.image_size = value


for _value in list(globals().values()):
    if isinstance(_value, type) and issubclass(_value, PreTrainedConfig) and _value.model_type:
        CONFIG_MAPPING[_value.model_type] = _value

__all__ = [
    name
    for name, value in globals().items()
    if isinstance(value, type) and issubclass(value, PreTrainedConfig)
]
