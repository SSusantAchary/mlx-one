"""SAM3 tensor selection and strict reference-layout weight contracts."""

from __future__ import annotations

from mlx_one.core.weights import WeightContract


def model_for_config(config):
    from .model import Sam3Model
    from .tracker import Sam3TrackerModel
    from .tracker_video import Sam3TrackerVideoModel
    from .video import Sam3VideoModel

    return {
        "sam3": Sam3Model,
        "sam3_tracker": Sam3TrackerModel,
        "sam3_tracker_video": Sam3TrackerVideoModel,
        "sam3_video": Sam3VideoModel,
    }[config.model_type](config)


def weight_contract(config):
    return WeightContract(
        {key: value.shape for key, value in model_for_config(config).state_dict().items()}
    )


def sanitize_weights(weights, config):
    """Select a component from a composite snapshot, never ignore unknown keys."""
    composite = any(key.startswith("detector_model.") for key in weights)
    if not composite or config.model_type == "sam3_video":
        return dict(weights)
    prefixes = ("detector_model.", "tracker_model.", "tracker_neck.")
    unexpected = sorted(key for key in weights if not key.startswith(prefixes))
    if unexpected:
        raise ValueError(f"unrecognized composite SAM3 tensors: {unexpected}")
    if config.model_type == "sam3":
        return {
            key.removeprefix("detector_model."): value
            for key, value in weights.items()
            if key.startswith("detector_model.")
        }
    result = {}
    for key, value in weights.items():
        if key.startswith("detector_model.vision_encoder.backbone."):
            result[key.removeprefix("detector_model.")] = value
        elif key.startswith("tracker_neck."):
            result["vision_encoder.neck." + key.removeprefix("tracker_neck.")] = value
        elif key.startswith("tracker_model."):
            result[key.removeprefix("tracker_model.")] = value
    if config.model_type == "sam3_tracker":
        # Interactive inference does not use temporal memory parameters.
        from .config import Sam3TrackerVideoConfig

        full_config = Sam3TrackerVideoConfig.from_dict(
            {**config.to_dict(), "model_type": "sam3_tracker_video"}
        )
        weight_contract(full_config).validate(result)
        expected = weight_contract(config).expected
        result = {key: value for key, value in result.items() if key in expected}
    return result
