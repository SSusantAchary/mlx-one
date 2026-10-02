"""Lazy native image segmentation and video tracking APIs."""

from importlib import import_module

_EXPORTS = {
    "load_segmentation_model": ("loading", "load_segmentation_model"),
    "LoadedSegmentationModel": ("loading", "LoadedSegmentationModel"),
    "SegmentationModelLoadError": ("loading", "SegmentationModelLoadError"),
    "segment_image": ("inference", "segment_image"),
    "predict_masks": ("inference", "predict_masks"),
    "generate_masks": ("inference", "generate_masks"),
    "Sam3VideoSession": ("video", "Sam3VideoSession"),
    "ConceptSegmentationResult": ("schemas", "ConceptSegmentationResult"),
    "InteractiveSegmentationResult": ("schemas", "InteractiveSegmentationResult"),
    "VideoFrameResult": ("schemas", "VideoFrameResult"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value
