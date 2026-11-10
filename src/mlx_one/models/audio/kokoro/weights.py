"""Backend-free exact shape contract for the converted Kokoro v1 profile."""

import json
from importlib.resources import files

from mlx_one.core.weights import WeightContract


def weight_contract(config):
    metadata = json.loads(
        files("mlx_one.models.audio.kokoro").joinpath("weight_shapes.json").read_text()
    )
    return WeightContract({name: tuple(shape) for name, shape in metadata.items()})


def sanitize_weights(weights, config):
    """Converted assets already use canonical native names and layouts."""
    return dict(weights)
