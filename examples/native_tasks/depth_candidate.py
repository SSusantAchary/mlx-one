"""Raw blocked relative-depth candidate. Produces a bounded NPZ, not metric meters."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_one.models.depth.depth_anything import REVISION, load_depth
from mlx_one.models.depth.interpolation import resize
from mlx_one.vision.depth_processing import preprocess_depth

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("image", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
pixels, size = preprocess_depth(args.image)
model = load_depth(args.checkpoint)
result = model(mx.array(pixels[None]))
depth = result["predicted_depth"]
original = resize(depth[..., None], (size[1], size[0]), mode="bicubic")[0, :, :, 0]
mx.eval(original)
with args.output.open("xb") as stream:
    np.savez_compressed(stream, relative_depth=np.asarray(original))
print(
    json.dumps(
        {
            "revision": REVISION,
            "semantics": "relative depth; no metric calibration",
            "shape": list(original.shape),
            "dtype": str(original.dtype),
            "artifact": str(args.output),
            "decision": "BLOCKED; intermediate FP32 gate",
        }
    )
)
