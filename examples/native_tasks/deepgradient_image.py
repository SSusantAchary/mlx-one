"""Generate a conceptual 768×512 editorial PNG for DeepGradient."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_one import ImageEngine
from mlx_one.image import IMAGE_PROFILES

PROMPT = (
    "A conceptual editorial illustration of machine learning in medicine. A sculptural "
    "translucent glass neural network grows into a small green botanical branch over a slate "
    "blue and warm ivory background. The image is metaphorical, with crisp silhouette and "
    "generous negative space. No text, letters, labels, logos, medical scans "
    "or diagnostic evidence."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    with ImageEngine(
        IMAGE_PROFILES[0].model, quantization=4, checkpoint=args.checkpoint, offline=args.offline
    ) as engine:
        result = engine.generate(PROMPT, width=768, height=512, seed=42)
    result.save(args.output)
    print(json.dumps(result.metadata, indent=2))


if __name__ == "__main__":
    main()
