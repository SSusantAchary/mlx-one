"""Experimental tasks; use exact local checkpoints recorded in the lockfile."""

import argparse

from mlx_one import token_classify, zero_shot_classify

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("ner_checkpoint")
parser.add_argument("clip_checkpoint")
parser.add_argument("image")
args = parser.parse_args()
print(
    token_classify(
        args.ner_checkpoint, "Marie Curie lived in Paris.", aggregation_strategy="simple"
    ).to_json()
)
print(zero_shot_classify(args.clip_checkpoint, args.image, ["cat", "dog"]).to_json())
