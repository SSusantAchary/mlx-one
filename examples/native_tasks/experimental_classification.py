"""Run locally downloaded, pinned experimental classifiers; no remote downloads."""

import argparse

from mlx_one.models.classification.distilbert import load_distilbert_classifier
from mlx_one.vision.vit_classification import load_vit_classifier

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("vit_checkpoint")
parser.add_argument("image")
parser.add_argument("distilbert_checkpoint")
args = parser.parse_args()
print(load_vit_classifier(args.vit_checkpoint).predict(args.image, top_k=3))
print(load_distilbert_classifier(args.distilbert_checkpoint).predict("A delightful film."))
