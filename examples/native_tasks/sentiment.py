"""Run the pinned local SST-2 checkpoint through the shared task service."""
import argparse

from mlx_one import ClassificationEngine

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint')
parser.add_argument('texts', nargs='+')
args = parser.parse_args()
with ClassificationEngine(args.checkpoint) as classifier:
    print(classifier.classify(args.texts).to_json())
