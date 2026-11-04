"""Internal blocked document-QA candidate; no release/task-support claim."""

import argparse
import json

from mlx_one.vision.document_qa import DocumentQAEngine

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint")
parser.add_argument("image")
parser.add_argument("question")
args = parser.parse_args()
with DocumentQAEngine(args.checkpoint) as engine:
    print(json.dumps(engine.ask(args.image, args.question).to_dict(), indent=2))
