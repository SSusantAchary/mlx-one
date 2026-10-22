"""Offline single-image candidate VLM example; requires exact processor assets."""
import argparse

from mlx_one import TextGenerationOptions
from mlx_one.vision import load_vlm_model, stream_vlm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    bundle = load_vlm_model(args.model, revision=args.revision, offline=True)
    for chunk in stream_vlm(bundle, "<image> Describe this image briefly.", [args.image],
                            options=TextGenerationOptions(max_tokens=32, temperature=0)):
        print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
