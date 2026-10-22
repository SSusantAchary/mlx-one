"""Offline candidate embedding API; this example is not checkpoint qualification."""
import argparse

from mlx_one import embed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    result = embed(args.model, ["A small cat.", "A kitten."],
                   revision=args.revision, offline=True)
    print(result.to_json())


if __name__ == "__main__":
    main()
