"""Offline streaming example with deterministic bounded reference decoding."""
import argparse

from mlx_one import CacheConfig, Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local checkpoint directory or cached canonical model ID")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt", default="Write a short greeting.")
    args = parser.parse_args()
    with Engine(args.model, revision=args.revision, offline=True,
                inference_mode="reference_dense",
                cache_config=CacheConfig(prefix_cache_enabled=False)) as engine:
        for chunk in engine.stream(args.prompt, max_tokens=32, temperature=0):
            print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
