"""Offline reference text example; executing it loads the selected local weights."""
import argparse

from mlx_one import CacheConfig, Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local checkpoint directory or cached canonical model ID")
    parser.add_argument("--revision", required=True, help="Immutable checkpoint revision")
    parser.add_argument("--prompt", default="Explain why the sky is blue in one sentence.")
    args = parser.parse_args()
    with Engine(args.model, revision=args.revision, offline=True,
                inference_mode="reference_dense",
                cache_config=CacheConfig(prefix_cache_enabled=False)) as engine:
        print(engine.generate(args.prompt, max_tokens=32, temperature=0).text)


if __name__ == "__main__":
    main()
