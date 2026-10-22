"""Experimental offline inference while the release contract awaits approval."""

import argparse

from mlx_one import CacheConfig, Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Local checkpoint directory")
    parser.add_argument("--prompt", default="The city was")
    args = parser.parse_args()
    with Engine(
        args.checkpoint,
        offline=True,
        inference_mode="reference_dense",
        cache_config=CacheConfig(prefix_cache_enabled=False),
    ) as engine:
        print(engine.generate(args.prompt, max_tokens=16).text)
        assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0


if __name__ == "__main__":
    main()
