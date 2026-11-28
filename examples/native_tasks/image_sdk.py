"""Write a PNG returned by the local image server using the OpenAI SDK."""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--prompt", default="A botanical scientific illustration")
    parser.add_argument("--size", default="768x512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    extra = {"seed": args.seed}
    if args.steps is not None:
        extra["steps"] = args.steps
    with OpenAI(
        base_url=args.url, api_key=os.environ.get("MLX_ONE_API_KEY", "local"), max_retries=0
    ) as client:
        result = client.images.generate(
            model=args.model,
            prompt=args.prompt,
            size=args.size,
            n=1,
            response_format="b64_json",
            extra_body=extra,
        )
    args.output.write_bytes(base64.b64decode(result.data[0].b64_json, validate=True))


if __name__ == "__main__":
    main()
