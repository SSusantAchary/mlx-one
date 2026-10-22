"""Call an experimental local server using its actual startup model identity."""

import argparse
import os

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8888")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("MLX_ONE_API_KEY")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(base_url=args.url, headers=headers, timeout=120) as client:
        models = client.get("/v1/models")
        models.raise_for_status()
        payload = {
            "model": models.json()["data"][0]["id"],
            "max_tokens": 16,
            "temperature": 0,
            "stream": args.stream,
        }
        route = "/v1/completions"
        if args.chat:
            route = "/v1/chat/completions"
            payload["messages"] = [{"role": "user", "content": "Say hello briefly."}]
        else:
            payload["prompt"] = "The city was"
        with client.stream("POST", route, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                print(line)


if __name__ == "__main__":
    main()
