"""Call the existing local chat endpoint; the server must already own a loaded model."""
import argparse
import json
import os
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Exact served model ID or configured alias")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("MLX_ONE_API_KEY"):
        headers["Authorization"] = f"Bearer {token}"
    payload = {"model": args.model, "messages": [{"role": "user", "content": "Hello"}],
               "max_tokens": 32, "temperature": 0}
    request = Request(args.base_url.rstrip("/") + "/v1/chat/completions",
                      data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urlopen(request, timeout=30) as response:
        print(json.dumps(json.load(response), indent=2))


if __name__ == "__main__":
    main()
