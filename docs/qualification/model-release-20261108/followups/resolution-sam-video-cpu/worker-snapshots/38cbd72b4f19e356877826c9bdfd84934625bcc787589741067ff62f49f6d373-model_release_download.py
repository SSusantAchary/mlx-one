"""Bounded, resumable HTTP-range transfer for task-owned large checkpoint files."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

import httpx


def download(repo, revision, file, destination, *, timeout=1800):
    from huggingface_hub import get_token, hf_hub_url

    expected = file["bytes"]
    target = destination / file["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected:
        return target
    url = hf_hub_url(repo, file["path"], revision=revision)
    token = get_token()
    headers = {"Authorization": "Bearer " + token} if token else {}
    deadline = time.monotonic() + timeout
    identity = file.get("sha256") or hashlib.sha256(file["path"].encode()).hexdigest()
    parts = destination / ".cache/release-range-download" / identity
    parts.mkdir(parents=True, exist_ok=True)
    width = 64 * 1024**2
    segments = [
        (i, start, min(start + width, expected) - 1)
        for i, start in enumerate(range(0, expected, width))
    ]

    def piece(segment):
        index, start, end = segment
        path = parts / f"{index:05d}"
        if path.is_file() and path.stat().st_size == end - start + 1:
            return path
        partial = path.with_suffix(".partial")
        for attempt in range(3):
            try:
                if time.monotonic() > deadline:
                    raise TimeoutError("checkpoint download deadline exceeded")
                request_headers = {**headers, "Range": f"bytes={start}-{end}"}
                with httpx.Client(
                    follow_redirects=True, timeout=httpx.Timeout(60, connect=20)
                ) as client:
                    with client.stream("GET", url, headers=request_headers) as response:
                        if response.status_code != 206:
                            raise RuntimeError(
                                f"HTTP {response.status_code}; range response required"
                            )
                        if (
                            response.headers.get("content-range")
                            != f"bytes {start}-{end}/{expected}"
                        ):
                            raise RuntimeError("unexpected HTTP byte range")
                        size = 0
                        with partial.open("wb") as stream:
                            for data in response.iter_bytes(chunk_size=1024**2):
                                if time.monotonic() > deadline:
                                    raise TimeoutError("checkpoint download deadline exceeded")
                                size += len(data)
                                if size > end - start + 1:
                                    raise ValueError("checkpoint range exceeds expected size")
                                stream.write(data)
                        if size != end - start + 1:
                            raise ValueError("incomplete checkpoint range")
                partial.replace(path)
                print("RANGE", file["path"], index + 1, "/", len(segments), flush=True)
                return path
            except Exception as exc:
                if attempt == 2 or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"checkpoint range transfer failed: {file['path']}: {type(exc).__name__}"
                    ) from None
                time.sleep(1 + attempt)
        raise RuntimeError("unreachable transfer state")

    with ThreadPoolExecutor(max_workers=4) as pool:
        complete = list(pool.map(piece, segments))
    temporary = target.with_name(target.name + ".assembling")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for path in complete:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(4 * 1024**2), b""):
                    digest.update(chunk)
                    output.write(chunk)
    if temporary.stat().st_size != expected or (
        file.get("sha256") and digest.hexdigest() != file["sha256"]
    ):
        raise ValueError("assembled checkpoint checksum mismatch")
    temporary.replace(target)
    return target
