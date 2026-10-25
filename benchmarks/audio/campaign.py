"""Run the primary audio inventory serially, using already downloaded checkpoints.

No downloads, cache deletion, or source qualification campaigns are performed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keys", nargs="+", help="Default: all primary checkpoint/tasks")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    catalog = json.loads(Path("benchmarks/audio/checkpoints.json").read_text())
    entries = [e for e in catalog["entries"] if not args.keys or e["key"] in args.keys]
    if not entries or args.keys and set(args.keys) != {e["key"] for e in entries}:
        parser.error("unknown checkpoint key")
    args.output.mkdir(parents=True)
    attempts = []
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    for entry in entries:
        key = entry["key"]
        directory = "A10" if key in {"A12", "A13"} else "A08-converted" if key == "A08" else key
        output = args.output / f"{key}-{entry['workload']}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.audio.run",
            "--model",
            str(args.models_root / directory),
            "--checkpoint",
            entry["checkpoint"],
            "--revision",
            entry["revision"],
            "--name",
            entry["name"],
            "--precision",
            entry["precision"],
            "--task",
            entry["task"],
            "--workload",
            entry["workload"],
            "--fixture-language",
            "English",
            "--output",
            str(output),
        ]
        if entry["task"] != "synthesize":
            command += ["--audio", str(args.fixtures / "librispeech-10s.wav")]
        if entry["task"] != "speech-to-speech":
            text = "tts.txt" if entry["task"] == "synthesize" else "librispeech.txt"
            command += ["--text-file", str(args.fixtures / text)]
        if entry["task"] in {"synthesize", "speech-to-speech"}:
            command += ["--save-first-audio", str(args.output / f"{key}-sample.wav")]
        if key in {"A01", "A02", "A03"}:
            command += ["--language", "en"]
        elif key in {"A06", "A07", "A11"}:
            command += ["--language", "English"]
        started = time.time()
        log = args.output / f"{key}.log"
        with log.open("x") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        attempts.append(
            {
                "key": key,
                "checkpoint": entry["checkpoint"],
                "revision": entry["revision"],
                "command": command,
                "exit_code": result.returncode,
                "wall_seconds": time.time() - started,
                "record": output.name if result.returncode == 0 else None,
                "log": log.name,
            }
        )
        (args.output / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
        print(key, "MEASURED" if result.returncode == 0 else "FAILED", flush=True)
    if any(a["exit_code"] for a in attempts):
        parser.exit(1, "Some checkpoints failed; inspect attempts.json and per-model logs.\n")


if __name__ == "__main__":
    main()
