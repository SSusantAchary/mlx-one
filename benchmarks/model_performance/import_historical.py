"""Import preserved measurements, without executing qualification or inference."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "docs/benchmarks"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive = args.archive
    environment = json.loads((archive / "environment.json").read_text())
    manifest = {r["id"]: r for r in json.loads((archive / "checkpoint-manifest.json").read_text())}
    runner = archive / "final-tool-snapshot/benchmarks/model_parity/kv_precision_performance.py"
    (DEST / "historical/runner.py.txt").write_bytes(runner.read_bytes())
    cohort = {
        "protocol": "historical-128-16",
        "measurement_type": "direct-model",
        "backend": "native MLX diagnostic dense; full sequence logits; production_fp16 KV",
        "hardware": environment["hardware"],
        "platform": environment["platform"],
        "python": environment["python"],
        "packages": environment["packages"],
        "metal_device_info": environment["metal_device_info"],
        "entry_commit": environment["entry_commit"],
        "dirty_worktree": True,
        "environment_sha256": digest(archive / "environment.json"),
        "archive_origin": "docs/qualification/alpha_release/" + archive.name,
        "runner": "historical/runner.py.txt",
        "runner_sha256": digest(runner),
        "warmups_per_shape": 1,
        "repetitions": 3,
        "statistic": "per-column median",
        "records": {},
    }
    for source in sorted((archive / "final-checks").glob("*-performance.json")):
        data = json.loads(source.read_text())
        model = manifest[source.name.split("-")[0]]
        assert (data["checkpoint"], data["revision"]) == (model["repo"], model["revision"])
        command = source.with_name(source.name.replace(".json", "-command.json"))
        receipt = json.loads(command.read_text())
        assert receipt["exit_code"] == 0
        (DEST / "historical" / source.name).write_bytes(source.read_bytes())
        cohort["records"][source.name] = {
            "sha256": digest(source),
            "command_receipt_sha256": digest(command),
            "source_files": model["files"],
            "production_source_sha256": receipt["production_source_sha256"],
        }
    (DEST / "historical/cohort.json").write_text(json.dumps(cohort, indent=2) + "\n")


if __name__ == "__main__":
    main()
