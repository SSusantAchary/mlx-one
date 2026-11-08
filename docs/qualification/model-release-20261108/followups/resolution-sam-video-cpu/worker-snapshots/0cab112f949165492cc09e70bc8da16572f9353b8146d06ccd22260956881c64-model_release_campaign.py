"""Run a pinned campaign serially; persist results before deleting owned weights.

The checkpoint plan must contain immutable revisions and Hub file metadata.
Only folders created with this campaign's ownership marker may be removed.
Existing user model folders are never modified or deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from model_release_worker import atomic, digest

PROTECTED = Path.home() / "mlx-one-models/gpt-oss-20b-MXFP4-Q4"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def selected(entry):
    if entry.get("assets"):
        return [f for f in entry["files"] if f["path"] in entry["assets"]]
    chosen = []
    for f in entry.get("files", []):
        p = Path(f["path"])
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("unsafe remote file name")
        if p.parts[0] in {"onnx", "openvino", "onnx_model", "coreml"}:
            continue
        if p.suffix == ".safetensors" and (
            len(p.parts) == 1
            or (entry["model_type"] == "lfm2_colbert" and f["path"] == "1_Dense/model.safetensors")
        ):
            chosen.append(f)
        elif p.suffix in {".json", ".txt", ".jinja", ".model", ".tiktoken"} and len(p.parts) <= 2:
            chosen.append(f)
        elif len(p.parts) == 1 and p.name.startswith(("LICENSE", "NOTICE", "README")):
            chosen.append(f)
        elif entry["model_type"] == "clip" and f["path"] == "pytorch_model.bin":
            chosen.append(f)
    return chosen


def verify(root, files):
    inventory = []
    for f in files:
        p = root / f["path"]
        if f.get("bytes") is not None and p.stat().st_size != f["bytes"]:
            raise ValueError("artifact size mismatch: " + f["path"])
        actual = digest(p)
        if f.get("sha256") and actual != f["sha256"]:
            raise ValueError("artifact checksum mismatch: " + f["path"])
        inventory.append(
            dict(
                path=f["path"],
                bytes=p.stat().st_size,
                sha256=actual,
                hub_lfs_sha256=f.get("sha256"),
            )
        )
    return inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--reference-python", type=Path)
    p.add_argument("--models", nargs="*")
    p.add_argument("--resume-owned-downloads", action="store_true")
    p.add_argument("--timeout", type=int, default=1200)
    args = p.parse_args()
    work, evidence = args.work.resolve(), args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    owned = work / "models"
    owned.mkdir(exist_ok=True)
    ownership = str(work)
    marker = owned / ".campaign-owned"
    if marker.exists():
        require(marker.read_text() == ownership, "campaign ownership differs")
    else:
        require(not list(owned.iterdir()), "refuse to adopt existing files")
        marker.write_text(ownership)
    entries = json.loads(args.plan.read_text())
    assert len({e["model_type"] for e in entries}) == len(entries)
    if args.models:
        assert set(args.models) <= {e["model_type"] for e in entries}
        entries = [e for e in entries if e["model_type"] in args.models]
    env = {
        **os.environ,
        "HF_HOME": str(work / "hf"),
        "HF_XET_CACHE": str(work / "xet"),
        "HF_HUB_DISABLE_XET": "1",
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HUB_ETAG_TIMEOUT": "30",
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
    }
    # Keep credentials in their existing file, never copy tokens into evidence.
    env["HF_TOKEN_PATH"] = str(Path.home() / ".cache/huggingface/token")
    os.environ.update(env)
    from huggingface_hub import hf_hub_download

    script = Path(__file__).with_name("model_release_worker.py").resolve()
    groups = []
    for entry in entries:
        if entry["kind"] == "segmentation":
            found = next((g for g in groups if g[0]["kind"] == "segmentation"), None)
            if found is not None:
                found.append(entry)
                continue
        groups.append([entry])
    for group in groups:
        pending = [e for e in group if not (evidence / (e["model_type"] + ".json")).exists()]
        if not pending:
            continue
        first = group[0]
        key = first["model_type"]
        assert key.replace("_", "").isalnum()
        container = owned / key
        if container.exists():
            require(args.resume_owned_downloads, "owned downloads remain; explicit resume required")
            require(
                (container / ".campaign-owned").read_text() == ownership,
                "checkpoint ownership differs",
            )
            atomic(
                evidence / (key + "-resume.json"),
                dict(owned_path=str(container), revision=first["revision"], resumed=time.time()),
            )
        else:
            container.mkdir()
            (container / ".campaign-owned").write_text(ownership)
        model = container / "checkpoint"
        model.mkdir(exist_ok=True)
        reports = []
        downloaded = 0
        stage = "download"
        started = time.time()
        try:
            files = selected(first)
            print("START", ",".join(e["model_type"] for e in pending), first["repo"], flush=True)
            preserved_files = files and all((PROTECTED / f["path"]).is_file() for f in files)
            if key == "gpt_oss" and preserved_files:
                inventory = verify(PROTECTED, files)
                model = PROTECTED
                print("REUSE_PROTECTED", model, flush=True)
            elif first.get("source") == "github":
                name = "silero_vad_16k.safetensors"
                url = f"https://raw.githubusercontent.com/{first['repo']}/{first['revision']}/src/silero_vad/data/{name}"
                urllib.request.urlretrieve(url, model / name)
                inventory = verify(
                    model,
                    [
                        dict(
                            path=name,
                            sha256="c59271c284ae9c8335d795d60e0bfdb71aaaceec578d9bd9ffc1b8153c319ea1",
                        )
                    ],
                )
                downloaded = sum(f["bytes"] for f in inventory)
            else:
                assert first.get("metadata_status") == "PASS", first.get(
                    "metadata_error", "missing metadata"
                )
                assert files and any(f["path"].endswith((".safetensors", ".bin")) for f in files), (
                    "no supported checkpoint weights"
                )

                def fetch(f, key=key, first=first, model=model):
                    print("FETCH", key, f["path"], f.get("bytes"), flush=True)
                    if (f.get("bytes") or 0) >= 128 * 1024**2:
                        from model_release_download import download

                        download(first["repo"], first["revision"], f, model)
                    else:
                        hf_hub_download(
                            first["repo"], f["path"], revision=first["revision"], local_dir=model
                        )

                with ThreadPoolExecutor(max_workers=3) as pool:
                    list(pool.map(fetch, files))
                stage = "verify"
                inventory = verify(model, files)
                downloaded = sum(f["bytes"] for f in inventory)
                atomic(
                    model / "mlx-one-download.json",
                    dict(
                        checkpoint=first["repo"],
                        revision=first["revision"],
                        assets={f["path"]: f["sha256"] for f in inventory},
                    ),
                )
            atomic(
                evidence / (key + "-assets.json"),
                dict(
                    checkpoint=first["repo"],
                    revision=first["revision"],
                    preserved_existing=model == PROTECTED,
                    files=inventory,
                ),
            )
            if key == "clip":
                stage = "convert"
                assert args.reference_python, (
                    "audited CLIP conversion requires isolated reference Python"
                )
                conversion = container / "converted"
                tool = script.parents[2] / "benchmarks/native_tasks/convert_clip_checkpoint.py"
                with (evidence / "clip-conversion.log").open("w") as log:
                    subprocess.run(
                        [str(args.reference_python), str(tool), str(model), str(conversion)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=300,
                    )
                atomic(
                    evidence / "clip-conversion.json",
                    json.loads((conversion / "conversion.json").read_text()),
                )
                model = conversion
            stage = "inference"
            for entry in pending:
                name = entry["model_type"]
                entry_path = evidence / (name + "-entry.json")
                atomic(entry_path, entry)
                output = evidence / (name + ".json")
                worker_env = {
                    **env,
                    "HF_HUB_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                    "PYTHONNOUSERSITE": "1",
                }
                worker_env.pop("PYTHONPATH", None)
                with (evidence / (name + ".log")).open("w") as log:
                    try:
                        completed = subprocess.run(
                            [
                                str(args.python),
                                "-u",
                                str(script),
                                "--entry",
                                str(entry_path),
                                "--model",
                                str(model),
                                "--fixtures",
                                str(work / "fixtures/inputs.json"),
                                "--output",
                                str(output),
                                *(
                                    ["--reference-python", str(args.reference_python)]
                                    if args.reference_python
                                    else []
                                ),
                            ],
                            cwd=work,
                            env=worker_env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout,
                        )
                        if not output.exists():
                            atomic(
                                output,
                                dict(
                                    model_type=name,
                                    status="FAIL",
                                    release_qualified=False,
                                    error=f"worker exited {completed.returncode} without a report",
                                    stage=stage,
                                ),
                            )
                    except subprocess.TimeoutExpired:
                        atomic(
                            output,
                            dict(
                                model_type=name,
                                status="TIMEOUT",
                                release_qualified=False,
                                timeout_seconds=args.timeout,
                                stage=stage,
                            ),
                        )
                result = json.loads(output.read_text())
                result.update(
                    checkpoint=entry["repo"],
                    revision=entry["revision"],
                    candidate_python=str(args.python),
                    campaign_worker_sha256=digest(script),
                )
                atomic(output, result)
                reports.append(output)
                print("RESULT", name, result["status"], result.get("error", "")[:200], flush=True)
        except Exception as exc:
            for entry in pending:
                output = evidence / (entry["model_type"] + ".json")
                if not output.exists():
                    atomic(
                        output,
                        dict(
                            model_type=entry["model_type"],
                            checkpoint=entry["repo"],
                            revision=entry.get("revision"),
                            status="FAIL",
                            release_qualified=False,
                            stage=stage,
                            error=str(exc),
                            exception=type(exc).__name__,
                        ),
                    )
                reports.append(output)
            print("ERROR", key, stage, str(exc)[:250], flush=True)
        finally:
            # Deletion follows durable report creation, including failed/partial downloads.
            require(
                all((evidence / (e["model_type"] + ".json")).is_file() for e in pending),
                "refuse cleanup without durable results",
            )
            require(
                container.parent == owned and container != PROTECTED and not container.is_symlink(),
                "refuse cleanup outside the owned checkpoint directory",
            )
            require(
                (container / ".campaign-owned").read_text() == ownership,
                "refuse cleanup after ownership changed",
            )
            reclaimed = sum(
                f.stat().st_size for f in container.rglob("*") if f.is_file() and not f.is_symlink()
            )
            cleanup = dict(
                model_types=[e["model_type"] for e in pending],
                owned_path=str(container),
                bytes_removed=reclaimed,
                downloaded_bytes_verified=downloaded,
                results_recorded_before_cleanup=True,
                protected_existing_untouched=True,
                started=started,
                recorded=time.time(),
                cleanup_complete=False,
            )
            cleanup_path = evidence / (key + "-cleanup.json")
            atomic(cleanup_path, cleanup)
            shutil.rmtree(container)
            cleanup.update(cleanup_complete=not container.exists(), finished=time.time())
            atomic(cleanup_path, cleanup)
            print("CLEANED", key, reclaimed, flush=True)
    summary = [
        json.loads(f.read_text())
        for f in evidence.glob("*.json")
        if f.stem in {e["model_type"] for e in json.loads(args.plan.read_text())}
    ]
    atomic(
        evidence / "campaign-summary.json",
        dict(
            schema_version="model-release-campaign-v1",
            expected_models=len(json.loads(args.plan.read_text())),
            executed_models=len(summary),
            functional_passed=sum(r["status"] == "PASS_FUNCTIONAL" for r in summary),
            release_qualified=sum(r.get("release_qualified", False) for r in summary),
            results=summary,
        ),
    )


if __name__ == "__main__":
    main()
