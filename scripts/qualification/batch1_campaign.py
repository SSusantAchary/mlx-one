"""Serial, pinned Batch 1 retrieval qualification; remove only owned downloads.

The metadata directory contains the pinned Hub inventories captured at planning.
This controller imports no inference framework; native and reference workers run
in separate environments. Evidence must live outside the disposable model root.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--native-python", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path, required=True)
    parser.add_argument("--nomic-reference", type=Path)
    parser.add_argument("--image-fixtures", type=Path)
    parser.add_argument("--models", nargs="+", default=["B01", "B02", "B04", "B10"])
    parser.add_argument(
        "--installed", action="store_true", help="Use installed native wheel, not source"
    )
    args = parser.parse_args()
    work = args.work.resolve()
    evidence = args.evidence.resolve()
    if evidence.is_relative_to(work):
        raise ValueError("evidence must be outside disposable model work")
    work.mkdir(parents=True, exist_ok=True)
    marker = work / ".batch1-owned"
    if not marker.exists():
        if list(work.iterdir()):
            raise ValueError("cannot adopt an existing directory")
        marker.write_text("mlx-one-batch1-v1\n")
    if marker.read_text() != "mlx-one-batch1-v1\n":
        raise ValueError("ownership marker mismatch")
    evidence.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(Path("src").resolve()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if args.installed:
        env.pop("PYTHONPATH", None)
    scripts = Path(__file__).resolve().parent
    for key in args.models:
        if key not in {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09", "B10"}:
            raise ValueError("unsupported retrieval checkpoint in this worker")
        metadata = json.loads((args.metadata / key / "summary.json").read_text())
        root = work / key
        if root.exists():
            raise ValueError("refuse to overwrite existing model work")
        root.mkdir()
        (root / ".batch1-owned").write_text(key)
        dest = evidence / key
        dest.mkdir(exist_ok=True)
        record = {
            "id": key,
            "checkpoint": metadata["checkpoint"],
            "revision": metadata["revision"],
            "license": metadata["license"],
            "assets": [],
            "status": "BLOCKED",
        }
        try:
            for name, info in metadata["files"].items():
                if not (
                    ("/" not in name and name.endswith((".json", ".txt", ".model")))
                    or name == "model.safetensors"
                    or (
                        key == "B03"
                        and name in {"pytorch_model.bin", "sparse_linear.pt", "colbert_linear.pt"}
                    )
                    or name == "1_Pooling/config.json"
                ):
                    continue
                path = root / name
                path.parent.mkdir(exist_ok=True, parents=True)
                url = f"https://huggingface.co/{metadata['checkpoint']}/resolve/{metadata['revision']}/{name}"
                with (
                    urllib.request.urlopen(url, timeout=120) as response,
                    path.open("wb") as output,
                ):
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                actual = digest(path)
                expected = info.get("lfs", {}).get("sha256") or metadata["metadata_sha256"].get(
                    name
                )
                if expected is not None and actual != expected:
                    raise ValueError("checksum mismatch: " + name)
                record["assets"].append(
                    dict(
                        path=name,
                        bytes=path.stat().st_size,
                        sha256=actual,
                        pinned_remote_sha256=expected,
                    )
                )
                print(key, "downloaded", name, path.stat().st_size, flush=True)
            native_root = root
            if key == "B03":
                native_root = root / "native"
                command = [
                    str(args.reference_python),
                    str(scripts / "batch1_convert.py"),
                    "--kind",
                    "bge-m3",
                    "--source",
                    str(root),
                    "--inventory",
                    str(args.metadata / key / "summary.json"),
                    "--output",
                    str(native_root),
                    "--report",
                    str(dest / "conversion.json"),
                ]
                with (dest / "conversion.log").open("w") as log:
                    subprocess.run(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=600,
                    )
            prefix = (
                {"query_prefix": "query: ", "document_prefix": "passage: "} if key == "B10" else {}
            )
            if key == "B09":
                prefix = dict(query_prefix="search_query: ", document_prefix="search_document: ")
            (root / "mlx_one_retrieval.json").write_text(json.dumps(prefix))
            documents = [
                "The capital of France is Paris.",
                "Bananas are yellow fruit.",
                "A compiler translates computer programs.",
                "Berlin is the capital of Germany.",
                "Madrid is the capital of Spain.",
            ]
            queries = [
                "What is the capital of France?",
                "Quelle est la capitale de la France?",
                "Was ist die Hauptstadt von Frankreich?",
            ]
            rerank = key in {"B01", "B07"}
            pooling = (
                json.loads((root / "1_Pooling/config.json").read_text())
                if not rerank and key not in {"B05", "B06"}
                else {}
            )
            max_length = (
                json.loads((root / "sentence_bert_config.json").read_text()).get(
                    "max_seq_length", 512
                )
                if (root / "sentence_bert_config.json").exists()
                else 512
            )
            spec = dict(
                task="reranking" if rerank else "embedding",
                query=queries[0],
                queries=queries,
                documents=documents,
                max_length=max_length,
                pooling="cls" if pooling.get("pooling_mode_cls_token") else "mean",
                normalize=key != "B04",
                bge_m3=key == "B03",
            )
            spec["first"] = (
                [queries[0]] * len(documents)
                if rerank
                else [prefix.get("query_prefix", "") + text for text in queries]
                + [prefix.get("document_prefix", "") + text for text in documents]
            )
            spec["second"] = documents if rerank else None
            (dest / "input.json").write_text(json.dumps(spec, indent=2) + "\n")
            for kind, python in [
                ("reference", args.reference_python),
                ("native", args.native_python),
            ]:
                output = dest / ("reference.npz" if kind == "reference" else "native.json")
                command = [
                    str(python),
                    str(scripts / f"batch1_{kind}.py"),
                    "--model",
                    str(native_root if kind == "native" else root),
                    "--input",
                    str(dest / "input.json"),
                    "--output",
                    str(output),
                ]
                if key in {"B05", "B06"}:
                    command = [
                        str(python),
                        str(scripts / "batch1_modality.py"),
                        "--mode",
                        kind,
                        "--task",
                        "t5" if key == "B05" else "vit",
                        "--model",
                        str(root),
                        "--output",
                        str(output),
                    ]
                    if key == "B06":
                        if args.image_fixtures is None:
                            raise ValueError("ViT requires the pinned image fixture directory")
                        command += ["--fixtures", str(args.image_fixtures)]
                if kind == "native":
                    command += ["--reference", str(dest / "reference.npz")]
                    if args.installed:
                        command += ["--check-interfaces"]
                elif key == "B09" and args.nomic_reference is not None:
                    command += ["--nomic-reference", str(args.nomic_reference)]
                with (dest / f"{kind}.log").open("w") as log:
                    run = subprocess.run(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200
                    )
                record[kind + "_exit_code"] = run.returncode
                if run.returncode:
                    raise RuntimeError(kind + " worker failed; see retained log")
            native = json.loads((dest / "native.json").read_text())
            record["status"] = (
                "VALIDATED_SCOPED" if native["status"] == "PASS" else "FAILED_VALIDATION"
            )
            record["reference"] = json.loads((dest / "reference.json").read_text())
        except Exception as exc:
            record["blocker"] = str(exc).replace(str(work), "<owned-work>")
        finally:
            # Persist the outcome before deleting bytes; a write failure leaves
            # the owned model recoverable instead of destroying unsaved evidence.
            (dest / "result.json").write_text(json.dumps(record, indent=2) + "\n")
            if (
                root.is_symlink()
                or root.resolve().parent != work
                or (root / ".batch1-owned").read_text() != key
            ):
                raise RuntimeError("cleanup ownership failed")
            deleted = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            shutil.rmtree(root)
            (dest / "cleanup.json").write_text(
                json.dumps(
                    dict(path=key, deleted_bytes=deleted, status="REMOVED_OWNED_DOWNLOADS"),
                    indent=2,
                )
                + "\n"
            )
        print(key, record["status"], record.get("blocker", ""), flush=True)


if __name__ == "__main__":
    main()
