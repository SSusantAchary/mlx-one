"""Real SAM video task/CLI verification under the user's MLX-LM support standard."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

from model_release_campaign import selected, verify
from model_release_worker import ForbidReferences, atomic, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    work, evidence = args.work.resolve(), args.evidence.resolve()
    container = work / "models/sam3_video"
    assert container.parent == work / "models" and not container.is_symlink()
    assert (container / ".campaign-owned").read_text() == str(work)
    model = container / "checkpoint"
    entries = [
        row
        for row in json.loads(args.plan.read_text())
        if row["model_type"] in {"sam3_video", "sam3_tracker_video"}
    ]
    evidence.mkdir(parents=True, exist_ok=True)
    snapshots = evidence / "worker-snapshots"
    snapshots.mkdir(exist_ok=True)
    checksum = digest(__file__)
    shutil.copyfile(__file__, snapshots / (checksum + ".py"))
    sys.meta_path.insert(0, ForbidReferences())
    for name in ["torch", "transformers", "mlx-lm", "mlx-vlm", "mlx-audio"]:
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise AssertionError("reference dependency installed in native candidate: " + name)
    import mlx.core as mx
    import mlx_one
    import numpy as np
    from click.testing import CliRunner

    from mlx_one.cli import cli
    from mlx_one.segmentation import load_segmentation_model

    assert Path(mlx_one.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    lock = (work / ".qualification-gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert mx.metal.is_available()
    inventory = verify(model, selected(entries[0]))
    atomic(
        evidence / "sam3_video-assets.json",
        dict(
            checkpoint=entries[0]["repo"],
            revision=entries[0]["revision"],
            preserved_existing=False,
            files=inventory,
        ),
    )
    fixture = json.loads((work / "fixtures/sam-natural/manifest.json").read_text())
    mini = work / "fixtures/sam-functional-cli"
    mini.mkdir(exist_ok=True)
    paths = []
    for source in fixture["frames"][:2]:
        path = Path(source["path"])
        assert digest(path) == source["sha256"]
        target = mini / path.name
        if not target.exists():
            shutil.copyfile(path, target)
        assert digest(target) == source["sha256"]
        paths.append(target)
    started = time.time()
    reports = []
    for entry in entries:
        name = entry["model_type"]
        report = dict(
            model_type=name,
            checkpoint=entry["repo"],
            revision=entry["revision"],
            release_standard="mlx-lm-aligned-functional-v1",
            release_qualified=False,
            installed_source=str(mlx_one.__file__),
            reference_packages_absent=True,
            installed_record_sha256=hashlib.sha256(
                importlib.metadata.distribution("mlx-one").read_text("RECORD").encode()
            ).hexdigest(),
            worker_sha256=checksum,
            device=mx.device_info(),
            started=time.time(),
            quality_status="REFERENCE_AND_ACCURACY_DIAGNOSTICS_RETAINED_SEPARATELY",
            fixtures=[dict(path=str(p), sha256=digest(p)) for p in paths],
        )
        try:
            task = "tracking" if name == "sam3_tracker_video" else "video"
            bundle = load_segmentation_model(
                model, task=task, revision=entry["revision"], offline=True
            )
            modes = []
            for streaming in [False, True]:
                kwargs = dict(streaming=streaming)
                if task == "tracking":
                    kwargs["original_size"] = (540, 960)
                with bundle.video_session(None if streaming else paths, **kwargs) as session:
                    if task == "tracking":
                        session.add_object(1, points=[[400, 200]], point_labels=[1])
                        session.add_object(2, points=[[240, 260]], point_labels=[1])
                    else:
                        session.add_prompt("person")
                    results = (
                        list(session.process_frame(path) for path in paths)
                        if streaming
                        else list(session.propagate())
                    )
                    arrays = {}
                    for index, result in enumerate(results):
                        assert result.frame_index == index and result.masks.shape[-2:] == (540, 960)
                        assert len(result.object_ids) > 0 and result.masks.any()
                        assert np.isfinite(result.scores).all() and np.isfinite(result.boxes).all()
                        assert ((result.scores >= 0) & (result.scores <= 1)).all()
                        assert ((result.boxes[:, 0::2] >= 0) & (result.boxes[:, 0::2] <= 960)).all()
                        assert ((result.boxes[:, 1::2] >= 0) & (result.boxes[:, 1::2] <= 540)).all()
                        for key, value in dict(
                            masks=result.masks,
                            boxes=result.boxes,
                            scores=result.scores,
                            object_ids=np.asarray(result.object_ids),
                        ).items():
                            arrays[f"frame_{index}_{key}"] = value
                    assert len(results) == 2 and results[0].object_ids == results[1].object_ids
                    target = evidence / f"{name}-{'stream' if streaming else 'offline'}-outputs.npz"
                    with target.open("xb") as stream:
                        np.savez_compressed(stream, **arrays)
                    modes.append(
                        dict(
                            streaming=streaming,
                            frames=2,
                            output=target.name,
                            output_sha256=digest(target),
                            object_ids=results[0].object_ids,
                        )
                    )
                del session
            del bundle
            gc.collect()
            mx.clear_cache()
            output = container / ("functional-cli-" + name)
            command = ["track", str(model), str(mini), "--offline", "--output", str(output)]
            if task == "tracking":
                prompts = container / "functional-prompts.json"
                atomic(
                    prompts,
                    [
                        dict(frame_index=0, object_id=i, points=[p], point_labels=[1])
                        for i, p in [(1, [400, 200]), (2, [240, 260])]
                    ],
                )
                command += ["--mode", "tracking", "--prompts-json", str(prompts)]
            else:
                command += ["--text", "person"]
            result = CliRunner().invoke(cli, command)
            report["interfaces"] = dict(
                python="PASS",
                streaming="PASS",
                cli=dict(exit_code=result.exit_code, output=result.output),
                http="NOT_ADVERTISED_FOR_SEGMENTATION",
            )
            assert result.exit_code == 0, result.output
            manifest = json.loads((output / "manifest.json").read_text())
            assert manifest["status"] == "complete" and len(manifest["results"]) == 2
            shutil.copytree(output, evidence / (name + "-cli-output"))
            report["interfaces"]["cli"].update(status="PASS", frames=2, manifest=manifest)
            report.update(
                status="PASS_FUNCTIONAL",
                functional_status="PASS",
                modes=modes,
                peak_mlx_bytes=mx.get_peak_memory(),
            )
        except Exception as exc:
            report.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        report["seconds"] = time.time() - report["started"]
        path = evidence / (name + ".json")
        atomic(path, report)
        reports.append(path)
        print(name, report["status"], report.get("error", ""), flush=True)
    assert all(path.is_file() for path in reports)
    assert (container / ".campaign-owned").read_text() == str(work)
    cleanup = dict(
        model_types=[row["model_type"] for row in entries],
        owned_path=str(container),
        started=started,
        recorded=time.time(),
        results_recorded_before_cleanup=True,
        protected_existing_untouched=True,
        bytes_removed=sum(
            p.stat().st_size for p in container.rglob("*") if p.is_file() and not p.is_symlink()
        ),
        cleanup_complete=False,
    )
    atomic(evidence / "sam3_video-cleanup.json", cleanup)
    shutil.rmtree(container)
    cleanup.update(cleanup_complete=not container.exists(), finished=time.time())
    atomic(evidence / "sam3_video-cleanup.json", cleanup)
    old = evidence.parent / "resolution-sam-video-cpu"
    for entry in entries:
        path = old / (entry["model_type"] + ".json")
        if not path.exists():
            atomic(
                path,
                dict(
                    model_type=entry["model_type"],
                    checkpoint=entry["repo"],
                    revision=entry["revision"],
                    status="SKIPPED_OPTIONAL",
                    release_qualified=False,
                    reason="User selected MLX-LM support standard; duplicate reference replay replaced by real native task/CLI verification.",
                ),
            )
    atomic(old / "sam3_video-cleanup.json", cleanup)
    print("CLEANED", container, cleanup["bytes_removed"], flush=True)


if __name__ == "__main__":
    main()
