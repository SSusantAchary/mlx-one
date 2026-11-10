"""Checkpoint release claims are independent of architecture registration."""

from __future__ import annotations

import json
from importlib.resources import files


def checkpoint_release_status(model_id, revision, *, architecture=None):
    batch = json.loads(files("mlx_one").joinpath("data/batch_1_sub_6b.json").read_text())
    exact = next(
        (
            row
            for row in batch["models"]
            if row["model_id"] == model_id
            and row["revision"] == revision
            and row["model_type"] == architecture
        ),
        None,
    )
    if exact is not None:
        return {
            "architecture_registered": True,
            "checkpoint_revision": revision,
            "status": exact["status"],
            "architecture_status": "IMPLEMENTED",
            "checkpoint_status": "SUPPORTED_PRIMARY_WORKFLOW",
            "alpha_supported": False,
            "numerical_contract": exact["numerical_status"],
            "admission_policy": batch["support_standard"],
            "release_approval": batch["release_approval"],
            "candidate_profile": exact["precision"],
            "primary_task": exact["task"],
            "known_limitations": exact["known_limitations"],
            "reason": (
                "Recorded native primary workflow; "
                "development support does not approve package publication."
            ),
        }
    manifest = json.loads(
        files("mlx_one").joinpath("data/compatibility/alpha_release.json").read_text()
    )
    candidate = next(
        (
            row
            for row in manifest["checkpoints"]
            if row["model_id"] == model_id
            and row["revision"] == revision
            and row.get("model_type") == architecture
        ),
        None,
    )
    return {
        "architecture_registered": architecture is not None,
        "checkpoint_revision": revision,
        "status": candidate["status"] if candidate else "EXPERIMENTAL_NOT_QUALIFIED",
        "architecture_status": (candidate.get("architecture_status") if candidate else None),
        "checkpoint_status": (candidate.get("checkpoint_status") if candidate else "NOT_VERIFIED"),
        "alpha_supported": False,
        "numerical_contract": manifest["numerical_contract"],
        "admission_policy": manifest["admission_policy"],
        "release_approval": manifest["release_approval"],
        "candidate_profile": candidate["primary_profile"] if candidate else None,
        "reason": (
            candidate["reason"]
            if candidate
            else "No exact checkpoint/revision/architecture functional evidence is attached."
        ),
    }
