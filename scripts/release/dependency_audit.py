"""Inventory the installed candidate dependencies and query the public OSV database."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    packages = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        packages.append(
            {
                "ecosystem": "PyPI",
                "name": name,
                "version": dist.version,
                "license_expression": dist.metadata.get("License-Expression"),
                "license_field_first_line": (dist.metadata.get("License") or "").splitlines()[:1],
                "license_classifiers": [
                    x for x in dist.metadata.get_all("Classifier", []) if x.startswith("License ::")
                ],
                "license_files": list(dist.metadata.get_all("License-File", [])),
                "runtime_dependency": name.lower() not in {"pip", "mlx-one"},
            }
        )
    ui = json.loads(args.ui_inventory.read_text())
    packages += [
        {
            "ecosystem": "npm",
            "name": n,
            "version": v["version"],
            "license_expression": v["declared_license"],
            "runtime_dependency": True,
        }
        for n, v in ui.items()
    ]
    queries = [
        {"package": {"ecosystem": p["ecosystem"], "name": p["name"]}, "version": p["version"]}
        for p in packages
    ]
    request = urllib.request.Request(
        "https://api.osv.dev/v1/querybatch",
        data=json.dumps({"queries": queries}).encode(),
        headers={"Content-Type": "application/json"},
    )
    report = {
        "timestamp": time.time(),
        "database": "https://api.osv.dev/v1/querybatch",
        "packages": packages,
        "query": queries,
        "scope": "Exact installed [server] dependencies plus bundled UI dependency inventory; "
        "no known advisory returned does not prove absence of all vulnerabilities.",
    }
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.load(response)
        if len(data.get("results", [])) != len(packages):
            raise ValueError("OSV result count differs from submitted package count")
        for package, result in zip(packages, data["results"], strict=True):
            package["advisories"] = result.get("vulns", [])
        report["raw_response"] = data
        report["runtime_advisory_count"] = sum(
            len(p["advisories"]) for p in packages if p["runtime_dependency"]
        )
        report["tooling_advisory_count"] = sum(
            len(p["advisories"]) for p in packages if not p["runtime_dependency"]
        )
        report["status"] = "PASS" if report["runtime_advisory_count"] == 0 else "FAIL"
    except Exception as exc:
        report.update(status="BLOCKED_ENVIRONMENT", error=str(exc))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"])


if __name__ == "__main__":
    main()
