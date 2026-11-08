"""Apply pinned browser dependency fixes after native performance measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 5 * 3600
    while not (args.run / "final-checks/complete.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("native final checks did not finish")
        time.sleep(5)
    env = {
        **os.environ,
        "npm_config_cache": str(Path("/tmp") / ("mlx-one-alpha-npm-" + args.run.name)),
        "npm_config_registry": "https://registry.npmjs.org",
    }
    records = []

    def execute(name, command, cwd=ROOT / "ui", *, required=True):
        started = time.time()
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, text=True, timeout=600
        )
        (args.run / (name + ".log")).write_text(result.stdout + result.stderr)
        records.append(
            {
                "id": name,
                "timestamp": started,
                "command": command,
                "cwd": str(cwd),
                "exit_code": result.returncode,
                "duration": time.time() - started,
            }
        )
        print(name, result.returncode, flush=True)
        if required and result.returncode:
            raise RuntimeError(name + " failed")
        return result

    execute(
        "ui-security-install", ["npm", "install", "--ignore-scripts", "--no-fund", "--no-audit"]
    )
    audit = execute("ui-npm-audit", ["npm", "audit", "--json"], required=False)
    (args.run / "ui-npm-audit.json").write_text(audit.stdout)
    execute("ui-typecheck", ["npm", "run", "check"])
    execute("ui-tests", ["npm", "test"])
    execute("ui-build", ["npm", "run", "build"])
    modules = ROOT / "ui/node_modules"
    names = [
        "svelte",
        "@sveltejs/kit",
        "@lucide/svelte",
        "dompurify",
        "marked",
        "vite",
        "tailwindcss",
        "clsx",
        "esm-env",
        "devalue",
    ]
    inventory, texts = {}, []
    for name in names:
        path = modules / name
        meta = json.loads((path / "package.json").read_text())
        licenses = [
            p
            for p in path.iterdir()
            if p.is_file() and p.name.lower().startswith(("license", "licence", "copying"))
        ]
        if not licenses:
            raise RuntimeError("missing license: " + name)
        inventory[name] = {
            "version": meta["version"],
            "declared_license": meta.get("license"),
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in licenses},
        }
        for p in licenses:
            texts.append(
                name
                + " "
                + meta["version"]
                + " — "
                + p.name
                + "\n"
                + "=" * 72
                + "\n"
                + p.read_text()
            )
    (ROOT / "src/mlx_one/ui/dist/THIRD_PARTY_NOTICES.txt").write_text(
        "Bundled UI and build dependency notices.\nDOMPurify is available under Apache-2.0 OR "
        "MPL-2.0; this distribution elects Apache-2.0.\n\n" + "\n\n".join(texts) + "\n"
    )
    (args.run / "ui-license-inventory-final.json").write_text(json.dumps(inventory, indent=2))
    execute(
        "post-ui-packaging-tests",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--no-cov",
            "tests/alpha_0_1/test_packaging_contract.py",
            "tests/test_server.py",
            "tests/api",
        ],
        cwd=ROOT,
    )
    (args.run / "ui-security-complete.json").write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "status": "PASS",
                "records": records,
                "inventory": inventory,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
