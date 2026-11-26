"""Explicit developer diagnostics, inspection and offline tuning."""

import argparse
import json
from dataclasses import asdict

from . import native_status
from .dispatcher import choose
from .registry import Registry, read_json
from .schemas import KernelRequest, NativeConfig, NativeKernelUnavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("--probe", action="store_true")
    listing = commands.add_parser("list")
    listing.add_argument("--manifest", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--request", required=True)
    tuning = commands.add_parser("tune")
    tuning.add_argument("--output", required=True)
    tuning.add_argument("--length", type=int, default=65536)
    tuning.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    tuning.add_argument("--trials", type=int, default=30)
    tuning.add_argument("--warmup", type=int, default=5)
    tuning.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    if args.command == "status":
        result = native_status(probe=args.probe)
    elif args.command == "list":
        entries, reason = Registry(args.manifest).entries()
        status = native_status()
        qualified = []
        for entry in entries:
            try:
                data = dict(entry["request"])
                data["shape"], data["strides"] = tuple(data["shape"]), tuple(data["strides"])
                decision = choose(
                    KernelRequest(**data),
                    NativeConfig(True),
                    Registry(args.manifest),
                    status_provider=lambda: status,
                )
                if decision.backend == "mlxboost_native":
                    qualified.append(asdict(decision))
            except (KeyError, TypeError, ValueError):
                continue
        result = dict(qualified=qualified, reason=reason or status.get("reason"))
    elif args.command == "inspect":
        data = read_json(args.request)
        for key in ("shape", "strides"):
            data[key] = tuple(data[key])
        result = asdict(
            choose(KernelRequest(**data), NativeConfig(enabled=True), Registry(args.manifest))
        )
    else:
        from .autotune import export_proposal, tune

        try:
            path, record = tune(
                args.output,
                length=args.length,
                dtype=args.dtype,
                trials=args.trials,
                warmup=args.warmup,
                budget_seconds=args.seconds,
            )
        except (NativeKernelUnavailable, ValueError, TimeoutError) as exc:
            parser.exit(2, f"tuning unavailable: {exc}\n")
        result = dict(
            evidence=path.name,
            review_proposal=export_proposal(path, record).name,
            ranking=record["ranking"],
            status="EXPERIMENTAL",
            approved_for_runtime=False,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
