"""Pinned P1 assets, offline profiles and isolated native qualification workers."""
from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

from mlx_one.inference_baseline import BaselineError, run_profile, source_provenance
from mlx_one.p1_profiles import CHECKPOINTS, profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",
                        choices=("assets", "profiles", "download", "convert", "run", "matrix"))
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="qwen3-small")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--inference-mode", choices=("reference_dense", "experimental_dense_batch",
                        "paged", "experimental_hybrid_batch"), default="experimental_dense_batch")
    parser.add_argument("--prompt-tokens", nargs="+", type=int, default=[128, 512, 2048, 8192])
    parser.add_argument("--max-legal-prompt", action="store_true")
    parser.add_argument("--output-tokens", nargs="+", type=int, default=[32, 128, 256])
    parser.add_argument("--slots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--chunk-tokens", nargs="+", type=int, default=[512])
    parser.add_argument("--page-tokens", nargs="+", type=int,
                        choices=(16, 32, 64, 128), default=[32])
    parser.add_argument("--workload", choices=("baseline", "parity", "prefix", "mixed", "turnover",
                        "cancellation", "slow_consumer", "multi_turn", "lifecycle",
                        "peak_calibration"), default="baseline")
    parser.add_argument("--ttft-budget-ms", type=float)
    parser.add_argument("--decode-gap-budget-ms", type=float)
    parser.add_argument("--max-queue-wait-ms", type=float)
    parser.add_argument("--max-service-gap-ms", type=float)
    parser.add_argument("--reference-run")
    parser.add_argument("--queue-size", type=int)
    parser.add_argument("--max-scheduled-tokens", type=int, default=2048)
    parser.add_argument("--max-prefill-tokens", type=int, default=1024)
    parser.add_argument("--serial-run")
    parser.add_argument("--b1-run")
    parser.add_argument("--no-prefix-run")
    parser.add_argument("--disable-prefix-reuse", action="store_true")
    parser.add_argument("--traffic-manifest")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/p1"))
    args = parser.parse_args(argv)
    asset = CHECKPOINTS[args.checkpoint]
    if args.action == "assets":
        from mlx_one.p1_profiles import asset_inventory

        print(json.dumps(asset_inventory(), indent=2))
        return 0
    if args.action == "download":
        from huggingface_hub import snapshot_download

        print(snapshot_download(asset.model_id, revision=asset.revision))
        return 0
    if args.action == "convert":
        if not asset.conversion_required or not args.checkpoint_path:
            parser.error("convert requires a SmolLM2 checkpoint and --checkpoint-path destination")
        from mlx_one.conversion import convert_text_model

        print(convert_text_model(asset.model_id, args.checkpoint_path, revision=asset.revision,
                                 quantize=True, bits=8, group_size=64))
        return 0
    if args.action == "matrix":
        from mlx_one.inference_qualification import export_matrix

        export_matrix(args.output, args.output / "matrix")
        return 0
    candidates, rejected = [], []
    traffic = {}
    if args.traffic_manifest:
        from mlx_one.traffic_replay import load_traffic_manifest

        records = load_traffic_manifest(args.traffic_manifest)
        traffic = {"traffic_manifest": str(Path(args.traffic_manifest).resolve()),
                   "traffic_identity": records["sha256"]}
    for requested_prompt in args.prompt_tokens:
        for output in args.output_tokens:
            prompt = asset.context_tokens - output if args.max_legal_prompt else requested_prompt
            for slots in args.slots:
                for chunk in args.chunk_tokens:
                    for page in args.page_tokens:
                        try:
                            candidates.append(profile(args.checkpoint, mode=args.inference_mode,
                                prompt_tokens=prompt, output_tokens=output, slots=slots,
                                chunk_tokens=chunk, page_tokens=page, workload=args.workload,
                                checkpoint_path=args.checkpoint_path,
                                ttft_budget_ms=args.ttft_budget_ms,
                                decode_gap_budget_ms=args.decode_gap_budget_ms,
                                max_queue_wait_ms=args.max_queue_wait_ms,
                                max_service_gap_ms=args.max_service_gap_ms,
                                serial_run=args.serial_run, b1_run=args.b1_run,
                                reference_run=args.reference_run, queue_size=args.queue_size,
                                max_scheduled_tokens=args.max_scheduled_tokens,
                                max_prefill_tokens=args.max_prefill_tokens,
                                no_prefix_run=args.no_prefix_run,
                                reuse_prefix=False if args.disable_prefix_reuse
                                else None, **traffic))
                        except (BaselineError, ValueError) as exc:
                            rejected.append({"prompt_tokens": prompt, "output_tokens": output,
                                "slots": slots, "chunk_tokens": chunk, "page_tokens": page,
                                "status": "rejected", "reason": str(exc)})
    candidates = list({p.profile_id: p for p in candidates}.values())
    if args.action == "profiles":
        print(json.dumps({"profiles": [p.to_dict() for p in candidates], "rejected": rejected,
                          "native_qualification": "pending"}, indent=2))
        return 0
    from mlx_one.hardware import detect_hardware

    detected = detect_hardware()
    hardware = {"chip": detected.chip, "memory_bytes": detected.memory_bytes}
    root = Path(__file__).resolve().parents[2]
    source = source_provenance(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = args.output / socket.gethostname() / args.checkpoint / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "rejected-profiles.json").write_text(json.dumps(rejected, indent=2))
    failed = False
    for item in candidates:
        outcome = run_profile(item, run_root / item.profile_id,
                              hardware=hardware, provenance=source)
        failed |= outcome["status"] != "completed"
        print(json.dumps({"profile_id": item.profile_id, **outcome}))
    return int(failed or not candidates)


if __name__ == "__main__":
    raise SystemExit(main())
