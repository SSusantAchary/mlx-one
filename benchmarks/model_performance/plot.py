"""Export evidence-backed text-model size/decode charts (optional matplotlib).

Run from the repository root: python -m benchmarks.model_performance.plot
Historical, standardized and HTTP results are shown separately, never pooled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from benchmarks.model_performance.report import ROOT, load_records

BLUE = "#1117AC"
INK = "#25304A"
GRAY = "#687187"
GRID = "#DDE0ED"
COLORS = {"Q4": "#7638C6", "Q8": "#3981F7", "FP32": "#FF674E", "BF16": "#42A76A"}
NAMES = {
    "M01": "GPT-2",
    "M02": "Qwen2.5",
    "M03": "Qwen3",
    "M04": "LFM2",
    "M05": "Qwen3.5",
    "M06": "SmolLM2",
    "M07": "Llama 3.2",
    "M08": "Qwen2.5",
    "M09": "Gemma 3",
    "M10": "Phi-3 mini",
    "M11": "LFM2.5 24B-A2B",
    "M12": "GPT-OSS MXFP4/Q4",
    "M13": "GPT-OSS TQ3",
}


def parameters_billion(label):
    match = re.match(r"^([\d.]+)([MB])(?:\s|$)", label)
    if not match:
        raise ValueError("unknown nominal parameter label: " + label)
    return float(match[1]) / (1000 if match[2] == "M" else 1)


def precision(record):
    profile = record["data"]["profile"]
    if profile == "affine_q4":
        return "Q4"
    if profile == "affine_q8":
        return "Q8"
    return record["model"].get("source_precision", "FP32")


def data():
    inventory, records = load_records()  # Validates revisions, hashes, counts and medians.
    points = []
    for r in records:
        e = r["environment"]
        points.append(
            {
                "id": r["model"]["id"],
                "checkpoint": r["model"]["model_id"],
                "revision": r["model"]["revision"],
                "parameters": r["model"]["parameters"],
                "parameters_billion": parameters_billion(r["model"]["parameters"]),
                "precision": precision(r),
                "primary": r["primary"],
                "decode_tokens_per_second": statistics.median(
                    s["decode_tokens_per_second"] for s in r["samples"]
                ),
                "protocol": e["protocol"],
                "measurement_type": "direct-model",
                "actual_input_tokens": [s["input_tokens"] for s in r["samples"]],
                "actual_output_tokens": [s["generated_tokens"] for s in r["samples"]],
                "environment": {
                    key: e[key]
                    for key in (
                        "hardware",
                        "packages",
                        "platform",
                        "python",
                        "backend",
                        "protocol",
                        "repetitions",
                        "warmups_per_shape",
                        "statistic",
                    )
                },
                "source_fingerprint": r["source_fingerprint"],
                "runner_sha256": r["runner_hash"],
                "evidence": r["path"],
                "evidence_sha256": hashlib.sha256((ROOT / r["path"]).read_bytes()).hexdigest(),
            }
        )
    http_path = ROOT / "docs/validation/gpt-oss-kilo-2026-10-23.json"
    http = json.loads(http_path.read_text())
    samples = http["hardware"]["checks"]["stream_tools"]
    if len(samples) != 5 or any(
        s["status"] != "PASS"
        or s["usage"]["prompt_tokens"] != 175
        or s["usage"]["completion_tokens"] != 91
        or s["native_metrics"]["prompt_tokens"] != 175
        or s["native_metrics"]["generated_tokens"] != 91
        or not math.isfinite(s["native_metrics"]["decode_tokens_per_second"])
        or s["native_metrics"]["decode_tokens_per_second"] <= 0
        for s in samples
    ):
        raise ValueError("HTTP evidence no longer matches its separately labeled protocol")
    serving = {
        "checkpoint": "mlx-community/gpt-oss-20b-MXFP4-Q4",
        "revision": None,
        "parameters_billion": 20,
        "active_parameters_billion": 3.6,
        "precision": "MXFP4 / Q4–Q8",
        "measurement_type": "native-http-serving",
        "input_tokens": 175,
        "output_tokens": 91,
        "repetitions": 5,
        "warmups": 1,
        "decode_tokens_per_second": statistics.median(
            s["native_metrics"]["decode_tokens_per_second"] for s in samples
        ),
        "hardware": http["host"],
        "mlx_version": None,
        "limitations": "Local checkpoint revision and MLX version not recorded; context only",
        "evidence": http_path.relative_to(ROOT).as_posix(),
        "evidence_sha256": hashlib.sha256(http_path.read_bytes()).hexdigest(),
    }
    missing = [
        {
            "id": m["id"],
            "checkpoint": m["model_id"],
            "revision": m["revision"],
            "parameters": m["parameters"],
            "precision": m["primary_precision"],
            "decode_tokens_per_second": None,
            "status": "HTTP only; no comparable direct-model result"
            if m["id"] == "M12"
            else "No comparable primary-precision measurement",
        }
        for m in inventory
        if not any(p["id"] == m["id"] and p["primary"] for p in points)
    ]
    return {
        "schema": "mlx-one-text-performance-plot-v1",
        "inventory_count": len(inventory),
        "size_basis": "Nominal total parameters, never quantized checkpoint bytes",
        "direct_model_points": points,
        "separate_http_context": serving,
        "missing_direct_model": missing,
    }


def setup():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "mlx-one-text-performance-v1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#929ACA",
            "axes.labelcolor": BLUE,
            "xtick.color": INK,
            "ytick.color": INK,
            "savefig.facecolor": "white",
        }
    )


def frame(dataset, title):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(19.6, 11.2), facecolor="white")
    fig.text(0.055, 0.941, title, fontsize=30, weight="bold", color=BLUE)
    fig.text(
        0.055,
        0.905,
        "Apple M4 · 32 GiB unified memory · Experimental measurements",
        fontsize=15,
        color=GRAY,
    )
    fig.text(
        0.075, 0.846, "Historical direct-model inference", fontsize=19, color=BLUE, weight="bold"
    )
    fig.text(
        0.075,
        0.816,
        "128 input / 16 output tokens · batch 1 · 1 warmup + 3 repetitions · median",
        fontsize=12,
        color=GRAY,
    )
    fig.text(0.775, 0.846, "Separate protocols", fontsize=18, color=BLUE, weight="bold")

    standard = [p for p in dataset["direct_model_points"] if p["protocol"] == "standard-128-128"]
    if len(standard) != 1 or standard[0]["id"] != "M01":
        raise ValueError("standardized coverage changed; update the explicit chart card")
    http = dataset["separate_http_context"]
    cards = [
        (
            0.615,
            "Standardized direct model",
            standard[0]["decode_tokens_per_second"],
            "GPT-2 · 124M · FP32",
            "128 input / 128 output tokens",
            "1 warmup + 3 repetitions · median",
            "MLX 0.32.2 · macOS 15.8.1",
        ),
        (
            0.350,
            "Native HTTP serving",
            http["decode_tokens_per_second"],
            "GPT-OSS 20B · MXFP4 / Q4–Q8",
            "175 input / 91 output tokens",
            "1 warmup + 5 repetitions · median",
            "Revision and MLX version unrecorded",
        ),
    ]
    for y, heading, speed, name, counts, repeats, note in cards:
        fig.add_artist(
            FancyBboxPatch(
                (0.76, y),
                0.217,
                0.185,
                transform=fig.transFigure,
                boxstyle="round,pad=0.012,rounding_size=0.009",
                edgecolor=GRID,
                facecolor="#F6F7FC",
                linewidth=1,
            )
        )
        fig.text(0.775, y + 0.156, heading, fontsize=12, color=BLUE, weight="bold")
        fig.text(0.775, y + 0.112, f"{speed:.2f}", fontsize=30, color=BLUE, weight="bold")
        fig.text(0.899, y + 0.12, "tokens/sec", fontsize=11, color=GRAY)
        fig.text(0.775, y + 0.081, name, fontsize=11, color=INK, weight="bold")
        fig.text(0.775, y + 0.057, counts, fontsize=10, color=GRAY)
        fig.text(0.775, y + 0.035, repeats, fontsize=10, color=GRAY)
        fig.text(0.775, y + 0.012, note, fontsize=9, color=GRAY)
    fig.text(
        0.768, 0.301, "Different protocols; compare within each cohort.", fontsize=10, color=GRAY
    )
    fig.text(
        0.055,
        0.218,
        "No comparable direct-model speed published",
        fontsize=16,
        color=BLUE,
        weight="bold",
    )
    for i, m in enumerate(dataset["missing_direct_model"]):
        x = 0.055 + i * 0.235
        fig.add_artist(
            FancyBboxPatch(
                (x, 0.087),
                0.216,
                0.105,
                transform=fig.transFigure,
                boxstyle="round,pad=0.01,rounding_size=0.006",
                edgecolor=GRID,
                facecolor="white",
                linewidth=1,
            )
        )
        fig.text(x + 0.005, 0.160, NAMES[m["id"]], fontsize=12, weight="bold", color=INK)
        fig.text(x + 0.005, 0.132, m["parameters"], fontsize=11, color=GRAY)
        label = (
            "HTTP result shown separately" if m["id"] == "M12" else "Unmeasured · no speed inferred"
        )
        fig.text(x + 0.005, 0.104, label, fontsize=10, color=GRAY)
    fig.text(
        0.055,
        0.043,
        "Historical: MLX 0.32.2 · macOS 15.7.9 · native MLX diagnostic dense · FP16 KV. "
        "Each model retains its recorded precision; size alone does not explain speed.",
        fontsize=10,
        color=GRAY,
    )
    fig.text(
        0.055,
        0.019,
        "All 13 text checkpoints are inventoried. Missing speed is not zero. "
        "Evidence, immutable direct-model revisions and chart data: docs/benchmarks/figures/.",
        fontsize=10,
        color=GRAY,
    )
    return fig


def scatter(dataset):
    from matplotlib.lines import Line2D

    fig = frame(dataset, "MLX-One | Model size vs decode speed")
    ax = fig.add_axes((0.075, 0.345, 0.635, 0.435))
    historical = [p for p in dataset["direct_model_points"] if p["protocol"] == "historical-128-16"]
    offsets = {
        "M01": (20, 23),
        "M04": (20, -18),
        "M02": (-55, 36),
        "M03": (-20, -53),
        "M07": (-86, 28),
        "M09": (-16, -47),
        "M08": (-5, 38),
        "M06": (20, -31),
        "M10": (-22, 33),
    }
    for p in historical:
        x, y = p["parameters_billion"], p["decode_tokens_per_second"]
        color = COLORS[p["precision"]] if p["primary"] else "#A0A7B9"
        ax.scatter(
            x,
            y,
            s=125 if p["primary"] else 75,
            facecolors=color if p["primary"] else "white",
            edgecolors="white" if p["primary"] else color,
            linewidths=1.8,
            zorder=4,
        )
        if p["primary"]:
            text = f"{NAMES[p['id']]} · {p['precision']}\n{y:.2f} TPS"
            ax.annotate(
                text,
                (x, y),
                xytext=offsets[p["id"]],
                textcoords="offset points",
                fontsize=10.5,
                fontweight="bold",
                color=INK,
                zorder=5,
                arrowprops={"arrowstyle": "-", "color": "#B4BACB", "lw": 0.8},
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            )
    ax.set_xscale("log")
    ax.set_xlim(0.095, 5.6)
    ax.set_ylim(0, 214)
    ticks = [0.125, 0.25, 0.5, 1, 2, 4]
    ax.set_xticks(ticks, ["0.125", "0.25", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.grid(axis="both", color=GRID, linestyle=(0, (2, 4)), linewidth=1)
    ax.set_xlabel(
        "Nominal model size (billions of parameters; log scale)", labelpad=13, fontsize=13
    )
    ax.set_ylabel("Decode tokens per second", labelpad=12, fontsize=14)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=c, markersize=8, label=k)
        for k, c in COLORS.items()
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            color="#A0A7B9",
            markersize=7,
            label="Stored BF16 (secondary)",
        )
    )
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10, ncol=2)
    return fig


def bars(dataset):
    from matplotlib.patches import Patch

    fig = frame(dataset, "MLX-One | Decode speed by model size")
    ax = fig.add_axes((0.075, 0.345, 0.635, 0.435))
    points = sorted(
        (
            p
            for p in dataset["direct_model_points"]
            if p["primary"] and p["protocol"] == "historical-128-16"
        ),
        key=lambda p: (p["parameters_billion"], p["id"]),
    )
    labels = []
    for i, p in enumerate(points):
        value = p["decode_tokens_per_second"]
        ax.bar(i, value, width=0.66, color=COLORS[p["precision"]], zorder=3)
        ax.text(
            i,
            value - 8,
            f"{value:.2f}",
            ha="center",
            va="top",
            color="white",
            fontsize=12,
            fontweight="bold",
            zorder=4,
        )
        size = p["parameters_billion"]
        size_label = f"{size * 1000:g}M" if size < 1 else f"{size:g}B"
        labels.append(f"{NAMES[p['id']]}\n{size_label}\n{p['precision']}")
    ax.set_xticks(range(len(points)), labels, fontsize=10)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.set_ylim(0, 214)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.grid(axis="y", color=GRID, linestyle=(0, (2, 4)), linewidth=1)
    ax.set_ylabel("Decode tokens per second", labelpad=12, fontsize=14)
    ax.legend(
        handles=[Patch(color=c, label=k) for k, c in COLORS.items()],
        frameon=False,
        loc="upper right",
        ncol=4,
        fontsize=10,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/benchmarks/figures")
    args = parser.parse_args()
    dataset = data()
    setup()
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, renderer in [
        ("text-model-size-vs-decode", scatter),
        ("text-model-decode-bars", bars),
    ]:
        fig = renderer(dataset)
        for extension in ("png", "svg"):
            path = args.output_dir / f"{name}.{extension}"
            fig.savefig(
                path,
                dpi=160,
                metadata={
                    "Description": "Measured MLX-One results; protocols separate",
                    **({"Date": None} if extension == "svg" else {}),
                },
            )
            print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        plt.close(fig)
    (args.output_dir / "text-model-performance-data.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
