"""Render VLM and native-audio performance from validated saved measurements.

Optional plotting dependency: matplotlib. No inference, model downloads or test
campaigns. Run: python -m benchmarks.model_performance.plot_multimodal
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.audio.result_schema import validate as validate_audio
from benchmarks.model_performance.plot import BLUE, COLORS, GRAY, GRID, INK, ROOT, setup
from benchmarks.vlm.readme_generator import records as vlm_records

PALETTE = COLORS | {"FP16": "#229DAB"}
# Published nominal labels, not tensor counts or quantized checkpoint bytes.
# Multimodal component boundaries differ; peak memory includes actual loaded components.
AUDIO_SIZES = {
    "A01": (0.039, "39M", "https://huggingface.co/openai/whisper-tiny"),
    "A02": (0.809, "809M", "https://huggingface.co/openai/whisper-large-v3-turbo"),
    "A03": (0.756, "756M", "https://huggingface.co/distil-whisper/distil-large-v3"),
    "A04": (0.027, "27M", "https://huggingface.co/UsefulSensors/moonshine-tiny"),
    "A05": (0.061, "61M", "https://huggingface.co/UsefulSensors/moonshine-base"),
    "A06": (0.6, "0.6B", "https://huggingface.co/Qwen/Qwen3-ASR-0.6B"),
    "A07": (1.7, "1.7B", "https://huggingface.co/Qwen/Qwen3-ASR-1.7B"),
    "A08": (0.6, "0.6B", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2"),
    "A09": (0.6, "0.6B", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"),
    "A10": (1.5, "1.5B", "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B"),
    "A11": (0.6, "0.6B", "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B"),
    "A12": (1.5, "1.5B", "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B"),
    "A13": (1.5, "1.5B", "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B"),
}
VLM_SIZES = {
    "V01": (2, "2B"),
    "V02": (3, "3B"),
    "V03": (0.8, "0.8B"),
    "V04": (0.45, "450M"),
    "V05": (3, "3B"),
}
SHORT = {
    "V01": "Qwen2-VL",
    "V02": "Qwen2.5-VL",
    "V03": "Qwen3.5",
    "V04": "LFM2-VL",
    "V05": "LFM2.5-VL",
    "A01": "Whisper\nTiny",
    "A02": "Whisper\nv3 Turbo",
    "A03": "Distil-Whisper\nLarge v3",
    "A04": "Moonshine\nTiny",
    "A05": "Moonshine\nBase",
    "A06": "Qwen3-ASR\n0.6B",
    "A07": "Qwen3-ASR\n1.7B",
    "A08": "Parakeet TDT\nv2",
    "A09": "Parakeet TDT\nv3",
    "A10": "LFM2.5\nAudio",
}


def read(path):
    return json.loads(path.read_text())


def evidence(path):
    return {
        "evidence": path.relative_to(ROOT).as_posix(),
        "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def same_cohort(points, keys):
    identities = {json.dumps({k: p[k] for k in keys}, sort_keys=True) for p in points}
    if len(identities) != 1:
        raise ValueError("incompatible measurement cohorts; render separate plots")


def data():
    plan, records = vlm_records()
    vlm = []
    for model in plan["checkpoints"]:
        candidates = [
            r
            for r in records
            if r["checkpoint"] == model["checkpoint"] and r["workload"] == "VLM-HF-S1"
        ]
        if len(candidates) != 1 or candidates[0]["status"] != "MEASURED":
            raise ValueError("expected one measured VLM-HF-S1 record per primary checkpoint")
        r = candidates[0]
        size, label = VLM_SIZES[model["id"]]
        path = ROOT / "docs/benchmarks/vlm/results" / f"{model['id']}-VLM-HF-S1.json"
        vlm.append(
            {
                **evidence(path),
                "id": model["id"],
                "name": model["name"],
                "checkpoint": r["checkpoint"],
                "revision": r["revision"],
                "nominal_parameters_billion": size,
                "size_label": label,
                "size_basis": "Nominal checkpoint-name label from pinned VLM benchmark plan",
                "precision": r["precision"],
                "status": r["status"],
                "metrics": r["metrics"],
                "protocol": r["protocol"],
                "measurement_type": r["measurement_type"],
                "hardware": r["hardware"],
                "mlx_version": r["mlx_version"],
                "implementation_sha256": r["implementation_sha256"],
                "runner_sha256": r["runner_sha256"],
                "prompt": r["prompt"],
                "image_sha256": r["image_sha256"],
                "source_sizes": r["source_sizes"],
                "actual_input_tokens": [s["prompt_tokens"] for s in r["samples"]],
                "actual_output_tokens": [s["output_tokens"] for s in r["samples"]],
                "batch_size": r["batch_size"],
                "warmups": r["warmups"],
                "repetitions": r["repetitions"],
                "statistic": r["statistic"],
            }
        )
    same_cohort(
        vlm,
        (
            "hardware",
            "mlx_version",
            "implementation_sha256",
            "runner_sha256",
            "measurement_type",
            "protocol",
            "prompt",
            "image_sha256",
            "source_sizes",
        ),
    )
    catalog = read(ROOT / "benchmarks/audio/checkpoints.json")["entries"]
    publication = read(ROOT / "docs/benchmarks/audio/publication.json")
    audio = []
    for source in publication["records"]:
        path = ROOT / source
        r = validate_audio(read(path))
        entries = [
            e
            for e in catalog
            if e["checkpoint"] == r["checkpoint"]
            and e["revision"] == r["revision"]
            and e["task"] == r["task"]
        ]
        if len(entries) != 1 or entries[0]["precision"] != r["precision"]:
            raise ValueError("audio record differs from primary checkpoint/task inventory")
        entry = entries[0]
        size, label, url = AUDIO_SIZES[entry["key"]]
        if r["measurement_type"] != "direct-model-prepared-pcm":
            raise ValueError("Engine, file-pipeline or HTTP results require separate charts")
        point = {
            **evidence(path),
            "id": entry["key"],
            "name": entry["name"],
            "checkpoint": r["checkpoint"],
            "revision": r["revision"],
            "precision": r["precision"],
            "task": r["task"],
            "workload": r["workload"],
            "status": "FAILED" if r.get("attempt_status") == "FAILED" else r["status"],
            "reason": r.get("reason"),
            "metrics": r["metrics"],
            "nominal_parameters_billion": size,
            "size_label": label,
            "size_source": url,
            "size_basis": "Publisher nominal size label; not a tensor count",
            "protocol": r["protocol"],
            "measurement_type": r["measurement_type"],
            "hardware": r["hardware"],
            "software": r["software"],
        }
        if r["status"] == "MEASURED":
            point.update(
                input=r["input"],
                implementation_sha256=r["implementation_sha256"],
                decoding=r["decoding"],
            )
            if r["task"] in {"synthesize", "speech-to-speech"}:
                point["actual_generated_samples"] = [s["output"]["samples"] for s in r["runs"]]
                point["actual_output_sample_rates"] = [
                    s["output"]["sample_rate"] for s in r["runs"]
                ]
        audio.append(point)
    measured = [p for p in audio if p["status"] == "MEASURED"]
    same_cohort(
        measured, ("hardware", "software", "protocol", "measurement_type", "implementation_sha256")
    )
    asr = [p for p in audio if p["task"] == "transcribe" and p["workload"] == "ASR-S1"]
    if len(asr) != 10 or any(p["status"] != "MEASURED" for p in asr):
        raise ValueError("ASR inventory changed; include missing entries explicitly")
    same_cohort(asr, ("input",))
    return {
        "schema": "mlx-one-multimodal-performance-plot-v1",
        "vlm": vlm,
        "asr": asr,
        "alignment": [p for p in audio if p["task"] == "align"],
        "speech": [p for p in audio if p["task"] in {"synthesize", "speech-to-speech"}],
        "size_note": "Nominal publisher size labels. Component scopes differ across families; "
        "not total multimodal parameter counts or checkpoint bytes. Peak memory is measured.",
    }


def base(title, subtitle, width=20, height=10.6):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig.text(0.055, 0.935, title, fontsize=29, color=BLUE, weight="bold")
    fig.text(0.055, 0.893, subtitle, fontsize=13, color=GRAY)
    return fig


def host_line(point, ending="Experimental"):
    hardware = point["hardware"]
    chip = hardware.get("device", hardware.get("chip"))
    version = point.get("mlx_version", point.get("software", {}).get("mlx"))
    return (
        f"{chip} · {hardware['memory_bytes'] / 1024**3:g} GiB · "
        f"native MLX/Metal · MLX {version} · {ending}"
    )


def axes_style(ax, ylabel, limit):
    ax.set_ylim(0, limit)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=12)
    ax.grid(axis="y", color=GRID, linestyle=(0, (2, 4)), linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=12)


def legend(ax, points):
    from matplotlib.patches import Patch

    precisions = list(dict.fromkeys(p["precision"].split(" /")[0] for p in points))
    ax.legend(
        handles=[Patch(color=PALETTE[p], label=p) for p in precisions],
        loc="upper right",
        frameon=False,
        ncol=len(precisions),
        fontsize=10,
    )


def main_chart(ax, points, metric, limit, mode, label_extras):
    if mode == "bars":
        for i, p in enumerate(points):
            value = p["metrics"][metric]
            ax.bar(i, value, width=0.62, color=PALETTE[p["precision"].split(" /")[0]], zorder=3)
            inside = value > limit * 0.1
            y = value - limit * 0.035 if inside else value + limit * 0.018
            ax.text(
                i,
                y,
                f"{value:.2f}",
                ha="center",
                va="top" if inside else "bottom",
                fontsize=13,
                weight="bold",
                color="white" if inside else INK,
            )
        labels = [
            f"{SHORT[p['id']]}\n{p['size_label']} · {p['precision']}\n{label_extras(p)}"
            for p in points
        ]
        ax.set_xticks(range(len(points)), labels, fontsize=9 if len(points) > 5 else 11)
    else:
        if metric == "decode_tps":
            offsets = {
                "V04": (12, 10),
                "V03": (12, 20),
                "V01": (-20, 24),
                "V02": (-125, -28),
                "V05": (-135, 20),
            }
        else:
            offsets = {
                "A04": (15, -15),
                "A01": (-20, 25),
                "A05": (15, 20),
                "A06": (-130, 60),
                "A08": (-125, 20),
                "A09": (20, -15),
                "A03": (-180, 70),
                "A02": (15, 60),
                "A10": (-25, 60),
                "A07": (-155, 10),
            }
        for p in points:
            x, y = p["nominal_parameters_billion"], p["metrics"][metric]
            ax.scatter(
                x,
                y,
                s=125,
                color=PALETTE[p["precision"].split(" /")[0]],
                edgecolors="white",
                linewidths=1,
                zorder=4,
            )
            name = SHORT[p["id"]].replace("\n", " ")
            ax.annotate(
                f"{name}\n{p['precision']} · {y:.2f}",
                (x, y),
                xytext=offsets[p["id"]],
                textcoords="offset points",
                fontsize=9.5,
                weight="bold",
                color=INK,
                zorder=5,
                arrowprops={"arrowstyle": "-", "color": "#B4BACB", "lw": 0.8},
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.4},
            )
        ax.set_xscale("log")
        if metric == "decode_tps":
            ax.set_xlim(0.34, 4.3)
            ticks = [0.45, 0.8, 1, 2, 3]
        else:
            ax.set_xlim(0.019, 2.7)
            ticks = [0.03, 0.06, 0.1, 0.3, 0.6, 1, 2]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Nominal publisher size (billions of parameters; log scale)", labelpad=12)
    legend(ax, points)


def companion(fig, points, metric, title, unit, height=0.40):
    ax = fig.add_axes((0.815, 0.34, 0.13, height))
    fig.text(0.785, 0.825, title, color=BLUE, fontsize=16, weight="bold")
    names = [SHORT[p["id"]].replace("\n", " ") for p in points]
    values = [p["metrics"][metric] * (100 if metric == "wer" else 1) for p in points]
    ax.barh(range(len(points)), values, height=0.45, color="#ACC6FA", zorder=3)
    ax.scatter(values, range(len(points)), s=23, color=BLUE, zorder=4)
    ax.set_yticks(range(len(points)), names, fontsize=8 if len(points) > 5 else 10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.27)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(unit, labelpad=10, color=GRAY)
    ax.grid(axis="x", color=GRID, linestyle=(0, (2, 4)), linewidth=0.8)
    for i, value in enumerate(values):
        ax.text(value + max(values) * 0.035, i, f"{value:.2f}", va="center", fontsize=9, color=INK)


def vlm_plot(dataset, mode):
    points = sorted(dataset["vlm"], key=lambda p: (p["nominal_parameters_billion"], p["id"]))
    fig = base(
        "MLX-One | Vision-language model performance",
        host_line(points[0]),
    )
    fig.text(0.075, 0.825, "Text decode speed", color=BLUE, fontsize=19, weight="bold")
    fig.text(
        0.075,
        0.791,
        "VLM-HF-S1 · one 512 × 512 image · fixed prompt · greedy output up to 128 tokens",
        color=GRAY,
        fontsize=11,
    )
    ax = fig.add_axes((0.075, 0.34, 0.62, 0.40))
    axes_style(ax, "Text decode tokens per second ↑", 185)
    main_chart(
        ax,
        points,
        "decode_tps",
        185,
        mode,
        lambda p: f"{p['metrics']['peak_mlx_memory_gib']:.3f} GiB peak",
    )
    companion(
        fig, points, "ttft_ms", "Time to first token ↓", "Milliseconds; includes image processing"
    )
    fig.text(
        0.055, 0.192, "Actual emitted tokens per repetition", color=BLUE, fontsize=15, weight="bold"
    )
    counts = "   ·   ".join(
        f"{p['name']}: {', '.join(map(str, sorted(set(p['actual_output_tokens']))))}"
        for p in points
    )
    fig.text(0.055, 0.157, counts, fontsize=11, color=INK)
    fig.text(
        0.055,
        0.116,
        "Each checkpoint keeps its native resize/tiling; fused image-token counts differ. "
        "TTFT includes image decoding, preprocessing, vision encoding and language prefill.",
        fontsize=10,
        color=GRAY,
    )
    fig.text(
        0.055,
        0.077,
        "Batch 1 · 1 warmup + 5 measured repetitions · per-column median · "
        "EOS respected; decode speed uses actual continuation steps. "
        "Peak memory includes loaded model components.",
        fontsize=10,
        color=GRAY,
    )
    fig.text(
        0.055,
        0.038,
        "Sizes are nominal checkpoint labels, not total multimodal tensor counts. "
        "Q4/Q8 exports retain some floating-point vision/projector weights. "
        "Semantic quality is not certified.",
        fontsize=10,
        color=GRAY,
    )
    return fig


def asr_plot(dataset, mode):
    from matplotlib.patches import FancyBboxPatch

    points = sorted(dataset["asr"], key=lambda p: (p["nominal_parameters_billion"], p["id"]))
    fig = base(
        "MLX-One | Speech recognition performance",
        host_line(points[0]),
    )
    fig.text(0.075, 0.825, "Input audio throughput", color=BLUE, fontsize=19, weight="bold")
    fig.text(
        0.075,
        0.791,
        "ASR-S1 · one complete English LibriSpeech utterance: 9.9 seconds + 0.1-second silence",
        fontsize=11,
        color=GRAY,
    )
    ax = fig.add_axes((0.075, 0.34, 0.62, 0.40))
    axes_style(ax, "Input audio seconds processed per second ↑", 103)
    main_chart(
        ax,
        points,
        "audio_seconds_per_second",
        103,
        mode,
        lambda p: f"{p['metrics']['peak_mlx_gib']:.3f} GiB peak",
    )
    if mode == "scatter":
        ax.set_xlabel("")
        fig.text(
            0.075,
            0.265,
            "Nominal publisher size (billions of parameters; log scale)",
            fontsize=11,
            color=BLUE,
        )
    companion(fig, points, "wer", "Single-clip WER ↓", "Word error rate (%)")
    fig.add_artist(
        FancyBboxPatch(
            (0.055, 0.108),
            0.89,
            0.095,
            transform=fig.transFigure,
            boxstyle="round,pad=0.012,rounding_size=0.008",
            edgecolor=GRID,
            facecolor="#F6F7FC",
            linewidth=1,
        )
    )
    alignments = dataset["alignment"]
    if len(alignments) != 1 or alignments[0]["status"] != "MEASURED":
        raise ValueError("alignment coverage changed; include missing status explicitly")
    a = alignments[0]["metrics"]
    fig.text(
        0.067,
        0.171,
        "Separate task: Qwen3-ForcedAligner 0.6B · Q8 / BF16 · 10-second input",
        fontsize=12,
        color=BLUE,
        weight="bold",
    )
    boundary = (
        "Unmeasured"
        if a["alignment_mean_error_ms"] is None
        else f"{a['alignment_mean_error_ms']:.2f} ms"
    )
    fig.text(
        0.067,
        0.135,
        f"Latency: {a['processing_seconds'] * 1000:.2f} ms     ·     "
        f"Peak MLX: {a['peak_mlx_gib']:.3f} GiB     ·     "
        f"Boundary accuracy: {boundary}     ·     WIP",
        fontsize=11,
        color=INK,
    )
    fig.text(
        0.055,
        0.064,
        "Batch 1 · 1 warmup + 5 repetitions · median · direct prepared-PCM task pipeline; "
        "file decoding and model loading excluded. Includes frontend, encoder and decoding.",
        fontsize=10,
        color=GRAY,
    )
    fig.text(
        0.055,
        0.030,
        "Nominal publisher size labels; component scopes differ. Peak MLX is measured. "
        "WER is one clip, not corpus accuracy. "
        "Parakeet v2 and alignment retain documented WIP limits.",
        fontsize=10,
        color=GRAY,
    )
    return fig


def speech_plot(dataset):
    from matplotlib.patches import FancyBboxPatch

    fig = base(
        "MLX-One | Speech generation performance",
        host_line(
            next(p for p in dataset["speech"] if p["status"] == "MEASURED"),
            "Separate workload panels",
        ),
        width=18.6,
        height=10.2,
    )
    rows = dataset["speech"]
    tts = [p for p in rows if p["task"] == "synthesize" and p["workload"] == "TTS-S1"]
    sts = [p for p in rows if p["task"] == "speech-to-speech" and p["workload"] == "STS-HF-Q1"]
    failed = [p for p in rows if p["workload"] == "STS-S1" and p["status"] == "FAILED"]
    if len(tts) != 1 or len(sts) != 1 or len(failed) != 1:
        raise ValueError(
            "speech coverage changed; failed and completed workloads must remain separate"
        )
    panels = [
        (tts[0], 0.08, "Text-to-speech · TTS-S1", "Fixed 30-word text · US male voice · 24 kHz"),
        (
            sts[0],
            0.565,
            "Speech-to-speech · STS-HF-Q1",
            "2.04-second spoken question · unconstrained response · 24 kHz",
        ),
    ]
    for p, x, heading, subtitle in panels:
        if p["status"] != "MEASURED":
            raise ValueError("missing generation speed cannot be plotted as a numeric value")
        m = p["metrics"]
        fig.text(x, 0.823, heading, fontsize=18, color=BLUE, weight="bold")
        fig.text(x, 0.790, subtitle, fontsize=10.5, color=GRAY)
        ax = fig.add_axes((x, 0.355, 0.35, 0.385))
        axes_style(ax, "Generated audio seconds per second ↑", 4.1)
        ax.bar([0], [m["audio_seconds_per_second"]], width=0.42, color=COLORS["Q4"])
        ax.text(
            0,
            m["audio_seconds_per_second"] - 0.18,
            f"{m['audio_seconds_per_second']:.2f}",
            fontsize=24,
            weight="bold",
            color="white",
            ha="center",
            va="top",
        )
        ax.set_xlim(-0.8, 0.8)
        ax.set_xticks([0], ["LFM2.5 Audio · nominal 1.5B · Q4 / BF16"], fontsize=12)
        fig.text(
            x,
            0.256,
            f"First audio: {m['time_to_first_audio_ms']:.2f} ms     ·     "
            f"Output: {m['output_audio_seconds']:.2f} s",
            fontsize=12,
            color=INK,
            weight="bold",
        )
        fig.text(
            x,
            0.220,
            f"RTF: {m['real_time_factor']:.3f}     ·     Peak MLX: {m['peak_mlx_gib']:.3f} GiB",
            fontsize=11,
            color=GRAY,
        )
    fig.add_artist(
        FancyBboxPatch(
            (0.055, 0.091),
            0.89,
            0.079,
            transform=fig.transFigure,
            boxstyle="round,pad=0.012,rounding_size=0.008",
            edgecolor="#EAC9C9",
            facecolor="#FFF7F7",
            linewidth=1,
        )
    )
    fig.text(
        0.068,
        0.139,
        "Primary 10-second STS-S1: FAILED · 40.96-second output bound exceeded · speed unavailable",
        fontsize=12,
        color="#A73737",
        weight="bold",
    )
    fig.text(
        0.068,
        0.110,
        "Sampled diagnostic also had unstable waveform amplitude. "
        "The completed short-question case does not resolve this failure.",
        fontsize=10.5,
        color=GRAY,
    )
    fig.text(
        0.055,
        0.048,
        "Batch 1 · 1 warmup + 5 repetitions · median · greedy text/audio · "
        "actual waveform duration determines throughput. "
        "Loaded codec is included in timing and peak memory.",
        fontsize=10,
        color=GRAY,
    )
    fig.text(
        0.055,
        0.020,
        "Direct prepared-PCM task pipeline; loading/file decode excluded. "
        "One checkpoint per panel; no size trend or perceptual-quality/MOS claim is inferred.",
        fontsize=10,
        color=GRAY,
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
    figures = [
        ("vlm-model-performance", lambda: vlm_plot(dataset, "bars")),
        ("audio-asr-performance", lambda: asr_plot(dataset, "bars")),
        ("audio-speech-generation", lambda: speech_plot(dataset)),
        ("vlm-model-size-vs-decode", lambda: vlm_plot(dataset, "scatter")),
        ("audio-model-size-vs-asr", lambda: asr_plot(dataset, "scatter")),
    ]
    for name, renderer in figures:
        fig = renderer()
        for extension in ("png", "svg"):
            path = args.output_dir / f"{name}.{extension}"
            fig.savefig(
                path,
                dpi=160,
                metadata={
                    "Description": "Actual MLX-One VLM/audio measurements",
                    **({"Date": None} if extension == "svg" else {}),
                },
            )
            print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        plt.close(fig)
    (args.output_dir / "multimodal-performance-data.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
