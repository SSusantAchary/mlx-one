"""Plot validated corpus throughput; never plot unmeasured checkpoint placeholders."""

import json

from benchmarks.ocr.corpus import EVIDENCE, ROOT, write
from benchmarks.ocr.corpus_report import load


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    inventory = json.loads((EVIDENCE.parent / "inventory.json").read_text())["checkpoints"]
    destination = ROOT / "docs/benchmarks/figures"
    destination.mkdir(parents=True, exist_ok=True)
    for hand in (False, True):
        records = [(e, r) for e, r in load(inventory, hand=hand) if r["status"] == "MEASURED"]
        if not records:
            continue
        fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
        violet, blue, orange = "#7137bc", "#4286ef", "#ff674d"
        values = [r["metrics"]["completed_pages_per_second"] for _, r in records]
        labels = [
            e["name"].replace(" Base ", "\nBase ")
            + "\n"
            + e["precision"]
            + (" (baseline)" if hand and e["id"] == "O01" else "")
            for e, _ in records
        ]
        colors = [
            violet
            if e["precision"] == "Affine Q4"
            else orange
            if e["precision"] == "FP32"
            else blue
            for e, _ in records
        ]
        bars = ax.bar(range(len(records)), values, color=colors, width=0.62, zorder=3)
        ax.bar_label(
            bars,
            labels=[f"{v:.3f}" for v in values],
            padding=7,
            fontsize=12,
            fontweight="bold",
            color="#271447",
        )
        ax.set_xticks(range(len(records)), labels, fontsize=10)
        ax.set_ylim(0, max(values) * 1.23)
        ax.set_ylabel("Images per second" if hand else "Pages per second", fontsize=12)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(axis="y", linestyle=(0, (2, 4)), color="#d5d8e7", zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#9ca3cd")
        title = "Handwriting · 10 line images" if hand else "Document OCR · 100 random pages"
        fig.suptitle(title, color="#141493", fontsize=22, fontweight="bold")
        record = records[0][1]
        ax.set_title(
            f"{record['hardware']['chip']} · "
            f"{record['hardware']['memory_bytes'] / 1024**3:g} GiB · "
            f"MLX {record['software']['mlx']} · native MLX/Metal · batch 1",
            fontsize=11,
            color="#50516c",
            pad=18,
        )
        budget = 128 if hand else 1023
        dataset = "handwritten_text_ocr test" if hand else "OmniAI OCR test"
        fig.supxlabel(
            f"{dataset} · seed 20261025 · direct-model inference\n"
            f"One request/image · {budget:,}-token cap · load/warmup excluded · "
            "throughput includes failed-request time\n"
            "Compare transcription error and truncations in the accompanying table.",
            fontsize=9,
            color="#50516c",
        )
        stem = "ocr-hand10-performance" if hand else "ocr-omni100-performance"
        fig.savefig(destination / (stem + ".png"), dpi=180, facecolor="white")
        fig.savefig(destination / (stem + ".svg"), facecolor="white")
        plt.close(fig)
        write(
            destination / (stem + ".json"),
            {
                "source": "validated per-model corpus JSON",
                "unit": "images/s" if hand else "pages/s",
                "values": [
                    {"checkpoint": e["checkpoint"], "revision": e["revision"], "throughput": v}
                    for (e, _), v in zip(records, values, strict=True)
                ],
            },
        )


if __name__ == "__main__":
    main()
