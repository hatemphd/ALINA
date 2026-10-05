"""Cross-question plots for EVALUATION.md, from results/q3/summary.csv and results/q4/summary.csv:
    results/summary/f1_bars.png        CBEM recall / precision / F1 per method (Q3a, Q3b, Q4), vidd_1 and vidd_2
    results/summary/time_vs_f1.png     pipeline ms per frame vs. CBEM F1, every scored run

Usage (repo root, after q3_score and q4_score):  uv run python -m experiments.summary_plots
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("results/summary")
VIDEOS = ("vidd_1", "vidd_2")


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def panels() -> dict[str, list[dict]]:
    q3, q4 = load(Path("results/q3/summary.csv")), load(Path("results/q4/summary.csv"))
    return {
        "Q3a: ROI on first frame": [r for r in q3 if r["mode"] == "first-frame" and r["cbem_f1"]],
        "Q3b: ROI on every frame": [r for r in q3 if r["mode"] == "every-frame" and r["cbem_f1"]],
        "Q4: color step (yellow mask)": [r for r in q4 if r["use"] == "yellow" and r["cbem_f1"]],
    }


def f1_bars(groups: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(len(groups), len(VIDEOS), figsize=(14, 11))
    for row, (title, rows) in enumerate(groups.items()):
        for col, video in enumerate(VIDEOS):
            ax = axes[row, col]
            sub = [r for r in rows if r["video"] == video]
            x = np.arange(len(sub))
            for i, (key, color) in enumerate((("cbem_recall", "tab:blue"), ("cbem_precision", "tab:orange"), ("cbem_f1", "tab:green"))):
                ax.bar(x + (i - 1) * 0.27, [float(r[key]) for r in sub], 0.27, color=color, label=key.replace("cbem_", ""))
            ax.set_xticks(x, [r["method"].replace("manual_q2_medium", "manual") for r in sub], rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 105)
            ax.set_title(f"{title} - {video}", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("ALINA metric vs. CBEM ground truth (%)")
    fig.tight_layout()
    fig.savefig(OUT / "f1_bars.png", dpi=100)
    plt.close(fig)


def time_vs_f1(groups: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for (title, rows), marker in zip(groups.items(), ("o", "s", "^")):
        for video, color in zip(VIDEOS, ("tab:red", "tab:blue")):
            sub = [r for r in rows if r["video"] == video]
            ax.scatter([float(r["ms_per_frame"]) / 1000 for r in sub], [float(r["cbem_f1"]) for r in sub], marker=marker, color=color, alpha=0.7, label=f"{title}, {video}")
    ax.set_xscale("log")
    ax.set_xlabel("ALINA pipeline seconds per frame (log)")
    ax.set_ylabel("CBEM F1 (%)")
    ax.legend(fontsize=7)
    ax.set_title("Speed vs. accuracy, every scored run")
    fig.tight_layout()
    fig.savefig(OUT / "time_vs_f1.png", dpi=100)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = panels()
    f1_bars(groups)
    time_vs_f1(groups)
    print(f"plots written to {OUT}")


if __name__ == "__main__":
    main()
