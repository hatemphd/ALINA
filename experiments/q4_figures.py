"""Q4 figures in results/q4/figures/:
    hsv_histograms.png       normalized H, S, V of background / yellow / white training pixels, with the baseline
                             bounds and the decision-tree boxes (trained without vidd_2) overlaid
    masks_<video>_<frame>.jpg  every method's bird's-eye | yellow mask | white mask for one frame, stacked

Usage (repo root, after experiments.q4_color_threshold):  uv run python -m experiments.q4_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .color_methods import METHODS  # noqa: E402
from .color_methods.common import VIDEOS  # noqa: E402
from .color_methods.m3_tree import hsv_features  # noqa: E402
from .color_methods.training import video_samples  # noqa: E402

OUT = Path("results/q4")
FIG = OUT / "figures"
PANEL_FRAMES = [("vidd_1", "00004"), ("vidd_1", "01063"), ("vidd_2", "00005"), ("vidd_3", "18970")]


def histograms() -> None:
    parts = [video_samples(v, hsv_features, "hsv") for v in VIDEOS]
    X, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    tree = json.loads((OUT / "runs" / "m3_tree" / "yellow" / "vidd_2" / "method.json").read_text())["boxes"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ch, (ax, name) in enumerate(zip(axes, ("H (normalized)", "S (normalized)", "V (normalized)"))):
        for cls, label, color in ((0, "background", "gray"), (1, "yellow (published labels)", "goldenrod"), (2, "white (traced)", "steelblue")):
            ax.hist(X[y == cls, ch], bins=64, range=(0, 255), density=True, alpha=0.5, color=color, label=label)
        ax.axvline((0, 70, 170)[ch], color="red", ls="--", label="baseline lower bound")
        for i, box in enumerate(tree["yellow"]):
            ax.axvspan(box["lower_hsv"][ch], box["upper_hsv"][ch], color="orange", alpha=0.12, label="tree yellow box" if i == 0 else None)
        ax.set_xlabel(name)
        ax.set_yticks([])
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Training pixels per class (all videos), baseline bound vs. learned tree boxes")
    fig.tight_layout()
    fig.savefig(FIG / "hsv_histograms.png", dpi=110)
    plt.close(fig)


def mask_grids() -> None:
    for video, stem in PANEL_FRAMES:
        rows = []
        for method in METHODS:
            path = OUT / "runs" / method / "yellow" / video / "masks" / f"{stem}.jpg"
            if not path.exists():
                continue
            panel = cv2.imread(str(path))
            cv2.putText(panel, method, (10, panel.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            rows.append(panel)
        if rows:
            cv2.imwrite(str(FIG / f"masks_{video}_{stem}.jpg"), np.vstack(rows))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    histograms()
    mask_grids()
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
