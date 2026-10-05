"""How lenient is ALINA's F1? Re-score the paper's validation pairs (data/gt_alina_labels) with stricter metrics.

Per frame, ALINA labels vs. CBEM ground truth:
    alina_f1      eval/evaluate.py: recall and precision on the *sets of x values* and *sets of y values*, averaged
    point_f1      the same formula on (x, y) points: a predicted pixel counts only if that exact pixel is in the truth
    tol3_f1       point F1 with a 3-pixel tolerance (a common choice for thin-line / edge benchmarks)
All are averaged over frames, as in eval/evaluate.py.

Usage (repo root):  uv run python -m experiments.metric_comparison
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from eval.metrics import calculate_f1, calculate_precision, calculate_recall

ROOT = Path("data/gt_alina_labels")


def load(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int32, ndmin=2).reshape(-1, 2)


def alina_f1(gt: np.ndarray, pred: np.ndarray) -> float:
    r = (calculate_recall(gt[:, 0], pred[:, 0]) + calculate_recall(gt[:, 1], pred[:, 1])) / 2
    p = (calculate_precision(gt[:, 0], pred[:, 0]) + calculate_precision(gt[:, 1], pred[:, 1])) / 2
    return calculate_f1(p, r)


def point_f1(gt: np.ndarray, pred: np.ndarray) -> float:
    g, q = set(map(tuple, gt)), set(map(tuple, pred))
    tp = len(g & q)
    return 200 * tp / (len(g) + len(q)) if g or q else 0.0


def tol_f1(gt: np.ndarray, pred: np.ndarray, tol: int = 3) -> float:
    h, w = int(max(gt[:, 1].max(), pred[:, 1].max())) + 1, int(max(gt[:, 0].max(), pred[:, 0].max())) + 1

    def dist_to(points):
        img = np.full((h, w), 255, np.uint8)
        img[points[:, 1], points[:, 0]] = 0
        return cv2.distanceTransform(img, cv2.DIST_L2, 3)

    p = 100 * np.mean(dist_to(gt)[pred[:, 1], pred[:, 0]] <= tol)
    r = 100 * np.mean(dist_to(pred)[gt[:, 1], gt[:, 0]] <= tol)
    return calculate_f1(p, r)


def main() -> None:
    print("video | frames | alina_f1 | point_f1 | tol3_f1")
    totals = {"alina": [], "point": [], "tol3": []}
    for k in (1, 2, 3):
        rows = []
        for gt_path in sorted((ROOT / "canny_textfiles" / f"canny_textfiles_{k}").glob("*.txt")):
            pred_path = ROOT / "ALINA_textfiles" / f"ALINA_textfiles_{k}" / gt_path.name
            if not pred_path.exists():
                continue
            gt, pred = load(gt_path), load(pred_path)
            rows.append((alina_f1(gt, pred), point_f1(gt, pred), tol_f1(gt, pred)))
        a, p, t = (np.mean([r[i] for r in rows]) for i in range(3))
        for key, vals in zip(totals, zip(*rows)):
            totals[key] += vals
        print(f"vidd_{k} | {len(rows)} | {a:.1f} | {p:.1f} | {t:.1f}")
    print(f"all | {len(totals['alina'])} | " + " | ".join(f"{np.mean(v):.1f}" for v in totals.values()))


if __name__ == "__main__":
    main()
