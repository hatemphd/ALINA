"""Score the Q3 runs (and the Q2 manual baseline) and write results/q3/summary.csv.

Metrics per method x video x mode:
    labeled          frames with at least one detected pixel, out of 50
    ms_per_frame     ALINA pipeline time (from timing.log)
    roi_s            mean time to propose one ROI
    fallbacks        frames where the method fell back to the centred default trapezoid
    cbem_*           accuracy vs CBEM ground truth (vidd_1: 7 frames, vidd_2: 10 frames, vidd_3: none),
                     using ALINA's own x-set / y-set recall, precision and F1 (eval/metrics.py)
    frames_scored    frames included; every-frame VLM runs exclude frames whose API call failed
    pub_f1           agreement with the published ALINA labels (data/Labeled_Data), frames with labels only.
                     This measures reproduction of the authors' labeling, not true accuracy.
    iou_manual       first-frame ROI overlap with the Q2 medium manual ROI
    jitter_px        every-frame mode: mean corner movement between consecutive frames
    iou_prev         every-frame mode: mean ROI overlap with the previous frame's ROI

Usage (repo root):  uv run python -m experiments.q3_score
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from eval.metrics import calculate_f1, calculate_precision, calculate_recall

from .roi_methods.common import CBEM_GT, CORNER_NAMES, PUBLISHED_LABELS, VIDEO_DIRS, frame_paths, load_coords, quad_iou

OUT = Path("results/q3")
Q2 = Path("results/q2")
WIDTH, HEIGHT = 1920, 1080


def frame_scores(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    recall = (calculate_recall(gt[:, 0], pred[:, 0]) + calculate_recall(gt[:, 1], pred[:, 1])) / 2
    precision = (calculate_precision(gt[:, 0], pred[:, 0]) + calculate_precision(gt[:, 1], pred[:, 1])) / 2
    return recall, precision, calculate_f1(precision, recall)


def score_against(reference_dir: Path, coords_dir: Path, stems: list[str]) -> list[tuple[float, float, float]]:
    scores = []
    for stem in stems:
        gt = load_coords(reference_dir / f"{stem}.txt")
        if len(gt) == 0:
            continue
        scores.append(frame_scores(gt, load_coords(coords_dir / f"{stem}.txt")))
    return scores


def timing(log: Path) -> tuple[int, float] | None:
    lines = log.read_text().splitlines()
    m = re.search(r"Done: (\d+)/\d+ labeled, avg ([\d.]+)ms", lines[-1]) if lines else None
    return (int(m.group(1)), float(m.group(2))) if m else None


def manual_corners(video: str) -> np.ndarray:
    c = json.loads((Q2 / video / "medium" / "roi.json").read_text())["corners_px"]
    return np.array([c[n] for n in CORNER_NAMES])


def per_frame_log(log: Path) -> dict[str, tuple[bool, float]]:
    """stem -> (labeled, seconds) from the per-frame lines of a timing.log."""
    out = {}
    for line in log.read_text().splitlines():
        m = re.match(r"\[\d+/\d+\] \[(Labeled|No lines found)\] (\S+)\.jpg: \d+ px, elapsed ([\d.]+)s", line)
        if m:
            out[m.group(2)] = (m.group(1) == "Labeled", float(m.group(3)))
    return out


def row_for(method: str, video: str, mode: str, coords_dir: Path, log: Path, proposals: dict | None) -> dict:
    stems = [p.stem for p in frame_paths(video)]
    if proposals is not None and mode == "every-frame":
        # frames whose VLM call failed (e.g. no API credits) got the fallback ROI; leave them out of every metric
        stems = [s for s in stems if not proposals[s]["note"].startswith("VLM error")]
    frames = per_frame_log(log)
    labeled = sum(frames[s][0] for s in stems)
    ms = 1000 * float(np.mean([frames[s][1] for s in stems]))
    cbem = score_against(CBEM_GT[video], coords_dir, [s for s in stems if (CBEM_GT[video] / f"{s}.txt").exists()])
    pub = score_against(PUBLISHED_LABELS[video], coords_dir, stems)
    row = {
        "method": method, "video": video, "mode": mode, "frames_scored": len(stems), "labeled": labeled, "ms_per_frame": round(ms, 1),
        "cbem_frames": len(cbem),
        "cbem_recall": round(float(np.mean([s[0] for s in cbem])), 2) if cbem else "",
        "cbem_precision": round(float(np.mean([s[1] for s in cbem])), 2) if cbem else "",
        "cbem_f1": round(float(np.mean([s[2] for s in cbem])), 2) if cbem else "",
        "pub_frames": len(pub),
        "pub_f1": round(float(np.mean([s[2] for s in pub])), 2) if pub else "",
        "roi_s": "", "fallbacks": "", "iou_manual": "", "jitter_px": "", "iou_prev": "",
    }
    if proposals is None:  # manual baseline
        row["iou_manual"] = 1.0
        return row
    used = [proposals[stems[0]]] if mode == "first-frame" else [proposals[s] for s in stems]
    row["roi_s"] = round(float(np.mean([r["seconds"] for r in used])), 3)
    row["fallbacks"] = sum(r["fallback"] for r in used)
    first = np.array([proposals[stems[0]]["corners"][n] for n in CORNER_NAMES])
    row["iou_manual"] = round(quad_iou(first, manual_corners(video), WIDTH, HEIGHT), 3)
    if mode == "every-frame":
        quads = [np.array([r["corners"][n] for n in CORNER_NAMES]) for r in used]
        shifts = [np.linalg.norm(a - b, axis=1).mean() for a, b in zip(quads, quads[1:])]
        ious = [quad_iou(a, b, WIDTH, HEIGHT) for a, b in zip(quads, quads[1:])]
        row["jitter_px"] = round(float(np.mean(shifts)), 1)
        row["iou_prev"] = round(float(np.mean(ious)), 3)
    return row


def main() -> None:
    rows = []
    for video in VIDEO_DIRS:
        q2 = Q2 / video / "medium"
        if (q2 / "timing.log").exists():
            rows.append(row_for("manual_q2_medium", video, "first-frame", q2 / "coords", q2 / "timing.log", None))
        for run in sorted((OUT / "runs").glob(f"*/{video}/*")):
            method, mode = run.parts[-3], run.parts[-1]
            if not (run / "timing.log").exists() or timing(run / "timing.log") is None:
                print(f"skipping unfinished run {run}")
                continue
            proposals = json.loads((OUT / "proposals" / method / f"{video}.json").read_text())
            rows.append(row_for(method, video, mode, run / "coords", run / "timing.log", proposals))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cols = ["method", "video", "mode", "frames_scored", "labeled", "ms_per_frame", "cbem_frames", "cbem_f1", "pub_f1", "iou_manual", "fallbacks", "jitter_px", "roi_s"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
