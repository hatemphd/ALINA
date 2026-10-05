"""Score the Q4 runs: writes results/q4/summary.csv (end-to-end) and results/q4/pixels.csv (mask quality).

End-to-end, per method x use x video (ALINA metrics, eval/metrics.py, as in Q3):
    labeled, ms_per_frame, mask_ms (color step alone), cbem_recall / precision / f1 (vidd_1: 7 frames, vidd_2: 10),
    pub_f1 (agreement with the published ALINA labels; reproduction, not accuracy)
Mask quality, per method (pooled pixel counts inside the usable bird's-eye ROI):
    yellow_iou / yellow_f1   vs. filled CBEM yellow paint (vidd_1 + vidd_2, rows the annotator traced)
    white_iou / white_f1     vs. the traced white stripe (vidd_1 frames 00002, 00004-00007)
    white_fp_pct             % of ROI pixels called white on frames without white paint (vidd_1 01063/01065, vidd_2 CBEM)

Usage (repo root):  uv run python -m experiments.q4_score
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .color_methods import METHODS
from .color_methods.common import VIDEOS, WHITE_TEST, cbem_frames
from .q3_score import per_frame_log, score_against, timing
from .roi_methods.common import CBEM_GT, PUBLISHED_LABELS, frame_paths

OUT = Path("results/q4")


def mean(values, digits=2):
    return round(float(np.mean(values)), digits) if len(values) else ""


def end_to_end(method: str, use: str, video: str) -> dict | None:
    run = OUT / "runs" / method / use / video
    if not (run / "timing.log").exists() or timing(run / "timing.log") is None:
        return None
    stems = [p.stem for p in frame_paths(video)]
    frames = per_frame_log(run / "timing.log")
    cbem = score_against(CBEM_GT[video], run / "coords", cbem_frames(video))
    pub = score_against(PUBLISHED_LABELS[video], run / "coords", stems)
    mask_ms = [float(x) for x in (run / "mask_ms.txt").read_text().split()]
    return {
        "method": method, "use": use, "video": video,
        "labeled": sum(frames[s][0] for s in stems),
        "ms_per_frame": round(1000 * float(np.mean([frames[s][1] for s in stems])), 1),
        "mask_ms": round(float(np.mean(mask_ms)), 1),
        "cbem_frames": len(cbem),
        "cbem_recall": mean([s[0] for s in cbem]), "cbem_precision": mean([s[1] for s in cbem]), "cbem_f1": mean([s[2] for s in cbem]),
        "pub_frames": len(pub), "pub_f1": mean([s[2] for s in pub]),
    }


def iou_f1(rows: list[dict]) -> tuple[float | str, float | str]:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    if tp + fp + fn == 0:
        return "", ""
    return round(tp / (tp + fp + fn), 3), round(2 * tp / (2 * tp + fp + fn), 3)


def mask_quality(method: str) -> dict | None:
    data = {}
    for video in VIDEOS:
        path = OUT / "runs" / method / "yellow" / video / "pixels.json"
        if not path.exists():
            return None
        data[video] = json.loads(path.read_text())
    row = {"method": method}
    for video in ("vidd_1", "vidd_2"):
        row[f"yellow_iou_{video}"], row[f"yellow_f1_{video}"] = iou_f1(data[video]["yellow"])
    row["yellow_iou"], row["yellow_f1"] = iou_f1(data["vidd_1"]["yellow"] + data["vidd_2"]["yellow"])
    white_pos = [r for r in data["vidd_1"]["white"] if r["frame"] in WHITE_TEST]
    row["white_iou"], row["white_f1"] = iou_f1(white_pos)
    white_neg = [r for r in data["vidd_1"]["white"] if r["frame"] not in WHITE_TEST] + data["vidd_2"]["white"]
    row["white_fp_pct"] = round(100 * sum(r["fp"] for r in white_neg) / sum(r["valid"] for r in white_neg), 2)
    return row


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    e2e = [r for use in ("yellow", "both") for m in METHODS for v in VIDEOS if (r := end_to_end(m, use, v))]
    pix = [r for m in METHODS if (r := mask_quality(m))]
    if e2e:
        write(OUT / "summary.csv", e2e)
        cols = ["method", "use", "video", "labeled", "ms_per_frame", "mask_ms", "cbem_recall", "cbem_precision", "cbem_f1", "pub_f1"]
        print(" | ".join(cols))
        for r in e2e:
            print(" | ".join(str(r[c]) for c in cols))
    if pix:
        write(OUT / "pixels.csv", pix)
        print()
        print(" | ".join(pix[0]))
        for r in pix:
            print(" | ".join(str(v) for v in r.values()))


if __name__ == "__main__":
    main()
