"""Q2 without the GUI: run ALINA on every video x ROI size using fixed trapezoids.

The trapezoids are the hand-chosen guide shapes in q2_make_roi_guides.py, so each
run is the same as clicking those corners in `alina label`, but reproducible.
Labeling uses ALINA's own process_image(); nothing in the pipeline is changed.

Per run, writes results/q2/<video>/<size>/:
    annotated/       frames with detected line pixels in red   (same as alina label)
    coords/          one "x y" per detected pixel per frame     (same as alina label)
    timing.log       per-frame status + summary line            (same format as alina label)
    roi.json         exact trapezoid corners in pixels and as fractions of the frame
    roi_overlay.jpg  first frame with the trapezoid drawn (stands in for the Confirm ROI window)
    birds_eye.jpg    first frame's ROI warped to the bird's-eye rectangle
    mask.jpg         HSV threshold mask of that bird's-eye view (after the left-column zeroing)
    compare.jpg      overlay | bird's-eye | mask, side by side

Usage (repo root, after README Q2 Step 1):
    uv run python scripts/q2_run_headless.py                 # all runs
    uv run python scripts/q2_run_headless.py vidd_2/medium   # selected runs
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2_make_roi_guides import GUIDES, trapezoid  # noqa: E402

from alina.color import normalize_color_features  # noqa: E402
from alina.config import PipelineConfig  # noqa: E402
from alina.pipeline import process_image  # noqa: E402
from alina.roi import roi_points_to_quad  # noqa: E402

INPUTS = {
    "vidd_1": Path("outputs/vidd_1_1080p"),
    "vidd_2": Path("data/Raw_Data/vidd_2"),
    "vidd_3": Path("data/Raw_Data/vidd_3"),
}
# (video, run name, guide size, extra PipelineConfig settings)
RUNS = [(video, size, size, {}) for video in INPUTS for size in ("tight", "medium", "loose")]
RUNS.append(("vidd_3", "medium_s40", "medium", {"yellow_lower": (0, 40, 170)}))
RUNS.append((
    "vidd_3", "medium_loose_thresh", "medium",
    {"yellow_lower": (0, 40, 150), "mask_ignore_left_columns": 100, "min_white_pixels": 50, "peak_pixel_threshold": 20},
))

OUT_ROOT = Path("results/q2")


def to_roi_points(corners: np.ndarray) -> np.ndarray:
    """4 corners (BL, TL, TR, BR) -> the (1, 6, 2) array select_roi() returns for 4 clicks."""
    bl, tl, tr, br = (tuple(int(v) for v in c) for c in corners)
    return np.array([[bl, tl, tl, tr, tr, br]], dtype=np.int32)


def warp_and_mask(img: np.ndarray, roi_points: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Repeat process_image()'s warp and threshold steps so they can be saved as images."""
    src = roi_points_to_quad(roi_points, dtype=np.float32)
    r = config.roi
    dst = np.array([r.dst_bottom_left, r.dst_top_left, r.dst_top_right, r.dst_bottom_right], dtype=np.float32).reshape(1, 4, 2)
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (img.shape[1], img.shape[0]))
    mask = cv2.inRange(normalize_color_features(warped), np.array(config.yellow_lower), np.array(config.yellow_upper))
    if config.mask_ignore_left_columns > 0:
        mask[:, : config.mask_ignore_left_columns] = 0
    (x0, y0), (x1, y1) = r.dst_top_left, r.dst_bottom_right
    return warped[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def save_roi_evidence(out: Path, first: np.ndarray, roi_points: np.ndarray, config: PipelineConfig, label: str) -> None:
    quad = roi_points_to_quad(roi_points, dtype=np.int32)
    h, w = first.shape[:2]
    corners = {name: [int(x), int(y)] for name, (x, y) in zip(("bottom_left", "top_left", "top_right", "bottom_right"), quad[0])}
    fractions = {name: [round(x / w, 3), round(y / h, 3)] for name, (x, y) in corners.items()}
    (out / "roi.json").write_text(json.dumps({"frame_size": [w, h], "corners_px": corners, "corners_fraction": fractions}, indent=2))

    overlay = first.copy()
    cv2.polylines(overlay, quad, True, (0, 0, 255), 3)
    cv2.putText(overlay, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    cv2.imwrite(str(out / "roi_overlay.jpg"), overlay)

    birds_eye, mask = warp_and_mask(first, roi_points, config)
    cv2.imwrite(str(out / "birds_eye.jpg"), birds_eye)
    cv2.imwrite(str(out / "mask.jpg"), mask)

    panel_h = 360
    panels = [cv2.resize(p, (int(p.shape[1] * panel_h / p.shape[0]), panel_h)) for p in (overlay, birds_eye, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))]
    for panel, title in zip(panels, ("camera + ROI", "bird's-eye warp", "HSV mask")):
        cv2.putText(panel, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(out / "compare.jpg"), np.hstack(panels))


def run(video: str, name: str, guide_size: str, overrides: dict) -> str:
    out = OUT_ROOT / video / name
    config = PipelineConfig(
        input_dir=INPUTS[video],
        output_images_dir=out / "annotated",
        output_coords_dir=out / "coords",
        log_file=out / "timing.log",
        **overrides,
    )
    frames = sorted(p for p in config.input_dir.iterdir() if p.suffix.lower() == ".jpg")
    first = cv2.imread(str(frames[0]))
    corners = trapezoid(GUIDES[video][1][guide_size], first.shape[1], first.shape[0])
    roi_points = to_roi_points(corners)
    save_roi_evidence(out, first, roi_points, config, f"{video} {name}")

    labeled, total_elapsed = 0, 0.0
    with open(config.log_file, "w") as log:
        for i, path in enumerate(frames, start=1):
            start = time.perf_counter()
            annotated, coords, had_lines = process_image(cv2.imread(str(path)), roi_points, config)
            elapsed = time.perf_counter() - start
            np.savetxt(config.output_coords_dir / f"{path.stem}.txt", coords, fmt="%6d")
            cv2.imwrite(str(config.output_images_dir / path.name), annotated)
            labeled += had_lines
            total_elapsed += elapsed
            status = "Labeled" if had_lines else "No lines found"
            line = f"[{i}/{len(frames)}] [{status}] {path.name}: {len(coords)} px, elapsed {elapsed:.3f}s ({datetime.now():%H:%M:%S})"
            log.write(line + "\n")
            log.flush()
        avg_ms = total_elapsed / len(frames) * 1000
        summary = f"Done: {labeled}/{len(frames)} labeled, avg {avg_ms:.1f}ms/frame ({1000 / avg_ms:.2f} fps), total {total_elapsed:.1f}s"
        log.write(summary + "\n")
    return summary


def main() -> None:
    selected = set(sys.argv[1:])
    for video, name, guide_size, overrides in RUNS:
        if selected and f"{video}/{name}" not in selected:
            continue
        print(f"{video}/{name}: running...", flush=True)
        print(f"{video}/{name}: {run(video, name, guide_size, overrides)}", flush=True)


if __name__ == "__main__":
    main()
