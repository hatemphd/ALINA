"""Draw target tight / medium / loose ROI trapezoids on each video's first frame.

The images are a clicking guide for Q2 (manual ROI). They are not ALINA output.
Run from the repo root after resizing vidd_1 (README Q2, Step 1):

    uv run python scripts/q2_make_roi_guides.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Fractions of frame width (x) and height (y), measured from the top-left corner.
# Each size: ((bottom-left x, bottom-right x, bottom y), (top-left x, top-right x, top y)).
# Bottom edge sits just above the aircraft nose; top edge just below the horizon.
GUIDES = {
    "vidd_1": (
        "outputs/vidd_1_1080p/00001.jpg",
        {
            "tight": ((0.35, 0.60, 0.68), (0.43, 0.53, 0.56)),
            "medium": ((0.25, 0.75, 0.68), (0.40, 0.60, 0.55)),
            "loose": ((0.05, 0.95, 0.68), (0.30, 0.70, 0.53)),
        },
    ),
    "vidd_2": (
        "data/Raw_Data/vidd_2/00001.jpg",
        {
            "tight": ((0.38, 0.58, 0.72), (0.45, 0.50, 0.63)),
            "medium": ((0.30, 0.70, 0.72), (0.43, 0.53, 0.63)),
            "loose": ((0.10, 0.90, 0.72), (0.35, 0.65, 0.62)),
        },
    ),
    "vidd_3": (
        "data/Raw_Data/vidd_3/18970.jpg",
        {
            "tight": ((0.45, 0.75, 0.73), (0.52, 0.70, 0.64)),
            "medium": ((0.35, 0.90, 0.73), (0.45, 0.80, 0.63)),
            "loose": ((0.15, 1.00, 0.73), (0.30, 0.98, 0.62)),
        },
    ),
}

COLORS = {"tight": (0, 0, 255), "medium": (0, 255, 255), "loose": (0, 255, 0)}  # BGR: red, yellow, green
OUTPUT_DIR = Path("results/q2/roi_guides")


def trapezoid(sizes: tuple, width: int, height: int) -> np.ndarray:
    (bl_x, br_x, bottom_y), (tl_x, tr_x, top_y) = sizes
    # Corners in ALINA's click order: Bottom-Left, Top-Left, Top-Right, Bottom-Right
    return np.array(
        [
            [bl_x * width, bottom_y * height],
            [tl_x * width, top_y * height],
            [tr_x * width, top_y * height],
            [br_x * width - 1, bottom_y * height],
        ],
        dtype=np.int32,
    )


def draw_guide(frame_path: str, sizes_by_name: dict) -> np.ndarray:
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {frame_path}")
    height, width = img.shape[:2]

    for name, sizes in sizes_by_name.items():
        corners = trapezoid(sizes, width, height)
        cv2.polylines(img, [corners], True, COLORS[name], 3)
        if name == "medium":
            for i, (x, y) in enumerate(corners, start=1):
                cv2.circle(img, (int(x), int(y)), 12, (255, 255, 255), -1)
                cv2.putText(img, str(i), (int(x) - 8, int(y) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    y = 40
    for name, color in COLORS.items():
        cv2.putText(img, name, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        y += 45
    return img


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for video, (frame_path, sizes_by_name) in GUIDES.items():
        out_path = OUTPUT_DIR / f"{video}_guide.jpg"
        cv2.imwrite(str(out_path), draw_guide(frame_path, sizes_by_name))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
