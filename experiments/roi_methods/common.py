"""Shared pieces for the Q3 ROI methods.

Every method answers two questions about a frame:
    1. Where is the ego centerline?  -> a line x = slope * y + intercept (image pixels)
    2. Where does the aircraft nose start?  -> the y of the nose top
and `build_trapezoid` turns those answers into ALINA's ROI.

Lesson from Q2: ALINA only detects a line that is thick and close to vertical in the
bird's-eye view. A homography maps straight lines to straight lines, so if the centerline
crosses the top and bottom edges of the trapezoid at the same fraction of their width,
it becomes exactly vertical after the warp. `LINE_FRACTION` puts it right of centre, clear
of the 300 columns ALINA zeroes on the left of the warped mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 1080p frames for every video (vidd_1 is the 4K set resized in Q2 Step 1)
VIDEO_DIRS = {
    "vidd_1": Path("outputs/vidd_1_1080p"),
    "vidd_2": Path("data/Raw_Data/vidd_2"),
    "vidd_3": Path("data/Raw_Data/vidd_3"),
}
PUBLISHED_LABELS = {v: Path(f"data/Labeled_Data/{v}/textfiles") for v in VIDEO_DIRS}
CBEM_GT = {v: Path(f"data/gt_alina_labels/canny_textfiles/canny_textfiles_{v[-1]}") for v in VIDEO_DIRS}


def frame_paths(video: str) -> list[Path]:
    return sorted(p for p in VIDEO_DIRS[video].iterdir() if p.suffix.lower() == ".jpg")


def load_coords(path: Path) -> np.ndarray:
    """ALINA coordinate file -> (N, 2) int array of x, y; empty if the file has no pixels."""
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 2), np.int32)
    return np.loadtxt(path, dtype=np.int32, ndmin=2)

LINE_FRACTION = 0.55
BOTTOM_WIDTH = 0.30  # of frame width
TOP_TO_BOTTOM_WIDTH = 0.35
# Trapezoid height, as a fraction of frame height. Only near paint passes ALINA's HSV threshold
# (far paint is dimmer), so a shallow ROI lets that bright paint fill the whole warped height.
DEPTH = 0.06
NOSE_MARGIN = 0.015  # gap between the bottom edge and the nose top, fraction of frame height
CORNER_NAMES = ("bottom_left", "top_left", "top_right", "bottom_right")


@dataclass
class Line:
    """x = slope * y + intercept, in image pixels."""

    slope: float
    intercept: float

    def x_at(self, y: float) -> float:
        return self.slope * y + self.intercept


@dataclass
class Proposal:
    corners: np.ndarray  # (4, 2) int: BL, TL, TR, BR
    line: Line | None = None
    nose_y: int | None = None
    fallback: bool = False
    note: str = ""


def fit_line(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray | None = None) -> Line:
    slope, intercept = np.polyfit(ys.astype(float), xs.astype(float), 1, w=weights)
    return Line(float(slope), float(intercept))


def detect_nose_top(img: np.ndarray) -> int:
    """Row where the aircraft nose begins.

    The nose top is a dark-above / bright-below edge in the lower middle of the frame. The horizon
    can give the same kind of edge, but it is always higher, so take the lowest strong peak.
    """
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 0).astype(np.float32)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    lo, hi = int(0.60 * h), int(0.82 * h)
    profile = sobel_y[lo:hi, int(0.40 * w) : int(0.62 * w)].mean(axis=1)
    profile = np.convolve(profile, np.ones(9) / 9, mode="same")
    strong = 0.5 * profile.max()
    peaks = [i for i in range(5, len(profile) - 5) if profile[i] >= strong and profile[i] == profile[i - 5 : i + 6].max()]
    return lo + (peaks[-1] if peaks else int(np.argmax(profile)))


def ground_band(img: np.ndarray, nose_y: int) -> tuple[int, int]:
    """Rows of pavement just ahead of the nose, where the methods look for the centerline."""
    h = img.shape[0]
    return max(0, nose_y - int(0.18 * h)), max(1, nose_y - int(NOSE_MARGIN * h))


def build_trapezoid(line: Line, nose_y: int, width: int, height: int) -> np.ndarray:
    """Trapezoid whose top and bottom edges cross `line` at LINE_FRACTION of their width."""
    y_bottom = int(np.clip(nose_y - NOSE_MARGIN * height, 0.3 * height, height - 1))
    y_top = int(y_bottom - DEPTH * height)
    w_bottom = BOTTOM_WIDTH * width
    w_top = w_bottom * TOP_TO_BOTTOM_WIDTH

    left_bottom = line.x_at(y_bottom) - LINE_FRACTION * w_bottom
    left_top = line.x_at(y_top) - LINE_FRACTION * w_top
    quad = np.array(
        [
            [left_bottom, y_bottom],
            [left_top, y_top],
            [left_top + w_top, y_top],
            [left_bottom + w_bottom, y_bottom],
        ]
    )
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    return np.round(quad).astype(np.int32)


def fallback_proposal(img: np.ndarray, note: str) -> Proposal:
    """Centred trapezoid above the detected nose, used when a method can't find the line."""
    h, w = img.shape[:2]
    nose_y = detect_nose_top(img)
    line = Line(0.0, w * 0.5)
    return Proposal(build_trapezoid(line, nose_y, w, h), line, nose_y, fallback=True, note=note)


def validate_corners(corners: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Clip to the frame and reject shapes ALINA can't warp sensibly. Returns None if invalid."""
    c = np.asarray(corners, dtype=float).reshape(4, 2)
    c[:, 0] = np.clip(c[:, 0], 0, width - 1)
    c[:, 1] = np.clip(c[:, 1], 0, height - 1)
    bl, tl, tr, br = c
    if not (tl[1] < bl[1] - 10 and tr[1] < br[1] - 10):
        return None
    if not (tr[0] - tl[0] > 10 and br[0] - bl[0] > 10):
        return None
    return np.round(c).astype(np.int32)


MAX_ABS_SLOPE = 3.0  # |dx/dy|; flatter than ~18 degrees from horizontal can't be the centerline ahead


def is_usable(corners: np.ndarray, line: Line | None, width: int, height: int) -> bool:
    """False for ROIs that would collapse in the warp: near-horizontal lines or degenerate trapezoids."""
    if line is not None and abs(line.slope) > MAX_ABS_SLOPE:
        return False
    return validate_corners(corners, width, height) is not None


def sanitize(img: np.ndarray, proposal: Proposal) -> Proposal:
    h, w = img.shape[:2]
    if is_usable(proposal.corners, proposal.line, w, h):
        return proposal
    return fallback_proposal(img, f"unusable ROI ({proposal.note})")


def to_roi_points(corners: np.ndarray) -> np.ndarray:
    """4 corners (BL, TL, TR, BR) -> the (1, 6, 2) array ALINA's select_roi() returns for 4 clicks."""
    bl, tl, tr, br = (tuple(int(v) for v in c) for c in corners)
    return np.array([[bl, tl, tl, tr, tr, br]], dtype=np.int32)


def quad_iou(a: np.ndarray, b: np.ndarray, width: int, height: int) -> float:
    ma = np.zeros((height, width), np.uint8)
    mb = np.zeros((height, width), np.uint8)
    cv2.fillPoly(ma, [np.asarray(a, np.int32).reshape(-1, 2)], 1)
    cv2.fillPoly(mb, [np.asarray(b, np.int32).reshape(-1, 2)], 1)
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else 0.0


def yellow_mask(img: np.ndarray, s_min: int = 60, v_min: int = 100) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (12, s_min, v_min), (40, 255, 255))
