"""Shared pieces for Q4: fixed ROI, bird's-eye warp, pixel features, pixel labels, and ALINA labeling with a
swappable color step.

Every method sees exactly what ALINA's threshold step sees: the bird's-eye warp of a fixed ROI. Only the
step that turns that warp into a binary mask changes; the warp, the left-column zeroing, the histogram,
CIRCLEDAT and the unwarp are ALINA's own.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from alina.color import normalize_color_features
from alina.config import PipelineConfig
from alina.histogram import calc_histogram
from alina.roi import roi_points_to_quad
from alina.traversal import circular_threshold_pixel_discovery_and_traversal

from ..roi_methods.common import CBEM_GT, PUBLISHED_LABELS, VIDEO_DIRS, frame_paths, load_coords, to_roi_points

VIDEOS = tuple(VIDEO_DIRS)
WIDTH, HEIGHT = 1920, 1080

# The Q3a M1 (Hough) first-frame ROI of each video: the only automatic ROI that labeled all three videos.
# Held fixed so that only the color step differs between methods.
FIXED_ROI = {
    "vidd_1": np.array([[535, 730], [760, 665], [962, 665], [1111, 730]]),
    "vidd_2": np.array([[576, 795], [786, 730], [988, 730], [1152, 795]]),
    "vidd_3": np.array([[583, 795], [957, 730], [1159, 730], [1159, 795]]),
}

_CFG = PipelineConfig(input_dir=".", output_images_dir="/tmp/q4_unused", output_coords_dir="/tmp/q4_unused")
DST = np.array([_CFG.roi.dst_bottom_left, _CFG.roi.dst_top_left, _CFG.roi.dst_top_right, _CFG.roi.dst_bottom_right], dtype=np.float32)
(X0, Y0), (X1, Y1) = _CFG.roi.dst_top_left, _CFG.roi.dst_bottom_right
# pixels ALINA can actually use: inside the warp rectangle, right of the zeroed left columns
EVAL = (slice(Y0, Y1), slice(_CFG.mask_ignore_left_columns, X1))

# White-marking labels, made the CBEM way: a polygon traced by hand around the marking (here the black-outlined
# white lane stripe of vidd_1), then an automatic split inside it (2-means on gray: white paint vs black outline).
# Only vidd_1 has white paint inside its ROI. Polygons cover rows 660-740, i.e. all ROI rows (665-730).
# Each entry: left edge x at y=660 and y=740, right edge x at y=660 and y=740 (white paint plus ~6 px of outline).
WHITE_POLYGONS = {
    "00002": (944, 964, 993, 1038),
    "00004": (943, 963, 992, 1036),
    "00005": (942, 962, 991, 1034),
    "00006": (940, 960, 988, 1031),
    "00007": (939, 959, 986, 1029),
    "00010": (944, 962, 994, 1036),
    "00014": (934, 954, 984, 1028),
    "00018": (932, 951, 981, 1021),
}
WHITE_TEST = ("00002", "00004", "00005", "00006", "00007")  # the vidd_1 CBEM frames that show the stripe
WHITE_TRAIN = ("00010", "00014", "00018")
WHITE_ROWS = (660, 740)


def homography(video: str) -> np.ndarray:
    quad = roi_points_to_quad(to_roi_points(FIXED_ROI[video]), dtype=np.float32)
    return cv2.getPerspectiveTransform(quad, DST.reshape(1, 4, 2))


def warp(img: np.ndarray, video: str, nearest: bool = False) -> np.ndarray:
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpPerspective(img, homography(video), (img.shape[1], img.shape[0]), flags=flags)


def pixel_features(warped: np.ndarray) -> np.ndarray:
    """(H, W, 6) float32: ALINA's normalized HSV plus CIE Lab, all 0..255."""
    hsv = normalize_color_features(warped).astype(np.float32)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.dstack((hsv, lab))


def context_features(warped: np.ndarray) -> np.ndarray:
    """(H, W, 20): pixel features, their 5x5 and 15x15 means, and 5x5 std of V and L (local contrast)."""
    f = pixel_features(warped)
    m5, m15 = cv2.blur(f, (5, 5)), cv2.blur(f, (15, 15))
    vl = f[:, :, [2, 3]]
    std5 = np.sqrt(np.maximum(cv2.blur(vl * vl, (5, 5)) - cv2.blur(vl, (5, 5)) ** 2, 0))
    return np.dstack((f, m5, m15, std5))


# ---------------------------------------------------------------- labels (all returned in warped coordinates)

def _coords_mask(coords: np.ndarray) -> np.ndarray:
    m = np.zeros((HEIGHT, WIDTH), np.uint8)
    if len(coords):
        m[coords[:, 1], coords[:, 0]] = 255
    return m


def fill_edge_pairs(edges: np.ndarray, gray: np.ndarray, max_width: int = 45, context: int = 60) -> np.ndarray:
    """CBEM ground truth marks the edges of the paint; fill each row between consecutive edges where the gap is
    paint, i.e. brighter than the surrounding pavement (black outlines between stripes stay unfilled).
    Uses brightness only, so no color threshold leaks into the ground truth."""
    filled = np.zeros_like(edges)
    for y in np.unique(np.nonzero(edges)[0]):
        xs = np.nonzero(edges[y])[0]
        runs = np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)
        row = gray[y]
        pavement = np.median(row[max(xs[0] - context, 0) : xs[-1] + context])
        for a, b in zip(runs, runs[1:]):
            gap = row[a[-1] + 1 : b[0]]
            if b[0] - a[-1] <= max_width and len(gap) and gap.mean() > pavement:
                filled[y, a[0] : b[-1] + 1] = 255
        if len(runs) == 1:
            filled[y, runs[0]] = 255
    return filled


def published_yellow(video: str, stem: str) -> np.ndarray:
    """Weak training labels: the authors' published ALINA output for this frame."""
    return warp(_coords_mask(load_coords(PUBLISHED_LABELS[video] / f"{stem}.txt")), video, nearest=True)


def cbem_frames(video: str) -> list[str]:
    stems = {p.stem for p in frame_paths(video)}
    return sorted(p.stem for p in CBEM_GT[video].glob("*.txt") if p.stem in stems)


def cbem_yellow(img: np.ndarray, video: str, stem: str) -> tuple[np.ndarray, np.ndarray]:
    """Filled CBEM yellow-paint mask and the warped rows it covers (the annotator traced only part of the line)."""
    coords = load_coords(CBEM_GT[video] / f"{stem}.txt")
    band = np.zeros((HEIGHT, WIDTH), np.uint8)
    band[coords[:, 1].min() : coords[:, 1].max() + 1] = 255
    filled = fill_edge_pairs(_coords_mask(coords), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return warp(filled, video, nearest=True), warp(band, video, nearest=True)


def white_label(img: np.ndarray, video: str, stem: str) -> np.ndarray:
    """White-paint mask in original coordinates (empty for frames without a traced polygon)."""
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    if video != "vidd_1" or stem not in WHITE_POLYGONS:
        return mask
    l0, l1, r0, r1 = WHITE_POLYGONS[stem]
    (ya, yb) = WHITE_ROWS
    poly = np.array([[l0, ya], [r0, ya], [r1, yb], [l1, yb]], np.int32)
    inside = np.zeros_like(mask)
    cv2.fillPoly(inside, [poly], 255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    values = gray[inside > 0].reshape(-1, 1)
    _, _, centers = cv2.kmeans(values, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 5, cv2.KMEANS_PP_CENTERS)
    split = float(centers.mean())
    mask[(inside > 0) & (gray > split)] = 255
    return mask


@lru_cache(maxsize=None)
def white_band(video: str) -> np.ndarray:
    """Warped rows covered by the white labels (all ROI rows); outside vidd_1 the whole warp is label-free of white."""
    band = np.zeros((HEIGHT, WIDTH), np.uint8)
    if video == "vidd_1":
        band[WHITE_ROWS[0] : WHITE_ROWS[1] + 1] = 255
    else:
        band[:] = 255
    return warp(band, video, nearest=True)


# ---------------------------------------------------------------- ALINA labeling with a swappable color step

@dataclass
class Masks:
    yellow: np.ndarray
    white: np.ndarray


def label_frame(img: np.ndarray, video: str, mask_fn, use: str = "yellow", config: PipelineConfig = _CFG):
    """process_image() with the cv2.inRange step replaced by mask_fn(warped) -> Masks.

    use: "yellow" feeds the yellow mask to ALINA (what the original threshold does), "both" feeds yellow | white.
    Returns coords (N, 2) in the original frame, had_lines, and the Masks.
    """
    warped = warp(img, video)
    masks = mask_fn(warped)
    mask = masks.yellow.copy() if use == "yellow" else cv2.bitwise_or(masks.yellow, masks.white)
    if config.mask_ignore_left_columns > 0:
        mask[:, : config.mask_ignore_left_columns] = 0

    avg_pixel, peak_value = calc_histogram(mask, min_white_pixels=config.min_white_pixels)
    if not avg_pixel or peak_value <= config.peak_pixel_threshold:
        return np.empty((0, 2), np.int32), False, masks

    binary = mask.copy()
    binary[avg_pixel[1]][avg_pixel[0]] = 255
    pixels = circular_threshold_pixel_discovery_and_traversal(binary, avg_pixel[0], avg_pixel[1], config.circular_threshold)
    black = np.zeros(mask.shape, np.uint8)
    for x, y in pixels:
        cv2.circle(black, (x, y), 1, 255, -1)
    unwarped = cv2.warpPerspective(black, np.linalg.inv(homography(video)), (img.shape[1], img.shape[0]))
    ys, xs = np.nonzero(unwarped)
    return np.column_stack((xs, ys)), True, masks


def frames(video: str) -> list[Path]:
    return frame_paths(video)
