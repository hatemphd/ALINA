"""Method 1 (classical CV baseline, no learning): fixed HSV yellow threshold + Hough line segments.

Steps: find the nose, threshold yellow paint in the pavement band ahead of it with fixed HSV
bounds, run Canny + probabilistic Hough on that mask, keep segments steeper than 20 degrees,
pick the group whose extension reaches the bottom edge closest to the image centre (the ego
centerline), and fit one line through it for the shared trapezoid.
"""

from __future__ import annotations

import cv2
import numpy as np

from .common import Line, Proposal, build_trapezoid, detect_nose_top, fallback_proposal, fit_line, ground_band, yellow_mask

NAME = "m1_hough"
MIN_ANGLE_DEG = 20


def propose(img: np.ndarray, seed: int = 42, video: str | None = None) -> Proposal:
    h, w = img.shape[:2]
    nose_y = detect_nose_top(img)
    y0, y1 = ground_band(img, nose_y)
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1] = yellow_mask(img[y0:y1])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    edges = cv2.Canny(mask, 50, 150)
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=int(0.02 * h), maxLineGap=int(0.02 * h))
    if segments is None:
        return fallback_proposal(img, "no Hough segments")

    segs = segments.reshape(-1, 4).astype(float)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    steep = np.degrees(np.arctan2(np.abs(dy), np.abs(dx))) >= MIN_ANGLE_DEG
    segs, dx, dy = segs[steep], dx[steep], dy[steep]
    if len(segs) == 0:
        return fallback_proposal(img, "no steep segments")

    # where each segment's extension meets the bottom of the band
    x_bottom = segs[:, 0] + (y1 - segs[:, 1]) * dx / dy
    ego = np.argmin(np.abs(x_bottom - w / 2))
    group = np.abs(x_bottom - x_bottom[ego]) < 0.04 * w
    pts = segs[group]
    xs = np.concatenate([pts[:, 0], pts[:, 2]])
    ys = np.concatenate([pts[:, 1], pts[:, 3]])
    lengths = np.hypot(dx[group], dy[group])
    line = fit_line(xs, ys, np.concatenate([lengths, lengths]))
    return Proposal(build_trapezoid(line, nose_y, w, h), line, nose_y, note=f"{int(group.sum())} segments")
