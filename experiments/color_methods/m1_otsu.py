"""M1 Otsu: per-frame thresholds on ALINA's normalized S and V, chosen automatically by Otsu's method.

yellow = S > t_S and V > t_V        (t_S, t_V: Otsu over the usable ROI pixels of this frame)
white  = S <= t_S and V > t_W       (t_W: Otsu over V of the low-saturation pixels only)
Otsu always splits, even a frame with only one material, so a split is accepted only if the two class means are
at least MIN_SEPARATION apart (0..255 scale); otherwise that color is reported absent.
"""

from __future__ import annotations

import cv2
import numpy as np

from alina.color import normalize_color_features

from .common import EVAL, Masks

NAME = "m1_otsu"
KIND = "Unsupervised, per frame"
SUPERVISED = False
MIN_SEPARATION = 60


def otsu(values: np.ndarray) -> tuple[float, float]:
    """Threshold and the distance between the two class means."""
    if len(values) < 2:
        return 255.0, 0.0
    t, _ = cv2.threshold(values.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = values[values <= t], values[values > t]
    return float(t), (float(hi.mean() - lo.mean()) if len(lo) and len(hi) else 0.0)


class Method:
    def __init__(self) -> None:
        self.history: list[tuple[float, float, float]] = []

    def fit(self, test_video: str, seed: int) -> None:
        pass

    def masks(self, warped: np.ndarray) -> Masks:
        hsv = normalize_color_features(warped)
        s, v = hsv[EVAL][:, :, 1].ravel(), hsv[EVAL][:, :, 2].ravel()
        ts, sep_s = otsu(s)
        tv, sep_v = otsu(v)
        low_s = s <= ts
        tw, sep_w = otsu(v[low_s])
        yellow = np.zeros(hsv.shape[:2], np.uint8)
        white = np.zeros_like(yellow)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        if sep_s >= MIN_SEPARATION and sep_v >= MIN_SEPARATION:
            yellow[EVAL][((S > ts) & (V > tv))[EVAL]] = 255
        if sep_w >= MIN_SEPARATION:
            white[EVAL][((S <= ts) & (V > tw))[EVAL]] = 255
        self.history.append((ts if sep_s >= MIN_SEPARATION else np.nan, tv if sep_v >= MIN_SEPARATION else np.nan, tw if sep_w >= MIN_SEPARATION else np.nan))
        return Masks(yellow, white)

    def describe(self) -> dict:
        h = np.array(self.history, float)

        def stat(col):
            c = h[:, col][~np.isnan(h[:, col])] if len(h) else np.array([])
            return {"median": round(float(np.median(c)), 1), "min": round(float(c.min()), 1), "max": round(float(c.max()), 1), "frames": int(len(c))} if len(c) else None

        return {"t_S": stat(0), "t_V": stat(1), "t_W": stat(2)}
