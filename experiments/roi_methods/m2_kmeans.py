"""Method 2 (unsupervised ML): K-means color clustering + RANSAC line regression.

Steps: find the nose, sample pixels from the pavement band ahead of it, cluster them in Lab color
space with K-means (k=10, lightness down-weighted so clusters split on colour), and take the
non-green cluster with the highest b* (most yellow) as paint.
RANSAC then fits x = slope * y + intercept to the paint pixels, ignoring outliers such as
grass edges or a second marking, and the fitted line goes into the shared trapezoid.
Unlike Method 1, nothing is a fixed color threshold: the "yellow" cluster adapts to each frame.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor

from .common import Line, Proposal, build_trapezoid, detect_nose_top, fallback_proposal, ground_band

NAME = "m2_kmeans"
K = 10
SAMPLE = 20000
WEIGHTS = np.array([0.25, 1.0, 1.0], np.float32)  # down-weight lightness so clusters split on colour
MIN_B = 135


def propose(img: np.ndarray, seed: int = 42, video: str | None = None) -> Proposal:
    h, w = img.shape[:2]
    nose_y = detect_nose_top(img)
    y0, y1 = ground_band(img, nose_y)
    x0, x1 = int(0.25 * w), int(0.75 * w)
    band = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32) * WEIGHTS

    rng = np.random.default_rng(seed)
    sample = band[rng.choice(len(band), size=min(SAMPLE, len(band)), replace=False)]
    km = KMeans(n_clusters=K, n_init=4, random_state=seed).fit(sample)
    centers = km.cluster_centers_ / WEIGHTS
    # yellow paint: bright, a* not green (grass has a* < 128), highest b* (yellowness); 128 is neutral
    candidates = np.where((centers[:, 1] >= 128) & (centers[:, 0] >= 100), centers[:, 2], -1)
    paint = int(np.argmax(candidates))
    if candidates[paint] < MIN_B:
        return fallback_proposal(img, "no yellow cluster")

    labels = km.predict(band).reshape(y1 - y0, x1 - x0)
    ys, xs = np.nonzero(labels == paint)
    if len(xs) < 200:
        return fallback_proposal(img, "too few paint pixels")
    if len(xs) > SAMPLE:
        keep = rng.choice(len(xs), size=SAMPLE, replace=False)
        xs, ys = xs[keep], ys[keep]
    ys, xs = ys + y0, xs + x0

    ransac = RANSACRegressor(residual_threshold=0.01 * w, random_state=seed).fit(ys.reshape(-1, 1), xs)
    line = Line(float(ransac.estimator_.coef_[0]), float(ransac.estimator_.intercept_))
    inliers = int(ransac.inlier_mask_.sum())
    return Proposal(build_trapezoid(line, nose_y, w, h), line, nose_y, note=f"{inliers} inlier px")
