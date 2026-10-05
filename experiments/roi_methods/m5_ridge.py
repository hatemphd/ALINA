"""Method 5 (supervised ML): Ridge regression from a tiny image thumbnail to the centerline position.

Training targets come from the published ALINA labels (data/Labeled_Data/<video>/textfiles):
RANSAC fits one line to each frame's labelled pixels, and the target is that line's x at two
fixed rows (60% and 72% of the frame height), as fractions of the frame width. Features are a
32x18 HSV thumbnail plus a 32x18 yellow-paint mask. Training is leave-one-video-out: the model
that labels vidd_k never sees vidd_k's frames, so there is no leakage into the evaluation.

Caveat: the targets are the authors' ALINA output, not independent ground truth, so this
method learns to reproduce the published labeling.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import (
    PUBLISHED_LABELS,
    VIDEO_DIRS,
    Line,
    Proposal,
    build_trapezoid,
    detect_nose_top,
    fallback_proposal,
    frame_paths,
    load_coords,
    yellow_mask,
)

NAME = "m5_ridge"
ROWS = (0.60, 0.72)
THUMB = (32, 18)


def features(img: np.ndarray) -> np.ndarray:
    hsv = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), THUMB, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    paint = cv2.resize(yellow_mask(img, s_min=40), THUMB, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return np.concatenate([hsv.ravel(), paint.ravel()])


def label_target(coords: np.ndarray, width: int, height: int, seed: int) -> np.ndarray | None:
    if len(coords) < 50 or np.ptp(coords[:, 1]) < 20:
        return None
    xs, ys = coords[:, 0].astype(float), coords[:, 1].astype(float)
    fit = RANSACRegressor(residual_threshold=0.01 * width, random_state=seed).fit(ys.reshape(-1, 1), xs)
    return np.array([fit.predict([[r * height]])[0] / width for r in ROWS])


@lru_cache(maxsize=None)
def model_for(held_out: str, seed: int = 42):
    X, Y = [], []
    for video in VIDEO_DIRS:
        if video == held_out:
            continue
        for path in frame_paths(video):
            img = cv2.imread(str(path))
            target = label_target(load_coords(PUBLISHED_LABELS[video] / f"{path.stem}.txt"), img.shape[1], img.shape[0], seed)
            if target is not None:
                X.append(features(img))
                Y.append(target)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 25)))
    return model.fit(np.array(X), np.array(Y)), len(X)


def propose(img: np.ndarray, seed: int = 42, video: str | None = None) -> Proposal:
    h, w = img.shape[:2]
    if video is None:
        return fallback_proposal(img, "video needed for leave-one-video-out")
    model, n_train = model_for(video, seed)
    x_top, x_bottom = model.predict(features(img)[None])[0] * w
    y_top, y_bottom = (r * h for r in ROWS)
    slope = (x_bottom - x_top) / (y_bottom - y_top)
    line = Line(float(slope), float(x_top - slope * y_top))
    nose_y = detect_nose_top(img)
    return Proposal(build_trapezoid(line, nose_y, w, h), line, nose_y, note=f"trained on {n_train} frames")
