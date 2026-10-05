"""Pixel training samples for the supervised Q4 methods (classes: 0 background, 1 yellow, 2 white).

Yellow: the published ALINA labels (data/Labeled_Data), warped into the fixed ROI. They are the authors' output,
        so they are weak labels; they are only used for training, never for the pixel-level test.
White:  the hand-traced white polygons of WHITE_TRAIN (vidd_1), the only white paint inside any ROI.
Background: ROI pixels at least 15 px (bird's-eye) away from any yellow or white label.

vidd_1 frames 00001-00020 show the white stripe; those without a white label are skipped so that unlabeled white
paint never becomes a background sample. Samples are cached per video and feature set in results/q4/train_cache/.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import EVAL, WHITE_TRAIN, frames, published_yellow, warp, white_label

CACHE = Path("results/q4/train_cache")
POS_PER_FRAME, NEG_PER_FRAME, MARGIN = 400, 1500, 15


def _usable(video: str, stem: str) -> bool:
    return not (video == "vidd_1" and stem < "00021" and stem not in WHITE_TRAIN)


def video_samples(video: str, feature_fn, feature_name: str, seed: int = 42, only: tuple[str, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE / f"{video}_{feature_name}_s{seed}{'_' + '-'.join(only) if only else ''}.npz"
    if path.exists():
        d = np.load(path)
        return d["X"], d["y"]
    rng = np.random.default_rng(seed)
    region = np.zeros((1080, 1920), bool)
    region[EVAL] = True
    xs, ys = [], []
    for f in frames(video):
        if not _usable(video, f.stem) or (only and f.stem not in only):
            continue
        img = cv2.imread(str(f))
        yellow = (published_yellow(video, f.stem) > 0) & region
        white = (warp(white_label(img, video, f.stem), video, nearest=True) > 0) & region
        if not yellow.any() and not white.any():
            continue
        near = cv2.dilate((yellow | white).astype(np.uint8), np.ones((2 * MARGIN + 1, 2 * MARGIN + 1), np.uint8)) > 0
        background = region & ~near
        feats = feature_fn(warp(img, video))
        for cls, m, n in ((1, yellow, POS_PER_FRAME), (2, white, POS_PER_FRAME), (0, background, NEG_PER_FRAME)):
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
            xs.append(feats.reshape(-1, feats.shape[2])[pick])
            ys.append(np.full(len(pick), cls, np.int8))
    X, y = np.concatenate(xs).astype(np.float32), np.concatenate(ys)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def training_set(test_video: str, videos: list[str], feature_fn, feature_name: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-video-out: train on every video except test_video.

    White paint exists only in vidd_1, so when vidd_1 is the test video its WHITE_TRAIN frames are still used for
    the white and background classes (their yellow samples are dropped). The white test frames are disjoint.
    """
    parts = [video_samples(v, feature_fn, feature_name, seed) for v in videos if v != test_video]
    if test_video == "vidd_1":
        X, y = video_samples("vidd_1", feature_fn, feature_name, seed, only=WHITE_TRAIN)
        parts.append((X[y != 1], y[y != 1]))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
