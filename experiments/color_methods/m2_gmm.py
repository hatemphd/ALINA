"""M2 Gaussian mixture: per-frame clustering of the ROI pixels in CIE Lab, then naming the clusters.

A 5-component GMM (seed 42) is fit on 20,000 pixels sampled from the usable ROI of the frame.
yellow cluster: highest mean b* (yellowness), accepted if mean b* >= 140 (128 is neutral)
white cluster:  among near-neutral clusters (chroma < 12), the brightest, accepted if its mean L is at least
                35 above the dominant (pavement) cluster
Every ROI pixel gets its most likely cluster, so the thresholds are the GMM's decision boundaries.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

from .common import EVAL, Masks

NAME = "m2_gmm"
KIND = "Unsupervised clustering, per frame"
SUPERVISED = False
K, SAMPLE, MIN_B, MAX_CHROMA, MIN_L_GAIN = 5, 20000, 140, 12, 35


class Method:
    def __init__(self) -> None:
        self.seed = 42
        self.found: list[dict] = []

    def fit(self, test_video: str, seed: int) -> None:
        self.seed = seed

    def masks(self, warped: np.ndarray) -> Masks:
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[EVAL].reshape(-1, 3).astype(np.float32)
        rng = np.random.default_rng(self.seed)
        gmm = GaussianMixture(K, covariance_type="full", random_state=self.seed).fit(lab[rng.choice(len(lab), SAMPLE, replace=False)])
        mu = gmm.means_
        chroma = np.hypot(mu[:, 1] - 128, mu[:, 2] - 128)
        pavement = int(np.argmax(gmm.weights_))
        yellow_k = int(np.argmax(mu[:, 2]))
        yellow_k = yellow_k if mu[yellow_k, 2] >= MIN_B and yellow_k != pavement else None
        neutral = [k for k in range(K) if chroma[k] < MAX_CHROMA and k != pavement and k != yellow_k]
        white_k = max(neutral, key=lambda k: mu[k, 0]) if neutral else None
        white_k = white_k if white_k is not None and mu[white_k, 0] >= mu[pavement, 0] + MIN_L_GAIN else None

        labels = gmm.predict(lab).reshape(warped[EVAL].shape[:2])
        yellow = np.zeros(warped.shape[:2], np.uint8)
        white = np.zeros_like(yellow)
        if yellow_k is not None:
            yellow[EVAL][labels == yellow_k] = 255
        if white_k is not None:
            white[EVAL][labels == white_k] = 255
        self.found.append({"yellow": None if yellow_k is None else mu[yellow_k].round(1).tolist(), "white": None if white_k is None else mu[white_k].round(1).tolist()})
        return Masks(yellow, white)

    def describe(self) -> dict:
        def summary(key):
            m = np.array([f[key] for f in self.found if f[key] is not None])
            return {"frames": int(len(m)), "median_Lab": np.median(m, axis=0).round(1).tolist()} if len(m) else None

        return {"yellow": summary("yellow"), "white": summary("white")}
