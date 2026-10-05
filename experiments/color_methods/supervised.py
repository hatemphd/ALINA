"""Shared fit/predict for the supervised pixel classifiers (classes 0 background, 1 yellow, 2 white)."""

from __future__ import annotations

import numpy as np

from .common import EVAL, VIDEOS, Masks
from .training import training_set


class PixelClassifier:
    feature_fn = None
    feature_name = ""

    def make_model(self, seed: int):
        raise NotImplementedError

    def fit(self, test_video: str, seed: int) -> None:
        X, y = training_set(test_video, list(VIDEOS), type(self).feature_fn, self.feature_name, seed)
        self.counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
        self.model = self.make_model(seed).fit(X, y)

    def masks(self, warped: np.ndarray) -> Masks:
        feats = type(self).feature_fn(warped)[EVAL]
        pred = self.model.predict(feats.reshape(-1, feats.shape[2])).reshape(feats.shape[:2])
        yellow = np.zeros(warped.shape[:2], np.uint8)
        white = np.zeros_like(yellow)
        yellow[EVAL][pred == 1] = 255
        white[EVAL][pred == 2] = 255
        return Masks(yellow, white)

    def describe(self) -> dict:
        return {"training_samples": self.counts}
