"""M4 logistic regression: a linear color boundary in HSV + Lab (6 features), balanced class weights.

Unlike an inRange box the boundary may be oblique, e.g. "high b* relative to L", which a per-channel range cannot express.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import pixel_features
from .supervised import PixelClassifier

NAME = "m4_logreg"
KIND = "Supervised, linear"
SUPERVISED = True
FEATURES = ["H", "S", "V", "L", "a", "b"]


class Method(PixelClassifier):
    feature_fn = staticmethod(pixel_features)
    feature_name = "hsvlab"

    def make_model(self, seed: int):
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))

    def describe(self) -> dict:
        lr = self.model[-1]
        coef = {int(c): dict(zip(FEATURES, lr.coef_[i].round(2).tolist())) for i, c in enumerate(lr.classes_)}
        return {**super().describe(), "standardized_coefficients": coef}
