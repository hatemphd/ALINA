"""M5 neural network with neighborhood context: an MLP (32-16 hidden units, seed 42) on 20 features per pixel:
HSV + Lab of the pixel, their 5x5 and 15x15 means, and the 5x5 standard deviation of V and L.

The context features let it use shape cues a pure color rule cannot (a thin bright stripe vs. a large bright area
such as the aircraft nose). It stands in for a small CNN, which needs PyTorch (not available on this Intel Mac).
"""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import context_features
from .supervised import PixelClassifier

NAME = "m5_mlp"
KIND = "Neural network, pixel + context"
SUPERVISED = True


class Method(PixelClassifier):
    feature_fn = staticmethod(context_features)
    feature_name = "context"

    def make_model(self, seed: int):
        return make_pipeline(StandardScaler(), MLPClassifier((32, 16), early_stopping=True, max_iter=200, random_state=seed))
