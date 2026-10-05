"""M3 decision tree: learns explicit HSV threshold boxes, a drop-in replacement for the hand-set inRange bounds.

A depth-3 tree on ALINA's normalized (H, S, V), balanced class weights, seed 42. Every leaf that predicts yellow
(or white) is an axis-aligned HSV box, i.e. one cv2.inRange(lower, upper); the mask is the union of those boxes.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from alina.color import normalize_color_features

from .supervised import PixelClassifier

NAME = "m3_tree"
KIND = "Supervised, explicit HSV boxes"
SUPERVISED = True
CLASSES = {1: "yellow", 2: "white"}


def hsv_features(warped: np.ndarray) -> np.ndarray:
    return normalize_color_features(warped).astype(np.float32)


class Method(PixelClassifier):
    feature_fn = staticmethod(hsv_features)
    feature_name = "hsv"

    def make_model(self, seed: int):
        return DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed)

    def boxes(self) -> dict[str, list[dict]]:
        t = self.model.tree_
        out = {name: [] for name in CLASSES.values()}

        def walk(node, lower, upper):
            if t.children_left[node] == -1:
                cls = int(self.model.classes_[np.argmax(t.value[node])])
                if cls in CLASSES:
                    out[CLASSES[cls]].append({"lower_hsv": [int(np.floor(v)) + 1 if v > 0 else 0 for v in lower], "upper_hsv": [int(np.floor(v)) for v in upper]})
                return
            f, thr = t.feature[node], t.threshold[node]
            up = list(upper)
            up[f] = min(up[f], thr)
            walk(t.children_left[node], lower, up)
            lo = list(lower)
            lo[f] = max(lo[f], thr)
            walk(t.children_right[node], lo, upper)

        walk(0, [0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
        return out

    def describe(self) -> dict:
        return {**super().describe(), "boxes": self.boxes()}
