"""M0 baseline: ALINA's hand-set range on normalized HSV, (0, 70, 170)-(255, 255, 255). No white rule."""

from __future__ import annotations

import cv2
import numpy as np

from alina.color import normalize_color_features
from alina.config import PipelineConfig

from .common import Masks

NAME = "m0_baseline"
KIND = "Rule (hand-set)"
SUPERVISED = False
_C = PipelineConfig(input_dir=".", output_images_dir="/tmp/q4_unused", output_coords_dir="/tmp/q4_unused")


class Method:
    def fit(self, test_video: str, seed: int) -> None:
        pass

    def masks(self, warped: np.ndarray) -> Masks:
        yellow = cv2.inRange(normalize_color_features(warped), np.array(_C.yellow_lower), np.array(_C.yellow_upper))
        return Masks(yellow, np.zeros_like(yellow))

    def describe(self) -> dict:
        return {"yellow": {"lower_hsv": list(_C.yellow_lower), "upper_hsv": list(_C.yellow_upper)}, "white": None}
