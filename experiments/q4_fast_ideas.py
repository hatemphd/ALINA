"""Fast follow-up to Q4: mask quality of extra color-step ideas, scored exactly like results/q4/pixels.csv.

Only the pixel-level test is run (no ALINA end-to-end labeling), on the scored frames of vidd_1 and vidd_2, with the
same fixed ROI, leave-one-video-out training data (results/q4/train_cache) and seed. m0 and m5 are re-scored as a
consistency check against results/q4/pixels.csv.

Ideas:
    i1_forest     random forest (100 trees, depth 12) on the 20 context features
    i2_boosting   histogram gradient boosting on the 20 context features
    i3_bayes      Gaussian naive Bayes on the 6 HSV + Lab features
    i4_mlp_open   m5 MLP followed by a 3x3 morphological opening of each mask
    i5_clahe      ALINA's hand-set threshold after CLAHE contrast equalization of the warp

Usage (repo root):  uv run python -m experiments.q4_fast_ideas
Writes results/q4_ideas/pixels.csv
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from .color_methods import m0_baseline, m5_mlp
from .color_methods.common import EVAL, VIDEOS, WHITE_TEST, Masks, cbem_frames, cbem_yellow, context_features, frame_paths, pixel_features, warp, white_band, white_label
from .color_methods.supervised import PixelClassifier
from .q4_color_threshold import counts, white_test_frames
from .q4_score import iou_f1

OUT = Path("results/q4_ideas")
SEED = 42


class Forest(PixelClassifier):
    feature_fn = staticmethod(context_features)
    feature_name = "context"

    def make_model(self, seed):
        return RandomForestClassifier(100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=seed)


class Boosting(PixelClassifier):
    feature_fn = staticmethod(context_features)
    feature_name = "context"

    def make_model(self, seed):
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed)


class Bayes(PixelClassifier):
    feature_fn = staticmethod(pixel_features)
    feature_name = "hsvlab"

    def make_model(self, seed):
        return GaussianNB()


class MlpOpen(m5_mlp.Method):
    def masks(self, warped):
        m = super().masks(warped)
        k = np.ones((3, 3), np.uint8)
        return Masks(cv2.morphologyEx(m.yellow, cv2.MORPH_OPEN, k), cv2.morphologyEx(m.white, cv2.MORPH_OPEN, k))


class Clahe(m0_baseline.Method):
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def masks(self, warped):
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return super().masks(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))


IDEAS = {
    "m0_baseline (re-run)": m0_baseline.Method,
    "m5_mlp (re-run)": m5_mlp.Method,
    "i1_forest": Forest,
    "i2_boosting": Boosting,
    "i3_bayes": Bayes,
    "i4_mlp_open": MlpOpen,
    "i5_clahe": Clahe,
}


def score(name: str, cls) -> dict:
    data, mask_ms, fit_s = {}, [], 0.0
    for video in ("vidd_1", "vidd_2"):
        model = cls()
        t = time.perf_counter()
        model.fit(video, SEED)
        fit_s += time.perf_counter() - t
        cbem, white = set(cbem_frames(video)), set(white_test_frames(video))
        data[video] = {"yellow": [], "white": []}
        for path in frame_paths(video):
            if path.stem not in cbem | white:
                continue
            img = cv2.imread(str(path))
            warped = warp(img, video)
            t = time.perf_counter()
            masks = model.masks(warped)
            mask_ms.append(1000 * (time.perf_counter() - t))
            if path.stem in cbem:
                truth, band = cbem_yellow(img, video, path.stem)
                data[video]["yellow"].append({"frame": path.stem, **counts(masks.yellow, truth, band)})
            if path.stem in white:
                truth = warp(white_label(img, video, path.stem), video, nearest=True)
                data[video]["white"].append({"frame": path.stem, **counts(masks.white, truth, white_band(video))})
    row = {"method": name}
    row["yellow_iou"], row["yellow_f1"] = iou_f1(data["vidd_1"]["yellow"] + data["vidd_2"]["yellow"])
    row["white_iou"], row["white_f1"] = iou_f1([r for r in data["vidd_1"]["white"] if r["frame"] in WHITE_TEST])
    neg = [r for r in data["vidd_1"]["white"] if r["frame"] not in WHITE_TEST] + data["vidd_2"]["white"]
    row["white_fp_pct"] = round(100 * sum(r["fp"] for r in neg) / sum(r["valid"] for r in neg), 2)
    row["mask_ms"] = round(float(np.mean(mask_ms)), 1)
    row["fit_s"] = round(fit_s, 1)
    return row


def main() -> None:
    assert set(VIDEOS) >= {"vidd_1", "vidd_2"}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cls in IDEAS.items():
        rows.append(score(name, cls))
        print(" | ".join(str(v) for v in rows[-1].values()), flush=True)
    with open(OUT / "pixels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
