"""Q4: replace ALINA's hand-set HSV threshold with learned color models.

Every method gets the same fixed ROI per video (FIXED_ROI in experiments/color_methods/common.py); only the step
that turns the bird's-eye warp into a binary mask changes. Supervised methods are trained leave-one-video-out.

--use yellow   feed the yellow mask to ALINA (what the original threshold does; CBEM ground truth is yellow paint)
--use both     feed yellow | white

Writes results/q4/runs/<method>/<use>/<video>/:
    coords/*.txt    detected line pixels per frame (same format as alina label)
    timing.log      per-frame status + summary line (same format as alina label)
    mask_ms.txt     time of the color step alone, per frame
    pixels.json     (yellow runs) pixel counts of the masks vs. CBEM yellow and the white labels
    method.json     learned thresholds / model summary
    masks/          bird's-eye | yellow mask | white mask, for frame 1 and the labeled test frames

Usage (repo root):
    uv run python -m experiments.q4_color_threshold                                    # everything
    uv run python -m experiments.q4_color_threshold --methods m3_tree --videos vidd_2 --use yellow
Then score with:  uv run python -m experiments.q4_score
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .color_methods import METHODS
from .color_methods.common import (
    EVAL,
    VIDEOS,
    WHITE_TEST,
    WHITE_TRAIN,
    X0,
    X1,
    Y0,
    Y1,
    cbem_frames,
    cbem_yellow,
    frames,
    label_frame,
    warp,
    white_band,
    white_label,
)
from .color_methods.training import video_samples

OUT = Path("results/q4")
USES = ("yellow", "both")


def counts(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict:
    p, t, v = pred[EVAL] > 0, truth[EVAL] > 0, valid[EVAL] > 0
    return {"tp": int((p & t & v).sum()), "fp": int((p & ~t & v).sum()), "fn": int((~p & t & v).sum()), "valid": int(v.sum())}


def white_test_frames(video: str) -> list[str]:
    """Frames with known white truth: the traced vidd_1 stripe frames, plus CBEM frames without white paint."""
    if video == "vidd_1":
        return sorted(set(WHITE_TEST) | set(cbem_frames(video)))
    return cbem_frames(video)


def save_panel(path: Path, warped: np.ndarray, masks) -> None:
    tiles = [warped[Y0:Y1, X0:X1], cv2.cvtColor(masks.yellow[Y0:Y1, X0:X1], cv2.COLOR_GRAY2BGR), cv2.cvtColor(masks.white[Y0:Y1, X0:X1], cv2.COLOR_GRAY2BGR)]
    for tile, title in zip(tiles, ("bird's-eye", "yellow mask", "white mask")):
        cv2.putText(tile, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite(str(path), cv2.resize(np.hstack(tiles), None, fx=0.4, fy=0.4))


def run_job(method: str, video: str, use: str, seed: int) -> str:
    out = OUT / "runs" / method / use / video
    (out / "coords").mkdir(parents=True, exist_ok=True)
    model = METHODS[method].Method()
    start = time.perf_counter()
    model.fit(video, seed)
    fit_s = time.perf_counter() - start

    def mask_fn(warped):
        t = time.perf_counter()
        m = model.masks(warped)
        mask_times.append(time.perf_counter() - t)
        return m

    paths = frames(video)
    cbem, white_frames = set(cbem_frames(video)), set(white_test_frames(video))
    if use == "yellow":
        (out / "masks").mkdir(exist_ok=True)
    keep_panel = {paths[0].stem} | cbem | white_frames
    pixels = {"yellow": [], "white": []}
    mask_times: list[float] = []
    labeled, total = 0, 0.0
    with open(out / "timing.log", "w") as log:
        for i, path in enumerate(paths, start=1):
            img = cv2.imread(str(path))
            t = time.perf_counter()
            coords, had_lines, masks = label_frame(img, video, mask_fn, use)
            elapsed = time.perf_counter() - t
            np.savetxt(out / "coords" / f"{path.stem}.txt", coords, fmt="%6d")
            labeled += had_lines
            total += elapsed
            log.write(f"[{i}/{len(paths)}] [{'Labeled' if had_lines else 'No lines found'}] {path.name}: {len(coords)} px, elapsed {elapsed:.3f}s ({datetime.now():%H:%M:%S})\n")
            log.flush()
            if use != "yellow":
                continue
            if path.stem in cbem:
                truth, band = cbem_yellow(img, video, path.stem)
                pixels["yellow"].append({"frame": path.stem, **counts(masks.yellow, truth, band)})
            if path.stem in white_frames:
                truth = warp(white_label(img, video, path.stem), video, nearest=True)
                pixels["white"].append({"frame": path.stem, **counts(masks.white, truth, white_band(video))})
            if path.stem in keep_panel:
                save_panel(out / "masks" / f"{path.stem}.jpg", warp(img, video), masks)
        avg_ms = total / len(paths) * 1000
        summary = f"Done: {labeled}/{len(paths)} labeled, avg {avg_ms:.1f}ms/frame ({1000 / avg_ms:.2f} fps), total {total:.1f}s"
        log.write(summary + "\n")
    (out / "mask_ms.txt").write_text("\n".join(f"{1000 * s:.1f}" for s in mask_times) + "\n")
    (out / "method.json").write_text(json.dumps({"method": method, "video": video, "seed": seed, "fit_s": round(fit_s, 1), **model.describe()}, indent=1, default=float))
    if use == "yellow":
        (out / "pixels.json").write_text(json.dumps(pixels, indent=1))
    return f"{method}/{use}/{video}: {summary}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--videos", nargs="+", default=list(VIDEOS), choices=list(VIDEOS))
    parser.add_argument("--use", nargs="+", default=list(USES), choices=USES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    # build the training-sample caches once, before the workers need them
    needed = {METHODS[m].Method.feature_name: METHODS[m].Method.feature_fn for m in args.methods if METHODS[m].SUPERVISED}
    for name, fn in needed.items():
        for video in VIDEOS:
            X, y = video_samples(video, fn, name, args.seed)
            print(f"training samples {name}/{video}: {len(y)} ({', '.join(f'class {c}: {(y == c).sum()}' for c in np.unique(y))})", flush=True)
        video_samples("vidd_1", fn, name, args.seed, only=WHITE_TRAIN)

    jobs = [(m, v, u) for u in args.use for m in args.methods for v in args.videos]
    print(f"Running {len(jobs)} jobs on {args.workers} processes", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_job, m, v, u, args.seed) for m, v, u in jobs]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)


if __name__ == "__main__":
    main()
