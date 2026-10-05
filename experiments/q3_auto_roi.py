"""Q3: run ALINA with automatically proposed ROIs.

Modes:
    first-frame  (Q3a) propose the ROI on frame 1 and reuse it for every frame, like a human clicking once
    every-frame  (Q3b) propose a new ROI on every frame

Each proposal is cached in results/q3/proposals/<method>/<video>.json, so VLM calls are made once
and re-runs are free and reproducible. Labeling uses ALINA's own process_image(), unchanged.

Writes results/q3/runs/<method>/<video>/<mode>/:
    coords/*.txt   detected line pixels per frame (same format as alina label)
    annotated/     red overlay, only for frame 1 and frames with CBEM ground truth (to save disk)
    timing.log     per-frame status + summary line (same format as alina label)
and, for first-frame mode, the ROI evidence (roi.json, roi_overlay.jpg, birds_eye.jpg, mask.jpg, compare.jpg).

Usage (repo root):
    uv run python -m experiments.q3_auto_roi                                  # all methods, videos, modes
    uv run python -m experiments.q3_auto_roi --methods m4_vlm_points --videos vidd_2 --modes first-frame
Then score with:  uv run python -m experiments.q3_score
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from alina.config import PipelineConfig
from alina.pipeline import process_image

from .roi_methods import METHODS
from .roi_methods.common import (
    CBEM_GT,
    CORNER_NAMES,
    VIDEO_DIRS,
    Line,
    fallback_proposal,
    frame_paths,
    is_usable,
    sanitize,
    to_roi_points,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from q2_run_headless import save_roi_evidence  # noqa: E402

OUT = Path("results/q3")
MODES = ("first-frame", "every-frame")
VLM_METHODS = {"m3_vlm_corners", "m4_vlm_points"}


def proposals_path(method: str, video: str) -> Path:
    return OUT / "proposals" / method / f"{video}.json"


def load_proposals(method: str, video: str) -> dict:
    path = proposals_path(method, video)
    return json.loads(path.read_text()) if path.exists() else {}


def propose_one(method: str, video: str, path: Path, seed: int) -> tuple[str, dict]:
    img = cv2.imread(str(path))
    start = time.perf_counter()
    p = sanitize(img, METHODS[method].propose(img, seed, video))
    seconds = time.perf_counter() - start
    return path.stem, to_record(p, seconds)


def to_record(p, seconds: float) -> dict:
    return {
        "corners": {name: [int(x), int(y)] for name, (x, y) in zip(CORNER_NAMES, p.corners)},
        "line": None if p.line is None else {"slope": p.line.slope, "intercept": p.line.intercept},
        "nose_y": p.nose_y,
        "fallback": p.fallback,
        "note": p.note,
        "seconds": round(seconds, 3),
    }


def resanitize(cache: dict, frames: list[Path]) -> int:
    """Apply sanitize() to cached proposals made before the check existed; keeps VLM replies, no new calls."""
    fixed = 0
    for f in frames:
        r = cache.get(f.stem)
        if r is None or r["fallback"]:
            continue
        line = None if r["line"] is None else Line(**r["line"])
        corners = np.array([r["corners"][n] for n in CORNER_NAMES])
        if not is_usable(corners, line, 1920, 1080):
            cache[f.stem] = to_record(fallback_proposal(cv2.imread(str(f)), f"unusable ROI ({r['note']})"), r["seconds"])
            fixed += 1
    return fixed


def ensure_proposals(method: str, video: str, frames: list[Path], seed: int, vlm_workers: int) -> dict:
    cache = load_proposals(method, video)
    # failed API calls (quota, rate limit, network) are not results: ask again instead of reusing the fallback
    cache = {s: r for s, r in cache.items() if not r["note"].startswith("VLM error")}
    missing = [f for f in frames if f.stem not in cache]
    fixed = resanitize(cache, frames)
    if fixed:
        print(f"  {method}/{video}: replaced {fixed} unusable cached ROI(s) with the fallback", flush=True)
    if missing or fixed:
        workers = vlm_workers if method in VLM_METHODS else 1
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(propose_one, method, video, f, seed) for f in missing]
            for i, fut in enumerate(as_completed(futures), 1):
                stem, record = fut.result()
                cache[stem] = record
                print(f"  {method}/{video}: proposed {i}/{len(missing)} ({stem}{', fallback' if record['fallback'] else ''})", flush=True)
        path = proposals_path(method, video)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(cache.items())), indent=1))
    return cache


def corners_of(record: dict) -> np.ndarray:
    return np.array([record["corners"][n] for n in CORNER_NAMES], dtype=np.int32)


def run_pipeline(method: str, video: str, mode: str, rois: list[tuple[str, list]]) -> str:
    """rois: (frame path, corners) per frame. Runs in a worker process."""
    out = OUT / "runs" / method / video / mode
    config = PipelineConfig(
        input_dir=VIDEO_DIRS[video],
        output_images_dir=out / "annotated",
        output_coords_dir=out / "coords",
        log_file=out / "timing.log",
    )
    keep_annotated = {Path(rois[0][0]).stem} | {p.stem for p in CBEM_GT[video].glob("*.txt")}
    labeled, total = 0, 0.0
    with open(config.log_file, "w") as log:
        for i, (path, corners) in enumerate(rois, start=1):
            path = Path(path)
            start = time.perf_counter()
            annotated, coords, had_lines = process_image(cv2.imread(str(path)), to_roi_points(np.array(corners)), config)
            elapsed = time.perf_counter() - start
            np.savetxt(config.output_coords_dir / f"{path.stem}.txt", coords, fmt="%6d")
            if path.stem in keep_annotated:
                cv2.imwrite(str(config.output_images_dir / path.name), annotated)
            labeled += had_lines
            total += elapsed
            status = "Labeled" if had_lines else "No lines found"
            log.write(f"[{i}/{len(rois)}] [{status}] {path.name}: {len(coords)} px, elapsed {elapsed:.3f}s ({datetime.now():%H:%M:%S})\n")
            log.flush()
        avg_ms = total / len(rois) * 1000
        summary = f"Done: {labeled}/{len(rois)} labeled, avg {avg_ms:.1f}ms/frame ({1000 / avg_ms:.2f} fps), total {total:.1f}s"
        log.write(summary + "\n")
    return f"{method}/{video}/{mode}: {summary}"


def draw_all_first_frame_rois(video: str, methods: list[str]) -> None:
    colors = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255)]
    first = frame_paths(video)[0]
    canvas = cv2.imread(str(first))
    manual = json.loads((Path("results/q2") / video / "medium" / "roi.json").read_text())["corners_px"]
    cv2.polylines(canvas, [np.array([manual[n] for n in CORNER_NAMES], np.int32)], True, (255, 255, 255), 3)
    cv2.putText(canvas, "manual (Q2 medium)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    for i, (method, color) in enumerate(zip(methods, colors), start=1):
        record = load_proposals(method, video).get(first.stem)
        if record:
            cv2.polylines(canvas, [corners_of(record)], True, color, 4)
            cv2.putText(canvas, method, (30, 50 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    out = OUT / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{video}_first_frame_rois.jpg"), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--videos", nargs="+", default=list(VIDEO_DIRS), choices=list(VIDEO_DIRS))
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6, help="parallel ALINA runs (processes)")
    parser.add_argument("--vlm-workers", type=int, default=6, help="parallel VLM requests (threads)")
    args = parser.parse_args()

    jobs = []
    for video in args.videos:
        frames = frame_paths(video)
        for method in args.methods:
            needed = frames if "every-frame" in args.modes else frames[:1]
            print(f"{method}/{video}: proposing ROIs for {len(needed)} frame(s)", flush=True)
            cache = ensure_proposals(method, video, needed, args.seed, args.vlm_workers)
            first = cache[frames[0].stem]
            for mode in args.modes:
                if mode == "first-frame":
                    rois = [(str(f), corners_of(first).tolist()) for f in frames]
                    out = OUT / "runs" / method / video / mode
                    out.mkdir(parents=True, exist_ok=True)
                    save_roi_evidence(out, cv2.imread(str(frames[0])), to_roi_points(corners_of(first)), PipelineConfig(VIDEO_DIRS[video], out / "annotated", out / "coords"), f"{video} {method}")
                else:
                    rois = [(str(f), corners_of(cache[f.stem]).tolist()) for f in frames]
                jobs.append((method, video, mode, rois))
        draw_all_first_frame_rois(video, list(METHODS))

    print(f"Running ALINA: {len(jobs)} runs on {args.workers} processes", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(run_pipeline, *job) for job in jobs]):
            print(fut.result(), flush=True)


if __name__ == "__main__":
    main()
