"""Demo video of the end-to-end pipeline with the best methods, next to the original manual pipeline.

Best pipeline (from the Q3/Q4 evaluation):
    ROI      Q3 M1 Hough, re-estimated on every frame (best automated ROI overall; the only one that handles vidd_3)
    color    ALINA's yellow threshold (still best end to end on yellow) + the Q4 decision tree's learned white box
Baseline:    the Q2 medium ROI drawn once by hand + ALINA's hand-set threshold.

Everything is rendered from saved results, so nothing is re-labeled:
    results/q3/proposals/m1_hough/<video>.json            per-frame ROIs
    results/q3/runs/m1_hough/<video>/every-frame/          detected line pixels and timing
    results/q2/<video>/medium/                             baseline ROI, line pixels and timing
    results/q4/runs/m3_tree/yellow/<video>/method.json     learned white box (tree trained without that video)
The white detections are drawn for display; ALINA's labels (red) come from the yellow threshold, as recommended.

Writes results/demo/alina_best_method.mp4 (H.264, 1280x720) and results/demo/alina_best_method_preview.gif.

Usage (repo root, after Q2-Q4):  uv run python -m experiments.make_demo_video
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from alina.color import normalize_color_features
from alina.config import PipelineConfig

from .q3_score import per_frame_log
from .roi_methods.common import CORNER_NAMES, frame_paths, load_coords

OUT = Path("results/demo")
W, H = 1280, 720
FPS, HOLD, CARD_SECONDS = 10, 5, 4  # each data frame is held HOLD/FPS = 0.5 s
GIF_PER_VIDEO, GIF_WIDTH = 8, 720  # GIF: frames where the best pipeline finds the line, baseline misses first
_CFG = PipelineConfig(input_dir=".", output_images_dir="/tmp/q4_unused", output_coords_dir="/tmp/q4_unused")
DST = np.array([_CFG.roi.dst_bottom_left, _CFG.roi.dst_top_left, _CFG.roi.dst_top_right, _CFG.roi.dst_bottom_right], np.float32)
(X0, Y0), (X1, Y1) = _CFG.roi.dst_top_left, _CFG.roi.dst_bottom_right

GREEN, RED, BLUE, WHITE, YELLOW, GRAY = (80, 220, 80), (0, 0, 255), (255, 140, 0), (255, 255, 255), (0, 220, 255), (170, 170, 170)

VIDEO_NOTES = {
    "vidd_1": ("Sunny, two yellow lines + a white lane stripe", "CBEM F1: manual 23.8 -> best 71.5"),
    "vidd_2": ("Overcast, straight centerline", "CBEM F1: manual 91.6 -> best 82.8 (good manual ROI on a straight line)"),
    "vidd_3": ("Curved taxiway, faded paint, no ground truth", "Frames labeled: manual 0/50 -> best 22/50"),
}


def text(img, s, org, scale=0.7, color=WHITE, thick=2):
    if thick >= 2:
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_8)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_8)


def corners_from(d: dict) -> np.ndarray:
    return np.array([d[n] for n in CORNER_NAMES], np.float32)


def load_inputs(video: str) -> dict:
    proposals = json.loads(Path(f"results/q3/proposals/m1_hough/{video}.json").read_text())
    best_run = Path(f"results/q3/runs/m1_hough/{video}/every-frame")
    base_run = Path(f"results/q2/{video}/medium")
    tree = json.loads(Path(f"results/q4/runs/m3_tree/yellow/{video}/method.json").read_text())
    return {
        "proposals": proposals,
        "best_dir": best_run / "coords",
        "best_log": per_frame_log(best_run / "timing.log"),
        "base_dir": base_run / "coords",
        "base_log": per_frame_log(base_run / "timing.log"),
        "base_roi": corners_from(json.loads((base_run / "roi.json").read_text())["corners_px"]),
        "white_boxes": tree["boxes"]["white"],
    }


def warp_masks(img: np.ndarray, quad: np.ndarray, white_boxes: list[dict]):
    M = cv2.getPerspectiveTransform(quad, DST)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    hsv = normalize_color_features(warped)
    yellow = cv2.inRange(hsv, np.array(_CFG.yellow_lower), np.array(_CFG.yellow_upper))
    white = np.zeros_like(yellow)
    for box in white_boxes:
        white |= cv2.inRange(hsv, np.array(box["lower_hsv"]), np.array(box["upper_hsv"]))
    inside = np.zeros_like(yellow)
    inside[Y0:Y1, _CFG.mask_ignore_left_columns : X1] = 255
    yellow &= inside
    white &= inside
    return M, warped, yellow, white


def camera_view(img, quad, coords, white_cam=None, roi_color=GREEN):
    view = img.copy()
    if white_cam is not None:
        view[white_cam > 0] = (0.4 * view[white_cam > 0] + 0.6 * np.array(BLUE)).astype(np.uint8)
    if len(coords):
        view[coords[:, 1], coords[:, 0]] = RED
    cv2.polylines(view, [quad.astype(np.int32)], True, roi_color, 4, cv2.LINE_AA)
    return view


def crop_lower(view: np.ndarray) -> np.ndarray:
    """Drop the sky: rows 380-1080 hold the taxiway, the ROI and the aircraft nose."""
    return view[380:1080]


def status(log: dict, stem: str, coords: np.ndarray) -> tuple[str, tuple]:
    labeled, seconds = log.get(stem, (False, 0.0))
    return (f"LINE FOUND  {len(coords)} px  {seconds:.1f}s" if labeled else "no line found"), (GREEN if labeled else (60, 60, 255))


def panel(video: str, path: Path, inp: dict, index: int, total: int) -> np.ndarray:
    img = cv2.imread(str(path))
    stem = path.stem
    best_quad = corners_from(inp["proposals"][stem]["corners"])
    M, warped, yellow, white = warp_masks(img, best_quad, inp["white_boxes"])
    white_cam = cv2.warpPerspective(white, np.linalg.inv(M), (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
    best_coords, base_coords = load_coords(inp["best_dir"] / f"{stem}.txt"), load_coords(inp["base_dir"] / f"{stem}.txt")

    canvas = np.full((H, W, 3), 25, np.uint8)
    # left column: best (top) and baseline (bottom) camera views, 800 x 292 each
    best = cv2.resize(crop_lower(camera_view(img, best_quad, best_coords, white_cam)), (800, 292))
    base = cv2.resize(crop_lower(camera_view(img, inp["base_roi"], base_coords, roi_color=GRAY)), (800, 292))
    canvas[60:352, 0:800] = best
    canvas[408:700, 0:800] = base
    text(canvas, "BEST: Hough ROI every frame + learned white box", (10, 50), 0.75, GREEN)
    text(canvas, "BASELINE: manual Q2 ROI + hand-set HSV threshold", (10, 398), 0.75, GRAY)
    s, c = status(inp["best_log"], stem, best_coords)
    text(canvas, s, (12, 342), 0.7, c)
    s, c = status(inp["base_log"], stem, base_coords)
    text(canvas, s, (12, 690), 0.7, c)

    # right column: bird's-eye view and mask of the best pipeline
    bird = cv2.resize(warped[Y0:Y1, X0:X1], (470, 286))
    mask = np.zeros((Y1 - Y0, X1 - X0, 3), np.uint8)
    mask[yellow[Y0:Y1, X0:X1] > 0] = YELLOW
    mask[white[Y0:Y1, X0:X1] > 0] = BLUE
    mask[:, : _CFG.mask_ignore_left_columns - X0] = (40, 40, 40)
    mask = cv2.resize(mask, (470, 286), interpolation=cv2.INTER_NEAREST)
    canvas[66:352, 805:1275] = bird
    canvas[414:700, 805:1275] = mask
    text(canvas, "bird's-eye view (best ROI)", (812, 50), 0.65)
    text(canvas, "mask: yellow paint / white paint", (812, 398), 0.65)
    text(canvas, "(gray: ALINA ignores these columns)", (812, 690), 0.5, GRAY, 1)

    text(canvas, f"{video}  frame {stem}  ({index}/{total})", (10, 22), 0.6, WHITE, 1)
    text(canvas, "red = ALINA line labels   blue = white paint", (700, 22), 0.6, WHITE, 1)
    return canvas


def card(lines: list[tuple[str, float, tuple]]) -> np.ndarray:
    canvas = np.full((H, W, 3), 25, np.uint8)
    y = 200
    for s, scale, color in lines:
        size = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        text(canvas, s, ((W - size[0]) // 2, y), scale, color)
        y += int(70 * scale) + 20
    return canvas


INTRO = [
    ("ALINA: automating the two manual steps", 1.2, WHITE),
    ("Q3: choose the ROI automatically   |   Q4: learn the color thresholds", 0.75, GRAY),
    ("Best pipeline: Hough ROI on every frame (Q3)", 0.8, GREEN),
    ("+ ALINA yellow threshold + decision-tree white box (Q4)", 0.8, GREEN),
    ("compared with the manual Q2 ROI and hand-set thresholds", 0.75, GRAY),
    ("Frames are sparse samples, shown at 2 per second", 0.6, GRAY),
]
OUTRO = [
    ("Takeaway", 1.2, WHITE),
    ("Choosing where to look (ROI) gave the big gains:", 0.8, GREEN),
    ("vidd_1 CBEM F1 23.8 -> 71.5,  vidd_3 0 -> 22 frames labeled", 0.75, WHITE),
    ("Learned color thresholds added white-line detection (white F1 0.96-0.98)", 0.75, WHITE),
    ("Simple ideas used well beat heavy models", 0.8, GREEN),
    ("Ground truth is small: 17 frames, none for vidd_3", 0.6, GRAY),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    video_path = OUT / "alina_best_method.mp4"
    video_path.unlink(missing_ok=True)  # the macOS H.264 writer refuses to overwrite
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit("could not open an H.264 video writer")
    gif: list[Image.Image] = []

    def hold(frame: np.ndarray, n: int) -> None:
        for _ in range(n):
            writer.write(frame)

    hold(card(INTRO), FPS * CARD_SECONDS)
    for video in ("vidd_1", "vidd_2", "vidd_3"):
        inp = load_inputs(video)
        title, result = VIDEO_NOTES[video]
        hold(card([(video, 1.4, WHITE), (title, 0.8, GRAY), (result, 0.8, GREEN)]), FPS * 3)
        paths = frame_paths(video)
        found = [p.stem for p in paths if inp["best_log"].get(p.stem, (False,))[0]]
        best_only = [s for s in found if not inp["base_log"].get(s, (False,))[0]]
        gif_stems = set((best_only + [s for s in found if s not in best_only])[:GIF_PER_VIDEO])
        for i, path in enumerate(paths, start=1):
            frame = panel(video, path, inp, i, len(paths))
            hold(frame, HOLD)
            if path.stem in gif_stems:
                small = cv2.resize(frame, (GIF_WIDTH, GIF_WIDTH * H // W), interpolation=cv2.INTER_AREA)
                gif.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        print(f"{video}: {len(paths)} frames rendered", flush=True)
    hold(card(OUTRO), FPS * (CARD_SECONDS + 1))
    writer.release()

    gif_path = OUT / "alina_best_method_preview.gif"
    gif[0].save(gif_path, save_all=True, append_images=gif[1:], duration=900, loop=0, optimize=True)
    print(f"wrote {video_path} ({video_path.stat().st_size / 1e6:.1f} MB) and {gif_path} ({gif_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
