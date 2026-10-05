"""Method 4 (multimodal LLM + geometry): GPT points at the centerline; the shared builder draws the ROI.

The model only does what it is good at, recognising things: it marks points along the ego
centerline and the row where the nose starts. The trapezoid then comes from build_trapezoid,
so the line ends up vertical and clear of ALINA's dead zone, the same as Methods 1, 2 and 5.
"""

from __future__ import annotations

import numpy as np

from .common import Proposal, build_trapezoid, fallback_proposal, fit_line
from .vlm import ask_json

NAME = "m4_vlm_points"

PROMPT = """This is a {width}x{height} pixel frame from a camera in a small aircraft's cockpit, taxiing at an airport.
The aircraft's nose is visible at the bottom of the frame.

1. Find the yellow taxiway centerline that the aircraft is following (the painted line on the pavement directly ahead).
   Give 5 points that lie exactly on that line, starting where it disappears behind the nose and going forward
   (up the image) for about 15% of the image height. If the line curves, follow the curve.
2. Give the y coordinate of the top edge of the aircraft's nose, directly below the centerline.

Reply with JSON only, in pixel coordinates of this image (x to the right, y down):
{{"centerline_points": [[x, y], [x, y], [x, y], [x, y], [x, y]], "nose_top_y": y}}"""


def propose(img: np.ndarray, seed: int = 42, video: str | None = None) -> Proposal:
    h, w = img.shape[:2]
    try:
        reply, scale = ask_json(img, PROMPT, seed)
        points = np.array(reply["centerline_points"], dtype=float).reshape(-1, 2) * scale
        nose_y = int(round(float(reply["nose_top_y"]) * scale))
    except Exception as exc:
        return fallback_proposal(img, f"VLM error: {type(exc).__name__}")

    if len(points) < 2 or np.ptp(points[:, 1]) < 0.01 * h or not (0.4 * h < nose_y < h):
        return fallback_proposal(img, "VLM points invalid")
    line = fit_line(points[:, 0], points[:, 1])
    return Proposal(build_trapezoid(line, nose_y, w, h), line, nose_y, note="VLM points")
