"""Method 3 (multimodal LLM, end to end): GPT returns the four ROI corners directly.

The prompt states ALINA's rules for a good ROI, learned in Q2. The reply is only validated
(inside the frame, top above bottom, non-degenerate) and forced to horizontal top and bottom
edges, as ALINA itself does. No geometry is added on top.
"""

from __future__ import annotations

import numpy as np

from .common import Proposal, fallback_proposal, validate_corners
from .vlm import ask_json

NAME = "m3_vlm_corners"

PROMPT = """This is a {width}x{height} pixel frame from a camera in a small aircraft's cockpit, taxiing at an airport.
The aircraft's nose is visible at the bottom of the frame.

I need a region of interest (ROI) for a lane-marking detector. Draw a trapezoid on the pavement that:
- contains the yellow taxiway centerline the aircraft is following, closest to the aircraft;
- has its bottom edge just above the aircraft's nose (no part of the aircraft inside), and its top edge further ahead;
- has horizontal top and bottom edges, with the top edge narrower than the bottom edge (perspective);
- has the centerline crossing both the top and bottom edges near their middle, so the line runs through the trapezoid from bottom to top;
- is roughly 30% of the frame width at the bottom and only about 6% of the frame height tall (a shallow strip just ahead of the nose);
- excludes sky, grass, buildings, other aircraft and the propeller where possible.

Reply with JSON only, in pixel coordinates of this image (x to the right, y down):
{{"bottom_left": [x, y], "top_left": [x, y], "top_right": [x, y], "bottom_right": [x, y]}}"""


def propose(img: np.ndarray, seed: int = 42, video: str | None = None) -> Proposal:
    h, w = img.shape[:2]
    try:
        reply, scale = ask_json(img, PROMPT, seed)
        corners = np.array([reply[k] for k in ("bottom_left", "top_left", "top_right", "bottom_right")], dtype=float) * scale
    except Exception as exc:  # network, refusal or malformed JSON
        return fallback_proposal(img, f"VLM error: {type(exc).__name__}")

    y_bottom = round((corners[0, 1] + corners[3, 1]) / 2)
    y_top = round((corners[1, 1] + corners[2, 1]) / 2)
    corners[[0, 3], 1] = y_bottom
    corners[[1, 2], 1] = y_top
    valid = validate_corners(corners, w, h)
    if valid is None:
        return fallback_proposal(img, "VLM corners invalid")
    return Proposal(valid, nose_y=None, note="VLM corners")
