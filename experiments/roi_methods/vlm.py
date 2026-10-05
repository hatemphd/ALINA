"""Minimal OpenAI vision call shared by Methods 3 and 4.

The API key is read from OPENAI_API_KEY, or from the git-ignored .env file at the repo root.
It is never printed or written anywhere.
"""

from __future__ import annotations

import base64
import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MODEL = os.environ.get("Q3_VLM_MODEL", "gpt-5.5")
SEND_WIDTH = 1280  # frames are downscaled to this width before sending; replies are scaled back


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return
    for raw in env.read_text().splitlines():
        key, sep, value = raw.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    _load_dotenv()
    return OpenAI(max_retries=6)  # backs off and retries on transient rate limits


def draw_grid(img: np.ndarray) -> np.ndarray:
    """Labelled pixel grid every 10% of the frame; vision models read positions off it far better."""
    out = img.copy()
    h, w = out.shape[:2]
    for i in range(1, 10):
        x, y = round(w * i / 10), round(h * i / 10)
        cv2.line(out, (x, 0), (x, h), (255, 255, 0), 1)
        cv2.line(out, (0, y), (w, y), (255, 255, 0), 1)
        cv2.putText(out, f"x={x}", (x + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(out, f"y={y}", (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return out


def ask_json(img: np.ndarray, prompt: str, seed: int = 42, model: str | None = None, grid: bool = True) -> tuple[dict, float]:
    """Send the frame and the prompt; return the parsed JSON reply and the scale back to full-size pixels."""
    model = model or MODEL
    h, w = img.shape[:2]
    scale = w / SEND_WIDTH
    small = cv2.resize(img, (SEND_WIDTH, round(h / scale)))
    if grid:
        small = draw_grid(small)
        prompt += "\nThe image has a labelled pixel grid (cyan lines every 10%) to help you read coordinates."
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("could not encode frame")
    url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    sampling = {} if model.startswith("gpt-5") else {"temperature": 0}  # GPT-5 models only allow the default
    response = _client().chat.completions.create(
        model=model,
        seed=seed,
        **sampling,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.format(width=small.shape[1], height=small.shape[0])},
                    {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
                ],
            }
        ],
    )
    return json.loads(response.choices[0].message.content), scale
