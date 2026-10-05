# What is on GitHub vs. Google Drive

[Back to README](../README.md) | [Saving the experiment data](saving_experiments_data.md) | [Building a video from images](building_video.md)

The repository on GitHub holds the full assignment: code, docs, the dataset, experiment results, figures, scores, cached GPT replies and the demo video. A few things are deliberately kept out by `.gitignore`. They stay on the local machine and in the Google Drive zip (see [saving_experiments_data.md](saving_experiments_data.md)).

---

## 1. Bulky experiment outputs you can regenerate (Google Drive only)

| Path | What it is |
|---|---|
| `results/q2/**/annotated/*` | Annotated frame images from the Q2 manual-ROI runs (50 per run) |
| `results/q3/**/annotated/*` | Annotated frame images from the Q3 runs (frame 1 and the CBEM frames per run) |
| `results/q4/train_cache/` | Cached training pixels for the Q4 machine-learning methods; rebuilt automatically on the next run |
| `*.zip` | The Google Drive archive itself, if it was created inside the repo |

Two Q2 frames are kept in git because [Q2_MANUAL_ROI.md](Q2_MANUAL_ROI.md) shows them:

- `results/q2/vidd_2/medium/annotated/00010.jpg`
- `results/q2/vidd_3/medium_loose_thresh/annotated/18990.jpg`

The coordinates, scores, logs and figures from every run are on GitHub, so every number in the report can be checked from the repo. Only the per-frame images are missing.

---

## 2. Secrets (never on GitHub)

| Path | What it is |
|---|---|
| `.env` | The OpenAI API key, used only by the Q3 GPT methods M3 and M4 |

`.env.example`, which holds only a placeholder, is on GitHub. See the README section on providing the OpenAI API key.

---

## 3. Local environment and system clutter

| Path | What it is |
|---|---|
| `.venv/` | The Python environment. Rebuild it with `uv sync` from the committed `uv.lock` |
| `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` | Python caches |
| `.DS_Store`, `.vscode/`, `.idea/` | macOS and editor files |
| `*.log` outside `results/` | Stray run logs (the logs under `results/` are kept) |

---

## 4. Checking exactly what was skipped

```bash
cd ~/ALINA
git status --ignored --short | grep '^!!'
du -sh results/q2/*/*/annotated results/q3/runs/*/*/*/annotated results/q4/train_cache .venv 2>/dev/null
```

---

## 5. Keep a copy before deleting anything

Keep the local `results/` folder until the Google Drive zip is uploaded and verified. Without either, the annotated frames can only be recovered by re-running the experiments, which took about 13.6 hours of pipeline time.
