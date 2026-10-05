# Q1. Getting set up: execution and analysis

[Back to README](../README.md#q1-getting-set-up) | [Assignment brief](../Assgn_4_Fall_26.pdf)

This file holds the detailed record for Q1. The commands to run and the short summary live in the [README](../README.md#q1-getting-set-up).

---

## Q1a. Read the ALINA paper

- Paper: [arXiv 2406.08775](https://arxiv.org/abs/2406.08775) (CVPR 2024 Workshops)
- Summary and Q&A: [BACKGROUND.md](../BACKGROUND.md)

### Takeaways that drive Q2–Q4

| Paper concept | Why it matters for this assignment | Code |
|---|---|---|
| ROI drawn once on the first frame | Q3 replaces this manual step with ML | [`alina/roi.py`](../alina/roi.py) |
| Bird's-eye perspective warp | Q2 studies how ROI size changes the warp | [`alina/pipeline.py`](../alina/pipeline.py) (`process_image`) |
| Hand-set HSV thresholds `(0, 70, 170)`–`(255, 255, 255)` | Q4 replaces them with ML | [`alina/config.py`](../alina/config.py), [`alina/pipeline.py`](../alina/pipeline.py) |
| Recall (detection rate) vs. CBEM ground truth | The metric all experiments must report | [`eval/evaluate.py`](../eval/evaluate.py), [`eval/metrics.py`](../eval/metrics.py) |

### Notes on the paper

*Draft for the author to edit. Each point is checked against the code or against our Q2–Q4 experiments.*

- **What ALINA is for:** labeling taxiway centerlines in aircraft camera footage semi-automatically, so that a person only draws an ROI once per video instead of annotating every frame. The output is a training dataset for line-detection models, not a real-time detector.
- **Pipeline in one line:** ROI, then bird's-eye warp, then normalized HSV threshold, then column histogram, then CIRCLEDAT traversal from the peak, then unwarp. Every step is classical computer vision; there is no learning anywhere in the pipeline.
- **The headline 98.45% detection rate** is reproduced exactly from the shipped output files (see Evidence below). Two caveats:
  - The metric compares the sets of x values and y values separately, not exact pixels, so it is generous.
  - It scores the authors' own outputs on frames where their ROI and thresholds were tuned.
- **The two manual inputs are the weak points.**
  - **The ROI:** in Q2, ROI size alone swung a video from 47/50 labeled frames to 0/50.
  - **The hand-set HSV thresholds:** in Q4 they detect no white paint at all, and they miss the dim paint of `vidd_3`.

  These are exactly the steps the assignment asks us to replace with ML (Q3 and Q4).
- **Hidden assumption: the line is thick and vertical after the warp.** The column histogram needs more than 200 mask pixels in a single column. Curved taxiways (`vidd_3`) and loose ROIs break this even when the color mask is correct.
- **Cost:** CIRCLEDAT is a pure-Python flood fill, so time per frame grows with the number of connected mask pixels: from about 50 ms (nothing found) to tens of seconds (large masks).
- **Validation is small:** 120 CBEM frames, of which only 17 have raw images in this repository (none for `vidd_3`). Any new method can be scored on just those 17.

---

## Q1b. Fork the repository and set it up locally (5 points)

### Fork

- Upstream: [github.com/hafeezkhan909/ALINA](https://github.com/hafeezkhan909/ALINA)
- Fork: [github.com/hatemphd/ALINA](https://github.com/hatemphd/ALINA)
- Local clone: `~/ALINA`, with `origin` set to `https://github.com/hatemphd/ALINA`

### Environment

| Item | Value |
|---|---|
| OS | macOS 15.7.4 (build 24G517), Intel x86_64 |
| uv version | 0.9.26 |
| Python version | 3.12.12 (chosen by uv; `uv python pin 3.11` was skipped, and the project accepts Python ≥ 3.9) |
| OpenCV / NumPy / Matplotlib | opencv-python 5.0.0.93 / numpy 2.5.3 / matplotlib 3.11.2 |
| ALINA package | `alina` 1.0.0, installed in editable mode from `~/ALINA` |

Setup steps follow [UV_ENV_SETUP.md](../UV_ENV_SETUP.md). `uv sync` created `.venv/`, `alina.egg-info/` (both git-ignored) and `uv.lock`, which should be committed so the exact versions above can be reproduced.

### Evidence

**`uv run alina --help`**

```
usage: alina [-h]
             {label,video-to-frames,rotate-frames,resize-images,cbem,evaluate,superimpose}
             ...

positional arguments:
  {label,video-to-frames,rotate-frames,resize-images,cbem,evaluate,superimpose}
    label               Detect and label taxiway line markings.
    video-to-frames     Extract frames from a video file.
    rotate-frames       Rotate a directory of frames.
    resize-images       Batch-resize a directory of images.
    cbem                Interactively create a context-based edge map (CBEM)
                        for one frame.
    evaluate            Compute precision/recall of ALINA output vs. CBEM
                        ground truth.
    superimpose         Overlay detected coordinates on a Canny edge map.

options:
  -h, --help            show this help message and exit
```

All 7 subcommands are available.

**`uv run alina evaluate` on the provided ground truth**

Expected (reproduces the paper's 98.45% detection rate):

```
Average Recall: 98.44245354113903%
Average Precision: 92.29333214425482%
Average F1: 95.16844076391271%
```

Our output (October 4, 2026):

```
Average Recall: 98.44245354113903%
Average Precision: 92.29333214425482%
Average F1: 95.16844076391271%
```

| Metric | Paper | Our run |
|---|---|---|
| Recall (detection rate) | 98.45% | 98.44% |
| Precision | not reported | 92.29% |
| F1 | not reported | 95.17% |

The setup reproduces the paper's detection rate. The 0.01-point difference is rounding plus averaging over the 120 ground-truth frames. Note that this scores ALINA's *shipped* output files against the ground truth; it does not re-run labeling. Precision is high partly because `eval/evaluate.py` matches x values and y values separately rather than exact pixel pairs (see [EVALUATION.md](EVALUATION.md)).

### Issues encountered

| Issue | Fix |
|---|---|
| None during `uv sync`, `alina --help`, or `alina evaluate` | – |
| `uv python pin 3.11` was skipped, so uv used Python 3.12.12 | No action needed (supported). Run `uv python pin 3.11 && uv sync` to match `UV_ENV_SETUP.md` exactly |
