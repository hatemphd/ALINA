# Demo video: end-to-end pipeline with the best methods

[Back to README](../README.md#demo-video) | [Report](REPORT.md) | [Q3 and Q4 Summary](Q3_Q4_Summary.md) | [Building a video from images](building_video.md)

![Demo preview](../results/demo/alina_best_method_preview.gif)

**Rendered clip:** [`results/demo/alina_best_method.mp4`](../results/demo/alina_best_method.mp4) (1:33, 1280×720, H.264, about 19 MB)
**Narrated video:** *add the YouTube or Google Drive link here after recording*

---

## What the clip shows

The **best pipeline** from the evaluation runs beside the **original manual pipeline** on all 150 frames:

| | ROI | Color step |
|---|---|---|
| **Best** (top row, green ROI) | Q3 M1 Hough, re-estimated on every frame. The best automated ROI overall and the only one that handles the curved `vidd_3` | ALINA's yellow threshold (still best end to end on yellow) plus the Q4 decision tree's learned white box |
| **Baseline** (bottom row, gray ROI) | Q2 medium trapezoid, drawn once by hand | ALINA's hand-set HSV threshold only |

Each frame panel shows:
- **Left:** the best (top) and baseline (bottom) camera views with the ROI, ALINA's line labels in **red**, and learned white paint in **blue**. Below each view: *LINE FOUND*, the pixel count and the processing time, or *no line found*.
- **Right:** the best pipeline's bird's-eye view, and its mask (yellow paint in yellow, white paint in blue; gray marks the columns ALINA ignores).

The frames are sparse samples from long recordings, so they are shown at 2 per second, with a title card per video.

**How it was made:**
- **No re-labeling:** it is rendered from saved results. The labels and per-frame ROIs come from the Q3 every-frame Hough run, the baseline from the Q2 medium run, and the white box from the Q4 tree trained without that video.
- **White is displayed, not traced:** the white detections are shown on screen, but ALINA's red labels come from the yellow threshold, as recommended in the report.

Regenerate (about 20 s):

```bash
uv run python -m experiments.make_demo_video
```

### Timeline (for narration)

| Time | Content |
|---|---|
| 0:00–0:04 | Intro card: the two manual steps and the best pipeline |
| 0:04–0:07 | `vidd_1` card: sunny, two yellow lines + white stripe; CBEM F1 23.8 → 71.5 |
| 0:07–0:32 | `vidd_1` frames: the best pipeline finds the line and the white stripe; the manual ROI often finds nothing |
| 0:32–0:35 | `vidd_2` card: overcast, straight line; CBEM F1 91.6 → 82.8 |
| 0:35–1:00 | `vidd_2` frames: both pipelines work; the manual ROI is already good here |
| 1:00–1:03 | `vidd_3` card: curved taxiway, faded paint; 0 → 22 frames labeled |
| 1:03–1:28 | `vidd_3` frames: the manual ROI never finds the line; the per-frame Hough ROI follows the curve on about half the frames |
| 1:28–1:33 | Takeaway card |

---

## Recording the narrated walkthrough (3–4 minutes)

**Setup:**
1. Open the README and `results/demo/alina_best_method.mp4` (QuickTime).
2. Open Terminal in `~/ALINA`.
3. Start a recording with **QuickTime Player → File → New Screen Recording**: click **Options**, choose your microphone, then record the full screen or a window. On recent macOS you can also press **Shift-Cmd-5**.

**Suggested script:**

1. **Problem (about 30 s).** Show the README overview.
   > "ALINA labels taxiway centerlines, but it has two manual steps: a person draws a region of interest on the first frame, and the color thresholds that decide what counts as paint are set by hand. In Q2 I found the ROI is fragile: its size alone took one video from 47 of 50 frames labeled to zero."

2. **Methods (about 45 s).** Scroll to the Q3 and Q4 method tables.
   > "For Q3 I tried five ways to choose the ROI automatically: Hough lines, k-means clustering, GPT-5.5 drawing the box, GPT-5.5 pointing at the line, and ridge regression. All share one trapezoid builder that keeps the line big and vertical in the bird's-eye view. For Q4 I tried five ways to learn the color thresholds: Otsu, a Gaussian mixture, a decision tree, logistic regression and a small neural network. The best combination was Hough lines for the ROI, plus ALINA's yellow threshold with the decision tree's learned white box."

3. **Live run (about 45 s).** In Terminal, run a short command and show its output, for example:

   ```bash
   uv run python -m experiments.q3_score
   ```

   > "Everything runs headless from these scripts; this one scores every run against the ground truth."

4. **Demo clip (about 90 s).** Play `alina_best_method.mp4` and narrate along the timeline above.
   - **`vidd_1`:** "Top is the best pipeline, bottom is the original. The automated ROI finds the line, in red, and the white stripe, in blue; the manual ROI misses it in many frames."
   - **`vidd_2`:** "On a straight taxiway, both work; a good manual ROI is already fine."
   - **`vidd_3`:** "On the curve, the manual ROI never finds the line. Re-estimating the ROI on every frame follows the curve."

5. **Results and limits (about 30 s).** Show the best-method table in the README.
   > "Choosing where to look gave the big gains: F1 from 24 to 72 on video one, and the curved video from zero to 22 frames labeled. Learned thresholds added white-line detection. The main limit is the small ground truth: 17 frames, none for the curved video."

**Export and share:**
- Save the recording, then use **File → Export As → 1080p** in QuickTime.
- Upload it to **YouTube as Unlisted**, or to **Google Drive** with "Anyone with the link can view".
- Paste the link at the top of this page and in the README's [Demo video](../README.md#demo-video) section.
- The rendered clip (`results/demo/`) is small enough to commit. Keep the narrated recording on YouTube or Drive; it is usually larger.
