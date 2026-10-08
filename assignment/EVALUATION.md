# Evaluation approach for Q3 and Q4

[Back to README](../README.md#evaluation-rigor) | [Assignment brief](../Assgn_4_Fall_26.pdf) | [Q3](Q3_AUTO_ROI.md) | [Q4](Q4_COLOR_THRESHOLD.md)

Q3a, Q3b, and Q4 are graded on creativity and on rigor: at least 5 methods each, with tables using ALINA's metrics plus any extra metrics needed (25 points). This file defines how every method is evaluated so the results are comparable.

---

## Ground truth

- **Source:** CBEM ground truth in [`data/gt_alina_labels/canny_textfiles/`](../data/gt_alina_labels/canny_textfiles/), 120 frames (40 per video). See [validation data](../BACKGROUND.md#validation-data).
- **Gap:** only 17 of those 120 frames have their raw image in `data/Raw_Data/` (7 in `vidd_1`, 10 in `vidd_2`, 0 in `vidd_3`). See [the gap](../BACKGROUND.md#validation-gap). A new method can only be scored on frames you can run it on.
- **Options to widen it:**
  - Create CBEM frames for `vidd_3` with `alina cbem` (interactive: trace a contour, Canny runs inside it). `vidd_3` has none, so its accuracy is unknown.
  - Obtain the full AssistTaxi frames that match the other 103 CBEM files.
  - Re-trace the duplicated `vidd_1` frames (see below).

  None of these was done here. The one extension made was **white-paint labels** for Q4 (below), because the dataset has none.
- **Dataset quirk:** the CBEM files are partly copies. `vidd_1` frames 00002–00007 share one identical file, and 01063/01065 share another, so `vidd_1`'s 7 scored frames contain only 2 distinct tracings. In 5 of them the tracing covers only the left of the two yellow lines.
- **Frames used in this study:**

| Purpose | `vidd_1` | `vidd_2` | `vidd_3` | Used in |
|---|---|---|---|---|
| All raw frames (runs, frames labeled, time) | 50 | 50 | 50 | Q2, Q3, Q4 |
| CBEM accuracy (ALINA metric) | 7: 00002, 00004–00007, 01063, 01065 | 10: 00001–00010 | 0 | Q3, Q4 |
| Published ALINA labels, non-empty (agreement; Q3 M5 and Q4 training) | 50 | 48 | 42 | Q3, Q4 |
| Filled CBEM yellow masks (pixel test) | 7 | 10 | 0 | Q4 |
| Hand-traced white stripe: test / train | 00002, 00004–00007 / 00010, 00014, 00018 | – | – | Q4 |
| Frames with no white paint (white false positives) | 01063, 01065 | 10 CBEM frames | – | Q4 |

`vidd_1` raw frames are the 4K originals resized to 1080p (`outputs/vidd_1_1080p`); `vidd_2` and `vidd_3` are used as provided. The data sources are explained in [data_used_in_q3.md](data_used_in_q3.md).

## Metrics

| Metric | Definition | Source |
|---|---|---|
| Recall (detection rate) | TP / (TP + FN) | ALINA paper, [`eval/metrics.py`](../eval/metrics.py) |
| Precision | TP / (TP + FP) | [`eval/metrics.py`](../eval/metrics.py) |
| F1 | 2PR / (P + R) | [`eval/metrics.py`](../eval/metrics.py) |
| Time per frame (ms) and FPS | Wall-clock per frame | `run_batch` log in [`alina/pipeline.py`](../alina/pipeline.py) |
| Frames labeled | Frames where a line was found | `run_batch` log |
| Published-label F1 (agreement) | ALINA metric vs. the authors' published labels; reproduction, not accuracy | [`q3_score.py`](../experiments/q3_score.py), [`q4_score.py`](../experiments/q4_score.py) |
| ROI IoU vs. manual | Overlap of an automated first-frame ROI with the Q2 medium ROI | Q3, `q3_score.py` |
| ROI time (s) | Time to propose one ROI (GPT calls included) | Q3, `q3_score.py` |
| Fallbacks | Frames where the ROI method failed and the centred fallback ROI was used | Q3, `q3_score.py` |
| ROI jitter (px) and IoU with previous ROI | Mean corner movement and overlap between consecutive per-frame ROIs | Q3b, `q3_score.py` |
| Mask IoU and pixel F1, yellow and white | Pooled pixel TP / FP / FN of the color mask inside the usable bird's-eye ROI, vs. filled CBEM (yellow) and the traced stripe (white). Exact pixel matching, so stricter than the x-set / y-set metric | Q4, `q4_score.py` |
| White false-positive % | Share of ROI pixels called white on frames with no white paint | Q4, `q4_score.py` |
| Color-step time (ms) | Time of the mask step alone, separate from the whole pipeline | Q4, `q4_score.py` |

**Caveat on ALINA's evaluator:** [`eval/evaluate.py`](../eval/evaluate.py) compares the *set of x values* and the *set of y values* separately and averages them. It does not match exact `(x, y)` pixel pairs, so its scores are generous. Report it for comparison with the paper, and consider adding a stricter pixel-pair metric (optionally with a small distance tolerance).

## Reproducibility

- **Seeds:** one seed, 42, passed through `--seed`. It is used by scikit-learn (`KMeans`, `RANSACRegressor`, `GaussianMixture`, `DecisionTreeClassifier`, `LogisticRegression`, `MLPClassifier`), by NumPy's `default_rng` for pixel sampling, by OpenCV k-means, and as the `seed` field of OpenAI requests. GPT-5 models accept neither `temperature` nor a guaranteed seed, so their replies can vary. Every reply is cached in `results/q3/proposals/`, and the reported results are reproduced exactly from the cache.
- **Versions:** pinned in `uv.lock` (Python 3.12.12, OpenCV 5.0.0.93, NumPy 2.5.3, scikit-learn 1.9.1, openai 2.48+). External model: OpenAI `gpt-5.5` (Q3 M3 and M4 only).
- **Repeated runs:** one run per method and video. All methods except the GPT ones are deterministic given the seed, so repeating them gives identical numbers. Re-running the GPT methods costs new API calls (the 146 calls that failed when the credits ran out during Q3b were completed in a second pass, but no full repeat was made). Run-to-run variation was therefore not measured as mean ± standard deviation. The closest evidence is the Q3 model comparison, where GPT-5.5 nose estimates stayed within about ±10 px across settings. This is a stated limitation.
- **Leakage control:** supervised methods are trained leave-one-video-out (Q3 M5, Q4 M3–M5). Q4 white uses disjoint train and test frames of `vidd_1`, the only video with white paint.
- **Outputs:** bulky intermediates go in `outputs/` (git-ignored). Tables, plots and per-run evidence go in `results/q2/`, `results/q3/`, `results/q4/` and `results/summary/`.

## Comparison tables and plots

### Best method per question

CBEM F1 is the ALINA metric vs. ground truth in %. `vidd_3` has no ground truth, so frames labeled out of 50 are shown.

| Question | Baseline | Best method | `vidd_1` CBEM F1 | `vidd_2` CBEM F1 | `vidd_3` frames labeled | Notes |
|---|---|---|---|---|---|---|
| Q2 (manual ROI) | – | Medium ROI (`vidd_2`), tight (`vidd_1`) | 23.8 (medium) | **91.6** (medium) | 0 (46 with loosened thresholds) | ROI size alone swings 0 → 47 frames |
| Q3a (first-frame ROI) | Manual: 23.8 / 91.6 / 0 | **M1 Hough** (`vidd_1`, `vidd_3`), **M4 GPT points** (`vidd_2`) | **70.0** (M1) | 91.3 (M4) | 16 (M1) | Line-centred, shallow trapezoid from Q2 |
| Q3b (every-frame ROI) | Same method, first frame | **M1 Hough** | **71.5** | 82.8 (M1); 91.0 (M4) | **22** | Helps curves and bad first ROIs, not straight taxiways |
| Q4 (color step) | Hand-set HSV: 70.0 / 83.2 / 16 | **M5 MLP** for masks, **M3 tree** for explicit thresholds | 70.0 (baseline); 67.0 (MLP) | 83.2 (baseline); 81.9 (MLP) | **35** (tree, LogReg) | MLP: yellow mask F1 0.907, white 0.983; baseline white F1 0 |

### Plots

Made by `uv run python -m experiments.summary_plots` (after `q3_score` and `q4_score`).

- **Recall, precision and F1 per method**, Q3a, Q3b and Q4, for `vidd_1` and `vidd_2`:

  ![F1 bars](../results/summary/f1_bars.png)

- **Speed vs. accuracy**, every scored run. Time is ALINA's whole pipeline per frame; it is dominated by CIRCLEDAT, not by the ROI or color method:

  ![Time vs F1](../results/summary/time_vs_f1.png)

Question-specific plots: the Q3 ROI overlays in `results/q3/overlays/`, and the Q4 HSV histograms and mask comparisons in `results/q4/figures/`.
