# Saving the experiment data

[Back to README](../README.md#saving-the-experiment-data) | [Q2](Q2_MANUAL_ROI.md) | [Q3](Q3_AUTO_ROI.md) | [Q4](Q4_COLOR_THRESHOLD.md)

The Q2–Q4 runs took about 13.6 hours of pipeline time (see [why the headless runner was needed](Q2_MANUAL_ROI.md#how-the-runs-were-executed)) and produced about 350 MB of outputs. This page explains how to keep a full copy on Google Drive, and which parts go to GitHub.

**The rule:** everything goes to Google Drive as one zip. GitHub gets the evidence and results that the write-ups use, but not the bulky, regenerable images and caches.

---

## 1. What the experiments produce

| Folder | Size | Contents |
|---|---|---|
| `results/q2/` | 207 MB | 11 manual-ROI runs: annotated frames, coords, `timing.log`, ROI evidence |
| `results/q3/` | 89 MB | 30 auto-ROI runs, cached GPT replies, `summary.csv`, overlays |
| `results/q4/` | 47 MB | 36 color-threshold runs, mask panels, `summary.csv`, `pixels.csv`, figures, training cache |
| `results/summary/` | 0.1 MB | Cross-question plots |
| `outputs/` | 15 MB | `vidd_1` frames resized to 1080p (Q2 Step 1) |

## 2. Save everything to Google Drive

Run this in Terminal. It creates one dated zip on your Desktop:

```bash
cd ~/ALINA && zip -r -q ~/Desktop/ALINA_results_$(date +%Y-%m-%d).zip results outputs -x "*.DS_Store" && ls -lh ~/Desktop/ALINA_results_*.zip
```

- **Size:** expect roughly 330–350 MB. Most of it is JPEG images, which don't compress further.
- **No secrets:** the key file `.env` is in neither folder, so it is never in the zip.
- **Check the archive:**

  ```bash
  unzip -l ~/Desktop/ALINA_results_*.zip | tail -1     # file count and total size
  unzip -l ~/Desktop/ALINA_results_*.zip | grep -c timing.log   # should be 77 (11 + 30 + 36 runs)
  ```

- **Upload:** open [drive.google.com](https://drive.google.com), click **New → File upload**, and choose the zip. Or drag it into a Drive folder, or into the Google Drive app's folder if it is installed. Keep the date in the name so later re-runs don't overwrite it.
- **Zips stay out of git:** `*.zip` is git-ignored, so an archive saved inside the repo by mistake is never committed.

### Restoring from the zip

```bash
cd ~/ALINA && unzip -o ~/Downloads/ALINA_results_YYYY-MM-DD.zip
```

This puts `results/` and `outputs/` back in place. The scoring and figure scripts (`q3_score`, `q4_score`, `q4_figures`, `summary_plots`) then work without re-running any experiment.

---

## 3. What goes to GitHub, and what doesn't

### Kept out of GitHub (Google Drive only)

These rules are in [`.gitignore`](../.gitignore), so `git add` skips them automatically:

| Ignored | Size | Why |
|---|---|---|
| `results/q2/**/annotated/*` | 195 MB | Every frame with red line pixels drawn on: bulky and regenerable |
| `results/q3/**/annotated/*` | 64 MB | Same, for Q3 |
| `results/q4/train_cache/` | 10 MB | Binary training-sample cache, rebuilt automatically on the next Q4 run |
| `outputs/` | 15 MB | Resized `vidd_1` frames, regenerable with Q2 Step 1 |
| `*.zip` | – | Archives belong on Drive |
| `.env` | – | **Your OpenAI API key; never commit it** |

**Exceptions:** two annotated frames are embedded in the Q2 write-up, so they stay in GitHub:
- `results/q2/vidd_2/medium/annotated/00010.jpg`
- `results/q2/vidd_3/medium_loose_thresh/annotated/18990.jpg`

The rules ignore the folders' *contents* rather than the folders themselves; that is what makes these exceptions possible.

The rules added to `.gitignore`:

```gitignore
# Bulky, regenerable experiment outputs: archived on Google Drive instead
results/q2/**/annotated/*
results/q3/**/annotated/*
!results/q2/vidd_2/medium/annotated/00010.jpg
!results/q2/vidd_3/medium_loose_thresh/annotated/18990.jpg
results/q4/train_cache/
*.zip
```

### Committed to GitHub (about 83 MB, 4,300 files)

| Kept | Why |
|---|---|
| `results/q3/summary.csv`, `results/q4/summary.csv`, `results/q4/pixels.csv` | The numbers in every results table |
| Each run's `timing.log` | Frames labeled and time per frame, in ALINA's own log format |
| `roi.json`, `roi_overlay.jpg`, `birds_eye.jpg`, `mask.jpg`, `compare.jpg` | ROI evidence for every Q2 run and Q3 first-frame run |
| Q4 `method.json`, `pixels.json`, `mask_ms.txt`, `masks/` | Learned thresholds, pixel counts, color-step timing, mask panels |
| `results/q3/overlays/`, `results/q4/figures/`, `results/summary/` | Plots embedded in the write-ups |
| `results/q3/proposals/` (0.3 MB) | **Cached GPT-5.5 replies.** They reproduce the GPT results without an API key or spending credits |
| `coords/` folders (about 59 MB of small text files) | Detected pixels per frame; the scoring scripts read these. With them, anyone can re-score in seconds instead of re-running about 13.6 h of experiments |

**Optional, smaller repo:** to keep the `coords/` folders on Drive only, add `results/**/coords/` to `.gitignore`. The summaries and evidence are still committed, but re-scoring would then need the zip or a full re-run.

---

## 4. Check before committing

```bash
cd ~/ALINA
git check-ignore -v .env                                        # must print a .gitignore rule: .env is ignored
git status --short | head -30                                   # overview of what will be committed
git ls-files --others --exclude-standard results | grep -c annotated   # should print 2 (the embedded frames)
git ls-files --others --exclude-standard -z | xargs -0 du -ck | tail -1   # total size to be committed, in KB
```

Then commit and push as usual:

```bash
git add -A
git diff --cached --name-only | grep -x ".env" && echo "STOP: .env is staged" || echo "OK: .env not staged"
git diff --cached --name-only | grep -c "/annotated/"   # should print 2
git commit -m "Assignment 4: Q2-Q4 experiments, results and write-ups"
git push origin main
```

Every file is far below GitHub's 100 MB per-file limit; the largest is under 1 MB.

---

## 5. After re-running experiments

1. Re-run the zip command (step 2). The new date keeps it separate from the old archive.
2. Upload the new zip to Drive.
3. Commit the updated summaries, logs and evidence. The `.gitignore` rules keep the bulky folders out automatically.
