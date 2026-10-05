# Building a video from the frame images

[Back to README](../README.md#demo-video) | [Demo video](DEMO_VIDEO.md) | [Saving the experiment data](saving_experiments_data.md)

This page shows how to turn a folder of labeled frames into an MP4 by hand, using ffmpeg, a short Python command, or iMovie. The ready-made demo, which compares the best pipeline against the manual one with captions and the bird's-eye view, is built by `uv run python -m experiments.make_demo_video`; see [DEMO_VIDEO.md](DEMO_VIDEO.md).

Run all commands from the repo root (`cd ~/ALINA`).

---

## 1. Which image folders have full frame sets

| Folder | Frames | Shows |
|---|---|---|
| `results/q2/<video>/<size>/annotated/` | All 50 | Q2 manual-ROI runs, with ALINA's line labels in red |
| `data/Labeled_Data/<video>/annotations/` | All 50 | The authors' original ALINA output |
| `results/q3/runs/<method>/<video>/<mode>/annotated/` | **Only frame 1 + the CBEM frames** | Q3 runs; the other frames were not saved, to save disk |

- **Sizes:** `<size>` is `tight`, `medium`, `loose`, and for `vidd_3` also `medium_s40` and `medium_loose_thresh`.
- **Not in git:** the `results/q2/**/annotated/` and `results/q3/**/annotated/` folders are git-ignored (see [saving_experiments_data.md](saving_experiments_data.md)). They exist in your local copy and in the Google Drive zip, but not in a fresh clone; re-run the experiment to regenerate them.
- **Best Q3 method on every frame:** there are no saved images to stitch. Use the rendered `results/demo/alina_best_method.mp4`, which draws the labels from the saved coordinates, or re-run that method.

---

## 2. Option 1: ffmpeg (standard tool, one command)

Install once:

```bash
brew install ffmpeg
```

Build a video from one folder:

```bash
ffmpeg -framerate 2 -pattern_type glob -i 'results/q2/vidd_2/medium/annotated/*.jpg' \
  -vf "scale=1280:-2" -c:v libx264 -pix_fmt yuv420p ~/Desktop/vidd_2_medium.mp4
```

| Part | Meaning |
|---|---|
| `-framerate 2` | 2 images per second. Use 5–10 for smoother playback, 1 to linger on each frame. The 50 frames are sparse samples, so 1–2 reads best |
| `-pattern_type glob -i '…/*.jpg'` | Takes the images in filename order (00001, 00002, …) |
| `scale=1280:-2` | Shrinks the 1080p frames to 720p for a smaller file |
| `-c:v libx264 -pix_fmt yuv420p` | H.264, which plays in QuickTime, browsers and YouTube |

### Side-by-side comparison from two folders

Both folders must contain the same frame names (true for any two runs on the same video). Example: `vidd_3`, default thresholds vs. loosened thresholds:

```bash
ffmpeg -framerate 2 -pattern_type glob -i 'results/q2/vidd_3/medium/annotated/*.jpg' \
       -framerate 2 -pattern_type glob -i 'results/q2/vidd_3/medium_loose_thresh/annotated/*.jpg' \
  -filter_complex "[0]scale=960:-2[a];[1]scale=960:-2[b];[a][b]hstack" \
  -c:v libx264 -pix_fmt yuv420p ~/Desktop/vidd_3_compare.mp4
```

Use `vstack` instead of `hstack` to stack them top and bottom.

### Joining several videos

```bash
printf "file '%s'\n" ~/Desktop/vidd_1_medium.mp4 ~/Desktop/vidd_2_medium.mp4 ~/Desktop/vidd_3_medium.mp4 > /tmp/list.txt
ffmpeg -f concat -safe 0 -i /tmp/list.txt -c copy ~/Desktop/all_videos.mp4
```

All parts need the same size and frame rate (true if they were built with the same command).

---

## 3. Option 2: Python with what is already installed (no ffmpeg)

OpenCV is already in the project environment:

```bash
uv run python -c "
import cv2, glob, os
files = sorted(glob.glob('results/q2/vidd_2/medium/annotated/*.jpg'))
out = os.path.expanduser('~/Desktop/vidd_2_medium.mp4')
if os.path.exists(out):
    os.remove(out)  # the macOS H.264 writer will not overwrite a file
w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'avc1'), 2, (1280, 720))
for f in files:
    w.write(cv2.resize(cv2.imread(f), (1280, 720)))
w.release()
print(len(files), 'frames ->', out)
"
```

- **Folder:** change it in the `glob(...)` line.
- **Speed:** change the frame rate (the `2` after `fourcc`).
- **Codec:** `avc1` (H.264) works through macOS's AVFoundation. On other systems, use `mp4v` if `avc1` fails.

---

## 4. Option 3: iMovie (no commands)

1. Open iMovie and choose **Create New → Movie**.
2. Open **iMovie → Settings** and set **Photo placement: Fit in Frame**. This stops the automatic zoom effect (Ken Burns) on every image.
3. Choose **Import Media** and select all 50 JPGs from the folder, then drag them onto the timeline in order.
4. Select all the images on the timeline, click the **clock** icon, and set **Duration** to 0.5 s (or 1 s).
5. Add title cards between videos with the **Titles** tab. Record narration with the **microphone** button under the viewer.
6. Export with **File → Share → File**, at 1080p.

---

## 5. Re-creating the full demo instead

The rendered demo already does the comparison, captions, bird's-eye view, masks and title cards, and takes about 20 s:

```bash
uv run python -m experiments.make_demo_video
# -> results/demo/alina_best_method.mp4 and results/demo/alina_best_method_preview.gif
```

To narrate it, see [DEMO_VIDEO.md](DEMO_VIDEO.md#recording-the-narrated-walkthrough-34-minutes).
