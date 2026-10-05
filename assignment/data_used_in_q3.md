# Data used in Q3

[Back to README](../README.md#q3-automating-roi-selection) | [Q3 details](Q3_AUTO_ROI.md) | [Q2 summary](Q2Summary.md)

Q3 uses two kinds of data: what ships with ALINA, and what Q2 produced.

## Original ALINA data

- **Raw frames** (`data/Raw_Data/vidd_1..3`, 50 per video; `vidd_1` uses the resized `outputs/vidd_1_1080p`): the input every ROI method and labeling run works on.
- **CBEM ground truth** (`data/gt_alina_labels/canny_textfiles_1..3`): the only true accuracy measure. Only 17 frames overlap the raw data, 7 in `vidd_1` and 10 in `vidd_2`. These give recall, precision and F1. `vidd_3` has none.
- **Published ALINA labels** (`data/Labeled_Data/vidd_1..3/textfiles`, all 150 frames): the authors' own ALINA output, used two ways.
  - As an agreement score on every frame, including `vidd_3`. It measures how closely we reproduce the authors' labeling, not true accuracy.
  - As training targets for the supervised method, always tested on a video it didn't train on.

## Q2 data

- **Manual ROIs** (each run's `roi.json`): the human baseline. Each automated ROI is compared with it by overlap (ROI IoU).
- **Q2 outputs** (`coords/` and `timing.log`): scored with the same metrics as the Q3 methods, so the table shows whether automation beats the hand-drawn ROI.
- **Q2 lessons**: built into every method's trapezoid. The line must end up thick and vertical in the bird's-eye view, clear of the 300-column dead zone, and the bottom edge sits just above the nose.
- **`vidd_3` finding**: Q3 keeps the default thresholds, which tests whether a well-placed ROI alone fixes the curved-line failure.
