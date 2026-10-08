# ML review and ideation: a student's study guide to every technique used

[Back to README](../README.md#report) | [ML methods in brief](ML_METHODS_EXPLAINED.md) | [Evaluation](EVALUATION.md) | [Q3](Q3_AUTO_ROI.md) | [Q4](Q4_COLOR_THRESHOLD.md)

This page is written for someone new to these techniques, or refreshing them. Each technique gets a **plain-English** explanation first, then a **deeper look** (how it works, its key settings, and what to watch out for), then **what happened** in this project.

It also covers:

- **How ALINA's scoring differs from a textbook F1**, with a quick experiment showing how much that matters.
- **Ideas for further experiments**, some of which were tried in a quick follow-up run, with results.

For the short version, see [ML_METHODS_EXPLAINED.md](ML_METHODS_EXPLAINED.md).

---

## Contents

1. [The basics you need first](#basics)
2. [Q3 techniques: finding the ROI](#q3)
3. [Q4 techniques: finding the paint colours](#q4)
4. [Supporting tools used by several methods](#supporting)
5. [How ALINA scores results, and how that differs from textbook F1](#metrics)
6. [Ideas for further experiments](#ideas)
7. [Fast follow-up experiment: five more colour methods](#fast-experiment)
8. [Self-test questions](#self-test)

---

<a id="basics"></a>
## 1. The basics you need first

**Supervised vs. unsupervised.**
- **Plain English:** supervised learning learns from examples that come with the right answer ("this pixel is yellow paint"). Unsupervised learning gets no answers and looks for structure by itself ("these pixels form a group").
- **Deeper:** supervised methods fit a function \( f(\text{features}) \to \text{label} \) by reducing a loss on labeled data. Unsupervised methods optimise something about the data itself, such as cluster tightness or likelihood under a model. In this project, Otsu, GMM, K-means and RANSAC are unsupervised. The decision tree, logistic regression, MLP and Ridge are supervised.

**Features.**
- **Plain English:** the numbers you describe each example with. For a pixel, that might be its hue, saturation and brightness.
- **Deeper:** the choice of features often matters more than the choice of model. Q4 used two feature sets:
  - **6 features per pixel:** ALINA's normalised HSV plus CIE Lab.
  - **20 context features:** those 6, plus their 5×5 and 15×15 neighbourhood averages, plus the local spread of brightness. These let a model see "a thin bright stripe" rather than just "a bright pixel".

**Training, testing and leave-one-video-out.**
- **Plain English:** never grade a model on the examples it studied. Here, each model is tested on a video it never saw during training.
- **Deeper:** pixels from the same video are highly correlated (same lighting, same camera). A random train/test split would leak that and inflate scores. Leave-one-video-out (train on two videos, test on the third) measures how well a model copes with new conditions, which is what matters on a real airfield.

**Overfitting.**
- **Plain English:** memorising the training examples instead of learning the general rule.
- **Deeper:** it shows up as a big gap between training and test scores. The usual defences are limiting model complexity (tree depth 3, a small MLP), regularisation (Ridge's penalty, early stopping) and more varied data. Q3's Ridge regression is the clearest overfitting case in this project.

**Class imbalance.**
- **Plain English:** paint is a tiny fraction of the pixels. A model can be 98% "accurate" by calling everything background.
- **Deeper:** this is why the project reports **precision, recall, F1 and IoU** instead of accuracy. It is also why the training sampler draws a fixed number of pixels per class (400 paint, 1,500 background per frame), and why logistic regression uses `class_weight="balanced"`.

**Weak labels.**
- **Plain English:** labels that are mostly right but were not made carefully by a person.
- **Deeper:** the yellow training labels are the authors' published ALINA outputs, so a model trained on them learns to imitate ALINA, including its mistakes. That is why those labels are used only for training. Testing uses the separate CBEM ground truth and hand-traced white polygons.

---

<a id="q3"></a>
## 2. Q3 techniques: finding the ROI

All five methods only have to find the taxiway line. Shared code then builds a trapezoid that makes the line come out big and vertical in the bird's-eye view.

### 2.1 Hough transform (M1: classical computer vision)

**Plain English.** Find the edges in the image. Every edge pixel then "votes" for all the straight lines that could pass through it. Lines that collect many votes are real lines in the picture.

**Deeper.**
- **How it works:** a line is written as \( \rho = x\cos\theta + y\sin\theta \), where \( \rho \) is its distance from the origin and \( \theta \) its angle. Each edge pixel adds one vote to every \( (\rho, \theta) \) cell it could lie on. Peaks in this vote grid (the "accumulator") are the detected lines.
- **The variant used:** OpenCV's probabilistic version, `HoughLinesP`, which returns line segments and is faster.
- **Key settings:** the vote threshold, the minimum segment length and the maximum gap allowed within a segment.
- **Pipeline:** colour pre-filter to keep yellowish pixels, Canny edges, then Hough. Keep steep segments near the centre of the view.
- **Strengths:** no training, very fast (0.04 s), and every result can be checked by eye.
- **Weaknesses:** it assumes straight lines (curves break it), and it needs sensible thresholds.

**What happened.** Best overall method. It was the only one to label anything on the curved `vidd_3`.

### 2.2 K-means clustering (M2, part 1: unsupervised)

**Plain English.** Sort all the pixels into K colour groups, so that each group is as tight as possible. Nobody tells the algorithm what the groups mean.

**Deeper.**
- **How it works:** pick K centres, assign every pixel to its nearest centre, move each centre to the average of its pixels, and repeat until nothing changes. This minimises \( \sum_i \lVert x_i - \mu_{c(i)} \rVert^2 \), the within-cluster squared distance.
- **Settings used:** K = 10 and k-means++ initialisation, which spreads out the starting centres. The "most yellow" cluster is treated as paint.
- **Weaknesses:** you must choose K. It assumes round clusters of similar size, and different random starts can give different answers.

### 2.3 RANSAC line fitting (M2, part 2: robust fitting)

**Plain English.** Draw a straight line through the paint pixels while ignoring stray pixels that don't belong. Repeatedly pick two random points, draw the line through them, and count how many other points lie close to it. Keep the line most points agree with.

**Deeper.**
- **The name:** Random Sample Consensus. Each iteration fits a model to a minimal random subset (two points for a line) and counts the **inliers** within a distance tolerance. The best model is then refitted on its inliers.
- **Why it beats least squares here:** ordinary least squares lets every outlier pull the line. RANSAC simply ignores them.
- **Settings:** the inlier tolerance and the number of iterations.

**What happened.** The combination worked on `vidd_2`. On `vidd_1` it locked onto a second yellow line in the first frame. Colour alone cannot tell which line the aircraft is following.

### 2.4 Vision-language model, GPT-5.5 (M3 and M4: foundation model)

**Plain English.** A very large AI model trained on huge numbers of images and texts. You show it a picture, describe the job in words, and it answers in text.

**Deeper.**
- **How it was used:** each prompt contained the frame with a labelled pixel grid drawn on it and a description of a good ROI. The model replied in JSON. Replies are cached in `results/q3/proposals/`, so results can be reproduced without paying again.
- **M3:** the model gives the four corners directly.
- **M4:** the model only marks points along the line and the aircraft nose; our geometry code builds the trapezoid. This split plays to each side's strength. The model is good at recognising what it sees but imprecise with exact coordinates, while code is exact.
- **Limits:** slow (15–37 s per call), costs money per call, and its replies vary between calls.

**What happened.** M4 beat M3 on every scored video, and matched the manual ROI on `vidd_2` (CBEM F1 91.3 vs. 91.6). The lesson: give a foundation model a narrow job.

### 2.5 Ridge regression (M5: supervised, linear)

**Plain English.** Learn a weighted sum of input numbers that predicts an output number, with a penalty that keeps the weights small so the model doesn't over-react to noise.

**Deeper.**
- **How it works:** it minimises \( \lVert y - Xw \rVert^2 + \alpha \lVert w \rVert^2 \). The \( \alpha \) term is L2 regularisation: larger \( \alpha \) means smaller weights and a smoother model.
- **Inputs and outputs:** the input was a 32×18 grey thumbnail (576 numbers). The outputs were the line's x positions at two fixed image rows, learned from the published labels of the other videos.
- **Why it struggled:** 576 inputs but effectively two training videos. The model learned *where lines usually are in those videos* rather than *how to find a line*. This is textbook overfitting caused by too little varied data.

---

<a id="q4"></a>
## 3. Q4 techniques: finding the paint colours

**The job:** in the bird's-eye view, label each pixel as yellow paint, white paint or background. The original ALINA uses one hand-set rule, normalised HSV between (0, 70, 170) and (255, 255, 255), and has no white rule.

### 3.1 Otsu's method (M1: unsupervised, per frame)

**Plain English.** Look at the histogram of one channel, such as brightness, and pick the cut that best separates it into two groups.

**Deeper.**
- **How it works:** try every threshold \( t \) and keep the one that maximises the **between-class variance** \( \omega_0 \omega_1 (\mu_0 - \mu_1)^2 \). Here \( \omega \) is the share of pixels on each side and \( \mu \) is each side's mean. This is equivalent to minimising the spread within each group.
- **Strength:** no labels, and the threshold adapts to every frame.
- **Weakness:** it *always* returns a cut, even when there is only one material in the frame.

**What happened.** On grey road with no white paint, it split the road into "light" and "dark", giving 30% false white.

### 3.2 Gaussian mixture model, GMM (M2: unsupervised clustering)

**Plain English.** Assume the pixels come from a few colour "blobs" (road, grass, yellow paint, white paint, shadow). Each blob has an average colour and a spread. Let the algorithm find the blobs, then name them: the most yellow blob is yellow paint, and the brightest neutral one is white paint.

**Deeper.**
- **The model:** the data is modelled as \( p(x) = \sum_k \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k) \), a weighted sum of Gaussian blobs.
- **How it is fitted:** with **Expectation-Maximisation**. The E-step computes each pixel's probability of belonging to each blob. The M-step re-estimates each blob's weight, mean and covariance from those probabilities.
- **Compared with K-means:** a GMM allows stretched, tilted clusters and gives soft membership instead of a hard assignment.
- **Colour space:** Lab, because its b* axis separates yellow from blue directly.
- **Weaknesses:** you must pick the number of blobs; like Otsu, it always finds groups; and it is slow, about 1 s per frame.

### 3.3 Decision tree (M3: supervised, readable thresholds)

**Plain English.** A flowchart of yes/no questions learned from examples, such as "saturation above 69?" then "brightness above 140?" then "yellow". Each path from the top to an answer is a box of colour ranges.

**Deeper.**
- **How it is built:** the tree grows greedily. At each node it picks the feature and threshold that most reduce **Gini impurity**, \( 1 - \sum_c p_c^2 \), where \( p_c \) is the share of class \( c \) at that node.
- **Settings:** depth was capped at 3, so the rules stay short and readable. This also limits overfitting.
- **Why it suits ALINA:** each leaf is literally a `cv2.inRange` box, so the learned rule drops straight into ALINA.
- **Weakness:** single trees are unstable (small data changes give different splits), and their boxes are always aligned to the axes.

**What happened.**
- **Yellow:** it rediscovered the hand-set saturation cut (about 70) and lowered the brightness cut (about 115–140).
- **White:** it learned a white rule, which the baseline doesn't have.
- **Mistake:** one box was too broad and caught the aircraft nose on `vidd_2`.

### 3.4 Logistic regression (M4: supervised, linear)

**Plain English.** Give every class a weighted score built from the colour values, and pick the class with the highest score. Unlike a box, the boundary between classes can be slanted.

**Deeper.**
- **How it works:** multinomial logistic regression computes \( z_c = w_c \cdot x + b_c \) for each class \( c \). It turns these scores into probabilities with the **softmax**, \( p_c = e^{z_c} / \sum_j e^{z_j} \), and is trained by minimising cross-entropy.
- **Settings:** inputs are standardised first (see [StandardScaler](#supporting)), and `class_weight="balanced"` counters the class imbalance.
- **Strength:** the learned coefficients show which colour channels matter. They are saved in each run's `method.json`.
- **Weakness:** it can only draw flat boundaries, so it can't express "yellow, unless it is also very bright and pale".

**What happened.** Trained on overcast videos, it called the pale sunlit yellow of `vidd_1` "white".

### 3.5 Multi-layer perceptron, MLP (M5: neural network with context)

**Plain English.** A small neural network: layers of simple units, each combining its inputs with learned weights and passing the result through a bend (the activation function). Stacking layers lets it learn curved, complicated boundaries.

**Deeper.**
- **Architecture:** 20 inputs, then 32 units, then 16 units, then 3 outputs. Hidden units use **ReLU** activations, \( \max(0, z) \), and the output uses softmax.
- **Training:** **backpropagation** with the Adam optimiser. **Early stopping** holds back 10% of the training data and stops when the score on it stops improving, which guards against overfitting.
- **Why the context features matter:** paint is a *thin stripe*. The 5×5 and 15×15 averages and the local brightness spread let the network tell a stripe from a large bright area, such as the aircraft nose or sunlit concrete.
- **Why not a CNN:** a convolutional network learns its own neighbourhood features and would be the natural choice, but PyTorch isn't available on this Intel Mac. The hand-made context features stand in for it.

**What happened.** Best masks overall: yellow F1 0.907, white F1 0.983, and almost no false white (0.01%).

---

<a id="supporting"></a>
## 4. Supporting tools used by several methods

| Tool | Plain English | Why it matters |
|---|---|---|
| **HSV colour space** | Describes a colour as hue (which colour), saturation (how vivid) and value (how bright) | Brightness changes mostly move V, so paint colour is more stable than in RGB. ALINA's own rule is written in HSV |
| **CIE Lab colour space** | L = lightness, a* = green–red, b* = blue–yellow | Yellow paint stands out on b*. Distances roughly match how different colours look to people |
| **StandardScaler** | Rescales each feature to mean 0 and spread 1 | Logistic regression and MLPs train badly when features have very different ranges |
| **Homography (bird's-eye warp)** | Stretches the camera view so the ground looks as if seen from above | Makes the line vertical and evenly wide, which ALINA's column histogram relies on |
| **Canny edge detector** | Finds sharp brightness changes, thinned to 1-pixel lines | Feeds the Hough transform, and the CBEM ground truth is made from Canny edges |
| **Morphological opening** | Erode then dilate: removes specks smaller than the kernel | Tried in the fast experiment to clean the masks |
| **Class weights / balanced sampling** | Make rare classes count as much as common ones | Without them, "everything is background" looks good |

---

<a id="metrics"></a>
## 5. How ALINA scores results, and how that differs from textbook F1

### 5.1 Textbook definitions

- **Precision:** of everything predicted as line, the share that is actually line. \( P = \frac{TP}{TP + FP} \)
- **Recall:** of everything that is actually line, the share that was found. \( R = \frac{TP}{TP + FN} \)
- **F1:** the harmonic mean, \( F_1 = \frac{2PR}{P + R} \). It is high only when both precision and recall are high.
- **IoU (Jaccard):** \( \frac{TP}{TP + FP + FN} \), the overlap divided by the union. It is always ≤ F1 (\( F_1 = \frac{2\,IoU}{1 + IoU} \)).

For a line detector, a TP is normally a predicted pixel that lies on (or within a few pixels of) a true line pixel.

### 5.2 What ALINA does instead (`eval/evaluate.py`, `eval/metrics.py`)

**Plain English.** ALINA does not compare points. It compares the *list of x values* used by the prediction with the list used by the ground truth, and separately the *list of y values*. It then averages the two.

**In detail:**

1. **Coordinates are split into sets of x values and sets of y values.** `calculate_recall(x_gt, x_pred)` turns both lists into Python sets. A predicted x counts as correct if **any** ground-truth pixel has the same x, at any height. The same applies to y.
2. **Duplicates collapse.** A line drawn 5 pixels wide still contributes each x value only once, so thick or repeated detections aren't penalised.
3. **Precision and recall are averaged across the two axes,** \( R = (R_x + R_y)/2 \) and \( P = (P_x + P_y)/2 \). F1 is then computed from those averages, not from separate x and y F1 scores.
4. **Scores are per frame, then averaged over frames** (macro average). A frame with a short line counts as much as a frame with a long one.
5. **Missing frames are skipped, not scored as zero.** If ALINA wrote no file for a frame, `run_evaluation` prints "Skipping" and leaves it out of the average.
6. **Values are percentages** (0–100), not 0–1.

**Why this makes the numbers look better than a point-by-point F1.** A prediction can get every x value and every y value "right" while placing its pixels in the wrong spots. For example, a line drawn at the right height range but mirrored left–right would still score well. Steps 1, 2 and 5 all push scores up, so ALINA's F1 is best read as *"does the detected line cover the same columns and rows as the true line?"*. It is not a measure of exact pixel overlap.

### 5.3 Quick experiment: the paper's validation data under three metrics

`experiments/metric_comparison.py` re-scores the paper's own validation pairs in `data/gt_alina_labels/` (ALINA labels vs. CBEM ground truth, 120 frame pairs).

| Video | Frames | ALINA F1 (x/y sets) | Point F1 (exact pixel) | Point F1 (3 px tolerance) |
|---|---|---|---|---|
| `vidd_1` | 40 | 98.0 | 21.0 | 76.6 |
| `vidd_2` | 40 | 92.8 | 27.7 | 90.0 |
| `vidd_3` | 40 | 94.7 | 37.7 | 95.8 |
| **All** | 120 | **95.2** | **28.8** | **87.4** |

The ALINA column reproduces the official tool exactly: `alina evaluate` on the same folders prints recall 98.44%, precision 92.29%, F1 95.17%.

**How to read it:**
- **Exact-pixel matching is too strict for thin lines** (28.8). CBEM marks the *edges* of the paint, while ALINA marks pixels along its *traversal path*, so they rarely land on the same pixel even when both are right.
- **A 3-pixel tolerance is the fairest comparison** (87.4). It is the usual choice for thin-line benchmarks.
- **ALINA's x/y-set metric is about 8 points more generous on average.** On `vidd_1` the gap is 21 points (98.0 vs. 76.6). On `vidd_3` the two agree.

So ALINA's headline number is honest about *coverage* but generous about *placement*.

**Other things to keep in mind about this ground truth:**
- **Duplicate files:** several CBEM files are byte-identical copies. In `vidd_1`, frames 00002–00007 are identical and 01063 equals 01065. The effective number of distinct test frames is smaller than the file count.
- **Edges, not paint:** CBEM traces the paint's edges, not the filled paint. Q4 filled the gaps between edge pairs to get a paint mask (`fill_edge_pairs` in `experiments/color_methods/common.py`).

### 5.4 How this project reported results

- **Same ALINA metric for comparability.** Q2–Q4 keep ALINA's x/y-set recall, precision and F1 (`experiments/q3_score.py`), so the numbers compare directly with the paper.
- **Unlabeled frames count as zero.** Every run writes a coordinate file for every frame, even when nothing was found, so a frame with no line scores 0 instead of being skipped. This is stricter than the original's step 5.
- **The number of frames labeled is reported separately.** A method can't look good by only answering easy frames.
- **Pixel-level IoU and F1 are added for the masks** (Q4), with standard TP/FP/FN counting inside the usable bird's-eye region.
- **False-white rate:** the percentage of pixels called white on frames that have no white paint. This catches methods that "always find something", such as Otsu.
- **Seeds are fixed** (42) and the training data is cached, so every number can be reproduced.

---

<a id="ideas"></a>
## 6. Ideas for further experiments

These ideas follow from what the Q3 and Q4 results showed. "Tested" means the idea was tried in the fast follow-up in [section 7](#fast-experiment).

### Colour step (Q4)

| Idea | Why it might help | Status |
|---|---|---|
| Random forest on the context features | Averages many trees, so it is more stable than one tree and more flexible than logistic regression | **Tested** |
| Gradient boosting | Often the strongest choice for tabular features | **Tested** |
| Gaussian naive Bayes | Trains almost instantly; checks how much a simple probabilistic model can do | **Tested** |
| Morphological clean-up of the MLP masks | Removes isolated false pixels before the histogram | **Tested** |
| CLAHE lighting normalisation before the hand-set rule | Evens out sun and shadow without any learning | **Tested** |
| Small CNN / U-Net segmentation | Learns its own context features; the natural next step once PyTorch is available | Not tested (no PyTorch) |
| Hysteresis thresholds on model probabilities | Keep weak paint pixels only if they touch strong ones, like Canny; should keep thin stripes connected | Not tested |
| Temporal smoothing across frames | Paint doesn't change between frames, so averaging masks or thresholds over time should cut flicker | Not tested |
| More white labels from `vidd_2`/`vidd_3`, chosen by active learning | White labels come from one stripe in one video, the weakest part of the training data | Not tested |
| Grey-world white balance | Corrects colour casts, which is what fooled logistic regression on sunlit `vidd_1` | Not tested |

### ROI step (Q3)

| Idea | Why it might help | Status |
|---|---|---|
| Polynomial (curved) RANSAC fit | `vidd_3` curves; straight-line models fail there | Not tested |
| Tracking the ROI across frames (Kalman filter) | The line moves smoothly; tracking would replace "find from scratch" every frame and fill gaps | Not tested |
| Hough + GPT-5.5 ensemble | Use Hough when it is confident and ask the model only when it isn't, cutting cost and failures | Not tested |
| Self-training Ridge/CNN on Hough outputs | Gives the supervised method many more (weak) labels than two videos of published labels | Not tested |
| Segment-anything style model (e.g. SAM) prompted with points | A promptable segmenter could outline the line exactly from M4's points | Not tested (no PyTorch) |

### Evaluation

| Idea | Why it helps |
|---|---|
| Report point F1 with a 3 px tolerance alongside ALINA's F1 | Shows placement accuracy, not just coverage ([section 5.3](#metrics)) |
| Remove duplicate CBEM frames before averaging | Avoids counting the same frame several times |
| Bootstrap confidence intervals over frames | With 17 scored frames, small differences between methods may not be real |

---

<a id="fast-experiment"></a>
## 7. Fast follow-up experiment: five more colour methods

A quick experiment (about 3.5 minutes on this laptop's CPU) tested five of the ideas above.

**Setup:**
- **Conditions:** the same mask-quality test as the Q4 table in `results/q4/pixels.csv`. That means the same fixed ROI, the same scored frames (`vidd_1` and `vidd_2`), leave-one-video-out training from the same cached samples, and seed 42.
- **What was run:** only the mask test, not full ALINA labeling, which is why it was fast.
- **Consistency check:** the baseline and the MLP were re-run, and their numbers match `results/q4/pixels.csv` exactly.

Command: `uv run python -m experiments.q4_fast_ideas`. Output: `results/q4_ideas/pixels.csv`.

| Method | Yellow IoU | Yellow F1 | White IoU | White F1 | False white % | Mask time (ms/frame) | Training time (s, 2 folds) |
|---|---|---|---|---|---|---|---|
| m0 baseline (re-run) | 0.793 | 0.884 | 0.000 | 0.000 | 0.00 | 13 | 0 |
| m5 MLP (re-run, best in Q4) | 0.831 | 0.907 | 0.967 | 0.983 | 0.01 | 1009 | 41 |
| i1 random forest | 0.821 | 0.902 | **0.969** | **0.984** | **0.00** | 1525 | 13 |
| i2 gradient boosting | 0.376 | 0.546 | 0.766 | 0.868 | 3.65 | 2005 | 4 |
| i3 Gaussian naive Bayes | 0.779 | 0.876 | 0.965 | 0.982 | **0.00** | 227 | 0.1 |
| i4 MLP + 3×3 opening | **0.832** | **0.908** | 0.967 | 0.983 | 0.01 | 921 | 43 |
| i5 CLAHE + hand-set rule | 0.795 | 0.886 | 0.000 | 0.000 | 0.00 | 25 | 0 |

**What it shows:**

- **Random forest:** matches the MLP on white (F1 0.984, no false white) and the Q4 tree on yellow (0.902). It is slower per frame but trains 3× faster. A solid alternative, not a clear improvement.
- **Naive Bayes:** surprisingly good at white (0.982, no false white), trains in a tenth of a second, and masks a frame 4× faster than the MLP. Its yellow (0.876) is slightly below the hand-set rule, so it is best as a cheap white detector.
- **Gradient boosting (untuned) failed on yellow** (0.546). It used no class weights and default settings, and fit the training videos' lighting too closely. This is a good reminder that "usually strongest on tabular data" depends on tuning and on the test conditions matching the training data.
- **Opening the MLP masks** gains only 0.001 yellow F1. The MLP's masks are already clean, so post-processing has little left to fix.
- **CLAHE** gains only 0.002 over the hand-set rule. Contrast equalisation doesn't fix the real weakness, which is that the rule has no notion of white or of shape.

**Takeaway:** none of the five quick ideas clearly beats the Q4 MLP. The remaining limits are the **labels** (one white stripe, weak yellow labels) and the **ROI** (straight-line assumption on curved `vidd_3`), not the choice of classifier. The most promising next steps are more varied white labels, a curve-aware ROI, and a small CNN once PyTorch is available.

These were mask-only tests, so the end-to-end ALINA scores of the new methods were not measured.

---

<a id="self-test"></a>
## 8. Self-test questions

1. Why does a random train/test split over pixels overestimate how well a colour model works on a new video?
2. Otsu gave 30% false white. What property of Otsu causes that, and which other method in this project shares it?
3. Why can a depth-3 decision tree be pasted into ALINA, but a logistic regression cannot?
4. What do the 15×15 mean features give the MLP that the 6 pixel features don't?
5. ALINA's F1 is 95.2 on its validation data, but exact-pixel F1 is 28.8. Explain both numbers, and say which 3-pixel-tolerance number you would report and why.
6. Why did M4 (GPT points + geometry) beat M3 (GPT corners) when they use the same model?
7. Ridge regression had 576 inputs and two training videos. What went wrong, and what would fix it?
8. Gradient boosting failed in the fast experiment. Name two changes you would try first.

<details>
<summary>Short answers</summary>

1. Pixels from the same video share lighting and camera, so the test pixels look like the training pixels. A new video has new lighting.
2. It always splits the data into two groups, even when only one material is present. GMM shares this: it always finds its K groups.
3. Each leaf of the tree is a box of per-channel ranges, which is exactly what `cv2.inRange` takes. Logistic regression's boundary is slanted across channels.
4. Shape and context: a thin stripe has a different neighbourhood average from a large bright area such as the aircraft nose.
5. ALINA's metric compares sets of x values and sets of y values, so coverage is enough. Exact-pixel matching fails because edges and traversal pixels rarely coincide. Report the 3-pixel-tolerance score (87.4) because it measures placement while allowing for the thickness of the line.
6. The model is good at recognising the line but poor at exact coordinates. M4 asks it only to recognise, and lets code produce the exact corners.
7. Too many inputs for too little varied data, so it memorised where lines are in two videos. More videos, or self-training on Hough outputs, would give it more examples; stronger regularisation would also help.
8. Balanced class weights (or balanced sample weights), and a smaller learning rate with fewer iterations or shallower trees. A different feature set, such as the 6 pixel features, would also be worth trying.

</details>
