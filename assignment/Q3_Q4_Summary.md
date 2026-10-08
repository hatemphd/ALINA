# Q3 and Q4 Summary: automating ALINA's two manual steps

[Back to README](../README.md#q3-and-q4-summary) | [Q2 summary](Q2Summary.md) | [Q3 details](Q3_AUTO_ROI.md) | [Q4 details](Q4_COLOR_THRESHOLD.md) | [Methods in plain English](ML_METHODS_EXPLAINED.md)

ALINA has two steps a person sets by hand: **where to look** (the ROI trapezoid) and **what counts as paint** (the HSV color threshold). Q3 replaced the first with ML, and Q4 the second. ALINA's own code was never changed: each new method plugs into the pipeline in place of the manual step.

Scores below use ALINA's metric against the CBEM ground truth (F1 in %). Only `vidd_1` (7 frames) and `vidd_2` (10 frames) have ground truth. For `vidd_3` the tables show how many of its 50 frames got a label.

---

## Q3: choosing the ROI automatically

**What was done.**
- **Five methods:** Hough lines (classical), K-means color clustering (unsupervised), GPT-5.5 drawing the corners, GPT-5.5 pointing at the line, and Ridge regression (supervised).
- **Shared trapezoid builder:** every method only finds the line. One shared builder then places a trapezoid that makes the line big and vertical in the bird's-eye view, the lesson from Q2.
- **Q3a:** each method proposes one ROI on the first frame, reused for all 50 frames.
- **Q3b:** each method proposes a new ROI on every frame.

| Video | Manual ROI (Q2) | Best automated, first frame | Same method, every frame |
|---|---|---|---|
| `vidd_1` | 23.8 | **70.0** (Hough) | 71.5 |
| `vidd_2` | **91.6** | 91.3 (GPT points) | 91.0 |
| `vidd_3` | 0 of 50 frames | **16** (Hough) | **22** |

**Conclusion.**
1. **Automating the ROI works.** It matches the hand-drawn ROI on the easy straight taxiway (`vidd_2`), triples the score on `vidd_1`, and gets the curved `vidd_3` labeled at all.
2. **The geometry mattered more than the model.** The gain came from the shared "line big and vertical" trapezoid. With it, simple Hough lines beat GPT-5.5 overall.
3. **Give a big model a narrow job.** GPT-5.5 did well when asked only to point at the line (64.2 on `vidd_1`), and badly when asked to draw the whole box (13.2).
4. **A new ROI every frame helps only sometimes.** It helps on curves and when the first ROI was wrong. It does nothing for a good ROI on a straight taxiway, and for GPT it costs one paid call per frame.

---

## Q4: learning the color threshold

**What was done.**
- **Five methods** compared with the hand-set rule (saturation ≥ 70, brightness ≥ 170): Otsu (per-frame automatic threshold), a Gaussian mixture (color clustering), a decision tree (learns explicit HSV boxes), logistic regression, and a small neural network (MLP) that also looks at neighboring pixels.
- **Fixed ROI:** every method used the same ROI per video, so only the color step changed.
- **Training:** the supervised models were tested on a video they never trained on.
- **White labels:** the dataset has no white-paint labels, so one white stripe in `vidd_1` was traced by hand, the same way CBEM is made.

| Method | Yellow mask F1 | White mask F1 | CBEM F1 `vidd_1` / `vidd_2` | `vidd_3` frames labeled |
|---|---|---|---|---|
| Hand-set rule | 0.884 | 0 | **70.0** / **83.2** | 16 |
| Decision tree | 0.901 | 0.960 | 67.6 / 54.6 | **35** |
| Neural network (MLP) | **0.907** | **0.983** | 67.0 / 81.9 | 18 |

(All six methods are in the [Q4 details](Q4_COLOR_THRESHOLD.md#results).)

**Conclusion.**
1. **Learned thresholds make better masks and can find white paint,** which the hand-set rule cannot do at all (white F1 0.98 vs. 0). The neural network is best for both colors.
2. **A better mask is not always a better label.** On the two videos with ground truth, the hand-set yellow rule still matches or beats every learned method end to end. ALINA keeps everything connected to the line, so one wrong blob costs more than many small errors: the decision tree called the aircraft nose "yellow", and ALINA's traversal pulled the nose in.
3. **Learned thresholds help most where the hand-set rule struggles.** On the dim, curved `vidd_3`, the decision tree labeled 35 of 50 frames vs. 16.
4. **Lighting is the real challenge.** Hue is useless after ALINA's normalization; saturation separates paint from road. Models trained on overcast videos can misread sunlit paint: logistic regression called pale yellow "white".
5. **Readable thresholds are possible.** The decision tree outputs plain `cv2.inRange` boxes. It rediscovered the hand-set saturation cut (about 70), lowered the brightness cut (about 115–140), and added a white rule (low saturation, high brightness).

---

## Final takeaway

**The ROI matters most, and simple ideas used well beat heavy models.**

- **Choosing where to look gave the big gains:** `vidd_1` 23.8 → 70.0, and `vidd_3` from no labels to 16–22 frames. A cheap classical method did this once Q2's lesson ("keep the line big and vertical") was built into the trapezoid.
- **Choosing what counts as paint gave smaller, more specific gains:** white-line detection, better masks, and more coverage on hard footage. It did not raise accuracy where the hand-set rule already worked.
- **Recommended setup:** Hough-based automatic ROI (re-estimated per frame on curvy taxiways), the hand-set yellow rule, plus the decision tree's white box, with the neural network as the option where the lighting differs from the training footage.
- **What limits these conclusions:** the evidence is small. There are only 17 ground-truth frames, none for `vidd_3`, and white labels from one stripe. More ground truth, especially for `vidd_3`, is the most valuable next step.
