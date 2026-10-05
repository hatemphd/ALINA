# ML techniques in plain English: what each one does, its purpose, and why it was chosen

[Back to README](../README.md#report) | [Q3 details](Q3_AUTO_ROI.md) | [Q4 details](Q4_COLOR_THRESHOLD.md)

This page explains every technique used in Q3 and Q4 without the math. The technical details and the numbers are in the Q3 and Q4 write-ups. For a longer study guide, covering how each algorithm works, how ALINA's F1 differs from a textbook F1, and ideas for further experiments, see [ML_review_ideation.md](ML_review_ideation.md).

---

## Why this mix of methods

The brief asks for at least 5 methods per question and rewards both creativity and rigor. The methods cover the main families of machine learning, so the comparison shows *which kind* of learning helps, not just which tool:

- **Rules and classical computer vision** (no learning): the baseline to beat.
- **Unsupervised learning** (no labels): finds structure in each frame by itself.
- **Supervised learning** (learns from labeled examples): from simple and readable to more flexible.
- **Neural networks:** the most flexible learner that runs on this laptop.
- **Foundation models** (Q3 only): a large pretrained vision-language model, OpenAI GPT-5.5.

Two practical limits shaped the choices:
- No PyTorch on this Intel Mac, so neural networks are small scikit-learn models.
- The OpenAI credits ran out during Q3, so Q4 uses only methods that run on a laptop CPU.

At least one method per question gives readable output a person can check: Hough lines in Q3, decision-tree boxes in Q4.

---

## Q3: finding the region of interest (ROI) automatically

**The job:** look at a camera frame, find the taxiway line the aircraft is following, and place ALINA's trapezoid on it.

All five methods share one rule learned in Q2: the trapezoid must make the line come out **big and vertical** in the bird's-eye view. In practice:
- the line crosses the top and bottom edges at the same position;
- the trapezoid is shallow;
- it ends just above the aircraft nose.

Each method only has to find the line; shared code builds the trapezoid from it.

### M1: Hough transform (classical computer vision)
- **In plain English:** keep the yellow-looking pixels and find their edges. Every edge pixel "votes" for the straight lines it could belong to, and lines with many votes are real. Keep the steep line closest to the centre of the view.
- **Purpose:** find the line using geometry, with no training.
- **Why chosen:** this is how lane finding has traditionally been done. It is fast (0.04 s), needs no data, and is the yardstick for the ML methods.
- **Outcome:** best overall. It is best on `vidd_1` and the only method that labels anything on the curved `vidd_3`.

### M2: K-means clustering + RANSAC (unsupervised)
- **In plain English:** sort the pavement pixels into 10 colour groups without being told what any of them are, and call the most yellow group "paint". Then draw the best straight line through the paint pixels; RANSAC ignores stray pixels that don't fit.
- **Purpose:** find the line by colour alone, with no labels and no hand-set threshold.
- **Why chosen:** it tests whether letting the data form its own colour groups is enough. RANSAC is the standard way to fit a line when some points are noise.
- **Outcome:** fine on `vidd_2`. On `vidd_1` it locked onto the other yellow line in the first frame.

### M3: GPT-5.5 draws the corners (foundation model, end to end)
- **In plain English:** show the frame to GPT-5.5 with a labelled pixel grid, explain what a good ROI looks like, and ask for the four corners.
- **Purpose:** test whether a large vision-language model can do the whole job in one step.
- **Why chosen:** it is the state-of-the-art "just ask the model" approach the brief suggests.
- **Outcome:** it recognises the scene but is imprecise with exact coordinates. It is also slow (15–37 s per call) and each call costs money.

### M4: GPT-5.5 points at the line (foundation model + geometry)
- **In plain English:** ask GPT-5.5 only to mark 5 points along the centre line and the top of the aircraft nose. Our code turns those points into the trapezoid.
- **Purpose:** let the model do what it is good at (recognising the line) and let code do what it is good at (exact corners).
- **Why chosen:** M3 vs. M4 is a controlled test: same model, different split of the job.
- **Outcome:** better than M3 on every scored video. It matches the manual ROI on `vidd_2` (CBEM F1 91.3 vs. 91.6).

### M5: Ridge regression (supervised)
- **In plain English:** shrink the frame to a tiny 32×18 thumbnail. From the authors' published labels on the other videos, learn where the line crosses two fixed rows of the image.
- **Purpose:** learn line positions from past examples.
- **Why chosen:** the simplest supervised method; it checks whether a small labeled dataset is enough.
- **Outcome:** with only two training videos, it learned *where lines usually are* in those videos rather than *how to find a line*. It often fell back to the default ROI.

**Running M3 and M4** needs an OpenAI API key, except when reusing the cached replies. See [Providing the OpenAI API key](../README.md#providing-the-openai-api-key-gpt-methods-m3-and-m4-only).

---

## Q4: finding the colour thresholds for line paint

**The job:** in the bird's-eye view, label every pixel as yellow paint, white paint, or background. The original ALINA uses a hand-picked rule (saturation ≥ 70 and brightness ≥ 170) and has no rule for white.

### Method 0: hand-set rule (baseline, not ML)
- **In plain English:** "bright and colourful = paint", with numbers a human typed in.
- **Purpose:** the thing being replaced; every method is compared to it.
- **Why kept:** without it there is no way to tell whether ML helped.

### Method 1: Otsu's method (unsupervised, per frame)
- **In plain English:** for each frame, look at all pixel brightness (or saturation) values. Pick the cut that best splits them into two tight groups, such as "dull road" and "vivid paint".
- **Purpose:** choose the thresholds fresh for every frame instead of fixing them once.
- **Why chosen:** the simplest data-driven replacement for a hand-picked number. It needs no labels and should adapt to lighting.
- **Weakness found:** it always splits the frame in two, even when there is no paint. It cut grey road into "light" and "dark", giving 30% false white.

### Method 2: Gaussian mixture model (unsupervised clustering, per frame)
- **In plain English:** assume the pixels come from a few colour groups (road, grass, yellow paint, white paint, shadow) and let the model find them. Each group is described by an average colour and a spread. Then name the groups: the most yellow is yellow paint, and the brightest neutral one is white paint.
- **Purpose:** find paint colours without labels and without forcing a single cut on one channel.
- **Why chosen:** the standard way to let the data form its own colour groups when there are more than two materials. It works in Lab colour, which has a dedicated yellow–blue axis.
- **Weakness found:** it always finds groups, even when one is just a yellowish patch of pavement. It is also slow, about 1 s per frame.

### Method 3: Decision tree (supervised, explicit thresholds)
- **In plain English:** learn a short list of yes/no questions from labeled pixels, for example "Is saturation above 69? Is brightness above 140? Then it is yellow." It is limited to 3 questions deep. Each answer path is a box of colour ranges, the same kind of rule ALINA already uses.
- **Purpose:** learn the thresholds from examples and output them as plain numbers that can be pasted into ALINA.
- **Why chosen:** it is the only learned method whose result is literally a set of `cv2.inRange` bounds. That makes it a true drop-in replacement and easy to explain.
- **Outcome:** it rediscovered the hand-set saturation cut (about 70), lowered the brightness cut (about 115–140), and learned a white rule the baseline lacks. One box was too broad and caught the aircraft nose on `vidd_2`.

### Method 4: Logistic regression (supervised, linear)
- **In plain English:** learn a weighted score from the colour values ("2 × yellowness − 1 × lightness + …") and pick the class with the highest score. The boundary can be slanted, not just an "above X, below Y" box.
- **Purpose:** test whether a slightly more flexible rule that combines colour spaces beats boxes.
- **Why chosen:** the standard simple supervised baseline: fast, stable, well understood.
- **Weakness found:** too rigid for lighting changes. Trained on the overcast videos, it called the pale sunlit yellow of `vidd_1` "white".

### Method 5: Small neural network with neighbourhood features (MLP)
- **In plain English:** a small neural network with two layers of 32 and 16 units. It sees each pixel's colour plus a summary of the area around it: average colour nearby at two sizes, and how much the brightness varies locally.
- **Purpose:** use shape as well as colour. A thin bright stripe is paint; a large bright area is the aircraft nose or sunlit concrete.
- **Why chosen:** the brief invites deep learning. A convolutional network (CNN) would be the usual choice, but it needs PyTorch, which isn't available here. The neighbourhood features give a small network the same ability to look at the surroundings.
- **Outcome:** the best masks for both yellow (F1 0.907) and white (F1 0.983), with almost no false white (0.01%).

### How the supervised methods (3–5) were trained
- **Yellow examples:** pixels from the authors' published ALINA labels.
- **White examples:** a white stripe traced by hand in `vidd_1`; the dataset has no white labels.
- **Background examples:** pixels well away from both.
- **Testing:** each model is tested on a video it never trained on ("leave-one-video-out"), which shows honestly how it copes with new lighting.

Q4 needs no API key.

---

## One-line takeaway

The best results came from simple ideas combined well. For the ROI (Q3), classical geometry worked best. For colour (Q4), the best options are readable learned thresholds (the tree) or a small context-aware network (the MLP). The big pretrained model (GPT-5.5) helped most when given a narrow job, pointing at the line, rather than the whole task.
