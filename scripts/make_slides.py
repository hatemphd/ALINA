"""Build the two slide decks in the repo root:

    Q3_Q4_Techniques_and_Results.pptx   the ML techniques of Q3 (ROI) and Q4 (color threshold) and their results
    ALINA_Explained.pptx                what ALINA is, how its pipeline works, its data and its evaluation

Numbers come from assignment/REPORT.md, results/q4/pixels.csv, results/q4_ideas/pixels.csv and
experiments/metric_comparison.py; figures from results/ and data/. Missing figures are skipped.

Usage (repo root):  uv run --with python-pptx python scripts/make_slides.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(".")
NAVY = RGBColor(0x14, 0x24, 0x3B)
AMBER = RGBColor(0xF2, 0xB1, 0x1B)
INK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xF3, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0xD9, 0xF2, 0xE0)
W, H = Inches(13.333), Inches(7.5)
DRIVE = "https://drive.google.com/file/d/11wKV1DYAJMh3jUcTpf9Wxy_7zr9-00rW/view?usp=drive_link"
REPO = "https://github.com/hatemphd/ALINA"


def first_image(*patterns: str) -> Path | None:
    for pattern in patterns:
        hits = sorted(Path(".").glob(pattern))
        if hits:
            return hits[0]
    return None


class Deck:
    def __init__(self, footer: str):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.footer = footer
        self.count = 0

    # ------------------------------------------------------------------ primitives

    def _slide(self, title: str | None, notes: str = ""):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.count += 1
        if title is not None:
            bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.0))
            bar.fill.solid()
            bar.fill.fore_color.rgb = NAVY
            bar.line.fill.background()
            accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.0), W, Inches(0.06))
            accent.fill.solid()
            accent.fill.fore_color.rgb = AMBER
            accent.line.fill.background()
            self._text(slide, title, Inches(0.5), Inches(0.12), Inches(12.3), Inches(0.8), 28, WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
            self._text(slide, f"{self.footer}   |   {self.count}", Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.35), 11, GREY, align=PP_ALIGN.RIGHT)
        if notes:
            slide.notes_slide.notes_text_frame.text = notes
        return slide

    @staticmethod
    def _runs(paragraph, text: str, size: int, color, bold: bool = False):
        for i, part in enumerate(text.split("**")):
            if not part:
                continue
            run = paragraph.add_run()
            run.text = part
            run.font.size = Pt(size)
            run.font.color.rgb = color
            run.font.bold = bold or i % 2 == 1

    def _text(self, slide, text, left, top, width, height, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
        box = slide.shapes.add_textbox(left, top, width, height)
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        lines = text if isinstance(text, list) else [text]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            self._runs(p, line, size, color, bold)
        return box

    def _bullets(self, slide, items, left, top, width, height, size=18):
        box = slide.shapes.add_textbox(left, top, width, height)
        tf = box.text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            sub = item.startswith("- ")
            text = item[2:] if sub else item
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(6 if not sub else 3)
            self._runs(p, ("      –  " if sub else "•  ") + text, size - 2 if sub else size, GREY if sub else INK)
        return box

    def _image(self, slide, path: Path | None, left, top, max_w, max_h, caption: str = ""):
        if path is None or not path.exists():
            return
        with Image.open(path) as im:
            ratio = im.width / im.height
        w, h = max_w, int(max_w / ratio)
        if h > max_h:
            h, w = max_h, int(max_h * ratio)
        x, y = left + (max_w - w) // 2, top + (max_h - h) // 2
        slide.shapes.add_picture(str(path), x, y, w, h)
        if caption:
            self._text(slide, caption, left, y + h + Inches(0.05), max_w, Inches(0.4), 12, GREY, align=PP_ALIGN.CENTER)

    def _table(self, slide, rows, left, top, width, height, size=13, highlight: set[int] = frozenset(), col_widths=None):
        shape = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width, height)
        table = shape.table
        if col_widths:
            total = sum(col_widths)
            for j, cw in enumerate(col_widths):
                table.columns[j].width = int(width * cw / total)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                cell = table.cell(i, j)
                cell.text = ""
                p = cell.text_frame.paragraphs[0]
                self._runs(p, str(value), size, WHITE if i == 0 else INK, bold=i == 0)
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY if i == 0 else (GREEN if i in highlight else (LIGHT if i % 2 else WHITE))
                cell.margin_left = cell.margin_right = Inches(0.06)
                cell.margin_top = cell.margin_bottom = Inches(0.03)
        return table

    # ------------------------------------------------------------------ slide types

    def title(self, title: str, subtitle: str, byline: str, notes: str = ""):
        slide = self._slide(None, notes)
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
        bg.fill.solid()
        bg.fill.fore_color.rgb = NAVY
        bg.line.fill.background()
        stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(3.55), Inches(2.2), Inches(0.09))
        stripe.fill.solid()
        stripe.fill.fore_color.rgb = AMBER
        stripe.line.fill.background()
        self._text(slide, title, Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.9), 40, WHITE, bold=True, anchor=MSO_ANCHOR.BOTTOM)
        self._text(slide, subtitle, Inches(0.8), Inches(3.8), Inches(11.7), Inches(1.2), 22, AMBER)
        self._text(slide, byline, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.8), 14, WHITE)

    def section(self, title: str, subtitle: str = "", notes: str = ""):
        slide = self._slide(None, notes)
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
        bg.fill.solid()
        bg.fill.fore_color.rgb = NAVY
        bg.line.fill.background()
        self._text(slide, title, Inches(0.8), Inches(2.6), Inches(11.7), Inches(1.2), 40, WHITE, bold=True)
        if subtitle:
            self._text(slide, subtitle, Inches(0.8), Inches(3.8), Inches(11.7), Inches(1.0), 20, AMBER)

    def bullets(self, title: str, items: list[str], notes: str = "", size: int = 20):
        slide = self._slide(title, notes)
        self._bullets(slide, items, Inches(0.7), Inches(1.4), Inches(12.0), Inches(5.5), size)

    def bullets_image(self, title: str, items: list[str], image: Path | None, caption: str = "", notes: str = "", size: int = 18, split: float = 6.3):
        slide = self._slide(title, notes)
        if image is None or not image.exists():
            self._bullets(slide, items, Inches(0.7), Inches(1.4), Inches(12.0), Inches(5.5), size)
            return
        self._bullets(slide, items, Inches(0.5), Inches(1.4), Inches(split), Inches(5.5), size)
        self._image(slide, image, Inches(split + 0.8), Inches(1.4), Inches(12.6 - split - 0.4), Inches(5.0), caption)

    def image(self, title: str, image: Path | None, caption: str = "", notes: str = "", top_text: str = ""):
        slide = self._slide(title, notes)
        top = Inches(1.3)
        if top_text:
            self._text(slide, top_text, Inches(0.7), Inches(1.2), Inches(12.0), Inches(0.6), 16, INK)
            top = Inches(1.85)
        self._image(slide, image, Inches(0.7), top, Inches(12.0), Inches(6.85) - top, caption)

    def table(self, title: str, rows, notes: str = "", takeaways: list[str] | None = None, size: int = 13, highlight=frozenset(), col_widths=None):
        slide = self._slide(title, notes)
        n = len(rows)
        row_h = Inches(0.42 if size >= 13 else 0.36)
        table_h = row_h * n
        self._table(slide, rows, Inches(0.5), Inches(1.35), Inches(12.3), table_h, size, highlight, col_widths)
        if takeaways:
            top = Inches(1.35) + table_h + Inches(0.25)
            self._bullets(slide, takeaways, Inches(0.6), top, Inches(12.1), Inches(6.9) - top, 16)

    def two_columns(self, title: str, left_title: str, left: list[str], right_title: str, right: list[str], notes: str = "", size: int = 16):
        slide = self._slide(title, notes)
        for x, head, items in ((Inches(0.5), left_title, left), (Inches(6.85), right_title, right)):
            card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.35), Inches(6.0), Inches(5.55))
            card.fill.solid()
            card.fill.fore_color.rgb = LIGHT
            card.line.color.rgb = AMBER
            card.adjustments[0] = 0.04
            self._text(slide, head, x + Inches(0.25), Inches(1.45), Inches(5.5), Inches(0.55), 20, NAVY, bold=True)
            self._bullets(slide, items, x + Inches(0.25), Inches(2.05), Inches(5.5), Inches(4.8), size)

    def save(self, name: str):
        OUT.mkdir(exist_ok=True)
        path = OUT / name
        self.prs.save(path)
        print(f"{path}: {self.count} slides")


# ====================================================================== deck 1: Q3 and Q4

def q3_q4_deck():
    d = Deck("Automating ALINA with ML: Q3 and Q4")
    q2_good = Path("results/q2/vidd_2/medium/annotated/00010.jpg")
    overlay = Path("results/q3/overlays/vidd_1_first_frame_rois.jpg")
    hsv = Path("results/q4/figures/hsv_histograms.png")
    f1_bars = Path("results/summary/f1_bars.png")
    time_f1 = Path("results/summary/time_vs_f1.png")
    tree_panel = first_image("results/q4/runs/m3_tree/yellow/vidd_2/masks/0000[2-9].jpg", "results/q4/runs/m3_tree/yellow/vidd_2/masks/*.jpg")
    mlp_panel = first_image("results/q4/runs/m5_mlp/yellow/vidd_1/masks/00002.jpg", "results/q4/runs/m5_mlp/yellow/vidd_1/masks/*.jpg")
    demo_gif = Path("results/demo/alina_best_method_preview.gif")

    d.title(
        "Automating ALINA's Manual Steps\nwith Machine Learning",
        "Q3: choosing the region of interest  ·  Q4: learning the colour threshold\nTechniques and results",
        f"Assignment 4  ·  {REPO}",
        notes="This deck covers questions 3 and 4: replacing ALINA's two hand-set inputs with machine-learning methods, and what the experiments showed.",
    )
    d.bullets_image(
        "The problem: ALINA has two manual inputs",
        [
            "ALINA labels taxiway centerlines with a **classical CV pipeline**; nothing in it learns",
            "**Manual input 1 — where to look:** a person clicks a trapezoid ROI on the first frame",
            "**Manual input 2 — what counts as paint:** a hand-set HSV range (0, 70, 170)–(255, 255, 255)",
            "**Q3** replaces the ROI with ML; **Q4** replaces the colour threshold",
            "ALINA's own code is **never modified**: each method plugs in where the manual step was",
        ],
        Path("assets/method.png"),
        "ALINA pipeline (from the paper)",
        notes="Everything downstream (warp, histogram, CIRCLEDAT traversal, unwarp) is ALINA's own code. Only the two manual inputs are replaced.",
    )
    d.table(
        "Lesson from Q2: the manual ROI is fragile",
        [
            ["Video", "Tight ROI", "Medium ROI", "Loose ROI"],
            ["vidd_1", "47 / 50", "28 / 50", "46 / 50"],
            ["vidd_2", "37 / 50", "38 / 50", "0 / 50"],
            ["vidd_3 (curved)", "0 / 50", "0 / 50 (46 with loosened thresholds)", "0 / 50"],
        ],
        takeaways=[
            "Frames labeled out of 50: ROI size alone swings a video between **47/50 and 0/50**",
            "Every ROI is warped to the same rectangle: a loose ROI makes the line **thin and slanted**, so no column passes the 200-pixel histogram test",
            "**Design rule carried into Q3:** a good ROI makes the line **big and vertical** in the bird's-eye view",
        ],
        col_widths=[2, 2, 4, 2],
        notes="Q2 motivated the Q3 design: the problem is not finding the line, it is framing it so ALINA's column histogram can see it.",
    )

    # ---------------- Q3
    d.section("Q3: Choosing the ROI automatically", "Five methods, one shared trapezoid builder")
    d.bullets(
        "Q3 design: every method only has to find the line",
        [
            "Each method outputs a **line estimate** (slope + intercept); a shared **trapezoid builder** places the ROI",
            "- Line crosses top and bottom edges at the same fraction (0.55) → it comes out **exactly vertical** after the homography",
            "- **Shallow** ROI (≈ 6% of frame height) ending just above the aircraft nose → the line **fills** the warp",
            "**Sanity check + fallback:** rejects near-horizontal lines and collapsed trapezoids (one bad vidd_3 ROI once produced 280,665 'line' pixels)",
            "**Two modes:** Q3a proposes one ROI on frame 1 and reuses it; Q3b proposes a new ROI on every frame",
            "**Evaluation:** ALINA's recall / precision / F1 against CBEM ground truth (vidd_1: 7 frames, vidd_2: 10), frames labeled, ROI time, jitter",
        ],
        notes="Separating 'find the line' from 'build the ROI' is what made the comparison fair: all methods get the same Q2-informed geometry.",
    )
    d.table(
        "Q3: the five ROI methods",
        [
            ["#", "Method", "Family", "Idea in one line", "Why included"],
            ["M1", "Hough transform", "Classical CV", "Yellow mask → Canny edges → line voting", "Traditional lane finding; fast; the yardstick"],
            ["M2", "K-means + RANSAC", "Unsupervised", "10 colour clusters in Lab; robust line fit through the yellow one", "Can colour grouping alone find the paint?"],
            ["M3", "GPT-5.5 corners", "Foundation model", "Ask the model for the 4 ROI corners", "The 'just ask the model' approach"],
            ["M4", "GPT-5.5 points", "Foundation model + geometry", "Model marks 5 centerline points + nose; code builds the ROI", "Same model as M3, narrower job"],
            ["M5", "Ridge regression", "Supervised", "32×18 thumbnail → line x at two rows", "Simplest learned predictor"],
        ],
        size=13,
        col_widths=[0.6, 2, 2, 4.2, 3.5],
    )
    d.two_columns(
        "Q3 techniques (1/2): classical and unsupervised",
        "M1  Hough transform",
        [
            "**Plain English:** every edge pixel votes for the lines it could lie on; lines with many votes are real",
            "Line written as ρ = x cos θ + y sin θ; peaks in the (ρ, θ) vote grid are lines",
            "Probabilistic variant (HoughLinesP) returns segments; keep the steep one nearest the centre",
            "**+** no training, 0.04 s, easy to check",
            "**–** assumes straight lines",
        ],
        "M2  K-means + RANSAC",
        [
            "**Plain English:** sort pixels into 10 colour groups, call the most yellow one paint, then fit a line that ignores strays",
            "K-means minimises within-cluster distance (k-means++ start)",
            "RANSAC: fit to 2 random points, count inliers, keep the best, refit",
            "**+** no labels, no hand-set threshold",
            "**–** colour alone can't tell **which** yellow line the aircraft follows",
        ],
    )
    d.two_columns(
        "Q3 techniques (2/2): foundation model and supervised",
        "M3 / M4  GPT-5.5 vision-language model",
        [
            "**Plain English:** show the frame with a labelled pixel grid, describe a good ROI, read back JSON",
            "**M3:** model gives the 4 corners (whole job)",
            "**M4:** model only points at the line + nose; code builds exact corners",
            "Replies cached in results/q3/proposals/ → reproducible without new API calls",
            "**–** 7–37 s per call, costs money, varies between calls",
        ],
        "M5  Ridge regression",
        [
            "**Plain English:** a weighted sum of thumbnail pixels predicts where the line crosses two rows",
            "Minimises ‖y − Xw‖² + α‖w‖² (L2 penalty keeps weights small)",
            "Trained leave-one-video-out on the published labels",
            "**–** 576 inputs, 2 training videos: it learned **where lines usually are**, not how to find one (overfitting)",
        ],
    )
    d.table(
        "Q3a results: one ROI from the first frame",
        [
            ["Video", "Method", "Frames labeled", "CBEM F1", "Published-label F1", "ROI time (s)"],
            ["vidd_1", "Manual (Q2 medium)", "28", "23.8", "46.5", "–"],
            ["vidd_1", "**M1 Hough**", "**45**", "**70.0**", "**62.3**", "0.05"],
            ["vidd_1", "M4 GPT points", "25", "64.2", "32.5", "27.9"],
            ["vidd_1", "M3 GPT corners", "20", "13.2", "5.8", "28.6"],
            ["vidd_2", "Manual (Q2 medium)", "38", "**91.6**", "76.7", "–"],
            ["vidd_2", "**M4 GPT points**", "39", "91.3", "**77.5**", "7.0"],
            ["vidd_2", "M1 Hough", "39", "83.2", "68.7", "0.03"],
            ["vidd_3", "Manual (Q2 medium)", "0", "–", "0.0", "–"],
            ["vidd_3", "**M1 Hough**", "**16**", "–", "**32.8**", "0.04"],
        ],
        size=12,
        highlight={2, 6, 9},
        takeaways=["vidd_1: **23.8 → 70.0**  ·  vidd_2: matches manual (91.3 vs 91.6)  ·  vidd_3: **0 → 16** frames labeled"],
        notes="M2, M3 and M5 rows for vidd_2 and vidd_3 are in the report. On vidd_3 only Hough labels anything with default thresholds.",
    )
    d.image(
        "Q3: what the methods proposed on vidd_1",
        overlay,
        "First-frame ROIs of all five methods on vidd_1 (results/q3/overlays/)",
        notes="M2 and M3 centred on vidd_1's other yellow line; the CBEM tracing covers only the left line in 5 of 7 frames.",
    )
    d.table(
        "Q3b results: a new ROI on every frame",
        [
            ["Video", "Method", "Frames labeled (first → every)", "CBEM F1 (first → every)", "ROI jitter (px)"],
            ["vidd_1", "M1 Hough", "45 → 47", "70.0 → **71.5**", "41"],
            ["vidd_1", "M2 K-means", "20 → 47", "15.9 → 38.5", "49"],
            ["vidd_1", "M3 GPT corners", "20 → 47", "13.2 → 25.1", "54"],
            ["vidd_1", "M5 Ridge", "43 → 32", "25.0 → 26.5", "112"],
            ["vidd_2", "M1 Hough", "39 → 37", "83.2 → 82.8", "37"],
            ["vidd_2", "M4 GPT points", "39 → 40", "91.3 → 91.0", "34"],
            ["vidd_3", "M1 Hough", "16 → 22", "–", "130"],
            ["vidd_3", "M2 K-means", "0 → 13", "–", "46"],
            ["vidd_3", "M3 GPT corners", "0 → **26**", "–", "209"],
        ],
        size=12,
        takeaways=[
            "**Helps** when the first ROI was wrong (M2, M3 recover) and on **curves** (vidd_3)",
            "**No gain** for a good ROI on a straight taxiway; **amplifies** a weak method (Ridge fell back on 35/50 frames)",
            "For GPT: one paid call (15–36 s) per frame; GPT on curved vidd_3: M3 labels **26** frames, but agrees less with published labels than Hough",
        ],
    )
    d.bullets(
        "Q3 takeaways",
        [
            "**Automating the ROI works:** matches the hand-drawn ROI on vidd_2, triples vidd_1, and gets curved vidd_3 labeled at all",
            "**Geometry mattered more than the model:** the shared 'big and vertical' trapezoid turned Q2's failures into detections",
            "**Simple beat heavy:** cheap Hough lines beat GPT-5.5 overall",
            "**Give a big model a narrow job:** GPT pointing at the line 64.2 vs drawing the box 13.2 (vidd_1)",
            "**Per-frame ROI** is worth it for curves and scene changes, paired with a sanity check",
        ],
    )

    # ---------------- Q4
    d.section("Q4: Learning the colour threshold", "Five ML methods vs. ALINA's hand-set HSV rule")
    d.bullets(
        "Q4 design: swap only the colour step",
        [
            "label_frame = ALINA's process_image with the single cv2.inRange call replaced by **masks(warped) → (yellow, white)**",
            "- With the baseline plugged in, it reproduces ALINA's output **pixel for pixel**",
            "**Fixed ROI per video** (the Q3a Hough ROI), so only the colour step differs between methods",
            "**Training:** leave-one-video-out; yellow labels from the published ALINA output (weak labels)",
            "**White labels created here:** the dataset has none, so one vidd_1 white stripe was traced by hand, CBEM-style (3 train / 5 test frames)",
            "**Two scores:** end-to-end ALINA F1 vs CBEM, and **pixel-level mask IoU / F1** (yellow vs filled CBEM, white vs traced stripe) + false-white rate",
        ],
    )
    d.table(
        "Q4: the methods",
        [
            ["#", "Method", "Family", "Idea in one line", "Why included"],
            ["0", "Hand-set HSV range", "Rule (baseline)", "S ≥ 70 and V ≥ 170 = paint; no white rule", "The step being replaced"],
            ["1", "Otsu per frame", "Unsupervised", "Best 2-group cut of S and V, recomputed every frame", "Simplest data-driven threshold"],
            ["2", "Gaussian mixture", "Unsupervised clustering", "5 colour blobs in Lab; name the yellow and white ones", "Handles several materials"],
            ["3", "Decision tree (depth 3)", "Supervised", "Learned yes/no questions = explicit inRange boxes", "True drop-in replacement, readable"],
            ["4", "Logistic regression", "Supervised, linear", "Weighted colour score per class (slanted boundary)", "Standard simple baseline"],
            ["5", "MLP (32-16) + context", "Neural network", "Pixel colour + 5×5 / 15×15 neighbourhood features", "Sees shape as well as colour; stands in for a CNN"],
        ],
        size=13,
        col_widths=[0.5, 2.3, 2.2, 4.4, 3.3],
    )
    d.two_columns(
        "Q4 techniques (1/2): unsupervised, per frame",
        "Otsu's method",
        [
            "**Plain English:** pick the cut that best splits a histogram into two tight groups",
            "Maximises between-class variance ω₀ω₁(μ₀ − μ₁)²",
            "**+** no labels; adapts to every frame's lighting",
            "**–** always returns a cut, even with one material → **30% false white** on plain road",
        ],
        "Gaussian mixture model (GMM)",
        [
            "**Plain English:** assume a few colour blobs (road, grass, yellow, white, shadow); find them; name them",
            "Fitted with Expectation-Maximisation; soft membership, tilted clusters",
            "Works in Lab (b* axis separates yellow)",
            "**–** always finds its K groups; ≈ 1 s per frame",
        ],
    )
    d.two_columns(
        "Q4 techniques (2/2): supervised",
        "Decision tree  ·  Logistic regression",
        [
            "**Tree:** flowchart of yes/no splits chosen to reduce Gini impurity; depth 3 keeps it readable",
            "- Each leaf is literally a **cv2.inRange box** → pastes into ALINA",
            "**LogReg:** softmax over weighted colour scores; balanced class weights",
            "- Slanted boundary, but still linear → misread sunlit yellow as white",
        ],
        "Neural network (MLP) with context",
        [
            "**Plain English:** layers of simple units learn curved boundaries",
            "20 inputs → 32 → 16 → 3 classes; ReLU, Adam, early stopping",
            "**Context features:** neighbourhood means (5×5, 15×15) + local brightness spread",
            "- Tells a **thin bright stripe** (paint) from a **large bright area** (aircraft nose)",
            "CNN not possible: no PyTorch on this Intel Mac",
        ],
    )
    d.table(
        "Q4 results: mask quality (pixel level)",
        [
            ["Method", "Yellow IoU", "Yellow F1", "White IoU", "White F1", "False white (% of ROI)"],
            ["Baseline HSV", "0.793", "0.884", "0", "0", "0"],
            ["M1 Otsu", "0.790", "0.883", "0.542", "0.703", "30.05"],
            ["M2 GMM", "0.686", "0.814", "0.871", "0.931", "1.63"],
            ["M3 Tree", "0.820", "0.901", "0.923", "0.960", "0.05"],
            ["M4 LogReg", "0.661", "0.796", "0.618", "0.764", "0.34"],
            ["**M5 MLP**", "**0.831**", "**0.907**", "**0.967**", "**0.983**", "**0.01**"],
        ],
        highlight={6},
        takeaways=[
            "Learned methods make **better masks** and **find white paint** (baseline white F1 = 0)",
            "The MLP is best on both colours, with almost no false white",
        ],
    )
    d.table(
        "Q4 results: end to end through ALINA",
        [
            ["Method", "vidd_1 R / P / F1", "vidd_2 R / P / F1", "Frames labeled (v1 / v2 / v3)", "Colour step (ms)"],
            ["**Baseline HSV**", "59.1 / 96.0 / **70.0**", "78.3 / 88.9 / **83.2**", "45 / 39 / 16", "23–28"],
            ["M1 Otsu", "52.6 / 99.4 / 66.2", "78.3 / 83.7 / 80.8", "43 / 29 / 2", "58–83"],
            ["M2 GMM", "52.5 / 99.5 / 66.0", "79.6 / 62.1 / 67.5", "44 / 34 / 19", "1,117–1,740"],
            ["M3 Tree", "54.2 / 99.2 / 67.6", "81.0 / 41.3 / 54.6", "45 / 48 / **35**", "114–132"],
            ["M4 LogReg", "16.0 / 42.9 / 22.1", "79.2 / 81.4 / 80.2", "29 / 41 / **35**", "298–347"],
            ["M5 MLP", "53.9 / 98.5 / 67.0", "81.0 / 82.8 / 81.9", "44 / 39 / 18", "1,380–1,612"],
        ],
        size=13,
        highlight={1},
        takeaways=[
            "**A better mask is not always a better label:** the hand-set yellow rule still wins end to end on vidd_1 / vidd_2",
            "Learned thresholds help most where the rule struggles: dim, curved vidd_3 → **35 vs 16** frames labeled",
        ],
    )
    d.bullets_image(
        "Why a better mask can give a worse label",
        [
            "ALINA keeps **everything connected** to the histogram peak (CIRCLEDAT)",
            "The tree's yellow mask on vidd_2 is good (F1 0.899), but one learned box also covers the **cream aircraft nose**",
            "The nose touches the line → the traversal pulls it in → **precision falls to 41%**",
            "Logistic regression, trained on overcast videos, called the pale **sunlit** yellow of vidd_1 'white' (F1 22.1)",
            "**Lesson:** one wrong blob touching the line costs more than many small errors",
        ],
        tree_panel,
        "Decision tree on vidd_2: bird's-eye | yellow mask | white mask",
        size=17,
    )
    d.bullets_image(
        "What the models learned about colour",
        [
            "**Hue is useless** after ALINA's min-max normalisation: paint and pavement share hue ≈ 25",
            "**Saturation does the work:** every learned rule converges on S ≈ 59–71 (hand-set: 70)",
            "The tree **lowers the brightness cut** to V ≥ 115–140 (hand-set: 170)",
            "It adds a **white rule** the baseline lacks: S ≤ ~66–70, V ≥ ~191–205",
        ],
        hsv,
        "Normalised HSV histograms: paint vs pavement",
        size=17,
    )
    d.image("Q4: the best masks (MLP, vidd_1)", mlp_panel, "MLP on vidd_1 frame 00002: bird's-eye | yellow mask | white mask")

    # ---------------- overall
    d.section("Results across questions", "Speed, best method per question, follow-up experiments")
    d.image(
        "Speed vs accuracy",
        time_f1,
        "",
        top_text="ROI proposal: < 0.4 s classical / supervised, 7–37 s GPT-5.5 · Colour step: 25 ms to 1.7 s · ALINA's pure-Python CIRCLEDAT dominates (1–37 s per frame)",
    )
    d.table(
        "Best method per question",
        [
            ["Question", "Best method", "vidd_1 CBEM F1", "vidd_2 CBEM F1", "vidd_3 frames labeled"],
            ["Q2 manual ROI", "Medium ROI", "23.8", "**91.6**", "0"],
            ["Q3a first-frame ROI", "M1 Hough (v1, v3), M4 GPT points (v2)", "**70.0**", "91.3", "16"],
            ["Q3b every-frame ROI", "M1 Hough", "**71.5**", "82.8", "**22**"],
            ["Q4 colour step", "MLP (masks), tree (explicit thresholds); hand-set rule best end to end", "70.0 / 67.0", "83.2 / 81.9", "**35** (tree)"],
        ],
        size=14,
        col_widths=[2.2, 5, 1.8, 1.8, 2],
    )
    d.image("F1 by question and method", f1_bars, "results/summary/f1_bars.png")
    d.table(
        "Fast follow-up: five more colour ideas (mask quality)",
        [
            ["Method", "Yellow F1", "White F1", "False white %", "Mask ms / frame", "Train s"],
            ["m0 baseline (re-run)", "0.884", "0.000", "0.00", "13", "0"],
            ["m5 MLP (re-run, best in Q4)", "0.907", "0.983", "0.01", "1009", "41"],
            ["Random forest", "0.902", "**0.984**", "**0.00**", "1525", "13"],
            ["Gradient boosting (untuned)", "0.546", "0.868", "3.65", "2005", "4"],
            ["Gaussian naive Bayes", "0.876", "0.982", "**0.00**", "227", "0.1"],
            ["MLP + 3×3 opening", "**0.908**", "0.983", "0.01", "921", "43"],
            ["CLAHE + hand-set rule", "0.886", "0.000", "0.00", "25", "0"],
        ],
        size=13,
        takeaways=[
            "Same frames, ROI, training data and seed as Q4; re-runs match Q4 exactly. **None clearly beats the MLP**",
            "Limits are the **labels** (one white stripe) and the **ROI** (straight lines on curved vidd_3), not the classifier",
        ],
        notes="uv run python -m experiments.q4_fast_ideas — about 3.5 minutes on the laptop CPU. Mask-only test; end-to-end scores not measured.",
    )
    d.table(
        "A note on the metric: ALINA's F1 is generous",
        [
            ["Metric on the paper's 120 validation frames", "F1"],
            ["ALINA formula: sets of x values and sets of y values, averaged", "**95.2**"],
            ["Exact pixel match", "28.8"],
            ["Pixel match within 3 px", "87.4"],
        ],
        size=15,
        col_widths=[8, 2],
        takeaways=[
            "ALINA counts a predicted x as correct if **any** true pixel has that x, at any height; duplicates collapse; missing files are skipped",
            "It measures **coverage** (right rows and columns), not exact placement",
            "This project kept ALINA's metric for comparability, but scored empty frames as **0** and added pixel-level IoU / F1",
        ],
        notes="experiments/metric_comparison.py. The ALINA column reproduces `alina evaluate` (F1 95.17).",
    )
    d.bullets(
        "Conclusions and recommendations",
        [
            "**The ROI matters most, and simple ideas used well beat heavy models**",
            "**Q3 (where to look):** the big gains — vidd_1 23.8 → 70.0, vidd_3 from no labels to 16–22 frames — from cheap Hough lines + Q2's geometry",
            "**Q4 (what counts as paint):** smaller, specific gains — white-line detection, better masks, more coverage on dim footage",
            "**Recommended setup:**",
            "- Hough-based automatic ROI, re-estimated per frame on curvy taxiways",
            "- Hand-set yellow rule + the decision tree's white box",
            "- MLP for footage whose lighting differs from the training data",
        ],
    )
    d.two_columns(
        "Limitations and next steps",
        "Limitations",
        [
            "Only **17** ground-truth frames; **none** for vidd_3",
            "Weak training labels (authors' ALINA output)",
            "White labels from **one** stripe in one video",
            "One run per method (GPT run-to-run variance not measured)",
            "No CNN (no PyTorch); lenient ALINA metric",
        ],
        "Next steps",
        [
            "CBEM ground truth for vidd_3 and more white-paint frames",
            "Train colour models with the aircraft nose as background",
            "Shape check before CIRCLEDAT so blobs touching the line aren't traced",
            "Curved (polynomial) RANSAC and ROI tracking across frames",
            "Small CNN / U-Net once PyTorch is available",
        ],
    )
    d.bullets_image(
        "Resources",
        [
            f"**Code and write-ups:** {REPO}",
            "**Report:** assignment/REPORT.md  ·  **Summary:** assignment/Q3_Q4_Summary.md",
            "**Study guide:** assignment/ML_review_ideation.md",
            f"**Full experiment data (Google Drive):** {DRIVE}",
            "**Demo video:** results/demo/alina_best_method.mp4",
        ],
        demo_gif if demo_gif.exists() else None,
        "Demo: best pipeline vs manual ROI",
        size=16,
    )
    d.save("Q3_Q4_Techniques_and_Results.pptx")


# ====================================================================== deck 2: ALINA explained

def alina_deck():
    d = Deck("ALINA explained")
    labeled = Path("data/Labeled_Data/vidd_1/annotations/00002.jpg")
    raw = first_image("data/Raw_Data/vidd_2/00001.jpg", "data/Raw_Data/vidd_2/*.jpg")
    labeled2 = Path("data/Labeled_Data/vidd_2/annotations/00001.jpg")
    q2_roi = first_image("results/q2/vidd_2/medium/roi_overlay.jpg")
    q2_bird = first_image("results/q2/vidd_2/medium/birds_eye.jpg")
    q2_mask = first_image("results/q2/vidd_2/medium/mask.jpg")
    canny = first_image("data/gt_alina_labels/canny_images/canny_images_1/*.jpg")

    d.title(
        "ALINA\nAdvanced Line Identification and Notation Algorithm",
        "How it labels taxiway centerlines automatically",
        "Khan, Ganeriwala, Bhattacharyya, Neogi, Muthalagu — CVPR 2024 Workshops (arXiv:2406.08775)",
    )
    d.bullets_image(
        "What is ALINA?",
        [
            "An **automatic labeling tool** for airport taxiway video",
            "Finds the **yellow centerline** in each frame and records exactly which pixels belong to it",
            "**Not a neural network:** a classical computer-vision pipeline",
            "Outputs per frame: an **annotated image** (line in red) and a **text file** of x y pixel coordinates",
            "Purpose: produce the **ground truth** that supervised ML models (taxiway / lane detectors) need",
        ],
        labeled,
        "ALINA output on vidd_1: centerline in red",
    )
    d.table(
        "Why it matters",
        [
            ["Problem", "How ALINA addresses it"],
            ["Manual labeling is slow, expensive and error-prone", "Labels automatically at ≈ 19.65 fps (50 ms per frame on a CPU)"],
            ["≈ 33% of aircraft accidents (2015–2022) happened during taxiing", "Creates training data for pilot-assist and autonomous taxi guidance"],
            ["Previous method (CDLEM) needed a new ROI at every scenario change: 120 ROIs", "One ROI per video: 3 ROIs for the same 60,249 frames"],
            ["CDLEM: 91.14% detection at 120 ms per frame", "ALINA: **98.45%** detection at 50 ms per frame (≈ 2.4× faster)"],
        ],
        size=15,
        col_widths=[5, 5],
        takeaways=["Scale: **60,249 frames** from 3 videos of the AssistTaxi dataset (Melbourne MLB and Grant-Valkaria X59 airports), sunny and cloudy"],
    )
    d.image("The pipeline at a glance", Path("assets/method.png"), "Method overview (from the paper)")
    d.bullets(
        "Pipeline overview: six steps per frame",
        [
            "**1. ROI:** a trapezoid in front of the aircraft nose, clicked once on the first frame",
            "**2. Bird's-eye warp:** a homography stretches the trapezoid to a top-down rectangle",
            "**3. Colour:** HSV conversion, min-max normalisation, threshold → binary mask",
            "**4. Column histogram:** count mask pixels per column; the peak column is the line",
            "**5. CIRCLEDAT:** traverse all mask pixels connected to that peak",
            "**6. Unwarp:** map the pixels back to the camera view; draw them in red and save x y",
            "Each step **shrinks the problem** or removes one source of error",
        ],
    )
    d.bullets_image(
        "Step 1–2: ROI and bird's-eye warp",
        [
            "The centerline always appears in roughly the same area in front of the nose",
            "The user clicks **4 points once** per video; reused for all frames",
            "A **homography** (3×3 matrix from 4 source + 4 destination points) warps the trapezoid to the rectangle (50, 100)–(1200, 800)",
            "In the bird's-eye view the line is a **roughly vertical stripe** of constant width",
            "Excludes sky, grass and parked aircraft; cuts computation",
        ],
        q2_bird or q2_roi,
        "Bird's-eye view of a medium ROI on vidd_2",
        size=17,
    )
    d.bullets_image(
        "Step 3: colour normalisation and threshold",
        [
            "Convert BGR → **HSV**: hue (which colour), saturation (how vivid), value (how bright)",
            "**Min-max normalise** each channel to 0–255 so sunny and cloudy frames are comparable",
            "Keep pixels in **(0, 70, 170)–(255, 255, 255)**: vivid and bright → white (255), else black",
            "Bounds chosen from histograms of real line pixels",
            "Why HSV: lighting changes mostly move V, so the paint signature is steadier than in RGB",
            "There is **no rule for white** paint",
        ],
        q2_mask,
        "Binary mask after thresholding",
        size=17,
    )
    d.bullets(
        "Step 4: column histogram — is there a line, and where?",
        [
            "For every column of the mask, **count the white pixels**",
            "A real line in the bird's-eye view is a tall column → a clear **peak**; noise makes only small bumps",
            "Line accepted if the peak column has **> 200 white pixels** (min_white_pixels) and the peak exceeds **peak_pixel_threshold**",
            "- The paper's ablation chose a peak threshold of 150 (0% false positives); the code default is 50",
            "Start point for the next step: the peak column at the **mean row** of its white pixels",
            "If no peak passes, the frame is saved with an **empty** label file",
            "Columns 0–300 of the mask are zeroed first (mask_ignore_left_columns)",
        ],
    )
    d.bullets(
        "Step 5: CIRCLEDAT — collect the whole line",
        [
            "**Circular Threshold Pixel Discovery and Traversal**, a new algorithm in the paper",
            "Depth-first search from the start pixel: from each white pixel, look at every pixel within **±15 px** (circular_threshold)",
            "Jumps small gaps, so it follows **curves and dashed lines**",
            "Collects only pixels **connected** to the line → unconnected noise is left out",
            "Cost **O(k)** in the number of line pixels vs O(m×n) for a sliding window (paper: 3.33 ms vs 10.90 ms)",
            "**Side effect:** anything touching the line (e.g. a bright aircraft nose in the mask) is pulled in too",
        ],
    )
    d.two_columns(
        "Step 6: unwarp and outputs",
        "Unwarp",
        [
            "The inverse homography M⁻¹ projects the collected pixels back to the camera view",
            "Pixels are painted red on the original frame",
            "Only the **single strongest** line is traced per frame",
        ],
        "Outputs per frame",
        [
            "annotations/<frame>.jpg — red overlay",
            "textfiles/<frame>.txt — one 'x y' per line pixel",
            "Empty text file when no line was found",
            "timing log: labeled / no lines found, ms per frame",
        ],
    )
    d.bullets_image(
        "Input and output example",
        [
            "**Input:** a folder of JPG frames per video + 4 ROI clicks",
            "- vidd_1 is 4K (resized to 1080p in this project); vidd_2 and vidd_3 are 1080p",
            "**Output:** the same frame with the centerline in red, plus its coordinates",
            "50 sample frames per video ship with the repo (data/Raw_Data, data/Labeled_Data)",
        ],
        labeled2 if labeled2.exists() else raw,
        "vidd_2 frame labeled by ALINA",
        size=17,
    )
    d.table(
        "The data in the repository",
        [
            ["Folder", "Contents", "Role"],
            ["data/Raw_Data/vidd_1..3", "50 raw frames per video", "Input"],
            ["data/Labeled_Data/vidd_1..3", "annotations/ (red overlays), textfiles/ (x y)", "Authors' sample output"],
            ["data/gt_alina_labels/canny_images", "CBEM edge-map images", "Visual reference for ground truth"],
            ["data/gt_alina_labels/canny_textfiles", "CBEM ground-truth coordinates (120 frames)", "Evaluation truth"],
            ["data/gt_alina_labels/ALINA_textfiles", "ALINA output on the same 120 frames", "What gets evaluated"],
        ],
        size=14,
        col_widths=[4, 5, 3],
        takeaways=["Only **17** of the 120 ground-truth frames have their raw image in the repo (vidd_1: 7, vidd_2: 10, vidd_3: 0)"],
    )
    d.bullets_image(
        "Ground truth: CBEM",
        [
            "**Canny-Based Edge Marking:** how the authors made ground truth",
            "120 frames picked at **scenario changes** (not at random), 40 per video",
            "A person outlines the marking; **Canny edge detection** inside the outline gives the edge pixels",
            "So the truth marks the **edges** of the paint, not the filled paint",
            "Dataset quirk: several CBEM files are byte-identical duplicates",
        ],
        canny,
        "CBEM edge image",
        size=17,
    )
    d.table(
        "How ALINA is evaluated",
        [
            ["Metric on the 120 validation frames", "Value"],
            ["Recall (detection rate) — paper", "98.45%"],
            ["Recall / precision / F1 — reproduced with alina evaluate", "98.44% / 92.29% / **95.17%**"],
            ["Same frames, exact pixel match F1", "28.8"],
            ["Same frames, pixel match within 3 px F1", "87.4"],
        ],
        size=15,
        col_widths=[7, 3],
        takeaways=[
            "ALINA compares the **set of x values** and the **set of y values** separately, then averages: a predicted x counts if any true pixel has it",
            "Per frame, then averaged over frames; frames without an output file are skipped",
            "The authors prioritise **recall**: for taxiway safety a missed marking is worse than an extra pixel",
        ],
    )
    d.table(
        "Key parameters (alina/config.py)",
        [
            ["Parameter", "Default", "Meaning"],
            ["yellow_lower / yellow_upper", "(0, 70, 170) / (255, 255, 255)", "Normalised HSV paint range"],
            ["min_white_pixels", "200", "White pixels needed in the peak column"],
            ["peak_pixel_threshold", "50", "Minimum histogram peak (paper ablation: 150)"],
            ["circular_threshold", "15", "CIRCLEDAT search radius (px)"],
            ["mask_ignore_left_columns", "300", "Mask columns zeroed before the histogram"],
            ["ROI destination rectangle", "(50, 100)–(1200, 800)", "Bird's-eye target of the homography"],
        ],
        size=14,
        col_widths=[3.5, 3.5, 5],
    )
    d.table(
        "Where the time goes (paper, ms per frame)",
        [
            ["Step", "Time (ms)"],
            ["Perspective warp", "4.41"],
            ["HSV normalisation", "5.91"],
            ["Thresholding", "1.05"],
            ["**Histogram**", "**28.71** (bottleneck)"],
            ["CIRCLEDAT", "3.33"],
            ["Remap", "6.68"],
            ["**Total**", "**50.09** (≈ 19.65 fps)"],
        ],
        size=15,
        col_widths=[5, 5],
        takeaways=["In this project's Python version, CIRCLEDAT was far slower (1–37 s per frame) because its cost grows with the connected mask area"],
    )
    d.bullets(
        "Running ALINA",
        [
            "Environment: uv sync  (Python 3.12, OpenCV, NumPy — pinned in uv.lock)",
            "**Label a folder of frames** (opens a window to click the ROI):",
            "- uv run alina label --input-dir data/Raw_Data/vidd_2 --output-images-dir … --output-coords-dir …",
            "**Evaluate** against CBEM:",
            "- uv run alina evaluate --canny-dirs data/gt_alina_labels/canny_textfiles/canny_textfiles_{1,2,3} --alina-dirs data/gt_alina_labels/ALINA_textfiles/ALINA_textfiles_{1,2,3}",
            "**Helpers:** alina video-to-frames, rotate-frames, resize-images, cbem, superimpose",
        ],
        size=18,
    )
    d.two_columns(
        "Strengths and limitations",
        "Strengths",
        [
            "Fast, CPU-only, no training data",
            "One ROI per video instead of one per scene",
            "Follows curves and dashes (CIRCLEDAT)",
            "Every step is inspectable",
        ],
        "Limitations",
        [
            "ROI fixed per video: camera moves or sharp turns push the line out",
            "ROI size is fragile: 47/50 → 0/50 labeled frames (Q2)",
            "Hand-tuned HSV bounds; no white-paint rule",
            "Only one line traced per frame",
            "Lenient x/y-set metric",
        ],
    )
    d.bullets(
        "From ALINA to this assignment",
        [
            "ALINA's two manual inputs are exactly what this assignment automates:",
            "- **Q3:** ML chooses the ROI (Hough, K-means + RANSAC, GPT-5.5, Ridge)",
            "- **Q4:** ML learns the colour threshold (Otsu, GMM, decision tree, logistic regression, MLP)",
            "Headline: automatic ROI raised vidd_1 CBEM F1 from **23.8 to 70.0** and labeled the curved vidd_3 for the first time",
            "Details: the companion deck **Q3_Q4_Techniques_and_Results.pptx** and assignment/REPORT.md",
        ],
    )
    d.bullets(
        "References",
        [
            "Khan, Ganeriwala, Bhattacharyya, Neogi, Muthalagu. ALINA: Advanced Line Identification and Notation Algorithm. CVPR 2024 Workshops. arXiv:2406.08775",
            "Project page: khanhafeez.github.io/alina-project-page",
            "Original code: github.com/hafeezkhan909/ALINA",
            f"This fork with Assignment 4: {REPO}",
        ],
        size=18,
    )
    d.save("ALINA_Explained.pptx")


if __name__ == "__main__":
    q3_q4_deck()
    alina_deck()
