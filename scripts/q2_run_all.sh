#!/usr/bin/env bash
# Q2: walk through the 9 manual-ROI labeling runs (3 videos x 3 ROI sizes).
# For each run: open the guide image, start `alina label`, capture the
# Confirm ROI window with screencapture, wait for labeling, check the outputs.
#
# Usage (from anywhere; the script moves to the repo root):
#   bash scripts/q2_run_all.sh        # start at run 1
#   bash scripts/q2_run_all.sh 4      # resume at run 4

set -u
cd "$(dirname "$0")/.." || exit 1

# video  input-folder  size  guide-colour
RUNS=(
  "vidd_1 outputs/vidd_1_1080p tight red"
  "vidd_1 outputs/vidd_1_1080p medium yellow"
  "vidd_1 outputs/vidd_1_1080p loose green"
  "vidd_2 data/Raw_Data/vidd_2 tight red"
  "vidd_2 data/Raw_Data/vidd_2 medium yellow"
  "vidd_2 data/Raw_Data/vidd_2 loose green"
  "vidd_3 data/Raw_Data/vidd_3 tight red"
  "vidd_3 data/Raw_Data/vidd_3 medium yellow"
  "vidd_3 data/Raw_Data/vidd_3 loose green"
)
TOTAL=${#RUNS[@]}
START=${1:-1}

die() { echo "Error: $*" >&2; exit 1; }
pause() { read -r -p "$1" _; }
ask_yes() { local ans; read -r -p "$1 [y/N] " ans; [[ "$ans" =~ ^[Yy] ]]; }

[[ "$START" =~ ^[1-9]$ ]] && [ "$START" -le "$TOTAL" ] || die "start run must be 1-$TOTAL"
command -v uv >/dev/null || die "uv not found (see UV_ENV_SETUP.md)"
[ -d outputs/vidd_1_1080p ] || die "outputs/vidd_1_1080p is missing; run README Q2 Step 1 first"
if [ ! -f results/q2/roi_guides/vidd_1_guide.jpg ]; then
  echo "Guide images not found; generating them..."
  uv run python scripts/q2_make_roi_guides.py || die "could not generate guides"
fi

LABEL_PID=""
trap 'echo; echo "Interrupted. Resume later with: bash scripts/q2_run_all.sh $RUN_NO"; [ -n "$LABEL_PID" ] && kill "$LABEL_PID" 2>/dev/null; exit 130' INT

for ((RUN_NO = START; RUN_NO <= TOTAL; RUN_NO++)); do
  read -r VIDEO INPUT SIZE COLOR <<<"${RUNS[RUN_NO - 1]}"
  OUT="results/q2/$VIDEO/$SIZE"

  while true; do
    echo
    echo "=================================================================="
    echo " Run $RUN_NO/$TOTAL: $VIDEO, $SIZE ROI   (follow the $COLOR shape)"
    echo " Input: $INPUT    Output: $OUT/"
    echo "=================================================================="

    if [ -f "$OUT/timing.log" ] && ! ask_yes "This run already has results ($(tail -n 1 "$OUT/timing.log")). Redo it?"; then
      echo "Skipping run $RUN_NO."
      break
    fi
    mkdir -p "$OUT"

    # 1. Guide image
    open "results/q2/roi_guides/${VIDEO}_guide.jpg"
    echo
    echo "1. The guide opened in Preview. Follow the $COLOR trapezoid."
    pause "   Press Enter to open the ALINA ROI window..."

    # 2. ALINA in the background, so this script can take the screenshot meanwhile
    uv run alina label --input-dir "$INPUT" \
      --output-images-dir "$OUT/annotated" \
      --output-coords-dir "$OUT/coords" \
      --log-file "$OUT/timing.log" &
    LABEL_PID=$!

    echo
    echo "2. In the 'Polyline' window, click: Bottom-Left, Top-Left, Top-Right, Bottom-Right."
    echo "   Then press any key IN THAT WINDOW. A 'Confirm ROI' window appears."
    echo "   Do NOT press a key in the Confirm ROI window yet."
    echo
    pause "3. Click back on this Terminal and press Enter to take the screenshot..."

    # 3. Screenshot of the Confirm ROI window
    echo "   The pointer is now a camera: click the 'Confirm ROI' window."
    rm -f "$OUT/roi_confirm.png"
    screencapture -o -iW "$OUT/roi_confirm.png"
    if [ -f "$OUT/roi_confirm.png" ]; then
      echo "   Saved $OUT/roi_confirm.png"
    else
      echo "   WARNING: no screenshot saved (cancelled, or Terminal lacks Screen Recording"
      echo "   permission; see README Q2 Step 0). Fallback: Cmd+Shift+4, Space, click the"
      echo "   window, then move the file to $OUT/roi_confirm.png"
    fi

    # 4. Labeling
    echo
    echo "4. Now click the 'Confirm ROI' window and press any key to start labeling."
    echo "   Waiting for ALINA to finish (about a minute)..."
    wait "$LABEL_PID"
    STATUS=$?
    LABEL_PID=""

    # 5. Check outputs
    N_IMG=$(ls "$OUT/annotated" 2>/dev/null | wc -l | tr -d ' ')
    N_TXT=$(ls "$OUT/coords" 2>/dev/null | wc -l | tr -d ' ')
    echo
    echo "5. Check for run $RUN_NO ($VIDEO, $SIZE):"
    echo "   exit status:      $STATUS"
    echo "   roi_confirm.png:  $([ -f "$OUT/roi_confirm.png" ] && echo yes || echo MISSING)"
    echo "   annotated frames: $N_IMG"
    echo "   coord files:      $N_TXT"
    echo "   summary:          $(tail -n 1 "$OUT/timing.log" 2>/dev/null || echo 'no timing.log')"

    if [ "$STATUS" -ne 0 ] || [ "$N_TXT" -eq 0 ]; then
      echo "   This run looks incomplete."
    fi
    ask_yes "   Redo this run (e.g. the trapezoid was wrong)?" || break
  done

  if [ "$RUN_NO" -lt "$TOTAL" ]; then
    read -r NEXT_VIDEO _ NEXT_SIZE _ <<<"${RUNS[RUN_NO]}"
    echo
    pause "Ready for next: run $((RUN_NO + 1))/$TOTAL ($NEXT_VIDEO, $NEXT_SIZE). Press Enter to continue, or Ctrl+C to stop..."
  fi
done

echo
echo "All runs finished. Summary:"
for f in results/q2/*/*/timing.log; do echo "  $f: $(tail -n 1 "$f")"; done
echo
echo "Next: README Q2 Step 4 (vidd_3 re-run with --yellow-lower 0 40 170), then Steps 5-6."
