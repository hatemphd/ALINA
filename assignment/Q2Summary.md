# Q2 Summary: manual ROI

[Back to README](../README.md#q2-run-alina-with-a-manual-roi) | [Full execution and analysis](Q2_MANUAL_ROI.md)

The ROI is not a minor setting. Its size alone can take ALINA from labeling most frames to labeling none, even when the colour detection is working.

1. **The warp always fills the same fixed rectangle.** A tight ROI magnifies the paint into thick bands. A loose ROI squeezes it into a thin stripe that may also look slanted.

2. **Detection expects a thick, near-vertical line.** After the warp, ALINA counts mask pixels in each vertical column and needs more than 200 in one column. Thin, slanted or curved lines spread their pixels across many columns and fail that test. This is why `vidd_2` loose and every default `vidd_3` run labeled 0 of 50 frames, even though the mask contained the line.

3. **The best ROI is medium-to-tight, centred on the line and ending just above the aircraft nose:**
   - `vidd_2`: medium gave the cleanest result, 38 of 50 frames.
   - `vidd_1`: tight (47) and loose (46) labeled the most frames, but each frame takes seconds to process.
   - `vidd_3`, a curved taxiway: no ROI size worked with the default settings. Loosening the thresholds labeled 46 frames, but added false detections and took about 12 s per frame.

4. **The takeaway for Q3:** a manual ROI is fragile, and one ROI drawn on the first frame doesn't suit a curving path. That is the case for automating the ROI, and possibly re-estimating it per frame.

In short, a good ROI keeps the line big and vertical in the bird's-eye view, and ALINA fails when the ROI doesn't do that.
