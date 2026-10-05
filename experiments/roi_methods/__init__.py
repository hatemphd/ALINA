"""Q3 automated ROI methods. Each module exposes NAME and propose(img, seed, video) -> Proposal."""

from . import m1_hough, m2_kmeans, m3_vlm_corners, m4_vlm_points, m5_ridge

METHODS = {m.NAME: m for m in (m1_hough, m2_kmeans, m3_vlm_corners, m4_vlm_points, m5_ridge)}
