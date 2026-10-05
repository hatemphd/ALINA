from . import m0_baseline, m1_otsu, m2_gmm, m3_tree, m4_logreg, m5_mlp

METHODS = {m.NAME: m for m in (m0_baseline, m1_otsu, m2_gmm, m3_tree, m4_logreg, m5_mlp)}
