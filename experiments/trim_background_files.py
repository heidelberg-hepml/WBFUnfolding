"""Write trimmed copies of the background files that keep only their uniform bulk.

The background files are not homogeneous in row order:
  ttbar: two sub-samples, the second one (fewer/softer jets) at BOTH file ends -> keep the middle block
  WWjj:  a leading block with shifted |Delta eta_jj| / p_T,j                   -> drop it, keep the rest
In both cases the kept rows are the larger, uniform sample. Reco and truth files of a chain are
row-aligned, so they get the same row range. Boundaries: change points located in the row-order structure
(ttbar sharp to < 50 rows; the WWjj shift is small and smooth).

Usage: WBF_DATA_DIR=/path/to/samples python experiments/trim_background_files.py [out_dir]
(out_dir defaults to $WBF_TRIMMED_DIR, else $WBF_DATA_DIR/trimmed)
"""
import json
import os
import sys

import numpy as np

SRC = os.environ["WBF_DATA_DIR"]
OUT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WBF_TRIMMED_DIR", os.path.join(SRC, "trimmed"))

# (files of one chain, kept row range [lo, hi))
RANGES = [
    (["reco-gen_ttbar_nobjet_nocjv_noolv_rec.npy", "reco-gen_ttbar_nobjet_nocjv_noolv_gen.npy"], (76039, 1608097)),
    (["reco-hard_ttbar_nobjet_nocjv_noolv_rec.npy", "reco-hard_ttbar_nobjet_nocjv_noolv_hard.npy"], (6209, 46136)),
    (["reco-gen_wwjj_rec.npy", "reco-gen_wwjj_gen.npy"], (126909, None)),
    (["reco-hard_wwjj_rec.npy", "reco-hard_wwjj_hard.npy"], (133773, None)),
]

os.makedirs(OUT, exist_ok=True)
summary = {}
for files, (lo, hi) in RANGES:
    n = {np.load(os.path.join(SRC, f), mmap_mode="r").shape[0] for f in files}
    assert len(n) == 1, f"{files}: reco and truth files are not row-aligned ({n})"
    n = n.pop()
    hi_ = n if hi is None else hi
    for f in files:
        a = np.load(os.path.join(SRC, f), mmap_mode="r")
        np.save(os.path.join(OUT, f), np.ascontiguousarray(a[lo:hi_]))
        summary[f] = {"source": os.path.join(SRC, f), "rows": [lo, hi_], "n_source": n, "n_kept": hi_ - lo}
        print(f"{f}: kept rows [{lo}, {hi_}) = {hi_ - lo} of {n} ({(hi_ - lo) / n:.1%})")
json.dump(summary, open(os.path.join(OUT, "trimming.json"), "w"), indent=1)
