"""Merge the weight dumps of several k-fold classifier runs on the SAME dataset into one dump with one
out-of-fold weight row per run: signal_approx_weights (n_runs, N). The runs must share cfg.seed (same
p_data events and order) and differ only in train.kfold_seed (fold assignment + member inits); this is
checked row by row. --bkg-yield-norm rescales each run's w = 1 - signal_approx_weights by c_k so that
sum(prior * w_k) = background yield of p_data (background scale taken as known); --bkg-yield-norm-ensemble
uses one common c = background yield / mean_k sum(prior * w_k) for all runs instead, so that the ensemble mean
subtracts the background yield and the runs keep their relative spread. The merged directory is a drop-in classifier dir for make_eval_block.py
and the reweighted unfolding (evaluate.unfold_data_weights), whose member spread is then the out-of-fold classifier spread.

Usage (CPU; from the repo root):
    python experiments/merge_kfold_dumps.py --out results/TransformerClassifier/<name> [--bkg-yield-norm[-ensemble]] RUN_DIR [RUN_DIR ...]
"""
import argparse
import os
import shutil

import numpy as np
import torch

SAME_KEYS = ("is_signal", "hww_index", "mask_wwjj_in_p0", "mask_ttbar_in_p0", "prior_weights", "src_pos_p0")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bkg-yield-norm", action="store_true",
                    help="rescale each run's w so that sum(prior * w) = background yield of p_data")
    ap.add_argument("--bkg-yield-norm-ensemble", action="store_true",
                    help="one common factor for all runs, so that the ensemble mean of sum(prior * w) = background yield")
    args = ap.parse_args()
    assert not (args.bkg_yield_norm and args.bkg_yield_norm_ensemble), "choose one normalization"

    dumps = [torch.load(os.path.join(r, "unfolding_weights_trn.pt"), map_location="cpu", weights_only=False)
             for r in args.runs]
    ref, ref_dir = dumps[0], args.runs[0]
    reco0 = np.load(os.path.join(ref_dir, "p_data_reco.npy"), mmap_mode="r")
    truth_name = "p_data_parton.npy" if ref.get("truth_level") == "parton" else "p_data_gen.npy"
    for r, d in zip(args.runs[1:], dumps[1:]):
        for k in SAME_KEYS:
            assert torch.equal(torch.as_tensor(d[k]), torch.as_tensor(ref[k])), f"{r}: '{k}' differs from {ref_dir}"
        assert ("ttbar_row_perm" in d) == ("ttbar_row_perm" in ref), f"{r}: ttbar shuffle differs"
        if "ttbar_row_perm" in ref:
            assert torch.equal(d["ttbar_row_perm"], ref["ttbar_row_perm"]), f"{r}: ttbar_row_perm differs"
        for k in ("hww_path", "ttbar_path", "wwjj_path", "truth_level", "ttbar_event_weights", "explicit_split"):
            assert d.get(k) == ref.get(k), f"{r}: '{k}' differs ({d.get(k)} vs {ref.get(k)})"
        assert np.array_equal(np.load(os.path.join(r, "p_data_reco.npy"), mmap_mode="r"), reco0), \
            f"{r}: p_data_reco.npy differs"
        print(f"  {r}: same {len(reco0)} events as {ref_dir}")

    merged = dict(ref)
    merged["signal_approx_weights"] = torch.cat([d["signal_approx_weights"] for d in dumps], dim=0)
    if args.bkg_yield_norm or args.bkg_yield_norm_ensemble:
        saw = merged["signal_approx_weights"].double()
        prior = torch.as_tensor(ref["prior_weights"]).double()
        bkg = ~torch.as_tensor(ref["is_signal"]).bool()
        c = prior[bkg].sum() / ((1.0 - saw) * prior).sum(dim=1)
        if args.bkg_yield_norm_ensemble:
            c = torch.full_like(c, (prior[bkg].sum() / ((1.0 - saw) * prior).sum(dim=1).mean()).item())
        merged["signal_approx_weights"] = (1.0 - c[:, None] * (1.0 - saw)).to(dumps[0]["signal_approx_weights"].dtype)
        merged["bkg_yield_norm"] = c
        print(f"  bkg-yield normalization c_k = {c.numpy().round(4)}")
    merged["n_ensemble"] = int(merged["signal_approx_weights"].shape[0])
    merged["merged_from"] = list(args.runs)
    os.makedirs(args.out, exist_ok=True)
    torch.save(merged, os.path.join(args.out, "unfolding_weights_trn.pt"))
    for f in ("p_data_reco.npy", truth_name):
        shutil.copyfile(os.path.join(ref_dir, f), os.path.join(args.out, f))
    print(f"wrote {args.out}: signal_approx_weights {tuple(merged['signal_approx_weights'].shape)}")


if __name__ == "__main__":
    main()
