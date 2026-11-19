"""Redo a reweighted-unfolding plot on CPU from its saved per-event inputs, optionally adding the
step-1 (ttbar truth-chain reweighting) uncertainty. No unfolding, no GPU.

Inputs in <reweight_dir> (written by the unfolding run, src/unfolding/main.py run_data_unfolding):
    unfolded_data_events.npz        per-event unfolded/truth observables, classifier weights, priors
    observables_unfolded_data.pkl   the plotted pages (labels + bins), used to rebuild the page list
and the classifier dump the run used (--classifier-dir, holding unfolding_weights_trn.pt + eval_block/).

--step1-members: each ttbar event's prior is rescaled by r_m(x)/r_mean(x) for every member m of the
ttbar reweighter (weights_particle_to_parton_members.npy of the step-1 run named in the dump), and the
ttbar total renormalized over the classifier's full trn split exactly as in training. The spread over
members enters the band in quadrature (classifier weights held fixed: the approximation).

Usage (from the repo root):
    python experiments/replot_unfolded_data.py <reweight_dir> --classifier-dir <clf_dir> [--step1-members]
        [--out PDF] [--n-bootstrap 200] [--weights-from-classifier]
"""
import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def wbf_observables(unfolder_config):
    """The unfolder's full observable list (same order as the npz rows), without loading data."""
    from types import SimpleNamespace
    from datasets.WBF import WBF

    cfg = yaml.safe_load(open(unfolder_config))["dataset"]
    stub = WBF.__new__(WBF)
    p = cfg["parameterization"]
    stub.parameterization = SimpleNamespace(
        naive=SimpleNamespace(use=p["naive"]["use"]),
        transverse_mass=SimpleNamespace(use=p["transverse_mass"]["use"]))
    stub.unfold_to = cfg["unfold_to"]
    WBF.init_observables(stub)
    return stub.observables


def step1_prior_variations(clf_dir, prior, eval_dump):
    """(M, N_eval) prior weights, one per step-1 member (non-ttbar rows unchanged)."""
    full = torch.load(os.path.join(clf_dir, "unfolding_weights_trn.pt"), map_location="cpu", weights_only=False)
    step1_run = os.path.dirname(full["ttbar_event_weights"])
    r_mean = np.load(full["ttbar_event_weights"])
    r_members = np.load(os.path.join(step1_run, "weights_particle_to_parton_members.npy"))  # (M, N_ttbar)
    lo = int(float(full["explicit_split"]["trn"][0]) * len(r_mean))
    tt_full = full["mask_ttbar_in_p0"].numpy().astype(bool)
    rows_full = lo + full["src_pos_p0"].numpy()[tt_full]          # file rows of all trn-split ttbar
    tt = eval_dump["mask_ttbar_in_p0"].numpy().astype(bool)
    rows = lo + eval_dump["src_pos_p0"].numpy()[tt]               # file rows of the eval-block ttbar
    if "ttbar_row_perm" in full:  # ttbar shuffled before splitting: positions -> file rows
        perm = full["ttbar_row_perm"].numpy()
        rows_full, rows = perm[rows_full], perm[rows]
    # the dumped priors must be alpha * r_mean / sum_trn(r_mean) on the ttbar rows
    c = prior[tt] / r_mean[rows]
    assert np.allclose(c, c[0], rtol=1e-4), "ttbar priors are not proportional to the step-1 weights"
    out = []
    for r_m in r_members:
        pv = prior.copy()
        pv[tt] = prior[tt] * (r_m[rows] / r_mean[rows]) * (r_mean[rows_full].sum() / r_m[rows_full].sum())
        out.append(pv)
    print(f"  step-1 variations: {len(out)} members of {step1_run} on {tt.sum()} ttbar events")
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reweight_dir")
    ap.add_argument("--classifier-dir", required=True)
    ap.add_argument("--step1-members", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--weights-from-classifier", action="store_true",
                    help="take the classifier weights from <classifier-dir>/eval_block instead of the npz "
                         "(e.g. a recalibrated dump of the same events)")
    args = ap.parse_args()

    import unfolding.plots as P
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("replot-unfolded")

    z = np.load(os.path.join(args.reweight_dir, "unfolded_data_events.npz"))
    pages = pickle.load(open(os.path.join(args.reweight_dir, "observables_unfolded_data.pkl"), "rb"))
    all_obs = wbf_observables(os.path.join(args.reweight_dir, "config.yaml"))
    assert len(all_obs) == z["obs_unf"].shape[0], (len(all_obs), z["obs_unf"].shape)
    channels = {e["obs"].channel for e in pages if e["obs"].channel is not None}
    keep = [i for i, o in enumerate(all_obs) if o.channel is None or o.channel in channels]
    assert [all_obs[i].tex_label for i in keep] == [e["obs"].tex_label for e in pages], "page list mismatch"

    prior, is_sig = z["prior_weights"], z["is_signal"]
    eval_dump = torch.load(os.path.join(args.classifier_dir, "eval_block", "unfolding_weights_trn.pt"),
                           map_location="cpu", weights_only=False)
    assert np.allclose(eval_dump["prior_weights"].numpy(), prior), "npz does not match the classifier eval block"
    assert np.array_equal(eval_dump["is_signal"].numpy().astype(bool), is_sig), "signal flags differ"
    weights = z["weights"]
    if args.weights_from_classifier:
        weights = eval_dump["signal_approx_weights"].numpy()
        print(f"  classifier weights from {args.classifier_dir}/eval_block: {weights.shape}")
    variations = step1_prior_variations(args.classifier_dir, prior, eval_dump) if args.step1_members else None

    plots = P.Plots.__new__(P.Plots)
    plots.logger, plots.channels, plots.title = logger, sorted(channels), None
    plots.model_name, plots.truth_name = pages[0].get("model_name", "TraCFM"), pages[0].get("truth_name", "Parton")
    out = args.out or os.path.join(args.reweight_dir, "observables_unfolded_data"
                                   + ("_step1unc" if args.step1_members else "_replot") + ".pdf")
    plots.plot_unfolded_data(
        file=out, obs_unf_data=[z["obs_unf"][i] for i in keep], obs_part_data=[z["obs_part"][i] for i in keep],
        bins=[e["bins"] for e in pages], observables=[all_obs[i] for i in keep], weights=weights,
        prior_weights=prior, is_signal=is_sig, n_bootstrap=args.n_bootstrap, prior_variations=variations,
        pickle_file=out.replace(".pdf", ".pkl"),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
