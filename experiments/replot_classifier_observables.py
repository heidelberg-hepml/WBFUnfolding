#!/usr/bin/env python
"""Replot a classifier run's ``observables_{trn,tst}.pdf`` without re-evaluating any model.

Sources (``--source``; ``auto`` picks the first available, preferring the CURRENT observable list):

1. ``plot_state`` -- ``<run_dir>/plot_state.pt`` (runs made after it was added). Rebuilt with the
   same ``BackgroundSubtractionPlots.plot_observables``; if the state holds the raw events, the
   observables are recomputed from today's ``init_observables`` (new observables show up).
2. ``dump`` -- ``unfolding_weights_trn.pt`` + ``p_data_reco.npy`` (``trn`` only). The clean-view page is
   rebuilt with today's observables. ``p_bkg^MC`` is rebuilt exactly: within the training pool it is
   the background rows NOT in ``p_data`` (``src_pos_p0``), so the page is identical to the original.
   ``--bkg-ref in-data`` uses the background inside ``p_data`` instead; ``both`` = reference check.
3. ``pickle`` -- ``observables_<split>.pkl``: the saved pages as they were (no new observables).

Fast layout checks: ``--n_class 1`` uses only the first ensemble member, ``--n-events N`` (dump
source) a random subset of N p_data events, ``--obs`` a single observable.

``plots.py`` is reloaded on every call, so style edits there show up without restarting a notebook.

Usage (from the repo root):
    python experiments/replot_classifier_observables.py <run_dir> [--split trn|tst] [--out PDF]
        [--obs SUBSTRING] [--source auto|plot_state|dump|pickle] [--n_class K] [--n-events N]
        [--bkg-ref independent|in-data|both] [--bkg-yield-norm]

Or from a notebook in this folder:
    from replot_classifier_observables import replot
    replot("../../results/TransformerClassifier/<run>", obs_filter="m_{T", n_class=1)
"""

import argparse
import importlib
import os
import pickle
import zipfile
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "src"), os.path.join(_HERE, "..")):  # experiments/, src/ and repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

import unfolding.plots as uplots  # noqa: E402
import classification.plots as cplots  # noqa: E402
from plot_reweighted_pdata import _get_observables, _to_ptphietam  # noqa: E402


def _independent_bkg(d, yields, frac=1.0, rng=None, max_njets=None):
    """Background MC rows of the dump's pool that are NOT in p_data: a stand-in for p_bkg^MC.

    ``src_pos_p0`` indexes the concatenated [ttbar pool, WWjj pool], each pool being the file rows
    ``[int(lo*N), int(hi*N))`` of the explicit split. Each process (ttbar with its chain-reweighting weights) is rescaled to ``yields[mask_key]``
    (its plotted p_data yield); ``frac < 1`` keeps a random fraction of the rows (fast checks);
    ``max_njets`` drops rows with more reco jets (raw column 20), as in p_data.
    Returns (PtPhiEtaM tensor, per-event weights).
    """
    lo_frac, hi_frac = d["explicit_split"]["trn"]
    src = np.asarray(d["src_pos_p0"])
    parts, weights, offset = [], [], 0
    for mask_key, path_key in (("mask_ttbar_in_p0", "ttbar_path"), ("mask_wwjj_in_p0", "wwjj_path")):
        m = np.asarray(d[mask_key], dtype=bool)
        a = np.load(d[path_key], mmap_mode="r")
        # per-event weights (ttbar chain reweighting, file order), else uniform
        ew = np.load(d["ttbar_event_weights"]) if path_key == "ttbar_path" and d.get("ttbar_event_weights") \
            else np.ones(len(a))
        if path_key == "ttbar_path" and "ttbar_row_perm" in d:  # rows shuffled before the split
            perm = np.asarray(d["ttbar_row_perm"])
            a, ew = np.asarray(a)[perm], ew[perm]
        lo, hi = int(lo_frac * len(a)), int(hi_frac * len(a))
        keep = np.ones(hi - lo, dtype=bool)
        keep[src[m] - offset] = False
        offset += hi - lo
        b, wb = np.asarray(a[lo:hi][keep]), np.asarray(ew[lo:hi][keep], dtype=float)
        if max_njets is not None:
            ok = b[:, 20] <= max_njets
            b, wb = b[ok], wb[ok]
        if frac < 1.0:
            sub = rng.random(len(b)) < frac
            b, wb = b[sub], wb[sub]
        parts.append(b)
        weights.append(wb * yields[mask_key] / wb.sum())
    return _to_ptphietam(np.concatenate(parts)), np.concatenate(weights)


def _lines_from_dump(run_dir, bkg_ref="independent", n_class=None, n_events=None, obs_filter=None,
                     bkg_yield_norm=False, max_njets=None):
    """Yield (lines, bins, obs) per observable, mirroring plot_observables' clean view (trn split).

    ``bkg_ref``: "independent" uses the pool's background MC rows not in p_data as p_bkg^MC (like
    the original, an independent sample); "in-data" uses the background inside p_data itself;
    "both" is the reference check: w*p_data against both backgrounds, one ratio line each.
    ``bkg_yield_norm``: rescale each member's w by c_k so that sum(prior * w_k) = background yield of p_data;
    "ensemble": one common c for all members, so that the ensemble MEAN subtracts the background yield.
    ``max_njets``: keep only events with at most this many reco jets (p_data and p_bkg^MC), before the
    normalization.
    """
    d = torch.load(os.path.join(run_dir, "unfolding_weights_trn.pt"), map_location="cpu")
    saw = np.asarray(d["signal_approx_weights"], dtype=float)  # (K, N) = 1 - w
    if saw.ndim == 1:
        saw = saw[None, :]
    saw = saw[:n_class]                                         # first n_class members (None = all)
    prior = np.asarray(d["prior_weights"], dtype=float)
    is_sig = np.asarray(d["is_signal"], dtype=bool)
    raw = np.load(os.path.join(run_dir, "p_data_reco.npy"), mmap_mode="r")
    if raw.shape[0] != saw.shape[1]:
        raise ValueError(f"p_data_reco.npy has {raw.shape[0]} rows but the dump has {saw.shape[1]}")
    sel = np.arange(raw.shape[0])
    if max_njets is not None:
        sel = np.nonzero(np.asarray(raw[:, 20]) <= max_njets)[0]
        print(f"  N_jets <= {max_njets}: {len(sel)} / {raw.shape[0]} p_data events, signal yield kept "
              f"{prior[sel][is_sig[sel]].sum() / prior[is_sig].sum():.4f}")
    if bkg_yield_norm:  # on the selected p_data, before any --n-events subset
        c = prior[sel][~is_sig[sel]].sum() / (prior[sel] * (1.0 - saw[:, sel])).sum(axis=1)
        if bkg_yield_norm == "ensemble":
            c = np.full_like(c, prior[sel][~is_sig[sel]].sum() / (prior[sel] * (1.0 - saw[:, sel])).sum(axis=1).mean())
        print(f"  bkg-yield normalization c_k = {np.round(c, 4)}")
        saw = 1.0 - c[:, None] * (1.0 - saw)
    frac, rng = 1.0, np.random.default_rng(0)
    if n_events is not None and n_events < len(sel):
        # independent bkg MC is thinned by the same fraction so yields stay comparable
        frac = n_events / len(sel)
        sel = np.sort(rng.choice(sel, n_events, replace=False))
    saw, prior, is_sig, raw = saw[:, sel], prior[sel], is_sig[sel], raw[sel]
    w = 1.0 - saw                                               # (K, N) per-member w
    N = saw.shape[1]
    norm = N / prior.sum()  # same norm_factor_p0 as plot_observables
    # per-process yield of the plotted p_data (the bkg MC stand-in is rescaled to it)
    yields = {k: prior[np.asarray(d[k], dtype=bool)[sel]].sum() * norm
              for k in ("mask_ttbar_in_p0", "mask_wwjj_in_p0")}

    p = _to_ptphietam(np.asarray(raw))
    if bkg_ref in ("independent", "both"):
        p_ind, w_ind = _independent_bkg(d, yields, frac=frac, rng=rng, max_njets=max_njets)
        bkg_label = r"$p_{\text{bkg}}^{\text{MC}}$"
    else:
        bkg_label = r"$\mathrm{bkg} \in p_{\text{data}}$"

    for obs in _get_observables():
        if obs_filter and obs_filter not in obs.tex_label:
            continue
        x = np.asarray(obs.compute(p))
        bins = np.asarray(obs.bins(x))
        hist = lambda sel, wts: cplots.compute_hist_data(bins, x[..., sel], weights=wts)
        y_p0, e_p0 = hist(slice(None), prior * norm)
        if bkg_ref in ("independent", "both"):
            x_ind = np.asarray(obs.compute(p_ind))
            y_ind, e_ind = cplots.compute_hist_data(bins, x_ind, weights=w_ind)
        if bkg_ref in ("in-data", "both"):
            y_dat, e_dat = hist(~is_sig, prior[~is_sig] * norm)
        y_bkg, e_bkg = (y_ind, e_ind) if bkg_ref == "independent" else (y_dat, e_dat)
        y_rew, e_rew = hist(slice(None), prior[None, :] * w * norm)
        y_sig, e_sig = hist(is_sig, prior[is_sig] * norm)
        y_rew2, e_rew2 = hist(slice(None), prior[None, :] * saw * norm)
        L = cplots.Line
        if bkg_ref == "both":
            # same w*p_data twice: identical in the main panel, one ratio line per reference
            lines = [
                L(y=y_p0, y_err=e_p0, label=r"$p_{\text{data}}$", color=cplots.P0_COLOR),
                L(y=y_dat, y_err=e_dat, label=r"$\mathrm{bkg} \in p_{\text{data}}$",
                  color=cplots.P1_COLOR),
                L(y=y_ind, y_err=e_ind, label=r"$p_{\text{bkg}}^{\text{MC}}$",
                  color=cplots.BKG_MC_COLOR),
                L(y=y_rew, y_err=e_rew, y_ref=y_dat,
                  label=r"$w \cdot p_{\text{data}}\ /\ \mathrm{bkg} \in p_{\text{data}}$",
                  color=cplots.P1_COLOR, linestyle="dashed"),
                L(y=y_rew, y_err=e_rew, y_ref=y_ind,
                  label=r"$w \cdot p_{\text{data}}\ /\ p_{\text{bkg}}^{\text{MC}}$",
                  color=cplots.BKG_MC_COLOR, linestyle="dashed"),
            ]
            yield {"lines": lines, "bins": bins, "obs": obs}
            continue
        lines = [
            L(y=y_p0, y_err=e_p0, label=r"$p_{\text{data}}$", color=cplots.P0_COLOR),
            L(y=y_bkg, y_err=e_bkg, label=bkg_label, color=cplots.P1_COLOR),
            L(y=y_rew, y_err=e_rew, y_ref=y_bkg, label=cplots.LABEL_W,
              color="darkorange", linestyle="dashed"),
            L(y=y_sig, y_err=e_sig, label=cplots.LABEL_SIG, color=cplots.DATA_COLOR),
            L(y=y_rew2, y_err=e_rew2, y_ref=y_sig, label=cplots.LABEL_1MW,
              color=cplots.REW_COLOR, linestyle="dotted"),
        ]
        yield {"lines": lines, "bins": bins, "obs": obs}


def _plot_from_state(run_dir, split, out, obs_filter=None, n_class=None):
    """Rebuild observables_<split>.pdf from plot_state.pt with BackgroundSubtractionPlots itself."""
    import logging
    import classification.main as cmain

    importlib.reload(cmain)
    dataset, plot_args = cmain.load_plot_state(run_dir)
    if plot_args["n_ensemble"] > 1:
        dataset.predictions = {s: v[:n_class] for s, v in plot_args["stacked_preds"].items()}
        dataset.compute_weights()
    if obs_filter:
        for s in dataset.obs:
            keep = [i for i, o in enumerate(dataset.obs[s]) if obs_filter in o.tex_label]
            for attr in ("obs", "obs_p0", "obs_p1", "bins"):
                getattr(dataset, attr)[s] = [getattr(dataset, attr)[s][i] for i in keep]
    plots = cplots.BackgroundSubtractionPlots(logging.getLogger("replot"), dataset, **plot_args["plots_kwargs"])
    plots.plot_observables(out, split=split)


def replot(run_dir, split="trn", out=None, obs_filter=None, bkg_ref="independent", source="auto",
           n_class=None, n_events=None, bkg_yield_norm=False, max_njets=None, **hist_overrides):
    """Write the replotted PDF and return its path. ``hist_overrides`` go straight to hist_plot."""
    importlib.reload(uplots)  # shared hist_plot / hist_line live here
    importlib.reload(cplots)
    run_name = os.path.basename(os.path.normpath(run_dir))
    if out is None:
        name = "reference_check" if bkg_ref == "both" else f"observables_{split}"
        out = os.path.join(_HERE, "figs", f"{run_name}_{name}.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    state = os.path.join(run_dir, "plot_state.pt")
    pkl = os.path.join(run_dir, f"observables_{split}.pkl")
    dump = os.path.join(run_dir, "unfolding_weights_trn.pt")
    if max_njets is not None and source == "auto":
        source = "dump"
    if source == "auto":
        if os.path.isfile(state) and bkg_ref == "independent" and _state_has_raw_events(state):
            source = "plot_state"
        elif split == "trn" and os.path.isfile(dump):
            source = "dump"
        elif os.path.isfile(pkl):
            source = "pickle"
        elif os.path.isfile(state):
            source = "plot_state"
        else:
            raise FileNotFoundError(f"Nothing to replot {split} from in {run_dir}")
    print(f"  source: {source}")

    if source == "plot_state":
        _plot_from_state(run_dir, split, out, obs_filter=obs_filter, n_class=n_class)
        print(f"  wrote {out}")
        return out
    if source == "dump":
        entries = _lines_from_dump(run_dir, bkg_ref=bkg_ref, n_class=n_class, n_events=n_events,
                                   obs_filter=obs_filter, bkg_yield_norm=bkg_yield_norm, max_njets=max_njets)
    else:
        with open(pkl, "rb") as f:
            entries = pickle.load(f)

    default_kwargs = dict(show_ratios=True, model_name="Rew.", no_scale=True, truth_name="Target")
    with PdfPages(out) as pp:
        for e in entries:
            if obs_filter and obs_filter not in e["obs"].tex_label:
                continue
            kwargs = {**default_kwargs, **e.get("hist_kwargs", {}),
                      **cplots.legend_layout(e["obs"]), **hist_overrides}
            cplots.hist_plot(pp, e["lines"], e["bins"], e["obs"], **kwargs)
    print(f"  wrote {out}")
    return out


def _state_has_raw_events(path):
    """plot_state.pt saved with raw p0/p1 events (needed to recompute today's observables)?"""
    with zipfile.ZipFile(path) as z:  # torch.save format is a zip; cheap peek at the pickle
        data_pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        return b"derived_channels" in z.read(data_pkl)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="classifier run directory")
    ap.add_argument("--split", choices=["trn", "tst"], default="trn")
    ap.add_argument("--out", default=None, help="output PDF (default: experiments/figs/<run>_observables_<split>.pdf)")
    ap.add_argument("--obs", default=None, help="only observables whose tex_label contains this substring")
    ap.add_argument("--bkg-ref", choices=["independent", "in-data", "both"], default="independent",
                    help="dump source: p_bkg^MC = pool MC rows not in p_data (exact original), or bkg "
                         "inside p_data; 'both' = reference check, w*p_data ratio against each")
    ap.add_argument("--source", choices=["auto", "plot_state", "dump", "pickle"], default="auto")
    ap.add_argument("--n_class", "--n-class", dest="n_class", type=int, default=None,
                    help="use only the first K ensemble members (fast layout check: --n_class 1)")
    ap.add_argument("--n-events", "--n_events", dest="n_events", type=int, default=None,
                    help="dump source: random subset of N p_data events (fast layout check)")
    ap.add_argument("--bkg-yield-norm", action="store_true",
                    help="dump source: rescale each member's w so that sum(w * p_data) = background yield")
    ap.add_argument("--bkg-yield-norm-ensemble", action="store_true",
                    help="dump source: as --bkg-yield-norm, but one common factor for the whole ensemble")
    ap.add_argument("--max-njets", type=int, default=None,
                    help="dump source: keep only events with at most this many reco jets (before normalizing)")
    args = ap.parse_args()
    replot(args.run_dir, split=args.split, out=args.out, obs_filter=args.obs, bkg_ref=args.bkg_ref,
           source=args.source, n_class=args.n_class, n_events=args.n_events, bkg_yield_norm="ensemble" if args.bkg_yield_norm_ensemble else args.bkg_yield_norm,
           max_njets=args.max_njets)
