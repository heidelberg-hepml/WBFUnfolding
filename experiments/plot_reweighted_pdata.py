#!/usr/bin/env python
"""Standalone diagnostic: reweighted p_data observables from a classifier dump.

Given a classifier run directory, load ``unfolding_weights_trn.pt`` + the ``p_data_{reco,gen}.npy``
dumps and plot, per observable, the background-subtracted signal estimate ``(1-w)*p_data`` against the
true signal (``is_signal``), plus the ``p_data`` mixture and the per-process (ttbar / WWjj) truth. This
is a pure CLASSIFIER diagnostic (no unfolding): it answers "does the reweighting recover the signal
shape?" without re-running anything.

By default it writes TWO PDFs: an "Events" (absolute yields) plot and a "Normalized" (unit-area) plot.

Usage (from the repo root):
    python experiments/plot_reweighted_pdata.py <run_dir> [--level reco|gen|both] [--out PDF]
                                                  [--split-sign] [--norm both|events|unit]

Or from a notebook in this folder:
    from plot_reweighted_pdata import plot_reweighted_pdata
    plot_reweighted_pdata("results/MLPClassifier/<run>", level="both", normalize=False)  # Events
    plot_reweighted_pdata("results/MLPClassifier/<run>", level="both", normalize=True)   # Normalized
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages

# Make the repo's `src` importable so the bare `from utils import ...` / `from datasets...` /
# `from classification...` package imports resolve regardless of the current working directory.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils import physics as phys  # noqa: E402
from datasets.classification import ClassificationDatasets  # noqa: E402
from classification.plots import Line, compute_hist_data, hist_plot, hist_line, corner_text  # noqa: E402

# colours (roughly matching the repo's plotting conventions)
C_DATA = "#888888"       # p_data mixture (grey)
C_SIGNAL = "#07078A"     # true signal (blue)  -- ratio reference
C_REW = "#E08000"        # (1-w)*p_data reweighted estimate (orange)
C_TTBAR = "#008B8B"      # ttbar truth (teal)
C_WWJJ = "#B23AEE"       # WWjj truth (purple)
C_BKG = "#000000"        # total background in p_data (ttbar+WWjj) (black)


@dataclass
class WeightMask:
    """A reweighted line restricted to events selected by a condition on the weights.

    ``select`` is a predicate on the per-event mean reweighting factor ``1-w`` (shape ``(N,)``) and
    returns a boolean mask ``(N,)``. The line plots ``(1-w)*p_data`` over the selected events, keeping
    the ensemble band (the ``(K,N)`` weights are column-sliced, not reduced).

    ``abs_weight``: if True, weight the selected events by ``|1-w|`` instead of ``1-w``, so a
    negative-weight component (e.g. the ``(1-w)<0`` sign split) plots as a positive line.

    ``secondary_axis``: if True, draw this line on a secondary right y-axis, unit-area normalized,
    while all other lines stay on the left axis with absolute (Events) yields -- a pure shape overlay
    to see where this component is enhanced relative to the unnormalized distributions.
    """
    label: str
    select: Callable          # (w1m: np.ndarray[N]) -> np.ndarray[bool, N]
    color: str = C_REW
    linestyle: str = "dotted"
    alpha: float = 0.6
    abs_weight: bool = False
    secondary_axis: bool = False


# Built-in preset: split the reweighted contribution by the sign of (1-w). The negative-weight
# component is the classic background-subtraction pathology (classifier assigned w > 1).
SIGN_SPLIT = [
    WeightMask(r"$|1-w|,\,(1-w)<0$", lambda w: w < 0, color="#C00000", linestyle="dotted",
               abs_weight=True),
    WeightMask(r"$(1-w)\geq 0$", lambda w: w >= 0, color="#00A000", linestyle="dashed"),
]

# The |1-w| negative-weight component on a secondary NORMALIZED (unit-area) right axis, everything
# else left/unnormalized -- shape overlay to see where the reweighting is enhanced.
ABS_NEG_SECONDARY = [
    WeightMask(r"$|1-w|,\,(1-w)<0$ (norm.)", lambda w: w < 0, color="#C00000", linestyle="dotted",
               alpha=0.9, abs_weight=True, secondary_axis=True),
]


def _get_observables():
    """The classifier's full observable list, without loading any data (bypasses __init__)."""
    stub = ClassificationDatasets.__new__(ClassificationDatasets)
    ClassificationDatasets.init_observables(stub)  # sets stub.observables
    return stub.observables


def _to_ptphietam(raw):
    """(N, 21) EPxPyPz + multiplicity  ->  (N, 21) PtPhiEtaM + multiplicity (mirrors classification.py)."""
    t = torch.as_tensor(np.asarray(raw), dtype=torch.float64)
    parts = [phys.EPxPyPz_to_PtPhiEtaM(t[:, i : i + 4]) for i in range(0, 20, 4)]
    return torch.cat(parts + [t[:, 20:21]], dim=1)


def _twin_hist_plot(pdf, left_lines, twin_lines, bins, observable, title=None,
                    model_name="Rew.", truth_name="Signal", show_ratios=True):
    """Like ``hist_plot`` but with a secondary right y-axis for ``twin_lines``.

    Left axis: absolute yields ("Events", no per-line scaling). Right axis: each twin line scaled to
    unit area ("Normalized"), a pure shape overlay to locate where that component is enhanced. The
    ratio panel shows the left lines that carry a ``y_ref`` (vs signal), matching ``hist_plot``.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n_panels = 1 + int(show_ratios)
        fig, axs = plt.subplots(
            n_panels, 1, sharex=True, figsize=(6, 4.5),
            gridspec_kw={"height_ratios": (12, 3)[:n_panels], "hspace": 0.0},
        )
        if n_panels == 1:
            axs = [axs]
        ax_main = axs[0]

        # left axis: absolute (Events) yields, no per-line scaling (no_scale semantics)
        for line in left_lines:
            hist_line(ax_main, bins, line.y, line.y_err, label=line.label, color=line.color,
                      fill=line.fill, linestyle=line.linestyle, alpha=line.alpha)
            if show_ratios and line.y_ref is not None:
                ratio = line.y / line.y_ref
                isnan = np.isnan(ratio) | np.isinf(ratio)
                if line.y_err is not None:
                    ratio_err = line.y_err / line.y_ref
                    ratio_err[..., isnan] = 0.0
                else:
                    ratio_err = None
                ratio[isnan] = 1.0
                hist_line(axs[1], bins, ratio, ratio_err, label=None, color=line.color,
                          linestyle=line.linestyle, alpha=line.alpha)

        # right axis: unit-area normalized shape overlay
        ax_tw = ax_main.twinx()
        for line in twin_lines:
            integral = np.sum((bins[1:] - bins[:-1]) * line.y)
            scale = 1.0 / integral if integral != 0.0 else 1.0
            hist_line(ax_tw, bins, line.y * scale,
                      line.y_err * scale if line.y_err is not None else None,
                      label=line.label, color=line.color, fill=line.fill,
                      linestyle=line.linestyle, alpha=line.alpha)

        ax_main.set_ylabel(r"$\mathrm{Events}$")
        ax_main.set_yscale(observable.yscale)
        ax_tw.set_ylabel(r"$\mathrm{Normalized}$")
        if title is not None:
            corner_text(ax_main, title, "left", "top")

        # merge legends from both axes into one
        h1, l1 = ax_main.get_legend_handles_labels()
        h2, l2 = ax_tw.get_legend_handles_labels()
        ax_main.legend(h1 + h2, l1 + l2, frameon=False, handlelength=1.0)

        if show_ratios:
            axs[1].set_ylabel(f"$\\frac{{\\text{{{model_name}}}}}{{\\text{{{truth_name}}}}}$")
            axs[1].set_yticks([0.9, 1, 1.1])
            axs[1].set_ylim([0.85, 1.15])
            axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.1, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.9, c="black", ls="dotted", lw=0.5)

        unit = "" if observable.unit is None else f"[\\mathrm{{{observable.unit}}}]"
        axs[-1].set_xlabel(f"${{{observable.tex_label}}}\\ {unit}$")
        axs[-1].set_xscale(observable.xscale)
        axs[-1].set_xlim(bins[0], bins[-1])
        # a touch more right margin than hist_plot's default so the twin ylabel isn't clipped
        fig.subplots_adjust(left=0.15, bottom=0.15, right=0.86, top=0.95)
        plt.savefig(pdf, format="pdf")
        plt.close()


def plot_reweighted_pdata(run_dir, level="both", out=None, weight_masks=None, normalize=False):
    """Plot reweighted p_data observables from a classifier run dir. Returns the output PDF path.

    ``weight_masks``: optional list of :class:`WeightMask`. Each adds a reweighted band line restricted
    to the events its ``select`` predicate keeps (e.g. ``SIGN_SPLIT`` for the (1-w)<0 / >=0 split).

    ``normalize``: if True, each line is scaled to unit area (``no_scale=False`` -> "Normalized" y-axis),
    a pure shape comparison. If False (default), the physical ``prior_weights`` (which sum to the tiny
    fiducial fraction Sum(alpha_i), not the event count) are rescaled by ``N / sum(prior)`` so the p_data
    integral equals the event count N ("Events" y-axis) -- the same norm_factor the classifier's own
    ``plot_observables`` uses.
    """
    weights_path = os.path.join(run_dir, "unfolding_weights_trn.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"No unfolding_weights_trn.pt in {run_dir}")
    d = torch.load(weights_path, map_location="cpu")

    saw = np.asarray(d["signal_approx_weights"], dtype=float)   # (K, N) = 1 - w
    if saw.ndim == 1:
        saw = saw[None, :]
    prior = np.asarray(d["prior_weights"], dtype=float)          # (N,)
    is_sig = np.asarray(d["is_signal"], dtype=bool)              # (N,)
    mask_tt = d.get("mask_ttbar_in_p0")
    mask_ww = d.get("mask_wwjj_in_p0")
    mask_tt = np.asarray(mask_tt, dtype=bool) if mask_tt is not None else None
    mask_ww = np.asarray(mask_ww, dtype=bool) if mask_ww is not None else None

    K, N = saw.shape
    w1m = saw.mean(axis=0)                                       # (N,) per-event mean (1-w)
    frac_neg = 100.0 * float(np.mean(w1m < 0))
    weight_masks = weight_masks or []
    # A secondary-axis mask forces an "Events left / normalized twin" layout (a shape overlay): the
    # left axis is always absolute yields regardless of `normalize`, and the flagged line is drawn on
    # a right unit-area axis by _twin_hist_plot.
    has_secondary = any(wm.secondary_axis for wm in weight_masks)
    if has_secondary:
        normalize = False
    # "Events" scaling: sum(prior) = Sum(alpha_i) (fiducial fraction, tiny), so rescale to integral = N.
    # When normalize=True the per-line unit-area scaling in hist_plot takes over and wscale is unused.
    wscale = 1.0 if normalize else float(N) / float(prior.sum())
    print(
        f"  run_dir          : {run_dir}\n"
        f"  dump_split       : {d.get('dump_split', '?')}   holdout_frac: {d.get('holdout_frac', '?')}\n"
        f"  ensemble (K)     : {K}\n"
        f"  N p_data         : {N}   (signal={int(is_sig.sum())}, background={int((~is_sig).sum())})\n"
        f"  (1-w)<0 events   : {frac_neg:.1f}%\n"
        f"  y-axis           : {'Normalized (unit area)' if normalize else f'Events (sum(prior)={prior.sum():.4g} -> x {wscale:.4g})'}"
    )
    if mask_tt is None or mask_ww is None:
        print("  note: per-process masks absent from dump -> skipping ttbar/WWjj truth lines")
    if has_secondary:
        print("  note: secondary-axis mask(s) drawn on a right NORMALIZED axis (shape overlay)")

    for wm in weight_masks:
        m = np.asarray(wm.select(w1m), dtype=bool)
        print(f"  weight mask '{wm.label}' selects {int(m.sum())}/{N} events "
              f"({100.0 * m.mean():.1f}%)")

    levels = ["reco", "gen"] if level == "both" else [level]
    if out is None:
        run_name = os.path.basename(os.path.normpath(run_dir))
        figs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
        suffix = "_secondary" if has_secondary else ("_normalized" if normalize else "")
        out = os.path.join(figs_dir, f"{run_name}{suffix}.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    observables = _get_observables()

    with PdfPages(out) as pdf:
        for lvl in levels:
            npy = os.path.join(run_dir, f"p_data_{lvl}.npy")
            if not os.path.isfile(npy):
                print(f"  [skip {lvl}] {npy} not found")
                continue
            raw = np.load(npy)
            if raw.shape[0] != N:
                raise ValueError(f"{npy} has {raw.shape[0]} rows but the dump has {N} events")
            p = _to_ptphietam(raw)
            print(f"  [{lvl}] plotting {len(observables)} observables from {npy}")
            # legend argument: reco pages carry the reco-level x_reco, gen pages the
            # particle/parton-level x_part (the x-axis keeps the bare observable name).
            arg = r"x_{\mathrm{reco}}" if lvl == "reco" else r"x_{\mathrm{part}}"

            for obs in observables:
                x = np.asarray(obs.compute(p))
                bins = np.asarray(obs.bins(x))

                y_data, e_data = compute_hist_data(bins, x, weights=prior * wscale)
                y_sig, e_sig = compute_hist_data(bins, x[is_sig], weights=prior[is_sig] * wscale)
                # total background in p_data (= ttbar + WWjj = complement of is_signal). Where this line
                # nearly meets p_data, the signal fraction is tiny -> the reweighted estimate fluctuates.
                bkg = ~is_sig
                y_bkg, e_bkg = compute_hist_data(bins, x[bkg], weights=prior[bkg] * wscale)
                # (1-w)*p_data: pass the full (K, N) weights so the ensemble becomes a mean+/-std band
                y_rew, e_rew = compute_hist_data(bins, x, weights=prior[None, :] * saw * wscale)

                lines = [
                    Line(y=y_data, y_err=e_data, label=rf"$p_{{\mathrm{{data}}}}({arg})$", color=C_DATA),
                    Line(y=y_bkg, y_err=e_bkg, label=rf"$p_{{\mathrm{{bkg}}}}({arg})$", color=C_BKG,
                         linestyle="solid", alpha=0.7),
                    Line(y=y_sig, y_err=e_sig, label=rf"$p_{{\mathrm{{signal}}}}({arg})$", color=C_SIGNAL),
                    Line(
                        y=y_rew, y_err=e_rew, y_ref=y_sig,
                        label=rf"$(1-w({arg}))\,p_{{\mathrm{{data}}}}$",
                        color=C_REW, linestyle="dotted",
                    ),
                ]
                # extra reweighted lines, each restricted to a custom mask on the weights.
                # secondary_axis masks are collected separately for the right (normalized) axis.
                twin_lines = []
                for wm in weight_masks:
                    m = np.asarray(wm.select(w1m), dtype=bool)
                    if not m.any():
                        continue
                    w_line = np.abs(saw) if wm.abs_weight else saw
                    y_m, e_m = compute_hist_data(bins, x[m], weights=(prior[None, :] * w_line * wscale)[:, m])
                    ln = Line(
                        y=y_m, y_err=e_m,
                        y_ref=None if wm.secondary_axis else y_sig,
                        label=rf"{wm.label}$\,p_{{\mathrm{{data}}}}({arg})$",
                        color=wm.color, linestyle=wm.linestyle, alpha=wm.alpha,
                    )
                    (twin_lines if wm.secondary_axis else lines).append(ln)
                if mask_tt is not None:
                    y_tt, e_tt = compute_hist_data(bins, x[mask_tt], weights=prior[mask_tt] * wscale)
                    lines.append(Line(y=y_tt, y_err=e_tt, label=rf"$t\bar{{t}}\,({arg})$",
                                      color=C_TTBAR, linestyle="dashed", alpha=0.6))
                if mask_ww is not None:
                    y_ww, e_ww = compute_hist_data(bins, x[mask_ww], weights=prior[mask_ww] * wscale)
                    lines.append(Line(y=y_ww, y_err=e_ww, label=rf"$WWjj\,({arg})$",
                                      color=C_WWJJ, linestyle="dashed", alpha=0.6))

                lines = [ln for ln in lines if ln is not None]
                if twin_lines:
                    _twin_hist_plot(
                        pdf, lines, twin_lines, bins, obs,
                        title=rf"$\mathrm{{{lvl}}}$", model_name="Rew.", truth_name="Signal",
                    )
                else:
                    hist_plot(
                        pdf, lines, bins, obs, show_ratios=True, no_scale=not normalize,
                        title=rf"$\mathrm{{{lvl}}}$", model_name="Rew.", truth_name="Signal",
                    )
    print(f"  saved -> {out}")
    return out


def plot_w_pdata(run_dir, level="both", out=None, normalize=False, w_thresh=1.0):
    """Plot ONLY p_data, p_bkg, w*p_data, and w*p_data restricted to w>w_thresh events.

    Here ``w = 1 - signal_approx_weights`` is the classifier weight. Since ``(1-w)*p_data`` estimates
    the signal, its complement ``w*p_data`` estimates the BACKGROUND, so it should track the true
    ``p_bkg`` (= complement of is_signal); the ``w>1`` subset isolates the negative-``(1-w)`` /
    over-subtraction events. ``normalize`` and the "Events" rescaling behave as in
    :func:`plot_reweighted_pdata`. Returns the output PDF path.
    """
    weights_path = os.path.join(run_dir, "unfolding_weights_trn.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"No unfolding_weights_trn.pt in {run_dir}")
    d = torch.load(weights_path, map_location="cpu")
    saw = np.asarray(d["signal_approx_weights"], dtype=float)   # (K, N) = 1 - w
    if saw.ndim == 1:
        saw = saw[None, :]
    prior = np.asarray(d["prior_weights"], dtype=float)          # (N,)
    is_sig = np.asarray(d["is_signal"], dtype=bool)              # (N,)
    K, N = saw.shape
    w = 1.0 - saw                                                # (K, N) classifier weight
    wmean = w.mean(axis=0)                                       # (N,) per-event mean w
    m_hi = wmean > w_thresh                                      # w>1 subset (== per-event (1-w)<0)
    m_lo = wmean < w_thresh                                      # w<1 subset (complement)
    wscale = 1.0 if normalize else float(N) / float(prior.sum())
    print(
        f"  run_dir          : {run_dir}\n"
        f"  ensemble (K)     : {K}   N p_data: {N}\n"
        f"  w>{w_thresh:g} events    : {100.0 * m_hi.mean():.1f}%  ({int(m_hi.sum())}/{N})\n"
        f"  w<{w_thresh:g} events    : {100.0 * m_lo.mean():.1f}%  ({int(m_lo.sum())}/{N})\n"
        f"  y-axis           : {'Normalized (unit area)' if normalize else f'Events (sum(prior)={prior.sum():.4g} -> x {wscale:.4g})'}"
    )

    levels = ["reco", "gen"] if level == "both" else [level]
    if out is None:
        run_name = os.path.basename(os.path.normpath(run_dir))
        figs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
        suffix = "_normalized" if normalize else ""
        out = os.path.join(figs_dir, f"{run_name}_wdata{suffix}.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    observables = _get_observables()
    bkg = ~is_sig

    with PdfPages(out) as pdf:
        for lvl in levels:
            npy = os.path.join(run_dir, f"p_data_{lvl}.npy")
            if not os.path.isfile(npy):
                print(f"  [skip {lvl}] {npy} not found")
                continue
            raw = np.load(npy)
            if raw.shape[0] != N:
                raise ValueError(f"{npy} has {raw.shape[0]} rows but the dump has {N} events")
            p = _to_ptphietam(raw)
            print(f"  [{lvl}] plotting {len(observables)} observables from {npy}")
            # legend argument: reco pages carry the reco-level x_reco, gen pages the
            # particle/parton-level x_part (the x-axis keeps the bare observable name).
            arg = r"x_{\mathrm{reco}}" if lvl == "reco" else r"x_{\mathrm{part}}"

            for obs in observables:
                x = np.asarray(obs.compute(p))
                bins = np.asarray(obs.bins(x))

                y_data, e_data = compute_hist_data(bins, x, weights=prior * wscale)
                y_bkg, e_bkg = compute_hist_data(bins, x[bkg], weights=prior[bkg] * wscale)
                # w*p_data ~ estimated background (ensemble mean+/-std band)
                y_w, e_w = compute_hist_data(bins, x, weights=prior[None, :] * w * wscale)

                lines = [
                    Line(y=y_data, y_err=e_data, label=rf"$p_{{\mathrm{{data}}}}({arg})$", color=C_DATA),
                    Line(y=y_bkg, y_err=e_bkg, label=rf"$p_{{\mathrm{{bkg}}}}({arg})$", color=C_BKG,
                         linestyle="solid", alpha=0.7),
                    Line(y=y_w, y_err=e_w, y_ref=y_bkg, label=rf"$w({arg})\,p_{{\mathrm{{data}}}}$",
                         color=C_REW, linestyle="dotted"),
                ]
                if m_hi.any():
                    y_hi, e_hi = compute_hist_data(bins, x[m_hi],
                                                   weights=(prior[None, :] * w * wscale)[:, m_hi])
                    lines.append(Line(y=y_hi, y_err=e_hi,
                                      label=rf"$w({arg})\,p_{{\mathrm{{data}}}}\,[w>{w_thresh:g}]$",
                                      color="#C00000", linestyle="dashed", alpha=0.8))
                if m_lo.any():
                    y_lo, e_lo = compute_hist_data(bins, x[m_lo],
                                                   weights=(prior[None, :] * w * wscale)[:, m_lo])
                    lines.append(Line(y=y_lo, y_err=e_lo,
                                      label=rf"$w({arg})\,p_{{\mathrm{{data}}}}\,[w<{w_thresh:g}]$",
                                      color="#00A000", linestyle="dashed", alpha=0.8))

                hist_plot(pdf, lines, bins, obs, show_ratios=True, no_scale=not normalize,
                          title=rf"$\mathrm{{{lvl}}}$", model_name="w data", truth_name="bkg")
    print(f"  saved -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="classifier run directory containing unfolding_weights_trn.pt")
    ap.add_argument("--level", choices=["reco", "gen", "both"], default="both",
                    help="observable level to plot (default: both)")
    ap.add_argument("--out", default=None,
                    help="output PDF for the unnormalized (Events) plot; the normalized plot gets a "
                         "'_normalized' suffix. default: experiments/figs/<run_name>.pdf")
    ap.add_argument("--split-sign", action="store_true",
                    help="add reweighted lines split by the sign of (1-w) (negative-weight diagnostic)")
    ap.add_argument("--abs-secondary", action="store_true",
                    help="single plot: all lines unnormalized (Events) plus |1-w| over (1-w)<0 events "
                         "on a secondary normalized right axis (shape/enhancement overlay)")
    ap.add_argument("--w-data", action="store_true",
                    help="focused plot: only p_data, p_bkg, w*p_data (~estimated background), and "
                         "w*p_data over w>1 events (the over-subtraction / negative-(1-w) piece)")
    ap.add_argument("--norm", choices=["both", "events", "unit"], default="both",
                    help="which y-axis scaling to produce: 'events' (absolute), 'unit' (normalized), "
                         "or 'both' (default)")
    args = ap.parse_args()

    def _norm_out(o):
        if o is None:
            return None
        root, ext = os.path.splitext(o)
        return f"{root}_normalized{ext}"

    if args.w_data:
        # focused plot: p_data, p_bkg, w*p_data, w*p_data[w>1]
        if args.norm in ("both", "events"):
            plot_w_pdata(args.run_dir, level=args.level, out=args.out, normalize=False)
        if args.norm in ("both", "unit"):
            plot_w_pdata(args.run_dir, level=args.level, out=_norm_out(args.out), normalize=True)
        return
    if args.abs_secondary:
        # one plot: Events on the left, |1-w|,(1-w)<0 on a normalized right axis
        plot_reweighted_pdata(args.run_dir, level=args.level, out=args.out,
                              weight_masks=ABS_NEG_SECONDARY)
        return
    weight_masks = SIGN_SPLIT if args.split_sign else None

    if args.norm in ("both", "events"):
        plot_reweighted_pdata(args.run_dir, level=args.level, out=args.out,
                              weight_masks=weight_masks, normalize=False)
    if args.norm in ("both", "unit"):
        plot_reweighted_pdata(args.run_dir, level=args.level, out=_norm_out(args.out),
                              weight_masks=weight_masks, normalize=True)


if __name__ == "__main__":
    main()
