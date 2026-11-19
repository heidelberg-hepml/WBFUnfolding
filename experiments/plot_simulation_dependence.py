"""Plots for results/simulation_dependence (CPU only, from the saved per-event evaluations).

For every evaluation task <run_dir>/<task>/{events.npz, meta.pkl} written by run_sim_dep
(src/unfolding/main.py):
  * single-label tasks -> <task>/observables.pdf: truth (parton), reco, unfolded; area-normalized,
    ratio to the truth (the usual unfolding plot);
  * --wrong-labels TASK --correct CORRECT_TASK -> <run_dir>/<TASK>_spread.pdf: the line is the
    CORRECT-label unfolding (task CORRECT, same events); the band spans, per bin, the min and max of the
    area-normalized unfoldings over all labels (correct + every label of TASK), in main and ratio panel.
    TASK itself gets no per-label plot.
  * with --sigma-band, only <run_dir>/<TASK>_spread_withsigmaband.pdf: the same line with a +-1 sigma band (sample
    std over all labels per bin, kept inside the envelope) and the envelope as a lighter band.
  * with --envelope, only <run_dir>/<TASK>_spread_envelope.pdf: the default spread plot (line + min/max envelope as
    one band, legend "TraCFM"), without re-plotting the per-task observables.pdf.
  * --random-label TASK -> <run_dir>/<TASK>_spread_withsigmaband.pdf, for a sample outside the model's conditions: the
    line is the unfolding with a random label per event, both bands come from the fixed labels.

Usage (from the repo root):
    python experiments/plot_simulation_dependence.py <run_dir> [--trained-on "SM"]
        [--wrong-labels eval_cpodd_pos_wronglabels --correct eval_cpodd_pos [--wrong-labels ... --correct ...]]
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SAMPLE_TITLES = {  # task name -> plot title (math-mode signs)
    "eval_sm": r"$\textrm{SM}$",
    "eval_cpeven_pos": r"$\textrm{CP-even}(+)$", "eval_cpeven_neg": r"$\textrm{CP-even}(-)$",
    "eval_cpodd_pos": r"$\textrm{CP-odd}(+)$", "eval_cpodd_neg": r"$\textrm{CP-odd}(-)$",
}
LABEL_NAMES = ["SM", "CP-even(+)", "CP-even(-)", "CP-odd(+)", "CP-odd(-)"]  # condition ids 0..4
CORRECT_NAMES = {"eval_sm": "SM", "eval_cpeven_pos": "CP-even(+)", "eval_cpeven_neg": "CP-even(-)",
                 "eval_cpodd_pos": "CP-odd(+)", "eval_cpodd_neg": "CP-odd(-)"}


JQ_LABELS = None  # --jq-labels obs|sub: axis labels name both reco jets and parton quarks


def jq_label(tex, style="obs"):
    """Reco-and-parton axis label for jet observables (pairs written jj/qq):
    style "obs": dphi_{jj} / dphi_{qq},  p_{T,j_1} / p_{T,q_1};
    style "sub": dphi_{jj/qq},           p_{T,j_1/q_1}."""
    t = tex.replace("j_{1}j_{2}", "jj")
    if "jj" not in t and "j_{" not in t:
        return tex
    if style == "obs":
        return rf"{t}\,/\,{t.replace('jj', 'qq').replace('j_{', 'q_{')}"
    return t.replace("jj", "jj/qq").replace("j_{1}", "j_1/q_1").replace("j_{2}", "j_2/q_2")


class _Obs:
    """Minimal observable for hist_plot (labels, unit, scales)."""

    def __init__(self, d):
        self.tex_label, self.unit, self.yscale, self.xscale = d["tex_label"], d["unit"], d["yscale"], "linear"
        if JQ_LABELS:
            self.tex_label = jq_label(self.tex_label, JQ_LABELS)
        self.key = d.get("key")


def load(task_dir):
    z = np.load(os.path.join(task_dir, "events.npz"))
    meta = pickle.load(open(os.path.join(task_dir, "meta.pkl"), "rb"))
    return z, meta


def keep(meta):
    """Indices of the observables to plot (primary channels the unfolder uses + all derived ones)."""
    ch = set(meta["channels"])
    return [i for i, o in enumerate(meta["observables"]) if o["channel"] is None or o["channel"] in ch]


def plot_task(task_dir, trained_on, P, out=None):
    from matplotlib.backends.backend_pdf import PdfPages

    z, meta = load(task_dir)
    name = os.path.basename(task_dir)
    title = SAMPLE_TITLES.get(name.replace("_wronglabels", ""), name)
    model = rf"\mathrm{{TraCFM}}" + (rf"_{{\mathrm{{{trained_on}}}}}" if trained_on else "")
    out = out or os.path.join(task_dir, "observables.pdf")
    with PdfPages(out) as pp:
        for i in keep(meta):
            o = meta["observables"][i]
            bins = o["bins"]
            yT, eT = P.compute_hist_data(bins, z["obs_part"][i])
            yR, eR = P.compute_hist_data(bins, z["obs_reco"][i])
            lines = [P.Line(y=yR, y_err=eR, label="Reco", color=P.RECO_COLOR),
                     P.Line(y=yT, y_err=eT, label="Parton", color=P.PART_COLOR)]
            for k, lab in enumerate(z["labels"]):
                yU, eU = P.compute_hist_data(bins, z["obs_unf"][k][i])
                cond = "" if lab == "true" else rf",\ c={LABEL_NAMES[int(lab)]}"
                lines.append(P.Line(y=yU, y_err=eU, y_ref=yT, label=rf"${model}{cond}$", color=P.CFM_COLOR))
            P.hist_plot(pp, lines, bins, _Obs(o), show_ratios=True, model_name="TraCFM", truth_name="Parton",
                        title=title, title_on_axs=True)
    print(f"  wrote {out} ({len(keep(meta))} pages)")


def plot_spread(run_dir, wrong_task, correct_task, P, sigma_band=False, suffix=None, out=None):
    from matplotlib.backends.backend_pdf import PdfPages

    zw, mw = load(os.path.join(run_dir, wrong_task))
    zc, mc = load(os.path.join(run_dir, correct_task))
    assert mw["n_events"] == mc["n_events"] and np.array_equal(zw["obs_part"], zc["obs_part"]), \
        "wrong- and correct-label tasks must unfold the same events"
    assert list(zc["labels"]) == ["true"], "correct task must use each event's own label"
    wrong = [LABEL_NAMES[int(l)] for l in zw["labels"]]
    correct = CORRECT_NAMES.get(correct_task, "true")
    suffix = suffix if suffix is not None else ('_withsigmaband' if sigma_band else '')
    out = out or os.path.join(run_dir, f"{wrong_task}_spread{suffix}.pdf")
    title = SAMPLE_TITLES.get(correct_task, correct_task)
    with PdfPages(out) as pp:
        for i in keep(mc):
            o = mc["observables"][i]
            bins = o["bins"]
            bw = np.diff(bins)
            norm = lambda y: y / np.sum(bw * y)
            yT, eT = P.compute_hist_data(bins, zc["obs_part"][i])
            yC, _ = P.compute_hist_data(bins, zc["obs_unf"][0][i])
            yW = [P.compute_hist_data(bins, zw["obs_unf"][k][i])[0] for k in range(len(wrong))]
            # band: per-bin min and max over all labels (correct + wrong), area-normalized, in yC's units
            nAll = np.array([norm(yC)] + [norm(y) for y in yW])
            band = np.array([nAll.min(axis=0), nAll.max(axis=0)]) * np.sum(bw * yC)
            yR, eR = P.compute_hist_data(bins, zc["obs_reco"][i])
            lines = [P.Line(y=yR, y_err=eR, y_ref=yT, label="Reco", color=P.RECO_COLOR,
                            ratio_linestyle="dashed"),
                     P.Line(y=yT, y_err=eT, label="Parton", color=P.PART_COLOR)]
            if sigma_band:
                # sample std over the labels, around the correct-label line, kept inside the envelope
                sig = nAll.std(axis=0, ddof=1) * np.sum(bw * yC)
                sig_band = np.array([np.maximum(yC - sig, band[0]), np.minimum(yC + sig, band[1])])
                lines += [P.Line(y=yC, y_err=band, y_ref=yT, label="Envelope", color=P.CFM_COLOR, alpha=0.4),
                          P.Line(y=yC, y_err=sig_band, y_ref=yT, label=r"TraCFM $\pm 1\sigma$", color=P.CFM_COLOR)]
            else:
                lines.append(P.Line(y=yC, y_err=band, y_ref=yT, label="TraCFM", color=P.CFM_COLOR))
            P.hist_plot(pp, lines, bins, _Obs(o), show_ratios=True, model_name="TraCFM", truth_name="Parton",
                        title=title, title_on_axs=True, ratio_label="Ratio to\nParton")
    print(f"  wrote {out}  (wrong labels: {', '.join(wrong)})")


def plot_random_label_spread(run_dir, task, P):
    """<TASK>_spread_withsigmaband.pdf for a sample outside the conditions: the line is the unfolding with a random
    label per event; envelope and sample std come from the fixed-label unfoldings only, in the line's units."""
    from matplotlib.backends.backend_pdf import PdfPages

    z, m = load(os.path.join(run_dir, task))
    labels = [str(l) for l in z["labels"]]
    k_rand = labels.index("random")
    fixed = [k for k in range(len(labels)) if k != k_rand]
    assert len(fixed) >= 2, "the band needs at least two fixed labels"
    out = os.path.join(run_dir, f"{task}_spread_withsigmaband.pdf")
    with PdfPages(out) as pp:
        for i in keep(m):
            o = m["observables"][i]
            bins = o["bins"]
            bw = np.diff(bins)
            norm = lambda y: y / np.sum(bw * y)
            yT, eT = P.compute_hist_data(bins, z["obs_part"][i])
            yL, _ = P.compute_hist_data(bins, z["obs_unf"][k_rand][i])
            yR, eR = P.compute_hist_data(bins, z["obs_reco"][i])
            nF = np.array([norm(P.compute_hist_data(bins, z["obs_unf"][k][i])[0]) for k in fixed])
            area = np.sum(bw * yL)
            band = np.array([nF.min(axis=0), nF.max(axis=0)]) * area
            sig = nF.std(axis=0, ddof=1) * area
            sig_band = np.array([np.maximum(yL - sig, band[0]), np.minimum(yL + sig, band[1])])
            lines = [P.Line(y=yR, y_err=eR, y_ref=yT, label="Reco", color=P.RECO_COLOR, ratio_linestyle="dashed"),
                     P.Line(y=yT, y_err=eT, label="Parton", color=P.PART_COLOR),
                     P.Line(y=yL, y_err=band, y_ref=yT, label="Envelope", color=P.CFM_COLOR, alpha=0.4),
                     P.Line(y=yL, y_err=sig_band, y_ref=yT, label=r"TraCFM $\pm 1\sigma$", color=P.CFM_COLOR)]
            P.hist_plot(pp, lines, bins, _Obs(o), show_ratios=True, model_name="TraCFM", truth_name="Parton",
                        title=SAMPLE_TITLES.get(task, task), title_on_axs=True, ratio_label="Ratio to\nParton")
    print(f"  wrote {out}  (labels: {', '.join(labels)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--trained-on", default=None, help='e.g. "even(+)" -> legend "TraCFM_{even(+)}"')
    ap.add_argument("--wrong-labels", action="append", default=[], help="repeatable, paired with --correct")
    ap.add_argument("--correct", action="append", default=[])
    ap.add_argument("--sigma-band", action="store_true",
                    help="only <TASK>_spread_withsigmaband.pdf: +-1 sigma over the labels, plus the lighter envelope")
    ap.add_argument("--envelope", action="store_true",
                    help="only <TASK>_spread_envelope.pdf: the line with the min/max envelope as its band, no sigma band")
    ap.add_argument("--random-label", action="append", default=[],
                    help="repeatable, only <TASK>_spread_withsigmaband.pdf: line = label 'random', bands = the fixed labels")
    ap.add_argument("--jq-labels", choices=["obs", "sub"], default=None,
                    help="axis labels name reco jets and parton quarks: obs = dphi_{jj} / dphi_{qq}, sub = dphi_{jj/qq}")
    args = ap.parse_args()
    global JQ_LABELS
    JQ_LABELS = args.jq_labels
    import unfolding.plots as P

    if args.random_label:
        for task in args.random_label:
            plot_random_label_spread(args.run_dir, task, P)
        return
    if args.sigma_band or args.envelope:
        for wrong_task, correct_task in zip(args.wrong_labels, args.correct):
            if args.sigma_band:
                plot_spread(args.run_dir, wrong_task, correct_task, P, sigma_band=True)
            if args.envelope:
                plot_spread(args.run_dir, wrong_task, correct_task, P, suffix="_envelope")
        return
    for task_dir in sorted(glob.glob(os.path.join(args.run_dir, "eval_*"))):
        if os.path.basename(task_dir) in args.wrong_labels:
            continue  # wrong-label task: only the combined spread plot, no per-label pages
        if os.path.isfile(os.path.join(task_dir, "events.npz")):
            plot_task(task_dir, args.trained_on, P)
    assert len(args.wrong_labels) == len(args.correct), "give one --correct per --wrong-labels"
    for wrong_task, correct_task in zip(args.wrong_labels, args.correct):
        plot_spread(args.run_dir, wrong_task, correct_task, P)


if __name__ == "__main__":
    main()
