"""Reweight the particle-cut ttbar reco sample onto the parton-cut one with the transformer classifier.

Both files pass the same reco selection; they differ only in the truth-level cut (the parton cut keeps
~3% of ttbar, 52k events, with a very different reco shape). The classifier pipeline's transformer
(config/model/Classifier-Transformer.yaml, same features + preprocessing as the background
subtraction) separates
    p0 = p^{particle cuts}(x_reco)   reco-gen_ttbar_*_rec.npy   (~1.53M events, trimmed)
    p1 = p^{parton cuts}(x_reco)     reco-hard_ttbar_*_rec.npy  (~40k events, trimmed)
with each class normalized to unit total weight, so the logit is log p1(x)/p0(x) and
    w(x_reco) = exp(logit)   (ensemble mean, renormalized to mean 1 over the particle-cut events)
makes the 1.69M particle-cut events follow the parton-cut reco distribution.

Both samples are shuffled (seeded) and split trn/val (dataset.trn_tst_val); the weights are computed
for EVERY particle-cut event and saved in file row order. The val events are never trained on, so the
closure on them is the overfitting check (plots + chi2 for all events and for val only).

Outputs in the run dir (results/TransformerClassifier/<date>-<run.name>/):
    weights_particle_to_parton.npy            (N_particle,) row-aligned with the particle-cut file
    weights_particle_to_parton_members.npy    (K, N_particle) per ensemble member
    ttbar_reweighting.pdf / ttbar_reweighting_val.pdf   production plots (all events / val only)
    ttbar_reweighting.pkl                     pickled lines (replot with unfolding.plots.hist_plot)
    train_metrics.pdf                         per-member loss / grad-norm curves
    weights.pdf                               log10 w per member + ensemble, per-event member spread
    class_trn.pdf / class_val.pdf             per member: classifier output for p0/p1, w, ROC/AUC
    classifier_outputs.pt                     the member outputs behind class_*.pdf
    summary.json, config.yaml, model_<k>/, out_00.log

Usage (from the repo root; hydra overrides of config/classify.yaml as extra args):
    python experiments/ttbar_chain_reweighting.py [train.ensemble=5 train.nits=50_000 ...]
    python experiments/ttbar_chain_reweighting.py --replot <run_dir>    # plots only, from the pickle
    python experiments/ttbar_chain_reweighting.py --rebuild <run_dir>   # CPU: histograms+plots from the saved weights
    python experiments/ttbar_chain_reweighting.py --heavy-weights <run_dir> [thr ...]  # CPU: + lines for w > thr
    python experiments/ttbar_chain_reweighting.py --evaluate <run_dir>  # GPU: class_{trn,val}.pdf (p0/p1 outputs, ROC/AUC)
"""
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

# trimmed ttbar files: only the uniform middle block of each (experiments/trim_background_files.py)
DATA = os.environ.get("WBF_TRIMMED_DIR",
                      os.path.join(os.environ.get("WBF_DATA_DIR", "/path/to/samples"), "trimmed"))
DEFAULT_OVERRIDES = [
    "model=Classifier-Transformer",
    "dataset.gaussianize=true",
    "dataset.explicit_split.use=false",
    "dataset.trn_tst_val=[0.9,0.0,0.1]",
    "+dataset.chain_reweighting.path_p0=" + os.path.join(DATA, "reco-gen_ttbar_nobjet_nocjv_noolv_rec.npy"),
    "+dataset.chain_reweighting.path_p1=" + os.path.join(DATA, "reco-hard_ttbar_nobjet_nocjv_noolv_rec.npy"),
    "+dataset.chain_reweighting.seed=0",
    "+dataset.chain_reweighting.kfold=null",   # e.g. 5: cross-fitted (out-of-fold) weights
    "+dataset.chain_reweighting.repeats=1",    # independent fold assignments, averaged
    "train.ensemble=5",
    "train.nits=50_000",
    "train.lr=0.0001",
    "train.scheduler=ReduceLROnPlateau",
    "train.early_stopping.use=false",
    "train.weight_decay=0.0",
    "seed=0",
    "run.name=ttbar-chain-reweighting",
]

PARTON_COLOR = "#17BECF"    # p^{parton cuts}
PARTICLE_COLOR = "#520B99"  # p^{particle cuts}
LABEL_PARTON = r"$p^{\text{parton cuts}}(x_{\text{reco}})$"
LABEL_PARTICLE = r"$p^{\text{particle cuts}}(x_{\text{reco}})$"
LABEL_REWEIGHTED = r"$w(x_{\text{reco}})\cdot p^{\text{particle cuts}}(x_{\text{reco}})$"


def plot_observables(out_pdf, entries):
    """Production plots: parton cuts (target) vs particle cuts, before (dotted) and after (solid)."""
    import unfolding.plots as P
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_pdf) as pp:
        for e in entries:
            P.hist_plot(pp, e["lines"], e["bins"], e["obs"], show_ratios=True, model_name="Rew.",
                        truth_name="Parton", title=r"$t\bar{t}$", title_on_axs=True)


def build_entries(obs_list, p0, p1, w):
    """Histogram lines per observable. p0/p1 = raw PtPhiEtaM tensors, w = weights of p0 rows."""
    import unfolding.plots as P

    entries, chi2 = [], {}
    for obs in obs_list:
        x0, x1 = np.asarray(obs.compute(p0)), np.asarray(obs.compute(p1))
        bins = np.asarray(obs.bins(torch.as_tensor(x1)))
        y1, e1 = P.compute_hist_data(bins, x1)
        y0, e0 = P.compute_hist_data(bins, x0)
        yw, ew = P.compute_hist_data(bins, x0, weights=w)
        lines = [
            P.Line(y=y1, y_err=e1, label=LABEL_PARTON, color=PARTON_COLOR),
            P.Line(y=y0, y_err=e0, y_ref=y1, label=LABEL_PARTICLE, color=PARTICLE_COLOR, linestyle="dotted"),
            P.Line(y=yw, y_err=ew, y_ref=y1, label=LABEL_REWEIGHTED, color=PARTICLE_COLOR),
        ]
        entries.append({"lines": lines, "bins": bins, "obs": obs})
        n1, nw = y1 / y1.sum(), yw / yw.sum()
        s = (e1 / y1.sum()) ** 2 + (ew / yw.sum()) ** 2
        m = s > 0
        chi2[obs.tex_label + (f" [{obs.key}]" if getattr(obs, "key", None) else "")] = float(
            ((n1 - nw)[m] ** 2 / s[m]).sum() / max(int(m.sum()), 1))
    return entries, chi2


def main(overrides):
    from classification.main import init_logger
    from classification.models import Classifier  # noqa: F401  (eval'd by cfg.model.model)
    from datasets.chain_reweighting import ChainReweightingDataset

    with initialize_config_dir(config_dir=os.path.join(_ROOT, "config"), version_base=None):
        cfg = compose("classify", overrides=DEFAULT_OVERRIDES + overrides)
    run_dir = os.path.join(_ROOT, "results", "TransformerClassifier",
                           datetime.now().strftime("%m%d_%H%M%S") + "-" + cfg.run.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger = init_logger(run_dir)
    logger.info(f"ttbar chain reweighting in {run_dir}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ChainReweightingDataset(logger, cfg.dataset)
    dims_in = len(ds.channels)
    if cfg.dataset.chain_reweighting.get("kfold", None):
        return main_kfold(run_dir, cfg, ds, logger, device)
    K = cfg.train.ensemble
    all_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(ds.p0_ppd_all[..., ds.channels].float(),
                                       torch.zeros(len(ds.p0_ppd_all)), torch.zeros(len(ds.p0_ppd_all))),
        batch_size=cfg.train.batch_size_sample, shuffle=False)
    logits = []
    for k in range(K):
        logger.info(f"  Ensemble member {k + 1}/{K}")
        torch.manual_seed(cfg.seed + k)
        member = eval(cfg.model.model)(logger=logger, cfg=cfg, dims_in=dims_in,
                                       model_path=os.path.join(run_dir, f"model_{k}"), device=device).to(device)
        member.init_dataloaders(ds)
        os.makedirs(os.path.join(run_dir, f"model_{k}"), exist_ok=True)
        member.run_training()
        torch.save(getattr(member, "losses", None), os.path.join(run_dir, f"model_{k}", "losses.pt"))
        member.load("final")
        pred, _ = member.evaluate("all particle-cut events", loader=all_loader)
        logits.append(pred.double().cpu())
    logits = torch.stack(logits)                                  # (K, N_particle), file row order
    w_members = torch.exp(torch.clamp(logits, max=50))
    w_members = w_members / w_members.mean(dim=1, keepdim=True)   # each member: mean 1
    w = w_members.mean(dim=0)
    w = (w / w.mean()).numpy()
    np.save(os.path.join(run_dir, "weights_particle_to_parton.npy"), w)
    np.save(os.path.join(run_dir, "weights_particle_to_parton_members.npy"), w_members.numpy())

    write_closure(run_dir, cfg, w, ds.rows_p0["val"], ds.rows_p1["val"], logger.info,
                  member_corr=float(np.corrcoef(w_members.numpy()).min()), ensemble=K)
    write_diagnostics(run_dir)
    evaluate_members(run_dir, cfg, ds, logger)
    write_classifier_plots(run_dir)
    logger.info(f"done -> {run_dir}")


def main_kfold(run_dir, cfg, ds, logger, device):
    """Cross-fitted weights: for each of `repeats` independent fold assignments, member (r, k) trains
    on the rows outside fold k and predicts the rows inside fold k, so EVERY event's weight comes from
    a member that never saw it. Per repeat the out-of-fold weights are normalized to mean 1 within each
    fold (removes member-to-member normalization offsets); the final weight is the mean over repeats."""
    from classification.models import Classifier  # noqa: F401  (eval'd by cfg.model.model)

    n_folds = int(cfg.dataset.chain_reweighting.kfold)
    repeats = int(cfg.dataset.chain_reweighting.get("repeats", 1))
    n0 = ds.p0_all.shape[0]
    w_rep = torch.zeros(repeats, n0, dtype=torch.float64)
    outputs = {"labels": [], "preds": [], "folds": []}
    losses = []
    for r in range(repeats):
        f0, f1 = ds.fold_assignment(n_folds, repeat=r)
        for k in range(n_folds):
            name = f"model_r{r}_f{k}"
            logger.info(f"  repeat {r + 1}/{repeats}, fold {k + 1}/{n_folds}: trained on the other folds")
            torch.manual_seed(cfg.seed + 100 * r + k)
            ds.set_fold(f0, f1, k)
            member = eval(cfg.model.model)(logger=logger, cfg=cfg, dims_in=len(ds.channels),
                                           model_path=os.path.join(run_dir, name), device=device).to(device)
            member.init_dataloaders(ds)
            os.makedirs(os.path.join(run_dir, name), exist_ok=True)
            member.run_training()
            losses.append(getattr(member, "losses", None))
            torch.save(losses[-1], os.path.join(run_dir, name, "losses.pt"))
            member.load("final")
            p, l = {}, {}
            for s in ("trn", "val"):
                p[s], l[s] = member.evaluate(s)
                p[s], l[s] = p[s].double().cpu(), l[s].cpu()
            outputs["preds"].append(p)
            outputs["labels"].append(l)
            outputs["folds"].append((r, k))
            # val = [p0 fold-k rows, p1 fold-k rows] in that order -> the p0 part is out-of-fold
            n_val0 = len(ds.rows_p0["val"])
            wk = torch.exp(torch.clamp(p["val"][:n_val0], max=50))
            w_rep[r, ds.rows_p0["val"]] = wk / wk.mean()
    torch.save(outputs, os.path.join(run_dir, "classifier_outputs.pt"))
    w = w_rep.mean(dim=0)
    w = (w / w.mean()).numpy()
    np.save(os.path.join(run_dir, "weights_particle_to_parton.npy"), w)
    np.save(os.path.join(run_dir, "weights_particle_to_parton_members.npy"), w_rep.numpy())
    write_closure(run_dir, cfg, w, None, None, logger.info, kfold=n_folds, repeats=repeats,
                  out_of_fold=True)
    write_diagnostics(run_dir)
    write_classifier_plots(run_dir)
    logger.info(f"done -> {run_dir}")


def write_closure(run_dir, cfg, w, rows_val0, rows_val1, log=print, **extra_summary):
    """Closure plots (all events and val only, never trained on), pickle and summary.json."""
    from datasets.chain_reweighting import ChainReweightingDataset, _load_ptphietam

    paths = cfg.dataset.chain_reweighting
    # plots always use the real reco jet multiplicity (a reco observable, input or not)
    p0, p1 = _load_ptphietam(paths.path_p0, True), _load_ptphietam(paths.path_p1, True)
    # observables computed from the raw 21 columns (no derived-channel lookups); all four dphi_jj
    # definitions are kept (the two unsigned ones are identical by construction)
    stub = ChainReweightingDataset.__new__(ChainReweightingDataset)
    ChainReweightingDataset.init_observables(stub)
    entries, chi2 = build_entries(stub.observables, p0, p1, w)
    plot_observables(os.path.join(run_dir, "ttbar_reweighting.pdf"), entries)
    if rows_val0 is None:  # cross-fitted: every weight is already out-of-fold
        entries_val, chi2_val = None, None
    else:
        entries_val, chi2_val = build_entries(stub.observables, p0[rows_val0], p1[rows_val1],
                                              w[np.asarray(rows_val0)])
        plot_observables(os.path.join(run_dir, "ttbar_reweighting_val.pdf"), entries_val)
    pickle.dump({"all": entries, "val": entries_val}, open(os.path.join(run_dir, "ttbar_reweighting.pkl"), "wb"))

    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    q = np.quantile(w, [0.0, 0.01, 0.5, 0.99, 0.999, 1.0])
    summary = {
        "n_particle": int(len(p0)), "n_parton": int(len(p1)), **extra_summary,
        "n_eff": n_eff, "n_eff_fraction": n_eff / len(w),
        "weight_quantiles": dict(zip(["min", "q01", "median", "q99", "q999", "max"], q.tolist())),
        "chi2_ndf_all": chi2, "chi2_ndf_val": chi2_val,
    }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    log(f"N_eff = {n_eff:.0f} of {len(w)} ({n_eff / len(w):.1%}); weight quantiles "
        f"[0,1%,50%,99%,99.9%,max] = {np.round(q, 3).tolist()}")
    for name, c in (("all", chi2), ("val", chi2_val)) if chi2_val is not None else (("all, out-of-fold", chi2),):
        worst = sorted(c.items(), key=lambda z: -z[1])[:6]
        log(f"shape chi2/ndf after reweighting ({name}), worst: " + ", ".join(f"{k}={v:.2f}" for k, v in worst))


def write_diagnostics(run_dir):
    """train_metrics.pdf (per member: loss pages, then grad-norm pages, as in the classifier runs) and
    weights.pdf (log10 w per member + ensemble; per-event spread between members)."""
    import glob
    import unfolding.plots as P
    from matplotlib.backends.backend_pdf import PdfPages
    from classification.plots import BackgroundSubtractionPlots
    from datasets.dataset import Observable

    losses = [torch.load(f, weights_only=False)
              for f in sorted(glob.glob(os.path.join(run_dir, "model_*", "losses.pt")))]
    if losses:
        BackgroundSubtractionPlots.plot_train_metrics_ensemble(
            None, os.path.join(run_dir, "train_metrics.pdf"), losses, logy=False)

    wm = np.load(os.path.join(run_dir, "weights_particle_to_parton_members.npy"))  # (K, N)
    w = np.load(os.path.join(run_dir, "weights_particle_to_parton.npy"))
    floor = 1e-6
    obs_w = Observable(compute=None, bins=None, tex_label=r"\log_{10} w(x_{\text{reco}})", yscale="log")
    bins = np.linspace(-6, 2, 49)
    member_colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2"]
    lines = [P.Line(y=P.compute_hist_data(bins, np.log10(np.maximum(w, floor)))[0],
                    label="ensemble mean", color=PARTICLE_COLOR)]
    lines += [P.Line(y=P.compute_hist_data(bins, np.log10(np.maximum(wk, floor)))[0],
                     label=f"member {k + 1}", color=member_colors[k % len(member_colors)],
                     linestyle="dotted", alpha=0.8)
              for k, wk in enumerate(wm)]
    rel = wm.std(axis=0) / np.maximum(wm.mean(axis=0), floor)
    obs_rel = Observable(compute=None, bins=None, tex_label=r"\sigma_{\text{members}}(w)\,/\,\bar{w}", yscale="log")
    bins_rel = np.linspace(0, 2.5, 51)
    with PdfPages(os.path.join(run_dir, "weights.pdf")) as pp:
        P.hist_plot(pp, lines, bins, obs_w, show_ratios=False, no_scale=True, ylabel="Events",
                    title=r"$t\bar{t}$", title_on_axs=True)
        P.hist_plot(pp, [P.Line(y=P.compute_hist_data(bins_rel, np.minimum(rel, 2.499))[0],
                                label="per-event member spread", color=PARTICLE_COLOR)],
                    bins_rel, obs_rel, show_ratios=False, no_scale=True, ylabel="Events",
                    title=r"$t\bar{t}$", title_on_axs=True)
    print(f"wrote {run_dir}/train_metrics.pdf and weights.pdf")


def evaluate_members(run_dir, cfg, ds, logger):
    """GPU: every saved member's output on p0 (particle cuts) and p1 (parton cuts), trn and val
    splits, saved to classifier_outputs.pt so the classifier plots can be redone on CPU."""
    from classification.models import Classifier  # noqa: F401  (eval'd by cfg.model.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"labels": None, "preds": []}
    for k in range(cfg.train.ensemble):
        member = eval(cfg.model.model)(logger=logger, cfg=cfg, dims_in=len(ds.channels),
                                       model_path=os.path.join(run_dir, f"model_{k}"), device=device).to(device)
        member.init_dataloaders(ds)
        member.load("final")
        p, l = {}, {}
        for split in ("trn", "val"):
            p[split], l[split] = member.evaluate(split)
            p[split], l[split] = p[split].cpu(), l[split].cpu()
        out["preds"].append(p)
        out["labels"] = l
    torch.save(out, os.path.join(run_dir, "classifier_outputs.pt"))
    return out


def write_classifier_plots(run_dir):
    """CPU: class_{trn,val}.pdf with the standard BackgroundSubtractionPlots.plot_classifier pages
    (output histograms of p0 and p1, w(x), 1-w, ROC/AUC), one block per ensemble member."""
    import logging
    from matplotlib.backends.backend_pdf import PdfPages
    from classification.plots import BackgroundSubtractionPlots
    from datasets.chain_reweighting import ChainReweightingDataset

    out = torch.load(os.path.join(run_dir, "classifier_outputs.pt"), weights_only=False)
    per_member = isinstance(out["labels"], list)  # k-fold: each member has its own trn/val events
    stub = ChainReweightingDataset.__new__(ChainReweightingDataset)
    stub.channels = []
    stub.obs, stub.obs_p0, stub.obs_p1, stub.bins = {}, {}, {}, {}

    def _set_labels(labels):
        stub.labels = labels
        n0 = {s: int((y == 0).sum()) for s, y in labels.items()}
        n1 = {s: int((y == 1).sum()) for s, y in labels.items()}
        stub.w_p0 = {s: torch.ones(n0[s], dtype=torch.float64) for s in labels}  # uniform, as in training
        stub.w_p1 = {s: torch.ones(n1[s], dtype=torch.float64) for s in labels}
        none0 = {s: torch.zeros(n0[s], dtype=torch.bool) for s in labels}
        none1 = {s: torch.zeros(n1[s], dtype=torch.bool) for s in labels}
        stub.mask_sig_in_p0, stub.mask_ttbar_in_p0, stub.mask_WWjj_in_p0 = none0, none0, none0
        stub.mask_bkg_in_p0 = none0  # no process split here (the plot class checks bkg == ttbar ^ WWjj)
        stub.mask_ttbar_in_p1, stub.mask_WWjj_in_p1 = none1, none1

    logger = logging.getLogger("ttbar-chain")
    splits = out["labels"][0].keys() if per_member else out["labels"].keys()
    for split in splits:
        with PdfPages(os.path.join(run_dir, f"class_{split}.pdf")) as pp:
            for k, p in enumerate(out["preds"]):
                _set_labels(out["labels"][k] if per_member else out["labels"])
                stub.predictions = {s: v.clone() for s, v in p.items()}
                stub.compute_weights()
                BackgroundSubtractionPlots(logger, stub, model_name=f"member {k + 1}", title=r"$t\bar{t}$",
                                           show_class_components={"p0": False, "p1": False},
                                           ).plot_classifier(pp, split=split)
    print(f"wrote {run_dir}/class_trn.pdf and class_val.pdf")


def evaluate_run(run_dir):
    """--evaluate (GPU): re-evaluate the saved members, then the classifier plots."""
    from classification.main import init_logger
    from datasets.chain_reweighting import ChainReweightingDataset

    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    logger = init_logger(run_dir)
    ds = ChainReweightingDataset(logger, cfg.dataset)
    evaluate_members(run_dir, cfg, ds, logger)
    write_classifier_plots(run_dir)


def rebuild(run_dir):
    """CPU only: redo histograms/plots/summary from the saved weights (no training, no GPU). The val
    rows are regenerated with the dataset's own seeded shuffle + split."""
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    paths = cfg.dataset.chain_reweighting
    if paths.get("kfold", None):
        w = np.load(os.path.join(run_dir, "weights_particle_to_parton.npy"))
        summary = json.load(open(os.path.join(run_dir, "summary.json")))
        keep = {k: summary[k] for k in ("kfold", "repeats", "out_of_fold") if k in summary}
        write_closure(run_dir, cfg, w, None, None, **keep)
        write_diagnostics(run_dir)
        write_classifier_plots(run_dir)
        print(f"rebuilt {run_dir}")
        return
    n0, n1 = np.load(paths.path_p0, mmap_mode="r").shape[0], np.load(paths.path_p1, mmap_mode="r").shape[0]
    g = torch.Generator().manual_seed(int(paths.get("seed", 0)))
    perm0, perm1 = torch.randperm(n0, generator=g), torch.randperm(n1, generator=g)
    val = lambda perm, n: perm[-int(n * cfg.dataset.trn_tst_val[2]):]   # = _split_data's "val"
    w = np.load(os.path.join(run_dir, "weights_particle_to_parton.npy"))
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    keep = {k: summary[k] for k in ("ensemble", "member_weight_corr_min") if k in summary}
    write_closure(run_dir, cfg, w, val(perm0, n0), val(perm1, n1), **keep)
    write_diagnostics(run_dir)
    print(f"rebuilt {run_dir}")


def plot_heavy_weights(run_dir, thresholds=(10.0, 30.0)):
    """CPU: ttbar_reweighting_heavy_w.pdf = the production plot plus, for each threshold, a twin of the
    reweighted line restricted to events with w > threshold, on the SAME normalization as the full
    reweighted line, so its height is the share of the reweighted ttbar carried by those events."""
    import unfolding.plots as P
    from matplotlib.backends.backend_pdf import PdfPages
    from datasets.chain_reweighting import ChainReweightingDataset, _load_ptphietam

    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    paths = cfg.dataset.chain_reweighting
    # plots always use the real reco jet multiplicity (a reco observable, input or not)
    p0, p1 = _load_ptphietam(paths.path_p0, True), _load_ptphietam(paths.path_p1, True)
    w = np.load(os.path.join(run_dir, "weights_particle_to_parton.npy"))
    stub = ChainReweightingDataset.__new__(ChainReweightingDataset)
    ChainReweightingDataset.init_observables(stub)
    styles = ["dashed", "dotted", "dashdot"]
    for thr in thresholds:
        m = w > thr
        print(f"  w > {thr:g}: {m.sum()} events ({m.mean():.4%}), {w[m].sum() / w.sum():.2%} of the total weight")
    with PdfPages(os.path.join(run_dir, "ttbar_reweighting_heavy_w.pdf")) as pp:
        for obs in stub.observables:
            x0, x1 = np.asarray(obs.compute(p0)), np.asarray(obs.compute(p1))
            bins = np.asarray(obs.bins(torch.as_tensor(x1)))
            bw = np.diff(bins)
            y1, e1 = P.compute_hist_data(bins, x1)
            y0, e0 = P.compute_hist_data(bins, x0)
            yw, ew = P.compute_hist_data(bins, x0, weights=w)
            n1, n0, nw = (np.sum(bw * y) for y in (y1, y0, yw))
            lines = [
                P.Line(y=y1 / n1, y_err=e1 / n1, label=LABEL_PARTON, color=PARTON_COLOR),
                P.Line(y=y0 / n0, y_err=e0 / n0, y_ref=y1 / n1, label=LABEL_PARTICLE, color=PARTICLE_COLOR,
                       linestyle="dotted"),
                P.Line(y=yw / nw, y_err=ew / nw, y_ref=y1 / n1, label=LABEL_REWEIGHTED, color=PARTICLE_COLOR),
            ]
            for thr, ls in zip(thresholds, styles):
                yh, eh = P.compute_hist_data(bins, x0, weights=np.where(w > thr, w, 0.0))
                lines.append(P.Line(y=yh / nw, y_err=eh / nw, label=rf"$w(x_{{\text{{reco}}}}) > {thr:g}$",
                                    color=PARTICLE_COLOR, linestyle=ls, alpha=0.7))
            P.hist_plot(pp, lines, bins, obs, show_ratios=True, model_name="Rew.", truth_name="Parton",
                        title=r"$t\bar{t}$", title_on_axs=True, no_scale=True, ylabel="Normalized")
    print(f"wrote {run_dir}/ttbar_reweighting_heavy_w.pdf")


def replot(run_dir):
    import importlib
    import unfolding.plots as P

    importlib.reload(P)
    d = pickle.load(open(os.path.join(run_dir, "ttbar_reweighting.pkl"), "rb"))
    plot_observables(os.path.join(run_dir, "ttbar_reweighting.pdf"), d["all"])
    if d.get("val") is not None:
        plot_observables(os.path.join(run_dir, "ttbar_reweighting_val.pdf"), d["val"])
    print(f"replotted {run_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--replot":
        replot(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == "--rebuild":
        rebuild(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == "--heavy-weights":
        plot_heavy_weights(sys.argv[2], tuple(float(x) for x in sys.argv[3:]) or (10.0, 30.0))
    elif len(sys.argv) > 2 and sys.argv[1] == "--evaluate":
        evaluate_run(sys.argv[2])
    else:
        main(sys.argv[1:])
