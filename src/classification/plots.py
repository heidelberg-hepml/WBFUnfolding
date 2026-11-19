import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from contextlib import nullcontext
from matplotlib.ticker import ScalarFormatter
from matplotlib.backends.backend_pdf import PdfPages
from datasets.dataset import Observable
from typing import Optional
from sklearn.metrics import roc_curve, roc_auc_score
# Shared with the unfolding plots so both produce identically formatted histograms.
from unfolding.plots import Line, hist_line, corner_text, hist_plot as _unfolding_hist_plot  # noqa: F401 (re-exported)

P1_COLOR = "#07078A"
DATA_COLOR = "#17BECF"
BKG_MC_COLOR = "#EF15C3"

P0_COLOR = "#015D14"
BKG_COLOR = "#634E2E"

REW_COLOR = "#8A0707"

# Legend labels of the clean-view reweighting lines. LEGEND_LAYOUTS and the replot scripts refer to
# these, so edit the text HERE (not at the Line) to keep the per-page legend layouts working.
LABEL_W = r"$w(x) \cdot p_{\text{data}}$"
LABEL_SIG = r"$p_{\mathrm{signal}}$"
LABEL_1MW = r"$[1-w(x)]\cdot p_{\text{data}}$"

HWW_COLOR = "#2CA02C"
TTBAR_COLOR = "#066E74"
WWJJ_COLOR = "#9467BD"


class BackgroundSubtractionPlots:
    """
    Implements the plotting pipeline to evaluate the performance of
    conditional generative networks.
    Args:
        logger: Logger object
        observables: List of observables
        losses: Dictionary with loss terms and learning rate as a function of the epoch
        x_part: True data at particle level
        x_reco: True data at reconstruction level
        x_gen: Generated particle level data
        x_part_ppd: Preprocessed particle level data
        x_reco_ppd: Preprocessed reconstruction level data
        debug: If True, additional debug information is printed
    """

    def __init__(
        self,
        logger,
        dataset,
        losses: dict = None,
        debug: bool = False,
        model_name: Optional[str] = "Classifier",
        title: Optional[str] = None,
        show_class_components: Optional[dict] = None,
    ):
        """
        Initializes the Plots object.
        Args:
            logger: Logger object
            dataset: Dataset object containing the data and observables
            losses: Dictionary with loss terms and learning rate as a function of the epoch
            debug: If True, additional debug information is printed
            model_name: Name of the model to be used in plot titles and labels
            title: Optional title for the plots
            show_class_components: Whether to show the contributions of different classes (signal, ttbar, WWjj) in the classifier output plots
        """
        self.logger = logger
        self.dataset = dataset
        self.losses = losses
        self.debug = debug
        self.model_name = model_name
        self.title = title if title is not None else None
        self.show_class_components = show_class_components

        self.channels = dataset.channels
        self.obs = dataset.obs

        self.predictions = dataset.predictions
        self.labels = {}
        self.prior_weights_p0 = {}
        self.prior_weights_p1 = {}
        self.class_w_p0 = {}
        self.class_w_p1 = {}
        self.class_w_raw_p0 = {}
        self.class_w_raw_p1 = {}

        self.norm_factor_p0 = {}
        self.norm_factor_p1 = {}
        if self.show_class_components:
            self.mask_sig_in_p0 = {}
            self.mask_bkg_in_p0 = {}
            self.mask_ttbar_in_p0 = {}
            self.mask_WWjj_in_p0 = {}
            self.mask_ttbar_in_p1 = {}
            self.mask_WWjj_in_p1 = {}

        for split in dataset.labels.keys():

            # Prior weights that follow fractions expected in fiducial region
            self.prior_weights_p0[split] = dataset.w_p0[split].cpu().numpy()
            self.prior_weights_p1[split] = dataset.w_p1[split].cpu().numpy()

            # Number of events in p0 and p1 for the given split
            N_p0 = len(dataset.labels[split][dataset.labels[split] == 0])
            N_p1 = len(dataset.labels[split][dataset.labels[split] == 1])

            # Obtain the norm factor to get the correct cross section normalization for the histograms (sum of weights should equal number of events in each class)
            self.norm_factor_p0[split] = N_p0 / np.sum(self.prior_weights_p0[split])
            self.norm_factor_p1[split] = N_p1 / np.sum(self.prior_weights_p1[split])

            self.predictions[split] = self.predictions[split].cpu().numpy()
            self.labels[split] = dataset.labels[split].cpu().numpy()
            mask_p0 = dataset.labels[split] == 0
            mask_p1 = dataset.labels[split] == 1
            if dataset.weights[split].ndim == 2:
                self.class_w_p0[split] = dataset.weights[split][:, mask_p0].cpu().numpy()
                self.class_w_p1[split] = dataset.weights[split][:, mask_p1].cpu().numpy()
            else:
                self.class_w_p0[split] = dataset.weights[split][mask_p0].cpu().numpy()
                self.class_w_p1[split] = dataset.weights[split][mask_p1].cpu().numpy()
            if hasattr(dataset, "density_ratio"):
                dr = dataset.density_ratio[split]
                if dr.ndim == 2:
                    self.class_w_raw_p0[split] = dr[:, mask_p0].cpu().numpy()
                    self.class_w_raw_p1[split] = dr[:, mask_p1].cpu().numpy()
                else:
                    self.class_w_raw_p0[split] = dr[mask_p0].cpu().numpy()
                    self.class_w_raw_p1[split] = dr[mask_p1].cpu().numpy()
            else:
                ratio_inv = (self.prior_weights_p0[split].sum()
                             / self.prior_weights_p1[split].sum())
                self.class_w_raw_p0[split] = self.class_w_p0[split] * ratio_inv
                self.class_w_raw_p1[split] = self.class_w_p1[split] * ratio_inv
            self.mask_sig_in_p0[split] = dataset.mask_sig_in_p0[split].cpu().numpy()
            self.mask_bkg_in_p0[split] = dataset.mask_bkg_in_p0[split].cpu().numpy()
            self.mask_ttbar_in_p0[split] = dataset.mask_ttbar_in_p0[split].cpu().numpy()
            self.mask_WWjj_in_p0[split] = dataset.mask_WWjj_in_p0[split].cpu().numpy()

            self.mask_ttbar_in_p1[split] = dataset.mask_ttbar_in_p1[split].cpu().numpy()
            self.mask_WWjj_in_p1[split] = dataset.mask_WWjj_in_p1[split].cpu().numpy()
            # check that (WWjj | ttbar) == bkg in p0
            assert np.all(
                self.mask_bkg_in_p0[split]
                == (self.mask_ttbar_in_p0[split] ^ self.mask_WWjj_in_p0[split])
            ), "Background in p0 should be ttbar or WWjj"

            self.logger.info(
                f"    In p0 {split} split: {np.sum(self.mask_sig_in_p0[split])} signal events, {np.sum(self.mask_bkg_in_p0[split])} background events (ttbar: {np.sum(self.mask_ttbar_in_p0[split])}, WWjj: {np.sum(self.mask_WWjj_in_p0[split])})."
            )
            self.logger.info(
                f"    In p1 {split} split: {len(self.mask_ttbar_in_p1[split])} MC background events (ttbar: {np.sum(self.mask_ttbar_in_p1[split])}, WWjj: {np.sum(self.mask_WWjj_in_p1[split])})."
            )

        self.obs_p0 = dataset.obs_p0
        self.obs_p1 = dataset.obs_p1
        self.has_ppd = False
        if hasattr(dataset, "obs_p0_ppd") and hasattr(dataset, "obs_p1_ppd"):
            self.obs_p0_ppd = dataset.obs_p0_ppd
            self.obs_p1_ppd = dataset.obs_p1_ppd
            self.has_ppd = True
        # full standardized tensors (per split) -> lets the ppd plot cover EVERY input channel,
        # including ones without a named observable, straight from what the network ingests.
        self.p0_ppd_full = getattr(dataset, "p0_ppd", None)
        self.p1_ppd_full = getattr(dataset, "p1_ppd", None)
        self.bins = dataset.bins

        self.has_refiner = hasattr(dataset, "refiner_weights")
        self.refiner_w_p0 = {}
        self.refiner_preds_p0 = {}
        self.refiner_labels_p0 = {}
        if self.has_refiner:
            for split in dataset.labels.keys():
                self.refiner_w_p0[split] = dataset.refiner_weights[split].cpu().numpy()
                self.refiner_preds_p0[split] = dataset.refiner_predictions[split].cpu().numpy()
                self.refiner_labels_p0[split] = dataset.refiner_y[split].cpu().numpy()

    # same style as unfolding/plots.py Plots, so both produce identically formatted plots
    plt.rc("font", family="serif", size=16)
    plt.rc("font", serif="Charter")
    plt.rc("axes", titlesize="medium")
    plt.rc("text.latex", preamble=r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}")
    plt.rc("text", usetex=True)
    plt.rcParams["axes.formatter.limits"] = [-2, 2.99]

    def plot_train_metrics(
        self,
        file: str,
        labels=None,
        logy=False,
        ylabel="Loss",
        rect=(0.2, 0.2, 0.83, 0.9),
    ):
        with PdfPages(file) as pp:
            iterations = range(1, len(self.losses["trn"]) + 1)
            lr = self.losses.get("lr", None)
            labels = ["Train", "Validation"] if labels is None else labels
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for loss, label in zip([self.losses["trn"], self.losses["val"]], labels):
                if len(loss) == len(iterations):
                    nb_of_train_losses_per_epoch = len(self.losses["trn"]) // len(
                        self.losses["val"]
                    )
                    # skip first 2 epochs
                    its = iterations[2 * nb_of_train_losses_per_epoch :]
                    loss = loss[2 * nb_of_train_losses_per_epoch :]
                else:
                    frac = len(self.losses["trn"]) / len(loss)
                    its = np.arange(1, len(loss) + 1) * frac
                    # skip first 2 epochs
                    loss = loss[2:]
                    its = its[2:]
                ax.plot(its, loss, label=label)
            if logy:
                ax.set_yscale("log")
            if lr is not None:
                axright = ax.twinx()
                axright.plot(iterations, lr, label="Learning Rate", color="crimson")
                axright.set_ylabel("Learning rate")
                axright.spines["right"].set_color("crimson")
                axright.yaxis.label.set_color("crimson")
                axright.tick_params(axis="y", colors="crimson")
            ax.set_xlabel("Number of iterations")
            ax.set_ylabel(ylabel)
            ax.xaxis.get_major_formatter().set_useOffset(False)
            ax.xaxis.get_major_formatter().set_scientific(False)
            ax.legend(
                ax.get_legend_handles_labels()[0]
                + axright.get_legend_handles_labels()[0],
                ax.get_legend_handles_labels()[1]
                + axright.get_legend_handles_labels()[1],
                frameon=False,
                loc="upper right",
            )
            fig.subplots_adjust(
                left=rect[0], bottom=rect[1], right=rect[2], top=rect[3]
            )
            fig.savefig(pp, format="pdf")
            plt.close()

            if "grad_norm" in self.losses:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                iterations = range(1, len(self.losses["grad_norm"]) + 1)
                ax.plot(
                    iterations,
                    self.losses["grad_norm"],
                    color="firebrick",
                    label="Gradient norm",
                )
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.get_major_formatter().set_useOffset(False)
                ax.xaxis.get_major_formatter().set_scientific(False)
                ax.set_xlabel("Number of iterations")
                ax.set_ylabel("Gradient norm")
                ax.set_yscale("log")
                ax.legend(frameon=False, loc="upper right")
                fig.subplots_adjust(
                    left=rect[0], bottom=rect[1], right=rect[2], top=rect[3]
                )
                fig.savefig(pp, format="pdf")
                plt.close()

            if "kl" in self.losses and "bce" in self.losses:
                iterations = range(1, len(self.losses["bce"]) + 1)
                fig, ax = plt.subplots(figsize=(6, 4.5))
                ax.plot(iterations, self.losses["bce"], label="BCE", color="steelblue")
                ax.plot(iterations, self.losses["kl"], label="KL", color="darkorange")
                ax.set_xlabel("Number of iterations")
                ax.set_ylabel("Loss")
                ax.set_yscale("log")
                ax.legend(frameon=False)
                fig.subplots_adjust(
                    left=rect[0], bottom=rect[1], right=rect[2], top=rect[3]
                )
                fig.savefig(pp, format="pdf")
                plt.close()

    def plot_train_metrics_ensemble(
        self,
        file,
        losses_list,
        labels=None,
        logy=False,
        ylabel="Loss",
        rect=(0.2, 0.2, 0.83, 0.9),
    ):
        """Ensemble train metrics: one loss page per member, then one grad-norm page per member.
        For K members the page order is loss_1..loss_K, grad_1..grad_K. losses_list is a list of
        per-member loss dicts (same shape as model.losses); None entries (e.g. re-plot runs) skip."""
        labels = ["Train", "Validation"] if labels is None else labels
        with PdfPages(file) as pp:
            # --- loss pages (one per member) ---
            for m, losses in enumerate(losses_list):
                if losses is None or "trn" not in losses or "val" not in losses:
                    continue
                iterations = range(1, len(losses["trn"]) + 1)
                lr = losses.get("lr", None)
                fig, ax = plt.subplots(figsize=(6, 4.5))
                for loss, label in zip([losses["trn"], losses["val"]], labels):
                    if len(loss) == len(iterations):
                        nb = max(len(losses["trn"]) // len(losses["val"]), 1)
                        its = iterations[2 * nb:]
                        loss = loss[2 * nb:]
                    else:
                        frac = len(losses["trn"]) / len(loss)
                        its = np.arange(1, len(loss) + 1) * frac
                        loss = loss[2:]
                        its = its[2:]
                    ax.plot(its, loss, label=label)
                if logy:
                    ax.set_yscale("log")
                handles, lbls = ax.get_legend_handles_labels()
                if lr is not None:
                    axright = ax.twinx()
                    axright.plot(iterations, lr, label="Learning Rate", color="crimson")
                    axright.set_ylabel("Learning rate")
                    axright.spines["right"].set_color("crimson")
                    axright.yaxis.label.set_color("crimson")
                    axright.tick_params(axis="y", colors="crimson")
                    h2, l2 = axright.get_legend_handles_labels()
                    handles, lbls = handles + h2, lbls + l2
                ax.set_xlabel("Number of iterations")
                ax.set_ylabel(ylabel)
                ax.set_title(f"Member {m + 1}", loc="right")
                ax.xaxis.get_major_formatter().set_useOffset(False)
                ax.xaxis.get_major_formatter().set_scientific(False)
                ax.legend(handles, lbls, frameon=False, loc="upper right")
                fig.subplots_adjust(left=rect[0], bottom=rect[1], right=rect[2], top=rect[3])
                fig.savefig(pp, format="pdf")
                plt.close()
            # --- grad-norm pages (one per member) ---
            for m, losses in enumerate(losses_list):
                if losses is None or "grad_norm" not in losses:
                    continue
                fig, ax = plt.subplots(figsize=(6, 4.5))
                iterations = range(1, len(losses["grad_norm"]) + 1)
                ax.plot(iterations, losses["grad_norm"], color="firebrick", label="Gradient norm")
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.get_major_formatter().set_useOffset(False)
                ax.xaxis.get_major_formatter().set_scientific(False)
                ax.set_xlabel("Number of iterations")
                ax.set_ylabel("Gradient norm")
                ax.set_yscale("log")
                ax.set_title(f"Member {m + 1}", loc="right")
                ax.legend(frameon=False, loc="upper right")
                fig.subplots_adjust(left=rect[0], bottom=rect[1], right=rect[2], top=rect[3])
                fig.savefig(pp, format="pdf")
                plt.close()

    def plot_classifier(
        self, file_or_pp, pickle_file: Optional[str] = None, split: str = "tst"
    ):
        """
        Makes histograms of classifier weights / output.
        Args:
            file_or_pp: Path to the output PDF file, or a pre-opened PdfPages object
                        (used by ensemble mode to append multiple members into one file).
            pickle_file: Path to the output pickle file (optional)
        """
        pickle_data = []
        ctx = PdfPages(file_or_pp) if isinstance(file_or_pp, str) else nullcontext(file_or_pp)
        with ctx as pp:
            # 1 - w  (raw density ratio r = p1/p0; negative where r > 1)
            one_minus_raw_p0 = 1 - self.class_w_raw_p0[split]
            one_minus_raw_p1 = 1 - self.class_w_raw_p1[split]
            all_1mw = np.concatenate([one_minus_raw_p0.ravel(), one_minus_raw_p1.ravel()])
            lo_1mw, hi_1mw = np.percentile(all_1mw, 1), np.percentile(all_1mw, 99)
            bins = np.linspace(lo_1mw, hi_1mw, 51)
            y_p0, y_p0_err = compute_hist_data(
                bins, one_minus_raw_p0, weights=self.prior_weights_p0[split]
            )
            y_p1, y_p1_err = compute_hist_data(
                bins, one_minus_raw_p1, weights=self.prior_weights_p1[split]
            )
            lines = [
                Line(y=y_p0, y_err=y_p0_err, label="$p_0$", color=P0_COLOR),
                Line(y=y_p1, y_err=y_p1_err, label="$p_1$", color=P1_COLOR),
            ]
            obs_one_minus_w = Observable(
                compute=None,
                tex_label=r"1 - w(x),\; w = p_1/p_0",
                bins=None,
            )
            hist_plot(
                pp, lines, bins, obs_one_minus_w, show_ratios=False, ylabel="Events", yscale="log",
            )
            # w (raw density ratio, log scale)
            raw_p0 = self.class_w_raw_p0[split]
            raw_p1 = self.class_w_raw_p1[split]
            wmin = max(min(float(np.min(raw_p0)), float(np.min(raw_p1))), 1e-12)
            wmax = max(float(np.max(raw_p0)), float(np.max(raw_p1)))
            wmax = max(wmax, wmin * (1.0 + 1e-6))
            bins = np.logspace(np.log10(wmin), np.log10(wmax), 51)
            y_p0, y_p0_err = compute_hist_data(
                bins, raw_p0, weights=self.prior_weights_p0[split]
            )
            y_p1, y_p1_err = compute_hist_data(
                bins, raw_p1, weights=self.prior_weights_p1[split]
            )
            lines = [
                Line(y=y_p0, y_err=y_p0_err, label="$p_0$", color=P0_COLOR),
                Line(y=y_p1, y_err=y_p1_err, label="$p_1$", color=P1_COLOR),
            ]
            obs_w_log = Observable(
                compute=None,
                tex_label=r"w(x) = p_1/p_0",
                bins=None,
                xscale="log",
                yscale="log",
            )
            hist_plot(pp, lines, bins, obs_w_log, show_ratios=False, ylabel="Events")
            # 1 - w_sc  (scaled physical weight w_sc = w * W_bkg/W_data; should be <= 1)
            wsc_p0 = self.class_w_p0[split]
            wsc_p1 = self.class_w_p1[split]
            wsc_1d_p0 = wsc_p0 if wsc_p0.ndim == 1 else wsc_p0.mean(axis=0)
            frac_neg = 100.0 * float(np.mean(wsc_1d_p0 > 1))
            one_minus_wsc_p0 = 1 - wsc_p0
            one_minus_wsc_p1 = 1 - wsc_p1
            all_1mwsc = np.concatenate([one_minus_wsc_p0.ravel(), one_minus_wsc_p1.ravel()])
            lo_sc, hi_sc = np.percentile(all_1mwsc, 1), np.percentile(all_1mwsc, 99)
            bins_sc = np.linspace(lo_sc, hi_sc, 51)
            y_p0sc, y_p0sc_err = compute_hist_data(
                bins_sc, one_minus_wsc_p0, weights=self.prior_weights_p0[split]
            )
            y_p1sc, y_p1sc_err = compute_hist_data(
                bins_sc, one_minus_wsc_p1, weights=self.prior_weights_p1[split]
            )
            lines = [
                Line(y=y_p0sc, y_err=y_p0sc_err, label="$p_0$", color=P0_COLOR),
                Line(y=y_p1sc, y_err=y_p1sc_err, label="$p_1$", color=P1_COLOR),
            ]
            obs_1mwsc = Observable(
                compute=None,
                tex_label=r"1 - w_{\text{sc}}(x),\; w_{\text{sc}} = w\cdot W_{bkg}/W_{data}",
                bins=None,
            )
            hist_plot(
                pp, lines, bins_sc, obs_1mwsc, show_ratios=False, ylabel="Events", yscale="log",
                title=f"{frac_neg:.1f}\\% of $p_0$ events have $w_{{sc}}>1$",
            )
            # w_sc (log scale)
            wsc_vals_p0 = wsc_1d_p0
            wsc_vals_p1 = wsc_p1 if wsc_p1.ndim == 1 else wsc_p1.mean(axis=0)
            wscmin = max(min(float(np.min(wsc_vals_p0)), float(np.min(wsc_vals_p1))), 1e-12)
            wscmax = max(float(np.max(wsc_vals_p0)), float(np.max(wsc_vals_p1)))
            wscmax = max(wscmax, wscmin * (1.0 + 1e-6))
            bins_wsc = np.logspace(np.log10(wscmin), np.log10(wscmax), 51)
            y_p0sc2, y_p0sc2_err = compute_hist_data(
                bins_wsc, wsc_vals_p0, weights=self.prior_weights_p0[split]
            )
            y_p1sc2, y_p1sc2_err = compute_hist_data(
                bins_wsc, wsc_vals_p1, weights=self.prior_weights_p1[split]
            )
            lines = [
                Line(y=y_p0sc2, y_err=y_p0sc2_err, label="$p_0$", color=P0_COLOR),
                Line(y=y_p1sc2, y_err=y_p1sc2_err, label="$p_1$", color=P1_COLOR),
            ]
            obs_wsc_log = Observable(
                compute=None,
                tex_label=r"w_{\text{sc}}(x) = w\cdot W_{bkg}/W_{data}",
                bins=None,
                xscale="log",
                yscale="log",
            )
            hist_plot(pp, lines, bins_wsc, obs_wsc_log, show_ratios=False, ylabel="Events")
            # Plot classifier discriminator output
            bins_disc = np.linspace(
                np.min(self.predictions[split]), np.max(self.predictions[split]), 51
            )
            if self.predictions[split].ndim == 2:
                pred_p0 = self.predictions[split][:, self.labels[split] == 0]
                pred_p1 = self.predictions[split][:, self.labels[split] == 1]
            else:
                pred_p0 = self.predictions[split][self.labels[split] == 0]
                pred_p1 = self.predictions[split][self.labels[split] == 1]

            y_disc_p0, y_disc_p0_err = compute_hist_data(
                bins_disc, pred_p0, weights=self.prior_weights_p0[split]
            )
            y_disc_p1, y_disc_p1_err = compute_hist_data(
                bins_disc, pred_p1, weights=self.prior_weights_p1[split]
            )
            lines = [
                Line(y=y_disc_p0, y_err=y_disc_p0_err, label="$p_0$", color=P0_COLOR),
                Line(y=y_disc_p1, y_err=y_disc_p1_err, label="$p_1$", color=P1_COLOR),
            ]
            obs_disc = Observable(
                compute=None,
                tex_label=r"\text{Classifier output}",
                bins=None,
                yscale="log",
            )
            hist_plot(
                pp, lines, bins_disc, obs_disc, show_ratios=False, ylabel="Events"
            )
            # ROC curve as final page
            self._add_roc_page(pp, split)

    def _add_roc_page(self, pp, split):
        y_true = self.labels[split]
        y_score = self.predictions[split]
        if y_score.ndim == 2:
            y_score = y_score.mean(axis=0)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        self.logger.info(f"    AUC ({split}): {auc:.4f}")
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.plot(fpr, tpr, color=REW_COLOR, lw=1.5, label=f"{self.model_name} (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", label="Random")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.legend(frameon=False, loc="lower right")
        if self.title is not None:
            corner_text(ax, self.title, "left", "top")
        fig.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.95)
        fig.savefig(pp, format="pdf")
        plt.close()

    def plot_roc(self, file: str, split: str = "tst"):
        """
        Plots ROC curve and computes AUC for the classifier.
        Args:
            file: Path to the output PDF file
            split: Data split to use
        """
        y_true = self.labels[split]
        y_score = self.predictions[split]  # logits
        auc = roc_auc_score(y_true, y_score)
        self.logger.info(f"    AUC ({split}): {auc:.4f}")

        with PdfPages(file) as pp:
            self._add_roc_page(pp, split)

    def plot_refiner(self, file: str, split: str = "tst"):
        """
        Plot refiner output: refinement factors exp(-r(x)) and ROC curve.
        Only active when the Neural Refiner (NRSW) was trained.
        """
        if not self.has_refiner:
            self.logger.warning("No refiner data found; skipping plot_refiner.")
            return
        with PdfPages(file) as pp:
            mask_pos = self.refiner_labels_p0[split] == 1  # label=1 when 1-w > 0
            mask_pp = self.refiner_labels_p0[split] == 0   # label=0 when 1-w < 0
            pw_pos = self.prior_weights_p0[split][mask_pos]
            pw_pp = self.prior_weights_p0[split][mask_pp]

            # Page 1: exp(-f), classical density-ratio intermediate
            rp = self.refiner_preds_p0[split]
            rf = np.exp(-rp)
            lo, hi = np.percentile(rf, 1), np.percentile(rf, 99)
            bins = np.linspace(max(lo, 0), hi, 51)
            y_pos, y_pos_err = compute_hist_data(bins, rf[mask_pos], weights=pw_pos)
            y_pp, y_pp_err = compute_hist_data(bins, rf[mask_pp], weights=pw_pp)
            lines = [
                Line(y=y_pos, y_err=y_pos_err, label=r"Positive $(1-w>0)$", color=HWW_COLOR),
                Line(y=y_pp, y_err=y_pp_err, label=r"Pseudo-pos.\ $(1-w<0)$", color=BKG_COLOR),
            ]
            hist_plot(
                pp, lines, bins,
                Observable(compute=None, tex_label=r"e^{-f_\theta(x)}", bins=None),
                show_ratios=False, ylabel="Events", yscale="log",
            )

            # Page 2: logit f(x)
            lo2, hi2 = np.percentile(rp, 1), np.percentile(rp, 99)
            bins2 = np.linspace(lo2, hi2, 51)
            y_pos2, y_pos2_err = compute_hist_data(bins2, rp[mask_pos], weights=pw_pos)
            y_pp2, y_pp2_err = compute_hist_data(bins2, rp[mask_pp], weights=pw_pp)
            lines2 = [
                Line(y=y_pos2, y_err=y_pos2_err, label=r"Positive $(1-w>0)$", color=HWW_COLOR),
                Line(y=y_pp2, y_err=y_pp2_err, label=r"Pseudo-pos.\ $(1-w<0)$", color=BKG_COLOR),
            ]
            hist_plot(
                pp, lines2, bins2,
                Observable(compute=None, tex_label=r"f_\theta(x)", bins=None),
                show_ratios=False, ylabel="Events", yscale="log",
            )

            # Page 3: r = tanh(f/2) = refiner_w_p0
            rr = self.refiner_w_p0[split]
            lo3, hi3 = np.percentile(rr, 1), np.percentile(rr, 99)
            bins3 = np.linspace(lo3, hi3, 51)
            y_pos3, y_pos3_err = compute_hist_data(bins3, rr[mask_pos], weights=pw_pos)
            y_pp3, y_pp3_err = compute_hist_data(bins3, rr[mask_pp], weights=pw_pp)
            lines3 = [
                Line(y=y_pos3, y_err=y_pos3_err, label=r"Positive $(1-w>0)$", color=HWW_COLOR),
                Line(y=y_pp3, y_err=y_pp3_err, label=r"Pseudo-pos.\ $(1-w<0)$", color=BKG_COLOR),
            ]
            hist_plot(
                pp, lines3, bins3,
                Observable(compute=None,
                           tex_label=r"r_\theta(x) = \tanh(f_\theta(x)/2)", bins=None),
                show_ratios=False, ylabel="Events", yscale="log",
            )

            # Page 4: |1-w| * r
            w_cls = self.class_w_p0[split]
            if w_cls.ndim == 2:
                w_cls = w_cls.mean(axis=0)
            refw = np.abs(1.0 - w_cls) * rr
            n_neg_before = int(((1.0 - w_cls) < 0).sum())
            n_neg_after = int((refw < 0).sum())
            pct_red = 100.0 * (n_neg_before - n_neg_after) / max(n_neg_before, 1)
            self.logger.info(
                f"    Negative events ({split}): before={n_neg_before}, "
                f"after={n_neg_after} ({pct_red:.1f}% reduction)"
            )
            lo4, hi4 = np.percentile(refw, 1), np.percentile(refw, 99)
            bins4 = np.linspace(lo4, hi4, 51)
            y_pos4, y_pos4_err = compute_hist_data(bins4, refw[mask_pos], weights=pw_pos)
            y_pp4, y_pp4_err = compute_hist_data(bins4, refw[mask_pp], weights=pw_pp)
            lines4 = [
                Line(y=y_pos4, y_err=y_pos4_err, label=r"Positive $(1-w>0)$", color=HWW_COLOR),
                Line(y=y_pp4, y_err=y_pp4_err, label=r"Pseudo-pos.\ $(1-w<0)$", color=BKG_COLOR),
            ]
            hist_plot(
                pp, lines4, bins4,
                Observable(compute=None,
                           tex_label=r"|1-w(x)|\cdot r_\theta(x)", bins=None),
                show_ratios=False, ylabel="Events",
                title=f"{pct_red:.1f}\\% less negative events",
            )

            # Page 5: ROC curve
            rl = self.refiner_labels_p0[split]
            fpr, tpr, _ = roc_curve(rl, rp)
            auc = roc_auc_score(rl, rp)
            self.logger.info(f"    Refiner AUC ({split}): {auc:.4f}")
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.plot(fpr, tpr, color=REW_COLOR, lw=1.5, label=f"Refiner (AUC = {auc:.4f})")
            ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", label="Random")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.legend(frameon=False, loc="lower right")
            if self.title is not None:
                corner_text(ax, self.title, "left", "top")
            fig.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.95)
            fig.savefig(pp, format="pdf")
            plt.close()

    def plot_observables(
        self, file: str, pickle_file: Optional[str] = None, split: str = "tst"
    ):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Path to the output PDF file
            pickle_file: Path to the output pickle file (optional)
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for i, (obs, bins, data_p0, data_p1) in enumerate(
                zip(
                    self.obs[split],
                    self.bins[split],
                    self.obs_p0[split],
                    self.obs_p1[split],
                )
            ):
                # if obs.channel is not None and obs.channel not in self.channels:
                #     # avoid plotting primary (obs.channel != None) channels not used for unfolding
                #     # secondary channels (derived observables -- obs.channel == None) are always plotted
                #     continue
                y_p0, y_p0_err = compute_hist_data(
                    bins,
                    data_p0,
                    weights=(self.prior_weights_p0[split]) * self.norm_factor_p0[split],
                )
                y_p1, y_p1_err = compute_hist_data(
                    bins,
                    data_p1,
                    weights=(self.prior_weights_p1[split]) * self.norm_factor_p0[split],
                )
                y_rew, y_rew_err = compute_hist_data(
                    bins,
                    data_p0,
                    weights=(self.prior_weights_p0[split] * self.class_w_p0[split])
                    * self.norm_factor_p0[split],
                )
                y_rew2, y_rew2_err = compute_hist_data(
                    bins,
                    data_p0,
                    weights=(self.prior_weights_p0[split] * (1-self.class_w_p0[split]))
                    * self.norm_factor_p0[split],
                )
                # signal-in-p0 histogram — needed by both debug and clean views
                y_sigp0, y_sigp0_err = compute_hist_data(
                    bins,
                    data=data_p0[self.mask_sig_in_p0[split]],
                    weights=self.prior_weights_p0[split][self.mask_sig_in_p0[split]]
                    * self.norm_factor_p0[split],
                )

                if self.debug:
                    # ---- Debug / diagnostic view ----------------------------------------
                    wsc_1d = self.class_w_p0[split] if self.class_w_p0[split].ndim == 1 \
                             else self.class_w_p0[split].mean(axis=0)
                    mask_negw = (1.0 - wsc_1d) < 0
                    y_negw, y_negw_err = compute_hist_data(
                        bins,
                        data_p0[mask_negw],
                        weights=self.prior_weights_p0[split][mask_negw] * self.norm_factor_p0[split],
                    )
                    lines = [
                        Line(
                            y=y_p0,
                            y_err=y_p0_err,
                            label=r"$p_{\text{data}}$",
                            color=P0_COLOR,
                        ),
                        Line(
                            y=y_negw,
                            y_err=y_negw_err,
                            label=r"$(1-w_{\text{sc}}(x))<0 \in p_{\text{data}}$",
                            color=DATA_COLOR,
                            linestyle="dotted",
                        ),
                        Line(
                            y=y_p1,
                            y_err=y_p1_err,
                            label=r"$p_{\text{bkg}}^{\text{MC}}$",
                            color=P1_COLOR,
                        ),
                        Line(
                            y=y_rew,
                            y_err=y_rew_err,
                            y_ref=y_p1,
                            label=r"$w_{\text{sc}}(x) \cdot\ p_{\text{data}}$",
                            color="darkorange",
                            linestyle="dashed",
                        ),
                    ]
                    if self.show_class_components.get("p0", False):
                        y_bkgp0, y_bkgp0_err = compute_hist_data(
                            bins,
                            data=data_p0[self.mask_bkg_in_p0[split]],
                            weights=self.prior_weights_p0[split][self.mask_bkg_in_p0[split]]
                            * self.norm_factor_p0[split],
                        )
                        if i == 1:
                            self.logger.info(
                                f"p0 integral: {np.sum(y_p0)}, signal integral: {np.sum(y_sigp0)}, bkg integral: {np.sum(y_bkgp0)}"
                            )
                            self.logger.info(
                                f"p1 integral: {np.sum(y_p1)}"
                            )
                        lines += [
                            Line(
                                y=y_sigp0,
                                y_err=y_sigp0_err,
                                label=r"$\mathrm{sig} \in p_{\text{data}}$",
                                color=HWW_COLOR,
                                linestyle="solid",
                                alpha=0.5,
                            ),
                            Line(
                                y=y_bkgp0,
                                y_err=y_bkgp0_err,
                                label=r"$\mathrm{bkg} \in p_{\text{data}}$",
                                color=BKG_COLOR,
                                linestyle="solid",
                                alpha=0.5,
                            ),
                            Line(
                                y=y_rew2,
                                y_err=y_rew2_err,
                                y_ref=y_sigp0,
                                label=r"$(1-w_{\text{sc}})\cdot p_{\text{data}}$",
                                linestyle="dashed",
                                color=REW_COLOR,
                            ),
                        ]
                    if self.show_class_components.get("p1", False):
                        y_ttbar_p1, y_ttbar_p1_err = compute_hist_data(
                            bins,
                            data=data_p1[self.mask_ttbar_in_p1[split]],
                            weights=self.prior_weights_p1[split][
                                self.mask_ttbar_in_p1[split]
                            ]
                            * self.norm_factor_p1[split],
                        )
                        y_WWjj_p1, y_WWjj_p1_err = compute_hist_data(
                            bins,
                            data=data_p1[self.mask_WWjj_in_p1[split]],
                            weights=self.prior_weights_p1[split][
                                self.mask_WWjj_in_p1[split]
                            ]
                            * self.norm_factor_p1[split],
                        )
                        if i == 0:
                            self.logger.info(
                                f"p1 integral: {np.sum(y_p1)}, ttbar integral: {np.sum(y_ttbar_p1)}, WWjj integral: {np.sum(y_WWjj_p1)}"
                            )
                        lines += [
                            Line(
                                y=y_ttbar_p1,
                                y_err=y_ttbar_p1_err,
                                label=r"$p_{\text{bkg}}^{\text{MC}}\ [t\bar{t}]$",
                                color=DATA_COLOR,
                                linestyle="dashed",
                                alpha=0.5,
                            ),
                            Line(
                                y=y_WWjj_p1,
                                y_err=y_WWjj_p1_err,
                                label=r"$p_{\text{bkg}}^{\text{MC}}\ [WWjj]$",
                                color=BKG_MC_COLOR,
                                linestyle="dashed",
                                alpha=0.5,
                            ),
                        ]
                else:
                    # ---- Clean view (default) -------------------------------------------
                    lines = [
                        Line(
                            y=y_p0,
                            y_err=y_p0_err,
                            label=r"$p_{\text{data}}$",
                            color=P0_COLOR,
                        ),
                        Line(
                            y=y_p1,
                            y_err=y_p1_err,
                            label=r"$p_{\text{bkg}}^{\text{MC}}$",
                            color=P1_COLOR,
                        ),
                        Line(
                            y=y_rew,
                            y_err=y_rew_err,
                            y_ref=y_p1,
                            label=LABEL_W,
                            color="darkorange",
                            linestyle="dashed",
                        ),
                        Line(
                            y=y_sigp0,
                            y_err=y_sigp0_err,
                            label=LABEL_SIG,
                            color=DATA_COLOR,
                            linestyle="solid",
                            alpha=1.0,
                        ),
                        Line(
                            y=y_rew2,
                            y_err=y_rew2_err,
                            y_ref=y_sigp0,
                            label=LABEL_1MW,
                            color=REW_COLOR,
                            linestyle="dotted",
                        ),
                    ]
                hist_kwargs = dict(
                    show_ratios=True,
                    model_name="Rew.",
                    no_scale=True,
                    truth_name="Target",
                    title=self.title,
                    **legend_layout(obs),
                )
                hist_plot(pp, lines, bins, obs, **hist_kwargs)
                if pickle_file is not None:
                    # hist_kwargs saved so a replot reproduces this page exactly
                    pickle_data.append(
                        {"lines": lines, "bins": bins, "obs": obs, "hist_kwargs": hist_kwargs}
                    )
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_refd_observables(
        self, file: str, pickle_file: Optional[str] = None, split: str = "trn"
    ):
        """
        Plots observable distributions including p_data reweighted by the refined factor
        |1-w| * r, where r = tanh(f/2) = refiner_w_p0.
        """
        if not self.has_refiner:
            self.logger.warning("No refiner data found; skipping plot_refd_observables.")
            return
        pickle_data = []
        r = self.refiner_w_p0[split]
        refd_factor = np.abs(1 - self.class_w_p0[split]) * r
        with PdfPages(file) as pp:
            for obs, bins, data_p0, data_p1 in zip(
                self.obs[split], self.bins[split],
                self.obs_p0[split], self.obs_p1[split],
            ):
                y_p0, y_p0_err = compute_hist_data(
                    bins, data_p0,
                    weights=self.prior_weights_p0[split] * self.norm_factor_p0[split],
                )
                y_p1, y_p1_err = compute_hist_data(
                    bins, data_p1,
                    weights=self.prior_weights_p1[split] * self.norm_factor_p0[split],
                )
                y_refd, y_refd_err = compute_hist_data(
                    bins, data_p0,
                    weights=self.prior_weights_p0[split] * refd_factor
                            * self.norm_factor_p0[split],
                )
                y_sig = None
                if hasattr(self, "mask_sig_in_p0"):
                    y_sig, y_sig_err = compute_hist_data(
                        bins,
                        data=data_p0[self.mask_sig_in_p0[split]],
                        weights=self.prior_weights_p0[split][self.mask_sig_in_p0[split]]
                                * self.norm_factor_p0[split],
                    )
                lines = [
                    Line(y=y_p0, y_err=y_p0_err,
                         label=r"$p_{\text{data}}$", color=P0_COLOR),
                    Line(y=y_p1, y_err=y_p1_err,
                         label=r"$p_{\text{bkg}}^{\text{MC}}$", color=P1_COLOR),
                    Line(y=y_refd, y_err=y_refd_err,
                         y_ref=y_sig if y_sig is not None else y_p1,
                         label=r"$p_{\text{data}}^{\text{refined}}$",
                         color=REW_COLOR, linestyle="dashed"),
                ]
                if y_sig is not None:
                    lines.insert(2, Line(
                        y=y_sig, y_err=y_sig_err,
                        label=r"$\mathrm{sig} \in p_{\text{data}}$",
                        color=HWW_COLOR, linestyle="solid", alpha=0.5,
                    ))
                hist_plot(
                    pp, lines, bins, obs,
                    show_ratios=True, model_name="Ref.", no_scale=True,
                    truth_name="Target", title=self.title,
                )
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_observables_ppd(
        self, file: str, pickle_file: Optional[str] = None, split: str = "tst"
    ):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Path to the output PDF file
            pickle_file: Path to the output pickle file (optional)
        """
        pickle_data = []
        # One panel per channel the network actually ingests, straight from the standardized
        # tensor it sees (self.p0_ppd_full / p1_ppd_full). Iterating self.channels guarantees
        # no input channel is dropped -- including derived features (mT, m_jj, dphi/deta).
        obs_by_channel = {o.channel: o for o in self.obs[split] if o.channel is not None}
        _np = lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        p0_full = _np(self.p0_ppd_full[split])  # may be a CUDA tensor on GPU runs
        p1_full = _np(self.p1_ppd_full[split])
        with PdfPages(file) as pp:
            for c in self.channels:
                obs = obs_by_channel.get(c) or Observable(
                    channel=c, compute=None, bins=None, tex_label=rf"\text{{channel }}{c}"
                )
                data_p0 = p0_full[:, c]
                data_p1 = p1_full[:, c]
                # tail-aware bins: 1-99 percentile of the pooled standardized values, so
                # long-tailed features (e.g. raw-standardized pt) aren't clipped at +/-2.
                pooled = np.concatenate([data_p0, data_p1])
                lo, hi = np.percentile(pooled, [1, 99])
                if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                    lo, hi = float(np.min(pooled)), float(np.max(pooled))
                    if hi <= lo:
                        hi = lo + 1.0
                bins = np.linspace(lo, hi, 31)
                y_p0, y_p0_err = compute_hist_data(
                    bins,
                    data_p0,
                    weights=self.prior_weights_p0[split] * self.norm_factor_p0[split],
                )
                y_p1, y_p1_err = compute_hist_data(
                    bins,
                    data_p1,
                    weights=self.prior_weights_p1[split] * self.norm_factor_p1[split],
                )
                y_rew, y_rew_err = compute_hist_data(
                    bins,
                    data_p0,
                    weights=self.prior_weights_p0[split]
                    * self.class_w_p0[split]
                    * self.norm_factor_p1[split],
                )
                y_rew2, y_rew2_err = compute_hist_data(
                    bins,
                    data_p0,
                    weights=self.prior_weights_p0[split]
                    * (1-self.class_w_p0[split])
                    * self.norm_factor_p1[split],
                )
                lines = [
                    Line(
                        y=y_p0,
                        y_err=y_p0_err,
                        label=r"$p_{\text{data}}$",
                        color=P0_COLOR,
                    ),
                    Line(
                        y=y_p1,
                        y_err=y_p1_err,
                        label=r"$p_{\text{bkg}}^{\text{MC}}$",
                        color=P1_COLOR,
                    ),
                    Line(
                        y=y_rew,
                        y_err=y_rew_err,
                        y_ref=y_p1,
                        label=r"$w_{\text{sc}}\cdot p_{\text{data}}$",
                        color="darkorange",
                        linestyle="dashed",
                    ),
                ]
                if self.show_class_components.get("p0", False):
                    y_sigp0, y_sigp0_err = compute_hist_data(
                        bins,
                        data=data_p0[self.mask_sig_in_p0[split]],
                        weights=self.prior_weights_p0[split][self.mask_sig_in_p0[split]]
                        * self.norm_factor_p0[split],
                    )
                    y_bkgp0, y_bkgp0_err = compute_hist_data(
                        bins,
                        data=data_p0[self.mask_bkg_in_p0[split]],
                        weights=self.prior_weights_p0[split][self.mask_bkg_in_p0[split]]
                        * self.norm_factor_p0[split],
                    )
                    lines += [
                        Line(
                            y=y_sigp0,
                            y_err=y_sigp0_err,
                            label=r"$\mathrm{sig} \in p_{\text{data}}$",
                            color=HWW_COLOR,
                            linestyle="solid",
                            alpha=0.5,
                        ),
                        Line(
                            y=y_bkgp0,
                            y_err=y_bkgp0_err,
                            label=r"$\mathrm{bkg} \in p_{\text{data}}$",
                            color=BKG_COLOR,
                            linestyle="solid",
                            alpha=0.5,
                        ),
                        Line(
                            y=y_rew2,
                            y_err=y_rew2_err,
                            y_ref=y_sigp0,
                            label=r"$(1-w_{\text{sc}})\cdot p_0$",
                            linestyle="dashed",
                            color=REW_COLOR,
                        ),
                    ]
                if self.show_class_components.get("p1", False):
                    y_ttbar_p1, y_ttbar_p1_err = compute_hist_data(
                        bins,
                        data=data_p1[self.mask_ttbar_in_p1[split]],
                        weights=self.prior_weights_p1[split][
                            self.mask_ttbar_in_p1[split]
                        ]
                        * self.norm_factor_p1[split],
                    )
                    y_WWjj_p1, y_WWjj_p1_err = compute_hist_data(
                        bins,
                        data=data_p1[self.mask_WWjj_in_p1[split]],
                        weights=self.prior_weights_p1[split][
                            self.mask_WWjj_in_p1[split]
                        ]
                        * self.norm_factor_p1[split],
                    )
                    lines += [
                        Line(
                            y=y_ttbar_p1,
                            y_err=y_ttbar_p1_err,
                            label=r"$t\bar{t} \in p_{\text{bkg}}^{\text{MC}}$",
                            color=DATA_COLOR,
                            linestyle="solid",
                            alpha=0.5,
                        ),
                        Line(
                            y=y_WWjj_p1,
                            y_err=y_WWjj_p1_err,
                            label=r"$WWjj \in p_{\text{bkg}}^{\text{MC}}$",
                            color=BKG_MC_COLOR,
                            linestyle="solid",
                            alpha=0.5,
                        ),
                    ]
                hist_kwargs = dict(
                    show_ratios=True,
                    model_name="Rew.",
                    no_scale=False,
                    truth_name="Target",
                    title=self.title,
                )
                hist_plot(pp, lines, bins, obs, **hist_kwargs)
                if pickle_file is not None:
                    # hist_kwargs saved so a replot reproduces this page exactly
                    pickle_data.append(
                        {"lines": lines, "bins": bins, "obs": obs, "hist_kwargs": hist_kwargs}
                    )
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    # def plot_migration(
    #     self, file: str, gt_part=False, pickle_file: Optional[str] = None
    # ):
    #     if gt_part:
    #         obs_part = self.obs_part
    #         name_part = "Particle" if self.unfold_to == "particle" else "Parton"
    #     else:
    #         obs_part = self.obs_unf
    #         name_part = "Unfold"
    #     ranges = [[0, 0.4], [0, 0.3], [0, 0.5], [0, 0.5], [0, 0.2], [0, 0.25]]
    #     ranges += [[0, 0.5] for _ in range(len(self.observables) - len(ranges))]

    #     pickle_data = []
    #     with PdfPages(file) as pp:
    #         for k, (obs, bins, data_hard, data_reco, r) in enumerate(
    #             zip(self.observables, self.bins, obs_part, self.obs_reco, ranges)
    #         ):
    #             if obs.channel is not None and obs.channel not in self.channels:
    #                 # avoid plotting primary (obs.channel != None) channels not used for unfolding
    #                 # secondary channels (derived observables -- obs.channel == None) are always plotted
    #                 continue
    #             cmap = plt.get_cmap("viridis")
    #             cmap.set_bad("white")
    #             h, x, y = np.histogram2d(data_reco, data_hard, bins=(bins, bins))
    #             h = np.ma.divide(h, np.sum(h, 1, keepdims=True)).filled(
    #                 0
    #             )  # normalize so each row sums up to 1 => h.T each column is a distribution: for a given reco event you have a pdf
    #             h[h == 0] = np.nan
    #             plt.pcolormesh(
    #                 bins,
    #                 bins,
    #                 h.T,
    #                 cmap=cmap,
    #                 rasterized=True,
    #                 vmin=r[0],
    #                 vmax=r[1],
    #             )
    #             plt.colorbar()

    #             unit = "" if obs.unit is None else f" [{obs.unit}]"
    #             plt.title(f"${{{obs.tex_label}}}${unit}")
    #             plt.xlim(bins[0], bins[-1])
    #             plt.ylim(bins[0], bins[-1])
    #             plt.xlabel("Reco")
    #             plt.ylabel(name_part)
    #             plt.savefig(pp, format="pdf", bbox_inches="tight")
    #             plt.close()

    #             if pickle_file is not None:
    #                 pickle_data.append({"h": h, "bins": bins, "ranges": r, "obs": obs})
    #     if pickle_file is not None:
    #         with open(pickle_file, "wb") as f:
    #             pickle.dump(pickle_data, f)

    # def plot_migration_diff(
    #     self, file: str, gt_part=False, pickle_file: Optional[str] = None
    # ):

    #     gt_obs_part = self.obs_part
    #     unf_obs_part = self.obs_unf

    #     name_part = (
    #         "Unfold $-$ Particle"
    #         if self.unfold_to == "particle"
    #         else "Unfold $-$ Parton"
    #     )
    #     ranges = [[0, 5], [0, 5], [0, 5], [0, 5], [0, 5], [0, 5]]
    #     ranges += [[0, 5] for _ in range(len(self.observables) - len(ranges))]

    #     pickle_data = []
    #     with PdfPages(file) as pp:
    #         for k, (obs, bins, unfold, gt, data_reco, r) in enumerate(
    #             zip(
    #                 self.observables,
    #                 self.bins,
    #                 unf_obs_part,
    #                 gt_obs_part,
    #                 self.obs_reco,
    #                 ranges,
    #             )
    #         ):
    #             if obs.channel is not None and obs.channel not in self.channels:
    #                 # avoid plotting primary (obs.channel != None) channels not used for unfolding
    #                 # secondary channels (derived observables -- obs.channel == None) are always plotted
    #                 continue
    #             cmap = plt.get_cmap("bwr")
    #             cmap.set_bad("white")
    #             h_unf, x_unf, y_unf = np.histogram2d(
    #                 data_reco, unfold, bins=(bins, bins)
    #             )
    #             # h_unf_norm = np.ma.divide(h_unf, np.sum(h_unf, 1, keepdims=True)).filled(
    #             #     0
    #             # )  # normalize so each row sums up to 1 => h.T each column is a distribution: for a given reco event you have a pdf

    #             h_gt, x_gt, y_gt = np.histogram2d(data_reco, gt, bins=(bins, bins))
    #             # h_gt_norm = np.ma.divide(h_gt, np.sum(h_gt, 1, keepdims=True)).filled(
    #             #     0
    #             # )  # normalize so each row sums up to 1 => h.T each column is a distribution: for a given reco event you have a pdf

    #             diff = h_unf - h_gt
    #             # Symmetric dynamic limits
    #             M = np.nanmax(np.abs(diff))
    #             norm = mpl.colors.CenteredNorm(vcenter=0.0, halfrange=M)
    #             plt.pcolormesh(
    #                 bins,
    #                 bins,
    #                 diff.T,
    #                 cmap=cmap,
    #                 norm=norm,
    #                 rasterized=True,
    #                 # vmin=r[0],
    #                 # vmax=r[1],
    #             )
    #             plt.colorbar()

    #             unit = "" if obs.unit is None else f" [{obs.unit}]"
    #             plt.title(f"${{{obs.tex_label}}}${unit}")
    #             plt.xlim(bins[0], bins[-1])
    #             plt.ylim(bins[0], bins[-1])
    #             plt.xlabel("Reco")
    #             plt.ylabel(name_part)
    #             plt.savefig(pp, format="pdf", bbox_inches="tight")
    #             plt.close()

    #             if pickle_file is not None:
    #                 pickle_data.append(
    #                     {"h_ratio": diff, "bins": bins, "ranges": r, "obs": obs}
    #                 )
    #     if pickle_file is not None:
    #         with open(pickle_file, "wb") as f:
    #             pickle.dump(pickle_data, f)

# Per-page legend layouts for the background-subtraction observables, so legends do not cover the
# curves. Keyed by Observable.key, else tex_label. "split_legend" groups get their own legend
# (labels -> loc, optionally anchored at bbox_to_anchor in axes coordinates); the remaining entries
# use "legend_kwargs" (default: matplotlib "best").
W_LABELS = [LABEL_W, r"$w_{\text{sc}}(x) \cdot\ p_{\text{data}}$"]
SIG_LABELS = [LABEL_SIG, r"$\mathrm{sig} \in p_{\text{data}}$"]
ONE_MINUS_W_LABELS = [LABEL_1MW, r"$(1-w_{\text{sc}})\cdot p_{\text{data}}$"]
LEGEND_LAYOUTS = {
    r"m_{j_{1}}": dict(
        # p_data, p_bkg^MC, p_signal stay in the main legend (top left, in that order)
        legend_kwargs={"loc": "upper left", "handlelength": 0.8},
        split_legend=[
            {"labels": W_LABELS, "loc": "upper right"},
            {"labels": ONE_MINUS_W_LABELS, "loc": "lower right", "bbox_to_anchor": (1.0, 0.12)},
        ],
    ),
    r"m_{T}": dict(split_legend=[
        {"labels": SIG_LABELS, "loc": "upper right"},
        {"labels": ONE_MINUS_W_LABELS, "loc": "lower right", "bbox_to_anchor": (1.0, 0.04)},
    ]),
    "dphi_jj_signed_eta": dict(
        # p_data, p_bkg^MC, w*p_data, p_signal top center; [1-w]*p_data between p_bkg^MC and p_signal at the center
        legend_kwargs={"loc": "upper center", "handlelength": 0.8},
        split_legend=[
            {"labels": ONE_MINUS_W_LABELS, "loc": "center", "bbox_to_anchor": (0.5, 0.185)},
        ],
    ),
}

def legend_layout(obs):
    """hist_plot kwargs (split_legend / legend_kwargs) for ``obs``; {} = single "best" legend."""
    key = getattr(obs, "key", None)
    return dict(LEGEND_LAYOUTS.get(key) or LEGEND_LAYOUTS.get(obs.tex_label) or {})


def compute_hist_data(
    bins: np.ndarray,
    data: np.ndarray,
    bayesian=False,
    weights=None,
    uncertainty: str = "counts",
    debug=False,
):
    data = np.asarray(data)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)

    # Auto-detect bayesian mode from array shapes
    is_bayesian = bayesian or data.ndim == 2 or (weights is not None and weights.ndim == 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if is_bayesian:
            if data.ndim == 2:
                # data is (N_samples, N_events): varying samples, fixed weights
                hists = np.stack(
                    [np.histogram(d, bins=bins, density=False, weights=weights)[0]
                     for d in data],
                    axis=0,
                )
            else:
                # weights is (N_samples, N_events): fixed data, varying weights
                hists = np.stack(
                    [np.histogram(data, bins=bins, density=False, weights=w)[0]
                     for w in weights],
                    axis=0,
                )
            y = np.mean(hists, axis=0)
            y_err = np.std(hists, axis=0)
        else:
            if weights is None:
                weights = np.ones_like(data, dtype=float)
            y, _ = np.histogram(data, bins=bins, weights=weights)
            sumw2, _ = np.histogram(data, bins=bins, weights=weights**2)
            y_err = np.sqrt(sumw2)
            if debug:
                print("y", y.mean(), y.std(), y.sum())
                print("w", weights.mean(), weights.std(), weights.sum())
        return y, y_err



def hist_plot(*args, ylabel=None, no_scale=False, **kwargs):
    """The unfolding ``hist_plot`` (identical styling); unnormalized pages are labelled Events."""
    if ylabel is None and no_scale:
        ylabel = "Events"
    return _unfolding_hist_plot(*args, ylabel=ylabel, no_scale=no_scale, **kwargs)
