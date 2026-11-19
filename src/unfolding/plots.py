import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass
from datasets.dataset import Observable
from typing import Optional

PART_COLOR = "#0B9999"
RECO_COLOR = "#52990B"
CFM_COLOR = "#990B0B"
PRIOR_COLOR = "#7E9412"  # Color for the Standard Model line, if applicable
SIGNAL_PART_COLOR = "#520B99"  # Orange — pure signal parton MC closure line
SIGNAL_TRUTH_COLOR = "#17BECF"  # p_signal in the reweighted-data plot; = classification DATA_COLOR
# Per-observable layout of the reweighted-data plot (tex_label -> extra hist_plot kwargs, e.g.
# legend_kwargs / headroom); observables not listed use matplotlib's "best" legend placement.
UNFOLDED_DATA_LAYOUT = {
    r"m_{T}": {"legend_kwargs": {"loc": "upper left", "handlelength": 0.8}, "headroom": 1.25},
}

# Per-background unfolding-closure pairs: (TraCFM(p_X) color, p_X(x_part) truth color).
# One same-ish hue per process so good closure shows as overlapping dashed lines.
BKG_CLOSURE_COLORS = [
    ("#6A0DAD", "#B23AEE"),  # WWjj  — purple pair
    ("#008B8B", "#00CED1"),  # ttbar — teal pair
]


@dataclass
class Line:
    y: np.ndarray
    y_err: Optional[np.ndarray] = None
    y_ref: Optional[np.ndarray] = None
    label: Optional[str] = None
    color: Optional[str] = None
    linestyle: Optional[str] = "solid"
    fill: bool = False
    vline: bool = False
    invert_ratio: bool = False
    alpha: float = 1.0
    ratio_err: Optional[np.ndarray] = None  # explicit ratio-panel band (e.g. bootstrap); overrides y_err/y
    ratio_linestyle: Optional[str] = None  # ratio-panel line style (default: linestyle)

def force_logy_ticks(ax, *, max_major=3):
    """
    Robust log-y ticks:
      - minor ticks at 2..9 * 10^n within visible ylim
      - major ticks: <= max_major, always in-range; prefers decade ticks if possible
    Call AFTER all plotting and AFTER final ylim is known (autoscale or manual).
    """
    if ax.get_yscale() != "log":
        return

    ymin, ymax = ax.get_ylim()
    if not (np.isfinite(ymin) and np.isfinite(ymax)) or ymin <= 0 or ymax <= 0:
        return
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    logmin = np.log10(ymin)
    logmax = np.log10(ymax)

    e_lo = int(np.floor(logmin)) - 1
    e_hi = int(np.ceil(logmax)) + 1

    # -----------------
    # MINOR ticks: 2..9 * 10^n within [ymin, ymax]
    # -----------------
    minor = []
    for e in range(e_lo, e_hi + 1):
        decade = 10.0**e
        for s in range(2, 10):  # 2..9
            v = s * decade
            if ymin <= v <= ymax:
                minor.append(v)

    ax.yaxis.set_minor_locator(ticker.FixedLocator(minor))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # -----------------
    # MAJOR ticks: prefer decade ticks inside the view
    # -----------------
    decade_inside = []
    for e in range(e_lo, e_hi + 1):
        v = 10.0**e
        if ymin <= v <= ymax:
            decade_inside.append(v)

    decade_inside = np.array(decade_inside, dtype=float)

    if len(decade_inside) >= 2:
        # too many? subsample evenly down to max_major
        if len(decade_inside) > max_major:
            idx = np.linspace(0, len(decade_inside) - 1, max_major)
            major = decade_inside[np.round(idx).astype(int)]
        else:
            major = decade_inside

    elif len(decade_inside) == 1:
        # one decade tick in range: add endpoints (geometric) to avoid "single major tick" look
        candidates = np.unique(np.array([ymin, decade_inside[0], ymax], dtype=float))
        # if still too many, reduce to max_major by picking evenly in log space
        if len(candidates) > max_major:
            logc = np.log10(candidates)
            idx = np.linspace(0, len(candidates) - 1, max_major)
            major = candidates[np.round(idx).astype(int)]
        else:
            major = candidates

    else:
        # no decade tick inside: place majors geometrically within range
        n = min(max_major, 3)
        major = np.geomspace(ymin, ymax, n)

    ax.yaxis.set_major_locator(ticker.FixedLocator(major))
    # formatter that behaves nicely for both decades and non-decade majors
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))

    ax.minorticks_on()
    ax.tick_params(axis="y", which="minor", length=3)

class TwoDecimalScalarFormatter(ticker.ScalarFormatter):
    def _set_format(self):
        self.format = "%1.2f"

class Plots:
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
        SM_dataset,
        EFTchw0p5_dataset,
        EFTchwtil5_dataset,
        losses: dict = None,
        debug: bool = False,
        unfold_to: str = "particle",
        model_name: Optional[str] = "CFM",
        title: Optional[str] = None,
        subsets: Optional[list] = None,
    ):
        self.logger = logger
        self.dataset = dataset
        self.dataset_ids = (
            dataset.dataset_ids if hasattr(dataset, "dataset_ids") else None
        )
        self.observables = dataset.observables
        self.observables_ppd = (
            dataset.observables_ppd if hasattr(dataset, "observables_ppd") else None
        )

        self.losses = losses
        self.channels = dataset.channels
        # physical channels of the unfolded events: with the balanced/valid parameterizations the
        # completion fills the condition channels too (jet pT/phi, pT_ll, ...), so plot those as well
        self.plot_channels = getattr(dataset, "plot_channels", dataset.channels)
        self.bins = dataset.bins
        if subsets is not None and len(subsets) > 0 and self.dataset_ids is not None:
            mask = [i.item() in subsets for i in dataset.dataset_ids]
        else:
            mask = np.ones(len(dataset.obs_part[0]), dtype=bool)
        self.obs_part = [part_obs[mask] for part_obs in dataset.obs_part]
        self.obs_reco = [reco_obs[mask] for reco_obs in dataset.obs_reco]
        self.logger.info(
            f"Plotting using {np.sum(mask)} events out of {len(mask)} total events."
        )
        self.obs_unf = dataset.obs_unf  # this is already evaluated from a subset

        for (obs_unf, obs_part, obs_reco) in zip(
            self.obs_unf, self.obs_part, self.obs_reco
        ):
            assert (
                len(obs_unf) == len(obs_part) == len(obs_reco)
            ), "Unfolded, particle, and reco data have different lengths!"

        self.prior_obs_part = (
            SM_dataset.obs_part
            if SM_dataset is not None
            else EFTchw0p5_dataset.obs_part
            if EFTchw0p5_dataset is not None
            else EFTchwtil5_dataset.obs_part
            if EFTchwtil5_dataset is not None
            else [None for _ in self.obs_part]
        )
        self.prior_label = (
            "SM"
            if SM_dataset is not None
            else r"$c_{hW}$"
            if EFTchw0p5_dataset is not None
            else r"$c_{h\tilde{W}}$"
            if EFTchwtil5_dataset is not None
            else "Prior"
        )

        self.obs_part_ppd = (
            dataset.obs_part_ppd if hasattr(dataset, "obs_part_ppd") else None
        )
        self.obs_reco_ppd = (
            dataset.obs_reco_ppd if hasattr(dataset, "obs_reco_ppd") else None
        )
        self.obs_unf_ppd = (
            dataset.obs_unf_ppd if hasattr(dataset, "obs_unf_ppd") else None
        )

        self.debug = debug
        self.unfold_to = unfold_to
        self.truth_name = "Parton" if unfold_to == "parton" else "Particle"
        self.model_name = model_name
        self.title = title if title is not None else None

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
        pickle_file: Optional[str] = None,
        ylabel="Loss",
        rect=(0.15, 0.15, 0.83, 0.9),
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
                yfmt = ScalarFormatterForceFormat()
                yfmt.set_powerlimits((0, 0))
                axright.yaxis.set_major_formatter(yfmt)
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
        if pickle_file is not None:
            append_to_pickle({"losses": self.losses}, pickle_file)

    def plot_observables(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Path to the output PDF file
            pickle_file: Path to the output pickle file (optional)
        """
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco, data_unf, data_prior in zip(
                self.observables,
                self.bins,
                self.obs_part,
                self.obs_reco,
                self.obs_unf,
                self.prior_obs_part,
            ):
                if obs.channel is not None and obs.channel not in self.plot_channels:
                    # avoid plotting primary (obs.channel != None) channels not used for unfolding
                    # secondary channels (derived observables -- obs.channel == None) are always plotted
                    continue
                y_hard, y_hard_err = compute_hist_data(bins, data_hard, bayesian=False)
                y_reco, y_reco_err = compute_hist_data(bins, data_reco, bayesian=False)
                y_unf, y_unf_err = compute_hist_data(bins, data_unf, bayesian=False)

                lines = []
                if data_prior is not None:
                    y_prior, y_prior_err = compute_hist_data(
                        bins, data_prior, bayesian=False
                    )
                    lines.append(
                        Line(
                            y=y_prior,
                            y_err=y_prior_err,
                            y_ref=y_unf,
                            label=self.prior_label,
                            color=PRIOR_COLOR,
                            linestyle="dashed",
                            invert_ratio=True,
                            # linewidth=1.5,
                        )
                    )
                lines.append(
                    Line(
                        y=y_reco,
                        y_err=y_reco_err,
                        label="Reco",
                        color=RECO_COLOR,
                    )
                )
                lines.append(
                    Line(
                        y=y_hard,
                        y_err=y_hard_err,
                        label="Particle" if self.unfold_to == "particle" else "Parton",
                        color=PART_COLOR,
                    )
                )
                lines.append(
                    Line(
                        y=y_unf,
                        y_err=y_unf_err,
                        y_ref=y_hard,
                        label=self.model_name,
                        color=CFM_COLOR,
                    )
                )
                hist_plot(
                    pp,
                    lines,
                    bins,
                    obs,
                    show_ratios=True,
                    model_name=self.model_name,
                    truth_name=self.truth_name,
                    title=self.title,
                    title_on_axs=True,
                )
                if pickle_file is not None:
                    append_to_pickle({"lines": lines, "bins": bins, "obs": obs,
                                      "model_name": self.model_name, "truth_name": self.truth_name,
                                      "title": self.title}, pickle_file)

    # tex labels of the dataset columns (WBF layout), for the preprocessed-input plots
    PPD_COLUMN_LABELS = {
        0: r"p_{T, \ell_{1}}", 1: r"\phi_{\ell_{1}}", 2: r"\eta_{\ell_{1}}", 3: r"m_{\ell_{1}}",
        4: r"p_{T, \ell_{2}}", 5: r"\phi_{\ell_{2}}", 6: r"\eta_{\ell_{2}}", 7: r"m_{\ell_{2}}",
        8: r"p_{T, j_{1}}", 9: r"\phi_{j_{1}}", 10: r"\eta_{j_{1}}", 11: r"m_{j_{1}}",
        12: r"p_{T, j_{2}}", 13: r"\phi_{j_{2}}", 14: r"\eta_{j_{2}}", 15: r"m_{j_{2}}",
        16: r"E_{T, \text{miss}}", 17: r"\phi_{\text{miss}}", 18: r"\eta_{\text{miss}}", 19: r"m_{\text{miss}}",
        20: r"N_{\text{jets}}", 21: r"p_{T, \ell\ell}", 22: r"\Delta\phi_{\ell\ell}", 23: r"\Delta\eta_{\ell\ell}",
        24: r"m_{T}", 25: r"\Delta\phi_{\ell\ell, \text{miss}}", 26: r"m_{\ell\ell}/m_{T}",
        27: r"\phi_{j_{1}} - \phi_{j_{2}}", 28: r"p_{T, j_{2}}/p_{T, j_{1}}", 29: r"\phi_{j_{1}} - \phi_{H}", 30: r"|\vec{k}| = |\vec{p}_{T,j_1} - \vec{p}_{T,j_2}|/2", 31: r"\phi_{k} - \phi_{H}", 32: r"\Delta\phi_{j_1 j_2}", 33: r"H_{T}", 34: r"\Delta\phi_{j_1 j_2}",
    }

    def plot_observables_ppd(self, file: str, pickle_file: Optional[str] = None):
        """
        Histograms of the preprocessed network inputs, one page per column: every reco condition
        column (Reco) and every unfolded column (Parton target and the unfolding). Columns that are
        both (the usual case) show all three lines with the ratio to the target.
        Args:
            file: Path to the output PDF file
            pickle_file: Path to the output pickle file (optional)
        """
        ds = self.dataset
        unf_ch = list(ds.channels)
        cond_ch = list(getattr(ds, "cond_channels", unf_ch))
        part = ds.part_ppd["tst"].detach().cpu().numpy()
        reco = ds.reco_ppd["tst"].detach().cpu().numpy()
        unf = ds.unfolded_ppd.detach().cpu().numpy() if hasattr(ds, "unfolded_ppd") else None
        if unf is not None and len(unf) != len(part):
            self.logger.info("    ppd plot: unfolded and test events differ in number (subsets); no unfolded lines")
            unf = None
        bins = np.linspace(-5, 5, 51)
        with PdfPages(file) as pp:
            for c in sorted(set(unf_ch) | set(cond_ch)):
                is_unf, is_cond = c in unf_ch, c in cond_ch
                lines = []
                if is_cond:
                    y, e = compute_hist_data(bins, reco[:, c], bayesian=False)
                    lines.append(Line(y=y, y_err=e, label="Reco", color=RECO_COLOR))
                if is_unf:
                    y_hard, e = compute_hist_data(bins, part[:, c], bayesian=False)
                    lines.append(Line(y=y_hard, y_err=e,
                                      label="Particle" if self.unfold_to == "particle" else "Parton",
                                      color=PART_COLOR))
                    if unf is not None:
                        y, e = compute_hist_data(bins, unf[:, c], bayesian=False)
                        lines.append(Line(y=y, y_err=e, y_ref=y_hard, label=self.model_name, color=CFM_COLOR))
                role = "unfolded + condition" if (is_unf and is_cond) else ("unfolded" if is_unf else "condition only")
                obs = Observable(compute=None, bins=None,
                                 tex_label=self.PPD_COLUMN_LABELS.get(c, f"\\text{{column {c}}}") + r"\ \text{(preprocessed)}")
                hist_plot(pp, lines, bins, obs, show_ratios=is_unf and unf is not None,
                          model_name=self.model_name, truth_name=self.truth_name,
                          title=rf"$\textrm{{column {c}: {role}}}$")
                if pickle_file is not None:
                    append_to_pickle({"lines": lines, "bins": bins, "obs": obs}, pickle_file)

    def plot_multiplicity_cut_observables(
        self, file: str, pickle_file: Optional[str] = None
    ):
        """
        Makes histograms of truth and generated distributions for observables cut on jet/quark multiplicity.
        Args:
            file: Path to the output PDF file
            pickle_file: Path to the output pickle file (optional)
        """
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco, data_unf in zip(
                self.observables,
                self.bins,
                self.obs_part,
                self.obs_reco,
                self.obs_unf,
            ):
                mults_idx = [
                    i
                    for i, a in enumerate(self.observables)
                    if "multiplicity" in a.tex_label
                ]
                assert (
                    len(mults_idx) == 1
                ), "Caught more than one (or None) 'multiplicity'"
                mults_idx = mults_idx[0]
                if "p_{T, j" in obs.tex_label or "p_{T, q" in obs.tex_label:
                    for i in range(2, 9):
                        hard_mults = (
                            data_hard[self.obs_part[mults_idx] == i]
                            if self.unfold_to == "particle"
                            else data_hard[self.obs_reco[mults_idx] == i]
                        )
                        reco_mults = data_reco[self.obs_reco[mults_idx] == i]
                        unf_mults = data_unf[self.obs_reco[mults_idx] == i]

                        y_hard, y_hard_err = compute_hist_data(
                            bins, hard_mults, bayesian=False
                        )
                        y_reco, y_reco_err = compute_hist_data(
                            bins, reco_mults, bayesian=False
                        )
                        y_unf, y_unf_err = compute_hist_data(
                            bins, unf_mults, bayesian=False
                        )

                        lines = [
                            Line(
                                y=y_reco,
                                y_err=y_reco_err,
                                label="Reco",
                                color=RECO_COLOR,
                            ),
                            Line(
                                y=y_hard,
                                y_err=y_hard_err,
                                label=(
                                    "Particle"
                                    if self.unfold_to == "particle"
                                    else "Parton"
                                ),
                                color=PART_COLOR,
                            ),
                            Line(
                                y=y_unf,
                                y_err=y_unf_err,
                                y_ref=y_hard,
                                label=self.model_name,
                                color=CFM_COLOR,
                            ),
                        ]
                        hist_plot(
                            pp,
                            lines,
                            bins,
                            obs,
                            show_ratios=True,
                            title=f"${i}j$",
                            model_name=self.model_name,
                            truth_name=self.truth_name,
                        )
                        if pickle_file is not None:
                            append_to_pickle({"lines": lines, "bins": bins, "obs": obs}, pickle_file)

    def plot_migration(
        self, file: str, gt_part=False, pickle_file: Optional[str] = None
    ):
        if gt_part:
            obs_part = self.obs_part
            name_part = "Particle" if self.unfold_to == "particle" else "Parton"
        else:
            obs_part = self.obs_unf
            name_part = "Unfold"
        ranges = [[0, 0.4], [0, 0.3], [0, 0.5], [0, 0.5], [0, 0.2], [0, 0.25]]
        ranges += [[0, 0.5] for _ in range(len(self.observables) - len(ranges))]

        with PdfPages(file) as pp:
            for k, (obs, bins, data_hard, data_reco, r) in enumerate(
                zip(self.observables, self.bins, obs_part, self.obs_reco, ranges)
            ):
                if obs.channel is not None and obs.channel not in self.plot_channels:
                    # avoid plotting primary (obs.channel != None) channels not used for unfolding
                    # secondary channels (derived observables -- obs.channel == None) are always plotted
                    continue
                cmap = plt.get_cmap("viridis")
                cmap.set_bad("white")
                h, x, y = np.histogram2d(data_reco, data_hard, bins=(bins, bins))
                h = np.ma.divide(h, np.sum(h, 1, keepdims=True)).filled(
                    0
                )  # normalize so each row sums up to 1 => h.T each column is a distribution: for a given reco event you have a pdf
                h[h == 0] = np.nan
                plt.pcolormesh(
                    bins,
                    bins,
                    h.T,
                    cmap=cmap,
                    rasterized=True,
                    vmin=r[0],
                    vmax=r[1],
                )
                plt.colorbar()

                unit = "" if obs.unit is None else f" [{obs.unit}]"
                plt.title(f"${{{obs.tex_label}}}${unit}")
                plt.xlim(bins[0], bins[-1])
                plt.ylim(bins[0], bins[-1])
                plt.xlabel("Reco")
                plt.ylabel(name_part)
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()

                if pickle_file is not None:
                    append_to_pickle({"h": h, "bins": bins, "ranges": r, "obs": obs}, pickle_file)

    def plot_migration_diff(
        self, file: str, gt_part=False, pickle_file: Optional[str] = None
    ):

        gt_obs_part = self.obs_part
        unf_obs_part = self.obs_unf

        name_part = (
            "Unfold $-$ Particle"
            if self.unfold_to == "particle"
            else "Unfold $-$ Parton"
        )
        ranges = [[0, 5], [0, 5], [0, 5], [0, 5], [0, 5], [0, 5]]
        ranges += [[0, 5] for _ in range(len(self.observables) - len(ranges))]

        with PdfPages(file) as pp:
            for k, (obs, bins, unfold, gt, data_reco, r) in enumerate(
                zip(
                    self.observables,
                    self.bins,
                    unf_obs_part,
                    gt_obs_part,
                    self.obs_reco,
                    ranges,
                )
            ):
                if obs.channel is not None and obs.channel not in self.plot_channels:
                    # avoid plotting primary (obs.channel != None) channels not used for unfolding
                    # secondary channels (derived observables -- obs.channel == None) are always plotted
                    continue
                cmap = plt.get_cmap("bwr")
                cmap.set_bad("white")
                h_unf, x_unf, y_unf = np.histogram2d(
                    data_reco, unfold, bins=(bins, bins)
                )
                # h_unf_norm = np.ma.divide(h_unf, np.sum(h_unf, 1, keepdims=True)).filled(
                #     0
                # )  # normalize so each row sums up to 1 => h.T each column is a distribution: for a given reco event you have a pdf

                h_gt, x_gt, y_gt = np.histogram2d(data_reco, gt, bins=(bins, bins))
                # h_gt_norm = np.ma.divide(h_gt, np.sum(h_gt, 1, keepdims=True)).filled(
                #     0
                # )  # normalize so each row sums up to 1 => h.T each column is a distribution: for a given reco event you have a pdf

                diff = h_unf - h_gt
                # Symmetric dynamic limits
                M = np.nanmax(np.abs(diff))
                norm = mpl.colors.CenteredNorm(vcenter=0.0, halfrange=M)
                plt.pcolormesh(
                    bins,
                    bins,
                    diff.T,
                    cmap=cmap,
                    norm=norm,
                    rasterized=True,
                    # vmin=r[0],
                    # vmax=r[1],
                )
                plt.colorbar()

                unit = "" if obs.unit is None else f" [{obs.unit}]"
                plt.title(f"${{{obs.tex_label}}}${unit}")
                plt.xlim(bins[0], bins[-1])
                plt.ylim(bins[0], bins[-1])
                plt.xlabel("Reco")
                plt.ylabel(name_part)
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()

                if pickle_file is not None:
                    append_to_pickle({"h_ratio": diff, "bins": bins, "ranges": r, "obs": obs}, pickle_file)

    def plot_unfolded_data(
        self,
        file: str,
        obs_unf_data: list,
        bins: list,
        observables: list,
        weights: np.ndarray,
        obs_part_data: Optional[list] = None,
        bkg_closures: Optional[list] = None,
        pickle_file: Optional[str] = None,
        prior_weights: Optional[np.ndarray] = None,
        is_signal: Optional[np.ndarray] = None,
        n_bootstrap: int = 0,
        bootstrap_seed: int = 0,
        prior_variations: Optional[np.ndarray] = None,
    ):
        """Plot unfolded p_data with per-background unfolding-closure cross-checks.

        For each observable the plot shows (all area-normalized):
          - unweighted unfolded p_data reco        (blue solid)    — full, background-inclusive
          - pure signal MC gen truth               (orange dashed) — signal reference
          - per background X in ``bkg_closures``, a dashed pair:
              * TraCFM(p_X) = unfolded p_data reco of the X-subset   (e.g. WWjj purple, ttbar teal)
              * p_X(x_part) = gen truth of the same X-subset         (lighter same-hue dashed)

        The per-background pairs answer "does the (signal-trained) unfolder reproduce each
        background's gen truth?": the ratio panel divides each ``TraCFM(p_X)`` line by its own
        ``p_X(x_part)`` y_ref (per-line area normalization), so a flat ratio ≈ 1 means that
        background unfolds cleanly. The blue line keeps a ratio against the full p_data gen truth
        (``obs_part_data``), whose reference line itself is not drawn.

        The reweighted ``(1-w)·TraCFM`` (red) and the ``p_data(x_part)`` (green) lines are currently
        commented out (see below); re-enable them to restore the signal/full-data closures.

        Args:
            obs_unf_data:  list of (N,) numpy arrays — unfolded p_data observables.
            bins:          list of bin-edge arrays.
            observables:   list of Observable objects (for labels / channel filtering).
            weights:       (K, N) numpy array — signal_approx_weights (used only by the commented
                           reweighted line; retained so it can be re-enabled).
            obs_part_data: optional list of (N,) numpy arrays — p_data gen-level (x_part) truth,
                           aligned with obs_unf_data. Used as the blue line's ratio reference.
            bkg_closures:  optional list of dicts, one per background, each with keys
                           "label" (mathtext, e.g. "WWjj"), "obs_unf" and "obs_part" (lists of
                           (n_X,) arrays aligned with `observables`). Adds the dashed closure pairs.
            pickle_file:   optional path for pickle output.
        """
        # Poisson bootstrap: one multiplicity per p_data event and replica, SHARED by every curve and
        # observable of a replica (signal truth uses the signal rows of the same events), so the
        # correlations between the reweighted curve and the truth carry into the ratio.
        boot = None
        if n_bootstrap > 0 and is_signal is not None and prior_weights is not None and obs_part_data is not None:
            rng = np.random.default_rng(bootstrap_seed)
            boot = rng.poisson(1.0, size=(n_bootstrap, len(prior_weights))).astype(np.float32)
            self.logger.info(f"    Bootstrap: {n_bootstrap} Poisson replicas of {len(prior_weights)} p_data events")
        with PdfPages(file) as pp:
            for idx, (obs, bins_i, data_unf) in enumerate(zip(
                observables, bins, obs_unf_data
            )):
                if obs.channel is not None and obs.channel not in self.plot_channels:
                    continue

                # --- Reweighted unfolded line (red): hist(unfolded, prior*(1-w)) per ensemble
                # member, mean +- std. Without prior_weights, falls back to (1-w) alone (legacy).
                K = weights.shape[0]
                eff = weights * prior_weights[None, :] if prior_weights is not None else weights
                y_unf_k = np.array([
                    compute_hist_data(bins_i, data_unf, bayesian=False, weights=eff[k])[0]
                    for k in range(K)
                ])   # (K, n_bins)
                y_unf_mean = y_unf_k.mean(axis=0)
                y_unf_std  = y_unf_k.std(axis=0)
                # -------------------------------------------------------------------------------

                # Unweighted unfolded line — full p_data, unit weights, single histogram
                y_unf_raw, y_unf_raw_err = compute_hist_data(bins_i, data_unf, bayesian=False)

                # Signal truth reference. When is_signal + prior_weights + obs_part_data are given
                # (p_data unfolding with a classifier dump), the truth is the gen truth of the
                # TRUE-signal rows of p_data itself, alpha-weighted — the exact target the
                # reweighted line should reproduce. Legacy fallback: own-dataset tst signal truth.
                if is_signal is not None and prior_weights is not None and obs_part_data is not None:
                    y_sig, y_sig_err = compute_hist_data(
                        bins_i, obs_part_data[idx][is_signal], bayesian=False,
                        weights=prior_weights[is_signal],
                    )
                else:
                    data_sig = self.obs_part[idx]
                    y_sig, y_sig_err = compute_hist_data(bins_i, data_sig, bayesian=False)

                # Uncertainties of the area-normalized curves (what is plotted). Member spread
                # (classifier training) of the reweighted curve, plus -- with the bootstrap -- the
                # statistical spread of both curves and of their ratio; combined in quadrature.
                bw = np.diff(bins_i)
                norm = lambda y: y / np.sum(bw * y, axis=-1, keepdims=True)
                n_unf_k = norm(y_unf_k)                      # (K, n_bins)
                n_unf, n_sig = norm(y_unf_mean), norm(y_sig)
                # sample variance (ddof=1): with few weights per event (e.g. 3 merged k-fold runs)
                # the population variance would underestimate the classifier spread
                var_member = n_unf_k.var(axis=0, ddof=1) if n_unf_k.shape[0] > 1 else np.zeros(n_unf_k.shape[1])
                ratio_err = None
                if boot is not None:
                    w_unf_mean = eff.mean(axis=0)
                    n_unf_b = norm(_boot_hists(bins_i, data_unf, w_unf_mean, boot))
                    n_sig_b = norm(_boot_hists(bins_i, obs_part_data[idx][is_signal],
                                               prior_weights[is_signal], boot[:, is_signal]))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r_b = n_unf_b / n_sig_b
                        ratio_err = np.sqrt(np.nanvar(np.where(np.isfinite(r_b), r_b, np.nan), axis=0)
                                            + var_member / n_sig**2)
                    var_unf = n_unf_b.var(axis=0) + var_member
                    var_sig = n_sig_b.var(axis=0)
                    var_diff = (n_unf_b - n_sig_b).var(axis=0) + var_member  # correlated difference
                else:
                    var_unf, var_sig = var_member, (y_sig_err / np.sum(bw * y_sig)) ** 2
                    var_diff = var_unf + var_sig
                # Variations of the prior weights (e.g. the ttbar weights of each step-1 reweighter
                # member, classifier weights held fixed): their spread is one more band term.
                if prior_variations is not None:
                    w_cls = weights.mean(axis=0)
                    n_var = norm(np.array([compute_hist_data(bins_i, data_unf, weights=w_cls * pv)[0]
                                           for pv in prior_variations]))
                    var_var = n_var.var(axis=0, ddof=1)
                    var_unf, var_diff = var_unf + var_var, var_diff + var_var
                    if ratio_err is not None:
                        ratio_err = np.sqrt(ratio_err ** 2 + var_var / n_sig ** 2)
                    yld_var = [float(np.sum(w_cls * pv * np.isfinite(data_unf)) / np.sum(prior_weights[is_signal]))
                               for pv in prior_variations]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rel = np.sqrt(var_var) / n_unf
                    self.logger.info(f"    {obs.tex_label}: prior variations -> max rel. shape spread "
                                     f"{np.nanmax(np.where(n_sig > 0.01 * n_sig.max(), rel, np.nan)):.4f}, "
                                     f"yield {min(yld_var):.4f}..{max(yld_var):.4f}")
                # y_err in the curves' own units (hist_plot rescales y and y_err by the same 1/area)
                y_unf_err = np.sqrt(var_unf) * np.sum(bw * y_unf_mean)
                y_sig_err = np.sqrt(var_sig) * np.sum(bw * y_sig)

                # chi2/ndf of the normalized shapes and the yield ratio; logged, not drawn on the plot.
                if is_signal is not None and prior_weights is not None:
                    m = var_diff > 0
                    chi2 = float((((n_unf - n_sig)[m] ** 2) / var_diff[m]).sum() / max(int(m.sum()), 1))
                    yld = float(y_unf_mean.sum() / y_sig.sum()) if y_sig.sum() != 0 else float("nan")
                    self.logger.info(f"    {obs.tex_label}: chi2/ndf={chi2:.2f}  yield={yld:.4f}"
                                     + ("  (bootstrap + member errors)" if boot is not None else ""))

                # p_data gen truth p_data(x_part) — used only as the blue line's ratio reference
                # (its own green line is commented out below). Available when obs_part_data passed in.
                if obs_part_data is not None:
                    y_pdata, y_pdata_err = compute_hist_data(
                        bins_i, obs_part_data[idx], bayesian=False
                    )
                else:
                    y_pdata, y_pdata_err = None, None

                lines = [
                    # signal truth first, so it leads the legend
                    Line(
                        y=y_sig,
                        y_err=y_sig_err,
                        label=r"$p_{\mathrm{signal}}(x_{\mathrm{part}})$",
                        color=SIGNAL_TRUTH_COLOR,  # same color/linestyle as p_signal in the classifier plots
                    ),
                    # Reweighted unfolded line (red)
                    Line(
                        y=y_unf_mean,
                        y_err=y_unf_err,
                        y_ref=y_sig,
                        ratio_err=ratio_err,
                        label=rf"$[1-w(x_{{\mathrm{{reco}}}})]\cdot\mathrm{{{self.model_name}}}(p_{{\mathrm{{data}}}}(x_{{\mathrm{{reco}}}}))$",
                        color=CFM_COLOR,
                    ),
                    # Unweighted unfolded line — full p_data, background-inclusive. Ratio reference is
                    # the full p_data gen truth (green, undrawn) when available, else the signal truth.
                    # Line(
                    #     y=y_unf_raw,
                    #     y_err=y_unf_raw_err,
                    #     y_ref=y_pdata if y_pdata is not None else y_sig,
                    #     label=rf"$\mathrm{{{self.model_name}}}(p_{{\mathrm{{data}}}}(x_{{\mathrm{{reco}}}}))$",
                    #     color=PART_COLOR,
                    # ),
                ]

                # p_data gen truth line (green) — COMMENTED OUT per request; re-enable to draw it.
                # if y_pdata is not None:
                #     lines.append(
                #         Line(
                #             y=y_pdata,
                #             y_err=y_pdata_err,
                #             label=r"$p_{\mathrm{data}}(x_{\mathrm{part}})$",
                #             color=RECO_COLOR,
                #             linestyle="dashed",
                #         ),
                #     )

                # Per-background closure pairs (all dashed): TraCFM(p_X) referenced to its own
                # gen truth p_X(x_part), plus the p_X(x_part) reference line itself.
                for j, bkg in enumerate(bkg_closures or []):
                    c_unf, c_truth = BKG_CLOSURE_COLORS[j % len(BKG_CLOSURE_COLORS)]
                    label = bkg["label"]
                    y_bunf, y_bunf_err = compute_hist_data(bins_i, bkg["obs_unf"][idx], bayesian=False)
                    y_btru, y_btru_err = compute_hist_data(bins_i, bkg["obs_part"][idx], bayesian=False)
                    lines.append(
                        Line(
                            y=y_bunf,
                            y_err=y_bunf_err,
                            y_ref=y_btru,
                            label=rf"$\mathrm{{{self.model_name}}}(p_{{{label}}})$",
                            color=c_unf,
                            linestyle="dashed",
                        ),
                    )
                    lines.append(
                        Line(
                            y=y_btru,
                            y_err=y_btru_err,
                            label=rf"$p_{{{label}}}(x_{{\mathrm{{part}}}})$",
                            color=c_truth,
                            linestyle="dashed",
                        ),
                    )
                hist_plot(
                    pp,
                    lines,
                    bins_i,
                    obs,
                    show_ratios=True,
                    model_name=self.model_name,
                    truth_name=self.truth_name,
                    title=None,
                    title_on_axs=True,
                    # shapes: every line area-normalized (the yields are logged above)
                    no_scale=False,
                    # blue truth drawn first (below), red unfolded line first in the legend
                    legend_order=[1, 0],
                    **UNFOLDED_DATA_LAYOUT.get(obs.tex_label, {}),
                )
                if pickle_file is not None:
                    append_to_pickle(
                        {
                            "lines": lines, "bins": bins_i, "obs": obs,
                            "model_name": self.model_name, "truth_name": self.truth_name,
                            "title": self.title,
                        },
                        pickle_file,
                    )


def _boot_hists(bins, x, w, boot, chunk=50):
    """(B, n_bins) weighted histograms of x, one per bootstrap replica (weights w * boot[b])."""
    x, w = np.asarray(x), np.asarray(w, dtype=np.float64)
    nb = len(bins) - 1
    idx = np.digitize(x, bins) - 1
    ok = (idx >= 0) & (idx < nb)
    idx, w, bt = idx[ok], w[ok], boot[:, ok]
    out = np.empty((boot.shape[0], nb))
    for s in range(0, boot.shape[0], chunk):
        bb = bt[s:s + chunk]
        flat = (np.arange(bb.shape[0])[:, None] * nb + idx[None, :]).ravel()
        out[s:s + chunk] = np.bincount(flat, weights=(bb * w[None, :]).ravel(),
                                       minlength=bb.shape[0] * nb).reshape(bb.shape[0], nb)
    return out


def compute_hist_data(bins: np.ndarray, data: np.ndarray, bayesian=False, weights=None):
    if bayesian:
        hists = np.stack(
            [
                np.histogram(d, bins=bins, density=False, weights=weights)[0]
                for d in data
            ],
            axis=0,
        )
        y = hists[0]
        y_err = np.std(hists, axis=0)
    else:
        y, _ = np.histogram(data, bins=bins, density=False, weights=weights)
        if weights is None:
            y_err = np.sqrt(y)
        else:
            # weighted histogram: err = sqrt(sum w^2) per bin, NOT sqrt(sum w)
            w2, _ = np.histogram(data, bins=bins, density=False, weights=np.asarray(weights) ** 2)
            y_err = np.sqrt(w2)
    return y, y_err


def hist_plot(
    pdf: PdfPages,
    lines: list[Line],
    bins: np.ndarray,
    observable: Observable,
    show_ratios: bool = True,
    title: Optional[str] = None,
    title_on_axs: bool = True,
    subtitle: Optional[str] = None,
    model_name: Optional[str] = "NN",
    truth_name: Optional[str] = "Truth",
    legend_kwargs: Optional[dict] = None,
    rect=(0.15, 0.15, 0.965, 0.94),
    no_scale: bool = False,
    yscale: Optional[str] = None,
    ylabel: Optional[str] = None,
    split_legend: Optional[dict] = None,
    headroom: Optional[float] = None,
    ratio_label: Optional[str] = None,
    legend_order: Optional[list[int]] = None,
    debug=False,
):
    """
    Makes a single histogram plot, used for the observable histograms and clustering
    histograms.
    Args:
        pdf: Multipage PDF object
        lines: List of line objects describing the histograms
        bins: Numpy array with the bin boundaries
        show_ratios: If True, show a panel with ratios
        split_legend: optional {"labels": [...], "loc": ..., "bbox_to_anchor": ...} or a list of
            them; each group's entries get their own legend at "loc" (anchored at bbox_to_anchor,
            in axes coordinates, if given). The remaining entries use legend_kwargs.
        headroom: optional factor on the upper y limit (e.g. 1.25) to make room for the legend
        ratio_label: optional ratio-panel y label (default: model_name over truth_name)
        legend_order: optional indices into the legend entries giving their order in the legend
            (the drawing order stays that of ``lines``); entries not listed follow in their order
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        n_panels = 1 + int(show_ratios)
        fig, axs = plt.subplots(
            n_panels,
            1,
            sharex=True,
            figsize=(6, 4.5),
            gridspec_kw={
                "height_ratios": (12, 3, 1)[:n_panels],
                "hspace": 0.00,
            },
        )
        if n_panels == 1:
            axs = [axs]
        legend_handles = []
        legend_labels = []
        for line in lines:
            if line.vline:
                vline_artist = axs[0].axvline(
                    line.y,
                    label=line.label,
                    color=line.color,
                    linestyle=line.linestyle,
                )
                legend_handles.append(vline_artist)
                legend_labels.append(line.label)
                continue
            integral = np.sum((bins[1:] - bins[:-1]) * line.y)
            scale = 1 / integral if integral != 0.0 else 1.0
            if line.y_ref is not None:
                ref_integral = np.sum((bins[1:] - bins[:-1]) * line.y_ref)
                ref_scale = 1 / ref_integral if ref_integral != 0.0 else 1.0
            if no_scale:
                scale = 1.0
                ref_scale = 1.0

            if debug:
                print("Actual values plotted:", line.y * scale)
            main_artist, band_artist = hist_line(
                axs[0],
                bins,
                line.y * scale,
                line.y_err * scale if line.y_err is not None else None,
                label=line.label,
                color=line.color,
                fill=line.fill,
                linestyle=line.linestyle,
                alpha=line.alpha,
            )
            if band_artist is None:
                legend_handles.append(main_artist)
            else:
                legend_handles.append((band_artist, main_artist))
            legend_labels.append(line.label)

            if line.y_ref is not None:
                ratio = (line.y * scale) / (line.y_ref * ref_scale)
                if line.invert_ratio:
                    ratio = (line.y_ref * ref_scale) / (line.y * scale)
                ratio_isnan = np.isnan(ratio)
                if line.y_err is not None:
                    if len(line.y_err.shape) == 2:
                        ratio_err = (
                            (line.y_err * scale) / (line.y_ref * ref_scale)
                            if not line.invert_ratio
                            else (line.y_ref * ref_scale)
                            * (line.y_err * scale)
                            / (line.y * scale) ** 2
                        )
                        ratio_err[:, ratio_isnan] = 1.0  # bounds, not offsets: collapse onto ratio = 1
                    else:
                        ratio_err = np.sqrt((line.y_err / line.y) ** 2)
                        ratio_err[ratio_isnan] = 0.0
                else:
                    ratio_err = None
                if line.ratio_err is not None:
                    ratio_err = np.array(line.ratio_err, dtype=float)
                    ratio_err[ratio_isnan] = 0.0
                ratio[ratio_isnan] = 1.0
                hist_line(
                    axs[1],
                    bins,
                    ratio,
                    ratio_err,
                    label=None,
                    color=line.color,
                    linestyle=line.ratio_linestyle or line.linestyle,
                    alpha=line.alpha,
                )
            # break

        axs[0].set_ylabel("Normalized" if ylabel is None else ylabel)
        axs[0].set_yscale(observable.yscale if yscale is None else yscale)
        if headroom is not None:
            axs[0].set_ylim(top=axs[0].get_ylim()[1] * headroom)
        if axs[0].get_yscale() == "log":
            force_logy_ticks(axs[0], max_major=3)
        else:
            y_formatter = TwoDecimalScalarFormatter(useMathText=True)
            y_formatter.set_powerlimits((-2, 2))
            axs[0].yaxis.set_major_formatter(y_formatter)
            axs[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins="auto", steps=[1, 2, 5, 10]))
        if title is not None and not title_on_axs:
            corner_text(axs[0], title, "left", "top")
            if subtitle is not None:
                corner_text(axs[0], subtitle, "left", "top", is_subtitle=True)
        elif title is not None and title_on_axs:
            axs[0].set_title(title, loc="right")
        else:
            pass
        if legend_order is not None:
            order = list(legend_order) + [i for i in range(len(legend_handles)) if i not in legend_order]
            legend_handles = [legend_handles[i] for i in order]
            legend_labels = [legend_labels[i] for i in order]
        # each split_legend group's entries go to their own legend at the group's loc
        groups = [split_legend] if isinstance(split_legend, dict) else (split_legend or [])
        for group in groups:
            moved = [i for i, l in enumerate(legend_labels) if l in group["labels"]]
            if not moved:
                continue
            extra = axs[0].legend(
                [legend_handles[i] for i in moved],
                [legend_labels[i] for i in moved],
                loc=group.get("loc", "upper right"),
                bbox_to_anchor=group.get("bbox_to_anchor"),
                frameon=False,
                handlelength=0.8,
            )
            axs[0].add_artist(extra)
            legend_handles = [h for i, h in enumerate(legend_handles) if i not in moved]
            legend_labels = [l for i, l in enumerate(legend_labels) if i not in moved]
        if legend_kwargs is None:
            axs[0].legend(
                legend_handles,
                legend_labels,
                loc="best",
                frameon=False,
                handlelength=0.8,
            )
        else:
            lk = dict(legend_kwargs)
            hl = lk.pop("handlelength", 0.7)
            loc = lk.pop("loc", "best")
            axs[0].legend(legend_handles, legend_labels, loc=loc, frameon=False, handlelength=hl, **lk)

        if show_ratios:
            axs[1].set_ylabel(
                ratio_label if ratio_label is not None
                else f"$\\frac{{\\text{{{model_name}}}}}{{\\text{{{truth_name}}}}}$"
            )
            axs[1].set_yticks([0.9, 1, 1.1])
            axs[1].set_ylim([0.85, 1.15])
            axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.1, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.9, c="black", ls="dotted", lw=0.5)

        unit = "" if observable.unit is None else f" [{observable.unit}]"

        axs[-1].set_xlabel(f"${{{observable.tex_label}}}${unit}")
        axs[-1].set_xscale(observable.xscale)
        axs[-1].set_xlim(bins[0], bins[-1])

        fig.subplots_adjust(left=rect[0], bottom=rect[1], right=rect[2], top=rect[3])
        # Pin ylabel to a fixed figure-space x so all PDF pages align consistently
        # regardless of tick label width. rect[0]-0.05 keeps it in the left margin.
        ylabel_x = -(rect[0] - 0.05) / (rect[2] - rect[0])
        for ax in axs:
            ax.yaxis.set_label_coords(ylabel_x, 0.5)
        if show_ratios and ratio_label is not None and "\n" in ratio_label:
            # a multi-line label is centred on that x: shift it right so its first line stays on the page
            axs[1].yaxis.label.set_linespacing(0.9)
            axs[1].yaxis.set_label_coords(0.75 * ylabel_x, 0.5)
        plt.savefig(pdf, format="pdf")
        plt.close()


def hist_line(
    ax: mpl.axes.Axes,
    bins: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "solid",
    fill: bool = False,
    alpha: float = 1.0,
):
    """
    Plot a stepped line for a histogram, optionally with error bars.
    Args:
        ax: Matplotlib Axes
        bins: Numpy array with bin boundaries
        y: Y values for the bins
        y_err: Y errors for the bins
        label: Label of the line
        color: Color of the line
        linestyle: line style
        fill: Filled histogram
        alpha: opacity multiplier for the line and its error band
    """

    dup_last = lambda a: np.append(a, a[-1])

    if fill:
        main_artist = ax.fill_between(
            bins,
            dup_last(y),
            label=label,
            facecolor=color,
            step="post",
            alpha=0.2 * alpha,
        )
    else:
        (main_artist,) = ax.step(
            bins,
            dup_last(y),
            label=label,
            color=color,
            linewidth=1.5,
            where="post",
            ls=linestyle,
            alpha=alpha,
        )
    band_artist = None
    if y_err is not None:
        if len(y_err.shape) == 2:
            y_low = y_err[0]
            y_high = y_err[1]
        else:
            y_low = y - y_err
            y_high = y + y_err

        ax.step(
            bins,
            dup_last(y_high),
            color=color,
            alpha=0.5 * alpha,
            linewidth=0.5,
            where="post",
        )
        ax.step(
            bins,
            dup_last(y_low),
            color=color,
            alpha=0.5 * alpha,
            linewidth=0.5,
            where="post",
        )
        band_artist = ax.fill_between(
            bins,
            dup_last(y_low),
            dup_last(y_high),
            facecolor=color,
            alpha=0.3 * alpha,
            step="post",
        )
    return main_artist, band_artist


def corner_text(
    ax: mpl.axes.Axes,
    text: str,
    horizontal_pos: str,
    vertical_pos: str,
    is_subtitle: bool = False,
):
    ax.text(
        x=0.95 if horizontal_pos == "right" else 0.05,
        y=(
            0.95 - 0.1 * (is_subtitle)
            if vertical_pos == "top"
            else 0.05 + 0.1 * (is_subtitle)
        ),
        s=text,
        horizontalalignment=horizontal_pos,
        verticalalignment=vertical_pos,
        transform=ax.transAxes,
    )
    # Dummy line for automatic legend placement
    plt.plot(
        0.8 if horizontal_pos == "right" else 0.2,
        0.8 - 0.1 * (is_subtitle) if vertical_pos == "top" else 0.2 + 0.1 * (is_subtitle),
        transform=ax.transAxes,
        color="none",
    )


def append_to_pickle(new_entry, pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except (FileNotFoundError, EOFError):
        pickle_data = []

    pickle_data.append(new_entry)
    with open(pickle_file, "wb") as f:
        pickle.dump(pickle_data, f)

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format her