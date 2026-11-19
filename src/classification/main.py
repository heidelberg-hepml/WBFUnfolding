import os
import glob
import hydra
import shutil
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from utils.logger import setup_logging
from utils.mlflow import mlflow, log_mlflow, LOGGING_ENABLED
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from datasets.classification import ClassificationDatasets
from datasets.dataset import compute_classifier_observables
from classification.models import Classifier, Refiner
from .plots import BackgroundSubtractionPlots


def init_logger(run_dir):
    """
    Initialize the logger. If we are working on a previous run, output to "out_{previos_run_number+1}.log"
    """
    log_files = glob.glob(os.path.join(run_dir, "out_*.log"))
    log_files.sort()
    if len(log_files) > 0:
        run_number = int(log_files[-1].split("_")[-1].split(".")[0])
        run_number += 1
    else:
        run_number = 0
    log_file_path = os.path.join(run_dir, f"out_{run_number:02d}.log")
    return setup_logging(log_file_path)


def fit_temperature(logits, labels, weights):
    """Single-scalar temperature T (>0) minimizing weighted BCE of sigmoid(logits/T) vs labels."""
    logits, labels, weights = logits.double(), labels.double(), weights.double()
    log_t = torch.zeros(1, dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def closure():
        opt.zero_grad()
        loss = (bce(logits / torch.exp(log_t), labels) * weights).sum() / weights.sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


def calibrate_predictions(dataset, cfg, logger):
    """Post-hoc calibration of the classifier logits, fit on the val split only (leak-free w.r.t. the
    APP/dump split), applied to every split in place before density-ratio weights are computed."""
    mode = cfg.train.get("calibration", "none")
    if mode in (None, "none", False):
        return
    preds = dataset.predictions
    W0 = dataset.w_p0["val"].sum()
    W1 = dataset.w_p1["val"].sum()
    w_val = torch.cat([dataset.w_p0["val"] / W0, dataset.w_p1["val"] / W1]).detach().cpu()
    y_val = dataset.labels["val"].detach().cpu()
    is_ens = preds["val"].ndim == 2
    K = preds["val"].shape[0] if is_ens else 1
    if mode == "temperature":
        temps = []
        for k in range(K):
            f_val = (preds["val"][k] if is_ens else preds["val"]).detach().cpu()
            T = fit_temperature(f_val, y_val, w_val)
            temps.append(round(T, 4))
            for split in preds:
                if is_ens:
                    preds[split][k] = preds[split][k] / T
                else:
                    preds[split] = preds[split] / T
        logger.info(f"[calibration] temperature (fit on val) per-member T = {temps}")
    elif mode == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        eps = 1e-6
        yv, wv = y_val.numpy(), w_val.numpy()
        for k in range(K):
            f_val = (preds["val"][k] if is_ens else preds["val"]).detach().cpu().double()
            iso = IsotonicRegression(y_min=eps, y_max=1 - eps, out_of_bounds="clip")
            iso.fit(torch.sigmoid(f_val).numpy(), yv, sample_weight=wv)
            for split in preds:
                cur = preds[split][k] if is_ens else preds[split]
                s = torch.sigmoid(cur.detach().cpu().double()).numpy()
                p_cal = np.clip(iso.predict(s), eps, 1 - eps)
                f_cal = torch.from_numpy(np.log(p_cal / (1 - p_cal))).to(cur.dtype).to(cur.device)
                if is_ens:
                    preds[split][k] = f_cal
                else:
                    preds[split] = f_cal
        logger.info("[calibration] isotonic (fit on val) applied per-member")
    else:
        raise NotImplementedError(f"calibration mode '{mode}' not implemented")


# Every dataset attribute that BackgroundSubtractionPlots / make_plots read. Saved to plot_state.pt
# so all plots of a run can be regenerated (and restyled) later without re-running the models.
PLOT_STATE_ATTRS = [
    "channels", "bins", "obs", "obs_p0", "obs_p1", "obs_p0_ppd", "obs_p1_ppd",
    "labels", "predictions", "weights", "density_ratio", "w_p0", "w_p1",
    "mask_sig_in_p0", "mask_bkg_in_p0", "mask_ttbar_in_p0", "mask_WWjj_in_p0",
    "mask_ttbar_in_p1", "mask_WWjj_in_p1", "p0_ppd", "p1_ppd",
    "refiner_weights", "refiner_predictions", "refiner_y",
    # raw events + derived-channel map: lets a replot recompute the CURRENT observable list
    "p0", "p1", "derived_channels",
]


def _to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(v) for v in x)
    return x


class PlotStateDataset:
    """Stand-in for ClassificationDatasets rebuilt from plot_state.pt (plot attributes only)."""

    compute_weights = ClassificationDatasets.compute_weights

    def __init__(self, attrs):
        self.__dict__.update(attrs)


def save_plot_state(logger, run_dir, dataset, plot_args):
    state = {
        "dataset": {a: _to_cpu(getattr(dataset, a)) for a in PLOT_STATE_ATTRS if hasattr(dataset, a)},
        "plot_args": _to_cpu(plot_args),
    }
    path = os.path.join(run_dir, "plot_state.pt")
    torch.save(state, path)
    logger.info(f"    Saved plot inputs -> {path}")


def load_plot_state(run_dir, refresh_observables=True, splits=None, include_ppd=True):
    """Returns (dataset stand-in, plot_args) such that make_plots(logger, out_dir, dataset, **plot_args)
    replots the run. With refresh_observables (and raw events saved), the observables are recomputed
    from the current init_observables, so observables added after the run show up too."""
    state = torch.load(os.path.join(run_dir, "plot_state.pt"), map_location="cpu", weights_only=False)
    dataset = PlotStateDataset(state["dataset"])
    if refresh_observables and hasattr(dataset, "p0") and hasattr(dataset, "p1"):
        ClassificationDatasets.init_observables(dataset)
        for split in list(dataset.obs.keys()):
            if splits is not None and split not in splits:
                continue
            res = compute_classifier_observables(dataset, split=split, include_ppd=include_ppd)
            dataset.bins[split], dataset.obs[split], dataset.obs_p0[split], dataset.obs_p1[split] = res[:4]
            if include_ppd:
                dataset.obs_p0_ppd[split], dataset.obs_p1_ppd[split] = res[4:]
    return dataset, state["plot_args"]


def make_plots(
    logger, run_dir, dataset, background_mode, plots_kwargs, n_ensemble, active_splits,
    member_preds_list=None, stacked_preds=None, member_losses_list=None, use_refiner=False,
):
    logger.info(f"Starting plots")

    plots_cls = {
        "subtraction": BackgroundSubtractionPlots,
    }[background_mode]

    if n_ensemble == 1:
        plots = plots_cls(logger, dataset, **plots_kwargs)
        if plots_kwargs.get("losses") is not None:
            logger.info(f"    Plotting train metrics")
            plots.plot_train_metrics(
                os.path.join(run_dir, f"train_metrics.pdf"), logy=False
            )
        logger.info(f"    Plotting classifier")
        plots.plot_classifier(os.path.join(run_dir, f"class_trn.pdf"), split="trn")
        if "tst" in active_splits:
            plots.plot_classifier(os.path.join(run_dir, f"class_tst.pdf"), split="tst")
    else:
        logger.info(f"    Plotting classifier ({n_ensemble} ensemble members)")
        for split_key, fname in (
            [("trn", "class_trn.pdf")]
            + ([("tst", "class_tst.pdf")] if "tst" in active_splits else [])
        ):
            with PdfPages(os.path.join(run_dir, fname)) as pp:
                for i in range(n_ensemble):
                    dataset.predictions = {s: member_preds_list[i][s] for s in active_splits}
                    dataset.compute_weights()
                    plots_cls(logger, dataset, **plots_kwargs).plot_classifier(pp, split=split_key)
        # Restore ensemble (stacked) state for observable plots
        dataset.predictions = stacked_preds
        dataset.compute_weights()
        plots = plots_cls(logger, dataset, **plots_kwargs)
        # Per-member train metrics: pages loss_1..loss_K then grad_1..grad_K
        if member_losses_list is not None and any(l is not None for l in member_losses_list):
            logger.info(f"    Plotting per-member train metrics ({n_ensemble} members)")
            plots.plot_train_metrics_ensemble(
                os.path.join(run_dir, "train_metrics.pdf"), member_losses_list, logy=False
            )
    if use_refiner and hasattr(dataset, "refiner_weights"):
        logger.info(f"    Plotting refiner")
        plots.plot_refiner(os.path.join(run_dir, f"refiner_trn.pdf"), split="trn")
        if "tst" in active_splits:
            plots.plot_refiner(os.path.join(run_dir, f"refiner_tst.pdf"), split="tst")
        logger.info(f"    Plotting refined observables")
        plots.plot_refd_observables(
            os.path.join(run_dir, "refd_observables_trn.pdf"), split="trn"
        )
    logger.info(f"    Plotting observables")
    plots.plot_observables(
        os.path.join(run_dir, f"observables_trn.pdf"),
        pickle_file=os.path.join(run_dir, f"observables_trn.pkl"),
        split="trn",
    )
    if "tst" in active_splits:
        plots.plot_observables(
            os.path.join(run_dir, f"observables_tst.pdf"),
            pickle_file=os.path.join(run_dir, f"observables_tst.pkl"),
            split="tst",
        )
    if hasattr(dataset, "obs_p0_ppd") and hasattr(dataset, "obs_p1_ppd"):
        logger.info(f"    Plotting ppd observables")
        plots.plot_observables_ppd(
            os.path.join(run_dir, f"observables_ppd_trn.pdf"),
            pickle_file=os.path.join(run_dir, f"observables_ppd_trn.pkl"),
            split="trn",
        )
        if "tst" in active_splits:
            plots.plot_observables_ppd(
                os.path.join(run_dir, f"observables_ppd_tst.pdf"),
                pickle_file=os.path.join(run_dir, f"observables_ppd_tst.pkl"),
                split="tst",
            )


def main(cfg: DictConfig):
    base_dir = hydra.utils.get_original_cwd()
    """
    Main function. Initialize directories and load params, initialize logger, execute the run
    """

    ### INITIALIZE DIRECTORIES AND LOAD PARAMS ###
    if cfg.run.type == "train":
        if not cfg.train.warm_start:
            # Initialize run directory
            results_dir = os.path.join(base_dir, "results")
            if cfg.model.model == "Classifier":
                nettype = {
                    "MLP": "MLPClassifier",
                    "Transformer": "TransformerClassifier",
                }[cfg.model.network]
            else:
                nettype = {"MLP": "CFM", "Transformer": "TraCFM"}[cfg.model.network]
            run_name = (
                nettype
                + "/"
                + datetime.now().strftime("%m%d_%H%M%S")
                + "-"
                + cfg.run.name
                if cfg.run.name is not None
                else nettype + "/" + datetime.now().strftime("%m%d_%H%M%S")
            )
            run_dir = os.path.join(results_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
        else:
            # Load already existing run directory
            run_dir = cfg.run.path
            old_dir = os.path.join(run_dir, "old")
            os.makedirs(old_dir, exist_ok=True)
            old_src = os.path.join(old_dir, "src")
            if os.path.exists(old_src):
                shutil.rmtree(old_src)
            for item in os.listdir(run_dir):
                item_path = os.path.join(run_dir, item)
                if (
                    (
                        os.path.isfile(item_path)
                        and item.startswith("out_")
                        and item.endswith(".log")
                    )
                    or item == "model"
                    or item.startswith("model_")  # ensemble member dirs
                    or (
                        os.path.isfile(item_path)
                        and item.startswith("config")
                        and item.endswith(".yaml")
                    )
                ):
                    continue
                if "old" in item:
                    continue
                os.rename(item_path, os.path.join(old_dir, item))

        # This is to keep plot as predefined run type
        cfg_to_save = OmegaConf.to_container(cfg, resolve=True)
        cfg_to_save["run"]["path"] = run_dir
        cfg_to_save["run"]["type"] = "plot"
        config_file = os.path.join(run_dir, "config.yaml")
        with open(config_file, "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.create(cfg_to_save)))
        shutil.copytree(
            os.path.join(base_dir, "src"),
            os.path.join(run_dir, "src"),
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "*__pycache__",
                "*egg-info",
                "playground",
                "template_files",
                "*__init__.py",
            ),
        )

    elif cfg.run.type == "plot":
        run_dir = cfg.run.path
        old_dir = os.path.join(run_dir, "old")
        os.makedirs(old_dir, exist_ok=True)
        old_src = os.path.join(old_dir, "src")
        if os.path.exists(old_src):
            shutil.rmtree(old_src)
        for item in os.listdir(run_dir):
            item_path = os.path.join(run_dir, item)
            if (
                (
                    os.path.isfile(item_path)
                    and item.startswith("out_")
                    and item.endswith(".log")
                )
                or item == "model"
                or item.startswith("model_")  # ensemble member dirs
                or (
                    os.path.isfile(item_path)
                    and item.startswith("config")
                    and item.endswith(".yaml")
                )
            ):
                continue
            if "old" in item:
                continue
            os.rename(item_path, os.path.join(old_dir, item))
    else:
        raise NotImplementedError(f"Run type {cfg.run.type} not recognized")

    logger = init_logger(run_dir)
    try:
        run(logger, run_dir, cfg)
    except Exception as e:
        logger.error(e, exc_info=True)


def run(logger, run_dir, cfg: DictConfig):
    """
    Run training and/or plotting of the model with the given parameters.
    """
    # seeding
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        logger.warning(f"Setting seed to {cfg.seed}.")
    if torch.cuda.is_available():
        if cfg.seed is not None:
            torch.cuda.manual_seed_all(cfg.seed)
            logger.warning(f"Setting CUDA seed to {cfg.seed}.")
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    ### INITALIZE RUN ###
    logger.info(f"Starting {cfg.run.type} run in {run_dir}")
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    logger.info(f"Device {device}")

    # INITIALIZE DATASET AND PREPROCESSING ###

    dataset = ClassificationDatasets(logger, cfg.dataset)
    active_splits = [s for s in ["trn", "val", "tst"] if len(dataset.p0[s]) > 0]

    ### INITIALIZE MODEL AND DATALOADERS ###
    dims_in = len(dataset.channels)
    logger.info("\n")
    logger.info(
        f"Building model { {'MLP' : 'MLPClassifier', 'Transformer' : 'TransformerClassifier'}[cfg.model.network] } with dims_in = {dims_in}."
    )
    model_path = os.path.join(run_dir, "model")

    model = eval(cfg.model.model)(
        logger=logger,
        cfg=cfg,
        dims_in=dims_in,
        model_path=model_path,
        device=device,
    ).to(device)

    logger.info(
        f"    Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    if LOGGING_ENABLED:
        log_mlflow(
            logger,
            "num_parameters",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            kind="param",
        )
    model.init_dataloaders(dataset)

    ### TRAIN/PLOT ###
    n_ensemble = cfg.train.get("ensemble", 1)

    if n_ensemble == 1:
        if cfg.run.type == "train":
            # TRAIN MODEL
            if not cfg.train.warm_start:
                os.makedirs(model_path, exist_ok=True)
            else:
                logger.info(f"Warm starting from {cfg.train.checkpoint}")
                model.load(cfg.train.get("checkpoint", "final"))
            model.run_training()
            if cfg.evaluate.get("evaluate_best", False):
                logger.info(f"Loading best model from {model_path}/best.pth")
                try:
                    model.load("best")
                except FileNotFoundError:
                    logger.warning(
                        f"Best model not found in {model_path}/best.pth. Loading final model instead."
                    )
                    model.load("final")
            else:
                try:
                    model.load("final")
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Final model not found in {model_path}/final.pth. Please train the model first."
                    )
        elif cfg.run.type == "plot":
            # LOAD AND PLOT MODEL
            if cfg.evaluate.get("model_name", None) is not None:
                model_name = cfg.evaluate.model_name
                try:
                    model.load(model_name)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Model {model_name} not found in {model_path}/{model_name}.pth. Please train the model first."
                    )
            else:
                model_name = (
                    "final" if not cfg.evaluate.get("evaluate_best", False) else "best"
                )
                if model_name == "best":
                    try:
                        model.load("best")
                    except FileNotFoundError:
                        logger.warning(
                            f"Best model not found in {model_path}/best.pth. Loading final model instead."
                        )
                        model.load("final")
                else:
                    model.load("final")

        ### EVALUATE MODEL ###
        preds = {}
        labels = {}
        for split in active_splits:
            (
                preds[split],
                labels[split],
            ) = model.evaluate(split)
        dataset.predictions = preds
        dataset.labels = labels
        calibrate_predictions(dataset, cfg, logger)
        dataset.compute_weights()

    else:
        ### ENSEMBLE TRAINING/LOADING ###
        logger.info(f"Running ensemble of {n_ensemble} classifiers.")
        member_preds_list = []  # list of n_ensemble dicts {split: 1-D tensor}
        member_losses_list = []  # per-member losses dict (for train_metrics pages); None when re-plotting
        member_labels = None

        # train.kfold: cross-fitting on the trn split. The trn p0/p1 events are dealt into
        # n_ensemble folds; member i trains on the other folds and is the ONLY member whose output is
        # used for fold-i events, so the trn weights (the ones dumped and applied) are out-of-fold.
        # val/tst are never trained on, so there all members are used as usual.
        kfold = bool(cfg.train.get("kfold", False)) and "trn" in active_splits
        if kfold:
            # kfold_seed: independent fold assignment on the SAME dataset (cfg.seed fixes the events
            # and their order), so several k-fold runs can be merged event by event
            kfold_seed = int(cfg.train.get("kfold_seed", 0))
            g = torch.Generator().manual_seed((cfg.seed if cfg.seed is not None else 0) + 1000 * kfold_seed)
            trn_full = {a: getattr(dataset, a)["trn"] for a in ("p0_ppd", "p1_ppd", "w_p0", "w_p1")}
            fold_p0 = torch.randperm(trn_full["p0_ppd"].shape[0], generator=g) % n_ensemble
            fold_p1 = torch.randperm(trn_full["p1_ppd"].shape[0], generator=g) % n_ensemble
            fold_trn = torch.cat([fold_p0, fold_p1])  # = order of the trn loader: p0 rows, then p1 rows
            oof_trn = None
            logger.info(f"  k-fold cross-fitting on trn: {n_ensemble} folds "
                        f"(p0 {trn_full['p0_ppd'].shape[0]}, p1 {trn_full['p1_ppd'].shape[0]} events)")

        for i in range(n_ensemble):
            logger.info(f"  Ensemble member {i + 1}/{n_ensemble}")
            if kfold:
                for a, fold in (("p0_ppd", fold_p0), ("p1_ppd", fold_p1), ("w_p0", fold_p0), ("w_p1", fold_p1)):
                    getattr(dataset, a)["trn"] = trn_full[a][(fold != i).to(trn_full[a].device)]
            if cfg.seed is not None:
                # kfold_seed > 0 also gives the members new initializations (unchanged for 0)
                member_seed = cfg.seed + i + 100 * int(cfg.train.get("kfold_seed", 0))
                torch.manual_seed(member_seed)
                np.random.seed(member_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(member_seed)

            member_path = os.path.join(run_dir, f"model_{i}")
            member = eval(cfg.model.model)(
                logger=logger,
                cfg=cfg,
                dims_in=dims_in,
                model_path=member_path,
                device=device,
            ).to(device)
            member.init_dataloaders(dataset)

            if cfg.run.type == "train":
                os.makedirs(member_path, exist_ok=True)
                member.run_training()
                # capture this member's loss/grad-norm history for the per-member train_metrics pages
                member_losses_list.append(getattr(member, "losses", None))
                # kept next to the checkpoint so plot-mode re-evaluations still have the curves
                torch.save(getattr(member, "losses", None), os.path.join(member_path, "losses.pt"))
                # honor evaluate_best like the single-model path (main.py:214-222): the ensemble
                # previously always dumped "final", ignoring the early-stopped "best" checkpoint.
                if cfg.evaluate.get("evaluate_best", False):
                    try:
                        member.load("best")
                    except FileNotFoundError:
                        logger.warning(
                            f"Best model not found in {member_path}/best.pth. Loading final instead."
                        )
                        member.load("final")
                else:
                    member.load("final")
            else:
                member.load(cfg.evaluate.get("model_name", "final"))
                losses_path = os.path.join(member_path, "losses.pt")
                member_losses_list.append(
                    torch.load(losses_path, weights_only=False) if os.path.isfile(losses_path) else None
                )

            if kfold:  # restore the full trn split for inference
                for a in trn_full:
                    getattr(dataset, a)["trn"] = trn_full[a]
                member.init_dataloaders(dataset)
            p, l = {}, {}
            for split in active_splits:
                p[split], l[split] = member.evaluate(split)
            member_preds_list.append(p)
            member_labels = l
            if kfold:
                if oof_trn is None:
                    oof_trn = torch.full_like(p["trn"], float("nan"))
                held = (fold_trn == i).to(p["trn"].device)
                oof_trn[held] = p["trn"][held]

        # Stack into shape (n_ensemble, N) per split
        stacked_preds = {
            split: torch.stack([mp[split] for mp in member_preds_list], dim=0)
            for split in active_splits
        }
        if kfold:
            assert not torch.isnan(oof_trn).any(), "k-fold: some trn events got no out-of-fold prediction"
            stacked_preds["trn"] = oof_trn.unsqueeze(0)  # (1, N_trn): one held-out output per event
            logger.info("  trn predictions = out-of-fold (each event from the member that never saw it)")
        dataset.predictions = stacked_preds
        dataset.labels = member_labels
        calibrate_predictions(dataset, cfg, logger)
        dataset.compute_weights()

    ### VAL-SPLIT YIELD-RECOVERY (selection metric; leak-free, never touches the APP/dump split) ###
    # Same estimator as the deficit we are attacking: sum(prior * (1-w)) over p0 vs the true signal
    # yield sum(prior[is_signal]). Reported on val so hyperparameters are chosen without tuning to APP.
    # Logged for trn too and for tst (held-out file rows): trn (in-sample, where the weights get
    # applied) vs val/tst (out-of-sample) is the overfitting check -- they should agree.
    if hasattr(dataset, "weights") and hasattr(dataset, "mask_sig_in_p0"):
        for ys in [s for s in ("val", "trn", "tst") if s in dataset.weights]:
            mp0 = (dataset.labels[ys] == 0).cpu()
            wv = dataset.weights[ys].detach().cpu()
            wv = wv[..., mp0] if wv.ndim == 2 else wv[mp0]
            if wv.ndim == 1:
                wv = wv.unsqueeze(0)
            saw_v = 1.0 - wv
            prior_v = dataset.w_p0[ys].detach().cpu()
            is_sig_v = dataset.mask_sig_in_p0[ys].cpu()
            target_v = float(prior_v[is_sig_v].sum())
            sums_v = (saw_v * prior_v).sum(dim=1)
            if target_v != 0:
                ratio_v = float(sums_v.mean() / target_v)
                per_member = [round(float(s / target_v), 4) for s in sums_v]
                logger.info(
                    f"[{ys} yield-recovery] sum(prior*w)/(N_D-N_B) = {ratio_v:.4f}  "
                    f"target={target_v:.5f}  per-member={per_member}"
                )

    ### SAVE WEIGHTS FOR UNFOLDER ###
    # dump_split is the held-out APP split ("tst") when dataset.holdout_frac > 0, else "trn".
    # Dumping the APP split makes p_data disjoint from the classifier's training set.
    dump_split = getattr(dataset, "dump_split", "trn")
    # Truth level selects reco paths + the truth dump filename (gen -> p_data_gen.npy [particle],
    # parton -> p_data_parton.npy). Must match the unfolder's unfold_to it will reweight.
    truth_level = cfg.dataset.get("truth_level", "gen")
    reco_key = ("path_reco_corresponding_to_parton" if truth_level == "parton"
                else "path_reco_corresponding_to_gen")
    bkg_level = cfg.dataset.get("background_truth_level", None) or truth_level
    bkg_reco_key = ("path_reco_corresponding_to_parton" if bkg_level == "parton"
                    else "path_reco_corresponding_to_gen")
    tt_level = cfg.dataset.get("ttbar_truth_level", None) or bkg_level
    tt_reco_key = ("path_reco_corresponding_to_parton" if tt_level == "parton"
                   else "path_reco_corresponding_to_gen")
    if cfg.evaluate.get("save_weights", False) and dump_split in active_splits:
        logger.info(f"Saving per-event reweighting weights for the unfolder (split='{dump_split}')...")
        mask_p0 = dataset.labels[dump_split] == 0
        w = dataset.weights[dump_split]
        # w has shape (N_all,) for single model or (K, N_all) for ensemble;
        # slice only the p0 (label==0) events to match dataset.p0[dump_split] row order.
        w_sc_p0 = w[..., mask_p0] if w.ndim == 2 else w[mask_p0]
        sig_approx = 1.0 - w_sc_p0
        if sig_approx.ndim == 1:
            sig_approx = sig_approx.unsqueeze(0)  # (1, N_p0) for single model
        out = {
            # (K, N_p0): column i = signal-approx weight for p_data event i
            "signal_approx_weights": sig_approx.cpu(),
            # row in original HWW npy array for each p_data event (-1 = background)
            "hww_index": dataset.idx_p0[dump_split].cpu(),
            # convenience boolean: True where the event is real signal (hww_index >= 0)
            "is_signal": dataset.mask_sig_in_p0[dump_split].cpu(),
            # per-process background masks in the same row order, so the unfolder can split the
            # background into WWjj / ttbar without value-matching against the source files.
            "mask_wwjj_in_p0": dataset.mask_WWjj_in_p0[dump_split].cpu(),
            "mask_ttbar_in_p0": dataset.mask_ttbar_in_p0[dump_split].cpu(),
            # fiducial prior weights in the same row order
            "prior_weights": dataset.w_p0[dump_split].cpu(),
            "n_ensemble": int(sig_approx.shape[0]),
            "seed": cfg.seed,
            "dump_split": dump_split,
            "holdout_frac": float(cfg.dataset.get("holdout_frac", 0.0)),
            "channels": list(dataset.channels),
            "background_mode": cfg.dataset.get("background_mode", "subtraction"),
            # reco paths for the truth level this classifier was trained on (parton -> reco-hard).
            "hww_path": cfg.dataset.HWW[reco_key],
            "ttbar_path": cfg.dataset.ttbar[tt_reco_key],
            "wwjj_path": cfg.dataset.WWjj[bkg_reco_key],
            "truth_level": truth_level,
            "background_truth_level": bkg_level,
            "ttbar_truth_level": tt_level,
            "ttbar_event_weights": cfg.dataset.get("ttbar_event_weights", None),
        }
        # explicit_split mode: source position per row (signal -> row in the pool's HWW slice,
        # background -> row in the pool's [ttbar ++ WWjj] block) + per-pool counts/ranges, so a
        # downstream filter can map every p_data row back to its global source-file row.
        if getattr(dataset, "src_pos_p0", None) is not None:
            out["src_pos_p0"] = dataset.src_pos_p0[dump_split].cpu()
        if getattr(dataset, "pool_counts", None) is not None:
            out["pool_counts"] = dataset.pool_counts.get(dump_split)
        if getattr(dataset, "ttbar_row_perm", None) is not None:
            # ttbar rows were shuffled before splitting: file row = ttbar_row_perm[position]
            out["ttbar_row_perm"] = dataset.ttbar_row_perm.cpu()
        if cfg.dataset.get("explicit_split", None) is not None:
            out["explicit_split"] = OmegaConf.to_container(cfg.dataset.explicit_split, resolve=True)
        weights_path = os.path.join(run_dir, "unfolding_weights_trn.pt")
        torch.save(out, weights_path)
        logger.info(
            f"    Saved signal_approx_weights {tuple(sig_approx.shape)} -> {weights_path}"
        )
        logger.info(
            f"    N_p0={sig_approx.shape[1]}, "
            f"N_signal={out['is_signal'].sum().item()}, "
            f"N_background={(~out['is_signal']).sum().item()}"
        )
        # Save raw EPxPyPz reco/gen for the same p_data events so the unfolder
        # can load them through its own WBF dataset class.
        if dataset.p0_raw_reco is not None:
            reco_path = os.path.join(run_dir, "p_data_reco.npy")
            np.save(reco_path, dataset.p0_raw_reco[dump_split].cpu().numpy())
            logger.info(f"    Saved p_data_reco {tuple(dataset.p0_raw_reco[dump_split].shape)} -> {reco_path}")
        if dataset.p0_raw_gen is not None:
            truth_fname = "p_data_parton.npy" if truth_level == "parton" else "p_data_gen.npy"
            gen_path = os.path.join(run_dir, truth_fname)
            np.save(gen_path, dataset.p0_raw_gen[dump_split].cpu().numpy())
            logger.info(f"    Saved {truth_fname} {tuple(dataset.p0_raw_gen[dump_split].shape)} -> {gen_path}")

    ### NEURAL REFINER (NRSW) ###
    use_refiner = cfg.get("refiner", {}).get("use", False)
    if use_refiner:
        logger.info("Setting up Neural Refiner (NRSW)...")
        dataset.setup_refiner_data()
        refiner_model_path = os.path.join(run_dir, "model_refiner")
        refiner = Refiner(
            logger=logger,
            cfg=cfg,
            dims_in=dims_in,
            model_path=refiner_model_path,
            device=device,
        ).to(device)
        refiner.init_dataloaders(dataset)
        if cfg.run.type == "train":
            os.makedirs(refiner_model_path, exist_ok=True)
            refiner.run_training()
            refiner.load("final")
        else:
            refiner.load(cfg.evaluate.get("model_name", "final"))
        refiner_preds = {}
        for split in active_splits:
            preds_r, _ = refiner.evaluate(split)
            refiner_preds[split] = preds_r
        dataset.refiner_predictions = refiner_preds
        dataset.compute_refiner_weights()

    # dataset.apply_preprocessing(
    #     reverse=True
    # )  # reverse preprocessing to obtain unfolded data

    if cfg.evaluate.save_samples:
        raise NotImplementedError("Saving samples not implemented yet.")
        # logger.info(f"Saving unfolded samples to {run_dir}/samples")
        # os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
        # torch.save(
        #     dataset.unfolded_ppd,
        #     os.path.join(run_dir, f"samples/unfolded_ppd.pt"),
        # )
        # torch.save(
        #     dataset.unfolded,
        #     os.path.join(run_dir, f"samples/unfolded.pt"),
        # )
    if cfg.evaluate.save_data:
        raise NotImplementedError("Saving data not implemented yet.")
        # logger.info(f"Saving data to {run_dir}/data")
        # os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
        # torch.save(
        #     {"part": dataset.part, "reco": dataset.reco},
        #     os.path.join(run_dir, f"samples/data_raw.pt"),
        # )
        # torch.save(
        #     {"part": dataset.part_ppd, "reco": dataset.reco_ppd},
        #     os.path.join(run_dir, f"samples/data_ppd.pt"),
        # )

    ### COMPUTE OBSERVABLES ###
    logger.info("Computing observables")
    dataset.bins = {}
    dataset.obs = {}
    dataset.obs_p0 = {}
    dataset.obs_p1 = {}
    dataset.obs_p0_ppd = {}
    dataset.obs_p1_ppd = {}
    for split in [s for s in ["trn", "tst"] if s in active_splits]:
        (
            dataset.bins[split],
            dataset.obs[split],
            dataset.obs_p0[split],
            dataset.obs_p1[split],
            dataset.obs_p0_ppd[split],
            dataset.obs_p1_ppd[split],
        ) = compute_classifier_observables(dataset, split=split, include_ppd=True)

    ### MAKE PLOTS ###
    plots_kwargs = dict(
        losses=model.losses if (n_ensemble == 1 and hasattr(model, "losses")) else None,
        debug=cfg.plot.get("debug", False),
        model_name=model.name if n_ensemble == 1 else cfg.evaluate.get("model_name", "final"),
        title=None,
        show_class_components=cfg.plot.get("show_class_components", None),
    )
    plot_args = dict(
        background_mode=cfg.dataset.get("background_mode", "subtraction"),
        plots_kwargs=plots_kwargs,
        n_ensemble=n_ensemble,
        active_splits=list(active_splits),
        member_preds_list=member_preds_list if n_ensemble > 1 else None,
        stacked_preds=stacked_preds if n_ensemble > 1 else None,
        member_losses_list=member_losses_list if n_ensemble > 1 else None,
        use_refiner=use_refiner,
    )
    if cfg.evaluate.get("save_plot_state", True):
        save_plot_state(logger, run_dir, dataset, plot_args)
    make_plots(logger, run_dir, dataset, **plot_args)

    if device == torch.device("cuda"):
        max_used = torch.cuda.max_memory_allocated()
        max_total = torch.cuda.mem_get_info()[1]
        logger.info(
            f"GPU RAM information: max_used = {max_used/1e9:.3} GB, max_total = {max_total/1e9:.3} GB"
        )
    logger.info("Run finished")
