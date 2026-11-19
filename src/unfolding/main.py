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
from datasets.WBF import WBF
from datasets.dataset import compute_observables
from unfolding.models import CFM, _condition_on_ids
from .plots import Plots


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


def main(cfg: DictConfig):
    base_dir = hydra.utils.get_original_cwd()
    """
    Main function. Initialize directories and load params, initialize logger, execute the run
    """

    ### INITIALIZE DIRECTORIES AND LOAD PARAMS ###
    if cfg.run.type == "train":
        resuming = bool(cfg.train.get("resume", False) and cfg.run.get("path", None)
                        and os.path.isdir(cfg.run.path))
        if not cfg.train.warm_start and not resuming:
            # Initialize run directory
            results_dir = os.path.join(base_dir, "results")
            # run.results_subdir (e.g. "simulation_dependence") replaces the default TraCFM/CFM folder
            run_name = (
                (cfg.run.get("results_subdir", None) or {"MLP": "CFM", "Transformer": "TraCFM"}[cfg.model.network])
                + "/"
                + datetime.now().strftime("%y%m%d_%H%M%S")
                + "-"
                + cfg.run.name
                if cfg.run.name is not None
                else {"MLP": "CFM", "Transformer": "TraCFM"}[cfg.model.network]
                + "/"
                + datetime.now().strftime("%y%m%d_%H%M%S")
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
        cfg_to_save["train"]["resume"] = False  # a later plot/eval run must not try to resume
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
        # a sim_dep-only run regenerates just its task folders: leave every other output in place
        sim_dep_names = {t["name"] for t in cfg.evaluate.get("sim_dep", None) or []}
        for item in os.listdir(run_dir):
            item_path = os.path.join(run_dir, item)
            if sim_dep_names and item not in sim_dep_names:
                continue
            if sim_dep_names and os.path.exists(os.path.join(old_dir, item)):
                shutil.rmtree(os.path.join(old_dir, item))  # a folder cannot be renamed over its older copy
            if (
                (
                    os.path.isfile(item_path)
                    and item.startswith("out_")
                    and item.endswith(".log")
                )
                or item == "model"
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
    else:
        if cfg.dataset.combined.use:
            cfg.seed = 0
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

            logger.warning(
                f"No seed provided, setting seed to {cfg.seed} for combined dataset, to ensure reproducibility."
            )
        else:
            pass
            logger.warning("No seed provided, results may not be reproducible.")
    if torch.cuda.is_available():
        if cfg.seed is not None:
            torch.cuda.manual_seed_all(cfg.seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    ### INITALIZE RUN ###
    logger.info(f"Starting {cfg.run.type} run in {run_dir}")
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    logger.info(f"Device {device}")

    ### INITIALIZE MLFLOW ###
    if LOGGING_ENABLED and device == "cuda" and cfg.run.type == "train":
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(
            "results/TraCFM" if cfg.model.network == "Transformer" else "results/CFM"
        )
        mlflow.start_run(run_name=cfg.run.name if cfg.run.name is not None else run_dir)
        # flatten and log top‐level params
        for key, val in {
            **cfg.train,
            **cfg.model,
            **cfg.dataset,
            **cfg.dataset.parameterization.naive,
            **cfg.dataset.parameterization.transverse_mass,
            **cfg.dataset.special,
            **{
                "run_type": cfg.run.type,
                "run_name": cfg.run.name,
            },
        }.items():
            log_mlflow(logger, key, str(val), kind="param")

    # INITIALIZE DATASET AND PREPROCESSING ###
    logger.info(f"Dataset: {cfg.dataset.process}")
    param_names = [
        p for p in cfg.dataset.parameterization if cfg.dataset.parameterization[p].use
    ]
    logger.info(f"    Using parameterization(s): {', '.join(param_names)}")
    dataset = WBF(logger, cfg.dataset)
    dataset.apply_preprocessing()
    dataset.split_data()

    ### INITIALIZE MODEL AND DATALOADERS ###
    dims_x = len(dataset.channels)
    dims_c = len(dataset.cond_channels)
    logger.info(
        f"Building model { {'MLP' : 'CFM', 'Transformer' : 'TraCFM'}[cfg.model.network] } with dims_x = {dims_x}, and dims_c = {dims_c}."
    )
    model_path = os.path.join(run_dir, "model")

    model = eval(cfg.model.model)(
        logger=logger,
        cfg=cfg,
        dims_x=dims_x,
        dims_c=dims_c,
        model_path=model_path,
        device=device,
    ).to(device)
    if (dataset.cond_channels != dataset.channels and set(dataset.channels) <= set(dataset.cond_channels)
            and hasattr(model.network, "compute_embedding")):
        # balanced parameterization: give each unfolded feature the position one-hot of the same
        # feature in the (larger) condition
        model.network.x_positions = [dataset.cond_channels.index(c) for c in dataset.channels]

    logger.info(
        f"    Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    logger.info(f"    Residual Dropout rate: {cfg.model.get('resid_pdrop', 0.0)}")
    logger.info(f"    Attention Dropout rate: {cfg.model.get('attn_pdrop', 0.0)}")

    if LOGGING_ENABLED:
        log_mlflow(
            logger,
            "num_parameters",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            kind="param",
        )
    # Iterative unfolding: per-event weights of the parton prior (from a reweighting classifier),
    # one per row of each split in dataset order; the CFM loss is weighted, p(x_reco | x_part) is kept.
    part_weights = None
    if cfg.train.get("part_weights", None):
        z = np.load(cfg.train.part_weights)
        part_weights = {}
        for split in ["trn", "tst", "val"]:
            n = dataset.part_ppd[split].shape[0]
            if z[split].shape[0] != n:
                raise ValueError(f"part_weights[{split}] has {z[split].shape[0]} rows, the dataset {n}")
            part_weights[split] = torch.as_tensor(z[split], dtype=dataset.part_ppd[split].dtype,
                                                  device=dataset.part_ppd[split].device).view(-1, 1)
        logger.info(f"    Parton prior weights from {cfg.train.part_weights}: trn mean "
                    f"{part_weights['trn'].mean().item():.4f}, max {part_weights['trn'].max().item():.2f}")
    model.init_dataloaders(dataset, weights=part_weights)

    # Fine-tuning: start from another run's network (fresh optimizer, scheduler and counters)
    if cfg.run.type == "train" and cfg.train.get("init_from", None) and not (
            cfg.train.get("resume", False) and os.path.isfile(os.path.join(model_path, "last.pth"))):
        state = torch.load(cfg.train.init_from, map_location=device, weights_only=False)
        model.network.load_state_dict(state["network"])
        logger.info(f"    Initialized the network from {cfg.train.init_from}")

    ### TRAIN/PLOT ###
    if cfg.run.type == "train":
        # TRAIN MODEL
        if not cfg.train.warm_start:
            os.makedirs(model_path, exist_ok=True)
        else:
            logger.info(f"Warm starting from {cfg.train.checkpoint}")
            model.load(cfg.train.get("checkpoint", "final"))
        if model.run_training() is False:
            logger.info("Training paused (train.max_hours / stop_after_its); resume to finish. No evaluation now.")
            return
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

    ### SIMULATION-DEPENDENCE EVALUATION (optional) ###
    # a training run always also does the standard evaluation + plots below; a plot run with sim_dep
    # only adds the sim_dep task folders
    if cfg.evaluate.get("sim_dep", None):
        run_sim_dep(logger, run_dir, cfg, model, dataset)
        # evaluate.skip_standard: quick runs that only need the sim_dep task folders
        if cfg.run.type == "plot" or cfg.evaluate.get("skip_standard", False):
            return

    ### EVALUATE MODEL ###
    subsets = cfg.evaluate.get("dataset_subsets", None)

    plot_title = (
        r"$\textrm{SM}$"
        if "signal" in cfg.dataset.process
        else r"$\textrm{CP-even}(+)$"
        if "EFT_chw0p5" in cfg.dataset.process
        else r"$\textrm{CP-odd}(+)$"
        if "EFT_chwtil5" in cfg.dataset.process
        else r"$\textrm{Unknown}$"
    )
    if cfg.dataset.combined.use and (subsets is None or len(subsets) == 0):
        # put all the training samples in the title
        names = {"signal": r"\textrm{SM}", "EFT_chw0p5": r"\textrm{CP-even}(+)", "EFT_chwm0p5": r"\textrm{CP-even}(-)",
                 "EFT_chwtil5": r"\textrm{CP-odd}(+)", "EFT_chwtilm2": r"\textrm{CP-odd}(-)",
                 "WWjj": r"\textrm{WWjj}", "ttbar": r"\textrm{ttbar}"}
        procs = cfg.dataset.combined.get("processes", ["signal", "EFT_chw0p5", "EFT_chwtil5"])
        plot_title = "$" + " + ".join(names[p] for p in procs) + "$"
    if subsets is not None and len(subsets) > 0:
        logger.info(
            f"Evaluating subsets: {[{0: 'SM', 1: 'CP-even', 2: 'CP-odd'}[i] for i in subsets]}"
        )
        plot_title = (
            r"$"
            + " + ".join(
                [
                    rf"\textrm{{{ {0: 'SM', 1: 'CP-even', 2: 'CP-odd'}[i] }}}"
                    for i in subsets
                ]
            )
            + r"$"
        )
        split = cfg.evaluate.get("subset_split", "tst")
        loader = model.make_subset_loader(
            split=split,
            keep_ids=list(cfg.evaluate.dataset_subsets),
            batch_size=cfg.train.batch_size_sample,
        )
        dataset.unfolded_ppd = model.evaluate(split=f"Subsets {subsets}", loader=loader)
        keep = torch.isin(dataset.dataset_ids[split].view(-1).cpu(), torch.tensor(list(subsets)))
        reco_ref = dataset.reco[split][keep.to(dataset.reco[split].device)]  # the unfolded events' reco
    else:
        dataset.unfolded_ppd = model.evaluate(split="tst")
        reco_ref = None
    dataset.apply_preprocessing(
        reverse=True, reco_ref=reco_ref
    )  # reverse preprocessing to obtain unfolded data

    if cfg.evaluate.save_samples:
        logger.info(f"Saving unfolded samples to {run_dir}/samples")
        os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
        torch.save(
            dataset.unfolded_ppd,
            os.path.join(run_dir, f"samples/unfolded_ppd.pt"),
        )
        torch.save(
            dataset.unfolded,
            os.path.join(run_dir, f"samples/unfolded.pt"),
        )
    if cfg.evaluate.save_data:
        logger.info(f"Saving data to {run_dir}/data")
        os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
        torch.save(
            {"part": dataset.part, "reco": dataset.reco},
            os.path.join(run_dir, f"samples/data_raw.pt"),
        )
        torch.save(
            {"part": dataset.part_ppd, "reco": dataset.reco_ppd},
            os.path.join(run_dir, f"samples/data_ppd.pt"),
        )

    ### COMPUTE OBSERVABLES ###
    logger.info("Computing observables")
    compute_observables(dataset, split="tst", include_ppd=True)
    if cfg.dataset.combined.use and cfg.dataset.combined.dataset_ids:
        dataset.dataset_ids = dataset.dataset_ids[
            "tst"
        ]  # We make sure that the dataset that is passed to the plots has the dataset ids assigned for the split

    plot_prior = False
    for prior in ["SM", "EFTchw0p5", "EFTchwtil5"]:
        if cfg.plot.get(f"include_{prior}_line", False):
            assert plot_prior is False, "Only one prior can be plotted at a time."
            plot_prior = True
            prior_dataset = WBF(logger, cfg.dataset, initiate_prior_only=prior)
            prior_dataset.apply_preprocessing()
            prior_dataset.split_data()
            compute_observables(prior_dataset, split="tst", include_ppd=False)

    ### MAKE PLOTS ###
    logger.info(f"Starting plots")
    plots = Plots(
        logger,
        dataset,
        SM_dataset=prior_dataset if cfg.plot.get("include_SM_line", False) else None,
        EFTchw0p5_dataset=prior_dataset
        if cfg.plot.get("include_EFTchw0p5_line", False)
        else None,
        EFTchwtil5_dataset=prior_dataset
        if cfg.plot.get("include_EFTchwtil5_line", False)
        else None,
        losses=model.losses if hasattr(model, "losses") else None,
        unfold_to=cfg.dataset.unfold_to,
        debug=False,
        model_name=model.name,
        title=plot_title if plot_title is not None else None,
        subsets=subsets if subsets is not None and len(subsets) > 0 else None,
    )

    if hasattr(model, "losses"):
        logger.info(f"    Plotting train metrics")
        plots.plot_train_metrics(
            os.path.join(run_dir, f"train_metrics.pdf"), pickle_file=os.path.join(run_dir, f"train_metrics.pkl"), logy=False
        )
    logger.info(f"    Plotting observables")
    plots.plot_observables(os.path.join(run_dir, f"observables.pdf"), pickle_file=os.path.join(run_dir, f"observables.pkl"))
    if hasattr(dataset, "observables_ppd"):
        logger.info(f"    Plotting ppd observables")
        plots.plot_observables_ppd(os.path.join(run_dir, f"observables_ppd.pdf"), pickle_file=os.path.join(run_dir, f"observables_ppd.pkl"))
    logger.info(f"    Plotting multiplicity-cut observables")
    if subsets is not None and len(subsets) < cfg.dataset.combined.get("n_datasets", 3):
        pass
    else:
        # Only plot multiplicity cut if evaluating on all datasets.
        # Otherwise it will throw an error because the dataset is initiated without dataset id (all combined) but the
        # unfolding is only performed on the subset list
        plots.plot_multiplicity_cut_observables(
            os.path.join(run_dir, f"multiplicity_cut.pdf"), pickle_file=os.path.join(run_dir, f"multiplicity_cut.pkl")
        )
    logger.info(f"    Plotting migration")
    plots.plot_migration(os.path.join(run_dir, f"migration.pdf"), pickle_file=os.path.join(run_dir, f"migration.pkl"))
    logger.info(f"    Plotting gt migration")
    plots.plot_migration(os.path.join(run_dir, f"gt_migration.pdf"), gt_part=True, pickle_file=os.path.join(run_dir, f"gt_migration.pkl"))
    plots.plot_migration_diff(os.path.join(run_dir, f"migration_diff.pdf"), pickle_file=os.path.join(run_dir, f"migration_diff.pkl"))

    ### UNFOLD P_DATA WITH CLASSIFIER WEIGHTS ###
    if cfg.evaluate.get("unfold_data_weights", None):
        run_data_unfolding(logger, run_dir, cfg, model, dataset, plots)

    if device == torch.device("cuda"):
        max_used = torch.cuda.max_memory_allocated()
        max_total = torch.cuda.mem_get_info()[1]
        logger.info(
            f"GPU RAM information: max_used = {max_used/1e9:.3} GB, max_total = {max_total/1e9:.3} GB"
        )
    logger.info("Run finished")


def _recover_process_masks(logger, cfg, is_signal):
    """Recover per-event process identity (ttbar vs WWjj) for p_data's background rows.

    The classifier tracks mask_ttbar_in_p0 / mask_WWjj_in_p0 but legacy dumps did not save them,
    and the run was unseeded (so the perm/perm_p0 shuffles can't be reproduced). The raw EPxPyPz
    reco rows in p_data_reco.npy are bit-identical copies of rows in the source ttbar/WWjj reco
    files (the classifier stores d_tensor.float()), so we recover identity by value-matching each
    background row against the ttbar reco file.

    Returns (mask_wwjj, mask_ttbar): bool np.ndarray of shape (N,), aligned with p_data_reco.npy
    row order (== unfolded-output order, since split_data preserves order and inference runs with
    shuffle=False). Signal rows are False in both masks.
    """
    n_data = cfg.dataset.get("n_data", None)
    reco_path = os.path.join(cfg.dataset.p_data.data_path, cfg.dataset.p_data.path_reco)
    pdata_reco = np.load(reco_path)
    pdata_reco = (pdata_reco[:n_data] if n_data is not None else pdata_reco).astype(np.float32)

    is_sig = is_signal.cpu().numpy().astype(bool)
    if is_sig.shape[0] != pdata_reco.shape[0]:
        raise ValueError(
            f"is_signal has {is_sig.shape[0]} rows but {reco_path} has {pdata_reco.shape[0]}; "
            f"they must originate from the same classifier run."
        )

    # Cast the source ttbar reco exactly like the classifier did (float64 -> float32) so the row
    # bytes match, then test membership for each background row.
    ttbar_path = cfg.dataset.ttbar.path_reco_corresponding_to_gen
    ttbar_reco = np.load(ttbar_path).astype(np.float64).astype(np.float32)
    ttbar_keys = {row.tobytes() for row in ttbar_reco}
    logger.info(f"    Value-matching {int((~is_sig).sum())} background rows against {ttbar_path}")

    mask_ttbar = np.zeros(pdata_reco.shape[0], dtype=bool)
    for i in np.nonzero(~is_sig)[0]:
        if pdata_reco[i].tobytes() in ttbar_keys:
            mask_ttbar[i] = True
    mask_wwjj = (~is_sig) & (~mask_ttbar)

    logger.info(
        f"    Recovered process masks: ttbar={int(mask_ttbar.sum())}, "
        f"WWjj={int(mask_wwjj.sum())}, signal={int(is_sig.sum())}"
    )
    return mask_wwjj, mask_ttbar


def run_sim_dep(logger, run_dir, cfg, model, dataset):
    """Simulation-dependence evaluations of a trained unfolder (results/simulation_dependence).

    cfg.evaluate.sim_dep: list of tasks, each saved to <run_dir>/<name>/events.npz (+ meta.pkl):
      {name, eval_process: <dataset key>}  -- unfold the TEST split of another sample (e.g. an EFT
          sample with an SM-trained model), standardized with the TRAINING sample's reco mean/std
          (same latent space as in training; as for p_data).
      {name, subset: <dataset id>, labels: [ids] | null}  -- conditioned model: unfold the test
          events of process <subset> once per forced condition label in `labels` (null = each
          event's own label).
      eval_process also takes labels, for a conditioned model. A label "random" draws one uniformly per event
      among the model's conditions.
    events.npz: obs_part / obs_reco (n_obs, N), obs_unf (n_labels, n_obs, N), labels.
    meta.pkl:   observable tex_labels/keys/units/bins, task, run_dir.
    """
    import pickle
    from omegaconf import OmegaConf

    def _obs(ds, data):
        vals = (o.compute(data) for o in ds.observables)
        return np.stack([np.asarray(v.cpu() if torch.is_tensor(v) else v, dtype=np.float32) for v in vals])

    for task in cfg.evaluate.sim_dep:
        task = OmegaConf.to_container(task, resolve=True)
        out = os.path.join(run_dir, task["name"])
        os.makedirs(out, exist_ok=True)
        logger.info(f"=== sim_dep task {task}")
        if task.get("eval_process"):
            ecfg = OmegaConf.to_container(cfg.dataset, resolve=True)
            ecfg["process"] = task["eval_process"]
            for k in ("data_path", "path_parton", "path_gen", "path_reco_corresponding_to_parton",
                      "path_reco_corresponding_to_gen"):
                ecfg[k] = ecfg[task["eval_process"]][k]
            ecfg["combined"]["use"] = False
            eds = WBF(logger, OmegaConf.create(ecfg))
            # standardize with the TRAINING sample's statistics (do not refit on the evaluated sample)
            eds.apply_preprocessing(reco_mean=dataset.reco_mean, reco_std=dataset.reco_std,
                                    part_mean=getattr(dataset, "part_mean", None), part_std=getattr(dataset, "part_std", None))
            eds.split_data()
            ch = eds.channels
            n_eval = eds.part_ppd["tst"].shape[0]
            tensors = [eds.part_ppd["tst"][..., ch], eds.reco_ppd["tst"][..., eds.cond_channels], torch.ones(n_eval, 1)]
            labels = task.get("labels") or [None]
            if cfg.dataset.combined.use and cfg.dataset.combined.dataset_ids:
                # Model.evaluate expects the id column of a combined run. An unconditioned model ignores it; a
                # conditioned one has no own label for a sample it was not trained on, so the task must force them.
                assert not (_condition_on_ids(cfg) and None in labels), \
                    "eval_process with a conditioned model needs explicit labels"
                tensors.append(torch.zeros(n_eval, 1, dtype=torch.int32, device=eds.part_ppd["tst"].device))
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors),
                                                 batch_size=cfg.train.batch_size_sample, shuffle=False)
            unf = []
            for lab in labels:
                logger.info(f"    {task['eval_process']}: condition label {'none' if lab is None else lab}")
                eds.unfolded_ppd = model.evaluate(split=f"{task['name']} ({task['eval_process']} tst)", loader=loader,
                                                  override_id=lab)
                eds.apply_preprocessing(reverse=True)
                unf.append(eds.unfolded)
            part, reco, obs_ds = eds.part["tst"], eds.reco["tst"], eds
            labels = ["true" if lab is None else lab for lab in labels]
        else:
            sid = int(task["subset"])
            loader = model.make_subset_loader(split="tst", keep_ids=[sid], batch_size=cfg.train.batch_size_sample)
            ids = dataset.dataset_ids["tst"].view(-1).cpu().numpy()
            mask = torch.as_tensor(ids == sid)
            labels = task.get("labels") or [None]
            unf = []
            for lab in labels:
                logger.info(f"    subset {sid}: condition label {'own' if lab is None else lab}")
                dataset.unfolded_ppd = model.evaluate(split=f"{task['name']} subset {sid}", loader=loader,
                                                      override_id=lab)
                dataset.apply_preprocessing(reverse=True,
                                            reco_ref=dataset.reco["tst"][mask.to(dataset.reco["tst"].device)])
                unf.append(dataset.unfolded)
            part, reco = dataset.part["tst"][mask.to(dataset.part["tst"].device)], dataset.reco["tst"][mask.to(dataset.reco["tst"].device)]
            labels = ["true" if lab is None else int(lab) for lab in labels]
            obs_ds = dataset
        np.savez_compressed(
            os.path.join(out, "events.npz"),
            obs_part=_obs(obs_ds, part), obs_reco=_obs(obs_ds, reco),
            obs_unf=np.stack([_obs(obs_ds, u) for u in unf]), labels=np.array([str(x) for x in labels]),
        )
        meta = {
            "task": task, "run_dir": run_dir, "n_events": int(part.shape[0]),
            "observables": [{"tex_label": o.tex_label, "key": getattr(o, "key", None), "unit": o.unit,
                             "yscale": o.yscale, "channel": o.channel,
                             "bins": np.asarray(o.bins(o.compute(part)).cpu())} for o in obs_ds.observables],
            "channels": list(getattr(dataset, "plot_channels", dataset.channels)),  # physical channels
        }
        pickle.dump(meta, open(os.path.join(out, "meta.pkl"), "wb"))
        logger.info(f"    saved {part.shape[0]} events x {len(labels)} label(s) -> {out}")


def run_data_unfolding(logger, run_dir, cfg, model, dataset, plots):
    """Unfold the full p_data reco through the trained unfolder, then compare
    hist(unfolded(p_data), weights=(1-w)) vs hist(p_data_parton, weights=(1-w)).

    Gated by cfg.evaluate.unfold_data_weights (path to unfolding_weights_trn.pt
    saved by the classifier). cfg.dataset.p_data.data_path must point at the
    classifier run_dir containing p_data_reco.npy and p_data_parton.npy.
    """
    weights_path = cfg.evaluate.get("unfold_data_weights", None)
    logger.info("=== Unfolding p_data with classifier ensemble weights ===")
    logger.info(f"    Loading weights from: {weights_path}")

    # Load classifier weights: signal_approx_weights = (1 - w_sc), shape (K, N)
    d = torch.load(weights_path, map_location="cpu")
    w = d["signal_approx_weights"]   # (K, N_p0_trn)
    K, N = w.shape
    logger.info(f"    Loaded weights: K={K} ensemble members, N={N} p_data events")
    logger.info(
        f"    N_signal={d['is_signal'].sum().item()}, "
        f"N_background={(~d['is_signal']).sum().item()}"
    )

    # Build a WBF dataset for p_data by overriding the process to 'p_data'.
    # Set trn_tst_val=[0.0,1.0,0.0] so split_data() puts all events in the 'tst' split.
    from omegaconf import OmegaConf
    pdata_cfg_dict = OmegaConf.to_container(cfg.dataset, resolve=True)
    pdata_cfg_dict["process"]     = "p_data"
    pdata_cfg_dict["trn_tst_val"] = [0.0, 1.0, 0.0]
    # Propagate the resolved paths (Hydra interpolation already resolved by to_container).
    # WBF.init_dataset reads path_reco_corresponding_to_{gen,parton} at the top level
    # (keyed on unfold_to), NOT a bare path_reco — so copy those, not path_reco.
    pdata_cfg_dict["data_path"]   = pdata_cfg_dict["p_data"]["data_path"]
    pdata_cfg_dict["path_parton"] = pdata_cfg_dict["p_data"]["path_parton"]
    pdata_cfg_dict["path_gen"]    = pdata_cfg_dict["p_data"]["path_gen"]
    pdata_cfg_dict["path_reco_corresponding_to_gen"]    = pdata_cfg_dict["p_data"]["path_reco_corresponding_to_gen"]
    pdata_cfg_dict["path_reco_corresponding_to_parton"] = pdata_cfg_dict["p_data"]["path_reco_corresponding_to_parton"]
    pdata_cfg = OmegaConf.create(pdata_cfg_dict)

    logger.info(f"    Loading p_data from: {pdata_cfg.data_path}")
    pdata_dataset = WBF(logger, pdata_cfg)

    # Preprocess using the training signal's reco_mean/reco_std — do NOT refit.
    # This places p_data in the exact same latent space as the training signal.
    pdata_dataset.apply_preprocessing(
        reco_mean=dataset.reco_mean,
        reco_std=dataset.reco_std,
        part_mean=getattr(dataset, "part_mean", None),
        part_std=getattr(dataset, "part_std", None),
    )
    pdata_dataset.split_data()

    N_pdata = pdata_dataset.reco_ppd["tst"].shape[0]
    if N_pdata != N:
        logger.warning(
            f"    WARNING: p_data reco has {N_pdata} events but weights tensor has {N} columns. "
            f"These must match exactly — ensure p_data_reco.npy was saved from the same classifier run."
        )

    # Build inference DataLoader: condition = preprocessed p_data reco, shuffle=False.
    device = next(model.parameters()).device
    channels = dataset.channels
    reco_ppd_pdata = pdata_dataset.reco_ppd["tst"][..., dataset.cond_channels].to(device)
    placeholder_part   = torch.zeros(N_pdata, len(channels), dtype=reco_ppd_pdata.dtype, device=reco_ppd_pdata.device)
    placeholder_weight = torch.ones(N_pdata, 1, dtype=reco_ppd_pdata.dtype, device=device)
    pdata_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(placeholder_part, reco_ppd_pdata, placeholder_weight),
        batch_size=cfg.train.batch_size_sample,
        shuffle=False,
    )

    # Run the unfolder on p_data reco → unfolded_ppd (channel-sliced, preprocessed parton)
    logger.info(f"    Running unfolder on {N_pdata} p_data events...")
    pdata_dataset.unfolded_ppd = model.evaluate(split="p_data inference", loader=pdata_loader)

    # Reverse-preprocess: unfolded_ppd → unfolded (physical PtPhiEtaM units)
    pdata_dataset.apply_preprocessing(reverse=True)

    # Compute per-observable arrays (stores obs_part, obs_reco, obs_unf, bins on pdata_dataset)
    logger.info("    Computing p_data observables")
    compute_observables(pdata_dataset, split="tst", include_ppd=False)

    # Per-process (WWjj / ttbar) decomposition of the unfolded p_data, for closure cross-checks.
    # Prefer masks saved by the classifier; fall back to value-matching for legacy dumps.
    if "mask_wwjj_in_p0" in d and "mask_ttbar_in_p0" in d:
        mask_wwjj  = d["mask_wwjj_in_p0"].cpu().numpy().astype(bool)
        mask_ttbar = d["mask_ttbar_in_p0"].cpu().numpy().astype(bool)
        logger.info("    Using per-process masks saved in the classifier weights file.")
    else:
        logger.info("    Per-process masks absent from weights file; recovering by value-matching.")
        mask_wwjj, mask_ttbar = _recover_process_masks(logger, cfg, d["is_signal"])

    # The blue line already unfolds all of p_data, so the per-process lines are just slices of the
    # already-computed observables (no extra inference). Each obs array is (N,) in p_data file order,
    # aligned with the masks.
    bkg_closures = [
        {"label": "WWjj",
         "obs_unf":  [a[mask_wwjj]  for a in pdata_dataset.obs_unf],
         "obs_part": [a[mask_wwjj]  for a in pdata_dataset.obs_part]},
        {"label": r"t\bar{t}",
         "obs_unf":  [a[mask_ttbar] for a in pdata_dataset.obs_unf],
         "obs_part": [a[mask_ttbar] for a in pdata_dataset.obs_part]},
    ]

    # Make the comparison plot: unfolded p_data + per-background closure lines vs signal truth
    logger.info("    Plotting unfolded p_data with per-background (WWjj, ttbar) closure lines")
    # Per-event inputs of the plot (unfolded + truth observables, classifier weights, priors,
    # signal flag), so the plot and its bootstrap can be redone on CPU without re-unfolding.
    np.savez_compressed(
        os.path.join(run_dir, "unfolded_data_events.npz"),
        obs_unf=np.stack([np.asarray(a, dtype=np.float32) for a in pdata_dataset.obs_unf]),
        obs_part=np.stack([np.asarray(a, dtype=np.float32) for a in pdata_dataset.obs_part]),
        weights=w.cpu().numpy(),
        prior_weights=d["prior_weights"].cpu().numpy() if "prior_weights" in d else np.array([]),
        is_signal=d["is_signal"].cpu().numpy().astype(bool) if "is_signal" in d else np.array([]),
    )
    logger.info(f"    Saved per-event plot inputs -> {run_dir}/unfolded_data_events.npz")
    plots.plot_unfolded_data(
        file=os.path.join(run_dir, "observables_unfolded_data.pdf"),
        obs_unf_data=pdata_dataset.obs_unf,
        obs_part_data=pdata_dataset.obs_part,   # p_data(x_part) gen truth (blue line's ratio ref)
        bkg_closures=None,              # per-process TraCFM(p_X) vs p_X(x_part) closures
        bins=pdata_dataset.bins,
        observables=pdata_dataset.observables,
        weights=w.cpu().numpy(),   # (K, N) numpy
        pickle_file=os.path.join(run_dir, "observables_unfolded_data.pkl"),
        # alpha-weighted comparison vs the gen truth of p_data's true-signal rows (when dumped)
        prior_weights=d["prior_weights"].cpu().numpy() if "prior_weights" in d else None,
        is_signal=d["is_signal"].cpu().numpy().astype(bool) if "is_signal" in d else None,
        n_bootstrap=int(cfg.evaluate.get("n_bootstrap", 200)),
    )
    logger.info(f"    Saved observables_unfolded_data.pdf/pkl in {run_dir}")
