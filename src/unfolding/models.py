import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from src.utils.mlflow import mlflow, LOGGING_ENABLED
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric
from .nets.layers import *


def _condition_on_ids(cfg):
    """Whether the model is *conditioned* on per-process dataset ids.

    Decoupled from combined.dataset_ids (which controls whether the id tensor is *loaded* — e.g. for
    balanced sampling) so training fractions can be balanced without telling the model the process.
    Defaults to dataset_ids for backward compatibility with the conditioned EFT runs.
    """
    comb = cfg.dataset.combined
    if not comb.get("use", False):
        return False
    return comb.get("condition_model", comb.get("dataset_ids", False))


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Model(nn.Module):
    def __init__(self, logger, cfg, dims_x, dims_c, model_path, device):
        super().__init__()
        self.logger = logger
        self.cfg = cfg
        self.dims_x = dims_x
        self.dims_c = dims_c
        self.model_path = model_path
        self.device = device

        self.bayesian = self.cfg.model.get("bayesian", False)
        if self.bayesian:
            self.bayesian_factor = self.cfg.model.get("bayesian_factor", 1.0e-3)
            self.logger.info(f"Using Bayesian factor {self.bayesian_factor}")
        self.state_dict_attrs = ["network", "optimizer"]  # add more if needed

    def init_network(self):
        if self.cfg.model.network == "MLP":
            if self.bayesian:
                layer_class = VBLinear
                self.name = "bCFM"
            else:
                layer_class = nn.Linear
                self.name = "CFM"
            layers = []
            layers.append(layer_class(self.dims_in, self.cfg.model.internal_size))
            layers.append(nn.ReLU())
            for _ in range(self.cfg.model.hidden_layers):
                layers.append(
                    layer_class(
                        self.cfg.model.internal_size,
                        self.cfg.model.internal_size,
                    )
                )
                layers.append(nn.ReLU())
            layers.append(layer_class(self.cfg.model.internal_size, self.dims_x))
            self.network = nn.Sequential(*layers)
            self.network.bayesian_layers = [
                layer for layer in layers if isinstance(layer, VBLinear)
            ]
        elif self.cfg.model.network == "Transformer":
            self.name = "TraCFM" if not self.bayesian else "bTraCFM"
            self.network = TransformerCFM(self.cfg, self.dims_x, self.dims_c)
            self.network.bayesian_layers = [
                self.network.final_layer
                if isinstance(self.network.final_layer, VBLinear)
                else None
            ]

    # Overwrite in child class
    def sample(self, c):
        pass

    # Overwrite in child class
    def batch_loss(self, x, c, weight):
        pass

    def init_dataloaders(self, dataset, weights=None):
        data_x = dataset.part_ppd
        data_c = dataset.reco_ppd
        dataset_ids = dataset.dataset_ids if hasattr(dataset, "dataset_ids") else None
        channels = dataset.channels
        cond_channels = getattr(dataset, "cond_channels", channels)

        # apply stuff to make it such that only the channels of the data are used
        if weights is None:
            weights = {
                f"{split}": torch.ones(data_x[f"{split}"].shape[0], 1)
                .to(data_x[f"{split}"].dtype)
                .to(data_x[f"{split}"].device)
                for split in ["trn", "tst", "val"]
            }
        for split in ["trn", "tst", "val"]:
            if dataset_ids is not None:
                globals()[f"{split}set"] = torch.utils.data.TensorDataset(
                    data_x[f"{split}"][..., channels],
                    data_c[f"{split}"][..., cond_channels],
                    weights[f"{split}"],
                    dataset_ids[f"{split}"],
                )
            else:
                globals()[f"{split}set"] = torch.utils.data.TensorDataset(
                    data_x[f"{split}"][..., channels],
                    data_c[f"{split}"][..., cond_channels],
                    weights[f"{split}"],
                )
        self.n_train_samples = len(trnset)
        self.n_val_samples = len(valset)
        self.bs = self.cfg.train.batch_size
        self.bs_sample = self.cfg.train.get("batch_size_sample", self.bs)

        mode = self.cfg.dataset.combined.get("balance_mode", "none")
        has_ids = len(trnset.tensors) >= 4
        if mode == "none" or not has_ids:
            self.trnloader = torch.utils.data.DataLoader(
                trnset, batch_size=self.bs, shuffle=True
            )
        elif mode == "approx" or mode == "strict":
            ids = trnset.tensors[3].view(-1)
            # inverse-frequency weights per sample
            uniques, counts = ids.unique(return_counts=True)
            invfreq = torch.zeros_like(ids, dtype=torch.double)
            for u, c in zip(uniques, counts):
                invfreq[ids == u] = 1.0 / float(c)
            sampler = torch.utils.data.WeightedRandomSampler(
                invfreq,
                num_samples=len(invfreq),
                replacement=False if mode == "approx" else True,
            )

            self.trnloader = torch.utils.data.DataLoader(
                trnset, batch_size=self.bs, sampler=sampler
            )
        else:
            raise ValueError(f"Unknown balance_mode='{mode}'")
        self.valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=self.bs,
            shuffle=True,
        )
        self.tstloader = torch.utils.data.DataLoader(
            tstset,
            batch_size=self.bs_sample,
            shuffle=False,
        )
        self.trnset = trnset
        self.valset = valset
        self.tstset = tstset

    def init_optimizer(self):
        optim = self.cfg.train.get("optimizer", "adam")
        lr = float(self.cfg.train.get("lr", 0.001))
        wd = self.cfg.train.get("weight_decay", 0.0)
        if optim == "adam" or optim == "Adam":
            optimizer = torch.optim.Adam(
                self.network.parameters(), lr=lr, weight_decay=wd
            )
        else:
            raise NotImplementedError(f"Optimizer {optim} not implemented")
        self.optimizer = optimizer
        self.logger.info(
            f"    Using optimizer {optim} with lr={lr} and weight decay={wd}"
        )

    def init_scheduler(self):
        sched = self.cfg.train.get("scheduler", None)
        if sched == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.cfg.train.nits
            )
        elif sched == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.cfg.train.get("lr_factor", 0.1),
                patience=self.cfg.train.get("lr_patience", 10),
            )
        elif sched is None:
            scheduler = None
        else:
            raise NotImplementedError(f"Scheduler {sched} not implemented")
        self.scheduler = scheduler
        self.logger.info(f"    Using scheduler {sched}") if not hasattr(
            self.scheduler, "patience"
        ) else self.logger.info(
            f"    Using scheduler {sched} with factor {getattr(self.scheduler, 'factor')} and patience {getattr(self.scheduler, 'patience')}."
        )

    def run_training(self):
        if LOGGING_ENABLED:
            self.mlflowclient = MlflowClient()

        nits = int(self.cfg.train.nits)  # total training iterations
        val_freq = self.cfg.train.get(
            "val_freq", len(self.trnloader)
        )  # how often to validate/save, default: per epoch

        self.init_optimizer()
        self.init_scheduler()
        warm_start = self.cfg.train.get("warm_start", False)

        resume_state = None
        if self.cfg.train.get("resume", False) and os.path.isfile(os.path.join(self.model_path, "last.pth")):
            # EXACT continuation of an interrupted run (train.resume): network, optimizer, scheduler,
            # early-stopping state, counters, loss histories and RNG states from the last epoch end.
            # The total budget stays cfg.train.nits (unlike warm_start, which adds nits again).
            resume_state = torch.load(os.path.join(self.model_path, "last.pth"), map_location=self.device,
                                      weights_only=False)
            self.network.load_state_dict(resume_state["network"])
            self.optimizer.load_state_dict(resume_state["optimizer"])
            if self.scheduler is not None and resume_state["scheduler"] is not None:
                self.scheduler.load_state_dict(resume_state["scheduler"])
            current_it, epoch_idx = resume_state["current_it"], resume_state["epoch_idx"]
            self.best_val_loss = resume_state["best_val_loss"]
            self.logger.info(
                f"Resuming EXACTLY from {self.model_path}/last.pth: iteration {current_it}, epoch {epoch_idx}, "
                f"lr={self.optimizer.param_groups[0]['lr']}, best val loss {self.best_val_loss:.8f}"
            )
        elif warm_start:
            self.logger.info(
                "Warm-starting model from "
                + os.path.join(
                    self.model_path, f"{self.cfg.train.get('checkpoint', 'final')}.pth"
                )
            )
            self.load(self.cfg.train.get("checkpoint", "final"))
            # infer how far we had trained already
            trn_hist = self.losses.get("trn", []) if hasattr(self, "losses") else []
            val_hist = self.losses.get("val", []) if hasattr(self, "losses") else []

            current_it = len(trn_hist)  # iterations completed so far
            epoch_idx = len(val_hist)  # how many validation passes done
            self.best_val_loss = min(val_hist) if len(val_hist) > 0 else 1e20
            nits += current_it
            # log resumed LR (comes from restored optimizer state)
            self.logger.info(
                f"    Resuming from iteration {current_it}, "
                f"epoch {epoch_idx}, "
                f"lr={self.optimizer.param_groups[0]['lr']}"
            )
        else:
            # bookkeeping
            current_it = 0
            epoch_idx = 0
            self.best_val_loss = 1e20

        # sanity: lr groups
        for i, g in enumerate(self.optimizer.param_groups):
            self.logger.info(
                f"Param_group[{i}]: lr={g['lr']}, wd={g.get('weight_decay', 0.0)}"
            )

        # iteration-wise logs
        trn_loss_it = []
        trn_lr = []
        grd_norm = []
        if self.bayesian:
            kl_it = []
            mse_it = []

        # epoch-wise logs
        trn_loss_epoch = []  # mean of last steps_per_epoch training iterations
        val_loss_epoch = []
        if self.bayesian:
            kl_epoch = []
            mse_epoch = []

        if self.cfg.train.early_stopping.get("use", False):
            patience = self.cfg.train.early_stopping.get("patience", 10)
            early_stopping = EarlyStopping(patience=patience)
            self.logger.info(f"    Using early stopping with patience {patience}")
        if resume_state is not None:
            h = resume_state["histories"]
            trn_loss_it, trn_lr, grd_norm = h["trn_loss_it"], h["trn_lr"], h["grd_norm"]
            trn_loss_epoch, val_loss_epoch = h["trn_loss_epoch"], h["val_loss_epoch"]
            if self.bayesian:
                kl_it, mse_it, kl_epoch, mse_epoch = h["kl_it"], h["mse_it"], h["kl_epoch"], h["mse_epoch"]
            if self.cfg.train.early_stopping.get("use", False):
                es = resume_state["early_stopping"]
                early_stopping.counter, early_stopping.best_loss, early_stopping.early_stop = (
                    es["counter"], es["best_loss"], es["early_stop"])
            self.losses = resume_state["losses"]
            # RNG states must be CPU ByteTensors (map_location moved them to self.device)
            torch.set_rng_state(resume_state["rng"]["torch"].cpu())
            if torch.cuda.is_available() and resume_state["rng"]["cuda"] is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in resume_state["rng"]["cuda"]])
            np.random.set_state(resume_state["rng"]["numpy"])
            random.setstate(resume_state["rng"]["python"])
        self.logger.info(
            f"Training model for {nits} iterations (= {nits // len(self.trnloader)} epochs)"
        )
        self.logger.info(
            f"    Validation frequency: {val_freq} its. Nb of train batches: {len(self.trnloader)}"
        )
        t0 = time.time()
        job_its = 0  # iterations run in THIS job (for train.stop_after_its)
        paused = False  # graceful stop before the end (train.max_hours / stop_after_its): resume later
        max_hours = self.cfg.train.get("max_hours", None)
        stop_after = self.cfg.train.get("stop_after_its", None)
        stop_training = bool(resume_state is not None and resume_state["finished"])
        if stop_training:
            self.logger.info("    Resumed run had already finished; writing final checkpoint only")
        while current_it < nits and not stop_training:
            self.network.train()
            # (re)initialize running epoch accumulators (for optional epoch-avg train stats)
            epoch_trn_losses = []
            epoch_kl_losses = [] if self.bayesian else None
            epoch_mse_losses = [] if self.bayesian else None

            for i, batch in enumerate(self.trnloader):
                if current_it >= nits or stop_training:
                    break

                if self.cfg.dataset.combined.get(
                    "use", False
                ) and self.cfg.dataset.combined.get("dataset_ids", False):
                    x_part, x_reco, weight, dataset_ids = batch
                    if not _condition_on_ids(self.cfg):
                        dataset_ids = None  # ids present for the sampler, but model not conditioned
                else:
                    x_part, x_reco, weight = batch
                    dataset_ids = None

                self.optimizer.zero_grad()
                loss, loss_terms = self.batch_loss(
                    x_part, x_reco, weight, dataset_ids=dataset_ids
                )
                loss.backward()
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.cfg.train.get("clip_grad_norm", 10),
                        error_if_nonfinite=False,
                    )
                    .cpu()
                    .item()
                )
                self.optimizer.step()
                if self.scheduler is not None and not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()

                # ---- logging (iteration-wise) ----
                li = float(loss.detach().cpu())
                trn_loss_it.append(li)
                epoch_trn_losses.append(li)
                if self.bayesian:
                    kl_it.append(loss_terms["kl"])
                    mse_it.append(loss_terms["mse"])
                    epoch_kl_losses.append(loss_terms["kl"])
                    epoch_mse_losses.append(loss_terms["mse"])
                trn_lr.append(self.optimizer.param_groups[0]["lr"])
                grd_norm.append(grad_norm)
                current_it += 1
                job_its += 1

                # validate after val_freq iterations or at the end of training
                has_completed_pass = current_it % val_freq == 0
                is_final_iter = current_it == nits

                if has_completed_pass or is_final_iter or stop_training:

                    # validate
                    avg_val, val_terms = self.validate()
                    epoch_mean_trn = torch.tensor(epoch_trn_losses).mean().item()
                    # epoch-avg train (last pass’ mean)
                    trn_loss_epoch.append(epoch_mean_trn)
                    val_loss_epoch.append(avg_val)

                    if self.bayesian:
                        kl_epoch.append(torch.tensor(epoch_kl_losses).mean().item())
                        mse_epoch.append(torch.tensor(epoch_mse_losses).mean().item())
                    # keep the latest losses available (before log_and_save so save("best") has them)
                    self.losses = {
                        "trn": trn_loss_it,
                        "val": val_loss_epoch,
                        "lr": trn_lr,
                        "grad_norm": grd_norm,
                    }

                    self.log_and_save(
                        t0=t0,
                        iteration=current_it,  # total iters so far
                        avg_trn_loss=epoch_mean_trn,  # mean trn loss since last validation window
                        avg_val_loss=avg_val,  # mean val loss over valloader
                        val_freq=val_freq,  # <- pass through exactly what you used
                    )
                    epoch_idx += 1
                    if self.scheduler is not None and isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        self.scheduler.step(avg_val)
                    if self.cfg.train.early_stopping.get("use", False):
                        early_stopping(avg_val)
                        if early_stopping.early_stop:
                            self.logger.info(
                                f"Early stopping after iteration {current_it} with validation loss {avg_val:.8f}"
                            )
                            stop_training = True
                    if self.cfg.train.get("resume", False):
                        self._save_last(
                            current_it, epoch_idx, early_stopping if self.cfg.train.early_stopping.get("use", False) else None,
                            finished=stop_training or current_it >= nits,
                            histories={"trn_loss_it": trn_loss_it, "trn_lr": trn_lr, "grd_norm": grd_norm,
                                       "trn_loss_epoch": trn_loss_epoch, "val_loss_epoch": val_loss_epoch,
                                       **({"kl_it": kl_it, "mse_it": mse_it, "kl_epoch": kl_epoch,
                                           "mse_epoch": mse_epoch} if self.bayesian else {})},
                        )
                    if stop_training:
                        break
                    # graceful pause at this epoch end (state is in last.pth); not finished, no final
                    if self.cfg.train.get("resume", False) and current_it < nits and (
                        (max_hours is not None and time.time() - t0 > 3600 * float(max_hours))
                        or (stop_after is not None and job_its >= int(stop_after))
                    ):
                        self.logger.info(f"Pausing at iteration {current_it} (epoch {epoch_idx}); "
                                         f"resume with train.resume=true run.path=<run dir>")
                        paused = True
                        break
            # end for batches
            if paused:
                break
        # end of training loop
        if paused:
            return False
        self.save("final")
        self.logger.info(
            f"Finished training after {time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}. Nb of its: {current_it}, epochs: {epoch_idx}"
        )
        self.logger.info(f"Best validation loss: {self.best_val_loss:.8f}")
        self.logger.info(f"Last validation loss: {self.losses['val'][-1]:.8f}")
        return True

    def _save_last(self, current_it, epoch_idx, early_stopping, finished, histories):
        """Full training state at an epoch end (train.resume): written to last.tmp then renamed, so an
        interrupted write never leaves a corrupt last.pth."""
        state = {
            "network": self.network.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "current_it": current_it, "epoch_idx": epoch_idx, "best_val_loss": self.best_val_loss,
            "early_stopping": None if early_stopping is None else {
                "counter": early_stopping.counter, "best_loss": early_stopping.best_loss,
                "early_stop": early_stopping.early_stop},
            "histories": histories, "losses": self.losses, "finished": bool(finished),
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy": np.random.get_state(), "python": random.getstate()},
        }
        tmp = os.path.join(self.model_path, "last.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(self.model_path, "last.pth"))

    @torch.no_grad()
    def validate(self):
        """Run full validation loader once; return mean loss and (optional) terms."""
        self.network.eval()
        losses = []
        kl_list = [] if self.bayesian else None
        mse_list = [] if self.bayesian else None

        for vbatch in self.valloader:
            if self.cfg.dataset.combined.get(
                "use", False
            ) and self.cfg.dataset.combined.get("dataset_ids", False):
                vx_part, vx_reco, vweight, vdataset_ids = vbatch
                if not _condition_on_ids(self.cfg):
                    vdataset_ids = None
            else:
                vx_part, vx_reco, vweight = vbatch
                vdataset_ids = None

            vloss, vterms = self.batch_loss(
                vx_part, vx_reco, vweight, dataset_ids=vdataset_ids
            )
            losses.append(vloss.detach().cpu())
            if self.bayesian:
                kl_list.append(vterms["kl"])
                mse_list.append(vterms["mse"])

        avg_val = torch.tensor(losses).mean().item()
        val_terms = None
        if self.bayesian:
            val_terms = {
                "kl": torch.tensor(kl_list).mean().item(),
                "mse": torch.tensor(mse_list).mean().item(),
            }
        return avg_val, val_terms

    def log_and_save(self, t0, iteration, avg_trn_loss, avg_val_loss, val_freq):
        if iteration == 1 * val_freq:
            self.logger.info(
                f"    Epoch {iteration // len(self.trnloader)} (it. {iteration}): tr_loss={avg_trn_loss:.8f}, val_loss={avg_val_loss:.8f}; ETA={time.strftime('%H:%M:%S', time.gmtime((time.time() - t0) * (self.cfg.train.nits - iteration) / iteration))}"
            )
        if (
            LOGGING_ENABLED
            and iteration
            % max(
                1,
                int(0.10 * self.cfg.train.nits // len(self.trnloader))
                * self.cfg.train.get("val_freq", len(self.trnloader)),
            )
            == 0
        ):
            self.mlflow_log_metrics(t0, iteration)

        if iteration % max(1, iteration) == 0:
            self.logger.info(
                f"    Epoch {iteration // len(self.trnloader)} (it. {iteration}): tr_loss={avg_trn_loss:.8f}, val_loss={avg_val_loss:.8f}"
            )
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save("best")
                self.logger.info(
                    f"    Saved best model with val_loss={avg_val_loss:.8f} at it. {iteration}"
                )

    def evaluate(self, split="tst", loader=None, override_id=None):
        """Sample the unfolding for every event of the split/loader. override_id: condition every
        event on this dataset id instead of its own, or on a random one per event with "random"
        (conditioned models only; None = true ids)."""
        if loader is None:
            if split is None or split == "tst":
                loader = self.tstloader
                self.logger.info("Evaluating model on tst set")
            elif split == "val":
                loader = self.valloader
                self.logger.info("Evaluating model on val set")
            elif split == "trn":
                loader = self.trnloader
                self.logger.info("Evaluating model on trn set")
        else:
            self.logger.info(f"Evaluating model on provided loader: {split}")
        predictions = []

        self.eval()
        with torch.no_grad():
            t0 = time.time()
            for i, batch in enumerate(loader):
                if self.cfg.dataset.combined.get(
                    "use", False
                ) and self.cfg.dataset.combined.get("dataset_ids", False):
                    x_part, x_reco, weight, dataset_ids = batch
                    if not _condition_on_ids(self.cfg):
                        dataset_ids = None
                    elif override_id == "random":  # one uniform draw per event among the model's conditions
                        dataset_ids = torch.randint_like(dataset_ids, self.cfg.dataset.combined.n_datasets)
                    elif override_id is not None:
                        dataset_ids = torch.full_like(dataset_ids, int(override_id))
                    unfold = self.sample(x_reco, dataset_ids).detach().cpu()
                else:
                    x_part, x_reco, weight = batch
                    unfold = self.sample(x_reco).detach().cpu()
                predictions.append(unfold)
                t1 = time.time()
                if i == 0:
                    self.logger.info(
                        f"    Total batches: {len(loader)}. Sampling time estimate: {time.strftime('%H:%M:%S', time.gmtime(round((t1-t0) * len(loader), 1)))}"
                    )
                log_every_percent = 0.25
                if i % max(1, int(len(loader) * log_every_percent)) == 0:
                    self.logger.info(f"    Sampled batch {i+1}/{len(loader)}")
            self.logger.info(f"    Finished sampling. Saving predictions")
        predictions = torch.cat(predictions)
        return predictions

    def make_subset_loader(self, split="tst", keep_ids=None, batch_size=None):
        """
        Return a DataLoader over only the samples whose dataset_id is in keep_ids.
        Requires dataset_ids to be present (combined.dataset_ids=True).
        """
        if keep_ids is None:
            keep_ids = []
        base = {"trn": self.trnset, "tst": self.tstset, "val": self.valset}[split]

        # If this tensor dataset has dataset_ids
        if len(base.tensors) < 4:
            raise ValueError("Subset-by-dataset_id requires combined.dataset_ids=True")

        ids = base.tensors[3].view(-1).cpu().numpy()
        mask = np.isin(ids, np.array(keep_ids))
        idx = np.nonzero(mask)[0].tolist()
        self.logger.info(
            f"Length of subset loader for split '{split}' with keep_ids={keep_ids}: {len(idx)} samples"
        )
        subset = torch.utils.data.Subset(base, idx)
        return torch.utils.data.DataLoader(
            subset, batch_size=(batch_size or self.bs_sample), shuffle=False
        )

    def mlflow_log_metrics(self, t0, iteration):
        if mlflow.active_run() is None:
            self.logger.warning("MLflow is not active. Cannot log metrics.")
        else:
            ts = int((time.time() - t0) * 1000)
            metrics_batch = []
            # for step_idx, loss_val in enumerate(self.losses["trn"]):
            #     metrics_batch.append(
            #         {
            #             "key":       "trn_loss_per_iter",
            #             "value":     loss_val,
            #             "timestamp": ts,
            #             "step":      step_idx,
            #         }
            #     )
            metrics_batch.append(
                {
                    "key": "trn_loss",
                    "value": self.losses["trn"][-1],
                    "timestamp": ts,
                    "step": iteration,
                }
            )
            metrics_batch.append(
                {
                    "key": "val_loss",
                    "value": self.losses["val"][-1],
                    "timestamp": ts,
                    "step": iteration,
                }
            )
            metrics_batch.append(
                {
                    "key": "lr",
                    "value": self.losses["lr"][-1],
                    "timestamp": ts,
                    "step": iteration,
                }
            )
            metrics_batch.append(
                {
                    "key": "grad_norm",
                    "value": self.losses["grad_norm"][-1],
                    "timestamp": ts,
                    "step": iteration,
                }
            )
            metrics = [
                Metric(
                    key=m["key"],
                    value=m["value"],
                    timestamp=m["timestamp"],
                    step=m["step"],
                )
                for m in metrics_batch
            ]
            self.mlflowclient.log_batch(
                run_id=mlflow.active_run().info.run_id, metrics=metrics
            )

    def save(self, name: str):
        """
        Saves the model, preprocessing, optimizer and losses.

        Args:
            name: File name for the model (without path and extension)
        """
        file = os.path.join(self.model_path, f"{name}.pth")
        torch.save(
            {
                **{
                    attr: getattr(self, attr).state_dict()
                    for attr in self.state_dict_attrs
                },
                "losses": self.losses,
            },
            file,
        )

    def load(self, name: str):
        """
        Loads the model, preprocessing, optimizer and losses.

        Args:
            name: File name for the model (without path and extension)
        """
        file = os.path.join(self.model_path, f"{name}.pth")
        state_dicts = torch.load(file, map_location=self.device)
        for attr in self.state_dict_attrs:
            try:
                getattr(self, attr).load_state_dict(state_dicts[attr])
            except AttributeError:
                pass
        self.losses = state_dicts["losses"]
        self.logger.info(f"    Loaded model from {file}.")


class CFM(Model):
    def __init__(self, logger, cfg, dims_x, dims_c, model_path, device):
        super().__init__(logger, cfg, dims_x, dims_c, model_path, device)
        self.dims_in = self.dims_x + self.dims_c + 1
        self.init_network()

    def sample(self, c, dataset_ids=None):
        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        if dataset_ids is not None:
            c = torch.cat([c, dataset_ids], dim=-1)

        def net_wrapper(t, x_t):
            t = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            v = self.network(torch.cat([t, x_t, c], dim=-1))
            return v

        x_0 = torch.randn((batch_size, self.dims_x)).to(device, dtype=dtype)
        # x_t = odeint(func=net_wrapper, y0=x_0, t=torch.linspace(0., 1., 1000).to(device, dtype=dtype))
        # return torch.swapaxes(x_t, 0, 1)
        x_t = odeint(
            func=net_wrapper,
            y0=x_0,
            t=torch.linspace(0.0, 1.0, self.cfg.model.get("n_steps", 2)).to(
                device, dtype=dtype
            ),
            rtol=1e-5,
            atol=1e-7,
        )

        return x_t[-1]

    def batch_loss(self, x, c, weight, dataset_ids=None):
        x_0 = torch.randn((x.size(0), self.dims_x)).to(x.device)
        t = torch.rand((x.size(0), 1)).to(x.device)
        x_t = (1 - t) * x_0 + t * x
        x_t_dot = x - x_0
        c = torch.cat([c, dataset_ids], dim=-1) if dataset_ids is not None else c
        v_pred = self.network(torch.cat([t, x_t, c], dim=-1))
        loss = ((v_pred - x_t_dot) ** 2 * weight).mean()
        if self.bayesian:
            kl_loss = sum([layer.kl() for layer in self.network.bayesian_layers])
            kl_loss = kl_loss * self.bayesian_factor * 1 / len(self.trnloader)

            loss_terms = {
                "loss": (loss + kl_loss).item(),
                "mse": loss.item(),
                "kl": kl_loss.item(),
            }
            loss = loss + kl_loss
        else:
            loss_terms = {
                "loss": loss.item(),
            }
        return loss, loss_terms


class TransformerCFM(nn.Module):
    def __init__(self, cfg, dims_x, dims_c):
        super().__init__()
        self.cfg = cfg
        self.dims_x = dims_x
        self.dims_c = dims_c
        self.dim_embedding = self.cfg.model.dim_embedding

        encoder_layer = TransformerEncoderLayerCustom(
            d_model=self.dim_embedding,
            nhead=self.cfg.model.n_head,
            dim_feedforward=self.cfg.model.dim_feedforward,
            dropout=self.cfg.model.get("resid_pdrop", 0.0),
            batch_first=True,
            attn_pdrop=self.cfg.model.get(
                "attn_pdrop", self.cfg.model.get("resid_pdrop", 0.0)
            ),
        )
        decoder_layer = TransformerDecoderLayerCustom(
            d_model=self.dim_embedding,
            nhead=self.cfg.model.n_head,
            dim_feedforward=self.cfg.model.dim_feedforward,
            dropout=self.cfg.model.get("resid_pdrop", 0.0),
            batch_first=True,
            attn_pdrop=self.cfg.model.get(
                "attn_pdrop", self.cfg.model.get("resid_pdrop", 0.0)
            ),
        )
        self.network = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=self.cfg.model.n_head,
            num_encoder_layers=self.cfg.model.n_encoder_layers,
            num_decoder_layers=self.cfg.model.n_decoder_layers,
            dim_feedforward=self.cfg.model.dim_feedforward,
            dropout=self.cfg.model.get("resid_pdrop", 0.0),
            batch_first=True,
            custom_encoder=nn.TransformerEncoder(
                encoder_layer, self.cfg.model.n_encoder_layers
            ),
            custom_decoder=nn.TransformerDecoder(
                decoder_layer, self.cfg.model.n_decoder_layers
            ),
        )
        self.bayesian = self.cfg.model.get("bayesian", False)
        if not self.bayesian:
            self.final_layer = nn.Linear(self.dim_embedding + 1, 1)
        else:
            self.final_layer = VBLinear(self.dim_embedding + 1, 1)
        if _condition_on_ids(self.cfg):
            self.dataset_id_embedding = nn.Embedding(
                num_embeddings=self.cfg.dataset.combined.get("n_datasets", 3),
                embedding_dim=self.cfg.dataset.combined.get("dim_embedding", 64),
            )

    def compute_embedding(
        self,
        p: torch.Tensor,
        n_components: int,
        t: torch.Tensor = None,
        dataset_ids: torch.Tensor = None,
        positions: list = None,
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        rows = slice(None, p.shape[1]) if positions is None else positions
        one_hot = torch.eye(n_components, device=p.device, dtype=p.dtype)[
            None, rows, :
        ].expand(p.shape[0], -1, -1)
        if t is None:
            p = p.unsqueeze(-1)
        else:
            p = torch.cat(
                [
                    p.unsqueeze(-1),
                    t.unsqueeze(-1).expand(t.shape[0], p.shape[1], 1),
                ],
                dim=-1,
            )
        if dataset_ids is not None:
            dataset_ids = self.dataset_id_embedding(dataset_ids)
            p = torch.cat(
                [
                    p,
                    dataset_ids.expand(
                        dataset_ids.shape[0], p.shape[1], dataset_ids.shape[-1]
                    ),
                ],
                dim=-1,
            )

        n_rest = self.dim_embedding - n_components - p.shape[-1]
        assert n_rest >= 0
        zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
        return torch.cat((p, one_hot, zeros), dim=-1)

    def forward(self, inputs):
        t = inputs[:, [0]]
        x = inputs[:, 1 : 1 + self.dims_x]
        c = inputs[:, 1 + self.dims_x : 1 + self.dims_x + self.dims_c]
        if _condition_on_ids(self.cfg):
            dataset_ids = inputs[:, -1:].long()
        else:
            dataset_ids = None

        x_positions = getattr(self, "x_positions", None)  # balanced param.: x is a subset of c
        x_embedding = self.compute_embedding(
            x, n_components=self.dims_x if x_positions is None else self.dims_c, t=t,
            dataset_ids=dataset_ids, positions=x_positions,
        )

        c_embedding = self.compute_embedding(
            c, n_components=self.dims_c, dataset_ids=dataset_ids
        )

        transformer_out = self.network(src=c_embedding, tgt=x_embedding)
        v_pred = self.final_layer(
            torch.cat(
                [t.unsqueeze(1).repeat(1, x.size(1), 1), transformer_out],
                dim=-1,
            )
        ).squeeze(
            -1
        )  # not sure of this to be honest

        return v_pred
