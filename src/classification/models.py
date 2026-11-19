import os
import time
import torch
import torch.nn as nn

# from src.utils.mlflow import mlflow, LOGGING_ENABLED
# from mlflow.tracking import MlflowClient
# from mlflow.entities import Metric
from src.unfolding.nets.layers import VBLinear, TransformerEncoderLayerCustom

LOGGING_ENABLED = False


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


class TransformerClassifierNet(nn.Module):
    """Encoder-only transformer classifier.

    Tokenizes each of the ``dims_in`` input channels as one token using the same scheme as
    the unfolding ``TransformerCFM.compute_embedding`` (src/unfolding/models.py:730):
    ``token = [value, one-hot-position, zero-pad]``. Self-attention runs over the channel
    tokens (encoder only — a classifier has a single input set, no source/target split),
    then a mean-pool over tokens feeds a single-logit head. Requires
    ``dim_embedding >= dims_in + 1``.
    """

    def __init__(self, cfg, dims_in):
        super().__init__()
        self.cfg = cfg
        self.dims_in = dims_in
        self.dim_embedding = cfg.model.dim_embedding
        assert self.dim_embedding >= dims_in + 1, (
            f"dim_embedding ({self.dim_embedding}) must be >= dims_in+1 ({dims_in + 1}) "
            "to fit [value, one-hot-position] in each token"
        )
        encoder_layer = TransformerEncoderLayerCustom(
            d_model=self.dim_embedding,
            nhead=cfg.model.n_head,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.get("resid_pdrop", 0.0),
            batch_first=True,
            attn_pdrop=cfg.model.get("attn_pdrop", cfg.model.get("resid_pdrop", 0.0)),
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.model.n_encoder_layers)
        self.head = nn.Linear(self.dim_embedding, 1)

    def compute_embedding(self, p: torch.Tensor) -> torch.Tensor:
        """(B, dims_in) -> (B, dims_in, dim_embedding), one token per channel."""
        one_hot = torch.eye(self.dims_in, device=p.device, dtype=p.dtype)[None].expand(
            p.shape[0], -1, -1
        )
        p = p.unsqueeze(-1)  # (B, dims_in, 1)
        n_rest = self.dim_embedding - self.dims_in - p.shape[-1]
        zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
        return torch.cat((p, one_hot, zeros), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.compute_embedding(x)  # (B, dims_in, dim_embedding)
        out = self.encoder(emb)          # (B, dims_in, dim_embedding)
        pooled = out.mean(dim=1)         # (B, dim_embedding)
        return self.head(pooled)         # (B, 1)


class Model(nn.Module):
    def __init__(self, logger, cfg, dims_in, model_path, device):
        super().__init__()
        self.logger = logger
        self.cfg = cfg
        self.dims_in = dims_in
        self.dims_out = 1
        self.model_path = model_path
        self.device = device

        self.bayesian = self.cfg.model.get("bayesian", False)
        if self.bayesian:
            self.bayesian_factor = self.cfg.model.get("bayesian_factor", 1.0)
            self.bayesian_samples = self.cfg.evaluate.get("bayesian_samples", 20)
            self.logger.info(f"Bayesian mode: factor={self.bayesian_factor}, samples={self.bayesian_samples}")
        self.state_dict_attrs = ["network", "optimizer"]  # add more if needed

    def init_network(self):
        if self.cfg.model.network == "MLP":
            if self.bayesian:
                layer_class = VBLinear
                self.name = "BayesianClassifier"
                layer_kwargs = {}
                if self.cfg.model.get("prior_prec") is not None:
                    layer_kwargs["prior_prec"] = self.cfg.model.prior_prec
                if self.cfg.model.get("std_init") is not None:
                    layer_kwargs["std_init"] = self.cfg.model.std_init
            else:
                layer_class = nn.Linear
                layer_kwargs = {}
                self.name = "MLPClassifier"
            p_drop = float(self.cfg.model.get("dropout", 0.0))
            layers = []
            layers.append(layer_class(self.dims_in, self.cfg.model.internal_size, **layer_kwargs))
            layers.append(nn.ReLU())
            if p_drop > 0:
                layers.append(nn.Dropout(p_drop))
            for _ in range(self.cfg.model.hidden_layers):
                layers.append(
                    layer_class(
                        self.cfg.model.internal_size,
                        self.cfg.model.internal_size,
                        **layer_kwargs,
                    )
                )
                layers.append(nn.ReLU())
                if p_drop > 0:
                    layers.append(nn.Dropout(p_drop))
            layers.append(layer_class(self.cfg.model.internal_size, self.dims_out, **layer_kwargs))
            self.network = nn.Sequential(*layers)
            self.network.bayesian_layers = [
                layer for layer in layers if isinstance(layer, VBLinear)
            ]
        elif self.cfg.model.network == "Transformer":
            self.name = "TransformerClassifier"
            self.network = TransformerClassifierNet(self.cfg, self.dims_in)
            self.network.bayesian_layers = []  # non-bayesian; keeps save/validate paths happy
        else:
            raise ValueError(f"Unknown network type: {self.cfg.model.network}")

    def reset_random_state(self):
        assert self.bayesian
        for layer in self.network.bayesian_layers:
            layer.reset_random()

    def sample_random_state(self):
        assert self.bayesian
        return [layer.sample_random_state() for layer in self.network.bayesian_layers]

    def import_random_state(self, states):
        assert self.bayesian
        for layer, s in zip(self.network.bayesian_layers, states):
            layer.import_random_state(s)

    def generate_random_state(self):
        assert self.bayesian
        self.random_states = [self.sample_random_state() for _ in range(self.bayesian_samples)]

    # Overwrite in child class
    def sample(self, c):
        pass

    # Overwrite in child class
    def batch_loss(self, x, c, weight):
        pass

    def init_dataloaders(self, dataset, weights=None):
        p1 = dataset.p1_ppd
        p0 = dataset.p0_ppd
        w_p0 = dataset.w_p0 if hasattr(dataset, "w_p0") else None
        w_p1 = dataset.w_p1 if hasattr(dataset, "w_p1") else None
        channels = dataset.channels

        # apply stuff to make it such that only the channels of the data are used
        if w_p0 is None:
            self.logger.warning(
                "No weights provided for p0. Defaulting to uniform weights."
            )
            w_p0 = {
                f"{split}": torch.ones(p0[f"{split}"].shape[0])
                .to(p0[f"{split}"].dtype)
                .to(p0[f"{split}"].device)
                for split in ["trn", "tst", "val"]
            }
        if w_p1 is None:
            self.logger.warning(
                "No weights provided for p1. Defaulting to uniform weights."
            )
            w_p1 = {
                f"{split}": torch.ones(p1[f"{split}"].shape[0])
                .to(p1[f"{split}"].dtype)
                .to(p1[f"{split}"].device)
                for split in ["trn", "tst", "val"]
            }

        for split in ["trn", "tst", "val"]:
            X = torch.cat(
                [p0[f"{split}"][..., channels], p1[f"{split}"][..., channels]], dim=0
            ).to(torch.float32)
            y = torch.cat(
                [
                    torch.zeros(p0[f"{split}"].shape[0], dtype=torch.float32),
                    torch.ones(p1[f"{split}"].shape[0], dtype=torch.float32),
                ],
                dim=0,
            )
            W0 = w_p0[f"{split}"].sum()
            W1 = w_p1[f"{split}"].sum()
            w = torch.cat(
                [w_p0[f"{split}"] / W0, w_p1[f"{split}"] / W1], dim=0
            ).to(torch.float32)
            # perm = torch.randperm(len(X))
            # X = X[perm]
            # y = y[perm]
            # w = w[perm]

            # Create dataset
            self.logger.info(f"    Shapes:")
            self.logger.info(
                f"        {split}: X={X.shape}, y={y.shape}, w={w.shape}. Npos: {(y==1).sum().item()}, Nneg: {(y==0).sum().item()}"
            )
            self.logger.info(
                f"        Negative weights: {(w[y==1] < 0).sum().item()}, Positive weights: {(w[y==1] > 0).sum().item()}"
            )
            globals()[f"{split}set"] = torch.utils.data.TensorDataset(X, y, w)
        self.n_trn_samples = len(trnset)
        self.n_val_samples = len(valset)
        self.bs = self.cfg.train.batch_size
        self.bs_sample = self.cfg.train.get("batch_size_sample", self.bs)

        # balance training classes according to label y
        trn_sampler = _construct_sampler(trnset)
        val_sampler = _construct_sampler(valset)

        self.trnloader = torch.utils.data.DataLoader(
            trnset,
            batch_size=self.bs,
            shuffle=True,
            # sampler=trn_sampler,
            pin_memory=True,
        )
        self.valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=self.bs,
            # sampler=val_sampler,
            shuffle=False,
            pin_memory=True,
        )
        self.tstloader = torch.utils.data.DataLoader(
            tstset,
            batch_size=self.bs_sample,
            shuffle=False,
            pin_memory=True,
        )
        self.inference_trnloader = torch.utils.data.DataLoader(
            trnset,
            batch_size=self.bs_sample,
            shuffle=False,
            pin_memory=True,
        )
        self.inference_valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=self.bs_sample,
            shuffle=False,
            pin_memory=True,
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
        elif sched == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.train.get(
                    "lr_max", self.cfg.train.get("lr", 0.001)
                ),
                pct_start=0.15,
                total_steps=self.cfg.train.nits,
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
        nits = int(self.cfg.train.nits)
        val_freq = self.cfg.train.get(
            "val_freq", len(self.trnloader)
        )  # how often to validate/save, default: per epoch
        self.init_optimizer()
        self.init_scheduler()
        # set training parameter
        warm_start = self.cfg.train.get("warm_start", False)
        if warm_start:
            raise NotImplementedError("Warm-starting not implemented yet.")
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
                f"    Param_group[{i}]: lr={g['lr']}, wd={g.get('weight_decay', 0.0)}"
            )
        # iteration-wise logs
        trn_loss_it = []
        trn_lr = []
        grd_norm = []
        if self.bayesian:
            kl_it = []
            bce_it = []

        # epoch-wise logs
        val_loss_epoch = []
        if self.bayesian:
            kl_epoch = []
            bce_epoch = []

        if self.cfg.train.early_stopping.get("use", False):
            patience = self.cfg.train.early_stopping.get("patience", 10)
            early_stopping = EarlyStopping(patience=patience)
            self.logger.info(f"    Using early stopping with patience {patience}")
        self.logger.info(
            f"    Training model for {nits} iterations (= {nits // len(self.trnloader)} epochs)"
        )
        self.logger.info(
            f"    Validation frequency: {val_freq} its. Nb of train batches: {len(self.trnloader)}"
        )
        t0 = time.time()
        stop_training = False

        while current_it < nits and not stop_training:
            self.network.train()
            # (re)initialize running epoch accumulators (for optional epoch-avg train stats)
            epoch_trn_losses = []
            epoch_kl_losses = [] if self.bayesian else None
            epoch_bce_losses = [] if self.bayesian else None
            for i, batch in enumerate(self.trnloader):
                if current_it >= nits or stop_training:
                    break
                xb, yb, w = batch
                self.optimizer.zero_grad()
                if self.bayesian:
                    loss, loss_terms = self.batch_loss(xb, yb, w, 1 / self.n_trn_samples)
                else:
                    loss = self.batch_loss(xb, yb, w, 1 / self.n_trn_samples)
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
                    bce_it.append(loss_terms["bce"])
                    epoch_kl_losses.append(loss_terms["kl"])
                    epoch_bce_losses.append(loss_terms["bce"])

                trn_lr.append(self.optimizer.param_groups[0]["lr"])
                grd_norm.append(grad_norm)
                current_it += 1

                # validate after val_freq iterations or at the end of training
                has_completed_pass = current_it % val_freq == 0
                is_final_iter = current_it == nits

                if has_completed_pass or is_final_iter or stop_training:
                    # validate
                    avg_val = self.validate()
                    epoch_mean_trn = torch.tensor(epoch_trn_losses).mean().item()
                    # epoch-avg train (last pass’ mean)
                    val_loss_epoch.append(avg_val)

                    if self.bayesian:
                        kl_epoch.append(torch.tensor(epoch_kl_losses).mean().item())
                        bce_epoch.append(torch.tensor(epoch_bce_losses).mean().item())

                    # keep the latest losses available (before log_and_save so save("best") has them)
                    self.losses = {
                        "trn": trn_loss_it,
                        "val": val_loss_epoch,
                        "lr": trn_lr,
                        "grad_norm": grd_norm,
                    }
                    if self.bayesian:
                        self.losses["kl"] = kl_it
                        self.losses["bce"] = bce_it

                    self.log_and_save(
                        t0=t0,
                        iteration=current_it,  # total iters so far
                        avg_trn_loss=epoch_mean_trn,  # mean trn loss since last validation window
                        avg_val_loss=avg_val,  # mean val loss over valloader
                        val_freq=val_freq,
                    )
                    epoch_idx += 1
                    if self.scheduler is not None and isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        self.scheduler.step(avg_val)
                    if self.cfg.train.get("early_stopping", {}).get("use", False):
                        early_stopping(avg_val)
                        if early_stopping.early_stop:
                            self.logger.info(
                                f"Early stopping after iteration {current_it} with validation loss {avg_val:.8f}"
                            )
                            stop_training = True
                            break
            # end for batches
        # end of training loop
        self.save("final")
        self.logger.info(
            f"Finished training after {time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}. Nb of its: {current_it}, epochs: {epoch_idx}"
        )
        self.logger.info(f"Best validation loss: {self.best_val_loss:.8f}")
        self.logger.info(f"Last validation loss: {self.losses['val'][-1]:.8f}")

    @torch.no_grad()
    def validate(self):
        """Run full validation loader once; return mean loss and (optional) terms."""
        self.network.eval()
        losses = []
        kl_list = [] if self.bayesian else None
        bce_list = [] if self.bayesian else None

        for vbatch in self.valloader:
            vx, vy, vw = vbatch
            if self.bayesian:
                vloss, vterms = self.batch_loss(vx, vy, vw, 1 / self.n_trn_samples)
                kl_list.append(vterms["kl"])
                bce_list.append(vterms["bce"])
            else:
                vloss = self.batch_loss(vx, vy, vw)
            losses.append(vloss.detach().cpu())

        avg_val = torch.tensor(losses).mean().item()
        val_terms = None
        if self.bayesian:
            val_terms = {
                "kl": torch.tensor(kl_list).mean().item(),
                "bce": torch.tensor(bce_list).mean().item(),
            }
        return avg_val  # , val_terms

    def log_and_save(self, t0, iteration, avg_trn_loss, avg_val_loss, val_freq):
        if iteration == 1 * val_freq:
            self.logger.info(
                f"        Epoch {iteration // len(self.trnloader)} (it. {iteration}): tr_loss={avg_trn_loss:.8f}, val_loss={avg_val_loss:.8f}; ETA={time.strftime('%H:%M:%S', time.gmtime((time.time() - t0) * (self.cfg.train.nits - iteration) / iteration))}"
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
            pass
            # self.mlflow_log_metrics(t0, iteration)

        if iteration % max(1, iteration) == 0:
            self.logger.info(
                f"        Epoch {iteration // len(self.trnloader)} (it. {iteration}): tr_loss={avg_trn_loss:.8f}, val_loss={avg_val_loss:.8f}"
            )
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save("best")
                self.logger.info(
                    f"        Saved best model with val_loss={avg_val_loss:.8f} at it. {iteration}"
                )

    def evaluate(self, split="tst", loader=None):
        if loader is None:
            if split is None or split == "tst":
                loader = self.tstloader
            elif split == "val":
                loader = self.inference_valloader
            elif split == "trn":
                loader = self.inference_trnloader
            self.logger.info(f"Evaluating model on {split} set")
        else:
            self.logger.info(f"Evaluating model on provided loader: {split}")
        bayesian_samples = self.bayesian_samples if self.bayesian else 1
        if self.bayesian:
            self.logger.info(f"    Bayesian evaluation: generating {bayesian_samples} weight samples")
            self.generate_random_state()

        all_predictions = []
        labels = []
        self.eval()
        with torch.no_grad():
            for n in range(bayesian_samples):
                if self.bayesian:
                    self.import_random_state(self.random_states[n])
                sample_preds = []
                t0 = time.time()
                for i, batch in enumerate(loader):
                    xb, yb, _ = batch
                    pred = (
                        self.network(xb.to(self.device, non_blocking=True))
                        .squeeze()
                        .detach()
                    )
                    sample_preds.append(pred)
                    if n == 0:
                        labels.append(yb.detach())
                        t1 = time.time()
                        if i == 0:
                            self.logger.info(
                                f"    Total batches: {len(loader)}. Eval time estimate: {time.strftime('%H:%M:%S', time.gmtime(round((t1-t0) * len(loader), 1)))}"
                            )
                        log_every_percent = 0.25
                        if i % max(1, int(len(loader) * log_every_percent)) == 0:
                            self.logger.info(f"        Evaluated batch {i+1}/{len(loader)}")
                all_predictions.append(torch.cat(sample_preds))
            self.logger.info(f"    Finished evaluating. Saving predictions")

        labels = torch.cat(labels)
        if bayesian_samples > 1:
            predictions = torch.stack(all_predictions)  # (N_samples, N_events)
        else:
            predictions = all_predictions[0]
        return predictions, labels

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
        self.logger.info(f"Loaded model from {file}.")


class Classifier(Model):
    def __init__(self, logger, cfg, dims_in, model_path, device):
        super().__init__(logger, cfg, dims_in, model_path, device)
        self.init_network()

    def kl(self) -> torch.Tensor:
        """
        Compute the KL divergence between weight prior and posterior

        Returns:
            Scalar tensor with KL divergence
        """
        assert self.bayesian
        return sum(layer.kl() for layer in self.network.bayesian_layers)
    
    def batch_loss(self, x, y, w, kl_scale: float = 0.0):
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        w = w.to(self.device, non_blocking=True)
        pred = self.network(x).squeeze()
        # print(f" Positive samples in batch: {(y==1).sum().item()}, Negative samples in batch: {(y==0).sum().item()}")
        # print(f" Positive weights in batch: {(w[y==1])[:5]}, Negative weights in batch: {(w[y==0])[:5]}")
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        bce = criterion(pred, y)
        loss = (bce * w).sum() / w.sum()
        if self.bayesian:
            kl_loss = self.bayesian_factor * kl_scale * self.kl() / self.dims_out
            loss_terms = {"bce": loss.item(), "kl": kl_loss.item()}
            loss = loss + kl_loss
            return loss, loss_terms
        return loss


class Refiner(Classifier):
    """
    Neural Refiner for sample weight refinement (NRSW).
    Trained on p0 events only: label 0 = positive (1-w > 0), label 1 = pseudo-positive (1-w < 0).
    Refinement factor = exp(-r(x)) where r(x) is the refiner logit.
    """

    def __init__(self, logger, cfg, dims_in, model_path, device):
        super().__init__(logger, cfg, dims_in, model_path, device)
        self.name = "Refiner"

    def init_dataloaders(self, dataset):
        channels = dataset.channels
        for split in ["trn", "tst", "val"]:
            X = dataset.refiner_X[split][..., channels].to(torch.float32)
            y = dataset.refiner_y[split].to(torch.float32)
            w = dataset.refiner_w[split].to(torch.float32)
            self.logger.info(
                f"    Refiner {split}: X={X.shape}, "
                f"pos={(y == 1).sum().item()}, pp={(y == 0).sum().item()}"
            )
            globals()[f"{split}set"] = torch.utils.data.TensorDataset(X, y, w)
        self.n_trn_samples = len(trnset)
        self.n_val_samples = len(valset)
        self.bs = self.cfg.train.batch_size
        self.bs_sample = self.cfg.train.get("batch_size_sample", self.bs)
        kw = dict(pin_memory=True)
        self.trnloader = torch.utils.data.DataLoader(trnset, batch_size=self.bs, shuffle=True, **kw)
        self.valloader = torch.utils.data.DataLoader(valset, batch_size=self.bs, shuffle=False, **kw)
        self.tstloader = torch.utils.data.DataLoader(tstset, batch_size=self.bs_sample, shuffle=False, **kw)
        self.inference_trnloader = torch.utils.data.DataLoader(trnset, batch_size=self.bs_sample, shuffle=False, **kw)
        self.inference_valloader = torch.utils.data.DataLoader(valset, batch_size=self.bs_sample, shuffle=False, **kw)
        self.trnset, self.valset, self.tstset = trnset, valset, tstset


def _construct_sampler(
    set: torch.utils.data.TensorDataset,
) -> torch.utils.data.WeightedRandomSampler:
    y_trn = set.tensors[1]
    n_0 = max(1, (y_trn == 0).sum().item())
    n_1 = max(1, (y_trn == 1).sum().item())
    sample_w = torch.empty_like(y_trn)
    sample_w[y_trn == 0] = 1.0 / n_0
    sample_w[y_trn == 1] = 1.0 / n_1
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_w,
        num_samples=len(sample_w),
        replacement=True,
    )
    return sampler
