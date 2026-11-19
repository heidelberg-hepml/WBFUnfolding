"""Two-sample dataset for reweighting one reco sample onto another with the classifier pipeline.

Used for the ttbar truth-chain reweighting: p0 = particle-cut ttbar reco (reco-gen_*_rec, ~1.69M),
p1 = parton-cut ttbar reco (reco-hard_*_rec, ~52k). Same reco selection, different truth cuts.
Feature construction (PtPhiEtaM + derived features) and preprocessing (gaussianize +
standardization) are inherited unchanged from ClassificationDatasets, so the transformer classifier
sees exactly the inputs it sees in the background-subtraction runs.
"""
import numpy as np
import torch

from utils import physics as phys
from .classification import ClassificationDatasets


def _load_ptphietam(path, use_njets=False):
    d = torch.tensor(np.load(path), dtype=torch.float64)
    njets = d[:, 20:21] if use_njets else torch.ones((d.shape[0], 1), dtype=torch.float64) * 2.0
    return torch.cat(
        [phys.EPxPyPz_to_PtPhiEtaM(d[:, i : i + 4]) for i in range(0, 20, 4)] + [njets],  # + jet multiplicity
        dim=1,
    )


class ChainReweightingDataset(ClassificationDatasets):
    """cfg.dataset.chain_reweighting.{path_p0, path_p1}; cfg.dataset.trn_tst_val splits both
    samples after a seeded shuffle. ``p0_ppd_all`` keeps the preprocessed p0 in FILE ROW ORDER,
    so the learned weights can be evaluated for every p0 event and saved row-aligned."""

    def init_dataset(self):
        paths = self.cfg.chain_reweighting
        use_njets = bool(self.cfg.get("use_njets", False))
        self.p0 = _load_ptphietam(paths.path_p0, use_njets)
        self.p1 = _load_ptphietam(paths.path_p1, use_njets)
        self.logger.info(f"    p0: {self.p0.shape[0]} events from {paths.path_p0}")
        self.logger.info(f"    p1: {self.p1.shape[0]} events from {paths.path_p1}")
        self.channels = [c for c in self.channels if c not in [3, 7]]  # fixed lepton masses
        if use_njets and 20 not in self.channels:
            self.channels = self.channels + [20]  # reco jet multiplicity as an input
            self.logger.info("    Jet multiplicity (channel 20) used as an input")
        self.compute_derived_features()
        self.apply_preprocessing()
        self.p0_ppd_all = self.p0_ppd  # file row order, unsplit
        self.p1_ppd_all = self.p1_ppd
        self.p0_all, self.p1_all = self.p0, self.p1

        g = torch.Generator().manual_seed(int(paths.get("seed", 0)))
        self.perm_p0 = torch.randperm(self.p0.shape[0], generator=g)
        self.perm_p1 = torch.randperm(self.p1.shape[0], generator=g)
        split = lambda t, perm: self._split_data(t[perm], self.cfg.trn_tst_val)
        self.p0, self.p1 = split(self.p0, self.perm_p0), split(self.p1, self.perm_p1)
        self.p0_ppd, self.p1_ppd = split(self.p0_ppd, self.perm_p0), split(self.p1_ppd, self.perm_p1)
        # file rows of each split (to report out-of-sample closure on the val events)
        self.rows_p0 = split(self.perm_p0, torch.arange(self.p0_ppd_all.shape[0]))
        self.rows_p1 = split(self.perm_p1, torch.arange(self.p1_ppd_all.shape[0]))
        # uniform event weights; the loader normalizes each class to unit total weight
        self.w_p0 = {s: torch.ones(v.shape[0], dtype=torch.float64) for s, v in self.p0.items()}
        self.w_p1 = {s: torch.ones(v.shape[0], dtype=torch.float64) for s, v in self.p1.items()}

    def fold_assignment(self, n_folds, repeat=0):
        """Seeded fold index per file row (balanced: every fold gets N/n_folds rows +-1), for p0 and
        p1; a different `repeat` gives an independent assignment (repeated k-fold)."""
        g = torch.Generator().manual_seed(int(self.cfg.chain_reweighting.get("seed", 0)) + 1000 * (repeat + 1))
        f0 = torch.randperm(self.p0_all.shape[0], generator=g) % n_folds
        f1 = torch.randperm(self.p1_all.shape[0], generator=g) % n_folds
        return f0, f1

    def set_fold(self, f0, f1, k):
        """Cross-fitting: trn = rows NOT in fold k, val = rows in fold k (held out; the member trained
        here predicts exactly these rows), tst empty. rows_p0/rows_p1 give each split's file rows."""
        self.rows_p0, self.rows_p1 = {}, {}
        for s, sel0, sel1 in (("trn", f0 != k, f1 != k), ("val", f0 == k, f1 == k),
                              ("tst", torch.zeros_like(f0, dtype=torch.bool), torch.zeros_like(f1, dtype=torch.bool))):
            self.rows_p0[s], self.rows_p1[s] = torch.where(sel0)[0], torch.where(sel1)[0]
        self.p0 = {s: self.p0_all[r] for s, r in self.rows_p0.items()}
        self.p1 = {s: self.p1_all[r] for s, r in self.rows_p1.items()}
        self.p0_ppd = {s: self.p0_ppd_all[r] for s, r in self.rows_p0.items()}
        self.p1_ppd = {s: self.p1_ppd_all[r] for s, r in self.rows_p1.items()}
        self.w_p0 = {s: torch.ones(len(r), dtype=torch.float64) for s, r in self.rows_p0.items()}
        self.w_p1 = {s: torch.ones(len(r), dtype=torch.float64) for s, r in self.rows_p1.items()}
