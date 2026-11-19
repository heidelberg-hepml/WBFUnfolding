# type: ignore
import os
import torch
import numpy as np
from utils import physics as phys
from .dataset import Observable, return_obs, get_hardcoded_bins


class ClassificationDatasets:
    def __init__(self, logger, cfg, **kwargs):
        self.logger = logger
        self.cfg = cfg
        self.channels = self.cfg.get("channels", None)
        if self.channels is None:
            raise ValueError("Channels not specified in the dataset parameters")
        if self.channels == []:
            # Use all of the useful channels by default
            self.channels = [
                0,
                1,
                2,  # l1
                4,
                5,
                6,  # l2
                8,
                9,
                10,
                11,  # j1/q1
                12,
                13,
                14,
                15,  # j2/q2
                16,
                17,  # 18, 19,    # miss
            ]

        if self.channels is None:
            raise ValueError("Channels not specified in the dataset parameters")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()
        self.init_observables()

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, value):
        """
        Channels should ALWAYS be sorted to avoid misalignments with the unfolded data
        With this we make sure that self.channels definitions are always sorted
        """
        self._channels = sorted(value)

    def init_dataset(self):
        n_data = self.cfg.get("n_data", None)
        # truth_level selects which reco is classified and which truth is dumped (see config).
        # parton -> reco-hard reco + *_hard truth (22 cols); gen -> reco-gen reco + *_gen truth (21 cols).
        self.truth_level = self.cfg.get("truth_level", "gen")
        reco_key = ("path_reco_corresponding_to_parton" if self.truth_level == "parton"
                    else "path_reco_corresponding_to_gen")
        truth_key = "path_parton" if self.truth_level == "parton" else "path_gen"
        # background_truth_level: which chain's reco files the BACKGROUND (ttbar, WWjj) comes from.
        # The background enters only through its reco, so e.g. a parton-level classifier can use the
        # (32x larger for ttbar) gen-chain background; both chains share the same reco selection.
        # Its truth columns are then the gen ones, a placeholder: background truth is never compared.
        self.background_truth_level = self.cfg.get("background_truth_level", None) or self.truth_level
        # ttbar_truth_level: chain of the ttbar reco alone (default: background_truth_level), e.g.
        # parton signal/WWjj + particle-cut ttbar reweighted onto the parton-cut ttbar shape with
        # per-event weights (ttbar_event_weights, row-aligned with the ttbar file; see
        # experiments/ttbar_chain_reweighting.py).
        self.ttbar_truth_level = self.cfg.get("ttbar_truth_level", None) or self.background_truth_level
        _reco = lambda lvl: ("path_reco_corresponding_to_parton" if lvl == "parton"
                             else "path_reco_corresponding_to_gen")
        _truth = lambda lvl: "path_parton" if lvl == "parton" else "path_gen"
        bkg_reco_key, bkg_truth_key = _reco(self.background_truth_level), _truth(self.background_truth_level)
        tt_reco_key, tt_truth_key = _reco(self.ttbar_truth_level), _truth(self.ttbar_truth_level)
        path_HWW = self.cfg.HWW[reco_key] if self.cfg.get("use_HWW", True) else None
        path_ttbar = self.cfg.ttbar[tt_reco_key] if self.cfg.get("use_ttbar", True) else None
        path_WWjj = self.cfg.WWjj[bkg_reco_key] if self.cfg.get("use_WWjj", False) else None

        if self.device == "cpu" and "remote" in path_HWW:
            self.logger.info(
                f"    Path {path_HWW} is remote but you are running on local cpu, changing it to data/"
            )
            data_path = "data"
            raise ValueError(
                "Running on CPU with remote data path is not supported. Please set use_HWW to False or run on GPU."
            )

        if path_HWW is None and path_ttbar is None and path_WWjj is None:
            raise ValueError(
                "Paths not specified for any of the datasets. Please specify at least two dataset to classify."
            )
        datasets = {}
        raw_reco = {}   # raw EPxPyPz reco tensors (N,21 float32) per source
        min_length = int(1e30)
        for name, data_path in zip(
            ["HWW", "ttbar", "WWjj"], [path_HWW, path_ttbar, path_WWjj]
        ):
            if data_path is None:
                continue
            if not os.path.exists(data_path):
                raise ValueError(f"Path {data_path} does not exist")
            else:
                d = (
                    np.load(data_path)[:n_data]
                    if n_data is not None
                    else np.load(data_path)
                )
                d_tensor = torch.tensor(d, dtype=torch.float64)
                PtPhiEtaM = torch.cat(
                    [
                        phys.EPxPyPz_to_PtPhiEtaM(d_tensor[:, 0:4]),
                        phys.EPxPyPz_to_PtPhiEtaM(d_tensor[:, 4:8]),
                        phys.EPxPyPz_to_PtPhiEtaM(d_tensor[:, 8:12]),
                        phys.EPxPyPz_to_PtPhiEtaM(d_tensor[:, 12:16]),
                        phys.EPxPyPz_to_PtPhiEtaM(d_tensor[:, 16:20]),
                        # jet multiplicity: the real reco count (col 20) with dataset.use_njets,
                        # else the legacy constant 2 (then it is not an input either)
                        d_tensor[:, 20:21] if self.cfg.get("use_njets", False)
                        else torch.ones((d_tensor.shape[0], 1)) * 2.0,
                    ],
                    dim=1,
                )
                datasets[name] = PtPhiEtaM
                raw_reco[name] = d_tensor.float()   # (N, 21) raw EPxPyPz for later dump
                min_length = min(min_length, len(datasets[name]))
                self.n_particles = d_tensor.shape[1] // 4
                self.logger.info(
                    f"    Data loaded for {data_path} with {len(PtPhiEtaM)} events."
                )

        # Optionally load gen files (raw EPxPyPz + multiplicity, 21 cols) for unfolding dump.
        # Requires path_gen to be set under each source in the config. The 21st column
        # (index 20, jet multiplicity) is kept so the unfolder's WBF dataset can read
        # part[:, 20:21] and the transverse_mass features land at indices 21-25.
        # Keep enough truth columns that the dump is column-identical to what the unfolder's WBF
        # reads from the real truth file: gen files have 21 cols, parton (*_hard) files 22.
        n_truth_cols = 22 if self.truth_level == "parton" else 21
        raw_gen = {}
        gen_paths = {
            "HWW":   self.cfg.HWW.get(truth_key, None)   if self.cfg.get("use_HWW",   True)  else None,
            "ttbar": self.cfg.ttbar.get(tt_truth_key, None)   if self.cfg.get("use_ttbar", True)  else None,
            "WWjj":  self.cfg.WWjj.get(bkg_truth_key, None)   if self.cfg.get("use_WWjj",  False) else None,
        }
        for name, path_gen in gen_paths.items():
            if path_gen is None or not os.path.exists(path_gen):
                if path_gen is not None:
                    self.logger.warning(f"    Truth path {path_gen} not found; skipping raw truth for {name}.")
                raw_gen[name] = None
                continue
            d_g = (
                np.load(path_gen)[:n_data, :n_truth_cols]
                if n_data is not None
                else np.load(path_gen)[:, :n_truth_cols]
            )
            if d_g.shape[1] < n_truth_cols:
                # gen-chain background under a parton-level classifier: pad the extra parton column
                # (ignored by the parton unfolder) so all p_data rows have the same width
                d_g = np.pad(d_g, ((0, 0), (0, n_truth_cols - d_g.shape[1])), constant_values=1.0)
            raw_gen[name] = torch.tensor(d_g, dtype=torch.float32)  # (N, 21|22) EPxPyPz + multiplicity [+ flag]
            level = {"HWW": self.truth_level, "ttbar": self.ttbar_truth_level}.get(name, self.background_truth_level)
            self.logger.info(f"    Truth ({level}) data loaded for {path_gen} with {raw_gen[name].shape[0]} events.")

        self.channels = [
            i for i in self.channels if i not in [3, 7]
        ]  # eliminate quark and lepton masses (fixed)
        if self.cfg.get("use_njets", False) and 20 not in self.channels:
            self.channels = self.channels + [20]  # reco jet multiplicity as a classifier input
            self.logger.info("    Jet multiplicity (channel 20) used as an input")
        self.HWW = datasets["HWW"]
        self.ttbar = datasets["ttbar"]
        self.WWjj = datasets["WWjj"]
        tt_w = self.cfg.get("ttbar_event_weights", None)
        if tt_w is not None:
            self.ttbar_event_weights = torch.tensor(np.load(tt_w), dtype=torch.float64)
            # row-aligned with the WHOLE ttbar file (weights made for another file version must fail here)
            n_file = np.load(path_ttbar, mmap_mode="r").shape[0]
            assert self.ttbar_event_weights.shape[0] == n_file, (
                f"ttbar_event_weights has {self.ttbar_event_weights.shape[0]} rows, ttbar file {n_file}")
            self.ttbar_event_weights = self.ttbar_event_weights[: self.ttbar.shape[0]]  # n_data
            assert torch.isfinite(self.ttbar_event_weights).all() and (self.ttbar_event_weights >= 0).all()
            self.logger.info(f"    ttbar event weights from {tt_w} (mean {self.ttbar_event_weights.mean():.4f})")
        else:
            self.ttbar_event_weights = None

        # Raw EPxPyPz arrays for building aligned p_data reco/parton dumps (may be None)
        self.HWW_raw_reco    = raw_reco.get("HWW")        # (N, 21) float32
        self.ttbar_raw_reco  = raw_reco.get("ttbar")      # (N, 21) float32
        self.WWjj_raw_reco   = raw_reco.get("WWjj")       # (N, 21) float32
        self.HWW_raw_gen  = raw_gen.get("HWW")      # (N, 21) float32, 5×4 four-vec cols + multiplicity
        self.ttbar_raw_gen = raw_gen.get("ttbar")   # (N, 21) float32
        self.WWjj_raw_gen  = raw_gen.get("WWjj")    # (N, 21) float32

        # Optional seeded shuffle of the ttbar rows before any split (the ttbar files are two
        # sub-samples, one at both file ends): every split then sees the same ttbar mixture.
        # ttbar_row_perm[i] = file row of the i-th ttbar row used from here on.
        self.ttbar_row_perm = None
        tt_seed = self.cfg.get("shuffle_ttbar_seed", None)
        if tt_seed is not None:
            n_tt = self.ttbar.shape[0]
            perm = torch.randperm(n_tt, generator=torch.Generator().manual_seed(int(tt_seed)))
            self.ttbar = self.ttbar[perm]
            if self.ttbar_event_weights is not None:
                self.ttbar_event_weights = self.ttbar_event_weights[perm]
            for attr in ("ttbar_raw_reco", "ttbar_raw_gen"):
                t = getattr(self, attr)
                if t is not None:
                    assert t.shape[0] == n_tt, f"{attr} has {t.shape[0]} rows, ttbar {n_tt}"
                    setattr(self, attr, t[perm])
            self.ttbar_row_perm = perm
            self.logger.info(f"    ttbar rows shuffled before splitting (seed {int(tt_seed)})")

        if self.cfg.get("background_mode", "subtraction") == "subtraction":
            self.construct_background_subtraction_classes()
        else:
            raise ValueError(f"Invalid background_mode: {self.cfg.background_mode}. Has to be subtraction")

        self.compute_derived_features()
        self.apply_preprocessing()
        self.fiducial_region_weights()
        self.split_data()
        self.normalize_weights()

    def fiducial_region_weights(self):
        # For each sample i of size n_i,
        # assign weights w_i according to
        # their expected fractions alpha_i in fiducial region
        # such that sum_{i}^{n_i} w_i = alpha_i

        # ATL-COM-PHYS-2020-843
        self.alpha_HWW = 114 / 789
        self.alpha_ttbar = 423 / 789
        self.alpha_WWjj = 252 / 789

    def construct_background_subtraction_classes(self):
        """
        Build p0 (= p_data) and p1 (= p_bkg_MC) for the background-subtraction classifier.

        When dataset.holdout_frac > 0 the data is partitioned by file order into two disjoint pools:
        a TRAIN pool (first 1 - holdout_frac of each sample) used to train the classifier, and an APP
        pool (last holdout_frac) from which p_data is built and on which the learned weights are
        applied + dumped. The APP pool lands entirely in the "tst" split (trn/val are carved from the
        TRAIN pool), so the dumped p_data is disjoint from classifier training. holdout_frac == 0
        reproduces the legacy single-pool behaviour (p_data == "trn" split, dumped from "trn").
        """
        holdout = float(self.cfg.get("holdout_frac", 0.0))
        self.logger.info(
            f"    Using {self.ttbar.shape[0]} ttbar and {self.WWjj.shape[0]} WWjj background events "
            f"(holdout_frac={holdout})."
        )
        raw_reco = (self.HWW_raw_reco, self.ttbar_raw_reco, self.WWjj_raw_reco)
        raw_gen = (self.HWW_raw_gen, self.ttbar_raw_gen, self.WWjj_raw_gen)

        es = self.cfg.get("explicit_split", None)
        if es is not None and bool(es.get("use", False)):
            # ---- explicit file-order split: each split = a [lo, hi) fraction slice of every
            # source file. Membership is decided by file position alone, so containment w.r.t.
            # the unfolder's contiguous split can be guaranteed exactly. Overrides holdout_frac/trn_tst_val.
            def _rng(t, lo, hi):
                if t is None:
                    return None
                n = t.shape[0]
                return t[int(n * lo) : int(n * hi)]

            pools, counts = {}, {}
            for split in ("trn", "tst", "val"):
                lo, hi = (float(x) for x in es[split])
                pools[split] = self._assemble_pool(
                    _rng(self.HWW, lo, hi), _rng(self.ttbar, lo, hi), _rng(self.WWjj, lo, hi),
                    tuple(_rng(t, lo, hi) for t in raw_reco),
                    tuple(_rng(t, lo, hi) for t in raw_gen),
                    ttbar_w=_rng(self.ttbar_event_weights, lo, hi),
                )
                counts[split] = {
                    "HWW": int(_rng(self.HWW, lo, hi).shape[0]),
                    "ttbar": int(_rng(self.ttbar, lo, hi).shape[0]),
                    "WWjj": int(_rng(self.WWjj, lo, hi).shape[0]),
                    "range": [lo, hi],
                }
                self.logger.info(f"    explicit_split[{split}] = [{lo}, {hi}) of each file: {counts[split]}")
            self._stitch_explicit_pools(pools)
            self.pool_counts = counts
            self.dump_split = str(es.get("dump_split", "trn"))
        elif holdout <= 0.0:
            # ---- legacy single-pool behaviour (p_data == train split) ------------------------
            pool = self._assemble_pool(self.HWW, self.ttbar, self.WWjj, raw_reco, raw_gen,
                                       ttbar_w=self.ttbar_event_weights)
            self._store_single_pool(pool)
            self._split_idx_p0 = None
            self._split_idx_p1 = None
            self.dump_split = "trn"
        else:
            # ---- honest two-pool split: APP (last holdout_frac, file order) held out ----------
            def _slice(t):
                if t is None:
                    return None, None
                n_app = int(t.shape[0] * holdout)
                n_trn = t.shape[0] - n_app
                return t[:n_trn], t[n_trn:]

            HWW_trn, HWW_app = _slice(self.HWW)
            ttbar_trn, ttbar_app = _slice(self.ttbar)
            WWjj_trn, WWjj_app = _slice(self.WWjj)
            rr_trn = tuple(_slice(t)[0] for t in raw_reco)
            rr_app = tuple(_slice(t)[1] for t in raw_reco)
            rg_trn = tuple(_slice(t)[0] for t in raw_gen)
            rg_app = tuple(_slice(t)[1] for t in raw_gen)

            ttw_trn, ttw_app = _slice(self.ttbar_event_weights)
            train_pool = self._assemble_pool(HWW_trn, ttbar_trn, WWjj_trn, rr_trn, rg_trn, ttbar_w=ttw_trn)
            app_pool = self._assemble_pool(HWW_app, ttbar_app, WWjj_app, rr_app, rg_app, ttbar_w=ttw_app)
            self._stitch_pools(train_pool, app_pool)
            self.dump_split = "tst"

        self.logger.info(
            f"    p0: {self.p0.shape[0]} events, of which {self.mask_sig_in_p0.sum()} signal and "
            f"{self.mask_bkg_in_p0.sum()} background "
            f"(ttbar: {self.mask_ttbar_in_p0.sum()}, WWjj: {self.mask_WWjj_in_p0.sum()})"
        )
        self.logger.info(
            f"    p1: {self.p1.shape[0]} MC background "
            f"(ttbar: {self.mask_ttbar_in_p1.sum()}, WWjj: {self.mask_WWjj_in_p1.sum()})"
        )

    def _assemble_pool(self, HWW, ttbar, WWjj, raw_reco=(None, None, None), raw_gen=(None, None, None),
                       ttbar_w=None):
        """Build (p_data, p_bkg_MC) + aligned masks/raw from one disjoint pool of events.

        p_data = [signal] + [half of the shuffled ttbar+WWjj]; p_bkg_MC = the other half. Mirrors the
        original single-pool construction but returns a dict so callers can stitch one or more pools.
        Weights are ones here (ttbar: ttbar_w, its per-event shape weights, if given) and rescaled
        per split and process by normalize_weights() to sum to alpha_process.
        """
        HWW_rr, ttbar_rr, WWjj_rr = raw_reco
        HWW_rg, ttbar_rg, WWjj_rg = raw_gen
        n_sig = HWW.shape[0]
        n_ttbar = ttbar.shape[0]
        n_WWjj = WWjj.shape[0]
        split_idx = (n_ttbar + n_WWjj) // 2

        # background = ttbar ++ WWjj, shuffled while tracking process identity
        p_background = torch.cat([ttbar, WWjj], dim=0)
        mask_ttbar_noWWjj = torch.zeros(p_background.shape[0], dtype=torch.bool)
        mask_ttbar_noWWjj[:n_ttbar] = True
        perm = torch.randperm(p_background.shape[0])
        p_background = p_background[perm]
        mask_ttbar_noWWjj = mask_ttbar_noWWjj[perm]
        w_background = torch.cat([
            ttbar_w.float() if ttbar_w is not None else torch.ones(n_ttbar),
            torch.ones(n_WWjj),
        ])[perm]

        # half -> p_bkg (into p_data), half -> p_bkg_MC (into p1)
        p_bkg = p_background[:split_idx]
        mask_ttbar_in_bkg = mask_ttbar_noWWjj[:split_idx]
        p_bkg_MC = p_background[split_idx:]
        mask_ttbar_in_bkg_MC = mask_ttbar_noWWjj[split_idx:]

        # p_data = signal ++ p_bkg
        n_bkg = p_bkg.shape[0]
        p_data = torch.cat([HWW, p_bkg], dim=0)
        mask_sig = torch.cat(
            [torch.ones(n_sig, dtype=torch.bool), torch.zeros(n_bkg, dtype=torch.bool)]
        )
        mask_ttbar = torch.cat([torch.zeros(n_sig, dtype=torch.bool), mask_ttbar_in_bkg])
        mask_WWjj = torch.cat([torch.zeros(n_sig, dtype=torch.bool), ~mask_ttbar_in_bkg])
        idx = torch.cat(
            [torch.arange(n_sig, dtype=torch.long), torch.full((n_bkg,), -1, dtype=torch.long)]
        )
        # source position within this pool: signal -> row in the pool's HWW slice (same as idx);
        # background -> row in the pool's concatenated [ttbar ++ WWjj] block (pre-shuffle), i.e.
        # perm[:split_idx]. Lets downstream code map every p_data row back to its source file row.
        src_pos = torch.cat([torch.arange(n_sig, dtype=torch.long), perm[:split_idx].to(torch.long)])

        perm_p0 = torch.randperm(p_data.shape[0])
        perm_p1 = torch.randperm(p_bkg_MC.shape[0])
        out = {
            "p_data": p_data[perm_p0],
            "w_data": torch.cat([torch.ones(n_sig), w_background[:split_idx]])[perm_p0],
            "mask_sig": mask_sig[perm_p0],
            "mask_bkg": (~mask_sig)[perm_p0],
            "mask_ttbar": mask_ttbar[perm_p0],
            "mask_WWjj": mask_WWjj[perm_p0],
            "idx": idx[perm_p0],
            "src_pos": src_pos[perm_p0],
            "p_bkg_MC": p_bkg_MC[perm_p1],
            "w_bkg_MC": w_background[split_idx:][perm_p1],
            "mask_ttbar_MC": mask_ttbar_in_bkg_MC[perm_p1],
            "mask_WWjj_MC": (~mask_ttbar_in_bkg_MC)[perm_p1],
        }

        # raw EPxPyPz reco/parton aligned with p_data (same perm + perm_p0 as above)
        if HWW_rr is not None and ttbar_rr is not None and WWjj_rr is not None:
            p_bkg_rr = torch.cat([ttbar_rr, WWjj_rr], dim=0)[perm][:split_idx]
            out["p_data_raw_reco"] = torch.cat([HWW_rr, p_bkg_rr], dim=0)[perm_p0]
        else:
            out["p_data_raw_reco"] = None
        if HWW_rg is not None and ttbar_rg is not None and WWjj_rg is not None:
            p_bkg_rg = torch.cat([ttbar_rg, WWjj_rg], dim=0)[perm][:split_idx]
            out["p_data_raw_gen"] = torch.cat([HWW_rg, p_bkg_rg], dim=0)[perm_p0]
        else:
            out["p_data_raw_gen"] = None
        return out

    def _store_single_pool(self, pool):
        """Map a single assembled pool onto the p0/p1 attributes (legacy layout, full tensors)."""
        self.p0 = pool["p_data"]
        self.w_p0 = pool["w_data"]
        self.mask_sig_in_p0 = pool["mask_sig"]
        self.mask_bkg_in_p0 = pool["mask_bkg"]
        self.mask_ttbar_in_p0 = pool["mask_ttbar"]
        self.mask_WWjj_in_p0 = pool["mask_WWjj"]
        self.idx_p0 = pool["idx"]
        self.src_pos_p0 = pool.get("src_pos")
        self.p0_raw_reco = pool["p_data_raw_reco"]
        self.p0_raw_gen = pool["p_data_raw_gen"]
        self.p1 = pool["p_bkg_MC"]
        self.w_p1 = pool["w_bkg_MC"]
        self.mask_ttbar_in_p1 = pool["mask_ttbar_MC"]
        self.mask_WWjj_in_p1 = pool["mask_WWjj_MC"]

    def _stitch_pools(self, train_pool, app_pool):
        """Concatenate [TRAIN, APP] pools into p0/p1 and record explicit split indices.

        trn/val are carved from the TRAIN pool (val fraction = trn_tst_val[2]); the whole APP pool
        becomes "tst". split_data() then slices every aligned p0/p1 array by these indices.
        """

        def _cat(key):
            a, b = train_pool[key], app_pool[key]
            if a is None or b is None:
                return None
            return torch.cat([a, b], dim=0)

        self.p0 = _cat("p_data")
        self.w_p0 = _cat("w_data")
        self.mask_sig_in_p0 = _cat("mask_sig")
        self.mask_bkg_in_p0 = _cat("mask_bkg")
        self.mask_ttbar_in_p0 = _cat("mask_ttbar")
        self.mask_WWjj_in_p0 = _cat("mask_WWjj")
        self.idx_p0 = _cat("idx")
        self.src_pos_p0 = _cat("src_pos")
        self.p0_raw_reco = _cat("p_data_raw_reco")
        self.p0_raw_gen = _cat("p_data_raw_gen")
        self.p1 = _cat("p_bkg_MC")
        self.w_p1 = _cat("w_bkg_MC")
        self.mask_ttbar_in_p1 = _cat("mask_ttbar_MC")
        self.mask_WWjj_in_p1 = _cat("mask_WWjj_MC")

        val_frac = float(self.cfg.trn_tst_val[2])
        n0_train = train_pool["p_data"].shape[0]
        n0_app = app_pool["p_data"].shape[0]
        n0_trn = n0_train - int(n0_train * val_frac)
        self._split_idx_p0 = {
            "trn": torch.arange(0, n0_trn),
            "tst": torch.arange(n0_train, n0_train + n0_app),
            "val": torch.arange(n0_trn, n0_train),
        }
        m1_train = train_pool["p_bkg_MC"].shape[0]
        m1_app = app_pool["p_bkg_MC"].shape[0]
        m1_trn = m1_train - int(m1_train * val_frac)
        self._split_idx_p1 = {
            "trn": torch.arange(0, m1_trn),
            "tst": torch.arange(m1_train, m1_train + m1_app),
            "val": torch.arange(m1_trn, m1_train),
        }
        self.logger.info(
            f"    Honest holdout split: p0 trn/val/tst = {n0_trn}/{n0_train - n0_trn}/{n0_app}, "
            f"p1 trn/val/tst = {m1_trn}/{m1_train - m1_trn}/{m1_app}"
        )

    def _stitch_explicit_pools(self, pools):
        """Concatenate the [trn, tst, val] pools of explicit_split mode into p0/p1 and record
        explicit split indices (one contiguous block per split). Mirror of _stitch_pools."""
        order = ("trn", "tst", "val")

        def _cat(key):
            arrs = [pools[s][key] for s in order]
            if any(a is None for a in arrs):
                return None
            return torch.cat(arrs, dim=0)

        self.p0 = _cat("p_data")
        self.w_p0 = _cat("w_data")
        self.mask_sig_in_p0 = _cat("mask_sig")
        self.mask_bkg_in_p0 = _cat("mask_bkg")
        self.mask_ttbar_in_p0 = _cat("mask_ttbar")
        self.mask_WWjj_in_p0 = _cat("mask_WWjj")
        self.idx_p0 = _cat("idx")
        self.src_pos_p0 = _cat("src_pos")
        self.p0_raw_reco = _cat("p_data_raw_reco")
        self.p0_raw_gen = _cat("p_data_raw_gen")
        self.p1 = _cat("p_bkg_MC")
        self.w_p1 = _cat("w_bkg_MC")
        self.mask_ttbar_in_p1 = _cat("mask_ttbar_MC")
        self.mask_WWjj_in_p1 = _cat("mask_WWjj_MC")

        idx0, idx1, o0, o1 = {}, {}, 0, 0
        for s in order:
            n0 = pools[s]["p_data"].shape[0]
            n1 = pools[s]["p_bkg_MC"].shape[0]
            idx0[s] = torch.arange(o0, o0 + n0)
            idx1[s] = torch.arange(o1, o1 + n1)
            o0 += n0
            o1 += n1
        self._split_idx_p0 = idx0
        self._split_idx_p1 = idx1
        self.logger.info(
            "    Explicit split: p0 trn/tst/val = "
            f"{len(idx0['trn'])}/{len(idx0['tst'])}/{len(idx0['val'])}, "
            f"p1 trn/tst/val = {len(idx1['trn'])}/{len(idx1['tst'])}/{len(idx1['val'])}"
        )

    def normalize_weights(self):
        """Normalize weights per split so that sum(w_component) = alpha_component.

        Must be called after split_data(). For each train/test/val split the events of component *i*
        get alpha_i * w / sum(w) over that component's events in the split: alpha_i / N_i for the
        (default) uniform weights, and the same total with the per-event ttbar shape weights.
        """

        def _norm(w, mask, alpha):
            if mask.sum().item() > 0:
                w[mask] = alpha * w[mask] / w[mask].sum()

        for split in self.w_p0.keys():
            _norm(self.w_p0[split], self.mask_sig_in_p0[split], self.alpha_HWW)
            _norm(self.w_p0[split], self.mask_ttbar_in_p0[split], self.alpha_ttbar)
            _norm(self.w_p0[split], self.mask_WWjj_in_p0[split], self.alpha_WWjj)
            if self.cfg.get("background_mode", "subtraction") == "subtraction":
                _norm(self.w_p1[split], self.mask_ttbar_in_p1[split], self.alpha_ttbar)
                _norm(self.w_p1[split], self.mask_WWjj_in_p1[split], self.alpha_WWjj)
            else:
                raise ValueError(f"Invalid background_mode: {self.cfg.background_mode}. Has to be subtraction")
            self.logger.info(
                f"    Weights normalized for {split}: "
                f"sum(w_p0)={self.w_p0[split].sum():.4f}, "
                f"sum(w_p1)={self.w_p1[split].sum():.4f}"
            )

    def split_data(self):
        """Split preprocessed data (p0_ppd, p1_ppd) into training/test/validation dicts."""
        if getattr(self, "_split_idx_p0", None) is not None:
            # Honest holdout: slice every aligned array by the explicit per-split indices
            # (p0-family by _split_idx_p0, p1-family by _split_idx_p1) instead of by fraction.
            idx0, idx1 = self._split_idx_p0, self._split_idx_p1

            def _by_idx(data, idx):
                return {s: data[idx[s]] for s in ["trn", "tst", "val"]}

            self.w_p0 = _by_idx(self.w_p0, idx0)
            self.w_p1 = _by_idx(self.w_p1, idx1)
            self.p0 = _by_idx(self.p0, idx0)
            self.p1 = _by_idx(self.p1, idx1)
            self.p0_ppd = _by_idx(self.p0_ppd, idx0)
            self.p1_ppd = _by_idx(self.p1_ppd, idx1)
            self.mask_sig_in_p0 = _by_idx(self.mask_sig_in_p0, idx0)
            self.mask_ttbar_in_p0 = _by_idx(self.mask_ttbar_in_p0, idx0)
            self.mask_ttbar_in_p1 = _by_idx(self.mask_ttbar_in_p1, idx1)
            self.mask_WWjj_in_p0 = _by_idx(self.mask_WWjj_in_p0, idx0)
            self.mask_WWjj_in_p1 = _by_idx(self.mask_WWjj_in_p1, idx1)
            self.mask_bkg_in_p0 = _by_idx(self.mask_bkg_in_p0, idx0)
            self.idx_p0 = _by_idx(self.idx_p0, idx0)
            if getattr(self, "src_pos_p0", None) is not None:
                self.src_pos_p0 = _by_idx(self.src_pos_p0, idx0)
            if self.p0_raw_reco is not None:
                self.p0_raw_reco = _by_idx(self.p0_raw_reco, idx0)
            if self.p0_raw_gen is not None:
                self.p0_raw_gen = _by_idx(self.p0_raw_gen, idx0)
            return

        self.w_p0 = self._split_data(self.w_p0, self.cfg.trn_tst_val)
        self.w_p1 = self._split_data(self.w_p1, self.cfg.trn_tst_val)
        self.p0 = self._split_data(self.p0, self.cfg.trn_tst_val)
        self.p1 = self._split_data(self.p1, self.cfg.trn_tst_val)
        self.p0_ppd = self._split_data(self.p0_ppd, self.cfg.trn_tst_val)
        self.p1_ppd = self._split_data(self.p1_ppd, self.cfg.trn_tst_val)

        self.mask_sig_in_p0 = self._split_data(
            self.mask_sig_in_p0, self.cfg.trn_tst_val
        )
        self.mask_ttbar_in_p0 = self._split_data(
            self.mask_ttbar_in_p0, self.cfg.trn_tst_val
        )
        self.mask_ttbar_in_p1 = self._split_data(
            self.mask_ttbar_in_p1, self.cfg.trn_tst_val
        )
        self.mask_WWjj_in_p0 = self._split_data(
            self.mask_WWjj_in_p0, self.cfg.trn_tst_val
        )
        self.mask_WWjj_in_p1 = self._split_data(
            self.mask_WWjj_in_p1, self.cfg.trn_tst_val
        )
        self.mask_bkg_in_p0 = self._split_data(
            self.mask_bkg_in_p0, self.cfg.trn_tst_val
        )
        self.idx_p0 = self._split_data(self.idx_p0, self.cfg.trn_tst_val)
        if getattr(self, "src_pos_p0", None) is not None:
            self.src_pos_p0 = self._split_data(self.src_pos_p0, self.cfg.trn_tst_val)
        if self.p0_raw_reco is not None:
            self.p0_raw_reco = self._split_data(self.p0_raw_reco, self.cfg.trn_tst_val)
        if self.p0_raw_gen is not None:
            self.p0_raw_gen = self._split_data(self.p0_raw_gen, self.cfg.trn_tst_val)

    def _split_data(self, data, trn_tst_val):
        trn_size = int(data.shape[0] * trn_tst_val[0])
        tst_size = int(data.shape[0] * trn_tst_val[1])
        val_size = int(data.shape[0] * trn_tst_val[2])
        split = {
            "trn": data[:trn_size],
            "tst": data[trn_size : trn_size + tst_size],
            "val": data[-val_size:],
        }
        return split

    def compute_derived_features(self):
        self.derived_channels = {}
        feats = self.cfg.get("features", None)
        if feats is None:
            return
        registry = {
            "dphi_ll": lambda p: phys.delta_phi(p, p[..., 1], p[..., 5]),
            "deta_ll": lambda p: phys.delta_eta(p, p[..., 2], p[..., 6]),
            "ET_ll":   lambda p: phys.get_transverse_dilepton_energy(p[..., 0:4], p[..., 4:8]),
            "dphi_jj": lambda p: phys.delta_phi(p, p[..., 9], p[..., 13], abs=True),
            # signed, eta-ordered: phi(forward jet) - phi(backward jet) -- keeps the CP-odd sign
            "dphi_jj_signed_eta": lambda p: phys.delta_phi(
                p,
                torch.where(p[..., 10] >= p[..., 14], p[..., 9], p[..., 13]),
                torch.where(p[..., 10] >= p[..., 14], p[..., 13], p[..., 9]),
            ),
            "deta_jj": lambda p: phys.delta_eta(p, p[..., 10], p[..., 14]),
            "m_jj":    lambda p: phys.invariant_mass(p, [p[..., 8:12], p[..., 12:16]]),
            "mT":      lambda p: phys.decay_transverse_mass(p)[0],
        }
        enabled = [n for n in registry if feats.get(n, False)]
        base_channels = list(self.channels) if feats.get("basic", True) else []
        base = self.p0.shape[1]
        new_channels = []
        for i, name in enumerate(enabled):
            col = base + i
            self.p0 = torch.cat([self.p0, registry[name](self.p0).unsqueeze(1)], dim=1)
            self.p1 = torch.cat([self.p1, registry[name](self.p1).unsqueeze(1)], dim=1)
            new_channels.append(col)
            self.derived_channels[name] = col
            self.logger.info(f"    Derived feature '{name}' -> channel {col}")
        self.channels = base_channels + new_channels
        self.logger.info(f"    Channels after derived features: {self.channels}")

    def compute_weights(self):
        assert hasattr(
            self, "predictions"
        ), "Predictions not found in the dataset. Cannot compute weights."
        if isinstance(self.predictions, dict):
            self.weights = {}
            self.density_ratio = {}
            for split in self.predictions.keys():
                r = torch.exp(torch.clamp(self.predictions[split], max=50))
                W0 = self.w_p0[split].sum()
                W1 = self.w_p1[split].sum()
                self.density_ratio[split] = r
                self.weights[split] = r * (W1 / W0)
                print(f"  [{split}] W1/W0 = {(W1/W0).item():.4f}  |  "
                      f"raw r range [{r.min().item():.3f}, {r.max().item():.3f}]")
        else:
            r = torch.exp(torch.clamp(self.predictions, max=50))
            W0 = self.w_p0.sum()
            W1 = self.w_p1.sum()
            self.density_ratio = r
            self.weights = r * (W1 / W0)

    def setup_refiner_data(self):
        """Build refiner training tensors from p0 events and classifier weights."""
        assert hasattr(self, "weights"), "Classifier weights must be computed first."
        self.refiner_X = {}
        self.refiner_y = {}
        self.refiner_w = {}
        for split in self.weights:
            # import ipdb; ipdb.set_trace()
            mask_p0 = self.labels[split] == 0
            w_cls = self.weights[split]
            if w_cls.ndim == 2:
                w_cls = w_cls[:, mask_p0].mean(dim=0)
            else:
                w_cls = w_cls[mask_p0]
            omw = 1.0 - w_cls
            self.refiner_y[split] = (omw > 0).float().cpu()
            self.refiner_w[split] = torch.abs(omw).cpu()
            self.refiner_X[split] = self.p0_ppd[split].cpu()
            n_pos = (self.refiner_y[split] == 1).sum().item()
            n_pp = (self.refiner_y[split] == 0).sum().item()
            self.logger.info(f"    Refiner {split}: {n_pos} positive, {n_pp} pseudo-positive")

    def compute_refiner_weights(self):
        """Refinement factor = exp(-r(x)) for each p0 event."""
        assert hasattr(self, "refiner_predictions"), "Refiner predictions not found."
        self.refiner_weights = {}
        for split in self.refiner_predictions:
            print(f"    Computing refiner weights for {split}...")
            print(f"    Refiner predictions stats: min={self.refiner_predictions[split].min():.4f}, max={self.refiner_predictions[split].max():.4f}, mean={self.refiner_predictions[split].mean():.4f}")
            self.refiner_weights[split] = torch.tanh(self.refiner_predictions[split]/2)

    def _gaussianize(self, x, eps=1e-10):
        """Physics transform mirroring the unfolder's Gaussianize (src/datasets/WBF.py):
        log for pt/mass/energy, arctanh(x/pi) for phi, identity for eta/Delta, a special map
        for mT. Returns a transformed clone; only the columns the classifier actually uses
        (``self.channels``) are touched, and eta/Delta-eta/abs-Delta-phi columns (which can be
        <= 0) are deliberately left untransformed so ``log`` is never applied to them."""
        x = x.clone()
        chans = set(self.channels)
        dc = self.derived_channels
        floor = 1e-6
        # object layout: [0:4]=l1, [4:8]=l2, [8:12]=j1, [12:16]=j2, [16:20]=MET
        pt_cols = [c for c in (0, 4, 8, 12, 16) if c in chans]    # object pT + E_T,miss -> log
        phi_cols = [c for c in (1, 5, 9, 13, 17) if c in chans]   # object phi + phi_miss -> arctanh
        mass_cols = [c for c in (3, 7, 11, 15, 19) if c in chans]  # object masses -> log
        # derived features (positive magnitudes -> log, signed dphi -> arctanh; deta/abs-dphi_jj stay identity)
        for name in ("ET_ll", "m_jj"):
            if dc.get(name) in chans:
                pt_cols.append(dc[name])
        for name in ("dphi_ll", "dphi_jj_signed_eta"):  # signed angles in (-pi, pi] -> arctanh
            if dc.get(name) in chans:
                phi_cols.append(dc[name])
        for c in pt_cols + mass_cols:
            x[:, c] = torch.log(x[:, c].clamp(min=floor))
        for c in phi_cols:
            x[:, c] = phys.stable_arctanh(x[:, c] / torch.pi, eps)
        mt = dc.get("mT")
        if mt is not None and mt in chans:
            z = 100.0 / torch.log(x[:, mt].clamp(min=1.0 + floor))
            x[:, mt] = torch.log(z.clamp(min=floor))
        return x

    def apply_preprocessing(self, reverse=False, eps=1e-10):
        """
        Standardize [p0, p1] combined to zero mean / unit variance. When
        ``dataset.gaussianize`` is set, a physics transform (log-pt / arctanh-phi / ...) is
        applied first (see ``_gaussianize``). Raw ``self.p0``/``self.p1`` are left untouched
        (the raw feature plots read them); only ``self.p0_ppd``/``self.p1_ppd`` are transformed.
        """
        if not reverse:
            gaussianize = bool(self.cfg.get("gaussianize", False))
            p0 = self._gaussianize(self.p0, eps) if gaussianize else self.p0
            p1 = self._gaussianize(self.p1, eps) if gaussianize else self.p1
            if gaussianize:
                self.logger.info("    Physics preprocessing (gaussianize) applied before standardization")

            # Compute mean and std from combined (optionally gaussianized) p0 + p1
            combined = torch.cat([p0, p1], dim=0)
            self.ppd_mean = combined.mean(dim=0)
            self.ppd_std = combined.std(dim=0)

            # Standardize
            self.p0_ppd = (p0 - self.ppd_mean) / (self.ppd_std + eps)
            self.p1_ppd = (p1 - self.ppd_mean) / (self.ppd_std + eps)

            assert torch.isfinite(self.p0_ppd).all(), "p0_ppd has non-finite values"
            assert torch.isfinite(self.p1_ppd).all(), "p1_ppd has non-finite values"

            # self.logger.info("    Preprocessing applied: standardization on combined [p0, p1]")
            self.logger.info(f"    Using channels: {self.channels}")
        else:
            raise NotImplementedError(
                "Reverse preprocessing not yet implemented for classification."
            )

    def init_observables(
        self, pt_bins: int = 30, phi_bins: int = 20
    ) -> list[Observable]:
        self.observables = []
        self.observables.append(
            Observable(
                channel=0,
                compute=lambda p: return_obs(p[..., :], p[..., 0]),
                tex_label=r"p_{T, \ell_{1}}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0, upper=300
                ),
                xscale="linear",
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=1,
                compute=lambda p: return_obs(p[..., :], p[..., 1]),
                tex_label=r"\phi_{\ell_{1}}",
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=2,
                compute=lambda p: return_obs(p[..., :], p[..., 2]),
                tex_label=r"\eta_{\ell_{1}}",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )
        # self.observables.append(
        #     Observable(
        #         channel=None,
        #         compute=lambda p: return_obs(p[..., :], p[..., 3]),
        #         tex_label=r"m_{\ell_{1}}",
        #         unit="GeV",
        #         bins=lambda obs: get_hardcoded_bins(
        #             n_bins=pt_bins + 1, lower=-0.05, upper=0.12
        #         ),
        #         yscale="linear",
        #     )
        # )
        self.observables.append(
            Observable(
                channel=4,
                compute=lambda p: (return_obs(p[..., :], p[..., 4])),
                tex_label=r"p_{T, \ell_{2}}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0, upper=290
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=5,
                compute=lambda p: return_obs(p[..., :], p[..., 5]),
                tex_label=r"\phi_{\ell_{2}}",
                unit="rad",
                bins=lambda obs: (
                    get_hardcoded_bins(n_bins=phi_bins + 1, lower=-3.5, upper=3.5)
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=6,
                compute=lambda p: return_obs(p[..., :], p[..., 6]),
                tex_label=r"\eta_{\ell_{2}}",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )
        # self.observables.append(
        #     Observable(
        #         channel=None,
        #         compute=lambda p: return_obs(p[..., :], p[..., 7]),
        #         tex_label=r"m_{\ell_{2}}",
        #         unit="GeV",
        #         bins=lambda obs: get_hardcoded_bins(
        #             n_bins=pt_bins + 1, lower=-0.05, upper=0.12
        #         ),
        #         yscale="linear",
        #     )
        # )
        self.observables.append(
            Observable(
                channel=8,
                compute=lambda p: return_obs(p[..., :], p[..., 8]),
                tex_label=(r"p_{T, j_{1}}"),
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0, upper=500
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=9,
                compute=lambda p: return_obs(p[..., :], p[..., 9]),
                tex_label=(r"\phi_{j_{1}}"),
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=10,
                compute=lambda p: return_obs(p[..., :], p[..., 10]),
                tex_label=(r"\eta_{j_{1}}"),
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=11,
                compute=lambda p: return_obs(p[..., :], p[..., 11]),
                tex_label=r"m_{j_{1}}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1,
                    lower=0,
                    upper=60 if False else 30,
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=12,
                compute=lambda p: return_obs(p[..., :], p[..., 12]),
                tex_label=(r"p_{T, j_{2}}"),
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0, upper=500
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=13,
                compute=lambda p: return_obs(p[..., :], p[..., 13]),
                tex_label=(r"\phi_{j_{2}}"),
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=14,
                compute=lambda p: return_obs(p[..., :], p[..., 14]),
                tex_label=(r"\eta_{j_{2}}"),
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=15,
                compute=lambda p: return_obs(p[..., :], p[..., 15]),
                tex_label=r"m_{j_{2}}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1,
                    lower=0,
                    upper=60 if False else 30,
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=16,
                compute=lambda p: return_obs(p[..., :], p[..., 16]),
                tex_label=r"E_{T, \text{ miss}}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0, upper=300
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=17,
                compute=lambda p: return_obs(p[..., :], p[..., 17]),
                tex_label=r"\phi_{\text{miss}}",
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-3.5, upper=3.5
                ),
            )
        )

        self.observables.append(
            Observable(
                channel=20,
                compute=lambda p: return_obs(p[..., :], p[..., 20]),
                tex_label=r"\text{Jet multiplicity }N",
                bins=lambda obs: get_hardcoded_bins(n_bins=11, lower=0, upper=10),
                yscale="log",
            )
        )

        # TODO implement pT_ll for naive parameterization
        # if False:
        #     self.observables.append(
        #         Observable(
        #             channel=None
        #             if not False
        #             else 21,
        #             compute=lambda p: 0
        #             if not False
        #             else return_obs(p[..., :], p[..., 21]),
        #             tex_label=r"p_{T, \ell\ell}"
        #             if not False
        #             else r"p_{T, \ell\ell}^{*}",
        #             unit="GeV",
        #             bins=lambda obs: get_hardcoded_bins(
        #                 n_bins=pt_bins + 1, lower=0, upper=300
        #             ),
        #             yscale="linear",
        #         )
        #     )

        def _feat(name, fallback):
            ch = getattr(self, "derived_channels", {}).get(name)
            if ch is None:
                return None, fallback
            return ch, (lambda p, c=ch: return_obs(p, p[..., c]))

        ch, comp = _feat("dphi_ll", lambda p: phys.delta_phi(p[..., :], p[..., 1], p[..., 5]))
        self.observables.append(
            Observable(
                channel=ch,
                compute=comp,
                tex_label=r"\Delta\phi_{\ell\ell}",
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-2, upper=2
                ),
                yscale="linear",
            )
        )
        ch, comp = _feat("deta_ll", lambda p: phys.delta_eta(p[..., :], p[..., 2], p[..., 6]))
        self.observables.append(
            Observable(
                channel=ch,
                compute=comp,
                tex_label=r"\Delta\eta_{\ell\ell}",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=-2, upper=2
                ),
                yscale="linear",
            )
        )

        ch, comp = _feat("ET_ll", lambda p: phys.get_transverse_dilepton_energy(p[..., 0:4], p[..., 4:8]))
        self.observables.append(
            Observable(
                channel=ch,
                compute=comp,
                tex_label=r"E_{T, \ell\ell}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0, upper=300
                ),
                yscale="linear",
            )
        )
        ch, comp = _feat("dphi_jj", lambda p: phys.delta_phi(p[..., :], p[..., 9], p[..., 13], abs=True))
        self.observables.append(
            Observable(
                channel=ch,
                compute=comp,
                tex_label=(r"|\Delta\phi_{j_{1}j_{2}}|"),
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-0.1, upper=3.5
                ),
                yscale="linear",
            )
        )
        # |dphi_jj|, eta-ordered: phi(forward jet) - phi(backward jet)
        self.observables.append(
            Observable(
                compute=lambda p: phys.delta_phi(
                    p[..., :],
                    torch.where(p[..., 10] >= p[..., 14], p[..., 9], p[..., 13]),
                    torch.where(p[..., 10] >= p[..., 14], p[..., 13], p[..., 9]),
                    abs=True,
                ),
                key="dphi_jj_abs_eta",
                tex_label=r"|\Delta\phi_{j_{1}j_{2}}|",
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-0.1, upper=3.5
                ),
                yscale="linear",
            )
        )
        # signed dphi_jj, pT-ordered: phi(j1) - phi(j2); CP-odd sign is washed out
        self.observables.append(
            Observable(
                compute=lambda p: phys.delta_phi(
                    p[..., :], p[..., 9], p[..., 13], abs=False
                ),
                key="dphi_jj_signed_pt",
                tex_label=r"\Delta\phi_{j_{1}j_{2}}",
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )
        # signed dphi_jj, eta-ordered: phi(forward jet) - phi(backward jet); CP-odd sensitive
        self.observables.append(
            Observable(
                compute=lambda p: phys.delta_phi(
                    p[..., :],
                    torch.where(p[..., 10] >= p[..., 14], p[..., 9], p[..., 13]),
                    torch.where(p[..., 10] >= p[..., 14], p[..., 13], p[..., 9]),
                    abs=False,
                ),
                key="dphi_jj_signed_eta",
                tex_label=r"\Delta\phi_{j_{1}j_{2}}",
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )
        ch, comp = _feat("deta_jj", lambda p: phys.delta_eta(p[..., :], p[..., 10], p[..., 14]))
        self.observables.append(
            Observable(
                channel=ch,
                compute=comp,
                tex_label=(r"\Delta\eta_{j_{1}j_{2}}"),
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=1.0, upper=10.0
                ),
                yscale="linear",
            )
        )
        ch, comp = _feat("m_jj", lambda p: phys.invariant_mass(p[..., :], [p[..., 8:12], p[..., 12:16]]))
        self.observables.append(
            Observable(
                channel=ch,
                compute=comp,
                tex_label=(r"m_{j_{1}j_{2}}"),
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=300.0, upper=6000.0
                ),
                yscale="log",
            )
        )

        ch, comp = _feat("mT", lambda p: phys.decay_transverse_mass(p[..., :])[0])
        self.observables.append(
            Observable(
                channel=ch,
                compute=comp,
                tex_label=r"m_{T}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0.0, upper=250.0
                ),
                yscale="linear",
            )
        )

        self.observables.append(
            Observable(
                compute=lambda p: phys.approximate_pT_Higgs(
                    p[..., :], p[..., 8:12], p[..., 12:16]
                ),
                tex_label=r"p_{T, H}\approx |- \vec{p_{T,j_1}^{*}} - \vec{p_{T,j_2}^{*}}|"
                if False
                else r"p_{T, H}\approx |- \vec{p}_{T,q_1}^{*} - \vec{p}_{T,q_2}^{*}|",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0.0, upper=850.0
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                compute=lambda p: phys.approximate_phi_Higgs(
                    p[..., :], p[..., 8:12], p[..., 12:16]
                ),
                tex_label=r"\phi_{H}\approx \phi_{j_1,j_2} + \pi"
                if False
                else r"\phi_{H}\approx \phi_{q_1,q_2} + \pi",
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )

        # Higgs transverse momentum from its decay products: vec pT_l1 + vec pT_l2 + vec pT_miss
        def higgs_pxpy(p):
            pT = [p[..., 0], p[..., 4], p[..., 16]]
            phi = [p[..., 1], p[..., 5], p[..., 17]]
            px = sum(a * torch.cos(b) for a, b in zip(pT, phi))
            py = sum(a * torch.sin(b) for a, b in zip(pT, phi))
            return px, py

        # p_T,H = |vec pT_l1 + vec pT_l2 + vec pT_miss|
        self.observables.append(
            Observable(
                compute=lambda p: torch.hypot(*higgs_pxpy(p)),
                tex_label=r"p_{T, H}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0.0, upper=850.0
                ),
                yscale="linear",
            )
        )
        # phi_H = azimuth of vec pT_l1 + vec pT_l2 + vec pT_miss
        self.observables.append(
            Observable(
                compute=lambda p: torch.atan2(*reversed(higgs_pxpy(p))),
                tex_label=r"\phi_{H}",
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                compute=lambda p: return_obs(p[..., :], p[..., 8] + p[..., 12]),
                tex_label=r"H_{T,2j}" if False else r"H_{T,2q}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0.0, upper=850.0
                ),
                yscale="linear",
            )
        )


def Gaussianize(
    self,
    x: torch.Tensor,
    pt_channels: list,
    phi_channels: list,
    eta_channels: list,
    mass_channels: list,
    mT_channel: int = None,
    reverse=False,
    eps: float = 1e-10,
) -> torch.Tensor:
    # check there is no overlap between channels
    assert len(set(pt_channels) & set(phi_channels)) == 0
    assert len(set(pt_channels) & set(eta_channels)) == 0
    assert len(set(pt_channels) & set(mass_channels)) == 0
    assert len(set(phi_channels) & set(eta_channels)) == 0
    assert len(set(phi_channels) & set(mass_channels)) == 0
    assert len(set(eta_channels) & set(mass_channels)) == 0

    for ch in pt_channels:
        x[:, ch] = gaussianize_pt(x[:, ch], eps, reverse)
    for ch in phi_channels:
        x[:, ch] = gaussianize_phi(x[:, ch], eps, reverse)
    for ch in eta_channels:
        x[:, ch] = gaussianize_eta(x[:, ch], eps, reverse)
    for ch in mass_channels:
        x[:, ch] = gaussianize_mass(x[:, ch], eps, reverse)
    if mT_channel is not None:
        if not reverse:
            z = 100 / torch.log(x[:, mT_channel])
            x[:, mT_channel] = gaussianize_pt(z, eps, reverse)
        else:
            z = gaussianize_pt(x[:, mT_channel], eps, reverse)
            x[:, mT_channel] = torch.exp((100 / z).clip(max=10))
    return x


def gaussianize_pt(
    x: torch.Tensor, epsilon: float, reverse: bool = False
) -> torch.Tensor:
    if not reverse:
        return torch.log(x)
    else:
        return torch.exp(x.clip(max=10))


def gaussianize_phi(
    x: torch.Tensor, epsilon: float, reverse: bool = False
) -> torch.Tensor:
    if not reverse:
        return phys.stable_arctanh(x / torch.pi, epsilon)
    else:
        return torch.tanh(x) * torch.pi


def gaussianize_eta(
    x: torch.Tensor, epsilon: float, reverse: bool = False
) -> torch.Tensor:
    if not reverse:
        return x
    else:
        return x


def gaussianize_mass(
    x: torch.Tensor, epsilon: float, reverse: bool = False
) -> torch.Tensor:
    if not reverse:
        return torch.log(x)
    else:
        return torch.exp(x.clip(max=10))
