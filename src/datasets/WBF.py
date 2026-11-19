# type: ignore
import os
import torch
import numpy as np
from utils import physics as phys
from .dataset import Observable, return_obs, get_hardcoded_bins, WBF_bins
from scipy.stats import johnsonsb, norm


class WBF:
    def __init__(self, logger, cfg, **kwargs):
        self.logger = logger
        self.cfg = cfg
        self.channels = self.cfg.get("channels", None)
        self.parameterization = self.cfg.get("parameterization", None)
        self.unfold_to = self.cfg.get("unfold_to", "parton")
        # the valid parameterization (default) and the jet relabelling are parton-level only: switched off
        # when unfolding to particle level (no exact p_T balance there)
        tm = self.parameterization.get("transverse_mass", None) if self.parameterization is not None else None
        self.parton_only_off = self.unfold_to != "parton"
        if self.parton_only_off and tm is not None and tm.get("valid", False):
            logger.info("    Particle-level unfolding: the parton-level valid parameterization is not used")
            tm.valid = False
        self.process = self.cfg.get("process", None)
        self.initiate_prior_only = (
            kwargs.get(f"initiate_prior_only", False)
            if kwargs.get(f"initiate_prior_only", False)
            else False
        )

        if self.parameterization.naive.use:
            self.channels = self.parameterization.naive.channels
        elif self.parameterization.transverse_mass.use:
            self.channels = self.parameterization.transverse_mass.channels

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
        data_path = self.cfg.get("data_path", None)
        if self.device == "cpu" and "remote" in data_path:
            self.logger.info(
                f"    Data path {data_path} is remote but you are running on local cpu, changing it to data/"
            )
            data_path = "data"
        if data_path is None:
            raise ValueError("Data path not specified in the parameters")

        if not os.path.exists(data_path):
            raise ValueError(f"Path {data_path} does not exist")
        if not self.initiate_prior_only:
            if not self.cfg.combined.use:
                if self.unfold_to == "particle":
                    path_part = os.path.join(data_path, self.cfg.path_gen)
                    path_reco = os.path.join(data_path, self.cfg.path_reco_corresponding_to_gen)
                elif self.unfold_to == "parton":
                    path_part = os.path.join(data_path, self.cfg.path_parton)
                    path_reco = os.path.join(data_path, self.cfg.path_reco_corresponding_to_parton)
                else:
                    raise ValueError("Unfolding to neither particle nor parton")
                try:
                    part = (
                        np.load(path_part)[:n_data]
                        if n_data is not None
                        else np.load(path_part)
                    )
                    reco = (
                        np.load(path_reco)[:n_data]
                        if n_data is not None
                        else np.load(path_reco)
                    )
                    self.logger.info(
                        f"    Data loaded for {path_part} and {path_reco} for process {self.process}"
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error loading file {path_part} or {path_reco}: {e}"
                    )
            else:

                # Combined training over an arbitrary list of processes (cfg.combined.processes;
                # default = signal + the two EFT variants). Each must have a full entry in the
                # dataset config (data_path + path_gen/path_parton + path_reco_corresponding_to_*).
                processes = list(
                    self.cfg.combined.get(
                        "processes", ["signal", "EFT_chw0p5", "EFT_chwtil5"]
                    )
                )
                holdout = float(self.cfg.get("holdout_frac", 0.0))
                try:
                    part = []
                    reco = []
                    dataset_ids = [] if self.cfg.combined.dataset_ids else None
                    for dataset_id, name in enumerate(processes):
                        pcfg = self.cfg[name]
                        if self.unfold_to == "particle":
                            p = os.path.join(pcfg.data_path, pcfg.path_gen)
                            r = os.path.join(pcfg.data_path, pcfg.path_reco_corresponding_to_gen)
                        elif self.unfold_to == "parton":
                            p = os.path.join(pcfg.data_path, pcfg.path_parton)
                            r = os.path.join(pcfg.data_path, pcfg.path_reco_corresponding_to_parton)
                        else:
                            raise ValueError("Unfolding to neither particle nor parton")
                        p_arr = np.load(p)[:n_data] if n_data is not None else np.load(p)
                        r_arr = np.load(r)[:n_data] if n_data is not None else np.load(r)
                        # Hold out the APP pool (last holdout_frac of each file, file order) so the
                        # unfolder never trains on the events that make up p_data. Matches the
                        # classifier's file-order holdout, keeping the closure leak-free.
                        if holdout > 0.0:
                            n_trn = p_arr.shape[0] - int(p_arr.shape[0] * holdout)
                            p_arr = p_arr[:n_trn]
                            r_arr = r_arr[:n_trn]
                        part.append(p_arr)
                        reco.append(r_arr)
                        self.logger.info(
                            f"    [combined] {name}: {p_arr.shape[0]} train-pool events "
                            f"(holdout_frac={holdout}) from {p} / {r}."
                        )
                        if dataset_ids is not None:
                            dataset_ids.append(np.full((p_arr.shape[0], 1), dataset_id))

                    part = np.concatenate(part, axis=0)
                    reco = np.concatenate(reco, axis=0)
                    if dataset_ids is not None:
                        dataset_ids = np.concatenate(dataset_ids, axis=0)
                    perm = np.random.permutation(part.shape[0])
                    part = part[perm]
                    reco = reco[perm]
                    if dataset_ids is not None:
                        dataset_ids = dataset_ids[perm]

                except Exception as e:
                    raise ValueError(f"Error loading combined processes {processes}: {e}")

        elif self.initiate_prior_only == "SM":
            path_reco = os.path.join(
                self.cfg.signal.data_path, self.cfg.signal.path_reco_corresponding_to_gen if self.unfold_to == "particle" else self.cfg.signal.path_reco_corresponding_to_parton
            )
            path_part = (
                os.path.join(self.cfg.signal.data_path, self.cfg.signal.path_gen)
                if self.unfold_to == "particle"
                else os.path.join(
                    self.cfg.signal.data_path, self.cfg.signal.path_parton
                )
            )
            try:
                part = (
                    np.load(path_part)[:n_data]
                    if n_data is not None
                    else np.load(path_part)
                )
                reco = (
                    np.load(path_reco)[:n_data]
                    if n_data is not None
                    else np.load(path_reco)
                )
                self.logger.info(
                    f"    Loaded data from {path_part} and {path_reco} for SM"
                )
            except Exception as e:
                raise ValueError(f"Error loading file {path_part} or {path_reco}: {e}")
        elif self.initiate_prior_only == "EFTchw0p5":
            path_reco = os.path.join(
                self.cfg.EFT_chw0p5.data_path, self.cfg.EFT_chw0p5.path_reco_corresponding_to_gen if self.unfold_to == "particle" else self.cfg.EFT_chw0p5.path_reco_corresponding_to_parton
            )
            path_part = (
                os.path.join(
                    self.cfg.EFT_chw0p5.data_path, self.cfg.EFT_chw0p5.path_gen
                )
                if self.unfold_to == "particle"
                else os.path.join(
                    self.cfg.EFT_chw0p5.data_path,
                    self.cfg.EFT_chw0p5.path_parton,
                )
            )
            try:
                part = (
                    np.load(path_part)[:n_data]
                    if n_data is not None
                    else np.load(path_part)
                )
                reco = (
                    np.load(path_reco)[:n_data]
                    if n_data is not None
                    else np.load(path_reco)
                )
                self.logger.info(
                    f"    Loaded data from {path_part} and {path_reco} for EFTchw0p5"
                )
            except Exception as e:
                raise ValueError(f"Error loading file {path_part} or {path_reco}: {e}")
        elif self.initiate_prior_only == "EFTchwtil5":
            path_reco = os.path.join(
                self.cfg.EFT_chwtil5.data_path, self.cfg.EFT_chwtil5.path_reco_corresponding_to_gen if self.unfold_to == "particle" else self.cfg.EFT_chwtil5.path_reco_corresponding_to_parton
            )
            path_part = (
                os.path.join(
                    self.cfg.EFT_chwtil5.data_path,
                    self.cfg.EFT_chwtil5.path_gen,
                )
                if self.unfold_to == "particle"
                else os.path.join(
                    self.cfg.EFT_chwtil5.data_path,
                    self.cfg.EFT_chwtil5.path_parton,
                )
            )
            try:
                part = (
                    np.load(path_part)[:n_data]
                    if n_data is not None
                    else np.load(path_part)
                )
                reco = (
                    np.load(path_reco)[:n_data]
                    if n_data is not None
                    else np.load(path_reco)
                )
                self.logger.info(
                    f"    Loaded data from {path_part} and {path_reco} for EFTchwtil5"
                )
            except Exception as e:
                raise ValueError(f"Error loading file {path_part} or {path_reco}: {e}")

        part = torch.tensor(part, device=self.device, dtype=torch.float32)
        reco = torch.tensor(reco, device=self.device, dtype=torch.float32)
        if self.cfg.combined.use and self.cfg.combined.dataset_ids:
            dataset_ids = torch.tensor(
                dataset_ids, device=self.device, dtype=torch.int32
            )
            self.dataset_ids = dataset_ids
        self.logger.info(
            f"    Trn: {self.cfg.trn_tst_val[0] * len(part):.0f}, Tst: {self.cfg.trn_tst_val[1] * len(part):.0f}, Val: {self.cfg.trn_tst_val[2] * len(part):.0f}"
        )
        self.n_components = part.shape[1] // 4
        if self.unfold_to == "particle":
            part_PtPhiEtaM = torch.cat(
                [
                    phys.EPxPyPz_to_PtPhiEtaM(part[:, 0:4]),
                    phys.EPxPyPz_to_PtPhiEtaM(part[:, 4:8]),
                    phys.EPxPyPz_to_PtPhiEtaM(part[:, 8:12]),
                    phys.EPxPyPz_to_PtPhiEtaM(part[:, 12:16]),
                    phys.EPxPyPz_to_PtPhiEtaM(part[:, 16:20]),
                    part[:, 20:21],
                ],
                dim=1,
            )
        elif self.unfold_to == "parton":
            part_PtPhiEtaM = torch.cat(
                [
                    phys.EPxPyPz_to_PtPhiEtaM(part[:, 0:4]),
                    phys.EPxPyPz_to_PtPhiEtaM(part[:, 4:8]),
                    phys.EPxPyPz_to_PtPhiEtaM(part[:, 8:12]),
                    phys.EPxPyPz_to_PtPhiEtaM(part[:, 12:16]),
                    phys.EPxPyPz_to_PtPhiEtaM(part[:, 16:20]),
                    torch.ones((part.shape[0], 1), device=self.device) * 2.0,
                ],
                dim=1,
            )
        else:
            raise ValueError("Unfolding to neither particle nor parton")

        reco_PtPhiEtaM = torch.cat(
            [
                phys.EPxPyPz_to_PtPhiEtaM(reco[:, 0:4]),
                phys.EPxPyPz_to_PtPhiEtaM(reco[:, 4:8]),
                phys.EPxPyPz_to_PtPhiEtaM(reco[:, 8:12]),
                phys.EPxPyPz_to_PtPhiEtaM(reco[:, 12:16]),
                phys.EPxPyPz_to_PtPhiEtaM(reco[:, 16:20]),
                reco[:, 20:21],
            ],
            dim=1,
        )
        if self.cfg.get("match_parton_jets", False) and self.unfold_to == "parton":
            # truth jet labels follow the p_T-ordered reco jets: swap q1 <-> q2 where that lowers
            # dR(q1, j1) + dR(q2, j2). Same four-vectors, only their order (parton files store the
            # quarks in random order w.r.t. reco, which gives every jet target a 50/50 two-mode posterior).
            def _dr(a, b):
                return torch.sqrt(phys.wrap_angle(part_PtPhiEtaM[:, a + 1] - reco_PtPhiEtaM[:, b + 1]) ** 2
                                  + (part_PtPhiEtaM[:, a + 2] - reco_PtPhiEtaM[:, b + 2]) ** 2)
            swap = _dr(8, 8) + _dr(12, 12) > _dr(8, 12) + _dr(12, 8)
            q1, q2 = part_PtPhiEtaM[:, 8:12].clone(), part_PtPhiEtaM[:, 12:16].clone()
            part_PtPhiEtaM[:, 8:12] = torch.where(swap[:, None], q2, q1)
            part_PtPhiEtaM[:, 12:16] = torch.where(swap[:, None], q1, q2)
            self.logger.info(f"    Truth jets matched to the reco jet order: swapped {100 * swap.float().mean():.1f}% of events")
        if self.parameterization.naive.use:
            if self.cfg.special.unfold_jets_only:
                self.channels = [
                    i
                    for i in self.channels
                    if i not in [0, 1, 2, 3, 4, 5, 6, 7, 16, 17]
                ]
            elif self.cfg.special.unfold_higgs_system_only:
                self.channels = [
                    i for i in self.channels if i not in [8, 9, 10, 11, 12, 13, 14, 15]
                ]
            else:
                pass
        elif self.parameterization.transverse_mass.use:
            PtPhiEtaM = {
                "part": part_PtPhiEtaM,
                "reco": reco_PtPhiEtaM,
            }
            extra_feats = {}
            updated = {}
            for level, data in PtPhiEtaM.items():
                ET_miss, pT_ll, dphi_ll_miss, mT = phys.higgs_system_pieces(data)
                # computing signed deltas
                dphi_ll = phys.delta_phi(data, data[..., 1], data[..., 5], abs=False)
                deta_ll = phys.delta_eta(data, data[..., 2], data[..., 6], abs=False)

                # extra features to append for this parameterization
                extra = {
                    "ET_miss": ET_miss.unsqueeze(-1),
                    "pT_ll": pT_ll.unsqueeze(-1),
                    "dphi_ll-miss": dphi_ll_miss.unsqueeze(-1),
                    "dphi_ll": dphi_ll.unsqueeze(-1),
                    "deta_ll": deta_ll.unsqueeze(-1),
                    "mT": mT.unsqueeze(-1),
                }
                extra_feats[level] = extra
                updated[level] = torch.cat(
                    [
                        data,
                        extra["pT_ll"],
                        extra["dphi_ll"],
                        extra["deta_ll"],
                        extra["mT"],
                        extra["dphi_ll-miss"],
                    ],
                    dim=1,
                )
            if self.parameterization.transverse_mass.get("valid", False):
                # valid-by-construction parameterization: append r = m_ll/m_T (26), signed
                # dphi_jj = phi_j1 - phi_j2 (27) and z = pT_j2/pT_j1 (28), file jet order
                for level, data in PtPhiEtaM.items():
                    _, m_ll = phys.decay_transverse_mass(data)
                    r = m_ll / updated[level][:, 24]
                    dphi_jj = phys.delta_phi(data, data[..., 9], data[..., 13], abs=False)
                    z = data[..., 12] / data[..., 8]
                    # (29) phi_j1 - phi_H, p_T,H = p_T,l1 + p_T,l2 + p_T,miss (valid_jets=direct)
                    Hx = sum(data[..., c] * torch.cos(data[..., c + 1]) for c in (0, 4, 16))
                    Hy = sum(data[..., c] * torch.sin(data[..., c + 1]) for c in (0, 4, 16))
                    dphi_j1H = phys.wrap_angle(data[..., 9] - torch.atan2(Hy, Hx))
                    # (30, 31) symmetric jets: |k| and psi = phi_k - phi_H, k = (p_j1 - p_j2)/2 (valid_jets=symmetric)
                    kx = (data[..., 8] * torch.cos(data[..., 9]) - data[..., 12] * torch.cos(data[..., 13])) / 2
                    ky = (data[..., 8] * torch.sin(data[..., 9]) - data[..., 12] * torch.sin(data[..., 13])) / 2
                    k_abs = torch.sqrt(kx**2 + ky**2)
                    psi = phys.wrap_angle(torch.atan2(ky, kx) - torch.atan2(Hy, Hx))
                    # (32) dphi_jj and (33) H_T = pT_j1 + pT_j2, (34) dphi_jj: raw values of the dphi_pt2 /
                    # ht_dphi jet coordinates (mapped per event in apply_preprocessing)
                    HT = data[..., 8] + data[..., 12]
                    updated[level] = torch.cat([updated[level], r[:, None], dphi_jj[:, None], z[:, None],
                                                dphi_j1H[:, None], k_abs[:, None], psi[:, None],
                                                dphi_jj[:, None], HT[:, None], dphi_jj[:, None]], dim=1)
            part_PtPhiEtaM = updated["part"]
            reco_PtPhiEtaM = updated["reco"]
            self.channels = [
                0,
                1,
                2,  # l1
                8,
                9,
                10,
                11,  # q1/j1
                12,
                13,
                14,
                15,  # q2/j2
                21,
                22,
                23,  # ll system
                24,
                25,  # mT and dphi_ll-miss
            ]
            # self.channels = [24]
            if self.cfg.special.unfold_jets_only:
                self.channels = [
                    i for i in self.channels if i not in [0, 1, 2, 21, 22, 23, 24, 25]
                ]
            elif self.cfg.special.unfold_higgs_system_only:
                self.channels = [
                    i for i in self.channels if i not in [8, 9, 10, 11, 12, 13, 14, 15]
                ]
            else:
                pass
        else:
            raise ValueError("Parameterization not specified in the parameters")

        # condition (reco) channels; differ from the unfolded ones only for the balanced parameterization
        self.cond_channels = list(self.channels)
        if self.parameterization.transverse_mass.use and self.parameterization.transverse_mass.get("balanced", False):
            # parton level balances p_T exactly: p_ll + p_miss = -(p_j1 + p_j2). pT_ll (21) and
            # dphi_ll-miss (25) are not unfolded but solved from the other features and m_T
            # (phys.complete_balanced, after reverse preprocessing); the reco condition keeps them.
            assert self.unfold_to == "parton", "balanced parameterization needs parton-level truth"
            self.channels = [i for i in self.channels if i not in (21, 25)]
        if self.parameterization.transverse_mass.use and self.parameterization.transverse_mass.get("valid", False):
            # every generated point is a physical event with exact parton p_T balance and m_T as generated:
            # unfold l1, dphi_ll (mapped into the fiducial |dphi_ll| < 1.4), deta_ll, m_T, dphi_ll-miss,
            # r = m_ll/m_T in (0,1), signed dphi_jj, z = pT_j2/pT_j1 > 0, eta_j1, eta_j2; pT_l2, MET and the
            # jet transverse momenta follow (phys.complete_valid). The reco condition keeps the 14 features.
            assert self.unfold_to == "parton", "valid parameterization needs parton-level truth"
            assert not self.parameterization.transverse_mass.get("balanced", False)
            if self.parameterization.transverse_mass.get("valid_jets", "ratio") == "dphi_pt2":
                # jets: unfold pT_q2 (12) and dphi_jj on its per-event range (32); q1 = -(p_H + p_q2)
                self.channels = [0, 1, 2, 10, 12, 14, 22, 23, 24, 25, 26, 32]
            elif self.parameterization.transverse_mass.get("valid_jets", "ratio") in ("ht_dphi", "ht_nu"):
                # jets: unfold H_T (33) and dphi_jj on its loop (34; ht_nu: as the elliptic angle nu)
                self.channels = [0, 1, 2, 10, 14, 22, 23, 24, 25, 26, 33, 34]
            elif self.parameterization.transverse_mass.get("valid_jets", "ratio") == "symmetric":
                # jets: unfold the half difference k = (p_j1 - p_j2)/2 (30, 31); p_j1,2 = -p_H/2 +- k
                self.channels = [0, 1, 2, 10, 14, 22, 23, 24, 25, 26, 30, 31]
            elif self.parameterization.transverse_mass.get("valid_jets", "ratio") == "direct":
                # jets: unfold pT_j1 (8) and phi_j1 - phi_H (29); j2 = -(p_H + p_j1), no division
                self.channels = [0, 1, 2, 8, 10, 14, 22, 23, 24, 25, 26, 29]
            else:
                self.channels = [0, 1, 2, 10, 14, 22, 23, 24, 25, 26, 27, 28]
            if self.parameterization.transverse_mass.get("valid_leptons", "ratio") == "direct":
                # leptons: unfold pT_l2 (4) instead of r = m_ll/m_T (26); m_ll has no division
                self.channels = sorted(4 if c == 26 else c for c in self.channels)
        if self.unfold_to == "parton":
            self.channels = [
                i for i in self.channels if i not in [3, 7, 11, 15]
            ]  # eliminate quark and lepton masses (fixed)
            # the reco jet masses are measured, independent information: keep them in the condition
            # (cond_jet_masses; only the parton-level masses are zero). Lepton masses are zero at both levels.
            drop = [3, 7] if self.cfg.get("cond_jet_masses", False) else [3, 7, 11, 15]
            self.cond_channels = sorted(i for i in self.cond_channels if i not in drop)
        # physical channels of the unfolded events (plots): the unfolded ones plus those the completion
        # fills; never masses that were not unfolded
        self.plot_channels = sorted(set(self.channels) | {i for i in self.cond_channels if i not in (3, 7, 11, 15)})
        self.part = part_PtPhiEtaM
        self.reco = reco_PtPhiEtaM

    def split_data(self):
        # separate preprocessed and raw data into training/test/validation splits
        self.part_ppd, self.reco_ppd = self._split_data(
            self.part_ppd, self.reco_ppd, self.cfg.trn_tst_val
        )
        self.part, self.reco = self._split_data(
            self.part, self.reco, self.cfg.trn_tst_val
        )
        if self.cfg.combined.use and self.cfg.combined.dataset_ids:
            self.dataset_ids, _ = self._split_data(
                self.dataset_ids, self.dataset_ids, self.cfg.trn_tst_val
            )

    def _split_data(self, part, reco, trn_tst_val):
        # split the data
        for i, split in enumerate(["trn", "tst", "val"]):
            globals()[f"{split}_slice"] = int(part.shape[0] * trn_tst_val[i])
        part_split = {
            "trn": part[:trn_slice],
            "tst": part[trn_slice : trn_slice + tst_slice],
            "val": part[-val_slice:],
        }
        reco_split = {
            "trn": reco[:trn_slice],
            "tst": reco[trn_slice : trn_slice + tst_slice],
            "val": reco[-val_slice:],
        }
        return part_split, reco_split

    def apply_preprocessing(self, reverse=False, reco_mean=None, reco_std=None, eps=1e-10,
                            part_mean=None, part_std=None, reco_ref=None):
        """Apply (or reverse) physics-specific Gaussianization + standardization.

        Args:
            reverse: if True, invert the preprocessing on ``self.unfolded_ppd``.
            reco_mean: optional external mean tensor (full-width, not channel-sliced).
                When provided in the forward pass, this mean is used for standardization
                instead of being refit from the current data — use this for p_data so it
                is standardized identically to the unfolder's training signal.
            reco_std: optional external std tensor (same convention as reco_mean).
            eps: small value added to std to avoid division by zero.
        """
        # create feature base channels (condition channels: a superset of the unfolded ones)
        prep_channels = sorted(set(getattr(self, "cond_channels", self.channels)) | set(self.channels))
        pt_channels = [i for i in prep_channels if i % 4 == 0 and i < 20]
        phi_channels = [i for i in prep_channels if i % 4 == 1 and i < 20]
        eta_channels = [i for i in prep_channels if i % 4 == 2 and i < 20]
        mass_channels = [i for i in prep_channels if i % 4 == 3 and i < 20]

        if self.parameterization.transverse_mass.use:
            if 21 in prep_channels:
                pt_channels.append(
                    21
                )  # appending pT_ll to apply pt_channels-type preprocessing
            if 25 in prep_channels:
                phi_channels.append(
                    25
                )  # appending dphi_ll-miss to apply phi_channels-type preprocessing

        else:
            pass
        logit_channels, halfphi_channels = [], []
        tm = self.parameterization.transverse_mass
        j2_pt_cut = tm.get("j2_pt_cut", None) if tm.use and tm.get("valid_jets", "ratio") == "direct" else None
        symmetric_jets = bool(tm.use and tm.get("valid", False) and tm.get("valid_jets", "ratio") == "symmetric")
        ht_dphi_jets = bool(tm.use and tm.get("valid", False) and tm.get("valid_jets", "ratio") in ("ht_dphi", "ht_nu"))
        jet_pt_cut = tm.get("jet_pt_cut", None) if symmetric_jets or ht_dphi_jets else None
        # jj_cuts: eta_j1/eta_j2 (10, 14) unfolded as per-event bounded (deta_jj, centre) coordinates, so that
        # every unfolded event passes m_jj > 450, |deta_jj| > 2.1, |eta_q| < 4.5 (phys.jj_eta_features)
        jj_cuts = bool(ht_dphi_jets and tm.get("jj_cuts", False))
        # truth-level p_T thresholds built into the transform: log(p_T - cut) <-> cut + exp(.), so every
        # unfolded value passes the cut (truth and unfolded only; the reco condition keeps log p_T)
        pt_cuts = {int(k): float(v) for k, v in (self.parameterization.transverse_mass.get("pt_cuts", None) or {}).items()}
        if not (tm.use and tm.get("valid", False)):
            pt_cuts = {}  # parton-level thresholds: only with the (parton-level) valid parameterization
        if self.parameterization.transverse_mass.use and self.parameterization.transverse_mass.get("valid", False):
            pt_channels.append(28)  # z = pT_j2/pT_j1 > 0
            phi_channels += [27, 29]  # signed dphi_jj, phi_j1 - phi_H
            if j2_pt_cut is not None:
                phi_channels.remove(29)  # bounded per event instead (truth: below; unfolded: complete_valid)
            logit_channels = [26]  # r = m_ll/m_T in (0, 1)
            halfphi_channels = [22]  # dphi_ll in the fiducial (-1.4, 1.4)
        # valid_leptons=direct: p_T,l1 (0), p_T,l2 (4) and m_T (24) get their own maps (_truth_forward,
        # phys.complete_valid); r (26) is not unfolded. reco_relative: unfolded truth relative to the reco
        # value of the same event (the condition), in each feature's own coordinate.
        lep_direct = bool(tm.use and tm.get("valid", False) and tm.get("valid_leptons", "ratio") == "direct")
        reco_relative = bool(tm.use and tm.get("valid", False) and tm.get("reco_relative", False))
        assert not (reco_relative and symmetric_jets), "reco_relative is not implemented for symmetric jets"
        t_pt = [c for c in pt_channels if not (lep_direct and c in (0, 4))]
        t_phi = list(phi_channels)
        # lep_mT_coord (direct leptons): offset = log(m_T - m_ll); standard = the usual m_T map, with
        # m_T >= m_ll enforced in the inverse
        lep_mT_offset = lep_direct and tm.get("lep_mT_coord", "offset") == "offset"
        t_mT = None if lep_mT_offset else (24 if tm.use else None)
        # periodic unfolded angles: residual as the wrapped difference, the other channels additively
        periodic = [c for c in self.channels if c in t_phi] if reco_relative else []
        # additive residuals; not the per-event-bounded phi_j1 - phi_H (29): reco jets do not balance, so
        # the reco angle can lie outside the bound and has no finite reference
        additive = [c for c in self.channels if c not in periodic and c not in (29, 32, 33, 34)
                    and not (jj_cuts and c in (10, 14))] if reco_relative else []

        def _truth_forward(P, angle_residual=None):
            """Truth-side forward map of a raw (N, n_feat) tensor (truth, or reco as the reference).
            angle_residual: values replacing the periodic channels before the angle map (wrapped
            truth - reco differences); every derived quantity is computed from P itself."""
            X = P.clone().to(self.device)
            if angle_residual is not None:
                X[:, periodic] = angle_residual.to(X.dtype).to(X.device)
            if j2_pt_cut is not None:
                # theta = phi_j1 - phi_H -> artanh(theta/theta_max): every unfolded theta gives pT_j2 > cut
                Hx = sum(P[:, c] * torch.cos(P[:, c + 1]) for c in (0, 4, 16))
                Hy = sum(P[:, c] * torch.sin(P[:, c + 1]) for c in (0, 4, 16))
                tmax = phys.j2_theta_max(torch.sqrt(Hx**2 + Hy**2).double(), P[:, 8].double(), float(j2_pt_cut))
                X[:, 29] = phys.stable_arctanh((P[:, 29].double() / tmax).clamp(-1, 1), eps).to(X.dtype).to(X.device)
            X = Gaussianize(self, X, pt_cuts=pt_cuts, pt_channels=t_pt, phi_channels=t_phi,
                            eta_channels=eta_channels, mass_channels=mass_channels, mT_channel=t_mT,
                            logit_channels=logit_channels, halfphi_channels=halfphi_channels, reverse=False)
            vj = tm.get("valid_jets", "ratio")
            if vj in ("dphi_pt2", "ht_dphi", "ht_nu"):
                Hx = sum(P[:, c] * torch.cos(P[:, c + 1]) for c in (0, 4, 16)).double()
                Hy = sum(P[:, c] * torch.sin(P[:, c + 1]) for c in (0, 4, 16)).double()
                pH, dphi = torch.sqrt(Hx**2 + Hy**2), phys.wrap_angle(P[:, 9] - P[:, 13]).double()
                if vj == "dphi_pt2":
                    X[:, 32] = phys.dphi_pt2_features(pH, P[:, 12].double(), P[:, 8].double(), dphi,
                                                      phys.DPHI_PT2_CUT, eps).to(X.dtype).to(X.device)
                else:
                    cut = None if jet_pt_cut is None else float(jet_pt_cut)
                    if vj == "ht_nu":  # quark-pair pT_H: exact geometry; the inverse uses leptons + MET
                        yS, ychi = phys.ht_nu_features(P[:, 8].double(), P[:, 12].double(), dphi, eps, cut)
                    else:
                        yS, ychi = phys.ht_dphi_features(pH, P[:, 8].double(), P[:, 12].double(), dphi, eps, cut)
                    X[:, 33], X[:, 34] = yS.to(X.dtype).to(X.device), ychi.to(X.dtype).to(X.device)
                if jj_cuts:
                    yd, yc = phys.jj_eta_features(P[:, 8].double(), P[:, 12].double(), dphi,
                                                  P[:, 10].double(), P[:, 14].double(), eps)
                    X[:, 10], X[:, 14] = yd.to(X.dtype).to(X.device), yc.to(X.dtype).to(X.device)
            if lep_direct:
                y0, y4, y24 = phys.direct_lepton_features(P)
                X[:, 0], X[:, 4] = (v.to(X.dtype).to(X.device) for v in (y0, y4))
                if lep_mT_offset:
                    X[:, 24] = y24.to(X.dtype).to(X.device)
            return X

        if not hasattr(self, "reco_mean") and reverse:
            raise ValueError(
                "Cannot reverse preprocess without initializing dataset first. Check init_dataset() attr."
            )
        else:
            pass
        self.logger.info(f"    Using channels: {self.channels}")
        self.logger.info(
            f"    Physics-specific preprocessing for channels (forward={not reverse}): pt {pt_channels}, phi {phi_channels}, eta {eta_channels}, mass {mass_channels}"
        )
        if not reverse:
            part_ppd = self.part.clone().to(self.device)
            reco_ppd = self.reco.clone().to(self.device)
            reco_ppd = Gaussianize(
                self,
                reco_ppd,
                pt_channels=pt_channels,
                phi_channels=phi_channels,
                eta_channels=eta_channels,
                mass_channels=mass_channels,
                mT_channel=24 if self.parameterization.transverse_mass.use else None,
                logit_channels=logit_channels,
                halfphi_channels=halfphi_channels,
                reverse=False,
            )
            if symmetric_jets:
                # (30, 31) -> log(k - k_min), artanh(t): every unfolded value gives both jets > jet_pt_cut
                P = self.part
                Hx = sum(P[:, c] * torch.cos(P[:, c + 1]) for c in (0, 4, 16)).double()
                Hy = sum(P[:, c] * torch.sin(P[:, c + 1]) for c in (0, 4, 16)).double()
                kx = (P[:, 8] * torch.cos(P[:, 9]) - P[:, 12] * torch.cos(P[:, 13])).double() / 2
                ky = (P[:, 8] * torch.sin(P[:, 9]) - P[:, 12] * torch.sin(P[:, 13])).double() / 2
                y, u = phys.symmetric_jet_features(torch.sqrt(Hx**2 + Hy**2), torch.atan2(Hy, Hx), kx, ky,
                                                   None if jet_pt_cut is None else float(jet_pt_cut), eps)
                part_ppd[:, 30] = y.to(part_ppd.dtype).to(part_ppd.device)
                part_ppd[:, 31] = u.to(part_ppd.dtype).to(part_ppd.device)
            if j2_pt_cut is not None or lep_direct or reco_relative or jj_cuts:
                sym30 = part_ppd[:, 30:32].clone() if symmetric_jets else None
                # periodic channels: wrapped difference to the reco angle, then the usual angle map
                part_ppd = _truth_forward(self.part, phys.wrap_angle(self.part[:, periodic] - self.reco[:, periodic])
                                          if periodic else None)
                if reco_relative:
                    part_ppd[:, additive] -= _truth_forward(self.reco)[:, additive]
                if sym30 is not None:
                    part_ppd[:, 30:32] = sym30
            else:
                part_ppd = Gaussianize(
                    self,
                    part_ppd,
                    pt_cuts=pt_cuts,
                    pt_channels=pt_channels,
                    phi_channels=phi_channels,
                    eta_channels=eta_channels,
                    mass_channels=mass_channels,
                    mT_channel=24 if self.parameterization.transverse_mass.use else None,
                    logit_channels=logit_channels,
                    halfphi_channels=halfphi_channels,
                    reverse=False,
                )

            # Standardize to N(0, 1).
            # If external mean/std are provided (e.g. from the trained unfolder), reuse them
            # so that p_data is placed in the same latent space as the training signal.
            n_fit = reco_ppd.shape[0]
            if reco_mean is not None and reco_std is not None:
                self.reco_mean = reco_mean.to(self.device)
                self.reco_std  = reco_std.to(self.device)
                self.logger.info("    Using external reco_mean/reco_std (not refit from data).")
            else:
                # standardize_on_trn: statistics from the training split only (its rows: the first
                # trn_tst_val[0] of the file order, as in split_data); old configs: all events
                n_fit = reco_ppd.shape[0]
                if self.cfg.get("standardize_on_trn", False) and float(self.cfg.trn_tst_val[0]) > 0:
                    n_fit = int(reco_ppd.shape[0] * float(self.cfg.trn_tst_val[0]))
                    self.logger.info(f"    Standardization statistics from the training split ({n_fit} events)")
                self.reco_mean = reco_ppd[:n_fit].mean(dim=0)
                self.reco_std  = reco_ppd[:n_fit].std(dim=0)
            assert len(self.reco_mean) == reco_ppd.shape[1]
            assert len(self.reco_std) == reco_ppd.shape[1]
            # the truth uses the reco statistics, except for the unfolded channels when their coordinates
            # differ from the reco ones (valid_leptons=direct, reco_relative, jj_cuts): own truth statistics
            self.part_mean, self.part_std = self.reco_mean.clone(), self.reco_std.clone()
            if lep_direct or reco_relative or jj_cuts:
                if part_mean is not None and part_std is not None:
                    self.part_mean, self.part_std = part_mean.to(self.device), part_std.to(self.device)
                else:
                    self.part_mean[self.channels] = part_ppd[:n_fit, self.channels].mean(dim=0)
                    self.part_std[self.channels] = part_ppd[:n_fit, self.channels].std(dim=0)
            part_ppd = (part_ppd - self.part_mean) / (self.part_std + eps)
            reco_ppd = (reco_ppd - self.reco_mean) / (self.reco_std + eps)

            assert torch.isfinite(
                part_ppd
            ).all(), f"{torch.isnan(part_ppd).sum()} {torch.isinf(part_ppd).sum()}"
            assert torch.isfinite(
                reco_ppd
            ).all(), f"{torch.isnan(reco_ppd).sum()} {torch.isinf(reco_ppd).sum()}"

            self.part_ppd = part_ppd
            self.reco_ppd = reco_ppd

        else:
            if hasattr(self, "unfolded_ppd"):
                total_channels = self.part_ppd["tst"].shape[-1]
                # Get the channels that were not unfolded
                channels_to_fill = list(set(range(total_channels)) - set(self.channels))
                # Initialize a new tensor with the same shape as the whole data (without slicing channels),
                # filled with zeros in the channels that were not unfolded
                new_unfolded_ppd = torch.zeros(
                    (self.unfolded_ppd.shape[0], total_channels),
                    device=self.unfolded_ppd.device,
                )
                new_unfolded_ppd[
                    :,
                    [i for i in range(total_channels) if i not in channels_to_fill],
                ] = self.unfolded_ppd
                self.unfolded_ppd = new_unfolded_ppd

                # Reverse the Standardization
                pm = getattr(self, "part_mean", self.reco_mean)
                ps = getattr(self, "part_std", self.reco_std)
                self.unfolded = self.unfolded_ppd * (ps.detach().cpu() + eps) + pm.detach().cpu()
                reco_tst = None
                if reco_relative:  # add back the reference of the SAME (test) events
                    # reco_ref: the reco events that were unfolded, when that is not the whole test split
                    # (subset evaluations); default: the test split
                    reco_tst = (reco_ref if reco_ref is not None else
                                (self.reco["tst"] if isinstance(self.reco, dict) else self.reco)).detach().cpu()
                    assert reco_tst.shape[0] == self.unfolded.shape[0], (
                        f"reco_relative: {self.unfolded.shape[0]} unfolded vs {reco_tst.shape[0]} reco test events")
                    ref = _truth_forward(reco_tst).detach().cpu()
                    self.unfolded[:, additive] += ref[:, additive].to(self.unfolded.dtype)
                # Reverse the Gaussianization (truth-side maps)
                self.unfolded = Gaussianize(
                    self,
                    self.unfolded,
                    pt_cuts=pt_cuts,
                    pt_channels=t_pt,
                    phi_channels=t_phi,
                    eta_channels=eta_channels,
                    mass_channels=mass_channels,
                    mT_channel=t_mT,
                    logit_channels=logit_channels,
                    halfphi_channels=halfphi_channels,
                    reverse=True,
                )
                if periodic:
                    self.unfolded[:, periodic] = phys.wrap_angle(
                        self.unfolded[:, periodic] + reco_tst[:, periodic].to(self.unfolded.dtype))
                if self.parameterization.transverse_mass.get("valid", False):
                    self.unfolded = phys.complete_valid(
                        self.unfolded, self.logger,
                        jets=self.parameterization.transverse_mass.get("valid_jets", "ratio"), j2_pt_cut=j2_pt_cut,
                        jet_pt_cut=None if jet_pt_cut is None else float(jet_pt_cut),
                        leptons="direct" if lep_direct else "ratio", mT_offset=lep_mT_offset, jj_cuts=jj_cuts)
                elif self.parameterization.transverse_mass.get("balanced", False):
                    self.unfolded = phys.complete_balanced(self.unfolded, self.logger)
            else:
                raise ValueError("Cannot reverse preprocess without unfolded data.")

    def init_observables(
        self, pt_bins: int = 35, phi_bins: int = 35
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
                ),  # WBF_bins.get_Ptl1_bins(),
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
        self.observables.append(
            Observable(
                channel=3,
                compute=lambda p: return_obs(p[..., :], p[..., 3]),
                tex_label=r"m_{\ell_{1}}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=-0.05, upper=0.12
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=4 if not self.parameterization.transverse_mass.use else None,
                compute=lambda p: (
                    return_obs(p[..., :], p[..., 4])
                    if not self.parameterization.transverse_mass.use
                    else phys.get_pT_l2(
                        p=p[..., :],
                        pT_l1=p[..., 0],
                        pT_ll=p[..., 21],
                        dphi_ll=p[..., 22],
                    )
                ),
                tex_label=r"p_{T, \ell_{2}}"
                if not self.parameterization.transverse_mass.use
                else r"p_{T, \ell_{2}}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0, upper=290
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=5 if not self.parameterization.transverse_mass.use else None,
                compute=lambda p: return_obs(p[..., :], p[..., 5])
                if not self.parameterization.transverse_mass.use
                else phys.delta_phi(
                    p[..., :], p[..., 1], p[..., 22]
                ),  # phi_l1 - (phi_l1 - phi_l2)
                tex_label=r"\phi_{\ell_{2}}"
                if not self.parameterization.transverse_mass.use
                else r"\phi_{\ell_{2}}",
                unit="rad",
                bins=lambda obs: (
                    get_hardcoded_bins(n_bins=phi_bins + 1, lower=-3.5, upper=3.5)
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=6 if not self.parameterization.transverse_mass.use else None,
                compute=lambda p: return_obs(p[..., :], p[..., 6])
                if not self.parameterization.transverse_mass.use
                else phys.delta_eta(
                    p[..., :], p[..., 2], p[..., 23]
                ),  # eta_l1 - (eta_l1 - eta_l2)
                tex_label=r"\eta_{\ell_{2}}"
                if not self.parameterization.transverse_mass.use
                else r"\eta_{\ell_{2}}",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=7,
                compute=lambda p: return_obs(p[..., :], p[..., 7]),
                tex_label=r"m_{\ell_{2}}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=-0.05, upper=0.12
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=8,
                compute=lambda p: return_obs(p[..., :], p[..., 8]),
                tex_label=(
                    r"p_{T, j_1}"
                    if self.unfold_to == "particle"
                    else r"p_{T, j_{1}}"
                ),
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0, upper=500
                ),  # WBF_bins.get_Ptj1_bins(),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=9,
                compute=lambda p: return_obs(p[..., :], p[..., 9]),
                tex_label=(
                    r"\phi_{j_1}"
                    if self.unfold_to == "particle"
                    else r"\phi_{j_{1}}"
                ),
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
                tex_label=(
                    r"\eta_{j_1}"
                    if self.unfold_to == "particle"
                    else r"\eta_{j_{1}}"
                ),
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
                tex_label=r"m_{j_1}" if self.unfold_to == "particle" else r"m_{j_{1}}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1,
                    lower=0,
                    upper=60 if self.unfold_to == "particle" else 30,
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=12,
                compute=lambda p: return_obs(p[..., :], p[..., 12]),
                tex_label=(
                    r"p_{T, j_2}"
                    if self.unfold_to == "particle"
                    else r"p_{T, j_{2}}"
                ),
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0, upper=500
                ),  # WBF_bins.get_Ptj2_bins(),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=13,
                compute=lambda p: return_obs(p[..., :], p[..., 13]),
                tex_label=(
                    r"\phi_{j_2}"
                    if self.unfold_to == "particle"
                    else r"\phi_{j_{2}}"
                ),
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
                tex_label=(
                    r"\eta_{j_2}"
                    if self.unfold_to == "particle"
                    else r"\eta_{j_{2}}"
                ),
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
                tex_label=r"m_{j_2}"
                if self.unfold_to == "particle"
                else r"m_{j_{2}}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1,
                    lower=0,
                    upper=60 if self.unfold_to == "particle" else 30,
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=16 if not self.parameterization.transverse_mass.use else None,
                compute=lambda p: return_obs(p[..., :], p[..., 16])
                if not self.parameterization.transverse_mass.use
                else (
                    (
                        return_obs(p[..., :], p[..., 24]) ** 2
                        - (
                            phys.get_transverse_dilepton_energy(
                                p[..., 0:4],
                                torch.cat(
                                    [
                                        phys.get_pT_l2(
                                            p=p[..., :],
                                            pT_l1=p[..., 0],
                                            pT_ll=p[..., 21],
                                            dphi_ll=p[..., 22],
                                        ).unsqueeze(1),
                                        phys.delta_phi(
                                            p[..., :], p[..., 1], p[..., 22]
                                        ).unsqueeze(1),
                                        phys.delta_eta(
                                            p[..., :], p[..., 2], p[..., 23]
                                        ).unsqueeze(1),
                                    ],
                                    dim=1,
                                ),
                            )
                            ** 2
                            - return_obs(p[..., :], p[..., 21]) ** 2
                        )
                    )
                    / (
                        2
                        * (
                            phys.get_transverse_dilepton_energy(
                                p[..., 0:4],
                                torch.cat(
                                    [
                                        phys.get_pT_l2(
                                            p=p[..., :],
                                            pT_l1=p[..., 0],
                                            pT_ll=p[..., 21],
                                            dphi_ll=p[..., 22],
                                        ).unsqueeze(1),
                                        phys.delta_phi(
                                            p[..., :], p[..., 1], p[..., 22]
                                        ).unsqueeze(1),
                                        phys.delta_eta(
                                            p[..., :], p[..., 2], p[..., 23]
                                        ).unsqueeze(1),
                                    ],
                                    dim=1,
                                ),
                            )
                            - return_obs(p[..., :], p[..., 21])
                            * torch.cos(return_obs(p[..., :], p[..., 25]))
                        )
                    )
                ),
                tex_label=r"E_{T, \text{ miss}}"
                if not self.parameterization.transverse_mass.use
                else r"E_{T, \text{ miss}}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0, upper=300
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=17 if not self.parameterization.transverse_mass.use else None,
                compute=lambda p: return_obs(p[..., :], p[..., 17])
                if not self.parameterization.transverse_mass.use
                else phys.get_phi_miss(
                    p[..., 0:4],
                    torch.cat(
                        [
                            phys.get_pT_l2(
                                p=p[..., :],
                                pT_l1=p[..., 0],
                                pT_ll=p[..., 21],
                                dphi_ll=p[..., 22],
                            ).unsqueeze(1),
                            phys.delta_phi(p[..., :], p[..., 1], p[..., 22]).unsqueeze(
                                1
                            ),
                            phys.delta_eta(p[..., :], p[..., 2], p[..., 23]).unsqueeze(
                                1
                            ),
                        ],
                        dim=1,
                    ),
                    return_obs(p[..., :], p[..., 25]),
                ),
                tex_label=r"\phi_{\text{miss}}"
                if not self.parameterization.transverse_mass.use
                else r"\phi_{\text{miss}}",
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
        if self.parameterization.transverse_mass.use:
            self.observables.append(
                Observable(
                    channel=None
                    if not self.parameterization.transverse_mass.use
                    else 21,
                    compute=lambda p: 0
                    if not self.parameterization.transverse_mass.use
                    else return_obs(p[..., :], p[..., 21]),
                    tex_label=r"p_{T, \ell\ell}"
                    if not self.parameterization.transverse_mass.use
                    else r"p_{T, \ell\ell}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=pt_bins + 1, lower=0, upper=300
                    ),
                    yscale="linear",
                )
            )

        self.observables.append(
            Observable(
                channel=None if not self.parameterization.transverse_mass.use else 22,
                compute=lambda p: phys.delta_phi(
                    p[..., :],
                    p[..., 1],
                    p[..., 5],
                )
                if not self.parameterization.transverse_mass.use
                else return_obs(p[..., :], p[..., 22]),
                tex_label=r"\Delta\phi_{\ell\ell}"
                if not self.parameterization.transverse_mass.use
                else r"\Delta\phi_{\ell\ell}",
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-2, upper=2  # from 0 to 1.4
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                channel=None if not self.parameterization.transverse_mass.use else 23,
                compute=lambda p: phys.delta_eta(
                    p[..., :],
                    p[..., 2],
                    p[..., 6],
                )
                if not self.parameterization.transverse_mass.use
                else return_obs(p[..., :], p[..., 23]),
                tex_label=r"\Delta\eta_{\ell\ell}"
                if not self.parameterization.transverse_mass.use
                else r"\Delta\eta_{\ell\ell}",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=-2, upper=2  # from 0 to 3
                ),
                yscale="linear",
            )
        )

        self.observables.append(
            Observable(
                channel=None,
                compute=lambda p: phys.get_transverse_dilepton_energy(
                    p[..., 0:4], p[..., 4:8]
                )
                if not self.parameterization.transverse_mass.use
                else phys.get_transverse_dilepton_energy(
                    p[..., 0:4],
                    torch.cat(
                        [
                            phys.get_pT_l2(
                                p=p[..., :],
                                pT_l1=p[..., 0],
                                pT_ll=p[..., 21],
                                dphi_ll=p[..., 22],
                            ).unsqueeze(1),
                            phys.delta_phi(p[..., :], p[..., 1], p[..., 22]).unsqueeze(
                                1
                            ),
                            phys.delta_eta(p[..., :], p[..., 2], p[..., 23]).unsqueeze(
                                1
                            ),
                        ],
                        dim=1,
                    ),
                ),
                tex_label=r"E_{T, \ell\ell}",
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=0, upper=300
                ),
                yscale="linear",
            )
        )

        if self.parameterization.transverse_mass.use:
            # TODO implement dphi_ll-miss for naive parameterization
            self.observables.append(
                Observable(
                    channel=26,
                    compute=lambda p: return_obs(p[..., :], p[..., 25]),
                    tex_label=r"\Delta \phi_{\ell \ell, \text{ miss}}",
                    unit="rad",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=phi_bins + 1, lower=-3.5, upper=3.5
                    ),
                    yscale="linear",
                )
            )
        self.observables.append(
            Observable(
                compute=lambda p: phys.delta_phi(
                    p[..., :], p[..., 9], p[..., 13], abs=True
                ),
                tex_label=(
                    r"|\Delta\phi_{j_{1}j_{2}}|"
                    if self.unfold_to == "particle"
                    else r"|\Delta\phi_{j_{1}j_{2}}|"
                ),
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-0.1, upper=3.5  # from 0 to 1.4
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
        self.observables.append(
            Observable(
                compute=lambda p: phys.delta_eta(p[..., :], p[..., 10], p[..., 14]),
                tex_label=(
                    r"\Delta\eta_{j_{1}j_{2}}"
                    if self.unfold_to == "particle"
                    else r"\Delta\eta_{j_{1}j_{2}}"
                ),
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=1.0, upper=10.0  # from 0 to 10
                ),
                yscale="linear",
            )
        )
        self.observables.append(
            Observable(
                compute=lambda p: phys.invariant_mass(
                    p[..., :], [p[..., 8:12], p[..., 12:16]]
                ),
                tex_label=(
                    r"m_{j_{1}j_{2}}"
                    if self.unfold_to == "particle"
                    else r"m_{j_{1}j_{2}}"
                ),
                unit="GeV",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=pt_bins + 1, lower=300.0, upper=6000.0
                ),
                yscale="log",
            )
        )
        if self.parameterization.naive.use:
            self.observables.append(
                Observable(
                    compute=lambda p: phys.decay_transverse_mass(p[..., :])[0],
                    tex_label=r"m_{T}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=pt_bins + 1, lower=0.0, upper=250.0
                    ),
                    yscale="linear",
                )
            )
        elif self.parameterization.transverse_mass.use:
            self.observables.append(
                Observable(
                    compute=lambda p: (
                        (
                            (
                                return_obs(p[..., :], p[..., 24]) ** 2
                                - (
                                    phys.get_transverse_dilepton_energy(
                                        p[..., 0:4],
                                        torch.cat(
                                            [
                                                phys.get_pT_l2(
                                                    p=p[..., :],
                                                    pT_l1=p[..., 0],
                                                    pT_ll=p[..., 21],
                                                    dphi_ll=p[..., 22],
                                                ).unsqueeze(1),
                                                phys.delta_phi(
                                                    p[..., :],
                                                    p[..., 1],
                                                    p[..., 22],
                                                ).unsqueeze(1),
                                                phys.delta_eta(
                                                    p[..., :],
                                                    p[..., 2],
                                                    p[..., 23],
                                                ).unsqueeze(1),
                                            ],
                                            dim=1,
                                        ),
                                    )
                                    ** 2
                                    - return_obs(p[..., :], p[..., 21]) ** 2
                                )
                            )
                            / (
                                2
                                * (
                                    phys.get_transverse_dilepton_energy(
                                        p[..., 0:4],
                                        torch.cat(
                                            [
                                                phys.get_pT_l2(
                                                    p=p[..., :],
                                                    pT_l1=p[..., 0],
                                                    pT_ll=p[..., 21],
                                                    dphi_ll=p[..., 22],
                                                ).unsqueeze(1),
                                                phys.delta_phi(
                                                    p[..., :],
                                                    p[..., 1],
                                                    p[..., 22],
                                                ).unsqueeze(1),
                                                phys.delta_eta(
                                                    p[..., :],
                                                    p[..., 2],
                                                    p[..., 23],
                                                ).unsqueeze(1),
                                            ],
                                            dim=1,
                                        ),
                                    )
                                    - return_obs(p[..., :], p[..., 21])
                                    * torch.cos(return_obs(p[..., :], p[..., 25]))
                                )
                            )
                        )
                        + phys.get_transverse_dilepton_energy(
                            p[..., 0:4],
                            torch.cat(
                                [
                                    phys.get_pT_l2(
                                        p=p[..., :],
                                        pT_l1=p[..., 0],
                                        pT_ll=p[..., 21],
                                        dphi_ll=p[..., 22],
                                    ).unsqueeze(1),
                                    phys.delta_phi(
                                        p[..., :], p[..., 1], p[..., 22]
                                    ).unsqueeze(1),
                                    phys.delta_eta(
                                        p[..., :], p[..., 2], p[..., 23]
                                    ).unsqueeze(1),
                                ],
                                dim=1,
                            ),
                        )
                    )
                    ** 2,
                    tex_label=r"(E_{T,\text{ miss}} + E_{T, \ell\ell})^{2}",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=pt_bins + 1, lower=0.0, upper=100000.0
                    ),
                    yscale="linear",
                )
            )

            self.observables.append(
                Observable(
                    channel=25,
                    compute=lambda p: return_obs(p[..., :], p[..., 25]),
                    tex_label=r"\Delta\phi_{\ell\ell,\text{ miss}}",
                    unit="rad",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=phi_bins + 1, lower=-3.5, upper=3.5
                    ),
                    yscale="linear",
                )
            )

            self.observables.append(
                Observable(
                    compute=lambda p: (
                        (
                            return_obs(p[..., :], p[..., 24]) ** 2
                            - (
                                phys.get_transverse_dilepton_energy(
                                    p[..., 0:4],
                                    torch.cat(
                                        [
                                            phys.get_pT_l2(
                                                p=p[..., :],
                                                pT_l1=p[..., 0],
                                                pT_ll=p[..., 21],
                                                dphi_ll=p[..., 22],
                                            ).unsqueeze(1),
                                            phys.delta_phi(
                                                p[..., :], p[..., 1], p[..., 22]
                                            ).unsqueeze(1),
                                            phys.delta_eta(
                                                p[..., :], p[..., 2], p[..., 23]
                                            ).unsqueeze(1),
                                        ],
                                        dim=1,
                                    ),
                                )
                                ** 2
                                - return_obs(p[..., :], p[..., 21]) ** 2
                            )
                        )
                        / (
                            2
                            * (
                                phys.get_transverse_dilepton_energy(
                                    p[..., 0:4],
                                    torch.cat(
                                        [
                                            phys.get_pT_l2(
                                                p=p[..., :],
                                                pT_l1=p[..., 0],
                                                pT_ll=p[..., 21],
                                                dphi_ll=p[..., 22],
                                            ).unsqueeze(1),
                                            phys.delta_phi(
                                                p[..., :], p[..., 1], p[..., 22]
                                            ).unsqueeze(1),
                                            phys.delta_eta(
                                                p[..., :], p[..., 2], p[..., 23]
                                            ).unsqueeze(1),
                                        ],
                                        dim=1,
                                    ),
                                )
                                - return_obs(p[..., :], p[..., 21])
                                * torch.cos(return_obs(p[..., :], p[..., 25]))
                            )
                        )
                    )
                    ** 2
                    + return_obs(p[..., :], p[..., 21]) ** 2
                    + 2
                    * (
                        (
                            return_obs(p[..., :], p[..., 24]) ** 2
                            - (
                                phys.get_transverse_dilepton_energy(
                                    p[..., 0:4],
                                    torch.cat(
                                        [
                                            phys.get_pT_l2(
                                                p=p[..., :],
                                                pT_l1=p[..., 0],
                                                pT_ll=p[..., 21],
                                                dphi_ll=p[..., 22],
                                            ).unsqueeze(1),
                                            phys.delta_phi(
                                                p[..., :], p[..., 1], p[..., 22]
                                            ).unsqueeze(1),
                                            phys.delta_eta(
                                                p[..., :], p[..., 2], p[..., 23]
                                            ).unsqueeze(1),
                                        ],
                                        dim=1,
                                    ),
                                )
                                ** 2
                                - return_obs(p[..., :], p[..., 21]) ** 2
                            )
                        )
                        / (
                            2
                            * (
                                phys.get_transverse_dilepton_energy(
                                    p[..., 0:4],
                                    torch.cat(
                                        [
                                            phys.get_pT_l2(
                                                p=p[..., :],
                                                pT_l1=p[..., 0],
                                                pT_ll=p[..., 21],
                                                dphi_ll=p[..., 22],
                                            ).unsqueeze(1),
                                            phys.delta_phi(
                                                p[..., :], p[..., 1], p[..., 22]
                                            ).unsqueeze(1),
                                            phys.delta_eta(
                                                p[..., :], p[..., 2], p[..., 23]
                                            ).unsqueeze(1),
                                        ],
                                        dim=1,
                                    ),
                                )
                                - return_obs(p[..., :], p[..., 21])
                                * torch.cos(return_obs(p[..., :], p[..., 25]))
                            )
                        )
                    )
                    * return_obs(p[..., :], p[..., 21])
                    * torch.cos(return_obs(p[..., :], p[..., 25])),
                    tex_label=r"E^{2}_{T,\text{ miss}} + p^{2}_{T,\ell\ell} + 2\cdot E_{T,\text{ miss}}\cdot p_{T,\ell\ell} \cdot \cos\Delta\phi_{\ell\ell,\text{ miss}}",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=pt_bins + 1, lower=0.0, upper=100000.0
                    ),
                    yscale="log",
                )
            )
            self.observables.append(
                Observable(
                    channel=24,
                    compute=lambda p: return_obs(p[..., :], p[..., 24]),
                    tex_label=r"m_{T}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=pt_bins + 1, lower=0.0, upper=250.0
                    ),
                    yscale="linear",
                )
            )
        else:
            raise NotImplementedError("Parameterization not implemented")

        self.observables.append(
            Observable(
                compute=lambda p: phys.approximate_pT_Higgs(
                    p[..., :], p[..., 8:12], p[..., 12:16]
                ),
                tex_label=r"p_{T, H}\approx |- \vec{p_{T,j_1}} - \vec{p_{T,j_2}}|"
                if self.unfold_to == "particle"
                else r"p_{T, H}\approx |- \vec{p}_{T,q_1} - \vec{p}_{T,q_2}|",
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
                if self.unfold_to == "particle"
                else r"\phi_{H}\approx \phi_{q_1,q_2} + \pi",
                unit="rad",
                bins=lambda obs: get_hardcoded_bins(
                    n_bins=phi_bins + 1, lower=-3.5, upper=3.5
                ),
                yscale="linear",
            )
        )

        # Higgs transverse momentum from its decay products: vec pT_l1 + vec pT_l2 + vec pT_miss.
        # Reuses the lepton/miss observables, so it works for both parameterizations.
        obs_by_label = {o.tex_label: o for o in self.observables}
        pT_phi_pairs = [
            (obs_by_label[r"p_{T, \ell_{1}}"], obs_by_label[r"\phi_{\ell_{1}}"]),
            (obs_by_label[r"p_{T, \ell_{2}}"], obs_by_label[r"\phi_{\ell_{2}}"]),
            (obs_by_label[r"E_{T, \text{ miss}}"], obs_by_label[r"\phi_{\text{miss}}"]),
        ]

        def higgs_pxpy(p):
            pT = [pt_obs.compute(p) for pt_obs, _ in pT_phi_pairs]
            phi = [phi_obs.compute(p) for _, phi_obs in pT_phi_pairs]
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
                tex_label=r"H_{T,2j}" if self.unfold_to == "particle" else r"H_{T,2q}",
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
    logit_channels: list = (),
    halfphi_channels: list = (),
    halfphi_bound: float = 1.4,
    pt_cuts: dict = None,
) -> torch.Tensor:
    # check there is no overlap between channels
    assert len(set(pt_channels) & set(phi_channels)) == 0
    assert len(set(pt_channels) & set(eta_channels)) == 0
    assert len(set(pt_channels) & set(mass_channels)) == 0
    assert len(set(phi_channels) & set(eta_channels)) == 0
    assert len(set(phi_channels) & set(mass_channels)) == 0
    assert len(set(eta_channels) & set(mass_channels)) == 0

    pt_cuts = pt_cuts or {}
    for ch in pt_channels:
        if ch in pt_cuts:  # p_T > cut: log(p_T - cut) <-> cut + exp(.)
            c = pt_cuts[ch]
            x[:, ch] = (torch.log((x[:, ch] - c).clamp(min=1e-3)) if not reverse
                        else c + torch.exp(x[:, ch].clip(max=10)))
        else:
            x[:, ch] = gaussianize_pt(x[:, ch], eps, reverse)
    for ch in phi_channels:
        x[:, ch] = gaussianize_phi(x[:, ch], eps, reverse)
    for ch in eta_channels:
        x[:, ch] = gaussianize_eta(x[:, ch], eps, reverse)
    for ch in mass_channels:
        x[:, ch] = gaussianize_mass(x[:, ch], eps, reverse)
    for ch in logit_channels:  # (0, 1) <-> R
        x[:, ch] = torch.logit(x[:, ch].clamp(eps, 1 - eps)) if not reverse else torch.sigmoid(x[:, ch])
    for ch in halfphi_channels:  # (-halfphi_bound, halfphi_bound) <-> R
        x[:, ch] = (phys.stable_arctanh(x[:, ch] / halfphi_bound, eps) if not reverse
                    else torch.tanh(x[:, ch]) * halfphi_bound)
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
