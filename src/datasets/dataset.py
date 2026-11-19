import torch
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class Observable:
    """
    Data class for an observable used for plotting
    Args:
        compute: Function that computes the observable value for the given momenta
        tex_label: Observable name in LaTeX for labels in plots
        bins: function that returns tensor with bin boundaries for given observable data
        xscale: X axis scale, "linear" (default) or "log", optional
        yscale: Y axis scale, "linear" (default) or "log", optional
        unit: Unit of the observable or None, if dimensionless, optional
        channel: Channel index for the observable, optional
        key: Short unique name, optional (tells apart observables sharing a tex_label)
    """

    compute: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    tex_label: str
    bins: Callable[[torch.Tensor], torch.Tensor]
    xscale: str = "linear"
    yscale: str = "linear"
    unit: Optional[str] = None
    channel: Optional[int] = None
    key: Optional[str] = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["compute"] = None
        d["bins"] = None
        return d


def get_hardcoded_bins(n_bins, lower, upper):
    return torch.linspace(lower, upper, n_bins + 1)


def make_overflow_bin(bins, overflow=torch.inf):
    bins[-1] = overflow
    return bins


def round(p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return torch.round(obs)


def return_obs(p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return obs


def compute_observables(dataset, split: str = "tst", include_ppd: bool = False):
    if not hasattr(dataset, "observables"):
        raise ValueError(
            "No observables defined for this dataset. Make sure init_observables() has been called."
        )
    observables = []
    obs_part = []
    obs_reco = []
    obs_unf = []
    if include_ppd:
        observables_ppd = []
        obs_part_ppd = []
        obs_reco_ppd = []
        obs_unf_ppd = []

    bins = []
    for obs in dataset.observables:
        observables.append(obs)
        o_part = obs.compute(dataset.part[split])
        obs_part.append(o_part.cpu().numpy())
        o_reco = obs.compute(dataset.reco[split])
        obs_reco.append(o_reco.cpu().numpy())
        o_unf = obs.compute(dataset.unfolded) if hasattr(dataset, "unfolded") else None
        obs_unf.append(o_unf.cpu().numpy()) if o_unf is not None else None
        bin = obs.bins(o_part)
        bins.append(bin.cpu().numpy())
        if include_ppd and obs.channel is not None:
            observables_ppd.append(obs)
            o_part_ppd = obs.compute(dataset.part_ppd[split])
            obs_part_ppd.append(o_part_ppd.cpu().numpy())
            o_reco_ppd = obs.compute(dataset.reco_ppd[split])
            obs_reco_ppd.append(o_reco_ppd.cpu().numpy())
            o_unf_ppd = (
                obs.compute(dataset.unfolded_ppd)
                if hasattr(dataset, "unfolded_ppd")
                else None
            )
            (
                obs_unf_ppd.append(o_unf_ppd.cpu().numpy())
                if o_unf_ppd is not None
                else None
            )

    dataset.observables = observables
    dataset.obs_part = obs_part
    dataset.obs_reco = obs_reco
    dataset.obs_unf = obs_unf
    dataset.bins = bins
    if include_ppd:
        dataset.observables_ppd = observables_ppd
        dataset.obs_part_ppd = obs_part_ppd
        dataset.obs_reco_ppd = obs_reco_ppd
        dataset.obs_unf_ppd = obs_unf_ppd


def compute_classifier_observables(
    dataset,
    split: str = "tst",
    include_ppd: bool = False,
    include_hww: bool = False,
    include_ttbar: bool = False,
    include_wjj: bool = False,
):
    if not hasattr(dataset, "observables"):
        raise ValueError(
            "No observables defined for this dataset. Make sure init_observables() has been called."
        )
    observables = []
    # observables_hww = []
    # observables_ttbar = []
    # observables_wjj = []
    observables_p0 = []
    observables_p1 = []
    if include_ppd:
        obs_ppd = []
        # obs_hww_ppd = []
        # obs_ttbar_ppd = []
        # obs_wjj_ppd = []
        observables_p0_ppd = []
        observables_p1_ppd = []

    bins = []
    for obs in dataset.observables:
        # o_HWW = obs.compute(dataset.HWW[split]) if hasattr(dataset, "HWW") else None
        # o_ttbar = obs.compute(dataset.ttbar[split]) if hasattr(dataset, "ttbar") else None
        # o_wjj = obs.compute(dataset.wjj[split]) if hasattr(dataset, "wjj") else None
        o_p0 = obs.compute(dataset.p0[split]) if hasattr(dataset, "p0") else None
        o_p1 = obs.compute(dataset.p1[split]) if hasattr(dataset, "p1") else None

        bin = obs.bins(o_p0)
        if include_ppd and obs.channel is not None:
            o_p0_ppd = (
                obs.compute(dataset.p0_ppd[split])
                if hasattr(dataset, "p0_ppd")
                else None
            )
            o_p1_ppd = (
                obs.compute(dataset.p1_ppd[split])
                if hasattr(dataset, "p1_ppd")
                else None
            )

        observables.append(obs)
        # observables_hww.append(o_HWW.cpu().numpy()) if o_HWW is not None else None
        # observables_ttbar.append(o_ttbar.cpu().numpy()) if o_ttbar is not None else None
        # observables_wjj.append(o_wjj.cpu().numpy()) if o_wjj is not None else None
        observables_p0.append(o_p0.cpu().numpy()) if o_p0 is not None else None
        observables_p1.append(o_p1.cpu().numpy()) if o_p1 is not None else None
        bins.append(bin.cpu().numpy())
        if include_ppd:
            if obs.channel is not None:
                observables_p0_ppd.append(
                    o_p0_ppd.cpu().numpy() if o_p0_ppd is not None else None
                )
                observables_p1_ppd.append(
                    o_p1_ppd.cpu().numpy() if o_p1_ppd is not None else None
                )
            else:
                observables_p0_ppd.append(None)
                observables_p1_ppd.append(None)

    out = [bins, observables, observables_p0, observables_p1]
    # out.append(observables_hww, observables_ttbar, observables_wjj)
    if include_ppd:
        out.append(observables_p0_ppd)
        out.append(observables_p1_ppd)
    return out


class WBF_bins:
    @staticmethod
    def get_Ptl1_bins():
        return torch.Tensor([22, 40, 50, 60, 70, 100, 500])

    @staticmethod
    def get_Ptl2_bins():
        return torch.Tensor([15, 22, 30, 40, 50, 200])

    @staticmethod
    def get_Ptj1_bins():
        return torch.Tensor([30, 90, 120, 160, 220, 700])

    @staticmethod
    def get_Ptj2_bins():
        return torch.Tensor([30, 45, 60, 90, 120, 350])

    @staticmethod
    def get_DeltaPhill_bins():
        return torch.Tensor([0, 0.2, 0.4, 0.6, 0.8, 1.4])

    @staticmethod
    def get_DeltaEtajj_bins():
        return torch.Tensor([0, 2.1, 4.0, 4.375, 5.0, 5.5, 6.25, 9.0])

    @staticmethod
    def get_Mjj_bins():
        return torch.Tensor([450, 700, 950, 1200, 1500, 2200, 6000])

    @staticmethod
    def get_DeltaPhijj_bins():
        return torch.Tensor([-torch.pi, -torch.pi / 2, 0, torch.pi / 2, torch.pi])

    @staticmethod
    def get_DeltaEtall_bins(overflow=True):
        bins = torch.Tensor([0, 0.4, 0.6, 0.8, 1.0, 5])
        if overflow:
            bins = make_overflow_bin(bins)
        return bins

    @staticmethod
    def get_Mll_bins(overflow=True):
        bins = torch.Tensor([10, 20, 30, 40, 55, 2000])
        if overflow:
            bins = make_overflow_bin(bins)
        return bins

    @staticmethod
    def get_MT_bins():
        return torch.linspace(0, 250, 11)

    @staticmethod
    def get_PtHiggs_bins():
        return torch.Tensor([0, 80, 120, 160, 260, 850])

    @staticmethod
    def get_Ptll_bins():
        return torch.Tensor([0, 60, 80, 100, 140, 1000])

    @staticmethod
    def get_CosThetaStar_bins():
        return torch.Tensor([0, 0.0625, 0.125, 0.1875, 0.3125, 1])
