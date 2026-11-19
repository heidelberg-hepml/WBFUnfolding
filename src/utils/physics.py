import torch


def lepton_eta_centrality(
    p: torch.Tensor, lepton_idx: list = [0, 4], jet_idx: list = [8, 12]
) -> torch.Tensor:
    sum_C_l = 0
    for l in lepton_idx:
        delta_eta_jj = torch.abs(p[..., jet_idx[0] + 2] - p[..., jet_idx[1] + 2])
        C_l = torch.abs(2 * p[..., l + 2] - (p[..., 10] + p[..., 14])) / delta_eta_jj
        sum_Cl += C_l
    return sum_C_l


def pT_Higgs(
    p: torch.Tensor,
    p_l1: torch.Tensor,
    p_l2: torch.Tensor,
    p_miss: torch.Tensor,
) -> torch.Tensor:
    p_l1_EPxPyPz = PtPhiEtaM_to_EPxPyPz(p_l1)
    p_l2_EPxPyPz = PtPhiEtaM_to_EPxPyPz(p_l2)
    p_miss_EPxPyPz = PtPhiEtaM_to_EPxPyPz(p_miss)
    p_H_EPxPyPz = p_l1_EPxPyPz + p_l2_EPxPyPz + p_miss_EPxPyPz
    p_H_PtPhiEtaM = EPxPyPz_to_PtPhiEtaM(p_H_EPxPyPz)
    return p_H_PtPhiEtaM[..., 0]


def approximate_pT_Higgs(
    p: torch.Tensor, p_j1: torch.Tensor, p_j2: torch.Tensor
) -> torch.Tensor:
    p_j1_EPxPyPz = PtPhiEtaM_to_EPxPyPz(p_j1)
    p_j2_EPxPyPz = PtPhiEtaM_to_EPxPyPz(p_j2)
    px_H = -p_j1_EPxPyPz[..., 1] - p_j2_EPxPyPz[..., 1]
    py_H = -p_j1_EPxPyPz[..., 2] - p_j2_EPxPyPz[..., 2]
    return torch.sqrt(px_H**2 + py_H**2)


def approximate_phi_Higgs(p: torch.Tensor, p_j1: torch.Tensor, p_j2: torch.Tensor):
    p_j1_EPxPyPz = PtPhiEtaM_to_EPxPyPz(p_j1)
    p_j2_EPxPyPz = PtPhiEtaM_to_EPxPyPz(p_j2)
    p_jj_PtPhiEtaM = EPxPyPz_to_PtPhiEtaM(p_j1_EPxPyPz + p_j2_EPxPyPz)
    return wrap_angle(p_jj_PtPhiEtaM[:, 1] + torch.pi)


def get_transverse_dilepton_energy(
    p_l1: torch.Tensor, p_l2: torch.Tensor
) -> torch.Tensor:
    p_l1_EPxPyPz = PtPhiEtaM_to_EPxPyPz(p_l1)
    p_l2_EPxPyPz = PtPhiEtaM_to_EPxPyPz(p_l2)
    p_ll_EPxPyPz = p_l1_EPxPyPz + p_l2_EPxPyPz
    ET_ll = torch.sqrt(
        torch.clamp(p_ll_EPxPyPz[..., 0] ** 2 - p_ll_EPxPyPz[..., 3] ** 2, min=0.0)
    )  # ET = sqrt(E^2 - pz^2)
    return ET_ll


def get_phi_miss(
    p_l1: torch.Tensor, p_l2: torch.Tensor, dphi_ll_miss: torch.Tensor
) -> torch.Tensor:
    """
    Obtains phi miss from dilepton pT and delta phi between dilepton and missing transverse momentum
    """
    p_ll_PtPhiEtaM = EPxPyPz_to_PtPhiEtaM(
        PtPhiEtaM_to_EPxPyPz(p_l1) + PtPhiEtaM_to_EPxPyPz(p_l2)
    )
    phi_miss = delta_phi(
        p_l1,
        p_ll_PtPhiEtaM[..., 1],
        dphi_ll_miss,
    )
    return wrap_angle(phi_miss)


def get_pT_l2(
    p: torch.Tensor,
    pT_l1: torch.Tensor,
    pT_ll: torch.Tensor,
    dphi_ll: torch.Tensor,
) -> torch.Tensor:
    # calculate transverse momentum of the second lepton as a function of delta_eta_ll, Et_ll, and pT_ll
    # pT2 = -p[..., 0] * torch.cosh(delta_eta(p, p[..., 2], p[..., 6])) + p[
    #     ..., 0
    # ] * torch.sqrt(
    #     torch.cosh(delta_eta(p, p[..., 2], p[..., 6])) ** 2
    #     - 1
    #     + Et_ll**2 / p[..., 0] ** 2
    # )
    # assert torch.isfinite(
    #     pT2
    # ).all(), f"{torch.isnan(pT2).sum(axis=0)} {torch.isinf(pT2).sum(axis=0)}"

    """
    calculates the transverse momentum of the second lepton as a function of delta_phi_ll, pT_l1, and pT_ll
    """
    pT_l2 = -pT_l1 * torch.cos(dphi_ll) + torch.sqrt(
        torch.clamp(pT_ll**2 - pT_l1**2 * torch.sin(dphi_ll) ** 2, min=0.0)
    )
    assert torch.isfinite(
        pT_l2
    ).all(), f"{torch.isnan(pT_l2).sum(axis=0)}{torch.isinf(pT_l2).sum(axis=0)}"
    return pT_l2


def _balanced_mT(pT_l2, pT_l1, phi_l1, dphi_ll, deta_ll, pjj_x, pjj_y):
    """m_T for a trial pT_l2, with the missing p_T from the balance p_miss = -(p_ll + p_jj)."""
    phi_l2 = phi_l1 - dphi_ll
    llx = pT_l1 * torch.cos(phi_l1) + pT_l2 * torch.cos(phi_l2)
    lly = pT_l1 * torch.sin(phi_l1) + pT_l2 * torch.sin(phi_l2)
    pT_ll2 = llx**2 + lly**2
    m_ll2 = 2 * pT_l1 * pT_l2 * (torch.cosh(deta_ll) - torch.cos(dphi_ll))  # massless leptons
    Et_ll = torch.sqrt(pT_ll2 + m_ll2)
    Et_miss = torch.sqrt((llx + pjj_x) ** 2 + (lly + pjj_y) ** 2)
    # p_ll + p_miss = -p_jj
    return torch.sqrt(torch.clamp((Et_ll + Et_miss) ** 2 - pjj_x**2 - pjj_y**2, min=0.0)), llx, lly


def complete_balanced(p: torch.Tensor, logger=None, n_iter: int = 60) -> torch.Tensor:
    """Balanced transverse-mass parameterization: fill pT_ll (21) and dphi_ll-miss (25) of the unfolded
    events from the unfolded l1 (0-2), jets (8-10, 12-14), dphi_ll (22), deta_ll (23) and m_T (24), using
    the parton-level balance p_l1 + p_l2 + p_miss + p_j1 + p_j2 = 0. pT_l2 is the root of
    m_T(pT_l2) = m_T in (0, pT_l1] (unique on parton truth), found by bisection; events without a root
    take the end of the interval closest to it.
    """
    q = p.double()
    pT_l1, phi_l1 = q[:, 0], q[:, 1]
    dphi_ll, deta_ll, mT = q[:, 22], q[:, 23], q[:, 24]
    pjj_x = q[:, 8] * torch.cos(q[:, 9]) + q[:, 12] * torch.cos(q[:, 13])
    pjj_y = q[:, 8] * torch.sin(q[:, 9]) + q[:, 12] * torch.sin(q[:, 13])
    args = (pT_l1, phi_l1, dphi_ll, deta_ll, pjj_x, pjj_y)
    f = lambda x: _balanced_mT(x, *args)[0] - mT
    lo, hi = 1e-3 * pT_l1, pT_l1.clone()
    f_lo, f_hi = f(lo), f(hi)
    has_root = (f_lo * f_hi) <= 0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        left = (f_lo * f_mid) <= 0
        hi = torch.where(left, mid, hi)
        lo = torch.where(left, lo, mid)
        f_lo = torch.where(left, f_lo, f_mid)
    x = torch.where(has_root, 0.5 * (lo + hi),
                    torch.where(f(1e-3 * pT_l1).abs() < f(pT_l1).abs(), 1e-3 * pT_l1, pT_l1))
    _, llx, lly = _balanced_mT(x, *args)
    phi_ll = torch.atan2(lly, llx)
    phi_miss = torch.atan2(-(lly + pjj_y), -(llx + pjj_x))
    out = p.clone()
    out[:, 21] = torch.sqrt(llx**2 + lly**2).to(p.dtype)
    out[:, 25] = wrap_angle(phi_ll - phi_miss).to(p.dtype)
    if logger is not None:
        logger.info(
            f"    balanced completion: no m_T root in (0, pT_l1] for {100 * (~has_root).float().mean():.2f}% "
            f"of events; |dphi_ll| > pi/2 (ambiguous pT_l2) for {100 * (dphi_ll.abs() > torch.pi / 2).float().mean():.2f}%"
        )
    return out


def complete_valid(p: torch.Tensor, logger=None, jets: str = "ratio", j2_pt_cut: float = None,
                   jet_pt_cut: float = None, leptons: str = "ratio", mT_offset: bool = True,
                   jj_cuts: bool = False) -> torch.Tensor:
    """Valid-by-construction transverse-mass parameterization (parton level): from the unfolded
    l1 (0-2), dphi_ll (22), deta_ll (23), m_T (24), dphi_ll-miss (25), r = m_ll/m_T (26), signed
    dphi_jj = phi_j1 - phi_j2 (27), z = pT_j2/pT_j1 (28), eta_j1 (10), eta_j2 (14), build the event:
      m_ll = r m_T;  pT_l2 = m_ll^2 / [2 pT_l1 (cosh deta_ll - cos dphi_ll)]  (leptons swapped if pT_l2 > pT_l1)
      E_T,miss = (m_T^2 - m_ll^2) / [2 (E_T,ll - pT_ll cos dphi_ll-miss)],  phi_miss = phi_ll - dphi_ll-miss
      p_T,H = p_T,ll + p_T,miss;  the dijet recoils: pT_jj = pT_H, phi_jj = phi_H + pi
      pT_j1 = pT_jj / sqrt(1 + z^2 + 2 z cos dphi_jj),  pT_j2 = z pT_j1
      phi_j1 = phi_jj + atan2(z sin dphi_jj, 1 + z cos dphi_jj),  phi_j2 = phi_j1 - dphi_jj
    jets="direct": pT_j1 (8) and phi_j1 - phi_H (29) are unfolded instead of z and dphi_jj, and
      p_T,j2 = -(p_T,H + p_T,j1)  (no division by sqrt(1 + z^2 + 2 z cos dphi_jj), which is small for
      back-to-back jets of similar pT); dphi_jj (27) and z (28) are then filled from the jets.
      With j2_pt_cut, column 29 holds u = artanh(theta/theta_max) (theta = phi_j1 - phi_H), theta_max being
      the largest |theta| with pT_j2 > j2_pt_cut: cos theta_max = (cut^2 - pT_H^2 - pT_j1^2)/(2 pT_H pT_j1)
      (theta_max = pi if that is <= -1); theta = theta_max tanh(u) is written back to column 29.
    jets="symmetric": the half difference k = (p_j1 - p_j2)/2 is unfolded, p_j1,2 = -p_H/2 +- k, as
      y = log(k - k_min) (30) and u = artanh(t) (31), see symmetric_jet_features; with jet_pt_cut both
      jets pass it. k (30) and psi = phi_k - phi_H (31) are written back.
    leptons="direct": columns 0, 4, 24 hold y0, y4, y24 of direct_lepton_features instead of pT_l1, m_T
      (and r is not used): pT_l2 = 15 + e^y4, pT_l1 = max(22, pT_l2) + e^y0,
      m_ll = sqrt(2 pT_l1 pT_l2 (cosh deta_ll - cos dphi_ll)) (no division), m_T = m_ll + e^y24.
    jj_cuts (jets="ht_dphi"/"ht_nu"): columns 10, 14 hold the jj_eta_features coordinates instead of
      eta_j1, eta_j2, so that m_jj > 450, |deta_jj| > 2.1 and |eta_q| < 4.5; eta_j1, eta_j2 are written back.
    Fills pT_ll (21), the jet pT/phi (8, 9, 12, 13), l2 (4-6) and MET (16, 17) explicitly, and writes the
    swapped l1/dphi_ll/deta_ll back. Every input point gives a physical event with exact p_T balance.
    """
    assert not jj_cuts or jets in ("ht_dphi", "ht_nu"), "jj_cuts needs valid_jets=ht_dphi or ht_nu"
    q = p.double()
    a, phi1, eta1 = q[:, 0], q[:, 1], q[:, 2]
    dphi, deta, mT, dllm = q[:, 22], q[:, 23], q[:, 24], q[:, 25]
    r, dphi_jj, z = q[:, 26], q[:, 27], q[:, 28]
    if leptons == "direct":
        b = LEP_PT_CUTS[1] + torch.exp(q[:, 4].clip(max=10))
        a = torch.clamp(b, min=LEP_PT_CUTS[0]) + torch.exp(q[:, 0].clip(max=10))
        m_ll = torch.sqrt(2 * a * b * (torch.cosh(deta) - torch.cos(dphi)).clamp(min=0))
        if mT_offset:
            mT = m_ll + torch.exp(q[:, 24].clip(max=10))
        else:  # m_T unfolded in its own map: keep m_T > m_ll (else no positive E_T,miss)
            n_clamped = int((q[:, 24] <= m_ll).sum())
            mT = torch.maximum(q[:, 24], m_ll * (1 + 1e-6) + 1e-6)
            if logger is not None:
                logger.info(f"    direct leptons: m_T raised to m_ll for {n_clamped} events ({100 * n_clamped / len(mT):.3f}%)")
        r = m_ll / mT
    else:
        m_ll = r * mT
        b = m_ll**2 / (2 * a * torch.clamp(torch.cosh(deta) - torch.cos(dphi), min=1e-12))
    swap = b > a
    a, b = torch.where(swap, b, a), torch.where(swap, a, b)
    phi1 = torch.where(swap, wrap_angle(phi1 - dphi), phi1)
    eta1 = torch.where(swap, eta1 - deta, eta1)
    dphi, deta = torch.where(swap, -dphi, dphi), torch.where(swap, -deta, deta)
    phi2, eta2 = wrap_angle(phi1 - dphi), eta1 - deta
    llx = a * torch.cos(phi1) + b * torch.cos(phi2)
    lly = a * torch.sin(phi1) + b * torch.sin(phi2)
    pT_ll, phi_ll = torch.sqrt(llx**2 + lly**2), torch.atan2(lly, llx)
    Et_ll = torch.sqrt(pT_ll**2 + m_ll**2)
    met = (mT**2 - m_ll**2) / (2 * (Et_ll - pT_ll * torch.cos(dllm)))
    phi_miss = wrap_angle(phi_ll - dllm)
    Hx, Hy = llx + met * torch.cos(phi_miss), lly + met * torch.sin(phi_miss)
    pT_jj, phi_jj = torch.sqrt(Hx**2 + Hy**2), torch.atan2(-Hy, -Hx)
    if jets in ("dphi_pt2", "ht_dphi", "ht_nu"):
        pH, phiH = torch.sqrt(Hx**2 + Hy**2), torch.atan2(Hy, Hx)
        if jets == "dphi_pt2":
            pT_j2 = q[:, 12]
            pT_j1, dphi_jj = dphi_pt2_inverse(pH, pT_j2, q[:, 32], DPHI_PT2_CUT)
        elif jets == "ht_nu":
            pT_j1, pT_j2, dphi_jj = ht_nu_inverse(pH, q[:, 33], q[:, 34], jet_pt_cut)
        else:
            pT_j1, pT_j2, dphi_jj = ht_dphi_inverse(pH, q[:, 33], q[:, 34], jet_pt_cut)
        phi_j1, phi_j2 = _orient(pH, phiH, pT_j1, pT_j2, dphi_jj)
        z = pT_j2 / pT_j1
        if jj_cuts:
            eta_j1, eta_j2 = jj_eta_inverse(pT_j1, pT_j2, dphi_jj, q[:, 10], q[:, 14])
    elif jets == "symmetric":
        pH = torch.sqrt(Hx**2 + Hy**2)
        k, psi = symmetric_jet_inverse(pH, q[:, 30], q[:, 31], jet_pt_cut)
        phi_k = torch.atan2(Hy, Hx) + psi
        j1x, j1y = -Hx / 2 + k * torch.cos(phi_k), -Hy / 2 + k * torch.sin(phi_k)
        j2x, j2y = -Hx / 2 - k * torch.cos(phi_k), -Hy / 2 - k * torch.sin(phi_k)
        pT_j1, phi_j1 = torch.sqrt(j1x**2 + j1y**2), torch.atan2(j1y, j1x)
        pT_j2, phi_j2 = torch.sqrt(j2x**2 + j2y**2), torch.atan2(j2y, j2x)
        dphi_jj, z = wrap_angle(phi_j1 - phi_j2), pT_j2 / pT_j1
    elif jets == "direct":
        pT_j1 = q[:, 8]
        theta = q[:, 29]
        if j2_pt_cut is not None:
            theta = j2_theta_max(torch.sqrt(Hx**2 + Hy**2), pT_j1, j2_pt_cut) * torch.tanh(theta)
        phi_j1 = wrap_angle(torch.atan2(Hy, Hx) + theta)
        j2x, j2y = -Hx - pT_j1 * torch.cos(phi_j1), -Hy - pT_j1 * torch.sin(phi_j1)
        pT_j2, phi_j2 = torch.sqrt(j2x**2 + j2y**2), torch.atan2(j2y, j2x)
        dphi_jj, z = wrap_angle(phi_j1 - phi_j2), pT_j2 / pT_j1
    else:
        pT_j1 = pT_jj / torch.sqrt(torch.clamp(1 + z**2 + 2 * z * torch.cos(dphi_jj), min=1e-12))
        phi_j1 = wrap_angle(phi_jj + torch.atan2(z * torch.sin(dphi_jj), 1 + z * torch.cos(dphi_jj)))
        pT_j2, phi_j2 = z * pT_j1, wrap_angle(phi_j1 - dphi_jj)
    out = p.clone()
    for col, val in {0: a, 1: phi1, 2: eta1, 3: 0 * a, 4: b, 5: phi2, 6: eta2, 7: 0 * a, 22: dphi, 23: deta,
                     24: mT, 26: r,
                     21: pT_ll, 16: met, 17: phi_miss, 8: pT_j1, 9: phi_j1, 12: pT_j2,
                     13: phi_j2, 27: dphi_jj, 28: z, **({29: theta} if jets == "direct" else {}),
                     **({30: k, 31: psi} if jets == "symmetric" else {}),
                     **({32: dphi_jj, 33: pT_j1 + pT_j2, 34: dphi_jj} if jets in ("dphi_pt2", "ht_dphi", "ht_nu") else {}),
                     **({10: eta_j1, 14: eta_j2} if jj_cuts else {})}.items():
        out[:, col] = val.to(p.dtype)
    if logger is not None:
        bal = torch.sqrt((Hx + pT_j1 * torch.cos(phi_j1) + pT_j2 * torch.cos(phi_j2)) ** 2
                         + (Hy + pT_j1 * torch.sin(phi_j1) + pT_j2 * torch.sin(phi_j2)) ** 2)
        logger.info(f"    valid completion: leptons swapped for {100 * swap.double().mean():.2f}% of events; "
                    f"non-finite {int((~torch.isfinite(out)).any(1).sum())}; max |sum p_T| {bal.max():.2e} GeV")
    return out


def j2_theta_max(pT_H: torch.Tensor, pT_j1: torch.Tensor, cut: float) -> torch.Tensor:
    """Largest |phi_j1 - phi_H| with |p_T,H + p_T,j1| = pT_j2 > cut (pi if every angle passes)."""
    c = (cut**2 - pT_H**2 - pT_j1**2) / (2 * pT_H * pT_j1).clamp(min=1e-12)
    return torch.arccos(c.clamp(-1.0, 1.0))


LEP_PT_CUTS = (22.0, 15.0)  # parton-level p_T,l1 / p_T,l2 thresholds
DPHI_PT2_CUT = 30.0  # parton-level quark p_T threshold of valid_jets=dphi_pt2 (derived q1)


def _orient(pH, phiH, pT1, pT2, dphi):
    """phi_q1, phi_q2 of two jets with p_q1 + p_q2 = -p_H and phi_q1 - phi_q2 = dphi."""
    zx, zy = pT2 + pT1 * torch.cos(dphi), pT1 * torch.sin(dphi)
    phi2 = wrap_angle(phiH + torch.pi - torch.atan2(zy, zx))
    return wrap_angle(phi2 + dphi), phi2


def _dphi_pt2_bound(pH, a, cut):
    """valid_jets=dphi_pt2: (loop, bound) per event. a = pT_q2 < pT_H: the coordinate is
    delta = dphi_jj - pi in (-bound, bound); otherwise the loop angle chi in (-bound, bound), with
    dphi_jj = pi + alpha sin chi (sin alpha = pT_H/a) and pT_q1 = a cos delta + sign(cos chi) sqrt(.).
    bound keeps pT_q1 > cut (pi if the cut is never active)."""
    loop = a >= pH
    # dphi at which pT_q1 = cut: pT_H^2 = cut^2 + a^2 + 2 cut a cos dphi*
    cstar = ((pH**2 - cut**2 - a**2) / (2 * cut * a)).clamp(-1, 1)
    dstar = torch.pi - torch.arccos(cstar)            # |delta| where pT_q1 = cut
    # unique branch: pT_q1 grows with |dphi|, so delta in (-(pi - dphi*), pi - dphi*) = (-dstar', dstar')
    b_unique = torch.where(pH - a >= cut, torch.full_like(a, torch.pi), dstar)
    # loop: on which branch does pT_q1 = cut sit at |delta| = dstar?
    alpha = torch.arcsin((pH / a).clamp(max=1.0))
    root = torch.sqrt((pH**2 - a**2 * torch.sin(dstar) ** 2).clamp(min=0))
    xminus = a * torch.cos(dstar) - root
    s = (dstar / alpha.clamp(min=1e-12)).clamp(-1, 1)
    chi_star = torch.where((xminus - cut).abs() <= (a * torch.cos(dstar) + root - cut).abs(),
                           torch.pi - torch.arcsin(s), torch.arcsin(s))
    b_loop = torch.where(a - pH >= cut, torch.full_like(a, torch.pi), chi_star)
    return loop, torch.where(loop, b_loop, b_unique), alpha


def dphi_pt2_features(pH, a, x, dphi, cut, eps=1e-10):
    """Truth -> u = artanh(angle / bound) for valid_jets=dphi_pt2 (a = pT_q2, x = pT_q1, dphi = phi_q1 - phi_q2)."""
    loop, bound, alpha = _dphi_pt2_bound(pH, a, cut)
    delta = wrap_angle(dphi - torch.pi)
    s = (delta / alpha.clamp(min=1e-12)).clamp(-1, 1)
    plus = (x - a * torch.cos(delta)) >= 0           # '+' root <-> cos chi >= 0
    chi = torch.where(plus, torch.arcsin(s), wrap_angle(torch.pi - torch.arcsin(s)))
    angle = torch.where(loop, chi, delta)
    return stable_arctanh((angle / bound).clamp(-1, 1), eps)


def dphi_pt2_inverse(pH, a, u, cut):
    """-> (pT_q1, dphi = phi_q1 - phi_q2)."""
    loop, bound, alpha = _dphi_pt2_bound(pH, a, cut)
    angle = bound * torch.tanh(u)
    delta = torch.where(loop, alpha * torch.sin(angle), angle)
    root = torch.sqrt((pH**2 - a**2 * torch.sin(delta) ** 2).clamp(min=0))
    sign = torch.where(loop, torch.sign(torch.cos(angle)), torch.ones_like(a))
    x = a * torch.cos(delta) + sign * root
    return x, wrap_angle(delta + torch.pi)


def _ht_dphi_bounds(pH, S, cut):
    """Both jets > cut: H_T >= S0 = max(pT_H, 2 cut) and |sin chi| >= sin chi_min, chi_min in [0, pi/2]
    (|pT_q1 - pT_q2| grows with |cos chi|; it equals H_T - 2 cut at P = cut (H_T - cut))."""
    if cut is None:
        return pH, torch.zeros_like(pH)
    beta = torch.arcsin((pH / S).clamp(max=1.0))
    c_min = ((S**2 - pH**2) / (2 * cut * (S - cut)).clamp(min=1e-12) - 1).clamp(-1, 1)
    s_min = (torch.arccos(c_min) / (2 * beta).clamp(min=1e-12)).clamp(0, 1)
    return torch.clamp(pH, min=2 * cut), torch.arcsin(s_min)


def ht_dphi_features(pH, x, y, dphi, eps=1e-10, cut=None):
    """valid_jets=ht_dphi: H_T = x + y unfolded as log(H_T - pT_H); dphi on the loop
    dphi = pi + 2 beta sin chi (sin beta = pT_H/H_T), q1 harder <-> cos chi >= 0; chi as artanh(chi/pi).
    With cut (jet_pt_cut): log(H_T - max(pT_H, 2 cut)) and u = pi (chi -+ chi_min)/(pi - 2 chi_min) as
    artanh(u/pi), so every unfolded value gives both jets > cut."""
    S = x + y
    beta = torch.arcsin((pH / S).clamp(max=1.0))
    delta = wrap_angle(dphi - torch.pi)
    s = (delta / (2 * beta).clamp(min=1e-12)).clamp(-1, 1)
    chi = torch.where(x >= y, torch.arcsin(s), wrap_angle(torch.pi - torch.arcsin(s)))
    S0, chi_min = _ht_dphi_bounds(pH, S, cut)
    u = torch.pi * (chi - torch.sign(chi) * chi_min) / (torch.pi - 2 * chi_min).clamp(min=1e-12)
    return torch.log((S - S0).clamp(min=1e-6)), stable_arctanh((u / torch.pi).clamp(-1, 1), eps)


def ht_dphi_inverse(pH, yS, ychi, cut=None):
    """-> (pT_q1, pT_q2, dphi)."""
    S = torch.clamp(pH, min=0 if cut is None else 2 * cut) + torch.exp(yS.clip(max=10))
    _, chi_min = _ht_dphi_bounds(pH, S, cut)
    u = torch.pi * torch.tanh(ychi)
    chi = u * (torch.pi - 2 * chi_min) / torch.pi + torch.sign(u) * chi_min
    beta = torch.arcsin((pH / S).clamp(max=1.0))
    delta = 2 * beta * torch.sin(chi)
    P = (S**2 - pH**2) / (2 * (1 + torch.cos(delta))).clamp(min=1e-12)
    root = torch.sqrt((S**2 - 4 * P).clamp(min=0))
    x = (S + torch.sign(torch.cos(chi)) * root) / 2
    return x, S - x, wrap_angle(delta + torch.pi)


def _ht_nu_bounds(pH, S, cut):
    """Both jets > cut: H_T >= S0 = max(pT_H, 2 cut) and |cos nu| <= kappa = min(1, (H_T - 2 cut)/pT_H)."""
    if cut is None:
        return pH, torch.zeros_like(pH)
    kappa = ((S - 2 * cut) / pH.clamp(min=1e-12)).clamp(0, 1)
    return torch.clamp(pH, min=2 * cut), torch.arccos(kappa)


def ht_nu_features(x, y, dphi, eps=1e-10, cut=None):
    """valid_jets=ht_nu: elliptic coordinates of the quark pair around the foci 0 and -p_T,H
    (pT_q1,2 = (H_T +- pT_H cos nu)/2). H_T as log(H_T - pT_H) (with cut: - max(pT_H, 2 cut)), nu as
    artanh(u/pi), u = pi (nu -+ nu_0)/(pi - 2 nu_0) (nu_0 = 0 without cut). pT_H is the quark-pair one;
    H_T - pT_H = 4 x y sin^2(dphi/2)/(H_T + pT_H) and nu = atan2(2 sqrt(x y) cos(dphi/2) sgn(dphi), x - y)
    have no cancellation, also for collinear quarks (H_T = pT_H)."""
    S = x + y
    pH = torch.sqrt((x**2 + y**2 + 2 * x * y * torch.cos(dphi)).clamp(min=0))
    nu = torch.atan2(2 * torch.sqrt(x * y) * torch.cos(dphi / 2) * torch.sign(dphi), x - y)
    S0, nu0 = _ht_nu_bounds(pH, S, cut)
    gap = 4 * x * y * torch.sin(dphi / 2) ** 2 / (S + pH) if cut is None else S - S0
    u = torch.pi * (nu - torch.sign(nu) * nu0) / (torch.pi - 2 * nu0).clamp(min=1e-12)
    return torch.log(gap.clamp(min=1e-6)), stable_arctanh((u / torch.pi).clamp(-1, 1), eps)


def ht_nu_inverse(pH, yS, ynu, cut=None):
    """-> (pT_q1, pT_q2, dphi); dphi/2 = sgn(nu) atan2(sqrt(H_T^2 - pT_H^2), pT_H |sin nu|)."""
    S0 = pH if cut is None else torch.clamp(pH, min=2 * cut)
    S = S0 + torch.exp(yS.clip(max=10))
    _, nu0 = _ht_nu_bounds(pH, S, cut)
    u = torch.pi * torch.tanh(ynu)
    nu = u * (torch.pi - 2 * nu0) / torch.pi + torch.sign(u) * nu0
    x = (S + pH * torch.cos(nu)) / 2
    dphi = 2 * torch.sign(nu) * torch.atan2(torch.sqrt(((S - pH) * (S + pH)).clamp(min=0)),
                                            pH * torch.abs(torch.sin(nu)))
    return x, S - x, wrap_angle(dphi)


JJ_CUTS = dict(m_jj=450.0, deta_jj=2.1, eta_q=4.5)  # parton-level VBF generator cuts (massless quarks)


def _deta_jj_min(pT1, pT2, dphi):
    """Smallest |deta_jj| passing both m_jj > 450 and |deta_jj| > 2.1, for fixed quark pT and dphi:
    m_jj^2 = 2 pT1 pT2 (cosh deta - cos dphi)."""
    c = (torch.cos(dphi) + JJ_CUTS["m_jj"] ** 2 / (2 * pT1 * pT2).clamp(min=1e-12)).clamp(min=1.0)
    return torch.clamp(torch.arccosh(c), min=JJ_CUTS["deta_jj"])


def jj_eta_features(pT1, pT2, dphi, eta1, eta2, eps=1e-10):
    """jj_cuts: the quark pseudorapidities as (signed deta_jj, centre), both bounded per event so that
    m_jj > 450, |deta_jj| > 2.1 and |eta_q| < 4.5 hold for every unfolded value:
      v = sgn(deta) (|deta| - d_min)/(2 eta_max - d_min) -> artanh(v)   (d_min from _deta_jj_min)
      c = (eta1 + eta2)/2 in |c| < eta_max - |deta|/2  -> artanh(c/(eta_max - |deta|/2))."""
    deta, emax = eta1 - eta2, JJ_CUTS["eta_q"]
    dmin = _deta_jj_min(pT1, pT2, dphi)
    v = torch.sign(deta) * ((deta.abs() - dmin) / (2 * emax - dmin)).clamp(0, 1)
    half = (emax - deta.abs() / 2).clamp(min=1e-12)
    return stable_arctanh(v, eps), stable_arctanh(((eta1 + eta2) / 2 / half).clamp(-1, 1), eps)


def jj_eta_inverse(pT1, pT2, dphi, yd, yc):
    """-> (eta_q1, eta_q2) from the jj_eta_features coordinates."""
    emax = JJ_CUTS["eta_q"]
    dmin = _deta_jj_min(pT1, pT2, dphi)
    v = torch.tanh(yd)
    ad = dmin + v.abs() * (2 * emax - dmin)
    c = torch.tanh(yc) * (emax - ad / 2)
    deta = torch.where(v < 0, -ad, ad)
    return c + deta / 2, c - deta / 2


def direct_lepton_features(P):
    """valid_leptons=direct coordinates of a raw (pT, phi, eta, m, ...) tensor:
    y0 = log(pT_l1 - max(22, pT_l2)), y4 = log(pT_l2 - 15), y24 = log(m_T - m_ll), with the massless
    m_ll = sqrt(2 pT_l1 pT_l2 (cosh deta_ll - cos dphi_ll)) used by the inverse (complete_valid)."""
    a, b = P[:, 0].double(), P[:, 4].double()
    m_ll = torch.sqrt(2 * a * b * (torch.cosh(P[:, 2].double() - P[:, 6].double())
                                   - torch.cos(P[:, 1].double() - P[:, 5].double())).clamp(min=0))
    y0 = torch.log((a - torch.clamp(b, min=LEP_PT_CUTS[0])).clamp(min=1e-3))
    y4 = torch.log((b - LEP_PT_CUTS[1]).clamp(min=1e-3))
    y24 = torch.log((P[:, 24].double() - m_ll).clamp(min=1e-3))
    return y0, y4, y24


def _sym_bounds(pH, k, cut):
    """k_min and a: both jets p_j1,2 = -p_H/2 +- k pass cut iff k > k_min and a < |psi| < pi - a."""
    if cut is None:
        return torch.zeros_like(pH), torch.zeros_like(pH)
    k_min = torch.sqrt((cut**2 - pH**2 / 4).clamp(min=0))
    if k is None:
        return k_min, None
    C = (pH**2 / 4 + k**2 - cut**2) / (pH * k).clamp(min=1e-12)
    return k_min, torch.arccos(C.clamp(-1.0, 1.0))


def symmetric_jet_features(pH, phiH, k_vec_x, k_vec_y, cut=None, eps=1e-10):
    """Truth -> unfolded coordinates of the symmetric jet parameterization: y = log(k - k_min),
    u = artanh(t) with psi = sign(t) (a + (pi - 2a)|t|), psi = phi_k - phi_H in (-pi, pi)."""
    k = torch.sqrt(k_vec_x**2 + k_vec_y**2)
    psi = wrap_angle(torch.atan2(k_vec_y, k_vec_x) - phiH)
    k_min, a = _sym_bounds(pH, k, cut)
    y = torch.log((k - k_min).clamp(min=1e-3))
    t = torch.sign(psi) * ((psi.abs() - a) / (torch.pi - 2 * a).clamp(min=1e-12)).clamp(0, 1)
    return y, stable_arctanh(t, eps)


def symmetric_jet_inverse(pH, y, u, cut=None):
    k_min, _ = _sym_bounds(pH, None, cut)
    k = k_min + torch.exp(y.clip(max=10))
    _, a = _sym_bounds(pH, k, cut)
    t = torch.tanh(u)
    return k, torch.sign(t) * (a + (torch.pi - 2 * a) * t.abs())


def wrap_angle(phi: torch.Tensor) -> torch.Tensor:
    """
    Wraps angle to [-pi, pi).
    """
    return (phi + torch.pi) % (2 * torch.pi) - torch.pi


def higgs_system_pieces(p: torch.Tensor) -> torch.Tensor:
    p_ll_EPxPyPz = PtPhiEtaM_to_EPxPyPz(p[..., 0:4]) + PtPhiEtaM_to_EPxPyPz(p[..., 4:8])
    p_ll_PtPhiEtaM = EPxPyPz_to_PtPhiEtaM(p_ll_EPxPyPz)
    p_miss_PtPhiEtaM = p[..., 16:20]
    m_miss = 0  # invariant missing mass
    # m_miss = p_ll_PtPhiEtaM[..., 3]
    Et_ll = torch.sqrt(
        p_ll_PtPhiEtaM[..., 0] ** 2 + p_ll_PtPhiEtaM[..., 3] ** 2
    )  # Et = sqrt(pt^2 + m^2)
    Et_miss = torch.sqrt(p_miss_PtPhiEtaM[..., 0] ** 2 + m_miss**2)
    Pt_ll = p_ll_PtPhiEtaM[..., 0]
    dphi_ll_miss = delta_phi(p, p_ll_PtPhiEtaM[..., 1], p_miss_PtPhiEtaM[..., 1])
    mT2 = (Et_miss + Et_ll) ** 2 - (
        Et_miss**2 + Pt_ll**2 + 2 * Et_miss * Pt_ll * torch.cos(dphi_ll_miss)
    )
    mT = torch.sqrt(mT2)
    return Et_miss, Pt_ll, dphi_ll_miss, mT


def decay_transverse_mass(p: torch.Tensor, eps=1e-6) -> torch.Tensor:
    p_ll_EPxPyPz = PtPhiEtaM_to_EPxPyPz(p[..., 0:4]) + PtPhiEtaM_to_EPxPyPz(p[..., 4:8])
    p_ll_PtPhiEtaM = EPxPyPz_to_PtPhiEtaM(p_ll_EPxPyPz)
    p_miss_PtPhiEtaM = p[..., 16:20]
    # check if m-LL is well defined
    m_ll2 = (
        eps
        + p_ll_EPxPyPz[..., 0] ** 2
        - p_ll_EPxPyPz[..., 1] ** 2
        - p_ll_EPxPyPz[..., 2] ** 2
        - p_ll_EPxPyPz[..., 3] ** 2
    )
    assert (m_ll2 >= 0).all(), f"m_ll: {m_ll2[~(m_ll2 >= 0)]}"
    m_ll = torch.sqrt(m_ll2)
    # assert (torch.isclose(m_ll, p_ll_PtPhiEtaM[..., 3], rtol=1e-2)).all(), f"m_ll: {m_ll[torch.isclose(m_ll, p_ll_PtPhiEtaM[..., 3], rtol=1e-2)]} != p_ll[3]: {p_ll_PtPhiEtaM[torch.isclose(m_ll, p_ll_PtPhiEtaM[..., 3], rtol=1e-2)]}"
    # m_miss = m_ll is correct at threshold but most strongly peaked
    # m_miss = 0 defines a proper bounded-from-above variable that seems to improve the Higgs mass extraction
    m_miss = 0  # invariant missing mass
    Et_ll = torch.sqrt(
        p_ll_PtPhiEtaM[..., 0] ** 2 + p_ll_PtPhiEtaM[..., 3] ** 2
    )  # Et = sqrt(pt^2 + m^2)
    Et_miss = torch.sqrt(p_miss_PtPhiEtaM[..., 0] ** 2 + m_miss**2)
    dphi = delta_phi(p, p_ll_PtPhiEtaM[..., 1], p_miss_PtPhiEtaM[..., 1])
    mT = torch.sqrt(
        m_ll**2
        + m_miss**2
        + 2
        * (
            Et_ll * Et_miss
            - p_ll_PtPhiEtaM[..., 0] * p_miss_PtPhiEtaM[..., 0] * torch.cos(dphi)
        )
    )
    return mT, m_ll


def invariant_mass(p: torch.Tensor, objects: list[torch.Tensor]) -> torch.Tensor:
    inv_p = torch.zeros(p.shape[0], 4).to(p.device)
    for object in objects:
        object_EPxPyPz = PtPhiEtaM_to_EPxPyPz(object)
        inv_p += object_EPxPyPz
    return torch.sqrt(
        inv_p[..., 0] ** 2
        - inv_p[..., 1] ** 2
        - inv_p[..., 2] ** 2
        - inv_p[..., 3] ** 2
    )


def delta_eta(
    p: torch.Tensor, eta1: torch.Tensor, eta2: torch.Tensor, abs: bool = False
) -> torch.Tensor:
    deta = eta1 - eta2
    return torch.abs(deta) if abs else deta


def delta_phi(
    p: torch.Tensor, phi1: torch.Tensor, phi2: torch.Tensor, abs: bool = False
) -> torch.Tensor:
    dphi = phi1 - phi2
    dphi = wrap_angle(dphi)
    return torch.abs(dphi) if abs else dphi


def PtPhiEtaM_to_EPxPyPz(PtPhiEtaM, cutoff=10):
    if PtPhiEtaM.shape[-1] == 4:
        pt, phi, eta, mass = PtPhiEtaM[:, torch.arange(4)].T
    elif PtPhiEtaM.shape[-1] == 3:
        pt, phi, eta = PtPhiEtaM[:, torch.arange(3)].T
        mass = torch.zeros_like(pt)  # mass is neglected
    else:
        raise ValueError(f"PtPhiEtaM has wrong shape {PtPhiEtaM.shape}")

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(torch.clip(eta, -cutoff, cutoff))
    E = torch.sqrt(mass**2 + px**2 + py**2 + pz**2)

    EPxPyPz = torch.stack((E, px, py, pz), axis=-1)
    assert torch.isfinite(
        EPxPyPz
    ).all(), f"{torch.isnan(EPxPyPz).sum(axis=0)} {torch.isinf(EPxPyPz).sum(axis=0)}"
    return EPxPyPz


def EPxPyPz_to_PtPhiEtaM(EPxPyPz):
    pt = get_pt(EPxPyPz)
    phi = get_phi(EPxPyPz)
    eta = get_eta(EPxPyPz)
    mass = get_mass(EPxPyPz)

    PtPhiEtaM = torch.stack((pt, phi, eta, mass), axis=-1)
    assert torch.isfinite(
        PtPhiEtaM
    ).all(), (
        f"{torch.isnan(PtPhiEtaM).sum(axis=0)} {torch.isinf(PtPhiEtaM).sum(axis=0)}"
    )
    return PtPhiEtaM


def get_pt(particle):
    return torch.sqrt(particle[..., 1] ** 2 + particle[..., 2] ** 2)


def get_phi(particle):
    return torch.arctan2(particle[..., 2], particle[..., 1])


def get_eta(particle, eps=1e-10):
    # eta = torch.arctanh(particle[...,3] / p_abs) # numerically unstable
    p_abs = torch.sqrt(torch.sum(particle[..., 1:] ** 2, axis=-1))
    eta = 0.5 * (
        torch.log(torch.clip(torch.abs(p_abs + particle[..., 3]), eps, None))
        - torch.log(torch.clip(torch.abs(p_abs - particle[..., 3]), eps, None))
    )
    return eta


def get_mass(particle, eps=1e-6):
    return torch.sqrt(
        torch.clip(
            particle[..., 0] ** 2 - torch.sum(particle[..., 1:] ** 2, axis=-1),
            eps,
            None,
        )
    )


def stable_arctanh(x, eps):
    # numerically stable implementation of arctanh that avoids log(0) issues
    return 0.5 * (
        torch.log(torch.clip(1 + x, min=eps, max=None))
        - torch.log(torch.clip(1 - x, min=eps, max=None))
    )
