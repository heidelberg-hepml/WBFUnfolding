"""Build <classifier_run>/eval_block/: the subset of the classifier's dumped p_data that lies in
the unfolder's TEST window. Only this subset is ever unfolded.

Window definition (signal) = EXACTLY the unfolder's tst slice, recomputed with WBF's own floor
arithmetic from the unfolder run's frozen config: [int(N*trn), int(N*trn) + int(N*tst)).
Background files get the same fractional window (the unfolder never trains on background, so any
background is safe to unfold; the same window keeps the mixture composition standard).

Hard assertions (the "100% sure" part):
  1. the kept signal reco rows byte-match np.load(hww_rec)[a:b] as a multiset (count + content);
  2. zero kept signal rows appear among the unfolder's TRAIN rows np.load(hww_rec)[:a] (byte check);
  3. the window is contained in the classifier's train pool range.

Standalone: numpy + torch + yaml only, no repo imports.
"""
import argparse
import os

import numpy as np
import torch
import yaml


def f32(a):
    return np.ascontiguousarray(np.asarray(a).astype(np.float64).astype(np.float32))


def row_bytes(a):
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier-dir", required=True)
    ap.add_argument("--unfolder-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <classifier-dir>/eval_block")
    ap.add_argument("--cap", type=int, default=None,
                    help="smoke only: subsample the eval block to N rows AFTER all assertions")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.classifier_dir, "eval_block")

    # ---- classifier dump -------------------------------------------------------------------
    d = torch.load(os.path.join(args.classifier_dir, "unfolding_weights_trn.pt"), map_location="cpu")
    assert d.get("dump_split") == "trn", f"expected dump_split='trn', got {d.get('dump_split')!r}"
    assert "src_pos_p0" in d and d.get("pool_counts"), "dump lacks src_pos_p0/pool_counts (explicit_split off?)"
    saw = d["signal_approx_weights"]            # (K, N)
    is_sig = d["is_signal"].numpy().astype(bool)
    hww_idx = d["hww_index"].numpy()            # signal: row within the pool's HWW slice; bkg: -1
    src_pos = d["src_pos_p0"].numpy()           # bkg: row within the pool's [ttbar ++ WWjj] block
    pc = d["pool_counts"]                       # {"HWW": n, "ttbar": n, "WWjj": n, "range": [lo, hi]}
    lo, hi = (float(x) for x in pc["range"])
    N_rows = saw.shape[1]
    assert is_sig.shape[0] == N_rows == src_pos.shape[0]

    # ---- unfolder frozen config ------------------------------------------------------------
    with open(os.path.join(args.unfolder_dir, "config.yaml")) as f:
        ucfg = yaml.safe_load(f)
    uds = ucfg["dataset"]
    assert not uds["combined"]["use"], "unfolder must be non-combined (contiguous file-order split)"
    assert uds.get("n_data") in (None, "null"), "unfolder n_data must be null (full file)"
    trn_f, tst_f, _ = (float(x) for x in uds["trn_tst_val"])
    # Truth level must be consistent between classifier (dump) and unfolder (unfold_to):
    #   particle <-> gen (reco-gen reco + p_data_gen.npy), parton <-> parton (reco-hard + p_data_parton.npy).
    unfold_to = ucfg["dataset"].get("unfold_to", "parton")
    truth_level = d.get("truth_level", "gen")
    expected_tl = "parton" if unfold_to == "parton" else "gen"
    assert truth_level == expected_tl, (
        f"truth-level mismatch: classifier dumped '{truth_level}' but unfolder unfold_to='{unfold_to}' "
        f"(expects '{expected_tl}')")
    reco_key = "path_reco_corresponding_to_parton" if unfold_to == "parton" else "path_reco_corresponding_to_gen"
    truth_fname = "p_data_parton.npy" if truth_level == "parton" else "p_data_gen.npy"
    # WBF joins data_path + relative filename; the classifier stores the absolute path.
    hww_rec_path = os.path.normpath(os.path.join(uds["data_path"], uds[reco_key]))
    assert hww_rec_path == os.path.normpath(d["hww_path"]), (
        f"classifier and unfolder read different signal files:\n  clf: {d['hww_path']}\n  unf: {hww_rec_path}")

    # ---- windows (EXACT WBF _split_data arithmetic) ------------------------------------------
    hww_rec = np.load(hww_rec_path, mmap_mode="r")
    N_sig = hww_rec.shape[0]
    a = int(N_sig * trn_f)                       # unfolder trn end
    b = a + int(N_sig * tst_f)                   # unfolder tst end
    sig_pool_lo, sig_pool_hi = int(N_sig * lo), int(N_sig * hi)
    print(f"signal file: N={N_sig}  unfolder tst window=[{a},{b})  classifier trn pool=[{sig_pool_lo},{sig_pool_hi})")
    # assertion 3: containment of the window in the classifier train pool
    assert sig_pool_lo <= a and b <= sig_pool_hi, "unfolder tst window NOT inside classifier train pool"

    sig_global = np.where(is_sig, sig_pool_lo + hww_idx, -1)
    keep_sig = is_sig & (sig_global >= a) & (sig_global < b)

    # background: same fractional window per file, decoded from src_pos (pos < n_tt -> ttbar).
    # With dataset.shuffle_ttbar_seed the ttbar positions index the SHUFFLED file (dump's
    # ttbar_row_perm), so the ttbar window is a random subset of the same size: fine, background
    # never enters the unfolder; only the signal rows must match the unfolder tst window (asserts 1+2).
    n_tt_pool = int(pc["ttbar"])
    keep_bkg = np.zeros(N_rows, dtype=bool)
    for name, path_key in (("ttbar", "ttbar_path"), ("WWjj", "wwjj_path")):
        M = np.load(d[path_key], mmap_mode="r").shape[0]
        wa = int(M * trn_f)
        wb = wa + int(M * tst_f)
        pool_lo = int(M * lo)
        if name == "ttbar":
            in_proc = (~is_sig) & (src_pos < n_tt_pool)
            grow = pool_lo + src_pos
        else:
            in_proc = (~is_sig) & (src_pos >= n_tt_pool)
            grow = pool_lo + (src_pos - n_tt_pool)
        keep_bkg |= in_proc & (grow >= wa) & (grow < wb)
        print(f"{name}: N={M}  window=[{wa},{wb})  kept {int((in_proc & (grow >= wa) & (grow < wb)).sum())} rows")
    keep = keep_sig | keep_bkg
    print(f"kept rows: {int(keep.sum())} / {N_rows}  (signal {int(keep_sig.sum())}, bkg {int(keep_bkg.sum())})")

    # ---- byte-level assertions 1 + 2 ----------------------------------------------------------
    p_reco = np.load(os.path.join(args.classifier_dir, "p_data_reco.npy"))
    assert p_reco.shape[0] == N_rows, (p_reco.shape, N_rows)
    kept_sig_rows = row_bytes(f32(p_reco[keep_sig]))
    window_rows = row_bytes(f32(hww_rec[a:b]))
    assert kept_sig_rows.shape[0] == window_rows.shape[0], (
        f"count mismatch: kept {kept_sig_rows.shape[0]} vs window {window_rows.shape[0]}")
    assert np.array_equal(np.sort(kept_sig_rows), np.sort(window_rows)), (
        "kept signal rows do NOT byte-match the unfolder tst window")
    print(f"ASSERT 1 OK: kept signal rows == unfolder tst rows, byte-for-byte ({window_rows.shape[0]} rows)")
    train_rows = set(row_bytes(f32(hww_rec[:a])).tolist())
    n_leak = sum(1 for r in kept_sig_rows.tolist() if r in train_rows)
    assert n_leak == 0, f"{n_leak} kept signal rows appear in the unfolder TRAIN block"
    print(f"ASSERT 2 OK: zero kept signal rows in the unfolder train block ([0,{a}))")

    # ---- optional smoke cap (after assertions) ------------------------------------------------
    keep_idx = np.where(keep)[0]
    if args.cap is not None and keep_idx.shape[0] > args.cap:
        rng = np.random.default_rng(7)
        keep_idx = np.sort(rng.choice(keep_idx, size=args.cap, replace=False))
        print(f"SMOKE CAP: subsampled eval block to {args.cap} rows (assertions ran on the full window)")

    # ---- write eval_block ----------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "p_data_reco.npy"), p_reco[keep_idx])
    p_gen = np.load(os.path.join(args.classifier_dir, truth_fname))
    np.save(os.path.join(out_dir, truth_fname), p_gen[keep_idx])
    out = dict(d)
    for k in ("signal_approx_weights",):
        out[k] = d[k][:, keep_idx]
    for k in ("hww_index", "is_signal", "mask_wwjj_in_p0", "mask_ttbar_in_p0", "prior_weights", "src_pos_p0"):
        out[k] = d[k][keep_idx]
    out["eval_block"] = {"window_sig": [a, b], "parent": os.path.abspath(args.classifier_dir),
                         "cap": args.cap, "n_rows": int(keep_idx.shape[0])}
    torch.save(out, os.path.join(out_dir, "unfolding_weights_trn.pt"))
    print(f"wrote {out_dir}: {keep_idx.shape[0]} rows "
          f"({int(out['is_signal'].sum())} signal, {int((~out['is_signal'].numpy().astype(bool)).sum())} bkg)")


if __name__ == "__main__":
    main()
