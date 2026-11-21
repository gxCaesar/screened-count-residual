#!/usr/bin/env python3
"""NV2-N3: Six Additional Post-Hoc Calibration & Spatial Comparators across All 20 Frozen Generators.

Expands the 12-arm comparative audit to an 18-arm benchmark across HEMIT (11), Orion-CRC (6),
and HER2match (3) under identical locked training/test splits and physical cutoffs:
  13. Isotonic       : Training-split nonparametric monotonic IsotonicRegression LUT (0..255 -> 0..255)
  14. PlattTemp      : Training-split logistic temperature/bias calibration around tau_c mapped back to u8
  15. DenseCRF       : Edge-preserving H&E-guided bilateral DenseCRF mean-field inference (5 iters, train-tuned)
  16. MacenkoQM      : H&E Macenko optical-density normalization residual awareness + quantile matching LUT
  17. TopHat         : Morphological white top-hat cellular peak enhancement (b + alpha * TopHat_r(b), train-tuned)
  18. BilateralSharp : Edge-preserving bilateral high-boost sharpening (b + alpha * (b - Bilateral(b)), train-tuned)

Every comparator fits strictly on the training split and evaluates on the locked test split via
exact ``journal_cohort_eval`` metrics (Fixed_tol_F1, Exact_r0_F1, ODS_F, AP, SSIM, PSNR, neg_tile_fp_pct).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from s2 import journal_cohort_eval as jce
from s2 import run_her2match_dab_campaign as rher
from s2 import run_nm_comparators20 as rcomp
from s2 import run_orion_multibackbone_survival as ror


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# 6 New Classical / Statistical / Spatial Post-Hoc Comparators (Train-Only Fit)
# ---------------------------------------------------------------------------

def apply_isotonic_lut_u8(tr_up: np.ndarray, tr_y: np.ndarray, ev_up: np.ndarray) -> tuple[np.ndarray, dict]:
    """Exact 256-bin weighted IsotonicRegression mapping upstream u8 -> expected physical u8."""
    flat_x = tr_up.ravel()
    flat_y = tr_y.ravel().astype(np.float64)
    cnt = np.bincount(flat_x, minlength=256).astype(np.float64)
    ysum = np.bincount(flat_x, weights=flat_y, minlength=256)
    valid = cnt > 0
    xs = np.arange(256, dtype=np.float64)[valid]
    ym = ysum[valid] / cnt[valid]
    iso = IsotonicRegression(y_min=0.0, y_max=255.0, out_of_bounds="clip")
    iso.fit(xs, ym, sample_weight=cnt[valid])
    lut = np.rint(np.clip(iso.predict(np.arange(256, dtype=np.float64)), 0.0, 255.0)).astype(np.uint8)
    return lut[ev_up], dict(lut_min=int(lut[0]), lut_max=int(lut[-1]))


def apply_platt_temp_u8(tr_up: np.ndarray, tr_y: np.ndarray, ev_up: np.ndarray, cutoff: int) -> tuple[np.ndarray, dict]:
    """Platt / logistic temperature-and-bias calibration fitted on training binary positivity y > cutoff.
    Maps calibrated probability p = sigmoid(a*x + b) monotonically into u8 so p=0.5 aligns with tau_c+0.5.
    """
    rng = np.random.default_rng(20261121)
    n_pix = tr_up.size
    take = min(400_000, n_pix)
    idx = rng.choice(n_pix, size=take, replace=False)
    x_s = (tr_up.ravel()[idx].astype(np.float64) / 255.0)[:, None]
    y_s = (tr_y.ravel()[idx] > cutoff).astype(np.int32)
    if y_s.sum() == 0 or y_s.sum() == len(y_s):
        return ev_up.copy(), dict(coef=0.0, intercept=0.0, degenerate=True)
    # Class-balanced logistic regression so rare markers (CD3/CD8a/HER2) don't collapse to all-0!
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=200)
    clf.fit(x_s, y_s)
    a = float(clf.coef_[0, 0])
    b = float(clf.intercept_[0])
    # Monotonic probability-aligned intensity mapping: p in [0,1] -> u8 where p=0.5 hits cutoff+0.5
    x_grid = np.arange(256, dtype=np.float64) / 255.0
    p_grid = 1.0 / (1.0 + np.exp(-np.clip(a * x_grid + b, -30.0, 30.0)))
    # Piecewise linear map: [0, 0.5] -> [0, cutoff+0.5], [0.5, 1.0] -> [cutoff+0.5, 255]
    cut_f = float(cutoff) + 0.5
    u_grid = np.where(
        p_grid <= 0.5,
        (p_grid / 0.5) * cut_f,
        cut_f + ((p_grid - 0.5) / 0.5) * (255.0 - cut_f),
    )
    lut = np.rint(np.clip(u_grid, 0.0, 255.0)).astype(np.uint8)
    return lut[ev_up], dict(coef=round(a, 4), intercept=round(b, 4), degenerate=False)


def _bilateral_smooth_torch(img_f32: torch.Tensor, guide_rgb: torch.Tensor | None, sigma_s: float, sigma_r: float, ksize: int = 7) -> torch.Tensor:
    """Fast GPU bilateral filter guided by img_f32 (and optional H&E RGB)."""
    # img_f32: (B, 1, H, W) in [0, 1]; guide_rgb: (B, 3, H, W) in [0, 1] or None
    pad = ksize // 2
    B, _, H, W = img_f32.shape
    num = torch.zeros_like(img_f32)
    den = torch.zeros_like(img_f32)
    x_pad = F.pad(img_f32, (pad, pad, pad, pad), mode="reflect")
    g_pad = F.pad(guide_rgb, (pad, pad, pad, pad), mode="reflect") if guide_rgb is not None else None
    inv_2ss = 0.5 / max(sigma_s * sigma_s, 1e-6)
    inv_2sr = 0.5 / max(sigma_r * sigma_r, 1e-6)

    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            ws = float(np.exp(-(dx * dx + dy * dy) * inv_2ss))
            nb = x_pad[:, :, pad + dy : pad + dy + H, pad + dx : pad + dx + W]
            if g_pad is not None:
                gb = g_pad[:, :, pad + dy : pad + dy + H, pad + dx : pad + dx + W]
                d2 = ((gb - guide_rgb) ** 2).sum(dim=1, keepdim=True)
            else:
                d2 = (nb - img_f32) ** 2
            w = ws * torch.exp(-d2 * inv_2sr)
            num += w * nb
            den += w
    return num / den.clamp_min(1e-6)


def apply_dense_crf_u8(
    tr_he: np.ndarray, tr_up: np.ndarray, tr_y: np.ndarray,
    ev_he: np.ndarray, ev_up: np.ndarray, cutoff: int, device: torch.device,
) -> tuple[np.ndarray, dict]:
    """H&E-guided bilateral DenseCRF mean-field inference (5 iters, Krahenbuhl & Koltun) tuned on train split."""
    # Fit unary Platt logit on train first so CRF unary potential is well-scaled, then tune bilateral compatibility w_bi
    rng = np.random.default_rng(20261121)
    idx = rng.choice(tr_up.size, size=min(300_000, tr_up.size), replace=False)
    xs = (tr_up.ravel()[idx].astype(np.float64) / 255.0)[:, None]
    ys = (tr_y.ravel()[idx] > cutoff).astype(np.int32)
    if ys.sum() == 0 or ys.sum() == len(ys):
        return ev_up.copy(), dict(w_bi=0.0, sigma_r=0.15, degenerate=True)
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=200).fit(xs, ys)
    a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
    cut_f = float(cutoff) + 0.5

    def _run_crf(he_arr: np.ndarray, up_arr: np.ndarray, w_bi: float, sigma_r: float) -> np.ndarray:
        n = len(up_arr)
        out = np.empty_like(up_arr)
        bs = 16
        with torch.no_grad():
            for s0 in range(0, n, bs):
                e0 = min(n, s0 + bs)
                u = torch.from_numpy(up_arr[s0:e0]).to(device, dtype=torch.float32).unsqueeze(1) / 255.0
                g = torch.from_numpy(he_arr[s0:e0]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                logit0 = (a * u + b).clamp(-15.0, 15.0)
                q = torch.sigmoid(logit0)
                for _ in range(5):
                    # Message passing: bilateral filter on Q minus self, Potts compatibility
                    q_msg = _bilateral_smooth_torch(q, g, sigma_s=3.0, sigma_r=sigma_r, ksize=7)
                    # Mean-field update: logit = unary + w_bi * (2*q_msg - 1)
                    q = torch.sigmoid((logit0 + w_bi * (2.0 * q_msg - 1.0)).clamp(-15.0, 15.0))
                # Map probability q back to u8 around cutoff+0.5
                u_out = torch.where(
                    q <= 0.5,
                    (q / 0.5) * cut_f,
                    cut_f + ((q - 0.5) / 0.5) * (255.0 - cut_f),
                )
                out[s0:e0] = torch.round(u_out.clamp(0.0, 255.0)).squeeze(1).cpu().numpy().astype(np.uint8)
        return out

    # Tune (w_bi, sigma_r) on <=120 training tiles for speed
    tr_sub = np.linspace(0, len(tr_up) - 1, num=min(120, len(tr_up)), dtype=int)
    best = (1.0, 0.15, -1.0)
    for w_bi in (0.5, 1.5, 3.0):
        for sr in (0.08, 0.18):
            pr = _run_crf(tr_he[tr_sub], tr_up[tr_sub], w_bi, sr)
            ev = jce.batch_curves_and_quality_torch(pr, tr_y[tr_sub], cutoff, device=device, compute_quality=False)
            ep = jce.endpoints_from_pooled_curves(ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cutoff)
            f1 = float(ep["Fixed_tol_F1"])
            if f1 > best[2]:
                best = (w_bi, sr, f1)
    ev_out = _run_crf(ev_he, ev_up, best[0], best[1])
    return ev_out, dict(w_bi=best[0], sigma_r=best[1], train_f1=round(best[2], 4))


def apply_macenko_qm_u8(
    tr_he: np.ndarray, tr_up: np.ndarray, tr_y: np.ndarray,
    ev_he: np.ndarray, ev_up: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Macenko H&E optical-density quantile-conditioned LUT: splits tiles into low/high H&E optical-density
    tertiles (approximating stain-intensity normalization regimes) and fits per-stratum empirical CDF matching."""
    def _tile_od(he_arr: np.ndarray) -> np.ndarray:
        # Mean optical density -log((I+1)/256)
        od = -np.log((he_arr.astype(np.float32) + 1.0) / 256.0)
        return od.mean(axis=(1, 2, 3))

    od_tr = _tile_od(tr_he)
    od_ev = _tile_od(ev_he)
    q1, q2 = float(np.quantile(od_tr, 1.0 / 3.0)), float(np.quantile(od_tr, 2.0 / 3.0))
    ev_out = np.empty_like(ev_up)
    for k, (lo, hi) in enumerate([(-1e9, q1), (q1, q2), (q2, 1e9)]):
        m_tr = (od_tr >= lo) & (od_tr < hi) if k < 2 else (od_tr >= lo)
        m_ev = (od_ev >= lo) & (od_ev < hi) if k < 2 else (od_ev >= lo)
        if not np.any(m_ev):
            continue
        if int(m_tr.sum()) < 5:
            m_tr = np.ones_like(m_tr, dtype=bool)
        _, ev_sub, _ = rcomp.apply_quantile_match_u8(tr_up[m_tr], tr_y[m_tr], ev_up[m_ev])
        ev_out[m_ev] = ev_sub
    return ev_out, dict(od_q33=round(q1, 4), od_q67=round(q2, 4))


def apply_tophat_u8(tr_up: np.ndarray, tr_y: np.ndarray, ev_up: np.ndarray, cutoff: int, device: torch.device) -> tuple[np.ndarray, dict]:
    """Morphological white top-hat cellular peak enhancement: clip(b + alpha * (b - opening_k(b)))."""
    def _tophat_batch(arr_u8: np.ndarray, ksize: int, alpha: float) -> np.ndarray:
        n = len(arr_u8)
        out = np.empty_like(arr_u8)
        pad = ksize // 2
        with torch.no_grad():
            for s0 in range(0, n, 32):
                e0 = min(n, s0 + 32)
                x = torch.from_numpy(arr_u8[s0:e0]).to(device, dtype=torch.float32).unsqueeze(1)
                # Grayscale morphological opening = dilation(erosion(x))
                er = -F.max_pool2d(-F.pad(x, (pad, pad, pad, pad), mode="reflect"), ksize, stride=1)
                op = F.max_pool2d(F.pad(er, (pad, pad, pad, pad), mode="reflect"), ksize, stride=1)
                th = (x - op).clamp_min(0.0)
                y = (x + alpha * th).clamp(0.0, 255.0)
                out[s0:e0] = torch.round(y).squeeze(1).cpu().numpy().astype(np.uint8)
        return out

    tr_sub = np.linspace(0, len(tr_up) - 1, num=min(150, len(tr_up)), dtype=int)
    ev0 = jce.batch_curves_and_quality_torch(tr_up[tr_sub], tr_y[tr_sub], cutoff, device=device, compute_quality=False)
    base_f1 = float(jce.endpoints_from_pooled_curves(ev0["K"].sum(0), ev0["U"].sum(0), ev0["V"].sum(0), ev0["T"].sum(0), int(ev0["G"].sum()), cutoff)["Fixed_tol_F1"])
    best = (9, 0.0, base_f1)
    for ksize in (7, 13, 21):
        for alpha in (0.5, 1.0, 2.0, 3.5):
            pr = _tophat_batch(tr_up[tr_sub], ksize, alpha)
            ev = jce.batch_curves_and_quality_torch(pr, tr_y[tr_sub], cutoff, device=device, compute_quality=False)
            f1 = float(jce.endpoints_from_pooled_curves(ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cutoff)["Fixed_tol_F1"])
            if f1 > best[2] + 1e-6:
                best = (ksize, alpha, f1)
    ev_out = ev_up.copy() if best[1] == 0.0 else _tophat_batch(ev_up, best[0], best[1])
    return ev_out, dict(ksize=best[0], alpha=best[1], train_f1=round(best[2], 4))


def apply_bilateral_sharp_u8(tr_up: np.ndarray, tr_y: np.ndarray, ev_up: np.ndarray, cutoff: int, device: torch.device) -> tuple[np.ndarray, dict]:
    """Edge-preserving bilateral high-boost sharpening: clip(b + alpha * (b - Bilateral(b)))."""
    def _bsharp_batch(arr_u8: np.ndarray, sigma_r: float, alpha: float) -> np.ndarray:
        n = len(arr_u8)
        out = np.empty_like(arr_u8)
        with torch.no_grad():
            for s0 in range(0, n, 16):
                e0 = min(n, s0 + 16)
                x = torch.from_numpy(arr_u8[s0:e0]).to(device, dtype=torch.float32).unsqueeze(1) / 255.0
                sm = _bilateral_smooth_torch(x, None, sigma_s=3.0, sigma_r=sigma_r, ksize=7)
                y = (x + alpha * (x - sm)).clamp(0.0, 1.0) * 255.0
                out[s0:e0] = torch.round(y).squeeze(1).cpu().numpy().astype(np.uint8)
        return out

    tr_sub = np.linspace(0, len(tr_up) - 1, num=min(120, len(tr_up)), dtype=int)
    ev0 = jce.batch_curves_and_quality_torch(tr_up[tr_sub], tr_y[tr_sub], cutoff, device=device, compute_quality=False)
    base_f1 = float(jce.endpoints_from_pooled_curves(ev0["K"].sum(0), ev0["U"].sum(0), ev0["V"].sum(0), ev0["T"].sum(0), int(ev0["G"].sum()), cutoff)["Fixed_tol_F1"])
    best = (0.15, 0.0, base_f1)
    for sr in (0.08, 0.20):
        for alpha in (0.5, 1.5, 3.0):
            pr = _bsharp_batch(tr_up[tr_sub], sr, alpha)
            ev = jce.batch_curves_and_quality_torch(pr, tr_y[tr_sub], cutoff, device=device, compute_quality=False)
            f1 = float(jce.endpoints_from_pooled_curves(ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cutoff)["Fixed_tol_F1"])
            if f1 > best[2] + 1e-6:
                best = (sr, alpha, f1)
    ev_out = ev_up.copy() if best[1] == 0.0 else _bsharp_batch(ev_up, best[0], best[1])
    return ev_out, dict(sigma_r=best[0], alpha=best[1], train_f1=round(best[2], 4))


def eval_6_new_arms_channel(
    tr_he: np.ndarray, tr_up: np.ndarray, tr_y: np.ndarray,
    ev_he: np.ndarray, ev_up: np.ndarray, ev_y: np.ndarray,
    cutoff: int, device: torch.device,
) -> dict[str, dict]:
    preds = {}
    metas = {}
    preds["Isotonic"], metas["Isotonic"] = apply_isotonic_lut_u8(tr_up, tr_y, ev_up)
    preds["PlattTemp"], metas["PlattTemp"] = apply_platt_temp_u8(tr_up, tr_y, ev_up, cutoff)
    preds["DenseCRF"], metas["DenseCRF"] = apply_dense_crf_u8(tr_he, tr_up, tr_y, ev_he, ev_up, cutoff, device)
    preds["MacenkoQM"], metas["MacenkoQM"] = apply_macenko_qm_u8(tr_he, tr_up, tr_y, ev_he, ev_up)
    preds["TopHat"], metas["TopHat"] = apply_tophat_u8(tr_up, tr_y, ev_up, cutoff, device)
    preds["BilateralSharp"], metas["BilateralSharp"] = apply_bilateral_sharp_u8(tr_up, tr_y, ev_up, cutoff, device)

    neg_t = (ev_y > cutoff).sum(axis=(1, 2)) == 0
    out = {}
    for arm, pr in preds.items():
        ev = jce.batch_curves_and_quality_torch(pr, ev_y, cutoff, device=device, compute_quality=True)
        ep = jce.endpoints_from_pooled_curves(
            ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cutoff
        )
        neg_fp = 100.0 * float((pr[neg_t] > cutoff).mean()) if np.any(neg_t) else None
        out[arm] = dict(
            Fixed_tol_F1=round(float(ep["Fixed_tol_F1"]), 6),
            Exact_r0_F1=round(float(ep["Fixed_exact_F1"]), 6),
            ODS_F=round(float(ep["ODS_F"]), 6),
            AP=round(float(ep["AP"]), 6),
            SSIM=round(float(ev["ssims"].mean()), 6),
            PSNR_dB=round(float(ev["psnrs"].mean()), 4),
            neg_tile_fp_pct=round(neg_fp, 4) if neg_fp is not None else None,
            meta=metas[arm],
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hemit", "orion", "her2"], required=True)
    ap.add_argument("--train-pkl", default="")
    ap.add_argument("--test-pkl", default="")
    ap.add_argument("--upstream-dir", required=True)
    ap.add_argument("--cv-manifest", default="")
    ap.add_argument("--her2-root", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", default="")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out_json = Path(args.out_json)
    up_dir = Path(args.upstream_dir)
    want = set(args.models.split(",")) if args.models else None

    results = dict(schema="nv2-n3-comparators6-v1", cohort=args.cohort, models={})

    if args.cohort == "hemit":
        h_tr = pickle.load(open(args.train_pkl, "rb"))
        h_ev = pickle.load(open(args.test_pkl, "rb"))
        cuts = (22, 30, 18)  # Canonical rhmc.HEMIT_CUTOFFS (panCK, CD3, DAPI)
        ch_names = ("panCK", "CD3", "DAPI")
        # Auto-discover all *_train_596.npz & *_test_945.npz pairs in upstream-dir
        for tr_f in sorted(up_dir.glob("*_train_596.npz")):
            mkey = tr_f.name.replace("_train_596.npz", "")
            ev_f = up_dir / f"{mkey}_test_945.npz"
            if not ev_f.exists():
                continue
            if want and mkey not in want:
                continue
            tm = time.time()
            up_tr = np.load(tr_f)["pred_u8"]
            up_ev = np.load(ev_f)["pred_u8"]
            per_ch = {}
            for ch, (cn, cut) in enumerate(zip(ch_names, cuts)):
                per_ch[cn] = eval_6_new_arms_channel(
                    h_tr["he_nhwc"], up_tr[:, ch], h_tr["targ_nchw"][:, ch],
                    h_ev["he_nhwc"], up_ev[:, ch], h_ev["targ_nchw"][:, ch],
                    cut, device,
                )
            macro = {}
            for arm in ("Isotonic", "PlattTemp", "DenseCRF", "MacenkoQM", "TopHat", "BilateralSharp"):
                macro[arm] = {
                    k: round(float(np.mean([per_ch[cn][arm][k] for cn in ch_names])), 6)
                    for k in ("Fixed_tol_F1", "Exact_r0_F1", "ODS_F", "AP", "SSIM", "PSNR_dB")
                }
                macro[arm]["cd3_neg_tile_fp_pct"] = per_ch["CD3"][arm]["neg_tile_fp_pct"]
            results["models"][mkey] = dict(macro=macro, per_channel=per_ch)
            print(f"[NV2-N3 HEMIT] {mkey} ({time.time()-tm:.1f}s): " + " ".join(f"{a}={macro[a]['Fixed_tol_F1']:.4f}" for a in macro), flush=True)
            _atomic_write_json(out_json, results)

    elif args.cohort == "orion":
        ob = pickle.load(open(args.train_pkl, "rb"))
        om = json.loads(Path(args.cv_manifest).read_text())
        f0 = om["folds"][0]
        pats_arr = np.asarray(ob["patients"])
        tr_idx = np.nonzero(np.isin(pats_arr, f0["inner_train_slides"]))[0]
        ev_idx = np.nonzero(np.isin(pats_arr, f0["outer_test_slides"]))[0]
        cuts, _ = ror.calibrate_orion_physical_cutoffs_from_train(ob["targ_nchw"][tr_idx], tr_idx, ob["nuclei_records"])
        cuts = [int(c) for c in cuts]
        for fnpz in sorted(up_dir.glob("*_all40_2804.npz")):
            mkey = fnpz.name.replace("_all40_2804.npz", "")
            if want and mkey not in want:
                continue
            tm = time.time()
            up_all = np.load(fnpz)["pred_u8"]
            per_ch = {}
            for ch in range(16):
                cn = ror.ORION_MARKERS[ch]
                per_ch[cn] = eval_6_new_arms_channel(
                    ob["he_nhwc"][tr_idx], up_all[tr_idx, ch], ob["targ_nchw"][tr_idx, ch],
                    ob["he_nhwc"][ev_idx], up_all[ev_idx, ch], ob["targ_nchw"][ev_idx, ch],
                    cuts[ch], device,
                )
            macro = {}
            for arm in ("Isotonic", "PlattTemp", "DenseCRF", "MacenkoQM", "TopHat", "BilateralSharp"):
                macro[arm] = {
                    k: round(float(np.mean([per_ch[ror.ORION_MARKERS[c]][arm][k] for c in range(16)])), 6)
                    for k in ("Fixed_tol_F1", "Exact_r0_F1", "ODS_F", "AP", "SSIM", "PSNR_dB")
                }
            results["models"][mkey] = dict(macro=macro, per_channel=per_ch)
            print(f"[NV2-N3 ORION] {mkey} ({time.time()-tm:.1f}s): " + " ".join(f"{a}={macro[a]['Fixed_tol_F1']:.4f}" for a in macro), flush=True)
            _atomic_write_json(out_json, results)

    elif args.cohort == "her2":
        with open(args.train_pkl, "rb") as fh:
            tr_bank = pickle.load(fh)
        with open(args.test_pkl, "rb") as fh:
            te_bank = pickle.load(fh)
        hopt_ckpt = None
        env_ckpt = os.environ.get("HOPTIMUS0_CKPT")
        cand_paths = [
            Path(args.her2_root) / "h_optimus_0.pth",
            Path(args.her2_root) / "checkpoints" / "h_optimus_0.pth",
        ]
        if env_ckpt:
            cand_paths.insert(0, Path(env_ckpt))
        for cand in cand_paths:
            if cand.exists():
                hopt_ckpt = cand
                break
        if out_json.exists():
            prev = json.loads(out_json.read_text())
            results["models"].update(prev.get("models", {}))
        for b_key in ("unet_l1_ssim", "pix2pix_cgan_ihc", "hoptimus_lora_ihc"):
            if want and b_key not in want:
                continue
            if b_key in results["models"]:
                continue
            tm = time.time()
            _, up_tr, up_ev = rher.train_or_load_her2_backbone(
                b_key, tr_bank, te_bank, 23, up_dir, device,
                hoptimus0_ckpt=hopt_ckpt, upstream_epochs=8, allow_train=False,
            )
            res_ch = eval_6_new_arms_channel(
                tr_bank["he_nhwc"], up_tr, tr_bank["dab_u8"],
                te_bank["he_nhwc"], up_ev, te_bank["dab_u8"],
                23, device,
            )
            results["models"][b_key] = dict(macro=res_ch)
            print(f"[NV2-N3 HER2] {b_key} ({time.time()-tm:.1f}s): " + " ".join(f"{a}={res_ch[a]['Fixed_tol_F1']:.4f}" for a in res_ch), flush=True)
            _atomic_write_json(out_json, results)

    results["wall_seconds"] = round(time.time() - t0, 1)
    _atomic_write_json(out_json, results)
    print("NV2_N3_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
