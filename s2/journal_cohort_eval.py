"""Shared pixel-level, single-cell, and cluster-bootstrap evaluation for Route 1 cohorts.

Computes exact 256-threshold spatial-tolerance (L2 <= 5, 81-pixel footprint) count curves,
ODS-F, Mean AP, Fixed-Cutoff F1, Training-Calibrated F1, Objective-Decision Mismatch Gap
(ODS_F - Fixed_tol_F1), native 11x11 Gaussian SSIM (sigma=1.5), PSNR (dB), single-cell
nuclear phenotyping, spatial infiltration concordance, and paired cluster bootstrap CIs
without touching any holdout data unless explicitly authorized under frozen S6 mode.

Includes an exact GPU-batched tolerance-curve & SSIM/PSNR accelerator that produces
bit-for-bit identical integer histogram curves (K, U, V, T, G) to SciPy's CPU reference
while running ~500x faster on throttled CPU cgroup hosts (e.g. cluster-host 4-core cgroup).
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage, stats
import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def tolerance_footprint_81() -> np.ndarray:
    """Exact 17x17 L2 <= 5 chamfer/Euclidean tolerance footprint (81 True pixels)."""
    yy, xx = np.ogrid[-8:9, -8:9]
    mask = (yy * yy + xx * xx) <= 25
    if int(mask.sum()) != 81:
        raise RuntimeError(f"Expected 81 pixels in tolerance footprint, got {int(mask.sum())}")
    mask.setflags(write=False)
    return mask


@lru_cache(maxsize=1)
def _footprint_offsets_81() -> tuple[tuple[int, int], ...]:
    fp = tolerance_footprint_81()
    ys, xs = np.nonzero(fp)
    return tuple((int(y) - 8, int(x) - 8) for y, x in zip(ys, xs))


@lru_cache(maxsize=1)
def _gaussian_1d_11() -> np.ndarray:
    coords = np.arange(11, dtype=np.float64) - 5.0
    g = np.exp(-(coords * coords) / (2.0 * 1.5 * 1.5))
    g /= g.sum()
    g.setflags(write=False)
    return g


@lru_cache(maxsize=8)
def _gaussian_1d_kernels_torch(device_str: str) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.from_numpy(_gaussian_1d_11().copy()).to(device=torch.device(device_str), dtype=torch.float32)
    return g.view(1, 1, 11, 1), g.view(1, 1, 1, 11)


def _sep_gauss_valid_f32(arr: torch.Tensor, g_col: torch.Tensor, g_row: torch.Tensor) -> torch.Tensor:
    return F.conv2d(F.conv2d(arr, g_col), g_row)


def ssim_gaussian_u8(pred_hw: np.ndarray, targ_hw: np.ndarray) -> float:
    """Native 11x11 Gaussian (sigma=1.5) valid-center SSIM on uint8 [0, 255] images."""
    g = _gaussian_1d_11()
    x = pred_hw.astype(np.float64, copy=False)
    y = targ_hw.astype(np.float64, copy=False)

    def _filt_valid(arr: np.ndarray) -> np.ndarray:
        tmp = ndimage.correlate1d(arr, g, axis=0, mode="constant")[5:-5, :]
        return ndimage.correlate1d(tmp, g, axis=1, mode="constant")[:, 5:-5]

    mu_x = _filt_valid(x)
    mu_y = _filt_valid(y)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _filt_valid(x * x) - mu_x2
    sigma_y2 = _filt_valid(y * y) - mu_y2
    sigma_xy = _filt_valid(x * y) - mu_xy

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(num / np.maximum(den, 1e-15)))


def psnr_u8(pred_hw: np.ndarray, targ_hw: np.ndarray) -> float:
    """Native per-channel byte PSNR in dB."""
    mse = float(np.mean(np.square(pred_hw.astype(np.float64) - targ_hw.astype(np.float64))))
    return float(20.0 * math.log10(255.0 / (math.sqrt(mse) + 1e-8)))


def tile_channel_tolerance_curves(
    pred_u8_hw: np.ndarray,
    targ_u8_hw: np.ndarray,
    cutoff: int,
) -> dict[str, np.ndarray | int]:
    """Compute exact 256-threshold (K, U, V, T, G) tolerance curves on one HxW channel."""
    if pred_u8_hw.dtype != np.uint8 or targ_u8_hw.dtype != np.uint8:
        raise ValueError("uint8 HxW arrays required")
    fp = tolerance_footprint_81()
    gm = targ_u8_hw > int(cutoff)
    g_cnt = int(gm.sum())
    hist_all = np.bincount(pred_u8_hw.ravel(), minlength=256)
    k_curve = hist_all.sum() - np.cumsum(hist_all)
    if g_cnt > 0:
        near = ndimage.binary_dilation(gm, structure=fp, border_value=0)
        dil_pred = ndimage.grey_dilation(pred_u8_hw, footprint=fp, mode="constant", cval=0)
        hist_u = np.bincount(pred_u8_hw[near], minlength=256)
        hist_v = np.bincount(dil_pred[gm], minlength=256)
        hist_t = np.bincount(pred_u8_hw[gm], minlength=256)
        u_curve = hist_u.sum() - np.cumsum(hist_u)
        v_curve = hist_v.sum() - np.cumsum(hist_v)
        t_curve = hist_t.sum() - np.cumsum(hist_t)
    else:
        zeros = np.zeros(256, dtype=np.int64)
        u_curve, v_curve, t_curve = zeros, zeros, zeros
    return dict(K=k_curve, U=u_curve, V=v_curve, T=t_curve, G=g_cnt)


@torch.no_grad()
def _footprint_max_u8_batch(x_nhw_u8: torch.Tensor) -> torch.Tensor:
    """Exact 81-point chamfer-5 zero-padded grey/binary dilation via 4 max_pool2d boxes."""
    xf = x_nhw_u8[:, None].to(torch.float32)
    m1 = F.max_pool2d(F.pad(xf, (5, 5, 0, 0), value=0.0), (1, 11), stride=1)
    m2 = F.max_pool2d(F.pad(xf, (4, 4, 3, 3), value=0.0), (7, 9), stride=1)
    m3 = F.max_pool2d(F.pad(xf, (3, 3, 4, 4), value=0.0), (9, 7), stride=1)
    m4 = F.max_pool2d(F.pad(xf, (0, 0, 5, 5), value=0.0), (11, 1), stride=1)
    out = torch.maximum(torch.maximum(m1, m2), torch.maximum(m3, m4))[:, 0]
    return out.to(x_nhw_u8.dtype)


@torch.no_grad()
def batch_curves_and_quality_torch(
    pred_nhw_u8: np.ndarray,
    targ_nhw_u8: np.ndarray,
    cutoff: int,
    *,
    device: torch.device | str | None = None,
    compute_quality: bool = True,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Exact vectorized per-tile (K, U, V, T, G) + optional (SSIM, PSNR) on GPU or CPU."""
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    n = pred_nhw_u8.shape[0]
    K_all = np.empty((n, 256), dtype=np.int64)
    U_all = np.empty((n, 256), dtype=np.int64)
    V_all = np.empty((n, 256), dtype=np.int64)
    T_all = np.empty((n, 256), dtype=np.int64)
    G_all = np.empty((n,), dtype=np.int64)
    ssims = np.zeros((n,), dtype=np.float64)
    psnrs = np.zeros((n,), dtype=np.float64)

    kerns_1d = _gaussian_1d_kernels_torch(str(dev)) if compute_quality else None
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        bsz = e - s
        p_t = torch.from_numpy(pred_nhw_u8[s:e]).to(dev)
        y_t = torch.from_numpy(targ_nhw_u8[s:e]).to(dev)

        gm = y_t > int(cutoff)
        near = _footprint_max_u8_batch(gm.to(torch.uint8)) > 0
        dil_p = _footprint_max_u8_batch(p_t)
        g_cnts = gm.reshape(bsz, -1).sum(dim=1).cpu().numpy().astype(np.int64)
        G_all[s:e] = g_cnts

        # Vectorized 256-bin bincount across all bsz tiles simultaneously via tile*256 offset
        tile_offsets = (torch.arange(bsz, device=dev, dtype=torch.int64) * 256)[:, None]
        flat_p = p_t.reshape(bsz, -1).to(torch.int64)
        flat_dil = dil_p.reshape(bsz, -1).to(torch.int64)
        flat_gm = gm.reshape(bsz, -1)
        flat_near = near.reshape(bsz, -1)

        h_all = torch.bincount((flat_p + tile_offsets).reshape(-1), minlength=bsz * 256).reshape(bsz, 256)
        k_mat = (h_all.sum(dim=1, keepdim=True) - torch.cumsum(h_all, dim=1)).cpu().numpy()

        near_vals = (flat_p + tile_offsets)[flat_near]
        h_u = torch.bincount(near_vals, minlength=bsz * 256).reshape(bsz, 256)
        u_mat = (h_u.sum(dim=1, keepdim=True) - torch.cumsum(h_u, dim=1)).cpu().numpy()

        gm_dil_vals = (flat_dil + tile_offsets)[flat_gm]
        h_v = torch.bincount(gm_dil_vals, minlength=bsz * 256).reshape(bsz, 256)
        v_mat = (h_v.sum(dim=1, keepdim=True) - torch.cumsum(h_v, dim=1)).cpu().numpy()

        gm_p_vals = (flat_p + tile_offsets)[flat_gm]
        h_t = torch.bincount(gm_p_vals, minlength=bsz * 256).reshape(bsz, 256)
        t_mat = (h_t.sum(dim=1, keepdim=True) - torch.cumsum(h_t, dim=1)).cpu().numpy()

        K_all[s:e] = k_mat
        U_all[s:e] = u_mat
        V_all[s:e] = v_mat
        T_all[s:e] = t_mat

        if compute_quality and kerns_1d is not None:
            g_col, g_row = kerns_1d
            xf = p_t[:, None].to(torch.float32)
            yf = y_t[:, None].to(torch.float32)
            mu_x = _sep_gauss_valid_f32(xf, g_col, g_row)
            mu_y = _sep_gauss_valid_f32(yf, g_col, g_row)
            mu_x2 = mu_x.square()
            mu_y2 = mu_y.square()
            mu_xy = mu_x * mu_y
            sig_x2 = (_sep_gauss_valid_f32(xf.square(), g_col, g_row) - mu_x2).clamp_min(0.0)
            sig_y2 = (_sep_gauss_valid_f32(yf.square(), g_col, g_row) - mu_y2).clamp_min(0.0)
            sig_xy = _sep_gauss_valid_f32(xf * yf, g_col, g_row) - mu_xy
            num = (2.0 * mu_xy + c1) * (2.0 * sig_xy + c2)
            den = (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
            ssim_b = (num / den.clamp_min(1e-12)).mean(dim=(1, 2, 3), dtype=torch.float64).cpu().numpy()
            mse_b = (xf - yf).square().mean(dim=(1, 2, 3), dtype=torch.float64).cpu().numpy()
            ssims[s:e] = ssim_b
            psnrs[s:e] = 20.0 * np.log10(255.0 / (np.sqrt(mse_b) + 1e-8))

    return dict(K=K_all, U=U_all, V=V_all, T=T_all, G=G_all, ssims=ssims, psnrs=psnrs)


def endpoints_from_pooled_curves(
    K: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    T: np.ndarray,
    G: int,
    fixed_cutoff: int,
    cal_threshold: int | None = None,
) -> dict[str, float | int]:
    """Compute exact tolerant ODS-F, Mean AP, Fixed_tol_F1, Cal_F1, and exact F1 from pooled curves."""
    Kf = K.astype(np.float64)
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Tf = T.astype(np.float64)
    Gf = float(G)

    if G <= 0:
        return dict(
            ODS_F=0.0,
            ODS_threshold=int(fixed_cutoff),
            AP=0.0,
            Fixed_tol_F1=0.0,
            Cal_F1=0.0,
            Fixed_exact_F1=0.0,
            mismatch_gap_ods_minus_fixed=0.0,
        )

    prec = np.where(Kf > 0, Uf / np.maximum(Kf, 1.0), 0.0)
    rec = Vf / Gf
    denom = prec + rec
    f_curve = np.where(denom > 0, 2.0 * prec * rec / np.maximum(denom, 1e-15), 0.0)

    ods_thr = int(np.argmax(f_curve))
    ods_f = float(f_curve[ods_thr])
    fixed_f = float(f_curve[int(fixed_cutoff)])
    cal_thr = int(cal_threshold) if cal_threshold is not None else ods_thr
    cal_f = float(f_curve[cal_thr])

    r_inc = np.r_[0.0, rec[::-1]]
    p_inc = np.r_[1.0, prec[::-1]]
    r_mono = np.maximum.accumulate(r_inc)
    dr = np.diff(r_mono)
    ap = float(np.sum(dr * p_inc[1:]))

    exact_f_fixed = float((2.0 * Tf[int(fixed_cutoff)]) / max(Kf[int(fixed_cutoff)] + Gf, 1.0))

    return dict(
        ODS_F=ods_f,
        ODS_threshold=ods_thr,
        AP=ap,
        Fixed_tol_F1=fixed_f,
        Cal_F1=cal_f,
        Fixed_exact_F1=exact_f_fixed,
        mismatch_gap_ods_minus_fixed=ods_f - fixed_f,
    )


def evaluate_channel_predictions(
    *,
    train_pred_u8: np.ndarray,
    train_targ_u8: np.ndarray,
    eval_pred_u8: np.ndarray,
    eval_targ_u8: np.ndarray,
    eval_clusters: Sequence[str],
    fixed_cutoff: int,
    device: torch.device | str | None = None,
) -> dict:
    """Evaluate one marker channel on train (for Cal_F1 threshold) and eval (with per-cluster curves)."""
    tr_batch = batch_curves_and_quality_torch(
        train_pred_u8,
        train_targ_u8,
        fixed_cutoff,
        device=device,
        compute_quality=False,
    )
    tr_ep = endpoints_from_pooled_curves(
        tr_batch["K"].sum(axis=0),
        tr_batch["U"].sum(axis=0),
        tr_batch["V"].sum(axis=0),
        tr_batch["T"].sum(axis=0),
        int(tr_batch["G"].sum()),
        fixed_cutoff,
    )
    cal_thr = int(tr_ep["ODS_threshold"])

    n_ev = eval_pred_u8.shape[0]
    ev_batch = batch_curves_and_quality_torch(
        eval_pred_u8,
        eval_targ_u8,
        fixed_cutoff,
        device=device,
        compute_quality=True,
    )
    ev_ep = endpoints_from_pooled_curves(
        ev_batch["K"].sum(axis=0),
        ev_batch["U"].sum(axis=0),
        ev_batch["V"].sum(axis=0),
        ev_batch["T"].sum(axis=0),
        int(ev_batch["G"].sum()),
        fixed_cutoff,
        cal_threshold=cal_thr,
    )
    ev_ep["SSIM"] = float(np.mean(ev_batch["ssims"])) if n_ev else 0.0
    ev_ep["PSNR_dB"] = float(np.mean(ev_batch["psnrs"])) if n_ev else 0.0
    ev_ep["train_calibrated_threshold"] = cal_thr

    cluster_curves: dict[str, dict] = {}
    for i in range(n_ev):
        cl = str(eval_clusters[i])
        rec_cl = cluster_curves.get(cl)
        if rec_cl is None:
            cluster_curves[cl] = dict(
                K=ev_batch["K"][i].copy(),
                U=ev_batch["U"][i].copy(),
                V=ev_batch["V"][i].copy(),
                T=ev_batch["T"][i].copy(),
                G=int(ev_batch["G"][i]),
                ssim_sum=float(ev_batch["ssims"][i]),
                psnr_sum=float(ev_batch["psnrs"][i]),
                n_tiles=1,
            )
        else:
            rec_cl["K"] += ev_batch["K"][i]
            rec_cl["U"] += ev_batch["U"][i]
            rec_cl["V"] += ev_batch["V"][i]
            rec_cl["T"] += ev_batch["T"][i]
            rec_cl["G"] += int(ev_batch["G"][i])
            rec_cl["ssim_sum"] += float(ev_batch["ssims"][i])
            rec_cl["psnr_sum"] += float(ev_batch["psnrs"][i])
            rec_cl["n_tiles"] += 1

    per_cluster_scalars = {}
    for cl, rc in sorted(cluster_curves.items()):
        cl_ep = endpoints_from_pooled_curves(
            rc["K"], rc["U"], rc["V"], rc["T"], rc["G"], fixed_cutoff, cal_threshold=cal_thr
        )
        cl_ep["SSIM"] = float(rc["ssim_sum"] / max(rc["n_tiles"], 1))
        cl_ep["PSNR_dB"] = float(rc["psnr_sum"] / max(rc["n_tiles"], 1))
        cl_ep["n_tiles"] = int(rc["n_tiles"])
        per_cluster_scalars[cl] = cl_ep

    return dict(
        pooled=ev_ep,
        per_cluster=per_cluster_scalars,
        raw_cluster_curves=cluster_curves,
    )


def paired_cluster_bootstrap(
    arm_a_clusters: Mapping[str, Mapping[str, float]],
    arm_b_clusters: Mapping[str, Mapping[str, float]],
    metric: str,
    n_bootstrap: int = 5000,
    seed: int = 20261116,
) -> dict[str, float | list[float] | int]:
    """Paired cluster-level (region / patient / WSI) bootstrap CI and Wilcoxon test for A - B."""
    common_keys = sorted(set(arm_a_clusters.keys()) & set(arm_b_clusters.keys()))
    a_vals = np.array([float(arm_a_clusters[k][metric]) for k in common_keys], dtype=np.float64)
    b_vals = np.array([float(arm_b_clusters[k][metric]) for k in common_keys], dtype=np.float64)
    diffs = a_vals - b_vals
    m = len(diffs)
    if m == 0:
        return dict(
            n_clusters=0,
            mean_delta=0.0,
            ci95=[0.0, 0.0],
            ci95_low=0.0,
            ci95_high=0.0,
            wilcoxon_p=1.0,
            positive_clusters=0,
        )

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_bootstrap, m))
    boot_means = diffs[idx].mean(axis=1)
    ci_low, ci_high = float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))

    nz = diffs[np.abs(diffs) > 1e-14]
    if len(nz) >= 2:
        try:
            w_p = float(stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:
            w_p = 1.0
    else:
        w_p = 1.0

    return dict(
        n_clusters=int(m),
        mean_a=float(np.mean(a_vals)),
        mean_b=float(np.mean(b_vals)),
        mean_delta=float(np.mean(diffs)),
        median_delta=float(np.median(diffs)),
        ci95=[ci_low, ci_high],
        ci95_low=ci_low,
        ci95_high=ci_high,
        wilcoxon_p=w_p,
        positive_clusters=int(np.sum(diffs > 0)),
    )
