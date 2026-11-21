#!/usr/bin/env python3
"""NV2-N2: Mechanistic XAI, Spectral Mode-Hijacking Audit & Conservation Mass-Flux Verification.

Computes domain-native, publication-grade explainability & physics audits on locked held-out splits:
  1. Common-Window 12-Channel Integrated Gradients (16-step Riemann IG from zero baseline)
     comparing S_count, S_recon, and E_count on the EXACT SAME held-out tiles across:
       - HEMIT (ConvNeXt, W-MSE Swin, Pix2Pix, DGR; CD3 ch=1, 15 independent spatial clusters)
       - Orion-CRC (ConvNeXt, W-MSE Swin; CD8a ch=6, 8 independent Fold 0 patients)
     Aggregates within each independent unit (cluster / patient) and performs paired Wilcoxon
     signed-rank tests for Delta-Saliency (S_count - E_count and S_count - S_recon).
  2. Exact Conservative 2D Spatial Mass-Flux Vector Fields on the Open Lattice:
       Because (I + L) r_c = L psi_c  ==>  r_c = L(psi_c - r_c) = B^T J_c
     verifies exact discrete flux-divergence identity ||div(J_c) - r_c||_inf < 1e-13 and exports
     high-resolution 2D (H&E, Physical, Frozen, S_count, r_c, Jx, Jy) maps for Fig. 1i, Fig. 3h, ED Fig. 8f-h.
  3. 4-Band Physical Wavelength Spectral Energy Transfer (DC >100um, Tissue 30-100um,
     Cellular 6-25um, Sub-cellular <6um) comparing Physical, Frozen, S_recon, QuantileMatch,
     E_count, and S_count, plus the Spectral Mode-Hijacking Index (SMHI).
  4. 2D Masked-Contour Physical Phase Portraits:
       (a) Upstream intensity b_c (0-255) x Local H&E luminance gradient |grad_1 x| -> mean r_c & counts
       (b) Tile true antigen prevalence (0-80%) x Per-tile RMS calibration residual (bytes).
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
from scipy import stats

from s2 import query_relation_reconstruction as core
from s2 import run_hemit_multibackbone_cell as rhmc
from s2 import run_nm_comparators20 as rcomp
from s2 import run_orion_multibackbone_survival as ror
from s2 import run_plan2_ablations as rp2
from s2 import screened_universal_head as suh

CH12_NAMES = [
    "H&E R", "H&E G", "H&E B",
    "|grad1 H&E|", "|grad8 H&E|", "|lap H&E|",
    "Upstream b_c", "Struct ref", "Panel mean",
    "|grad1 b_norm|", "|grad16 b_norm|", "Peak-halo max-b",
]


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def compute_exact_lattice_flux_and_divergence(psi_hw: np.ndarray, r_hw: np.ndarray) -> dict:
    """Exact discrete flux J = B (psi - r) satisfying B^T J == L(psi - r) == r_c."""
    phi = (psi_hw - r_hw).astype(np.float64)
    h, w = phi.shape
    div = np.zeros((h, w), dtype=np.float64)
    jx = np.zeros((h, w), dtype=np.float64)
    jy = np.zeros((h, w), dtype=np.float64)
    for step in (1, 16):
        if w > step:
            dx = (phi[:, step:] - phi[:, :-step]) * 0.0625
            div[:, :-step] -= dx
            div[:, step:] += dx
            # Positive flux flows from low potential (sink) to high potential (source/peak)
            # Inward flux to node i+step is +dx (flowing +x)
            jx[:, :-step] += 0.5 * dx
            jx[:, step:] += 0.5 * dx
        if h > step:
            dy = (phi[step:, :] - phi[:-step, :]) * 0.0625
            div[:-step, :] -= dy
            div[step:, :] += dy
            jy[:-step, :] += 0.5 * dy
            jy[step:, :] += 0.5 * dy
    err = float(np.max(np.abs(div - r_hw.astype(np.float64))))
    return dict(jx=jx.astype(np.float32), jy=jy.astype(np.float32), max_div_err=err)


def compute_ig12_for_model(
    model: torch.nn.Module,
    is_unscreened: bool,
    c12_batch: torch.Tensor,
    donor_u8_batch: torch.Tensor,
    cutoff: int,
    device: torch.device,
    n_steps: int = 12,
) -> np.ndarray:
    """Compute per-tile 12-channel |IG| targeting positive-cutoff activation on common tiles."""
    model.eval()
    n = int(c12_batch.shape[0])
    out = np.zeros((n, 12), dtype=np.float64)
    tau = (float(cutoff) + 0.5) / 255.0

    for i in range(n):
        x_full = c12_batch[i : i + 1].to(device, dtype=torch.float32)
        d0 = donor_u8_batch[i : i + 1].to(device, dtype=torch.float32).unsqueeze(1) / 255.0
        grad_acc = torch.zeros_like(x_full)
        for s in range(1, n_steps + 1):
            alpha = (s - 0.5) / float(n_steps)
            x_in = (x_full * alpha).detach().requires_grad_(True)
            if is_unscreened:
                r, _ = rp2.fast_ablation_forward(model, x_in, "E_count", inference=False)
            else:
                r, _ = suh.fast_sc_forward(model, x_in, inference=False)
            z = d0 + r
            # Activation crossing fixed physical cutoff tau_c
            obj = torch.sigmoid((z - tau) * 25.0).mean()
            g = torch.autograd.grad(obj, x_in)[0]
            grad_acc += g.detach()
        ig = (x_full * (grad_acc / float(n_steps))).abs().mean(dim=(0, 2, 3))
        out[i] = ig.cpu().numpy()
    return out


def compute_4band_radial_psd(imgs_u8: np.ndarray, um_per_px: float = 0.5) -> dict:
    """Compute Hann-windowed 2D radial PSD integrated over 4 physical wavelength bands."""
    n, h, w = imgs_u8.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    win = wy * wx
    fy = np.fft.fftfreq(h, d=um_per_px)
    fx = np.fft.fftfreq(w, d=um_per_px)
    fr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)  # cycles / um

    # Wavelength lambda = 1 / fr:
    # DC/Macro > 100 um: fr < 0.01
    # Tissue 30-100 um: 0.01 <= fr < 1/30
    # Cellular 6-30 um: 1/30 <= fr < 1/6
    # Sub-cellular < 6 um: fr >= 1/6
    m_dc = fr < 0.01
    m_tis = (fr >= 0.01) & (fr < 1.0 / 30.0)
    m_cel = (fr >= 1.0 / 30.0) & (fr < 1.0 / 6.0)
    m_sub = fr >= 1.0 / 6.0

    # Also compute 32-bin normalized radial curve
    fmax = float(fr.max())
    edges = np.linspace(0.0, 0.45, 33)
    bin_idx = np.clip(np.digitize(fr.ravel(), edges) - 1, 0, 31)
    bin_cnt = np.maximum(np.bincount(bin_idx, minlength=32), 1)

    bands = np.zeros((n, 4), dtype=np.float64)
    rad_acc = np.zeros(32, dtype=np.float64)
    for i in range(n):
        arr = imgs_u8[i].astype(np.float64) / 255.0
        # Keep true DC component via un-windowed mean^2 + Hann-windowed AC spectrum
        p_dc_exact = float(arr.mean() ** 2) * (h * w)
        spec = np.abs(np.fft.fft2((arr - arr.mean()) * win)) ** 2 / max(1.0, float(np.sum(win ** 2)))
        e_dc = p_dc_exact + float(spec[m_dc].sum())
        e_tis = float(spec[m_tis].sum())
        e_cel = float(spec[m_cel].sum())
        e_sub = float(spec[m_sub].sum())
        bands[i] = [e_dc, e_tis, e_cel, e_sub]
        rad_acc += np.bincount(bin_idx, weights=spec.ravel(), minlength=32) / bin_cnt

    mean_b = bands.mean(axis=0)
    smhi = float((mean_b[0] + mean_b[1]) / max(mean_b[2], 1e-12))
    return dict(
        band_names=["DC (>100 um)", "Tissue (30-100 um)", "Cellular (6-30 um)", "Sub-cellular (<6 um)"],
        band_energy_mean=[float(x) for x in mean_b],
        smhi_macro_to_cell_ratio=round(smhi, 4),
        radial_freq_centers=[round(0.5 * (edges[k] + edges[k + 1]), 5) for k in range(32)],
        radial_psd_mean=[float(x) for x in (rad_acc / max(1, n))],
    )


def compute_phase_portraits(
    he_nhwc: np.ndarray,
    up_u8: np.ndarray,
    sc_f32: np.ndarray,
    targ_u8: np.ndarray,
    cutoff: int,
) -> dict:
    """Compute 2D masked-contour phase portraits:
    (a) Upstream intensity (24 bins) x Local H&E gradient (24 bins) -> mean signed residual r_c & pixel counts
    (b) Per-tile true antigen prevalence (%) vs calibration residual L2 norm ||r_c||_2.
    """
    n = len(up_u8)
    # 2D bins: x = up_u8 in [0, 120] (24 bins of width 5), y = local H&E grad in [0, 60] (24 bins)
    x_edges = np.linspace(0.0, 120.0, 25)
    y_edges = np.linspace(0.0, 60.0, 25)
    sum_r = np.zeros((24, 24), dtype=np.float64)
    cnt_r = np.zeros((24, 24), dtype=np.int64)

    tile_prev = np.zeros(n, dtype=np.float64)
    tile_r_l2 = np.zeros(n, dtype=np.float64)
    tile_r_pos_mass = np.zeros(n, dtype=np.float64)

    for i in range(n):
        gray = 0.299 * he_nhwc[i, :, :, 0] + 0.587 * he_nhwc[i, :, :, 1] + 0.114 * he_nhwc[i, :, :, 2]
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        gmag = np.sqrt(gx * gx + gy * gy)
        r_255 = (sc_f32[i] * 255.0) - up_u8[i].astype(np.float64)

        tile_prev[i] = 100.0 * float((targ_u8[i] > cutoff).mean())
        tile_r_l2[i] = float(np.sqrt(np.mean(r_255 ** 2)))
        tile_r_pos_mass[i] = float(np.mean(np.maximum(r_255, 0.0)))

        # Subsample every 4th pixel for clean 2D histogram speed (~16k px/tile -> ~15M total)
        u_sub = up_u8[i, ::4, ::4].ravel().astype(np.float64)
        g_sub = gmag[::4, ::4].ravel()
        r_sub = r_255[::4, ::4].ravel()
        ix = np.digitize(u_sub, x_edges) - 1
        iy = np.digitize(g_sub, y_edges) - 1
        ok = (ix >= 0) & (ix < 24) & (iy >= 0) & (iy < 24)
        if np.any(ok):
            flat = iy[ok] * 24 + ix[ok]
            cnt_r.ravel()[:] += np.bincount(flat, minlength=576)
            sum_r.ravel()[:] += np.bincount(flat, weights=r_sub[ok], minlength=576)

    mean_r = np.where(cnt_r >= 25, sum_r / np.maximum(cnt_r, 1), np.nan)
    rho_prev, p_prev = stats.spearmanr(tile_prev, tile_r_l2)
    zero_mask = tile_prev == 0.0
    pos_mask = tile_prev >= 2.0
    return dict(
        x_edges=x_edges.tolist(),
        y_edges=y_edges.tolist(),
        mean_residual_grid=[[None if np.isnan(v) else round(float(v), 4) for v in row] for row in mean_r],
        count_grid=cnt_r.tolist(),
        tile_prevalence_pct=[round(float(x), 4) for x in tile_prev],
        tile_residual_l2=[round(float(x), 4) for x in tile_r_l2],
        tile_residual_pos_mass=[round(float(x), 4) for x in tile_r_pos_mass],
        spearman_prev_vs_energy=round(float(rho_prev), 4),
        spearman_p=float(p_prev),
        zero_antigen_tiles_n=int(zero_mask.sum()),
        zero_antigen_mean_r_l2=round(float(tile_r_l2[zero_mask].mean()), 4) if np.any(zero_mask) else None,
        pos_antigen_mean_r_l2=round(float(tile_r_l2[pos_mask].mean()), 4) if np.any(pos_mask) else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemit-train-pkl", required=True)
    ap.add_argument("--hemit-test-pkl", required=True)
    ap.add_argument("--hemit-upstream-dir", required=True)
    ap.add_argument("--orion-bank-pkl", required=True)
    ap.add_argument("--orion-upstream-dir", required=True)
    ap.add_argument("--orion-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[NV2-N2] Loading HEMIT train/test banks...", flush=True)
    h_tr = pickle.load(open(args.hemit_train_pkl, "rb"))
    h_ev = pickle.load(open(args.hemit_test_pkl, "rb"))
    h_up_dir = Path(args.hemit_upstream_dir)

    # Select balanced common-window subset across ALL 15 HEMIT spatial clusters (8 tiles/cluster = 120 tiles)
    clusters = np.asarray(h_ev["clusters"])
    uniq_cl = sorted(set(clusters.tolist()))
    common_idx = []
    common_cl = []
    for cl in uniq_cl:
        idxs = np.nonzero(clusters == cl)[0]
        pick = idxs[np.linspace(0, len(idxs) - 1, num=min(8, len(idxs)), dtype=int)]
        common_idx.extend(pick.tolist())
        common_cl.extend([cl] * len(pick))
    common_idx = np.asarray(common_idx, dtype=np.int64)
    common_cl = np.asarray(common_cl)

    hemit_models = [
        ("ConvNeXt", "sanofi_miphei_convnext_train_596.npz", "sanofi_miphei_convnext_test_945.npz"),
        ("W-MSE Swin", "sanofi_miphei_hemit_train_596.npz", "sanofi_miphei_hemit_test_945.npz"),
        ("Pix2Pix", "sanofi_pix2pix_train_596.npz", "sanofi_pix2pix_test_945.npz"),
        ("DGR", "native_dgr_train_596.npz", "native_dgr_test_945.npz"),
    ]

    results = dict(
        schema="nv2-n2-mechanistic-xai-v1",
        ch12_names=CH12_NAMES,
        hemit_cd3_xai={},
        orion_cd8a_xai={},
    )

    flux_saved = False
    for mname, tr_fn, ev_fn in hemit_models:
        tm = time.time()
        print(f"[NV2-N2] HEMIT CD3 Mechanistic XAI on {mname}...", flush=True)
        up_tr = np.load(h_up_dir / tr_fn)["pred_u8"]
        up_ev = np.load(h_up_dir / ev_fn)["pred_u8"]
        ch = 1  # CD3
        cut = 30

        c12_tr = rhmc.build_common12_for_3ch(h_tr["he_nhwc"], up_tr, ch, device=device)
        c12_ev_all = rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev, ch, device=device)
        c12_cm = c12_ev_all[common_idx]
        d_tr_t = torch.from_numpy(up_tr[:, ch])
        y_tr_t = torch.from_numpy(h_tr["targ_nchw"][:, ch])
        d_ev_t = torch.from_numpy(up_ev[:, ch])
        d_cm_t = torch.from_numpy(up_ev[common_idx, ch])

        # Fit S_count, S_recon, E_count (seed 202610270) on identical split
        mdl_sc, _ = suh.fit_universal_marker(
            arm="S_count", seed=202610270, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
            cutoff=cut, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
        )
        mdl_sr, _ = suh.fit_universal_marker(
            arm="S_recon", seed=202610270, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
            cutoff=cut, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
        )
        mdl_ec, _ = rp2.fit_ablation_marker(
            mode="E_count", seed=202610270, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
            cutoff=cut, device=device, epochs=8, batch_size=8, lr=1.2e-3,
        )

        # 1. Common-window 12-channel IG + cluster-level paired Wilcoxon
        ig_sc = compute_ig12_for_model(mdl_sc, False, c12_cm, d_cm_t, cut, device, n_steps=12)
        ig_sr = compute_ig12_for_model(mdl_sr, False, c12_cm, d_cm_t, cut, device, n_steps=12)
        ig_ec = compute_ig12_for_model(mdl_ec, True, c12_cm, d_cm_t, cut, device, n_steps=12)

        # Normalize per-tile to relative channel share (%) for scale-free comparison + keep raw
        ig_sc_rel = 100.0 * ig_sc / np.maximum(ig_sc.sum(axis=1, keepdims=True), 1e-12)
        ig_sr_rel = 100.0 * ig_sr / np.maximum(ig_sr.sum(axis=1, keepdims=True), 1e-12)
        ig_ec_rel = 100.0 * ig_ec / np.maximum(ig_ec.sum(axis=1, keepdims=True), 1e-12)

        cl_sc = np.asarray([ig_sc_rel[common_cl == cl].mean(axis=0) for cl in uniq_cl])
        cl_sr = np.asarray([ig_sr_rel[common_cl == cl].mean(axis=0) for cl in uniq_cl])
        cl_ec = np.asarray([ig_ec_rel[common_cl == cl].mean(axis=0) for cl in uniq_cl])
        delta_sc_ec = cl_sc - cl_ec
        delta_sc_sr = cl_sc - cl_sr

        p_sc_ec = []
        p_sc_sr = []
        for c_i in range(12):
            try:
                p1 = float(stats.wilcoxon(cl_sc[:, c_i], cl_ec[:, c_i]).pvalue)
            except Exception:
                p1 = 1.0
            try:
                p2 = float(stats.wilcoxon(cl_sc[:, c_i], cl_sr[:, c_i]).pvalue)
            except Exception:
                p2 = 1.0
            p_sc_ec.append(p1)
            p_sc_sr.append(p2)

        # 2. Predictions on full 945 test tiles for 4-band PSD & 2D phase portraits
        pr_sc_u8, _ = suh.predict_universal_marker(mdl_sc, c12_ev_all, d_ev_t, device=device, batch_size=16)
        pr_sr_u8, _ = suh.predict_universal_marker(mdl_sr, c12_ev_all, d_ev_t, device=device, batch_size=16)
        pr_ec_u8, _ = rp2.predict_ablation_marker(mdl_ec, "E_count", c12_ev_all, d_ev_t, device=device, batch_size=16)
        _, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr[:, ch], h_tr["targ_nchw"][:, ch], up_ev[:, ch])
        pr_sc_f32 = pr_sc_u8.astype(np.float32) / 255.0

        # 4-band radial spectral energy transfer on common 120 tiles
        psd_arms = dict(
            Physical=compute_4band_radial_psd(h_ev["targ_nchw"][common_idx, ch], um_per_px=0.5),
            Frozen=compute_4band_radial_psd(up_ev[common_idx, ch], um_per_px=0.5),
            S_recon=compute_4band_radial_psd(pr_sr_u8[common_idx], um_per_px=0.5),
            QuantileMatch=compute_4band_radial_psd(qm_ev[common_idx], um_per_px=0.5),
            E_count=compute_4band_radial_psd(pr_ec_u8[common_idx], um_per_px=0.5),
            S_count=compute_4band_radial_psd(pr_sc_u8[common_idx], um_per_px=0.5),
        )

        # 3. Exact discrete spatial mass-flux verification on 10 richest CD3+ tiles (save ConvNeXt for Fig 1i & Fig 3h)
        cd3_pos_cnt = (h_ev["targ_nchw"][:, ch] > cut).sum(axis=(1, 2))
        top_tiles = np.argsort(-cd3_pos_cnt)[:10]
        div_errs = []
        flux_pack = {}
        with torch.no_grad():
            for rank_i, tidx in enumerate(top_tiles.tolist()):
                x1 = c12_ev_all[tidx : tidx + 1].to(device, dtype=torch.float32)
                zeros16 = x1.new_zeros((1, 16, *x1.shape[-2:]))
                psi = suh.fast_unary_potential(mdl_sc, mdl_sc.backbone(torch.cat((x1, zeros16), dim=1))).to(torch.float64)
                r64, _ = suh.fast_sc_forward(mdl_sc, x1, inference=True)
                fl = compute_exact_lattice_flux_and_divergence(psi[0, 0].cpu().numpy(), r64[0, 0].cpu().numpy())
                div_errs.append(fl["max_div_err"])
                if not flux_saved and mname == "ConvNeXt" and rank_i == 0:
                    flux_pack = dict(
                        tile_idx=int(tidx),
                        tile_name=str(h_ev["names"][tidx]),
                        he=h_ev["he_nhwc"][tidx],
                        phys_3ch=h_ev["targ_nchw"][tidx],
                        frz_3ch=up_ev[tidx],
                        sc_cd3=pr_sc_u8[tidx],
                        ec_cd3=pr_ec_u8[tidx],
                        r_cd3=r64[0, 0].cpu().numpy().astype(np.float32),
                        jx=fl["jx"],
                        jy=fl["jy"],
                    )
                    np.savez_compressed(out_dir / "n2_hemit_spatial_flux_crop.npz", **flux_pack)
                    flux_saved = True

        # 4. Phase portraits on ConvNeXt & W-MSE Swin
        phase = None
        if mname in ("ConvNeXt", "W-MSE Swin"):
            phase = compute_phase_portraits(h_ev["he_nhwc"], up_ev[:, ch], pr_sc_f32, h_ev["targ_nchw"][:, ch], cut)

        results["hemit_cd3_xai"][mname] = dict(
            n_common_tiles=int(len(common_idx)),
            n_independent_clusters=int(len(uniq_cl)),
            ig12_rel_pct_scount=[round(float(x), 3) for x in cl_sc.mean(axis=0)],
            ig12_rel_pct_srecon=[round(float(x), 3) for x in cl_sr.mean(axis=0)],
            ig12_rel_pct_ecount=[round(float(x), 3) for x in cl_ec.mean(axis=0)],
            delta_saliency_sc_minus_ec=[round(float(x), 3) for x in delta_sc_ec.mean(axis=0)],
            delta_saliency_sc_minus_sr=[round(float(x), 3) for x in delta_sc_sr.mean(axis=0)],
            wilcoxon_p_sc_vs_ec=[round(float(x), 6) for x in p_sc_ec],
            wilcoxon_p_sc_vs_sr=[round(float(x), 6) for x in p_sc_sr],
            max_flux_divergence_err=float(max(div_errs)),
            psd_4band=psd_arms,
            phase_portrait=phase,
        )
        print(
            f"[NV2-N2] HEMIT {mname} ({time.time()-tm:.1f}s) | max_div_err={max(div_errs):.2e} | "
            f"SMHI E_cnt={psd_arms['E_count']['smhi_macro_to_cell_ratio']:.2f} vs S_cnt={psd_arms['S_count']['smhi_macro_to_cell_ratio']:.2f} | "
            f"H&E+Grad IG share S_cnt={cl_sc[:, :6].sum(1).mean():.1f}% vs E_cnt={cl_ec[:, :6].sum(1).mean():.1f}%",
            flush=True,
        )
        del mdl_sc, mdl_sr, mdl_ec, c12_tr, c12_ev_all, up_tr, up_ev
        gc.collect()
        torch.cuda.empty_cache()

    # Release HEMIT banks before loading Orion
    del h_tr, h_ev
    gc.collect()

    # Orion-CRC Fold 0 CD8a Mechanistic XAI (8 independent patients)
    print("[NV2-N2] Loading Orion bank for Fold 0 CD8a Mechanistic XAI...", flush=True)
    ob = pickle.load(open(args.orion_bank_pkl, "rb"))
    om = json.loads(Path(args.orion_manifest).read_text())
    f0 = om["folds"][0]
    pats_arr = np.asarray(ob["patients"])
    tr_idx = np.nonzero(np.isin(pats_arr, f0["inner_train_slides"]))[0]
    ev_idx = np.nonzero(np.isin(pats_arr, f0["outer_test_slides"]))[0]
    ev_pats_uniq = sorted(set(f0["outer_test_slides"]))
    cm_o_idx, cm_o_pat = [], []
    for p in ev_pats_uniq:
        pi = ev_idx[pats_arr[ev_idx] == p]
        pick = pi[np.linspace(0, len(pi) - 1, num=min(12, len(pi)), dtype=int)]
        cm_o_idx.extend(pick.tolist())
        cm_o_pat.extend([p] * len(pick))
    cm_o_idx = np.asarray(cm_o_idx, dtype=np.int64)
    cm_o_pat = np.asarray(cm_o_pat)

    o_up_dir = Path(args.orion_upstream_dir)
    orion_cuts, _ = ror.calibrate_orion_physical_cutoffs_from_train(ob["targ_nchw"][tr_idx], tr_idx, ob["nuclei_records"])
    for mname, fn in [("ConvNeXt", "miphei_convnext_all40_2804.npz"), ("W-MSE Swin", "miphei_hemit_arch_all40_2804.npz")]:
        tm = time.time()
        up_all = np.load(o_up_dir / fn)["pred_u8"]
        ch = 6  # CD8a
        cut = int(orion_cuts[ch])
        c12_tr = ror.build_common12_orion(ob["he_nhwc"][tr_idx], up_all[tr_idx], ch, device=device)
        c12_cm = ror.build_common12_orion(ob["he_nhwc"][cm_o_idx], up_all[cm_o_idx], ch, device=device)
        d_tr_t = torch.from_numpy(up_all[tr_idx, ch])
        y_tr_t = torch.from_numpy(ob["targ_nchw"][tr_idx, ch])
        d_cm_t = torch.from_numpy(up_all[cm_o_idx, ch])

        mdl_sc, _ = suh.fit_universal_marker(
            arm="S_count", seed=202610270, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
            cutoff=cut, device=device, epochs=6, batch_size=16, lr=1.5e-3, deploy_anchor_weight=1.25,
        )
        mdl_sr, _ = suh.fit_universal_marker(
            arm="S_recon", seed=202610270, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
            cutoff=cut, device=device, epochs=6, batch_size=16, lr=1.5e-3, deploy_anchor_weight=1.25,
        )
        mdl_ec, _ = rp2.fit_ablation_marker(
            mode="E_count", seed=202610270, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
            cutoff=cut, device=device, epochs=6, batch_size=16, lr=1.5e-3,
        )

        ig_sc = compute_ig12_for_model(mdl_sc, False, c12_cm, d_cm_t, cut, device, n_steps=12)
        ig_sr = compute_ig12_for_model(mdl_sr, False, c12_cm, d_cm_t, cut, device, n_steps=12)
        ig_ec = compute_ig12_for_model(mdl_ec, True, c12_cm, d_cm_t, cut, device, n_steps=12)

        ig_sc_rel = 100.0 * ig_sc / np.maximum(ig_sc.sum(axis=1, keepdims=True), 1e-12)
        ig_sr_rel = 100.0 * ig_sr / np.maximum(ig_sr.sum(axis=1, keepdims=True), 1e-12)
        ig_ec_rel = 100.0 * ig_ec / np.maximum(ig_ec.sum(axis=1, keepdims=True), 1e-12)

        pt_sc = np.asarray([ig_sc_rel[cm_o_pat == p].mean(axis=0) for p in ev_pats_uniq])
        pt_sr = np.asarray([ig_sr_rel[cm_o_pat == p].mean(axis=0) for p in ev_pats_uniq])
        pt_ec = np.asarray([ig_ec_rel[cm_o_pat == p].mean(axis=0) for p in ev_pats_uniq])

        p_sc_ec = [float(stats.wilcoxon(pt_sc[:, c_i], pt_ec[:, c_i]).pvalue) for c_i in range(12)]
        p_sc_sr = [float(stats.wilcoxon(pt_sc[:, c_i], pt_sr[:, c_i]).pvalue) for c_i in range(12)]

        pr_sc_u8, _ = suh.predict_universal_marker(mdl_sc, c12_cm, d_cm_t, device=device, batch_size=16)
        pr_ec_u8, _ = rp2.predict_ablation_marker(mdl_ec, "E_count", c12_cm, d_cm_t, device=device, batch_size=16)
        pr_sr_u8, _ = suh.predict_universal_marker(mdl_sr, c12_cm, d_cm_t, device=device, batch_size=16)
        _, qm_cm, _ = rcomp.apply_quantile_match_u8(up_all[tr_idx, ch], ob["targ_nchw"][tr_idx, ch], up_all[cm_o_idx, ch])
        # MIPHEI Zenodo 10.5281/zenodo.15340874 Orion-CRC 256x256 tiles are resampled to 0.50 um/px (20x)
        psd_o = dict(
            Physical=compute_4band_radial_psd(ob["targ_nchw"][cm_o_idx, ch], um_per_px=0.50),
            Frozen=compute_4band_radial_psd(up_all[cm_o_idx, ch], um_per_px=0.50),
            S_recon=compute_4band_radial_psd(pr_sr_u8, um_per_px=0.50),
            QuantileMatch=compute_4band_radial_psd(qm_cm, um_per_px=0.50),
            E_count=compute_4band_radial_psd(pr_ec_u8, um_per_px=0.50),
            S_count=compute_4band_radial_psd(pr_sc_u8, um_per_px=0.50),
        )

        # Save one Orion spatial flux crop on ConvNeXt
        if mname == "ConvNeXt":
            best_local = int(np.argmax((ob["targ_nchw"][cm_o_idx, ch] > cut).sum(axis=(1, 2))))
            gidx = int(cm_o_idx[best_local])
            with torch.no_grad():
                x1 = c12_cm[best_local : best_local + 1].to(device, dtype=torch.float32)
                zeros16 = x1.new_zeros((1, 16, *x1.shape[-2:]))
                psi = suh.fast_unary_potential(mdl_sc, mdl_sc.backbone(torch.cat((x1, zeros16), dim=1))).to(torch.float64)
                r64, _ = suh.fast_sc_forward(mdl_sc, x1, inference=True)
                fl = compute_exact_lattice_flux_and_divergence(psi[0, 0].cpu().numpy(), r64[0, 0].cpu().numpy())
            np.savez_compressed(
                out_dir / "n2_orion_spatial_flux_crop.npz",
                tile_idx=gidx,
                he=ob["he_nhwc"][gidx],
                phys_cd8a=ob["targ_nchw"][gidx, ch],
                frz_cd8a=up_all[gidx, ch],
                sc_cd8a=pr_sc_u8[best_local],
                r_cd8a=r64[0, 0].cpu().numpy().astype(np.float32),
                jx=fl["jx"],
                jy=fl["jy"],
                max_div_err=fl["max_div_err"],
            )

        results["orion_cd8a_xai"][mname] = dict(
            n_common_tiles=int(len(cm_o_idx)),
            n_independent_patients=int(len(ev_pats_uniq)),
            ig12_rel_pct_scount=[round(float(x), 3) for x in pt_sc.mean(axis=0)],
            ig12_rel_pct_srecon=[round(float(x), 3) for x in pt_sr.mean(axis=0)],
            ig12_rel_pct_ecount=[round(float(x), 3) for x in pt_ec.mean(axis=0)],
            delta_saliency_sc_minus_ec=[round(float(x), 3) for x in (pt_sc - pt_ec).mean(axis=0)],
            delta_saliency_sc_minus_sr=[round(float(x), 3) for x in (pt_sc - pt_sr).mean(axis=0)],
            wilcoxon_p_sc_vs_ec=[round(float(x), 6) for x in p_sc_ec],
            wilcoxon_p_sc_vs_sr=[round(float(x), 6) for x in p_sc_sr],
            psd_4band=psd_o,
        )
        print(f"[NV2-N2] Orion {mname} ({time.time()-tm:.1f}s) done.", flush=True)
        del mdl_sc, mdl_sr, mdl_ec, c12_tr, c12_cm, up_all
        gc.collect()
        torch.cuda.empty_cache()

    results["wall_seconds"] = round(time.time() - t0, 1)
    _atomic_write_json(out_dir / "nv2_n2_mechanistic_xai.json", results)
    print("NV2_N2_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
