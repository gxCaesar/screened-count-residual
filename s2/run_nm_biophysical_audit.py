#!/usr/bin/env python3
"""EXP-4: Deep Biophysical, Spectral, Tolerance-Radius, Cutoff-Perturbation & Microscopy Audit.

Evaluates the EXACT frozen S6 checkpoints (without modifying any S6 file) across HEMIT,
Orion-CRC Fold 0, and HER2match to produce publication-grade analytical & visual source data:
  1. ``tolerance_radius_sweep.json``       : F1 at chamfer radii r in {0, 1, 3, 5, 7} px (incl. exact r=0!).
  2. ``threshold_curves_and_histograms.json``: Full 255-threshold F1(t), cutoff perturbation tau +/- delta,
                                             and 64-bin intensity distributions around tau_c.
  3. ``spectral_and_cg_audit.json``        : Empirical 2D radial Fourier PSD of real residuals r_c +
                                             empirical CG convergence k=1..22 vs analytical (sqrt(2)-1)/(sqrt(2)+1) bound.
  4. ``single_cell_deep_audit.json``       : HEMIT 4-state cell confusion matrices + 945-tile count scatters;
                                             Orion 15x15 co-occurrence correlation matrices;
                                             HER2 500-tile burden scatter + negative-slide specificity audit.
  5. ``microscopy_multizoom_{hemit,orion,her2}.npz``: 2 rich ROIs per cohort with H&E, Physical, Frozen,
                                             S_recon, S_count, residual_f32, masks, and high-contrast 1D line scans.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

from . import journal_cohort_eval as jce
from . import query_relation_reconstruction as core
from . import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _disk_offsets(radius: int) -> np.ndarray:
    """Boolean structuring element for Euclidean disk dx^2 + dy^2 <= r^2."""
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= (radius * radius)


def eval_tolerance_radii_gpu(
    pred_u8: np.ndarray,
    targ_u8: np.ndarray,
    cutoff: int,
    radii: tuple[int, ...] = (0, 1, 3, 5, 7),
    device: torch.device = torch.device("cuda:0"),
) -> dict[str, float]:
    """Compute exact pooled tolerant F1 at multiple spatial tolerance radii (incl. r=0 exact pixel F1)."""
    out = {}
    n = pred_u8.shape[0]
    bs = 32
    for r in radii:
        se = _disk_offsets(r)
        tp_p = 0.0
        tp_g = 0.0
        n_p = 0.0
        n_g = 0.0
        se_t = torch.from_numpy(se.astype(np.float32)).to(device).view(1, 1, se.shape[0], se.shape[1])
        pad = r
        for s in range(0, n, bs):
            e = min(n, s + bs)
            p_b = (torch.from_numpy(pred_u8[s:e]).to(device) > cutoff).to(torch.float32).unsqueeze(1)
            g_b = (torch.from_numpy(targ_u8[s:e]).to(device) > cutoff).to(torch.float32).unsqueeze(1)
            if r == 0:
                p_dil = p_b
                g_dil = g_b
            else:
                p_dil = (F.conv2d(p_b, se_t, padding=pad) > 0.5).to(torch.float32)
                g_dil = (F.conv2d(g_b, se_t, padding=pad) > 0.5).to(torch.float32)
            tp_p += float((p_b * g_dil).sum().item())
            tp_g += float((g_b * p_dil).sum().item())
            n_p += float(p_b.sum().item())
            n_g += float(g_b.sum().item())
        prec = tp_p / max(n_p, 1.0)
        rec = tp_g / max(n_g, 1.0)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out[f"r{r}"] = float(f1)
    return out


def eval_threshold_sweep_and_hist(
    pred_u8: np.ndarray,
    targ_u8: np.ndarray,
    cutoff: int,
    device: torch.device,
) -> dict:
    """Compute full 255-threshold F1(t), cutoff perturbations tau +/- delta, and 64-bin intensity PDF."""
    res = jce.batch_curves_and_quality_torch(pred_u8, targ_u8, cutoff, device=device, compute_quality=False)
    K = res["K"].sum(axis=0).astype(np.float64)
    U = res["U"].sum(axis=0).astype(np.float64)
    V = res["V"].sum(axis=0).astype(np.float64)
    G = float(res["G"].sum())
    prec = np.divide(U, K, out=np.zeros_like(U), where=K > 0)
    rec = np.divide(V, G, out=np.zeros_like(V), where=G > 0)
    f1_curve = np.divide(2.0 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)

    deltas = (-12, -8, -4, -2, 0, 2, 4, 8, 12)
    pert = {}
    for d in deltas:
        t_idx = int(np.clip(cutoff + d, 0, 254))
        pert[str(d)] = float(f1_curve[t_idx])

    hist_64, _ = np.histogram(pred_u8.ravel(), bins=64, range=(0, 256), density=False)
    hist_frac = (hist_64 / max(1.0, float(hist_64.sum()))).tolist()
    return dict(
        f1_curve_255=[round(float(x), 5) for x in f1_curve[:255]],
        cutoff=int(cutoff),
        ods_t=int(np.argmax(f1_curve[:255])),
        ods_f1=float(np.max(f1_curve[:255])),
        fixed_f1=float(f1_curve[cutoff]),
        perturb=pert,
        hist64_frac=[round(float(x), 6) for x in hist_frac],
    )


def compute_radial_psd(resid_nhw: np.ndarray, n_bins: int = 40) -> tuple[list[float], list[float]]:
    """Compute radially averaged 2D Fourier Power Spectral Density normalized to unit total energy."""
    # resid_nhw: float32 (N, H, W)
    n, h, w = resid_nhw.shape
    fft2 = np.fft.fft2(resid_nhw.astype(np.float64))
    pwr = np.mean(np.abs(fft2) ** 2, axis=0) / float(h * w)
    fy = np.fft.fftfreq(h) * 2.0  # in [-1, 1] -> omega/pi
    fx = np.fft.fftfreq(w) * 2.0
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    kr = np.sqrt(kx * kx + ky * ky)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (0.5 * (edges[:-1] + edges[1:])).tolist()
    radial = []
    # Exact DC bin first
    dc_power = float(pwr[0, 0])
    total_pwr = max(float(pwr.sum()), 1e-30)
    for b in range(n_bins):
        mask = (kr >= edges[b]) & (kr < edges[b + 1])
        radial.append(float(pwr[mask].mean() / total_pwr) if np.any(mask) else 0.0)
    return centers, radial, dc_power / total_pwr


def measure_cg_convergence(
    model: core.RelationModel,
    common_sample: torch.Tensor,
    device: torch.device,
) -> dict:
    """Measure exact empirical FP64 CG relative residual vs step k = 1..22 against theory."""
    model.eval()
    with torch.no_grad():
        c = common_sample[:16].to(device, dtype=torch.float32)
        z16 = model.token_projection.weight.new_zeros((c.shape[0], 16, *c.shape[-2:]))
        h = model.backbone(torch.cat((c, z16), dim=1))
        pot = suh.fast_unary_potential(model, h).to(torch.float64)
        rhs = suh._fast_laplacian_1_16(pot)
        rhs_norm = rhs.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-30)

        sol = torch.zeros_like(rhs)
        res = rhs.clone()
        direction = res.clone()
        rr = res.square().sum(dim=(-2, -1), keepdim=True)
        steps = list(range(1, 23))
        emp_rel = []
        emp_mean_abs = []
        for k in steps:
            op = direction + suh._fast_laplacian_1_16(direction)
            denom = (direction * op).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-300)
            alpha = rr / denom
            sol = sol + alpha * direction
            res = res - alpha * op
            rel = float((res.square().sum(dim=(-2, -1), keepdim=True).sqrt() / rhs_norm).max().item())
            emp_rel.append(rel)
            emp_mean_abs.append(float(sol.mean(dim=(-2, -1)).abs().max().item()))
            new_rr = res.square().sum(dim=(-2, -1), keepdim=True)
            beta = new_rr / rr.clamp_min(1e-300)
            direction = res + beta * direction
            rr = new_rr
        q = (np.sqrt(2.0) - 1.0) / (np.sqrt(2.0) + 1.0)
        theory = [float(2.0 * (q ** k)) for k in steps]
    return dict(steps=steps, empirical_max_rel=emp_rel, empirical_dc_abs=emp_mean_abs, analytical_bound=theory)


def _predict_with_resid_2seed(
    state_dir: Path,
    prefix: str,
    arm: str,
    seeds: tuple[int, ...],
    c12: torch.Tensor,
    donor_u8: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, core.RelationModel]:
    preds = []
    resids = []
    last_m = None
    n = c12.shape[0]
    for sd in seeds:
        m = suh.build_coordinate_screened_model(sd, device)
        m.load_state_dict(torch.load(state_dir / f"{prefix}_{arm}_{sd}.pt", map_location=device, weights_only=True))
        m.eval()
        p_out = np.empty((n, c12.shape[2], c12.shape[3]), dtype=np.float32)
        r_out = np.empty((n, c12.shape[2], c12.shape[3]), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n, batch_size):
                e = min(n, s + batch_size)
                cb = c12[s:e].to(device, dtype=torch.float32)
                db = donor_u8[s:e].to(device, dtype=torch.float32).unsqueeze(1) / 255.0
                r_b, _ = suh.fast_sc_forward(m, cb, inference=True)
                r_f32 = r_b.to(torch.float32)
                z_u8 = torch.round((db + r_f32).clamp(0.0, 1.0) * 255.0)
                p_out[s:e] = z_u8[:, 0].cpu().numpy()
                r_out[s:e] = r_f32[:, 0].cpu().numpy()
        preds.append(p_out)
        resids.append(r_out)
        last_m = m
    pred_ens = np.round(np.mean(preds, axis=0)).astype(np.uint8)
    resid_ens = np.mean(resids, axis=0).astype(np.float32)
    return pred_ens, resid_ens, last_m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-cache-dir", required=True)
    ap.add_argument("--hemit-state-dir", required=True)
    ap.add_argument("--orion-state-dir", required=True)
    ap.add_argument("--her2-state-dir", required=True)
    ap.add_argument("--cv-manifest", required=True)
    ap.add_argument("--her2-cal-json", required=True)
    ap.add_argument("--hoptimus0-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from . import run_hemit_multibackbone_cell as rhmc
    from . import run_her2match_dab_campaign as rhd
    from . import run_orion_multibackbone_survival as roms

    t0 = time.monotonic()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bc_dir = Path(args.bank_cache_dir).resolve()
    seeds = (202610270, 202610271)

    tol_sweep: dict = {}
    thr_hist: dict = {}
    spec_cg: dict = {}
    cell_deep: dict = {}

    # ------------------------------------------------------------------
    # 1. HEMIT (Sanofi MIPHEI-ViT hero + ConvNeXt + DiffusionFT + Swin)
    # ------------------------------------------------------------------
    print("[1/3] Auditing HEMIT frozen S6 checkpoints...", flush=True)
    with open(bc_dir / "hemit_bank_test_945.pkl", "rb") as fh:
        hem_ev = pickle.load(fh)
    with open(bc_dir / "hemit_bank_train_596.pkl", "rb") as fh:
        hem_tr = pickle.load(fh)
    hem_state = Path(args.hemit_state_dir).resolve()
    hem_bks = ("sanofi_miphei_vit", "sanofi_unetr_hoptimus", "sanofi_miphei_convnext", "sanofi_diffusion_ft", "sanofi_hemit_swin")

    tol_sweep["HEMIT"] = {}
    thr_hist["HEMIT"] = {}
    cell_deep["HEMIT"] = {}

    for b_key in hem_bks:
        with np.load(hem_state / "upstream_cache" / f"{b_key}_test_945.npz") as z:
            up_ev = z["pred_u8"]
        with np.load(hem_state / "upstream_cache" / f"{b_key}_train_596.npz") as z:
            up_tr = z["pred_u8"]

        sr_ev = np.empty_like(up_ev)
        sc_ev = np.empty_like(up_ev)
        sc_tr = np.empty_like(up_tr)
        sr_tr = np.empty_like(up_tr)
        sc_resid_ev = np.empty(up_ev.shape, dtype=np.float32)

        tol_sweep["HEMIT"][b_key] = {}
        thr_hist["HEMIT"][b_key] = {}

        for ch, (m_nm, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
            c12_ev = rhmc.build_common12_for_3ch(hem_ev["he_nhwc"], up_ev, ch, device=device)
            d_ev_t = torch.from_numpy(up_ev[:, ch])
            sr_c, _, _ = _predict_with_resid_2seed(hem_state / b_key, m_nm, "S_recon", seeds, c12_ev, d_ev_t, device)
            sc_c, r_c, last_sc = _predict_with_resid_2seed(hem_state / b_key, m_nm, "S_count", seeds, c12_ev, d_ev_t, device)
            sr_ev[:, ch], sc_ev[:, ch], sc_resid_ev[:, ch] = sr_c, sc_c, r_c

            if b_key == "sanofi_miphei_vit" and m_nm == "CD3":
                spec_cg["HEMIT_cg_convergence"] = measure_cg_convergence(last_sc, c12_ev, device)
            if b_key == "sanofi_miphei_vit":
                freqs, psd, dc_frac = compute_radial_psd(r_c[:128])
                spec_cg[f"HEMIT_psd_{m_nm}"] = dict(freqs=freqs, radial_psd=psd, dc_power_frac=dc_frac)

            del c12_ev
            if b_key == "sanofi_miphei_vit":
                # Also get train predictions for exact single-cell threshold & confusion audit
                c12_tr = rhmc.build_common12_for_3ch(hem_tr["he_nhwc"], up_tr, ch, device=device)
                d_tr_t = torch.from_numpy(up_tr[:, ch])
                sr_tr[:, ch], _, _ = _predict_with_resid_2seed(hem_state / b_key, m_nm, "S_recon", seeds, c12_tr, d_tr_t, device)
                sc_tr[:, ch], _, _ = _predict_with_resid_2seed(hem_state / b_key, m_nm, "S_count", seeds, c12_tr, d_tr_t, device)
                del c12_tr

            # Tolerance-radius sweep & threshold curves
            tol_sweep["HEMIT"][b_key][m_nm] = dict(
                upstream=eval_tolerance_radii_gpu(up_ev[:, ch], hem_ev["targ_nchw"][:, ch], cutoff, device=device),
                S_recon=eval_tolerance_radii_gpu(sr_c, hem_ev["targ_nchw"][:, ch], cutoff, device=device),
                S_count=eval_tolerance_radii_gpu(sc_c, hem_ev["targ_nchw"][:, ch], cutoff, device=device),
            )
            if b_key == "sanofi_miphei_vit":
                gt_h, _ = np.histogram(hem_ev["targ_nchw"][:, ch].ravel(), bins=64, range=(0, 256), density=False)
                thr_hist["HEMIT"][m_nm] = dict(
                    gt_hist64_frac=[round(float(x), 6) for x in (gt_h / max(1.0, float(gt_h.sum())))],
                    upstream=eval_threshold_sweep_and_hist(up_ev[:, ch], hem_ev["targ_nchw"][:, ch], cutoff, device),
                    S_recon=eval_threshold_sweep_and_hist(sr_c, hem_ev["targ_nchw"][:, ch], cutoff, device),
                    S_count=eval_threshold_sweep_and_hist(sc_c, hem_ev["targ_nchw"][:, ch], cutoff, device),
                )
            gc.collect()
            torch.cuda.empty_cache()

        if b_key == "sanofi_miphei_vit":
            # 4-state single-cell confusion matrix (DN=0, panCK+=1, CD3+=2, DP=3) + 945-tile cell count scatters
            tr_ref = rhmc.extract_pred_cell_means_for_records(up_tr, hem_tr["nuclei_records"])
            tau_pck = float(rhmc.ccs.evaluate_single_cell_marker(
                train_gt_scores=tr_ref["gt_means"][:, 0], train_pred_scores=tr_ref["pred_means"][:, 0],
                train_labels=tr_ref["panck_pos"], eval_pred_scores=tr_ref["pred_means"][:, 0],
                eval_labels=tr_ref["panck_pos"], eval_tile_ids=tr_ref["tile_ids"],
            )["tau_physical"])
            tau_cd3 = float(rhmc.ccs.evaluate_single_cell_marker(
                train_gt_scores=tr_ref["gt_means"][:, 1], train_pred_scores=tr_ref["pred_means"][:, 1],
                train_labels=tr_ref["cd3_pos"], eval_pred_scores=tr_ref["pred_means"][:, 1],
                eval_labels=tr_ref["cd3_pos"], eval_tile_ids=tr_ref["tile_ids"],
            )["tau_physical"])

            ev_ref = rhmc.extract_pred_cell_means_for_records(up_ev, hem_ev["nuclei_records"])
            gt_state = ev_ref["panck_pos"].astype(np.int32) + 2 * ev_ref["cd3_pos"].astype(np.int32)
            tile_gt_cd3 = np.bincount(ev_ref["tile_ids"], weights=ev_ref["cd3_pos"], minlength=len(hem_ev["names"]))
            tile_gt_pck = np.bincount(ev_ref["tile_ids"], weights=ev_ref["panck_pos"], minlength=len(hem_ev["names"]))

            cm_dict = {}
            tile_counts = dict(gt_cd3=tile_gt_cd3.astype(int).tolist(), gt_panck=tile_gt_pck.astype(int).tolist())
            for arm_nm, arr_ev in (("upstream", up_ev), ("S_recon", sr_ev), ("S_count", sc_ev)):
                ec = rhmc.extract_pred_cell_means_for_records(arr_ev, hem_ev["nuclei_records"])
                p_pck = (ec["pred_means"][:, 0] > tau_pck).astype(np.int32)
                p_cd3 = (ec["pred_means"][:, 1] > tau_cd3).astype(np.int32)
                pr_state = p_pck + 2 * p_cd3
                cm = np.zeros((4, 4), dtype=np.int64)
                for gs, ps in zip(gt_state, pr_state):
                    cm[int(gs), int(ps)] += 1
                cm_dict[arm_nm] = cm.tolist()
                tile_counts[f"{arm_nm}_cd3"] = np.bincount(ec["tile_ids"], weights=p_cd3, minlength=len(hem_ev["names"])).astype(int).tolist()
                tile_counts[f"{arm_nm}_panck"] = np.bincount(ec["tile_ids"], weights=p_pck, minlength=len(hem_ev["names"])).astype(int).tolist()
            cell_deep["HEMIT"]["sanofi_miphei_vit"] = dict(
                tau_panck=tau_pck, tau_cd3=tau_cd3, states=["DN", "panCK+", "CD3+", "DP"],
                confusion_4x4=cm_dict, tile_counts_945=tile_counts,
            )

            # Export 2 rich co-populated tumor+immune HEMIT tiles with nuclear masks & residual maps
            cands = []
            for ti, rec in enumerate(hem_ev["nuclei_records"]):
                if rec is None:
                    continue
                n_pck = int(rec["panck_pos"].sum())
                n_cd3 = int(rec["cd3_pos"].sum())
                if n_pck >= 45 and n_cd3 >= 18:
                    # Pick tiles where S_count sharply recovers CD3 over Frozen
                    g_cd3 = hem_ev["targ_nchw"][ti, 1] > rhmc.HEMIT_CUTOFFS[1]
                    u_rec = float(np.mean((up_ev[ti, 1] > rhmc.HEMIT_CUTOFFS[1]) & g_cd3)) / max(float(g_cd3.mean()), 1e-6)
                    s_rec = float(np.mean((sc_ev[ti, 1] > rhmc.HEMIT_CUTOFFS[1]) & g_cd3)) / max(float(g_cd3.mean()), 1e-6)
                    cands.append((s_rec - u_rec + 0.002 * min(n_cd3, 60), ti, n_pck, n_cd3))
            cands.sort(reverse=True)
            chosen = [cands[0][1], cands[min(5, len(cands) - 1)][1]]
            masks_out = []
            for ti in chosen:
                rec = hem_ev["nuclei_records"][ti]
                if "mask_hw" in rec:
                    masks_out.append(rec["mask_hw"].astype(np.int32))
                else:
                    # Reconstruct mask from idx_spec dict if mask_hw stripped
                    m_hw = np.zeros((512, 512), dtype=np.int32)
                    spec = rec.get("idx_spec")
                    if isinstance(spec, dict) and "fg_pos" in spec and "inv" in spec:
                        m_hw.ravel()[spec["fg_pos"]] = spec["inv"].astype(np.int32) + 1
                    masks_out.append(m_hw)
            np.savez_compressed(
                out_dir / "microscopy_multizoom_hemit.npz",
                backbone="sanofi_miphei_vit",
                tile_indices=np.array(chosen, dtype=np.int32),
                names=np.array([hem_ev["names"][i] for i in chosen]),
                cutoffs=np.array(rhmc.HEMIT_CUTOFFS, dtype=np.int32),
                he_rgb=hem_ev["he_nhwc"][chosen],
                phys_nchw=hem_ev["targ_nchw"][chosen],
                up_nchw=up_ev[chosen],
                sr_nchw=sr_ev[chosen],
                sc_nchw=sc_ev[chosen],
                resid_nchw=sc_resid_ev[chosen],
                nuc_masks=np.stack(masks_out, axis=0),
            )
    del hem_ev, hem_tr
    gc.collect()

    # ------------------------------------------------------------------
    # 2. ORION-CRC Fold 0 (miphei_convnext hero + miphei_hemit_arch + rosie_orion)
    # ------------------------------------------------------------------
    print("[2/3] Auditing Orion-CRC Fold 0 frozen S6 checkpoints...", flush=True)
    cv_manifest = json.loads(Path(args.cv_manifest).read_text())
    with open(bc_dir / "orion_bank_all40_2804_crop256.pkl", "rb") as fh:
        ori_b = pickle.load(fh)
    fold0 = cv_manifest["folds"][0]
    ev_idx = np.nonzero(np.isin(ori_b["patients"], list(set(roms._fold_patients(fold0, "outer_test")))))[0]
    ori_state = Path(args.orion_state_dir).resolve()

    tol_sweep["Orion-CRC"] = {}
    thr_hist["Orion-CRC"] = {}
    cell_deep["Orion-CRC"] = {}

    for b_key in ("miphei_convnext", "miphei_hemit_arch", "rosie_orion"):
        with np.load(ori_state / "upstream_cache" / f"{b_key}_all40_2804.npz") as z:
            up_ev = z["pred_u8"][ev_idx]
        c_doc = json.loads((ori_state / b_key / "fold0" / "fold0_physical_cutoffs.json").read_text())
        cutoffs_u8, tau_float = list(c_doc["cutoffs_u8"]), dict(c_doc["tau_phys_float"])

        sr_ev = np.empty_like(up_ev)
        sc_ev = np.empty_like(up_ev)
        sc_resid_ev = np.empty(up_ev.shape, dtype=np.float32)
        tol_sweep["Orion-CRC"][b_key] = {}

        for ch16, m_nm in enumerate(roms.ORION_MARKERS):
            cutoff = int(cutoffs_u8[ch16])
            c12_ev = roms.build_common12_orion(ori_b["he_nhwc"][ev_idx], up_ev, ch16, device=device)
            d_ev_t = torch.from_numpy(up_ev[:, ch16])
            sr_c, _, _ = _predict_with_resid_2seed(ori_state / b_key / "fold0", m_nm, "S_recon", seeds, c12_ev, d_ev_t, device, batch_size=32)
            sc_c, r_c, _ = _predict_with_resid_2seed(ori_state / b_key / "fold0", m_nm, "S_count", seeds, c12_ev, d_ev_t, device, batch_size=32)
            sr_ev[:, ch16], sc_ev[:, ch16], sc_resid_ev[:, ch16] = sr_c, sc_c, r_c

            if m_nm in ("Pan-CK", "CD3e", "CD8a", "CD68", "E-cadherin"):
                tol_sweep["Orion-CRC"][b_key][m_nm] = dict(
                    upstream=eval_tolerance_radii_gpu(up_ev[:, ch16], ori_b["targ_nchw"][ev_idx, ch16], cutoff, device=device),
                    S_recon=eval_tolerance_radii_gpu(sr_c, ori_b["targ_nchw"][ev_idx, ch16], cutoff, device=device),
                    S_count=eval_tolerance_radii_gpu(sc_c, ori_b["targ_nchw"][ev_idx, ch16], cutoff, device=device),
                )
            if b_key == "miphei_convnext" and m_nm in ("Pan-CK", "CD3e", "CD8a"):
                gt_h, _ = np.histogram(ori_b["targ_nchw"][ev_idx, ch16].ravel(), bins=64, range=(0, 256), density=False)
                thr_hist["Orion-CRC"][m_nm] = dict(
                    gt_hist64_frac=[round(float(x), 6) for x in (gt_h / max(1.0, float(gt_h.sum())))],
                    upstream=eval_threshold_sweep_and_hist(up_ev[:, ch16], ori_b["targ_nchw"][ev_idx, ch16], cutoff, device),
                    S_recon=eval_threshold_sweep_and_hist(sr_c, ori_b["targ_nchw"][ev_idx, ch16], cutoff, device),
                    S_count=eval_threshold_sweep_and_hist(sc_c, ori_b["targ_nchw"][ev_idx, ch16], cutoff, device),
                )
                freqs, psd, dc_frac = compute_radial_psd(r_c[:128])
                spec_cg[f"Orion_psd_{m_nm}"] = dict(freqs=freqs, radial_psd=psd, dc_power_frac=dc_frac)
            del c12_ev
            gc.collect()
            torch.cuda.empty_cache()

        # 15x15 marker co-occurrence correlation matrices
        ev_c_up = roms.gather_subset_cells(up_ev, ev_idx, ori_b["nuclei_records"])
        ev_c_sr = roms.gather_subset_cells(sr_ev, ev_idx, ori_b["nuclei_records"])
        ev_c_sc = roms.gather_subset_cells(sc_ev, ev_idx, ori_b["nuclei_records"])
        gt_lbl = ev_c_up["labels15"].astype(np.float64)
        def _corr15(mat: np.ndarray) -> np.ndarray:
            c = np.zeros((15, 15), dtype=np.float64)
            for i in range(15):
                si = float(np.std(mat[:, i]))
                for j in range(15):
                    sj = float(np.std(mat[:, j]))
                    c[i, j] = 1.0 if i == j else (float(np.corrcoef(mat[:, i], mat[:, j])[0, 1]) if si > 1e-12 and sj > 1e-12 else 0.0)
            return c
        gt_cooc = _corr15(gt_lbl)
        def _pred_bin(ec):
            return np.stack([(ec["pred_means"][:, ch16] > float(tau_float[roms.ORION_MARKERS[ch16]])).astype(np.float64) for ch16 in range(1, 16)], axis=1)
        up_cooc, sr_cooc, sc_cooc = _corr15(_pred_bin(ev_c_up)), _corr15(_pred_bin(ev_c_sr)), _corr15(_pred_bin(ev_c_sc))
        cell_deep["Orion-CRC"][b_key] = dict(
            markers15=list(roms.ORION_MARKERS[1:]),
            gt_cooc=gt_cooc.tolist(), up_cooc=up_cooc.tolist(), sr_cooc=sr_cooc.tolist(), sc_cooc=sc_cooc.tolist(),
            up_mae=float(np.mean(np.abs(up_cooc - gt_cooc))),
            sr_mae=float(np.mean(np.abs(sr_cooc - gt_cooc))),
            sc_mae=float(np.mean(np.abs(sc_cooc - gt_cooc))),
        )

        if b_key == "miphei_convnext":
            # Export 2 rich prognostic triad tiles (Pan-CK=14, CD3e=10, CD8a=6)
            tri = [14, 10, 6]
            cands = []
            for loc_i, glob_i in enumerate(ev_idx):
                gt_pck = float(np.mean(ori_b["targ_nchw"][glob_i, 14] > cutoffs_u8[14]))
                gt_cd8 = float(np.mean(ori_b["targ_nchw"][glob_i, 6] > cutoffs_u8[6]))
                if 0.15 <= gt_pck <= 0.65 and gt_cd8 >= 0.03:
                    g8 = ori_b["targ_nchw"][glob_i, 6] > cutoffs_u8[6]
                    gain = float(np.mean((sc_ev[loc_i, 6] > cutoffs_u8[6]) & g8) - np.mean((up_ev[loc_i, 6] > cutoffs_u8[6]) & g8)) / max(gt_cd8, 1e-6)
                    cands.append((gain + 0.5 * gt_cd8, loc_i, int(glob_i)))
            cands.sort(reverse=True)
            chosen_loc = [cands[0][1], cands[min(4, len(cands) - 1)][1]]
            chosen_glob = [int(ev_idx[i]) for i in chosen_loc]
            np.savez_compressed(
                out_dir / "microscopy_multizoom_orion.npz",
                backbone="miphei_convnext",
                triad_indices=np.array(tri, dtype=np.int32),
                triad_names=np.array(["Pan-CK", "CD3e", "CD8a"]),
                triad_cutoffs=np.array([cutoffs_u8[c] for c in tri], dtype=np.int32),
                patients=np.array([str(ori_b["patients"][g]) for g in chosen_glob]),
                he_rgb=ori_b["he_nhwc"][chosen_glob],
                phys_nchw=ori_b["targ_nchw"][chosen_glob][:, tri],
                up_nchw=up_ev[chosen_loc][:, tri],
                sr_nchw=sr_ev[chosen_loc][:, tri],
                sc_nchw=sc_ev[chosen_loc][:, tri],
                resid_nchw=sc_resid_ev[chosen_loc][:, tri],
            )
    del ori_b
    gc.collect()

    # ------------------------------------------------------------------
    # 3. HER2match (all 3 brightfield generators)
    # ------------------------------------------------------------------
    print("[3/3] Auditing HER2match frozen S6 checkpoints...", flush=True)
    cal_doc = json.loads(Path(args.her2_cal_json).read_text())
    cutoff_u8 = int(round(float(cal_doc["threshold"]) / float(cal_doc["histogram"]["range"][1]) * 255.0))
    with open(bc_dir / "her2_bank_train_480_512.pkl", "rb") as fh:
        h_tr = pickle.load(fh)
    with open(bc_dir / "her2_bank_test_500_512.pkl", "rb") as fh:
        h_ev = pickle.load(fh)
    her2_state = Path(args.her2_state_dir).resolve()
    hopt_path = Path(args.hoptimus0_ckpt).resolve()

    tol_sweep["HER2match"] = {}
    thr_hist["HER2match"] = {}
    cell_deep["HER2match"] = {}

    for b_key in ("hoptimus_lora_ihc", "unet_l1_ssim", "pix2pix_cgan_ihc"):
        _, _, up_ev = rhd.train_or_load_her2_backbone(
            b_key, h_tr, h_ev, cutoff_u8, her2_state, device, hoptimus0_ckpt=hopt_path, upstream_epochs=8, allow_train=False,
        )
        c12_ev = rhd.build_common12_her2(h_ev["he_nhwc"], up_ev, h_ev["hem_u8"], device=device)
        d_ev_t = torch.from_numpy(up_ev)
        sr_ev, _, _ = _predict_with_resid_2seed(her2_state / b_key, "HER2_DAB", "S_recon", seeds, c12_ev, d_ev_t, device)
        sc_ev, r_ev, _ = _predict_with_resid_2seed(her2_state / b_key, "HER2_DAB", "S_count", seeds, c12_ev, d_ev_t, device)
        del c12_ev

        tol_sweep["HER2match"][b_key] = dict(
            upstream=eval_tolerance_radii_gpu(up_ev, h_ev["dab_u8"], cutoff_u8, device=device),
            S_recon=eval_tolerance_radii_gpu(sr_ev, h_ev["dab_u8"], cutoff_u8, device=device),
            S_count=eval_tolerance_radii_gpu(sc_ev, h_ev["dab_u8"], cutoff_u8, device=device),
        )
        gt_h, _ = np.histogram(h_ev["dab_u8"].ravel(), bins=64, range=(0, 256), density=False)
        thr_hist["HER2match"][b_key] = dict(
            gt_hist64_frac=[round(float(x), 6) for x in (gt_h / max(1.0, float(gt_h.sum())))],
            upstream=eval_threshold_sweep_and_hist(up_ev, h_ev["dab_u8"], cutoff_u8, device),
            S_recon=eval_threshold_sweep_and_hist(sr_ev, h_ev["dab_u8"], cutoff_u8, device),
            S_count=eval_threshold_sweep_and_hist(sc_ev, h_ev["dab_u8"], cutoff_u8, device),
        )
        if b_key == "hoptimus_lora_ihc":
            freqs, psd, dc_frac = compute_radial_psd(r_ev[:128])
            spec_cg["HER2_psd_DAB"] = dict(freqs=freqs, radial_psd=psd, dc_power_frac=dc_frac)
            # Export 2 HER2match brightfield tiles: 1 strong HER2-3+ positive ROI + 1 HER2-negative specificity ROI
            wsis_arr = [str(w) for w in h_ev["wsis"]]
            gt_b_all = (h_ev["dab_u8"] > cutoff_u8).mean(axis=(1, 2))
            up_b_all = (up_ev > cutoff_u8).mean(axis=(1, 2))
            sc_b_all = (sc_ev > cutoff_u8).mean(axis=(1, 2))
            pos_cands = [
                (float(sc_b_all[i] - up_b_all[i]) + 0.5 * float(gt_b_all[i]), i)
                for i in range(len(wsis_arr))
                if wsis_arr[i] in ("wsi010", "wsi015") and 0.12 <= float(gt_b_all[i]) <= 0.60
            ]
            neg_cands = [
                (float(up_b_all[i]) - float(sc_b_all[i]), i)
                for i in range(len(wsis_arr))
                if wsis_arr[i] in ("wsi001", "wsi008", "wsi016") and float(gt_b_all[i]) <= 0.015
            ]
            pos_cands.sort(reverse=True)
            neg_cands.sort(reverse=True)
            chosen_h2 = [pos_cands[0][1] if pos_cands else 0, neg_cands[0][1] if neg_cands else 1]
            np.savez_compressed(
                out_dir / "microscopy_multizoom_her2.npz",
                backbone="hoptimus_lora_ihc",
                cutoff=np.array(cutoff_u8, dtype=np.int32),
                tile_indices=np.array(chosen_h2, dtype=np.int32),
                wsis=np.array([wsis_arr[i] for i in chosen_h2]),
                he_rgb=h_ev["he_nhwc"][chosen_h2],
                phys_dab=h_ev["dab_u8"][chosen_h2],
                up_dab=up_ev[chosen_h2],
                sr_dab=sr_ev[chosen_h2],
                sc_dab=sc_ev[chosen_h2],
                resid_dab=r_ev[chosen_h2],
            )

        # Per-tile DAB+ burden across all 500 tiles + negative-slide specificity
        wsis = [str(w) for w in h_ev["wsis"]]
        gt_b = (h_ev["dab_u8"] > cutoff_u8).mean(axis=(1, 2))
        up_b = (up_ev > cutoff_u8).mean(axis=(1, 2))
        sr_b = (sr_ev > cutoff_u8).mean(axis=(1, 2))
        sc_b = (sc_ev > cutoff_u8).mean(axis=(1, 2))
        neg_mask = np.array([w in ("wsi001", "wsi008", "wsi016") for w in wsis])
        cell_deep["HER2match"][b_key] = dict(
            wsis=wsis,
            gt_tile_burden=[round(float(x), 5) for x in gt_b],
            up_tile_burden=[round(float(x), 5) for x in up_b],
            sr_tile_burden=[round(float(x), 5) for x in sr_b],
            sc_tile_burden=[round(float(x), 5) for x in sc_b],
            neg_slides_fp_rate=dict(
                upstream=float(up_b[neg_mask].mean()),
                S_recon=float(sr_b[neg_mask].mean()),
                S_count=float(sc_b[neg_mask].mean()),
                gt_pos=float(gt_b[neg_mask].mean()),
            ),
        )
        gc.collect()
        torch.cuda.empty_cache()

    _atomic_write_json(out_dir / "tolerance_radius_sweep.json", tol_sweep)
    _atomic_write_json(out_dir / "threshold_curves_and_histograms.json", thr_hist)
    _atomic_write_json(out_dir / "spectral_and_cg_audit.json", spec_cg)
    _atomic_write_json(out_dir / "single_cell_deep_audit.json", cell_deep)
    print(f"BIOPHYSICAL_AUDIT_COMPLETE in {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
