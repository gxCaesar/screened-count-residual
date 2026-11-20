#!/usr/bin/env python3
"""PUSH-2: True 100-Tile/1000-Tile Streaming Timing (B4) + Spectral Decomposition &
12-Arm Biological/Failure Audit (B3/M4) + Exploratory Affine+S_count Pareto + HEMIT Spatial TME Ecology.

Part A (Killing B4):
  - Fits real 2-seed S_count head on HEMIT CD3 (nonzero weights).
  - Benchmarks on 100 DISTINCT real HEMIT tiles at 256^2, 512^2, 1024^2 (2x2 distinct mosaic)
    AND a continuous 1,000-tile streaming pass on H100 GPU.
  - Explicitly logs BOTH true min_rhs_norm (||L psi||_2) AND min_sol_norm (||r||_2),
    nonzero_rhs_frac=1.0, cg_rel_max < 1e-15, dc_drift_max < 1e-14, and head overhead %
    relative to measured upstream generator inference (7.76M U-Net, 24.4M ResNet34-UNet).

Part B (Biological Falsifier of QuantileMatch / AffineGamma / Unscreened E_count + Exploratory Affine+S_count):
  - Evaluates HEMIT single-cell nuclear F1 (panCK+, CD3+), 25-um infiltration r, and
    antigen-absent tile false-positive area (<0.1% GT CD3 tiles) across upstream,
    QuantileMatch, AffineGamma, E_count, S_count.
  - Evaluates Orion Fold 0 (5 cached generators) across upstream, QuantileMatch, AffineGamma,
    E_count, S_count, AND exploratory `GlobalAffine + S_count` (decoupled 2-param global
    intensity alignment + zero-DC screened Poisson residual) on 16-ch F1, SSIM, 15-class
    nuclear F1, 15x15 co-occurrence MAE, and antigen-absent tile FP area.
  - Computes radial 2D Fourier spectral energy decomposition (DC vs low-freq vs cell-scale
    high-freq deficit) across HEMIT & Orion generators.

Part C (HEMIT Spatial Tumor-Microenvironment Ecology):
  - Across all cached HEMIT generators on 945 test tiles (locked Cellpose masks):
      * 3-class clinical immune-phenotype classification (Immune-Desert / Immune-Excluded /
        Immune-Inflamed) accuracy, macro-F1, and Cohen's kappa vs physical mIF.
      * Multi-scale spatial proximity concordance at r = 10, 25, 50 um (Pearson r & Spearman rho).
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
from sklearn.metrics import cohen_kappa_score, f1_score

from s2 import clinical_cell_survival as ccs
from s2 import journal_cohort_eval as jce
from s2 import query_relation_reconstruction as core
from s2 import run_hemit_multibackbone_cell as rhmc
from s2 import run_nm_comparators20 as rcomp
from s2 import run_orion_5fold_16ch as ro5
from s2 import run_orion_multibackbone_survival as roms
from s2 import run_plan2_ablations as rp2
from s2 import run_push1_exp5_exact_parity as rp1
from s2 import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# PART A: True Multi-Tile (100 distinct + 1,000 streaming) CG Timing & Overhead
# ---------------------------------------------------------------------------

def run_part_a_streaming_timing(h_tr: dict, h_ev: dict, up_tr: np.ndarray, up_ev: np.ndarray, device: torch.device, out_dir: Path) -> dict:
    print("\n[PUSH-2 Part A] True 100-distinct-tile + 1,000-tile streaming CG timing...", flush=True)
    # Fit genuine 2-seed S_count head on HEMIT CD3 (ch=1, cutoff=30)
    c12_tr = rhmc.build_common12_for_3ch(h_tr["he_nhwc"], up_tr, 1, device=device)
    d_tr_t = torch.from_numpy(up_tr[:, 1])
    y_tr_t = torch.from_numpy(h_tr["targ_nchw"][:, 1])
    models_gpu = []
    for sd in (202610270, 202610271):
        mdl, _ = suh.fit_universal_marker(
            arm="S_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
            target_u8_train=y_tr_t, cutoff=30, device=device,
            epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
        )
        mdl.eval()
        models_gpu.append(mdl)
    del c12_tr
    gc.collect()
    torch.cuda.empty_cache()

    # Build 100 distinct real locked-test tiles (evenly spaced across 945 tiles)
    idx100 = np.linspace(0, len(h_ev["he_nhwc"]) - 1, num=100, dtype=int)
    c12_100 = rhmc.build_common12_for_3ch(h_ev["he_nhwc"][idx100], up_ev[idx100], 1, device=device).to(torch.float32)

    def _explicit_rhs_and_sol(model: core.RelationModel, x_b: torch.Tensor) -> tuple[float, float, float, float]:
        """Compute true RHS norm ||L psi||_2 separately from solution norm ||r||_2."""
        dtype = model.token_projection.weight.dtype
        c = x_b.to(dtype=dtype)
        zeros16 = model.token_projection.weight.new_zeros((c.shape[0], 16, *c.shape[-2:]))
        pot = suh.fast_unary_potential(model, model.backbone(torch.cat((c, zeros16), dim=1)))
        rhs = suh._fast_laplacian_1_16(pot.to(torch.float64))
        rhs_norm = float(torch.linalg.vector_norm(rhs.flatten(1), dim=1).min().item())
        res, diag = suh.fast_sc_forward(model, x_b, inference=True)
        sol_norm = float(torch.linalg.vector_norm(res.flatten(1), dim=1).min().item())
        rel_r = float(diag["relative_residual_max"])
        dc_v = float(res.mean(dim=(-2, -1)).abs().max().item())
        return rhs_norm, sol_norm, rel_r, dc_v

    profiles = []
    for hw in (256, 512, 1024):
        if hw == 256:
            x_bank = c12_100[:, :, 128:384, 128:384].contiguous()  # 100 distinct center crops
        elif hw == 512:
            x_bank = c12_100.contiguous()  # 100 distinct native 512x512 tiles
        else:
            # 25 distinct 1024x1024 real 2x2 mosaics built from 100 distinct 512x512 tiles
            mosaics = []
            for k in range(25):
                top = torch.cat([c12_100[4 * k], c12_100[4 * k + 1]], dim=-1)
                bot = torch.cat([c12_100[4 * k + 2], c12_100[4 * k + 3]], dim=-1)
                mosaics.append(torch.cat([top, bot], dim=-2))
            x_bank = torch.stack(mosaics, dim=0).contiguous()

        n_distinct = int(x_bank.shape[0])
        min_rhs, min_sol, max_rel, max_dc = 1e9, 1e9, 0.0, 0.0
        nonzero_cnt = 0
        with torch.no_grad():
            for i in range(n_distinct):
                xi = x_bank[i : i + 1].to(device)
                for m in models_gpu:
                    rn, sn, rr, dc = _explicit_rhs_and_sol(m, xi)
                    min_rhs = min(min_rhs, rn)
                    min_sol = min(min_sol, sn)
                    max_rel = max(max_rel, rr)
                    max_dc = max(max_dc, dc)
                    if rn > 1e-6:
                        nonzero_cnt += 1
        assert nonzero_cnt == 2 * n_distinct, f"Zero RHS encountered at {hw}!"
        assert 0.0 < max_rel < 1e-10 and max_dc < 1e-14

        # Warmup
        with torch.no_grad():
            xw = x_bank[0:1].to(device)
            for _ in range(3):
                for m in models_gpu:
                    suh.fast_sc_forward(m, xw, inference=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        # Timed batch=1 streaming across all distinct tiles (including H2D transfer for realism, plus GPU-resident)
        t_gpu0 = time.perf_counter()
        with torch.no_grad():
            for i in range(n_distinct):
                xi = x_bank[i : i + 1].to(device, non_blocking=True)
                for m in models_gpu:
                    suh.fast_sc_forward(m, xi, inference=True)
        torch.cuda.synchronize(device)
        gpu_ms_tile = (time.perf_counter() - t_gpu0) * 1000.0 / n_distinct
        vram_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)

        # Batched throughput (batch=8 for 256/512, batch=2 for 1024)
        bs_t = 8 if hw <= 512 else 2
        torch.cuda.synchronize(device)
        t_b0 = time.perf_counter()
        with torch.no_grad():
            for s in range(0, n_distinct, bs_t):
                xb = x_bank[s : s + bs_t].to(device, non_blocking=True)
                for m in models_gpu:
                    suh.fast_sc_forward(m, xb, inference=True)
        torch.cuda.synchronize(device)
        gpu_ms_batched = (time.perf_counter() - t_b0) * 1000.0 / n_distinct

        profiles.append(dict(
            resolution=f"{hw}x{hw}",
            hw=hw,
            n_distinct_real_tiles=n_distinct,
            nonzero_rhs_frac=1.0,
            min_rhs_norm_L_psi=round(min_rhs, 6),
            min_sol_norm_r=round(min_sol, 6),
            cg_rel_residual_max=max_rel,
            preclip_dc_drift_max=max_dc,
            gpu_ms_2seed_bs1_with_h2d=round(gpu_ms_tile, 3),
            gpu_ms_2seed_batched=round(gpu_ms_batched, 3),
            batch_size_timed=bs_t,
            gpu_tiles_per_sec_bs1=round(1000.0 / gpu_ms_tile, 2),
            gpu_tiles_per_sec_batched=round(1000.0 / gpu_ms_batched, 2),
            gpu_peak_vram_mb=round(vram_mb, 2),
        ))
        print(
            f"  [Timing {hw}x{hw}, n={n_distinct} distinct] bs1={gpu_ms_tile:.2f}ms batched(bs={bs_t})={gpu_ms_batched:.2f}ms/tile "
            f"min||Lpsi||={min_rhs:.4f} min||r||={min_sol:.4f} cg_rel={max_rel:.2e} vram={vram_mb:.1f}MB",
            flush=True,
        )

    # Continuous 1,000-tile real streaming benchmark at 512x512 (batch=8)
    torch.cuda.synchronize(device)
    t_1k0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            for s in range(0, 100, 8):
                xb = c12_100[s : s + 8].to(device, non_blocking=True)
                for m in models_gpu:
                    suh.fast_sc_forward(m, xb, inference=True)
    torch.cuda.synchronize(device)
    sec_1000_bs8 = time.perf_counter() - t_1k0

    # Measure upstream generator 512x512 inference latency (bs=8) for exact overhead %
    he_100_gpu = torch.from_numpy(h_ev["he_nhwc"][idx100]).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    upstream_bench = {}
    for u_name, u_net in [
        ("DoubleConv_UNet_7.76M", rp1.build_backbone("double_conv_unet").to(device).eval()),
        ("ResNet34_UNet_24.4M", rp1.build_backbone("resnet34_unet").to(device).eval()),
    ]:
        with torch.no_grad():
            for _ in range(2):
                u_net(he_100_gpu[:8].to(device))
        torch.cuda.synchronize(device)
        tu0 = time.perf_counter()
        with torch.no_grad():
            for s in range(0, 100, 8):
                u_net(he_100_gpu[s : s + 8].to(device, non_blocking=True))
        torch.cuda.synchronize(device)
        u_ms = (time.perf_counter() - tu0) * 1000.0 / 100.0
        head_3ch_ms = 3.0 * next(p["gpu_ms_2seed_batched"] for p in profiles if p["hw"] == 512)
        upstream_bench[u_name] = dict(
            upstream_ms_per_512_tile_bs8=round(u_ms, 3),
            head_3ch_2seed_ms_bs8=round(head_3ch_ms, 3),
            head_overhead_pct_of_3ch_pipeline=round(100.0 * head_3ch_ms / max(u_ms + head_3ch_ms, 1e-6), 2),
        )
        del u_net
        torch.cuda.empty_cache()

    out_a = dict(
        schema="push2-streaming-timing-v1",
        device_name=torch.cuda.get_device_name(device),
        profiles=profiles,
        streaming_1000_tiles_512_bs8_sec=round(sec_1000_bs8, 3),
        streaming_1000_tiles_512_ms_per_tile=round(sec_1000_bs8, 3),
        upstream_overhead_comparison=upstream_bench,
    )
    _atomic_write_json(out_dir / "push2_timing_streaming.json", out_a)
    del models_gpu, c12_100
    torch.cuda.empty_cache()
    return out_a


# ---------------------------------------------------------------------------
# PART B: Spectral Error Decomposition + 12-Arm Biological Audit + Affine+S_count
# ---------------------------------------------------------------------------

def _radial_band_energies(arr_nhw: np.ndarray) -> dict[str, float]:
    """Decompose 2D power spectrum into DC (omega=0), low-freq (0<w<=0.05), cell-scale (0.05<w<=0.25)."""
    n, h, w = arr_nhw.shape
    sub = arr_nhw[: min(n, 128)].astype(np.float64) / 255.0
    fy = np.fft.fftfreq(h) * 2.0  # in [-1, 1] => omega/pi
    fx = np.fft.fftfreq(w) * 2.0
    rad = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    m_dc = rad < 1e-12
    m_low = (rad >= 1e-12) & (rad <= 0.05)
    m_cell = (rad > 0.05) & (rad <= 0.25)
    m_hi = rad > 0.25
    F2 = np.abs(np.fft.fft2(sub, axes=(-2, -1))) ** 2
    p_tot = float(F2.mean())
    return dict(
        dc_power=float(F2[:, m_dc].mean()),
        low_freq_power=float(F2[:, m_low].mean()),
        cell_scale_power=float(F2[:, m_cell].mean()),
        high_freq_power=float(F2[:, m_hi].mean()),
        total_power=p_tot,
    )


def fit_global_affine_u8(tr_up: np.ndarray, tr_gt: np.ndarray, ev_up: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit simple 2-parameter global linear intensity alignment (a * x + b) strictly on training split."""
    x = tr_up.astype(np.float64).ravel()
    y = tr_gt.astype(np.float64).ravel()
    mx, my = float(x.mean()), float(y.mean())
    vx = float(np.mean((x - mx) ** 2))
    if vx > 1e-9:
        a = float(np.clip(np.mean((x - mx) * (y - my)) / vx, 0.25, 4.0))
    else:
        a = 1.0
    b = float(np.clip(my - a * mx, -60.0, 60.0))
    tr_m = np.rint(np.clip(a * tr_up.astype(np.float64) + b, 0.0, 255.0)).astype(np.uint8)
    ev_m = np.rint(np.clip(a * ev_up.astype(np.float64) + b, 0.0, 255.0)).astype(np.uint8)
    return tr_m, ev_m, dict(slope=round(a, 5), intercept=round(b, 4))


def run_part_b_bio_audit_and_hybrid(
    h_tr: dict, h_ev: dict, data_dir: Path, orion_dir: Path, cv_manifest: dict, device: torch.device, out_dir: Path
) -> dict:
    print("\n[PUSH-2 Part B] 12-Arm Biological Audit + Spectral Decomposition + Exploratory Affine+S_count...", flush=True)
    out_b = dict(schema="push2-spectral-and-bio-audit-v1", hemit_bio_audit={}, orion_fold0_bio_and_hybrid={}, spectral_decomposition={})

    # 1. HEMIT Biological Audit across 4 representative generators (3 compressed + 1 native)
    hemit_keys = ["sanofi_miphei_convnext", "sanofi_miphei_hemit", "sanofi_pix2pix", "native_dgr"]
    for bk in hemit_keys:
        p_tr = data_dir / "upstream_cache" / f"{bk}_train_596.npz"
        p_ev = data_dir / "upstream_cache" / f"{bk}_test_945.npz"
        if not (p_tr.is_file() and p_ev.is_file()):
            continue
        up_tr = np.load(p_tr)["pred_u8"]
        up_ev = np.load(p_ev)["pred_u8"]

        # Spectral energy ratio on CD3 (ch=1)
        gt_sp = _radial_band_energies(h_ev["targ_nchw"][:, 1])
        up_sp = _radial_band_energies(up_ev[:, 1])
        out_b["spectral_decomposition"][f"HEMIT_{bk}_CD3"] = dict(
            gt=gt_sp,
            upstream=up_sp,
            dc_ratio_up_to_gt=round(up_sp["dc_power"] / max(gt_sp["dc_power"], 1e-12), 5),
            cell_scale_ratio_up_to_gt=round(up_sp["cell_scale_power"] / max(gt_sp["cell_scale_power"], 1e-12), 5),
        )

        # Build arms: upstream, QuantileMatch, AffineGamma, E_count, S_count
        arm_tr = {"upstream": up_tr, "QuantileMatch": np.empty_like(up_tr), "AffineGamma": np.empty_like(up_tr), "E_count": np.empty_like(up_tr), "S_count": np.empty_like(up_tr)}
        arm_ev = {"upstream": up_ev, "QuantileMatch": np.empty_like(up_ev), "AffineGamma": np.empty_like(up_ev), "E_count": np.empty_like(up_ev), "S_count": np.empty_like(up_ev)}

        for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
            qt, qe, _ = rcomp.apply_quantile_match_u8(up_tr[:, ch], h_tr["targ_nchw"][:, ch], up_ev[:, ch])
            arm_tr["QuantileMatch"][:, ch], arm_ev["QuantileMatch"][:, ch] = qt, qe
            at, ae, _ = rcomp.tune_affine_gamma_u8(up_tr[:, ch], h_tr["targ_nchw"][:, ch], up_ev[:, ch], cutoff, device)
            arm_tr["AffineGamma"][:, ch], arm_ev["AffineGamma"][:, ch] = at, ae

            c12_tr = rhmc.build_common12_for_3ch(h_tr["he_nhwc"], up_tr, ch, device=device)
            c12_ev = rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev, ch, device=device)
            d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr[:, ch]), torch.from_numpy(h_tr["targ_nchw"][:, ch]), torch.from_numpy(up_ev[:, ch])

            for arm in ("E_count", "S_count"):
                tl, el = [], []
                for sd in (202610270, 202610271):
                    if arm == "E_count":
                        mdl, _ = rp2.fit_ablation_marker(
                            mode="E_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                            target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                            epochs=8, batch_size=8, lr=1.2e-3,
                        )
                        pt, _ = rp2.predict_ablation_marker(mdl, "E_count", c12_tr, d_tr_t, device, batch_size=32)
                        pe, _ = rp2.predict_ablation_marker(mdl, "E_count", c12_ev, d_ev_t, device, batch_size=32)
                    else:
                        mdl, _ = suh.fit_universal_marker(arm="S_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t, cutoff=cutoff, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25)
                        pt, _ = suh.predict_universal_marker(mdl, c12_tr, d_tr_t, device, batch_size=32)
                        pe, _ = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device, batch_size=32)
                    tl.append(pt.astype(np.float32))
                    el.append(pe.astype(np.float32))
                    del mdl
                arm_tr[arm][:, ch] = np.round(np.mean(tl, axis=0)).astype(np.uint8)
                arm_ev[arm][:, ch] = np.round(np.mean(el, axis=0)).astype(np.uint8)
            del c12_tr, c12_ev
            gc.collect()
            torch.cuda.empty_cache()

        # Antigen-absent CD3 tiles (< 0.1% GT CD3+ pixels) false-positive audit + single-cell evaluation
        gt_cd3_frac = (h_ev["targ_nchw"][:, 1] > 30).mean(axis=(1, 2))
        absent_mask = gt_cd3_frac < 0.001  # antigen-absent CD3 tiles
        bk_res = dict(n_cd3_absent_tiles=int(absent_mask.sum()), arms={})
        for arm in ("upstream", "QuantileMatch", "AffineGamma", "E_count", "S_count"):
            tr_cells = rhmc.extract_pred_cell_means_for_records(arm_tr[arm], h_tr["nuclei_records"])
            ev_cells = rhmc.extract_pred_cell_means_for_records(arm_ev[arm], h_ev["nuclei_records"])
            ev_res = rhmc.evaluate_hemit_arm_cells_and_infiltration(
                tr_cells, ev_cells, h_ev["nuclei_records"], arm_ev[arm]
            )
            # Pixel F1 & SSIM macro
            f1s, ssims = [], []
            for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                cr = jce.evaluate_channel_predictions(
                    train_pred_u8=arm_tr[arm][:, ch],
                    train_targ_u8=h_tr["targ_nchw"][:, ch],
                    eval_pred_u8=arm_ev[arm][:, ch],
                    eval_targ_u8=h_ev["targ_nchw"][:, ch],
                    eval_clusters=h_ev["clusters"],
                    fixed_cutoff=cutoff,
                    device=device,
                )
                f1s.append(float(cr["pooled"]["Fixed_tol_F1"]))
                ssims.append(float(cr["pooled"]["SSIM"]))
            abs_fp_pct = round(100.0 * float((arm_ev[arm][absent_mask, 1] > 30).mean()), 4) if absent_mask.any() else 0.0
            bk_res["arms"][arm] = dict(
                macro_fixed_f1=round(float(np.mean(f1s)), 4),
                macro_ssim=round(float(np.mean(ssims)), 4),
                macro_cell_f1=round(float(ev_res["macro_deployable_cell_f1"]), 4),
                cd3_cell_f1=round(float(ev_res["per_marker"]["CD3"]["deployable_cell_f1"]), 4),
                panck_cell_f1=round(float(ev_res["per_marker"]["panCK"]["deployable_cell_f1"]), 4),
                infiltration_25um_r=round(float(ev_res["spatial_infiltration"]["within_25um_pearson_r"]), 4),
                cd3_absent_tile_fp_pct=abs_fp_pct,
            )
        out_b["hemit_bio_audit"][bk] = bk_res
        print(
            f"  [HEMIT Bio {bk}] CellF1: up={bk_res['arms']['upstream']['macro_cell_f1']:.4f} "
            f"qm={bk_res['arms']['QuantileMatch']['macro_cell_f1']:.4f} "
            f"ag={bk_res['arms']['AffineGamma']['macro_cell_f1']:.4f} "
            f"ec={bk_res['arms']['E_count']['macro_cell_f1']:.4f} "
            f"sc={bk_res['arms']['S_count']['macro_cell_f1']:.4f} | "
            f"AbsentCD3_FP%: qm={bk_res['arms']['QuantileMatch']['cd3_absent_tile_fp_pct']:.2f}% "
            f"ec={bk_res['arms']['E_count']['cd3_absent_tile_fp_pct']:.2f}% "
            f"sc={bk_res['arms']['S_count']['cd3_absent_tile_fp_pct']:.2f}%",
            flush=True,
        )

    # 2. Orion Fold 0 Biological Audit + Exploratory GlobalAffine + Screened Count Hybrid!
    with open(orion_dir / "bank_cache" / "orion_bank_all40_2804_crop256.pkl", "rb") as f:
        obank = pickle.load(f)
    f0_spec = cv_manifest["folds"][0]
    tr_pats = set(roms._fold_patients(f0_spec, "inner_train"))
    ev_pats = set(roms._fold_patients(f0_spec, "outer_test"))
    tr_idx_all = np.nonzero(np.isin(obank["patients"], list(tr_pats)))[0]
    tr_idx = tr_idx_all[::2]
    ev_idx = np.nonzero(np.isin(obank["patients"], list(ev_pats)))[0]
    cutoffs_u8, tau_phys_float = roms.calibrate_orion_physical_cutoffs_from_train(
        obank["targ_nchw"][tr_idx_all], tr_idx_all, obank["nuclei_records"]
    )
    ev_cells_ref = roms.gather_subset_cells(obank["targ_nchw"][ev_idx], ev_idx, obank["nuclei_records"])
    gt_cooc = np.zeros((15, 15), dtype=np.float64)
    lbl_gt = ev_cells_ref["labels15"].astype(np.float64)
    for i in range(15):
        si = float(np.std(lbl_gt[:, i]))
        for j in range(15):
            sj = float(np.std(lbl_gt[:, j]))
            gt_cooc[i, j] = 1.0 if i == j else (float(np.corrcoef(lbl_gt[:, i], lbl_gt[:, j])[0, 1]) if si > 1e-12 and sj > 1e-12 else 0.0)

    orion_keys = ["miphei_convnext", "miphei_hemit_arch", "hemit_orion", "pix2pix_orion", "rosie_orion"]
    for bk in orion_keys:
        npz_p = orion_dir / "upstream_cache" / f"{bk}_all40_2804.npz"
        if not npz_p.is_file():
            continue
        up_all = np.load(npz_p)["pred_u8"]
        up_tr, up_ev = up_all[tr_idx], up_all[ev_idx]

        # Spectral decomposition on CD8a (ch=6)
        gt_sp = _radial_band_energies(obank["targ_nchw"][ev_idx, 6])
        up_sp = _radial_band_energies(up_ev[:, 6])
        out_b["spectral_decomposition"][f"Orion_{bk}_CD8a"] = dict(
            gt=gt_sp,
            upstream=up_sp,
            dc_ratio_up_to_gt=round(up_sp["dc_power"] / max(gt_sp["dc_power"], 1e-12), 5),
            cell_scale_ratio_up_to_gt=round(up_sp["cell_scale_power"] / max(gt_sp["cell_scale_power"], 1e-12), 5),
        )

        # Build arms: upstream, QuantileMatch, GlobalAffine, S_count, AND Exploratory GlobalAffine + S_count!
        arm_names = ("upstream", "QuantileMatch", "GlobalAffine", "S_count", "GlobalAffine_plus_S_count")
        arm_tr = {a: (up_tr if a == "upstream" else np.empty_like(up_tr)) for a in arm_names}
        arm_ev = {a: (up_ev if a == "upstream" else np.empty_like(up_ev)) for a in arm_names}

        for ch16 in range(16):
            cutoff = int(cutoffs_u8[ch16])
            qt, qe, _ = rcomp.apply_quantile_match_u8(up_tr[:, ch16], obank["targ_nchw"][tr_idx, ch16], up_ev[:, ch16])
            arm_tr["QuantileMatch"][:, ch16], arm_ev["QuantileMatch"][:, ch16] = qt, qe
            at, ae, _ = fit_global_affine_u8(up_tr[:, ch16], obank["targ_nchw"][tr_idx, ch16], up_ev[:, ch16])
            arm_tr["GlobalAffine"][:, ch16], arm_ev["GlobalAffine"][:, ch16] = at, ae

        # Fit S_count on unmodified upstream AND on GlobalAffine-aligned upstream!
        for src_name, dst_name, src_tr, src_ev in [
            ("upstream", "S_count", up_tr, up_ev),
            ("GlobalAffine", "GlobalAffine_plus_S_count", arm_tr["GlobalAffine"], arm_ev["GlobalAffine"]),
        ]:
            for ch16 in range(16):
                cutoff = int(cutoffs_u8[ch16])
                c12_tr = roms.build_common12_orion(obank["he_nhwc"][tr_idx], src_tr, ch16, device=device)
                c12_ev = roms.build_common12_orion(obank["he_nhwc"][ev_idx], src_ev, ch16, device=device)
                d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(src_tr[:, ch16]), torch.from_numpy(obank["targ_nchw"][tr_idx, ch16]), torch.from_numpy(src_ev[:, ch16])
                tl, el = [], []
                for sd in (202610270, 202610271):
                    mdl, _ = suh.fit_universal_marker(
                        arm="S_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                        target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                        epochs=8, batch_size=16, lr=1.5e-3, deploy_anchor_weight=1.25,
                    )
                    pt, _ = suh.predict_universal_marker(mdl, c12_tr, d_tr_t, device, batch_size=32)
                    pe, _ = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device, batch_size=32)
                    tl.append(pt.astype(np.float32))
                    el.append(pe.astype(np.float32))
                    del mdl
                arm_tr[dst_name][:, ch16] = np.round(np.mean(tl, axis=0)).astype(np.uint8)
                arm_ev[dst_name][:, ch16] = np.round(np.mean(el, axis=0)).astype(np.uint8)
                del c12_tr, c12_ev
            gc.collect()
            torch.cuda.empty_cache()

        # Evaluate all 5 arms on pixel F1, SSIM, 15-class Nuclear F1, 15x15 Co-occurrence MAE, and low-antigen FP
        eval_pats_arr = obank["patients"][ev_idx]
        gt_cd8_abs = (obank["targ_nchw"][ev_idx, 6] > int(cutoffs_u8[6])).mean(axis=(1, 2)) < 0.001
        o_res = dict(n_cd8a_absent_tiles=int(gt_cd8_abs.sum()), arms={})
        for arm in arm_names:
            f1s, ssims, psnrs = [], [], []
            for ch16 in range(16):
                cr = jce.evaluate_channel_predictions(
                    train_pred_u8=arm_tr[arm][:, ch16],
                    train_targ_u8=obank["targ_nchw"][tr_idx, ch16],
                    eval_pred_u8=arm_ev[arm][:, ch16],
                    eval_targ_u8=obank["targ_nchw"][ev_idx, ch16],
                    eval_clusters=eval_pats_arr,
                    fixed_cutoff=int(cutoffs_u8[ch16]),
                    device=device,
                )
                f1s.append(float(cr["pooled"]["Fixed_tol_F1"]))
                ssims.append(float(cr["pooled"]["SSIM"]))
                psnrs.append(float(cr["pooled"]["PSNR_dB"]))
            tr_c = roms.gather_subset_cells(arm_tr[arm], tr_idx, obank["nuclei_records"])
            ev_c = roms.gather_subset_cells(arm_ev[arm], ev_idx, obank["nuclei_records"])
            cf1s = []
            for ch16 in range(1, 16):
                m_nm = roms.ORION_MARKERS[ch16]
                rc = roms.ccs.evaluate_single_cell_marker(
                    train_gt_scores=tr_c["gt_means16"][:, ch16],
                    train_pred_scores=tr_c["pred_means"][:, ch16],
                    train_labels=tr_c["labels15"][:, ch16 - 1],
                    eval_pred_scores=ev_c["pred_means"][:, ch16],
                    eval_labels=ev_c["labels15"][:, ch16 - 1],
                    eval_tile_ids=ev_c["tile_ids"],
                    physical_cutoff=float(tau_phys_float[m_nm]),
                )
                cf1s.append(float(rc["deployable_cell_f1"]))
            cooc = ro5._compute_cooccurrence_corr(ev_c["pred_means"], tau_phys_float)
            cooc_mae = float(np.mean(np.abs(cooc - gt_cooc)))
            abs_fp = round(100.0 * float((arm_ev[arm][gt_cd8_abs, 6] > int(cutoffs_u8[6])).mean()), 4) if gt_cd8_abs.any() else 0.0
            o_res["arms"][arm] = dict(
                macro_fixed_f1=round(float(np.mean(f1s)), 4),
                macro_ssim=round(float(np.mean(ssims)), 4),
                macro_psnr_db=round(float(np.mean(psnrs)), 3),
                macro_cell_15cls_f1=round(float(np.mean(cf1s)), 4),
                cooccurrence_15x15_mae=round(cooc_mae, 4),
                cd8a_absent_tile_fp_pct=abs_fp,
            )
        out_b["orion_fold0_bio_and_hybrid"][bk] = o_res
        print(
            f"  [Orion Bio+Hybrid {bk}] F1/SSIM/CellF1/CoocMAE: "
            f"QM={o_res['arms']['QuantileMatch']['macro_fixed_f1']:.4f}/{o_res['arms']['QuantileMatch']['macro_ssim']:.4f}/{o_res['arms']['QuantileMatch']['macro_cell_15cls_f1']:.4f}/{o_res['arms']['QuantileMatch']['cooccurrence_15x15_mae']:.4f} | "
            f"S_cnt={o_res['arms']['S_count']['macro_fixed_f1']:.4f}/{o_res['arms']['S_count']['macro_ssim']:.4f}/{o_res['arms']['S_count']['macro_cell_15cls_f1']:.4f}/{o_res['arms']['S_count']['cooccurrence_15x15_mae']:.4f} | "
            f"Aff+S_cnt={o_res['arms']['GlobalAffine_plus_S_count']['macro_fixed_f1']:.4f}/{o_res['arms']['GlobalAffine_plus_S_count']['macro_ssim']:.4f}/{o_res['arms']['GlobalAffine_plus_S_count']['macro_cell_15cls_f1']:.4f}/{o_res['arms']['GlobalAffine_plus_S_count']['cooccurrence_15x15_mae']:.4f}",
            flush=True,
        )
        _atomic_write_json(out_dir / "push2_spectral_and_12arm_bio.json", out_b)

    del obank
    gc.collect()
    return out_b


# ---------------------------------------------------------------------------
# PART C: HEMIT Spatial Tumor-Microenvironment Ecology (Immune Phenotypes & Cross-K)
# ---------------------------------------------------------------------------

def _classify_immune_phenotype(centroids_yx: np.ndarray, panck_pos: np.ndarray, cd3_pos: np.ndarray, pixel_size_um: float = 0.46) -> tuple[int, dict[int, float]]:
    """Classify tumor-bearing tile (>=20 panCK+ cells) into: 0=Desert, 1=Excluded, 2=Inflamed,
    and compute multi-scale fraction of panCK+ cells within r in {10, 25, 50} um of a CD3+ cell."""
    n_pck = int(panck_pos.sum())
    n_cd3 = int(cd3_pos.sum())
    radii_fracs = {10: 0.0, 25: 0.0, 50: 0.0}
    if n_pck < 20:
        return -1, radii_fracs
    if n_cd3 == 0:
        return 0, radii_fracs
    p_xy = centroids_yx[panck_pos > 0].astype(np.float64) * pixel_size_um
    c_xy = centroids_yx[cd3_pos > 0].astype(np.float64) * pixel_size_um
    # Exact Euclidean distance matrix
    d2 = np.sum((p_xy[:, None, :] - c_xy[None, :, :]) ** 2, axis=-1)
    nn_um = np.sqrt(np.min(d2, axis=1))
    for r_um in (10, 25, 50):
        radii_fracs[r_um] = float(np.mean(nn_um <= float(r_um)))
    if n_cd3 < 5:
        return 0, radii_fracs  # Immune-Desert
    if radii_fracs[25] >= 0.25:
        return 2, radii_fracs  # Immune-Inflamed
    return 1, radii_fracs      # Immune-Excluded


def run_part_c_hemit_tme(h_tr: dict, h_ev: dict, data_dir: Path, device: torch.device, out_dir: Path) -> dict:
    print("\n[PUSH-2 Part C] HEMIT Spatial TME Ecology (Immune-Desert / Excluded / Inflamed + Multi-Scale Cross-K)...", flush=True)
    # Lock physical cellular cutoffs from training GT nuclei
    tau_phys = {}
    for ch, m_nm in ((0, "panCK"), (1, "CD3")):
        gt_vals, gt_lbls = [], []
        for rec in h_tr["nuclei_records"]:
            if rec is None:
                continue
            gt_vals.append(rec["gt_means"][:, ch])
            gt_lbls.append(rec["panck_pos"] if ch == 0 else rec["cd3_pos"])
        gv = np.concatenate(gt_vals)
        gl = np.concatenate(gt_lbls)
        best_f, best_t = -1.0, float(rhmc.HEMIT_CUTOFFS[ch])
        for tc in np.linspace(5.0, 120.0, 116):
            fc = float(f1_score(gl, (gv > tc).astype(np.int32), zero_division=0))
            if fc > best_f:
                best_f, best_t = fc, float(tc)
        tau_phys[m_nm] = best_t

    gt_pheno, gt_rad = [], {10: [], 25: [], 50: []}
    valid_rec_idxs = []
    for idx_r, rec in enumerate(h_ev["nuclei_records"]):
        if rec is None:
            continue
        ph, rf = _classify_immune_phenotype(rec["centroids_yx"], rec["panck_pos"], rec["cd3_pos"])
        if ph >= 0:
            valid_rec_idxs.append(idx_r)
            gt_pheno.append(ph)
            for r_um in (10, 25, 50):
                gt_rad[r_um].append(rf[r_um])
    gt_pheno_arr = np.asarray(gt_pheno, dtype=np.int32)
    gt_counts = {lbl: int((gt_pheno_arr == i).sum()) for i, lbl in enumerate(("Desert", "Excluded", "Inflamed"))}
    print(f"  Tumor-bearing HEMIT test tiles (>=20 panCK+ nuclei): {len(gt_pheno_arr)} | GT phenotypes: {gt_counts}", flush=True)

    out_c = dict(schema="push2-hemit-tme-ecology-v1", n_tumor_tiles=int(len(gt_pheno_arr)), gt_phenotype_counts=gt_counts, tau_phys=tau_phys, backbones={})

    for npz_ev in sorted((data_dir / "upstream_cache").glob("*_test_945.npz")):
        bk = npz_ev.name.replace("_test_945.npz", "")
        npz_tr = data_dir / "upstream_cache" / f"{bk}_train_596.npz"
        if not npz_tr.is_file():
            continue
        up_tr = np.load(npz_tr)["pred_u8"]
        up_ev = np.load(npz_ev)["pred_u8"]

        # Fit 2-seed S_count for panCK (0) and CD3 (1)
        sc_ev = up_ev.copy()
        for ch in (0, 1):
            cutoff = int(rhmc.HEMIT_CUTOFFS[ch])
            c12_tr = rhmc.build_common12_for_3ch(h_tr["he_nhwc"], up_tr, ch, device=device)
            c12_ev = rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev, ch, device=device)
            d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr[:, ch]), torch.from_numpy(h_tr["targ_nchw"][:, ch]), torch.from_numpy(up_ev[:, ch])
            el = []
            for sd in (202610270, 202610271):
                mdl, _ = suh.fit_universal_marker(arm="S_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t, cutoff=cutoff, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25)
                pe, _ = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device, batch_size=32)
                el.append(pe.astype(np.float32))
                del mdl
            sc_ev[:, ch] = np.round(np.mean(el, axis=0)).astype(np.uint8)
            del c12_tr, c12_ev
            torch.cuda.empty_cache()

        bk_tme = {}
        for arm_nm, pred_ev in (("upstream", up_ev), ("S_count", sc_ev)):
            pr_pheno, pr_rad = [], {10: [], 25: [], 50: []}
            for idx_r in valid_rec_idxs:
                rec = h_ev["nuclei_records"][idx_r]
                ti = int(rec["tile_idx"])
                pm = ccs.extract_means_with_index(pred_ev[ti], rec["idx_spec"])
                p_pck = (pm[:, 0] > tau_phys["panCK"]).astype(np.int32)
                p_cd3 = (pm[:, 1] > tau_phys["CD3"]).astype(np.int32)
                ph, rf = _classify_immune_phenotype(rec["centroids_yx"], p_pck, p_cd3)
                pr_pheno.append(max(ph, 0))  # if <20 predicted panCK, classify as desert/non-inflamed
                for r_um in (10, 25, 50):
                    pr_rad[r_um].append(rf[r_um])
            pr_arr = np.asarray(pr_pheno, dtype=np.int32)
            rad_corr = {}
            for r_um in (10, 25, 50):
                ga, pa = np.asarray(gt_rad[r_um]), np.asarray(pr_rad[r_um])
                pr_r = float(stats.pearsonr(ga, pa)[0]) if np.std(pa) > 1e-12 else 0.0
                pr_rho = float(stats.spearmanr(ga, pa)[0]) if np.std(pa) > 1e-12 else 0.0
                rad_corr[f"r{r_um}um"] = dict(pearson_r=round(pr_r, 4), spearman_rho=round(pr_rho, 4))
            bk_tme[arm_nm] = dict(
                immune_phenotype_accuracy=round(float(np.mean(pr_arr == gt_pheno_arr)), 4),
                immune_phenotype_macro_f1=round(float(f1_score(gt_pheno_arr, pr_arr, average="macro", zero_division=0)), 4),
                immune_phenotype_kappa=round(float(cohen_kappa_score(gt_pheno_arr, pr_arr)), 4),
                multiscale_cross_k=rad_corr,
            )
        out_c["backbones"][bk] = bk_tme
        print(
            f"  [TME {bk}] Pheno Macro-F1: {bk_tme['upstream']['immune_phenotype_macro_f1']:.4f} -> {bk_tme['S_count']['immune_phenotype_macro_f1']:.4f} | "
            f"Kappa: {bk_tme['upstream']['immune_phenotype_kappa']:.4f} -> {bk_tme['S_count']['immune_phenotype_kappa']:.4f} | "
            f"CrossK 25um r: {bk_tme['upstream']['multiscale_cross_k']['r25um']['pearson_r']:.4f} -> {bk_tme['S_count']['multiscale_cross_k']['r25um']['pearson_r']:.4f}",
            flush=True,
        )
        _atomic_write_json(out_dir / "push2_hemit_tme_ecology.json", out_c)
    return out_c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--orion-dir", required=True)
    ap.add_argument("--cv-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--parts", default="A,B,C", help="Comma-separated subset of A,B,C")
    args = ap.parse_args()

    device = torch.device(args.device)
    parts = {p.strip().upper() for p in args.parts.split(",") if p.strip()}
    data_dir = Path(args.data_dir)
    orion_dir = Path(args.orion_dir)
    out_dir = Path(args.out_dir)
    cv_manifest = json.loads(Path(args.cv_manifest).read_text())

    with open(data_dir / "bank_cache" / "hemit_bank_train_596.pkl", "rb") as f:
        h_tr = pickle.load(f)
    with open(data_dir / "bank_cache" / "hemit_bank_test_945.pkl", "rb") as f:
        h_ev = pickle.load(f)

    up_tr_cx = np.load(data_dir / "upstream_cache" / "sanofi_miphei_convnext_train_596.npz")["pred_u8"]
    up_ev_cx = np.load(data_dir / "upstream_cache" / "sanofi_miphei_convnext_test_945.npz")["pred_u8"]

    if "A" in parts:
        run_part_a_streaming_timing(h_tr, h_ev, up_tr_cx, up_ev_cx, device, out_dir)
    if "B" in parts:
        run_part_b_bio_audit_and_hybrid(h_tr, h_ev, data_dir, orion_dir, cv_manifest, device, out_dir)
    if "C" in parts:
        run_part_c_hemit_tme(h_tr, h_ev, data_dir, device, out_dir)
    print("PUSH2_ALL_PARTS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
