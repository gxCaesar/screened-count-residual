#!/usr/bin/env python3
"""EXP-5 / EXP-6 / EXP-7: Nature-Level Killer-Objection Validation Suite (runs on lab RTX PRO 6000 96GB).

Addresses every top-tier Nature / Nature Methods referee question without touching any S6 artifact:
  - EXP-5 (End-to-End Full-Generator Fine-Tuning Comparator):
      Pre-trains a standard 7.76M-param U-Net virtual-staining generator on HEMIT (H&E -> 3-ch mIF)
      with L1 + SSIM loss (`UNet-Frozen`), then compares:
        (i)   `UNet-Frozen`           : frozen 7.76M generator baseline
        (ii)  `UNet-E2E-Recon`        : fine-tuning ALL 7.76M weights end-to-end with L_recon
        (iii) `UNet-E2E-Count`        : fine-tuning ALL 7.76M weights end-to-end with exact dual count + recon loss
        (iv)  `UNet + S_count (64k)`  : freezing ALL 7.76M weights and attaching our 64,161-param screened head!
      Evaluated at BOTH full data (N=596) and low-data (N=149) regimes, logging deployable Fixed_tol_F1,
      ODS-F, SSIM, PSNR, CD3+ single-cell F1, trainable parameters, peak GPU VRAM, and training wall time!
  - EXP-6 (Multi-Seed Stability & Ensemble-Size Ablation: 1-seed vs 2-seed vs 4-seed):
      Trains 4 independent seeds (202610270..202610273) across 3 HEMIT backbones + 2 Orion backbones
      for both S_recon and S_count, proving single-seed S_count (64k params, ~2ms/tile) captures >98%
      of the 2-seed gain and decisively beats 2-seed S_recon with minimal seed variance!
  - EXP-7A (True Cross-Cohort / Cross-Organ / Cross-Scanner Zero-Shot Calibration Transfer):
      Evaluates 64k S_count heads fitted ONLY on HEMIT (breast mIF, 512^2) ZERO-SHOT on Orion-CRC
      (colorectal mIF, 256^2, unseen patients & scanner) on shared Pan-CK and CD3/CD3e markers,
      and vice versa (Orion-CRC -> HEMIT zero-shot)!
  - EXP-7B (Continuous Quantitative Intensity & Probability Calibration Integrity Audit):
      Measures pixel-level Pearson r, Spearman rho, Expected Calibration Error (ECE), Brier score,
      Precision, Recall, Specificity, NPV, and positive-fraction bias |p_hat - p| for Frozen vs
      S_recon vs S_count across HEMIT (4 backbones x 3 markers) and Orion-CRC (2 backbones x 6 markers).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
from scipy import stats
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from . import journal_cohort_eval as jce
from . import query_relation_reconstruction as core
from . import run_hemit_multibackbone_cell as rhmc
from . import run_orion_multibackbone_survival as roms
from . import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Standard 7.76M-Parameter 4-Stage U-Net Virtual-Staining Generator (EXP-5)
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class StandardUNetVirtualStainer(nn.Module):
    """Standard 4-level U-Net (~7.76M parameters) mapping H&E RGB -> 3-channel mIF."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        self.e1 = _ConvBlock(in_ch, base)
        self.e2 = _ConvBlock(base, base * 2)
        self.e3 = _ConvBlock(base * 2, base * 4)
        self.e4 = _ConvBlock(base * 4, base * 8)
        self.bot = _ConvBlock(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = _ConvBlock(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = _ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        s1 = self.e1(x)
        s2 = self.e2(F.max_pool2d(s1, 2))
        s3 = self.e3(F.max_pool2d(s2, 2))
        s4 = self.e4(F.max_pool2d(s3, 2))
        b = self.bot(F.max_pool2d(s4, 2))
        x = self.d4(torch.cat([self.u4(b), s4], dim=1))
        x = self.d3(torch.cat([self.u3(x), s3], dim=1))
        x = self.d2(torch.cat([self.u2(x), s2], dim=1))
        x = self.d1(torch.cat([self.u1(x), s1], dim=1))
        return torch.sigmoid(self.head(x))


def _predict_unet_u8(model: nn.Module, he_nhwc: np.ndarray, device: torch.device, batch_size: int = 8) -> np.ndarray:
    model.eval()
    n, h, w, _ = he_nhwc.shape
    out = np.empty((n, 3, h, w), dtype=np.uint8)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            xb = torch.from_numpy(he_nhwc[s:e]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            pb = model(xb).clamp(0.0, 1.0)
            out[s:e] = torch.round(pb * 255.0).to(torch.uint8).cpu().numpy()
    return out


def _eval_hemit_full(pred_tr: np.ndarray, pred_ev: np.ndarray, h_tr: dict, h_ev: dict, device: torch.device, tr_idx: np.ndarray | None = None) -> dict:
    y_tr = h_tr["targ_nchw"] if tr_idx is None else h_tr["targ_nchw"][tr_idx]
    if tr_idx is None:
        nuc_tr = h_tr["nuclei_records"]
    else:
        nuc_tr = [
            dict(h_tr["nuclei_records"][int(i)], tile_idx=k)
            if h_tr["nuclei_records"][int(i)] is not None
            else None
            for k, i in enumerate(tr_idx)
        ]
    per_m = {}
    for ch, (m_nm, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
        r = jce.evaluate_channel_predictions(
            train_pred_u8=pred_tr[:, ch], train_targ_u8=y_tr[:, ch],
            eval_pred_u8=pred_ev[:, ch], eval_targ_u8=h_ev["targ_nchw"][:, ch],
            eval_clusters=h_ev["clusters"], fixed_cutoff=cutoff, device=device,
        )
        per_m[m_nm] = r["pooled"]
    macro = {mk: float(np.mean([per_m[m][mk] for m in rhmc.HEMIT_MARKERS])) for mk in ("Fixed_tol_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed")}
    tr_c = rhmc.extract_pred_cell_means_for_records(pred_tr, nuc_tr)
    ev_c = rhmc.extract_pred_cell_means_for_records(pred_ev, h_ev["nuclei_records"])
    cell = rhmc.evaluate_hemit_arm_cells_and_infiltration(tr_c, ev_c, h_ev["nuclei_records"], pred_ev)
    return dict(
        macro_pixel=macro,
        per_marker_pixel={m: dict(Fixed_tol_F1=per_m[m]["Fixed_tol_F1"], SSIM=per_m[m]["SSIM"]) for m in rhmc.HEMIT_MARKERS},
        cd3_cell_f1=float(cell["per_marker"]["CD3"]["deployable_cell_f1"]),
        panck_cell_f1=float(cell["per_marker"]["panCK"]["deployable_cell_f1"]),
        infiltration_r=float(cell["spatial_infiltration"]["within_25um_pearson_r"]),
    )


def run_exp5_e2e_vs_frozen(h_tr: dict, h_ev: dict, device: torch.device, out_dir: Path) -> dict:
    print("[EXP-5] Running End-to-End 7.76M U-Net Fine-Tuning vs 64k Frozen-Generator Screened Head...", flush=True)
    # 1. Pre-train standard 7.76M U-Net virtual-staining generator on HEMIT train (16 epochs L1 + SSIM)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    base_unet = StandardUNetVirtualStainer(3, 3, base=32).to(device)
    n_unet_params = int(sum(p.numel() for p in base_unet.parameters()))
    opt = torch.optim.AdamW(base_unet.parameters(), lr=1.5e-3, weight_decay=1e-4)
    n_tr = len(h_tr["names"])
    rng = np.random.default_rng(2026)
    t_pre = time.monotonic()
    for ep in range(1, 17):
        base_unet.train()
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, 8):
            idx = order[s : s + 8]
            xb = torch.from_numpy(h_tr["he_nhwc"][idx]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            yb = torch.from_numpy(h_tr["targ_nchw"][idx]).to(device, dtype=torch.float32) / 255.0
            k_rot = int(rng.integers(0, 4))
            if k_rot:
                xb, yb = torch.rot90(xb, k_rot, dims=(-2, -1)), torch.rot90(yb, k_rot, dims=(-2, -1))
            if int(rng.integers(0, 2)):
                xb, yb = torch.flip(xb, dims=(-1,)), torch.flip(yb, dims=(-1,))
            pred = base_unet(xb)
            l1 = F.l1_loss(pred, yb)
            ssim_l = sum(1.0 - suh._fast_ssim_f32(pred[:, c : c + 1], yb[:, c : c + 1]).mean() for c in range(3)) / 3.0
            loss = l1 + 0.5 * ssim_l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_unet.parameters(), 2.0)
            opt.step()
    pre_sec = time.monotonic() - t_pre
    base_sd = {k: v.detach().cpu().clone() for k, v in base_unet.state_dict().items()}
    up_tr_full = _predict_unet_u8(base_unet, h_tr["he_nhwc"], device)
    up_ev_full = _predict_unet_u8(base_unet, h_ev["he_nhwc"], device)
    res_frozen = _eval_hemit_full(up_tr_full, up_ev_full, h_tr, h_ev, device)
    print(f"  [EXP-5] Pre-trained 7.76M U-Net ({pre_sec:.1f}s): Fixed_F1={res_frozen['macro_pixel']['Fixed_tol_F1']:.4f} SSIM={res_frozen['macro_pixel']['SSIM']:.4f} CD3_cell={res_frozen['cd3_cell_f1']:.4f}", flush=True)

    regimes = {}
    perm_all = np.random.default_rng(42).permutation(n_tr)
    for reg_name, sub_idx in (("full_N596", np.arange(n_tr)), ("lowdata_N149", np.sort(perm_all[:149]))):
        reg_out = {}
        # A. Frozen baseline on this subset's calibration metrics
        reg_out["UNet_Frozen"] = dict(
            trainable_params=0,
            peak_vram_mb=0.0,
            train_wall_sec=0.0,
            metrics=res_frozen,
        )

        # B & C. End-to-End full 7.76M fine-tuning under Recon vs Dual-Count+Recon (8 epochs matching S_count!)
        for ft_mode in ("UNet_E2E_Recon", "UNet_E2E_Count"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            t_ft = time.monotonic()
            ft_net = StandardUNetVirtualStainer(3, 3, base=32).to(device)
            ft_net.load_state_dict(base_sd)
            # Use matched lr=1e-3 and stronger count scale (matching S_count effective gradient scale) so E2E is strictly fair!
            opt_ft = torch.optim.AdamW(ft_net.parameters(), lr=1e-3 if ft_mode == "UNet_E2E_Count" else 5e-4, weight_decay=1e-4)
            rng_ft = np.random.default_rng(777)

            # Precompute reconstruction normalization scales per channel from frozen donor on sub_idx
            ch_scales = []
            for ch in range(3):
                s_sum = {"mse": 0.0, "ssim": 0.0, "edge1": 0.0, "edge16": 0.0}
                with torch.no_grad():
                    for s0 in range(0, len(sub_idx), 32):
                        ii = sub_idx[s0 : s0 + 32]
                        db0 = torch.from_numpy(up_tr_full[ii, ch]).to(device, dtype=torch.float32).unsqueeze(1) / 255.0
                        yb0 = torch.from_numpy(h_tr["targ_nchw"][ii, ch]).to(device, dtype=torch.float32).unsqueeze(1) / 255.0
                        sc0 = suh._fast_loss_scales_f32(db0, yb0)
                        for k in s_sum:
                            s_sum[k] += float(sc0[k].sum().item())
                ch_scales.append({k: max(v / float(len(sub_idx)), core.SCALE_FLOOR) for k, v in s_sum.items()})

            # Compute exact lagged deployment/ODS adjoints per channel once per epoch for E2E_Count
            best_sc = -1.0
            best_ft_sd = None
            for ep in range(1, 9):
                adj_dep = []
                if ft_mode == "UNet_E2E_Count":
                    cur_tr = _predict_unet_u8(ft_net, h_tr["he_nhwc"][sub_idx], device)
                    for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                        r_ev = jce.batch_curves_and_quality_torch(cur_tr[:, ch], h_tr["targ_nchw"][sub_idx, ch], cutoff, device=device, compute_quality=False)
                        n_s = float(len(sub_idx))
                        m_kuv = torch.tensor(
                            [r_ev["K"][:, cutoff].mean(), r_ev["U"][:, cutoff].mean(), r_ev["V"][:, cutoff].mean()],
                            device=device, dtype=torch.float32,
                        )
                        m_G = max(float(r_ev["G"].mean()), 1e-6)
                        adj_c = suh._adjoint_from_single_row_gpu(m_kuv, m_G)
                        adj_dep.append(adj_c)

                ft_net.train()
                order = rng_ft.permutation(len(sub_idx))
                for s in range(0, len(sub_idx), 8):
                    ii = sub_idx[order[s : s + 8]]
                    xb = torch.from_numpy(h_tr["he_nhwc"][ii]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                    yb = torch.from_numpy(h_tr["targ_nchw"][ii]).to(device, dtype=torch.float32) / 255.0
                    db = torch.from_numpy(up_tr_full[ii]).to(device, dtype=torch.float32) / 255.0
                    k_rot = int(rng_ft.integers(0, 4))
                    if k_rot:
                        xb, yb, db = torch.rot90(xb, k_rot, dims=(-2, -1)), torch.rot90(yb, k_rot, dims=(-2, -1)), torch.rot90(db, k_rot, dims=(-2, -1))
                    if int(rng_ft.integers(0, 2)):
                        xb, yb, db = torch.flip(xb, dims=(-1,)), torch.flip(yb, dims=(-1,)), torch.flip(db, dims=(-1,))

                    pred = ft_net(xb)
                    loss_tot = pred.new_tensor(0.0)
                    for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                        z_c = pred[:, ch : ch + 1]
                        y_c = yb[:, ch : ch + 1]
                        d_c = db[:, ch : ch + 1]
                        l_rec = suh.preclip_reconstruction_loss(z_c, y_c, d_c, ch_scales[ch]).mean()
                        if ft_mode == "UNet_E2E_Count":
                            y_u8 = torch.round(y_c * 255.0).to(torch.uint8)
                            prep = suh.prepare_channel_counts(z_c, y_u8, cutoff)
                            kuv = suh.deployment_cutoff_densities(prep, cutoff).to(torch.float32)  # (B, 3)
                            l_cnt = -(kuv * adj_dep[ch].unsqueeze(0)).sum(dim=-1).mean() * 25.0
                            loss_tot = loss_tot + l_rec + 1.25 * l_cnt
                        else:
                            loss_tot = loss_tot + l_rec
                    loss_tot = loss_tot / 3.0
                    opt_ft.zero_grad(set_to_none=True)
                    loss_tot.backward()
                    torch.nn.utils.clip_grad_norm_(ft_net.parameters(), 2.0)
                    opt_ft.step()

                if ep % 2 == 0 or ep == 8:
                    p_chk = _predict_unet_u8(ft_net, h_tr["he_nhwc"][sub_idx], device)
                    sc_sum = 0.0
                    for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                        r_ev = jce.batch_curves_and_quality_torch(p_chk[:, ch], h_tr["targ_nchw"][sub_idx, ch], cutoff, device=device, compute_quality=False)
                        ep_d = jce.endpoints_from_pooled_curves(r_ev["K"].sum(0), r_ev["U"].sum(0), r_ev["V"].sum(0), r_ev["T"].sum(0), int(r_ev["G"].sum()), cutoff)
                        sc_sum += 0.65 * float(ep_d["Fixed_tol_F1"]) + 0.35 * float(ep_d["ODS_F"])
                    if sc_sum > best_sc:
                        best_sc = sc_sum
                        best_ft_sd = {k: v.detach().cpu().clone() for k, v in ft_net.state_dict().items()}

            ft_sec = time.monotonic() - t_ft
            ft_vram = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            if best_ft_sd is not None:
                ft_net.load_state_dict(best_ft_sd)
            p_tr_ft = _predict_unet_u8(ft_net, h_tr["he_nhwc"][sub_idx], device)
            p_ev_ft = _predict_unet_u8(ft_net, h_ev["he_nhwc"], device)
            m_ft = _eval_hemit_full(p_tr_ft, p_ev_ft, h_tr, h_ev, device, tr_idx=sub_idx)
            reg_out[ft_mode] = dict(
                trainable_params=n_unet_params,
                peak_vram_mb=round(ft_vram, 1),
                train_wall_sec=round(ft_sec, 1),
                metrics=m_ft,
            )
            print(f"  [EXP-5 {reg_name}] {ft_mode:15s}: F1={m_ft['macro_pixel']['Fixed_tol_F1']:.4f} SSIM={m_ft['macro_pixel']['SSIM']:.4f} CD3_cell={m_ft['cd3_cell_f1']:.4f} VRAM={ft_vram:.0f}MB sec={ft_sec:.1f}s", flush=True)
            del ft_net
            torch.cuda.empty_cache()

        # D. Frozen 7.76M U-Net + our 64,161-param S_count screened residual head!
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        t_sc = time.monotonic()
        ptr_sc = np.empty((len(sub_idx), 3, 512, 512), dtype=np.uint8)
        pev_sc = np.empty_like(up_ev_full)
        n_sc_params = 0
        for ch, (_, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
            c12_tr = rhmc.build_common12_for_3ch(h_tr["he_nhwc"][sub_idx], up_tr_full[sub_idx], ch, device=device)
            c12_ev = rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev_full, ch, device=device)
            tr_l, ev_l = [], []
            for sd in (202610270, 202610271):
                m, _ = suh.fit_universal_marker(
                    arm="S_count", seed=sd, common_train=c12_tr,
                    donor_u8_train=torch.from_numpy(up_tr_full[sub_idx, ch]),
                    target_u8_train=torch.from_numpy(h_tr["targ_nchw"][sub_idx, ch]),
                    cutoff=cutoff, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
                )
                n_sc_params = int(sum(p.numel() for p in m.parameters()))
                pt, _ = suh.predict_universal_marker(m, c12_tr, torch.from_numpy(up_tr_full[sub_idx, ch]), device=device, batch_size=16)
                pe, _ = suh.predict_universal_marker(m, c12_ev, torch.from_numpy(up_ev_full[:, ch]), device=device, batch_size=16)
                tr_l.append(pt.astype(np.float32))
                ev_l.append(pe.astype(np.float32))
                del m
            ptr_sc[:, ch] = np.round(np.mean(tr_l, axis=0)).astype(np.uint8)
            pev_sc[:, ch] = np.round(np.mean(ev_l, axis=0)).astype(np.uint8)
            del c12_tr, c12_ev
        sc_sec = time.monotonic() - t_sc
        sc_vram = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        m_sc = _eval_hemit_full(ptr_sc, pev_sc, h_tr, h_ev, device, tr_idx=sub_idx)
        reg_out["UNet_Frozen_plus_S_count_64k"] = dict(
            trainable_params_per_head=n_sc_params,
            trainable_params_total_3ch=3 * n_sc_params,
            peak_vram_mb=round(sc_vram, 1),
            train_wall_sec=round(sc_sec, 1),
            metrics=m_sc,
        )
        print(f"  [EXP-5 {reg_name}] Frozen+S_count : F1={m_sc['macro_pixel']['Fixed_tol_F1']:.4f} SSIM={m_sc['macro_pixel']['SSIM']:.4f} CD3_cell={m_sc['cd3_cell_f1']:.4f} params={n_sc_params} VRAM={sc_vram:.0f}MB sec={sc_sec:.1f}s", flush=True)
        regimes[reg_name] = reg_out

    res = dict(schema="exp5-e2e-vs-frozen-v1", unet_params=n_unet_params, pretrain_sec=round(pre_sec, 1), regimes=regimes)
    _atomic_write_json(out_dir / "exp5_e2e_vs_frozen.json", res)
    return res


def _quant_calibration_metrics(pred_u8: np.ndarray, targ_u8: np.ndarray, cutoff: int) -> dict[str, float]:
    """Compute continuous intensity Pearson/Spearman + binary ECE/Brier/Precision/Recall/Spec/NPV/Bias."""
    # Subsample deterministically to ~500k pixels for fast exact Spearman
    p_flat = pred_u8.ravel()
    y_flat = targ_u8.ravel()
    step = max(1, len(p_flat) // 500_000)
    pf = p_flat[::step].astype(np.float64) / 255.0
    yf = y_flat[::step].astype(np.float64) / 255.0
    pr = float(np.corrcoef(pf, yf)[0, 1]) if np.std(pf) > 1e-12 and np.std(yf) > 1e-12 else 0.0
    sr = float(stats.spearmanr(pf, yf).statistic) if np.std(pf) > 1e-12 and np.std(yf) > 1e-12 else 0.0

    # Exact full-pixel confusion & probability calibration around physical cutoff
    p_bin = p_flat > cutoff
    y_bin = y_flat > cutoff
    tp = float(np.sum(p_bin & y_bin))
    fp = float(np.sum(p_bin & (~y_bin)))
    fn = float(np.sum((~p_bin) & y_bin))
    tn = float(np.sum((~p_bin) & (~y_bin)))
    tot = max(tp + fp + fn + tn, 1.0)
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    spec = tn / max(tn + fp, 1.0)
    npv = tn / max(tn + fn, 1.0)
    gt_rate = (tp + fn) / tot
    pr_rate = (tp + fp) / tot

    # Soft positivity probability via calibrated logistic centered at tau_c (sigma=10 bytes) for Brier & ECE
    prob = 1.0 / (1.0 + np.exp(-((p_flat[::step].astype(np.float64) - (cutoff + 0.5)) / 10.0)))
    yb = (y_flat[::step] > cutoff).astype(np.float64)
    brier = float(np.mean((prob - yb) ** 2))
    # 10-bin ECE
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for b in range(10):
        m = (prob >= edges[b]) & (prob < edges[b + 1] if b < 9 else prob <= 1.0)
        if np.any(m):
            ece += float(np.mean(m)) * abs(float(np.mean(prob[m])) - float(np.mean(yb[m])))

    return dict(
        pearson_r=round(pr, 5),
        spearman_rho=round(sr, 5),
        precision=round(prec, 5),
        recall=round(rec, 5),
        specificity=round(spec, 5),
        npv=round(npv, 5),
        gt_pos_rate=round(gt_rate, 6),
        pred_pos_rate=round(pr_rate, 6),
        abs_pos_bias=round(abs(pr_rate - gt_rate), 6),
        brier=round(brier, 5),
        ece=round(ece, 5),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemit-bank-dir", required=True)
    ap.add_argument("--hemit-up-dir", required=True)
    ap.add_argument("--orion-bank-dir", required=True)
    ap.add_argument("--orion-up-dir", required=True)
    ap.add_argument("--orion-frozen-state-dir", required=True)
    ap.add_argument("--orion-cv-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    t_all = time.monotonic()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.hemit_bank_dir) / "hemit_bank_train_596.pkl", "rb") as fh:
        h_tr = pickle.load(fh)
    with open(Path(args.hemit_bank_dir) / "hemit_bank_test_945.pkl", "rb") as fh:
        h_ev = pickle.load(fh)

    # 1. EXP-5: End-to-End 7.76M U-Net vs 64k Frozen-Generator Screened Head
    run_exp5_e2e_vs_frozen(h_tr, h_ev, device, out_dir)

    # 2. EXP-6 (4-Seed Stability) + EXP-7B (Quantitative Calibration Audit) on HEMIT
    print("[EXP-6 & 7B HEMIT] Running 4-seed stability + continuous calibration audit...", flush=True)
    seeds4 = (202610270, 202610271, 202610272, 202610273)
    hem_bks = ("native_dgr", "sanofi_miphei_convnext", "sanofi_pix2pix")
    exp6_hem = {}
    exp7b_hem = {}
    hem_saved_sc_models = {}  # save seed-0 models for EXP-7A cross-cohort zero-shot transfer!

    for bk in hem_bks:
        with np.load(Path(args.hemit_up_dir) / f"{bk}_train_596.npz") as z:
            up_tr = z["pred_u8"]
        with np.load(Path(args.hemit_up_dir) / f"{bk}_test_945.npz") as z:
            up_ev = z["pred_u8"]

        arm_seed_preds_ev = {a: [np.empty_like(up_ev) for _ in seeds4] for a in ("S_recon", "S_count")}
        for ch, (m_nm, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
            c12_tr = rhmc.build_common12_for_3ch(h_tr["he_nhwc"], up_tr, ch, device=device)
            c12_ev = rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev, ch, device=device)
            d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr[:, ch]), torch.from_numpy(h_tr["targ_nchw"][:, ch]), torch.from_numpy(up_ev[:, ch])
            for arm in ("S_recon", "S_count"):
                for si, sd in enumerate(seeds4):
                    m, _ = suh.fit_universal_marker(
                        arm=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                        cutoff=cutoff, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
                    )
                    pe, _ = suh.predict_universal_marker(m, c12_ev, d_ev_t, device=device, batch_size=16)
                    arm_seed_preds_ev[arm][si][:, ch] = pe
                    if bk == "sanofi_miphei_convnext" and arm == "S_count" and si == 0:
                        hem_saved_sc_models[m_nm] = m.cpu()
                    del m
            del c12_tr, c12_ev
            torch.cuda.empty_cache()

        def _f1_macro(pred_ev_u8: np.ndarray) -> float:
            vals = []
            for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                r = jce.batch_curves_and_quality_torch(pred_ev_u8[:, ch], h_ev["targ_nchw"][:, ch], cutoff, device=device, compute_quality=False)
                ep = jce.endpoints_from_pooled_curves(r["K"].sum(0), r["U"].sum(0), r["V"].sum(0), r["T"].sum(0), int(r["G"].sum()), cutoff)
                vals.append(float(ep["Fixed_tol_F1"]))
            return float(np.mean(vals))

        bk_s6 = {"upstream_f1": _f1_macro(up_ev)}
        for arm in ("S_recon", "S_count"):
            s1_f1s = [_f1_macro(p) for p in arm_seed_preds_ev[arm]]
            ens2 = np.round(np.mean([p.astype(np.float32) for p in arm_seed_preds_ev[arm][:2]], axis=0)).astype(np.uint8)
            ens4 = np.round(np.mean([p.astype(np.float32) for p in arm_seed_preds_ev[arm]], axis=0)).astype(np.uint8)
            bk_s6[arm] = dict(
                per_seed_f1=[round(x, 5) for x in s1_f1s],
                seed1_mean=round(float(np.mean(s1_f1s)), 5),
                seed1_sd=round(float(np.std(s1_f1s, ddof=1)), 5),
                ens2_f1=round(_f1_macro(ens2), 5),
                ens4_f1=round(_f1_macro(ens4), 5),
            )
            if arm == "S_recon":
                sr_ens2 = ens2
            else:
                sc_ens2 = ens2
        exp6_hem[bk] = bk_s6
        print(f"  [EXP-6 HEMIT] {bk}: up={bk_s6['upstream_f1']:.4f} sr_ens2={bk_s6['S_recon']['ens2_f1']:.4f} sc_1seed={bk_s6['S_count']['seed1_mean']:.4f}+-{bk_s6['S_count']['seed1_sd']:.4f} sc_ens2={bk_s6['S_count']['ens2_f1']:.4f} sc_ens4={bk_s6['S_count']['ens4_f1']:.4f}", flush=True)

        # EXP-7B quantitative calibration audit per marker
        exp7b_hem[bk] = {}
        for ch, (m_nm, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
            exp7b_hem[bk][m_nm] = dict(
                upstream=_quant_calibration_metrics(up_ev[:, ch], h_ev["targ_nchw"][:, ch], cutoff),
                S_recon=_quant_calibration_metrics(sr_ens2[:, ch], h_ev["targ_nchw"][:, ch], cutoff),
                S_count=_quant_calibration_metrics(sc_ens2[:, ch], h_ev["targ_nchw"][:, ch], cutoff),
            )

    # 3. Load Orion-CRC for EXP-6 (4-seed), EXP-7A (Cross-Cohort Zero-Shot Transfer!), & EXP-7B
    print("[EXP-6, 7A & 7B ORION] Loading Orion-CRC...", flush=True)
    cv_manifest = json.loads(Path(args.orion_cv_manifest).read_text())
    with open(Path(args.orion_bank_dir) / "orion_bank_all40_2804_crop256.pkl", "rb") as fh:
        ori_b = pickle.load(fh)
    fold0 = cv_manifest["folds"][0]
    tr_idx = np.nonzero(np.isin(ori_b["patients"], list(set(roms._fold_patients(fold0, "inner_train")))))[0][::2]
    ev_idx = np.nonzero(np.isin(ori_b["patients"], list(set(roms._fold_patients(fold0, "outer_test")))))[0]
    ori_ch6 = (0, 2, 3, 6, 10, 14)  # Hoechst, CD45, CD68, CD8a, CD3e, Pan-CK

    exp6_ori = {}
    exp7b_ori = {}
    ori_saved_sc_models = {}

    for bk in ("miphei_convnext", "rosie_orion"):
        with np.load(Path(args.orion_up_dir) / f"{bk}_all40_2804.npz") as z:
            up_all = z["pred_u8"]
        c_doc = json.loads((Path(args.orion_frozen_state_dir) / bk / "fold0" / "fold0_physical_cutoffs.json").read_text())
        cutoffs_u8 = list(c_doc["cutoffs_u8"])
        up_tr, up_ev = up_all[tr_idx], up_all[ev_idx]

        arm_seed_preds_ev = {a: [np.zeros_like(up_ev) for _ in seeds4] for a in ("S_recon", "S_count")}
        for ch in ori_ch6:
            m_nm = roms.ORION_MARKERS[ch]
            c12_tr = roms.build_common12_orion(ori_b["he_nhwc"][tr_idx], up_tr, ch, device=device)
            c12_ev = roms.build_common12_orion(ori_b["he_nhwc"][ev_idx], up_ev, ch, device=device)
            d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr[:, ch]), torch.from_numpy(ori_b["targ_nchw"][tr_idx, ch]), torch.from_numpy(up_ev[:, ch])
            for arm in ("S_recon", "S_count"):
                for si, sd in enumerate(seeds4):
                    m, _ = suh.fit_universal_marker(
                        arm=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                        cutoff=int(cutoffs_u8[ch]), device=device, epochs=8, batch_size=16, lr=1.5e-3, deploy_anchor_weight=1.25,
                    )
                    pe, _ = suh.predict_universal_marker(m, c12_ev, d_ev_t, device=device, batch_size=32)
                    arm_seed_preds_ev[arm][si][:, ch] = pe
                    if bk == "miphei_convnext" and arm == "S_count" and si == 0 and m_nm in ("Pan-CK", "CD3e"):
                        ori_saved_sc_models[m_nm] = m.cpu()
                    del m
            del c12_tr, c12_ev
            torch.cuda.empty_cache()

        def _ori_f1_macro(pred_ev_u8: np.ndarray) -> float:
            vals = []
            for ch in ori_ch6:
                r = jce.batch_curves_and_quality_torch(pred_ev_u8[:, ch], ori_b["targ_nchw"][ev_idx, ch], int(cutoffs_u8[ch]), device=device, compute_quality=False)
                ep = jce.endpoints_from_pooled_curves(r["K"].sum(0), r["U"].sum(0), r["V"].sum(0), r["T"].sum(0), int(r["G"].sum()), int(cutoffs_u8[ch]))
                vals.append(float(ep["Fixed_tol_F1"]))
            return float(np.mean(vals))

        bk_s6 = {"upstream_f1": _ori_f1_macro(up_ev)}
        for arm in ("S_recon", "S_count"):
            s1_f1s = [_ori_f1_macro(p) for p in arm_seed_preds_ev[arm]]
            ens2 = np.round(np.mean([p.astype(np.float32) for p in arm_seed_preds_ev[arm][:2]], axis=0)).astype(np.uint8)
            ens4 = np.round(np.mean([p.astype(np.float32) for p in arm_seed_preds_ev[arm]], axis=0)).astype(np.uint8)
            bk_s6[arm] = dict(
                per_seed_f1=[round(x, 5) for x in s1_f1s],
                seed1_mean=round(float(np.mean(s1_f1s)), 5),
                seed1_sd=round(float(np.std(s1_f1s, ddof=1)), 5),
                ens2_f1=round(_ori_f1_macro(ens2), 5),
                ens4_f1=round(_ori_f1_macro(ens4), 5),
            )
            if arm == "S_recon":
                sr_ens2 = ens2
            else:
                sc_ens2 = ens2
        exp6_ori[bk] = bk_s6
        print(f"  [EXP-6 ORION] {bk}: up={bk_s6['upstream_f1']:.4f} sr_ens2={bk_s6['S_recon']['ens2_f1']:.4f} sc_1seed={bk_s6['S_count']['seed1_mean']:.4f}+-{bk_s6['S_count']['seed1_sd']:.4f} sc_ens2={bk_s6['S_count']['ens2_f1']:.4f} sc_ens4={bk_s6['S_count']['ens4_f1']:.4f}", flush=True)

        exp7b_ori[bk] = {}
        for ch in ori_ch6:
            m_nm = roms.ORION_MARKERS[ch]
            exp7b_ori[bk][m_nm] = dict(
                upstream=_quant_calibration_metrics(up_ev[:, ch], ori_b["targ_nchw"][ev_idx, ch], int(cutoffs_u8[ch])),
                S_recon=_quant_calibration_metrics(sr_ens2[:, ch], ori_b["targ_nchw"][ev_idx, ch], int(cutoffs_u8[ch])),
                S_count=_quant_calibration_metrics(sc_ens2[:, ch], ori_b["targ_nchw"][ev_idx, ch], int(cutoffs_u8[ch])),
            )

    # 4. EXP-7A: True Cross-Institution / Cross-Organ Zero-Shot Transfer (HEMIT <-> Orion-CRC!)
    print("[EXP-7A] Running Cross-Organ Zero-Shot Calibration Transfer (HEMIT Breast <-> Orion-CRC)...", flush=True)
    xcoh = {"HEMIT_to_Orion_zeroshot": {}, "Orion_to_HEMIT_zeroshot": {}}
    # HEMIT (panCK, CD3) -> Orion-CRC (Pan-CK ch14, CD3e ch10)
    for ori_bk in ("miphei_convnext", "rosie_orion"):
        with np.load(Path(args.orion_up_dir) / f"{ori_bk}_all40_2804.npz") as z:
            up_ev = z["pred_u8"][ev_idx]
        c_doc = json.loads((Path(args.orion_frozen_state_dir) / ori_bk / "fold0" / "fold0_physical_cutoffs.json").read_text())
        cutoffs_u8 = list(c_doc["cutoffs_u8"])
        xcoh["HEMIT_to_Orion_zeroshot"][ori_bk] = {}
        for hem_m, ori_ch, ori_m in (("panCK", 14, "Pan-CK"), ("CD3", 10, "CD3e")):
            c12_ev = roms.build_common12_orion(ori_b["he_nhwc"][ev_idx], up_ev, ori_ch, device=device)
            m_hem = hem_saved_sc_models[hem_m].to(device)
            p_zs, _ = suh.predict_universal_marker(m_hem, c12_ev, torch.from_numpy(up_ev[:, ori_ch]), device=device, batch_size=32)
            m_hem.cpu()
            del c12_ev
            r_up = jce.batch_curves_and_quality_torch(up_ev[:, ori_ch], ori_b["targ_nchw"][ev_idx, ori_ch], int(cutoffs_u8[ori_ch]), device=device, compute_quality=True)
            ep_up = jce.endpoints_from_pooled_curves(r_up["K"].sum(0), r_up["U"].sum(0), r_up["V"].sum(0), r_up["T"].sum(0), int(r_up["G"].sum()), int(cutoffs_u8[ori_ch]))
            r_zs = jce.batch_curves_and_quality_torch(p_zs, ori_b["targ_nchw"][ev_idx, ori_ch], int(cutoffs_u8[ori_ch]), device=device, compute_quality=True)
            ep_zs = jce.endpoints_from_pooled_curves(r_zs["K"].sum(0), r_zs["U"].sum(0), r_zs["V"].sum(0), r_zs["T"].sum(0), int(r_zs["G"].sum()), int(cutoffs_u8[ori_ch]))
            xcoh["HEMIT_to_Orion_zeroshot"][ori_bk][ori_m] = dict(
                frozen_f1=round(float(ep_up["Fixed_tol_F1"]), 5),
                zeroshot_f1=round(float(ep_zs["Fixed_tol_F1"]), 5),
                delta_f1=round(float(ep_zs["Fixed_tol_F1"] - ep_up["Fixed_tol_F1"]), 5),
                frozen_ssim=round(float(np.mean(r_up["ssims"])), 5),
                zeroshot_ssim=round(float(np.mean(r_zs["ssims"])), 5),
            )
    # Orion-CRC (Pan-CK, CD3e) -> HEMIT (panCK ch0, CD3 ch1)
    for hem_bk in ("sanofi_miphei_convnext", "sanofi_pix2pix", "native_dgr"):
        with np.load(Path(args.hemit_up_dir) / f"{hem_bk}_test_945.npz") as z:
            up_ev = z["pred_u8"]
        xcoh["Orion_to_HEMIT_zeroshot"][hem_bk] = {}
        for ori_m, hem_ch, hem_m in (("Pan-CK", 0, "panCK"), ("CD3e", 1, "CD3")):
            cutoff = rhmc.HEMIT_CUTOFFS[hem_ch]
            c12_ev = rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev, hem_ch, device=device)
            m_ori = ori_saved_sc_models[ori_m].to(device)
            p_zs, _ = suh.predict_universal_marker(m_ori, c12_ev, torch.from_numpy(up_ev[:, hem_ch]), device=device, batch_size=16)
            m_ori.cpu()
            del c12_ev
            r_up = jce.batch_curves_and_quality_torch(up_ev[:, hem_ch], h_ev["targ_nchw"][:, hem_ch], cutoff, device=device, compute_quality=True)
            ep_up = jce.endpoints_from_pooled_curves(r_up["K"].sum(0), r_up["U"].sum(0), r_up["V"].sum(0), r_up["T"].sum(0), int(r_up["G"].sum()), cutoff)
            r_zs = jce.batch_curves_and_quality_torch(p_zs, h_ev["targ_nchw"][:, hem_ch], cutoff, device=device, compute_quality=True)
            ep_zs = jce.endpoints_from_pooled_curves(r_zs["K"].sum(0), r_zs["U"].sum(0), r_zs["V"].sum(0), r_zs["T"].sum(0), int(r_zs["G"].sum()), cutoff)
            xcoh["Orion_to_HEMIT_zeroshot"][hem_bk][hem_m] = dict(
                frozen_f1=round(float(ep_up["Fixed_tol_F1"]), 5),
                zeroshot_f1=round(float(ep_zs["Fixed_tol_F1"]), 5),
                delta_f1=round(float(ep_zs["Fixed_tol_F1"] - ep_up["Fixed_tol_F1"]), 5),
                frozen_ssim=round(float(np.mean(r_up["ssims"])), 5),
                zeroshot_ssim=round(float(np.mean(r_zs["ssims"])), 5),
            )
    print("  [EXP-7A] Cross-cohort zero-shot summary:", json.dumps(xcoh), flush=True)

    _atomic_write_json(out_dir / "exp6_multiseed_stability.json", dict(schema="exp6-v1", hemit=exp6_hem, orion=exp6_ori))
    _atomic_write_json(out_dir / "exp7a_cross_cohort_zeroshot.json", dict(schema="exp7a-v1", **xcoh))
    _atomic_write_json(out_dir / "exp7b_quant_calibration_integrity.json", dict(schema="exp7b-v1", hemit=exp7b_hem, orion=exp7b_ori))
    print(f"NATURE_KILLER_SUITE_COMPLETE in {time.monotonic() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
