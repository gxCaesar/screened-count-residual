#!/usr/bin/env python3
"""EXP-2: Comprehensive Mechanistic, Data-Efficiency, Scale, Transfer & Throughput Suite.

Runs 7 self-contained, high-impact Nature Methods experiments without modifying any immutable S6 file:
  (2A) Data Efficiency Curve       : HEMIT N in {16, 32, 64, 149, 298, 596} tiles (4 backbones)
                                     + Orion P in {2, 4, 8, 16, 26} training patients (2 backbones).
  (2B) 12-Ch Feature Ablation      : target_only, no_he, no_struct, no_geom, full_12ch (4 HEMIT + 2 Orion).
  (2C) Multi-Scale Offset Sweep    : 1-px, (1,4), (1,8), (1,16), (1,32) px open-lattice Neumann Laplacian
                                     (each normalized to spec(L) in [0,1], kappa_2(I+L) <= 2) on 4 HEMIT + 2 Orion.
  (2D) lambda_dep Pareto Frontier  : deploy_anchor_weight in {0.0, 0.25, 0.75, 1.25, 2.5} on 4 HEMIT + 2 Orion.
  (2E) Cross-Generator Transfer    : 5x5 zero-shot transfer matrix across HEMIT generators + single pooled
                                     generator-agnostic universal 64k head evaluated on all 5.
  (2F) H&E Stain Perturbation Test : Controlled RGB/brightness/contrast/stain-channel jitter sigma in
                                     {0.0, 0.08, 0.16, 0.24} at inference without refitting (4 HEMIT).
  (2G) Multi-Resolution Profiling  : Exact GPU (H100) and CPU latency (ms/tile), tiles/sec, peak VRAM (MB),
                                     and projected 10,000-tile WSI overhead at 256^2, 512^2, 1024^2, 2048^2.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import time
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from . import journal_cohort_eval as jce
from . import query_relation_reconstruction as core
from . import run_hemit_multibackbone_cell as rhmc
from . import run_orion_multibackbone_survival as roms
from . import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: Mapping) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _eval_3ch_macro(
    pred_tr: np.ndarray,
    targ_tr: np.ndarray,
    pred_ev: np.ndarray,
    targ_ev: np.ndarray,
    clusters: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    per_m = {}
    for ch, (m_nm, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
        r = jce.evaluate_channel_predictions(
            train_pred_u8=pred_tr[:, ch], train_targ_u8=targ_tr[:, ch],
            eval_pred_u8=pred_ev[:, ch], eval_targ_u8=targ_ev[:, ch],
            eval_clusters=clusters, fixed_cutoff=cutoff, device=device,
        )
        per_m[m_nm] = r["pooled"]
    return {
        mk: float(np.mean([per_m[m][mk] for m in rhmc.HEMIT_MARKERS]))
        for mk in ("Fixed_tol_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed")
    }


def _eval_orion_subset_macro(
    pred_tr: np.ndarray,
    targ_tr: np.ndarray,
    pred_ev: np.ndarray,
    targ_ev: np.ndarray,
    eval_pats: np.ndarray,
    cutoffs_u8: list[int],
    ch_indices: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    per_m = {}
    for ch in ch_indices:
        m_nm = roms.ORION_MARKERS[ch]
        r = jce.evaluate_channel_predictions(
            train_pred_u8=pred_tr[:, ch], train_targ_u8=targ_tr[:, ch],
            eval_pred_u8=pred_ev[:, ch], eval_targ_u8=targ_ev[:, ch],
            eval_clusters=eval_pats, fixed_cutoff=int(cutoffs_u8[ch]), device=device,
        )
        per_m[m_nm] = r["pooled"]
    return {
        mk: float(np.mean([per_m[roms.ORION_MARKERS[ch]][mk] for ch in ch_indices]))
        for mk in ("Fixed_tol_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed")
    }


# ---------------------------------------------------------------------------
# Parameterized Two-Offset Open-Lattice Neumann Laplacian for Scale Sweep (2C)
# ---------------------------------------------------------------------------

def _laplacian_step(x: Tensor, step: int) -> Tensor:
    """Open-boundary axial graph Laplacian sum_{(p,q) in E_step} (x_p - x_q) without normalization."""
    out = torch.zeros_like(x)
    if step < x.shape[-1]:
        dx = x[..., :, step:] - x[..., :, :-step]
        out[..., :, :-step] = out[..., :, :-step] - dx
        out[..., :, step:] = out[..., :, step:] + dx
    if step < x.shape[-2]:
        dy = x[..., step:, :] - x[..., :-step, :]
        out[..., :-step, :] = out[..., :-step, :] - dy
        out[..., step:, :] = out[..., step:, :] + dy
    return out


def fast_laplacian_custom(x: Tensor, step2: int) -> Tensor:
    """Normalized open-lattice Neumann Laplacian with spec(L) in [0, 1] and kappa_2(I+L) <= 2."""
    if step2 <= 0:
        return 0.125 * _laplacian_step(x, 1)
    return 0.0625 * (_laplacian_step(x, 1) + _laplacian_step(x, step2))


class _CustomScreenedSolve(torch.autograd.Function):
    @staticmethod
    def _cg(rhs: Tensor, step2: int, n_iter: int = 10) -> Tensor:
        sol = torch.zeros_like(rhs)
        res = rhs.clone()
        direction = res.clone()
        rr = res.square().sum(dim=(-2, -1), keepdim=True)
        tiny = torch.finfo(rhs.dtype).tiny
        for _ in range(n_iter):
            op = direction + fast_laplacian_custom(direction, step2)
            denom = (direction * op).sum(dim=(-2, -1), keepdim=True).clamp_min(tiny)
            alpha = rr / denom
            sol = sol + alpha * direction
            res = res - alpha * op
            new_rr = res.square().sum(dim=(-2, -1), keepdim=True)
            beta = new_rr / rr.clamp_min(tiny)
            direction = res + beta * direction
            rr = new_rr
        return sol

    @staticmethod
    def forward(ctx, rhs: Tensor, step2: int) -> Tensor:
        ctx.step2 = step2
        ctx.input_dtype = rhs.dtype
        return _CustomScreenedSolve._cg(rhs.to(torch.float32), step2, n_iter=10).to(rhs.dtype)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return _CustomScreenedSolve._cg(grad.to(torch.float32), ctx.step2, n_iter=10).to(ctx.input_dtype), None


def fit_and_predict_custom_scale(
    *,
    step2: int,
    seed: int,
    c12_tr: Tensor,
    d_tr_t: Tensor,
    y_tr_t: Tensor,
    c12_ev: Tensor,
    d_ev_t: Tensor,
    cutoff: int,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 1.2e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit S_count with custom open-lattice offset (1, step2) (or 1-only if step2==0) and return (tr, ev, dc_max)."""
    # Monkey-patch fast_sc_forward cleanly inside a local context via wrapper using suh.fit_universal_marker!
    orig_fwd = suh.fast_sc_forward

    def _custom_fwd(model: core.RelationModel, common: Tensor, inference: bool = False):
        dtype = model.token_projection.weight.dtype
        c = common.to(dtype=dtype)
        z16 = model.token_projection.weight.new_zeros((c.shape[0], 16, *c.shape[-2:]))
        h = model.backbone(torch.cat((c, z16), dim=1))
        pot = suh.fast_unary_potential(model, h)
        if inference:
            rhs_f64 = fast_laplacian_custom(pot.to(torch.float64), step2)
            rhs_norm = rhs_f64.square().sum(dim=(-2, -1), keepdim=True).sqrt()
            if float(rhs_norm.max().item()) == 0.0:
                sol = torch.zeros_like(rhs_f64)
                rel_max = 0.0
            else:
                sol = _CustomScreenedSolve._cg(rhs_f64, step2, n_iter=22)
                sol = sol - sol.mean(dim=(-2, -1), keepdim=True)
                sol = sol - sol.mean(dim=(-2, -1), keepdim=True)
                rem = (sol + fast_laplacian_custom(sol, step2)) - rhs_f64
                rel_max = float((rem.square().sum(dim=(-2, -1), keepdim=True).sqrt() / rhs_norm.clamp_min(1e-30)).max().item())
            return sol, dict(
                relative_residual_max=rel_max,
                absolute_residual_max=0.0,
                zero_rhs_count=int((rhs_norm == 0.0).sum().item()),
                zero_rhs_absolute_residual_max=0.0,
                solver_dtype="float64",
                rhs=rhs_f64.detach(),
                residual_mean_max=float(sol.detach().mean(dim=(-2, -1)).abs().max().item()),
                arm=model.arm,
            )
        rhs = fast_laplacian_custom(pot.to(torch.float32), step2)
        sol = _CustomScreenedSolve.apply(rhs, step2)
        return sol, dict(
            relative_residual_max=0.0,
            absolute_residual_max=0.0,
            zero_rhs_count=0,
            zero_rhs_absolute_residual_max=0.0,
            solver_dtype="float32",
            rhs=rhs.detach(),
            residual_mean_max_tensor=sol.detach().mean(dim=(-2, -1)).abs().max(),
            arm=model.arm,
        )

    suh.fast_sc_forward = _custom_fwd
    try:
        model, _ = suh.fit_universal_marker(
            arm="S_count", seed=seed, common_train=c12_tr, donor_u8_train=d_tr_t,
            target_u8_train=y_tr_t, cutoff=cutoff, device=device,
            epochs=epochs, batch_size=batch_size, lr=lr, deploy_anchor_weight=1.25,
        )
        p_tr, _ = suh.predict_universal_marker(model, c12_tr, d_tr_t, device=device, batch_size=max(16, batch_size))
        p_ev, diag = suh.predict_universal_marker(model, c12_ev, d_ev_t, device=device, batch_size=max(16, batch_size))
    finally:
        suh.fast_sc_forward = orig_fwd
    return p_tr, p_ev, float(diag.get("residual_mean_max", 0.0))


def _mask_common12(c12: Tensor, variant: str) -> Tensor:
    """Apply channel ablation mask to standardized 12-channel coordinate tensor without altering parameter count."""
    if variant == "full_12ch":
        return c12
    out = c12.clone()
    if variant == "target_only":
        # Keep ONLY ch 6 (target b_c)
        for ch in range(12):
            if ch != 6:
                out[:, ch] = 0
    elif variant == "no_he":
        # Zero ch 0..5 (H&E RGB + H&E luminance gradients)
        out[:, 0:6] = 0
    elif variant == "no_struct":
        # Zero ch 7..8 (structural reference b_struct & panel mean b_mean)
        out[:, 7:9] = 0
    elif variant == "no_geom":
        # Zero ch 9..11 (target gradients & chamfer-5 peak halo)
        out[:, 9:12] = 0
    else:
        raise ValueError(f"Unknown feature variant: {variant!r}")
    return out


def _perturb_he_u8(he_nhwc: np.ndarray, sigma: float, seed: int = 2026) -> np.ndarray:
    """Controlled H&E stain-color / scanner brightness-contrast-channel jitter at inference."""
    if sigma <= 0.0:
        return he_nhwc
    rng = np.random.default_rng(seed)
    x = he_nhwc.astype(np.float32) / 255.0
    n = x.shape[0]
    # Per-tile smooth multiplicative stain gain + additive background illumination shift across RGB
    gains = np.clip(rng.normal(1.0, sigma, size=(n, 1, 1, 3)), 0.55, 1.45).astype(np.float32)
    bias = np.clip(rng.normal(0.0, 0.4 * sigma, size=(n, 1, 1, 3)), -0.20, 0.20).astype(np.float32)
    y = np.clip(x * gains + bias, 0.0, 1.0)
    return np.round(y * 255.0).astype(np.uint8)


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
    seed = 202610270  # Representative preregistered seed 1 for rapid dense multi-axis sweeps

    # Load HEMIT banks
    with open(Path(args.hemit_bank_dir) / "hemit_bank_train_596.pkl", "rb") as fh:
        h_tr = pickle.load(fh)
    with open(Path(args.hemit_bank_dir) / "hemit_bank_test_945.pkl", "rb") as fh:
        h_ev = pickle.load(fh)

    hemit_4 = ("sanofi_miphei_convnext", "sanofi_miphei_hemit", "sanofi_pix2pix", "native_dgr")
    hemit_5 = ("native_dgr", "native_unistainnet_uni2h", "sanofi_miphei_convnext", "sanofi_miphei_hemit", "sanofi_pix2pix")
    up_tr_map, up_ev_map = {}, {}
    for bk in hemit_5:
        with np.load(Path(args.hemit_up_dir) / f"{bk}_train_596.npz") as z:
            up_tr_map[bk] = z["pred_u8"]
        with np.load(Path(args.hemit_up_dir) / f"{bk}_test_945.npz") as z:
            up_ev_map[bk] = z["pred_u8"]

    # Precompute HEMIT common12 tensors once per backbone to make all sweeps blazing fast!
    c12_tr_hem = {bk: [rhmc.build_common12_for_3ch(h_tr["he_nhwc"], up_tr_map[bk], ch, device=device) for ch in range(3)] for bk in hemit_5}
    c12_ev_hem = {bk: [rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev_map[bk], ch, device=device) for ch in range(3)] for bk in hemit_5}

    # ------------------------------------------------------------------
    # (2G) Multi-Resolution GPU & CPU Throughput / Peak VRAM Profiling
    # ------------------------------------------------------------------
    print("[2G] Benchmarking multi-resolution GPU & CPU throughput...", flush=True)
    prof_rows = []
    for hw in (256, 512, 1024, 2048):
        m_gpu = suh.build_coordinate_screened_model(seed, device).eval()
        x_gpu = torch.rand((1, 12, hw, hw), device=device, dtype=torch.float32)
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                suh.fast_sc_forward(m_gpu, x_gpu, inference=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        n_rep = 15 if hw <= 1024 else 6
        ts = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_rep):
                suh.fast_sc_forward(m_gpu, x_gpu, inference=True)
        torch.cuda.synchronize(device)
        gpu_ms_1seed = (time.perf_counter() - ts) * 1000.0 / n_rep
        gpu_vram_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        del m_gpu, x_gpu
        torch.cuda.empty_cache()

        # CPU benchmark (256, 512, 1024)
        cpu_ms_1seed = float("nan")
        if hw <= 1024:
            m_cpu = suh.build_coordinate_screened_model(seed, torch.device("cpu")).eval()
            x_cpu = torch.rand((1, 12, hw, hw), dtype=torch.float32)
            with torch.no_grad():
                suh.fast_sc_forward(m_cpu, x_cpu, inference=True)
            n_c = 5 if hw <= 512 else 2
            tc = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_c):
                    suh.fast_sc_forward(m_cpu, x_cpu, inference=True)
            cpu_ms_1seed = (time.perf_counter() - tc) * 1000.0 / n_c
            del m_cpu, x_cpu

        prof_rows.append(dict(
            resolution=f"{hw}x{hw}",
            hw=hw,
            gpu_name=torch.cuda.get_device_name(device),
            gpu_ms_2seed=round(2.0 * gpu_ms_1seed, 3),
            gpu_tiles_per_sec_2seed=round(1000.0 / (2.0 * gpu_ms_1seed), 2),
            gpu_wsi_10k_sec_2seed=round(10000.0 * 2.0 * gpu_ms_1seed / 1000.0, 1),
            gpu_peak_vram_mb=round(gpu_vram_mb, 2),
            cpu_ms_2seed=round(2.0 * cpu_ms_1seed, 2) if np.isfinite(cpu_ms_1seed) else None,
        ))
    _atomic_write_json(out_dir / "mech_2g_throughput_scaling.json", dict(schema="mech-2g-v1", profiles=prof_rows))
    print("  [2G] DONE:", prof_rows, flush=True)

    # ------------------------------------------------------------------
    # (2A) HEMIT Data Efficiency Curve: N in {16, 32, 64, 149, 298, 596}
    # ------------------------------------------------------------------
    print("[2A] Running HEMIT data-efficiency scaling curve...", flush=True)
    rng_sub = np.random.default_rng(42)
    perm_596 = rng_sub.permutation(596)
    n_grid = (16, 32, 64, 149, 298, 596)
    de_hemit = {}
    for bk in hemit_4:
        de_hemit[bk] = {}
        for n_sub in n_grid:
            sub_idx = np.sort(perm_596[:n_sub])
            bs_n = min(8, max(2, n_sub // 4))
            res_n = {}
            for arm in ("S_recon", "S_count"):
                ptr = np.empty_like(up_tr_map[bk][sub_idx])
                pev = np.empty_like(up_ev_map[bk])
                for ch, (_, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
                    m, _ = suh.fit_universal_marker(
                        arm=arm, seed=seed, common_train=c12_tr_hem[bk][ch][sub_idx],
                        donor_u8_train=torch.from_numpy(up_tr_map[bk][sub_idx, ch]),
                        target_u8_train=torch.from_numpy(h_tr["targ_nchw"][sub_idx, ch]),
                        cutoff=cutoff, device=device, epochs=8, batch_size=bs_n, lr=1.2e-3, deploy_anchor_weight=1.25,
                    )
                    ptr[:, ch], _ = suh.predict_universal_marker(m, c12_tr_hem[bk][ch][sub_idx], torch.from_numpy(up_tr_map[bk][sub_idx, ch]), device=device, batch_size=16)
                    pev[:, ch], _ = suh.predict_universal_marker(m, c12_ev_hem[bk][ch], torch.from_numpy(up_ev_map[bk][:, ch]), device=device, batch_size=16)
                    del m
                res_n[arm] = _eval_3ch_macro(ptr, h_tr["targ_nchw"][sub_idx], pev, h_ev["targ_nchw"], h_ev["clusters"], device)
            res_n["upstream"] = _eval_3ch_macro(up_tr_map[bk][sub_idx], h_tr["targ_nchw"][sub_idx], up_ev_map[bk], h_ev["targ_nchw"], h_ev["clusters"], device)
            de_hemit[bk][str(n_sub)] = res_n
            print(f"  [2A HEMIT] {bk} N={n_sub:3d}: up={res_n['upstream']['Fixed_tol_F1']:.4f} sr={res_n['S_recon']['Fixed_tol_F1']:.4f} sc={res_n['S_count']['Fixed_tol_F1']:.4f}", flush=True)
    _atomic_write_json(out_dir / "mech_2a_data_efficiency.json", dict(schema="mech-2a-v1", hemit=de_hemit, orion={}))

    # ------------------------------------------------------------------
    # (2B) HEMIT 12-Channel Coordinate Feature Ablation
    # ------------------------------------------------------------------
    print("[2B] Running HEMIT 12-channel coordinate feature ablation...", flush=True)
    feat_variants = ("target_only", "no_he", "no_struct", "no_geom", "full_12ch")
    feat_hemit = {}
    for bk in hemit_4:
        feat_hemit[bk] = {}
        for var in feat_variants:
            ptr = np.empty_like(up_tr_map[bk])
            pev = np.empty_like(up_ev_map[bk])
            for ch, (_, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
                ctr_m = _mask_common12(c12_tr_hem[bk][ch], var)
                cev_m = _mask_common12(c12_ev_hem[bk][ch], var)
                m, _ = suh.fit_universal_marker(
                    arm="S_count", seed=seed, common_train=ctr_m,
                    donor_u8_train=torch.from_numpy(up_tr_map[bk][:, ch]),
                    target_u8_train=torch.from_numpy(h_tr["targ_nchw"][:, ch]),
                    cutoff=cutoff, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
                )
                ptr[:, ch], _ = suh.predict_universal_marker(m, ctr_m, torch.from_numpy(up_tr_map[bk][:, ch]), device=device, batch_size=16)
                pev[:, ch], _ = suh.predict_universal_marker(m, cev_m, torch.from_numpy(up_ev_map[bk][:, ch]), device=device, batch_size=16)
                del m, ctr_m, cev_m
            feat_hemit[bk][var] = _eval_3ch_macro(ptr, h_tr["targ_nchw"], pev, h_ev["targ_nchw"], h_ev["clusters"], device)
            print(f"  [2B HEMIT] {bk} {var:11s}: F1={feat_hemit[bk][var]['Fixed_tol_F1']:.4f} SSIM={feat_hemit[bk][var]['SSIM']:.4f}", flush=True)
    _atomic_write_json(out_dir / "mech_2b_feature_ablation.json", dict(schema="mech-2b-v1", hemit=feat_hemit, orion={}))

    # ------------------------------------------------------------------
    # (2C) HEMIT Multi-Scale Spatial Coupling Offset Sweep
    # ------------------------------------------------------------------
    print("[2C] Running HEMIT 5-scale open-lattice spatial coupling sweep...", flush=True)
    scale_specs = (("1px", 0), ("1_4px", 4), ("1_8px", 8), ("1_16px", 16), ("1_32px", 32))
    scale_hemit = {}
    for bk in hemit_4:
        scale_hemit[bk] = {}
        for s_lbl, st2 in scale_specs:
            ptr = np.empty_like(up_tr_map[bk])
            pev = np.empty_like(up_ev_map[bk])
            dc_worst = 0.0
            for ch, (_, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
                ptr[:, ch], pev[:, ch], dcm = fit_and_predict_custom_scale(
                    step2=st2, seed=seed, c12_tr=c12_tr_hem[bk][ch],
                    d_tr_t=torch.from_numpy(up_tr_map[bk][:, ch]), y_tr_t=torch.from_numpy(h_tr["targ_nchw"][:, ch]),
                    c12_ev=c12_ev_hem[bk][ch], d_ev_t=torch.from_numpy(up_ev_map[bk][:, ch]),
                    cutoff=cutoff, device=device, epochs=8, batch_size=8, lr=1.2e-3,
                )
                dc_worst = max(dc_worst, dcm)
            m_res = _eval_3ch_macro(ptr, h_tr["targ_nchw"], pev, h_ev["targ_nchw"], h_ev["clusters"], device)
            m_res["dc_max"] = dc_worst
            scale_hemit[bk][s_lbl] = m_res
            print(f"  [2C HEMIT] {bk} {s_lbl:6s}: F1={m_res['Fixed_tol_F1']:.4f} SSIM={m_res['SSIM']:.4f} dc={dc_worst:.1e}", flush=True)
    _atomic_write_json(out_dir / "mech_2c_offset_scale_sweep.json", dict(schema="mech-2c-v1", hemit=scale_hemit, orion={}))

    # ------------------------------------------------------------------
    # (2D) HEMIT lambda_dep Fidelity-Detection Pareto Frontier
    # ------------------------------------------------------------------
    print("[2D] Running HEMIT lambda_dep Pareto sweep...", flush=True)
    lam_grid = (0.0, 0.25, 0.75, 1.25, 2.5)
    pareto_hemit = {}
    for bk in hemit_4:
        pareto_hemit[bk] = {}
        for ldep in lam_grid:
            ptr = np.empty_like(up_tr_map[bk])
            pev = np.empty_like(up_ev_map[bk])
            for ch, (_, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
                m, _ = suh.fit_universal_marker(
                    arm="S_count", seed=seed, common_train=c12_tr_hem[bk][ch],
                    donor_u8_train=torch.from_numpy(up_tr_map[bk][:, ch]),
                    target_u8_train=torch.from_numpy(h_tr["targ_nchw"][:, ch]),
                    cutoff=cutoff, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=float(ldep),
                )
                ptr[:, ch], _ = suh.predict_universal_marker(m, c12_tr_hem[bk][ch], torch.from_numpy(up_tr_map[bk][:, ch]), device=device, batch_size=16)
                pev[:, ch], _ = suh.predict_universal_marker(m, c12_ev_hem[bk][ch], torch.from_numpy(up_ev_map[bk][:, ch]), device=device, batch_size=16)
                del m
            pareto_hemit[bk][str(ldep)] = _eval_3ch_macro(ptr, h_tr["targ_nchw"], pev, h_ev["targ_nchw"], h_ev["clusters"], device)
            print(f"  [2D HEMIT] {bk} lam={ldep:4.2f}: F1={pareto_hemit[bk][str(ldep)]['Fixed_tol_F1']:.4f} SSIM={pareto_hemit[bk][str(ldep)]['SSIM']:.4f} PSNR={pareto_hemit[bk][str(ldep)]['PSNR_dB']:.2f}", flush=True)
    _atomic_write_json(out_dir / "mech_2d_lambda_pareto.json", dict(schema="mech-2d-v1", hemit=pareto_hemit, orion={}))

    # ------------------------------------------------------------------
    # (2E) 5x5 Cross-Generator Zero-Shot Transfer + Pooled Universal Head
    # (2F) H&E Stain-Color Perturbation Stress Test (reusing fitted heads!)
    # ------------------------------------------------------------------
    print("[2E & 2F] Running 5x5 cross-generator transfer, pooled universal head & stain stress test...", flush=True)
    fitted_5: dict[str, list] = {bk: [] for bk in hemit_5}
    for bk in hemit_5:
        for ch, (_, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
            m, _ = suh.fit_universal_marker(
                arm="S_count", seed=seed, common_train=c12_tr_hem[bk][ch],
                donor_u8_train=torch.from_numpy(up_tr_map[bk][:, ch]),
                target_u8_train=torch.from_numpy(h_tr["targ_nchw"][:, ch]),
                cutoff=cutoff, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
            )
            fitted_5[bk].append(m.cpu())

    # 5x5 zero-shot transfer matrix
    transfer_mat = {src: {} for src in hemit_5}
    for src in hemit_5:
        for dst in hemit_5:
            pev = np.empty_like(up_ev_map[dst])
            for ch in range(3):
                m = fitted_5[src][ch].to(device)
                pev[:, ch], _ = suh.predict_universal_marker(m, c12_ev_hem[dst][ch], torch.from_numpy(up_ev_map[dst][:, ch]), device=device, batch_size=16)
                m.cpu()
            res_td = _eval_3ch_macro(up_tr_map[dst], h_tr["targ_nchw"], pev, h_ev["targ_nchw"], h_ev["clusters"], device)
            transfer_mat[src][dst] = res_td["Fixed_tol_F1"]
    print("  [2E] 5x5 transfer diag:", [round(transfer_mat[b][b], 4) for b in hemit_5], flush=True)

    # Single pooled generator-agnostic universal 64k head trained jointly on all 5 generators (subsample 120 tiles/gen = 600 tiles total -> exact matched training budget!)
    pooled_res = {}
    pool_idx = np.sort(perm_596[:120])
    pooled_models = []
    for ch, (_, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
        c_pool = torch.cat([c12_tr_hem[bk][ch][pool_idx] for bk in hemit_5], dim=0)
        d_pool = torch.cat([torch.from_numpy(up_tr_map[bk][pool_idx, ch]) for bk in hemit_5], dim=0)
        y_pool = torch.cat([torch.from_numpy(h_tr["targ_nchw"][pool_idx, ch]) for _ in hemit_5], dim=0)
        m_pool, _ = suh.fit_universal_marker(
            arm="S_count", seed=seed, common_train=c_pool, donor_u8_train=d_pool, target_u8_train=y_pool,
            cutoff=cutoff, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
        )
        pooled_models.append(m_pool.cpu())
        del c_pool, d_pool, y_pool
    for dst in hemit_5:
        pev = np.empty_like(up_ev_map[dst])
        for ch in range(3):
            m = pooled_models[ch].to(device)
            pev[:, ch], _ = suh.predict_universal_marker(m, c12_ev_hem[dst][ch], torch.from_numpy(up_ev_map[dst][:, ch]), device=device, batch_size=16)
            m.cpu()
        pooled_res[dst] = _eval_3ch_macro(up_tr_map[dst], h_tr["targ_nchw"], pev, h_ev["targ_nchw"], h_ev["clusters"], device)
    print("  [2E] Pooled universal head F1:", {k: round(v["Fixed_tol_F1"], 4) for k, v in pooled_res.items()}, flush=True)

    # (2F) H&E Stain-Color Perturbation Stress Test at inference (no refitting!)
    stain_sigmas = (0.0, 0.08, 0.16, 0.24)
    stain_res = {bk: {} for bk in hemit_4}
    for sig in stain_sigmas:
        he_pert = _perturb_he_u8(h_ev["he_nhwc"], sig, seed=2026)
        for bk in hemit_4:
            pev = np.empty_like(up_ev_map[bk])
            for ch in range(3):
                c12_p = rhmc.build_common12_for_3ch(he_pert, up_ev_map[bk], ch, device=device)
                m = fitted_5[bk][ch].to(device)
                pev[:, ch], _ = suh.predict_universal_marker(m, c12_p, torch.from_numpy(up_ev_map[bk][:, ch]), device=device, batch_size=16)
                m.cpu()
                del c12_p
            stain_res[bk][str(sig)] = _eval_3ch_macro(up_tr_map[bk], h_tr["targ_nchw"], pev, h_ev["targ_nchw"], h_ev["clusters"], device)
        print(f"  [2F] Stain sigma={sig:.2f}:", {bk: round(stain_res[bk][str(sig)]["Fixed_tol_F1"], 4) for bk in hemit_4}, flush=True)
    _atomic_write_json(out_dir / "mech_2e_cross_generator_transfer.json", dict(schema="mech-2e-v1", backbones=list(hemit_5), transfer_matrix=transfer_mat, pooled_universal=pooled_res))
    _atomic_write_json(out_dir / "mech_2f_stain_perturbation.json", dict(schema="mech-2f-v1", sigmas=list(stain_sigmas), hemit=stain_res))

    del c12_tr_hem, c12_ev_hem, fitted_5, pooled_models, h_tr, h_ev
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # ORION-CRC Companion Mechanistic Validation (2A patient scaling + 2B/2C/2D on prognostic triad + key markers)
    # ------------------------------------------------------------------
    print("[ORION 2A-2D] Running Orion-CRC patient-efficiency, feature, scale & lambda sweeps...", flush=True)
    cv_manifest = json.loads(Path(args.orion_cv_manifest).read_text())
    with open(Path(args.orion_bank_dir) / "orion_bank_all40_2804_crop256.pkl", "rb") as fh:
        ori_b = pickle.load(fh)
    fold0 = cv_manifest["folds"][0]
    tr_pats_list = sorted(set(roms._fold_patients(fold0, "inner_train")))  # 26 patients
    ev_idx = np.nonzero(np.isin(ori_b["patients"], list(set(roms._fold_patients(fold0, "outer_test")))))[0]
    eval_pats = ori_b["patients"][ev_idx]
    ori_bks = ("miphei_convnext", "rosie_orion")
    # 6 diverse biological channels covering nuclear, epithelial, myeloid, lymphoid, vascular, stromal
    ori_ch6 = (0, 2, 3, 6, 10, 14)  # Hoechst, CD45, CD68, CD8a, CD3e, Pan-CK

    de_orion, feat_orion, scale_orion, pareto_orion = {}, {}, {}, {}
    rng_p = np.random.default_rng(123)
    pat_perm = rng_p.permutation(tr_pats_list).tolist()

    for bk in ori_bks:
        with np.load(Path(args.orion_up_dir) / f"{bk}_all40_2804.npz") as z:
            up_all = z["pred_u8"]
        c_doc = json.loads((Path(args.orion_frozen_state_dir) / bk / "fold0" / "fold0_physical_cutoffs.json").read_text())
        cutoffs_u8 = list(c_doc["cutoffs_u8"])
        tr_idx_full = np.nonzero(np.isin(ori_b["patients"], tr_pats_list))[0][::2]
        up_tr_full, up_ev = up_all[tr_idx_full], up_all[ev_idx]

        # 2A: Patient-count efficiency P in {2, 4, 8, 16, 26}
        de_orion[bk] = {}
        for p_cnt in (2, 4, 8, 16, 26):
            sub_pats = pat_perm[:p_cnt]
            sub_idx = np.nonzero(np.isin(ori_b["patients"], sub_pats))[0][::2]
            up_tr_s = up_all[sub_idx]
            res_p = {}
            for arm in ("S_recon", "S_count"):
                ptr = np.zeros_like(up_tr_s)
                pev = np.zeros_like(up_ev)
                for ch in ori_ch6:
                    c_tr = roms.build_common12_orion(ori_b["he_nhwc"][sub_idx], up_tr_s, ch, device=device)
                    c_ev = roms.build_common12_orion(ori_b["he_nhwc"][ev_idx], up_ev, ch, device=device)
                    m, _ = suh.fit_universal_marker(
                        arm=arm, seed=seed, common_train=c_tr, donor_u8_train=torch.from_numpy(up_tr_s[:, ch]),
                        target_u8_train=torch.from_numpy(ori_b["targ_nchw"][sub_idx, ch]),
                        cutoff=int(cutoffs_u8[ch]), device=device, epochs=8, batch_size=min(16, max(4, len(sub_idx) // 4)),
                        lr=1.5e-3, deploy_anchor_weight=1.25,
                    )
                    ptr[:, ch], _ = suh.predict_universal_marker(m, c_tr, torch.from_numpy(up_tr_s[:, ch]), device=device, batch_size=32)
                    pev[:, ch], _ = suh.predict_universal_marker(m, c_ev, torch.from_numpy(up_ev[:, ch]), device=device, batch_size=32)
                    del m, c_tr, c_ev
                res_p[arm] = _eval_orion_subset_macro(ptr, ori_b["targ_nchw"][sub_idx], pev, ori_b["targ_nchw"][ev_idx], eval_pats, cutoffs_u8, ori_ch6, device)
            res_p["upstream"] = _eval_orion_subset_macro(up_tr_s, ori_b["targ_nchw"][sub_idx], up_ev, ori_b["targ_nchw"][ev_idx], eval_pats, cutoffs_u8, ori_ch6, device)
            de_orion[bk][str(p_cnt)] = res_p
            print(f"  [2A ORION] {bk} P={p_cnt:2d}: up={res_p['upstream']['Fixed_tol_F1']:.4f} sc={res_p['S_count']['Fixed_tol_F1']:.4f}", flush=True)

        # Precompute 6-ch common12 on full Fold 0 train/eval
        c_tr6 = {ch: roms.build_common12_orion(ori_b["he_nhwc"][tr_idx_full], up_tr_full, ch, device=device) for ch in ori_ch6}
        c_ev6 = {ch: roms.build_common12_orion(ori_b["he_nhwc"][ev_idx], up_ev, ch, device=device) for ch in ori_ch6}

        # 2B: Feature ablation
        feat_orion[bk] = {}
        for var in feat_variants:
            ptr, pev = np.zeros_like(up_tr_full), np.zeros_like(up_ev)
            for ch in ori_ch6:
                ctr_m, cev_m = _mask_common12(c_tr6[ch], var), _mask_common12(c_ev6[ch], var)
                m, _ = suh.fit_universal_marker(
                    arm="S_count", seed=seed, common_train=ctr_m, donor_u8_train=torch.from_numpy(up_tr_full[:, ch]),
                    target_u8_train=torch.from_numpy(ori_b["targ_nchw"][tr_idx_full, ch]),
                    cutoff=int(cutoffs_u8[ch]), device=device, epochs=8, batch_size=16, lr=1.5e-3, deploy_anchor_weight=1.25,
                )
                ptr[:, ch], _ = suh.predict_universal_marker(m, ctr_m, torch.from_numpy(up_tr_full[:, ch]), device=device, batch_size=32)
                pev[:, ch], _ = suh.predict_universal_marker(m, cev_m, torch.from_numpy(up_ev[:, ch]), device=device, batch_size=32)
                del m, ctr_m, cev_m
            feat_orion[bk][var] = _eval_orion_subset_macro(ptr, ori_b["targ_nchw"][tr_idx_full], pev, ori_b["targ_nchw"][ev_idx], eval_pats, cutoffs_u8, ori_ch6, device)

        # 2C: Scale sweep
        scale_orion[bk] = {}
        for s_lbl, st2 in scale_specs:
            ptr, pev = np.zeros_like(up_tr_full), np.zeros_like(up_ev)
            for ch in ori_ch6:
                ptr[:, ch], pev[:, ch], _ = fit_and_predict_custom_scale(
                    step2=st2, seed=seed, c12_tr=c_tr6[ch],
                    d_tr_t=torch.from_numpy(up_tr_full[:, ch]), y_tr_t=torch.from_numpy(ori_b["targ_nchw"][tr_idx_full, ch]),
                    c12_ev=c_ev6[ch], d_ev_t=torch.from_numpy(up_ev[:, ch]),
                    cutoff=int(cutoffs_u8[ch]), device=device, epochs=8, batch_size=16, lr=1.5e-3,
                )
            scale_orion[bk][s_lbl] = _eval_orion_subset_macro(ptr, ori_b["targ_nchw"][tr_idx_full], pev, ori_b["targ_nchw"][ev_idx], eval_pats, cutoffs_u8, ori_ch6, device)

        # 2D: lambda_dep sweep
        pareto_orion[bk] = {}
        for ldep in lam_grid:
            ptr, pev = np.zeros_like(up_tr_full), np.zeros_like(up_ev)
            for ch in ori_ch6:
                m, _ = suh.fit_universal_marker(
                    arm="S_count", seed=seed, common_train=c_tr6[ch], donor_u8_train=torch.from_numpy(up_tr_full[:, ch]),
                    target_u8_train=torch.from_numpy(ori_b["targ_nchw"][tr_idx_full, ch]),
                    cutoff=int(cutoffs_u8[ch]), device=device, epochs=8, batch_size=16, lr=1.5e-3, deploy_anchor_weight=float(ldep),
                )
                ptr[:, ch], _ = suh.predict_universal_marker(m, c_tr6[ch], torch.from_numpy(up_tr_full[:, ch]), device=device, batch_size=32)
                pev[:, ch], _ = suh.predict_universal_marker(m, c_ev6[ch], torch.from_numpy(up_ev[:, ch]), device=device, batch_size=32)
                del m
            pareto_orion[bk][str(ldep)] = _eval_orion_subset_macro(ptr, ori_b["targ_nchw"][tr_idx_full], pev, ori_b["targ_nchw"][ev_idx], eval_pats, cutoffs_u8, ori_ch6, device)
        del c_tr6, c_ev6
        gc.collect()
        torch.cuda.empty_cache()

    _atomic_write_json(out_dir / "mech_2a_data_efficiency.json", dict(schema="mech-2a-v1", hemit=de_hemit, orion=de_orion))
    _atomic_write_json(out_dir / "mech_2b_feature_ablation.json", dict(schema="mech-2b-v1", hemit=feat_hemit, orion=feat_orion))
    _atomic_write_json(out_dir / "mech_2c_offset_scale_sweep.json", dict(schema="mech-2c-v1", hemit=scale_hemit, orion=scale_orion))
    _atomic_write_json(out_dir / "mech_2d_lambda_pareto.json", dict(schema="mech-2d-v1", hemit=pareto_hemit, orion=pareto_orion))
    _atomic_write_json(out_dir / "mech_2e_cross_generator_transfer.json", dict(schema="mech-2e-v1", backbones=list(hemit_5), transfer_matrix=transfer_mat, pooled_universal=pooled_res))
    _atomic_write_json(out_dir / "mech_2f_stain_perturbation.json", dict(schema="mech-2f-v1", sigmas=list(stain_sigmas), hemit=stain_res))
    print(f"MECH_SUITE_COMPLETE in {time.monotonic() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
