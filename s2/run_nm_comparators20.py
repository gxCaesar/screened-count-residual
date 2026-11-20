#!/usr/bin/env python3
"""EXP-3: Modern Objective & Intensity-Calibration Comparators across all 20 frozen generators.

Evaluates 4 additional rigorous baselines strictly on training splits and scored on locked test splits:
  1. ``QuantileMatch``: exact 256-bin training-split CDF / histogram quantile matching LUT.
  2. ``AffineGamma``  : training-tuned monotonic power-law ``clip(a * (x/255)^g + d, 0, 1)``
                        maximizing training deployable tolerant F1 (with identity fallback).
  3. ``S_dice``       : identical 64,161-param screened Poisson head trained with standard
                        Soft-Dice loss at physical cutoff ``tau_c`` + L_recon + matched 2-ep selector.
  4. ``S_bce``        : identical 64,161-param screened Poisson head trained with standard
                        class-balanced Focal-BCE loss at ``tau_c`` + L_recon + matched 2-ep selector.

Never touches or modifies any immutable S6 artifact.
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
from . import query_relation_count_objective as qobj
from . import query_relation_reconstruction as core
from . import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: Mapping) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Monotonic Intensity Calibration Baselines (QuantileMatch & AffineGamma)
# ---------------------------------------------------------------------------

def apply_quantile_match_u8(
    train_up_u8: np.ndarray,
    train_targ_u8: np.ndarray,
    eval_up_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Exact 256-bin empirical CDF quantile matching fitted on training split."""
    src_hist = np.bincount(train_up_u8.ravel(), minlength=256).astype(np.float64)
    ref_hist = np.bincount(train_targ_u8.ravel(), minlength=256).astype(np.float64)
    src_cdf = np.cumsum(src_hist) / max(1.0, float(src_hist.sum()))
    ref_cdf = np.cumsum(ref_hist) / max(1.0, float(ref_hist.sum()))
    # Map each source bin i to smallest ref bin j with ref_cdf[j] >= src_cdf[i]
    lut = np.searchsorted(ref_cdf, src_cdf, side="left")
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[train_up_u8], lut[eval_up_u8], dict(lut_min=int(lut[0]), lut_max=int(lut[-1]))


def tune_affine_gamma_u8(
    train_up_u8: np.ndarray,
    train_targ_u8: np.ndarray,
    eval_up_u8: np.ndarray,
    cutoff: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Grid-search monotonic clip(a * (x/255)^gamma + d, 0, 1) on training Fixed_tol_F1."""
    def _eval_f1(pred_tr: np.ndarray) -> float:
        res = jce.batch_curves_and_quality_torch(
            pred_tr, train_targ_u8, cutoff, device=device, compute_quality=False
        )
        ep = jce.endpoints_from_pooled_curves(
            res["K"].sum(axis=0),
            res["U"].sum(axis=0),
            res["V"].sum(axis=0),
            res["T"].sum(axis=0),
            int(res["G"].sum()),
            cutoff,
        )
        return float(ep["Fixed_tol_F1"])

    base_f1 = _eval_f1(train_up_u8)
    best = (1.0, 1.0, 0.0, base_f1)
    x_bins = np.arange(256, dtype=np.float64) / 255.0

    for gamma in (0.5, 0.75, 1.0, 1.35):
        xg = np.power(x_bins, gamma)
        for gain in (0.85, 1.0, 1.25, 1.6, 2.2):
            for shift in (-0.05, 0.0, 0.05, 0.12, 0.22):
                lut = np.round(np.clip(gain * xg + shift, 0.0, 1.0) * 255.0).astype(np.uint8)
                f1 = _eval_f1(lut[train_up_u8])
                if f1 > best[3] + 1e-6:
                    best = (gain, gamma, shift, f1)

    gain, gamma, shift, best_f1 = best
    lut = np.round(np.clip(gain * np.power(x_bins, gamma) + shift, 0.0, 1.0) * 255.0).astype(np.uint8)
    meta = dict(gain=gain, gamma=gamma, shift=shift, train_f1=best_f1, train_base_f1=base_f1)
    return lut[train_up_u8], lut[eval_up_u8], meta


# ---------------------------------------------------------------------------
# Differentiable Alternative Loss Comparators on Identical 64k Screened Head
# ---------------------------------------------------------------------------

def fit_alt_loss_marker(
    *,
    mode: str,
    seed: int,
    common_train: Tensor,
    donor_u8_train: Tensor,
    target_u8_train: Tensor,
    cutoff: int,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 1.2e-3,
    weight_decay: float = 1e-4,
    aux_weight: float = 0.25,
) -> tuple[core.RelationModel, dict]:
    """Train identical 64k screened Poisson head under ``S_dice`` or ``S_bce`` + L_recon + 2-ep selector."""
    assert mode in ("S_dice", "S_bce")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = suh.build_coordinate_screened_model(seed, device)
    n = int(common_train.shape[0])
    gt_pos_full = (target_u8_train > int(cutoff))
    pos_rate = float(gt_pos_full.to(torch.float32).mean().item())
    pos_weight = float(np.clip((1.0 - pos_rate) / max(pos_rate, 1e-4), 1.0, 25.0))

    # Compute exact dataset reconstruction normalization scales matching S_count
    scale_sums = {"mse": 0.0, "ssim": 0.0, "edge1": 0.0, "edge16": 0.0}
    with torch.no_grad():
        for s0 in range(0, n, 32):
            e0 = min(n, s0 + 32)
            db0 = donor_u8_train[s0:e0].to(device, dtype=torch.float32).unsqueeze(1) / 255.0
            yb0 = target_u8_train[s0:e0].to(device, dtype=torch.float32).unsqueeze(1) / 255.0
            sc0 = suh._fast_loss_scales_f32(db0, yb0)
            for k in scale_sums:
                scale_sums[k] += float(sc0[k].sum().item())
    scales = {k: max(v / float(n), core.SCALE_FLOOR) for k, v in scale_sums.items()}

    # Precompute 3-scale temperature weights identical to S_count soft-step
    temps = (1.2, 5.0, 20.0)
    tw = (0.45, 0.35, 0.20)
    cut_shift = float(cutoff) + 0.5

    def _soft_prob(a_255: Tensor) -> Tensor:
        p = torch.zeros_like(a_255)
        for tj, wj in zip(temps, tw):
            lo = torch.sigmoid(torch.tensor((0.0 - cut_shift) / tj, device=a_255.device, dtype=a_255.dtype))
            hi = torch.sigmoid(torch.tensor((255.0 - cut_shift) / tj, device=a_255.device, dtype=a_255.dtype))
            raw = torch.sigmoid((a_255 - cut_shift) / tj)
            p = p + wj * ((raw - lo) / (hi - lo).clamp_min(1e-6)).clamp(1e-6, 1.0 - 1e-6)
        return p.clamp(1e-6, 1.0 - 1e-6)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup = max(1, int(round(0.10 * total_steps)))

    def _lr_scale(step_idx: int) -> float:
        if step_idx < warmup:
            return float(step_idx + 1) / float(warmup)
        prog = float(step_idx - warmup) / float(max(1, total_steps - warmup))
        return 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, max(0.0, prog)))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_scale)
    rng = np.random.default_rng(seed + 1337)

    best_score = -1.0
    best_ep = epochs
    best_sd = None

    def _train_composite_score() -> float:
        p_u8, _ = suh.predict_universal_marker(model, common_train, donor_u8_train, device=device, batch_size=max(16, batch_size))
        res = jce.batch_curves_and_quality_torch(p_u8, target_u8_train.numpy(), cutoff, device=device, compute_quality=False)
        ep = jce.endpoints_from_pooled_curves(
            res["K"].sum(axis=0), res["U"].sum(axis=0), res["V"].sum(axis=0), res["T"].sum(axis=0), int(res["G"].sum()), cutoff
        )
        return 0.65 * float(ep["Fixed_tol_F1"]) + 0.35 * float(ep["ODS_F"])

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s : s + batch_size]
            c_b = common_train[idx].to(device, dtype=torch.float32)
            d_b = donor_u8_train[idx].to(device, dtype=torch.float32).unsqueeze(1) / 255.0
            y_b = target_u8_train[idx].to(device, dtype=torch.float32).unsqueeze(1) / 255.0
            g_b = (target_u8_train[idx].to(device) > int(cutoff)).to(torch.float32).unsqueeze(1)

            # Dihedral D4 augmentation matching S_count
            k_rot = int(rng.integers(0, 4))
            do_flip = bool(rng.integers(0, 2))
            if k_rot:
                c_b = torch.rot90(c_b, k_rot, dims=(-2, -1))
                d_b = torch.rot90(d_b, k_rot, dims=(-2, -1))
                y_b = torch.rot90(y_b, k_rot, dims=(-2, -1))
                g_b = torch.rot90(g_b, k_rot, dims=(-2, -1))
            if do_flip:
                c_b = torch.flip(c_b, dims=(-1,))
                d_b = torch.flip(d_b, dims=(-1,))
                y_b = torch.flip(y_b, dims=(-1,))
                g_b = torch.flip(g_b, dims=(-1,))

            resid, _ = suh.fast_sc_forward(model, c_b, inference=False)
            z_b = d_b + resid
            z_clip = z_b.clamp(0.0, 1.0)
            a_255 = z_clip * 255.0

            # Exact normalized reconstruction anchor identical to S_count
            recon_loss = suh.preclip_reconstruction_loss(z_b, y_b, d_b, scales).mean()

            prob = _soft_prob(a_255)
            if mode == "S_dice":
                inter = (prob * g_b).sum()
                denom = prob.sum() + g_b.sum() + 1.0
                aux_loss = 1.0 - (2.0 * inter + 1.0) / denom
                lam_eff = 2.5
            else:
                # Class-balanced Focal-BCE (gamma = 2.0)
                pt = torch.where(g_b > 0.5, prob, 1.0 - prob)
                alpha_w = torch.where(g_b > 0.5, torch.full_like(prob, pos_weight), torch.ones_like(prob))
                bce = -(g_b * torch.log(prob) + (1.0 - g_b) * torch.log(1.0 - prob))
                focal = alpha_w * ((1.0 - pt) ** 2.0) * bce
                aux_loss = focal.mean()
                lam_eff = 2.5

            loss = recon_loss + lam_eff * aux_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()

        if ep % 2 == 0 or ep == epochs:
            sc = _train_composite_score()
            if sc > best_score:
                best_score = sc
                best_ep = ep
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_sd is not None:
        model.load_state_dict(best_sd)
    return model, dict(mode=mode, selected_train_epoch=best_ep, selected_train_composite=best_score)


# ---------------------------------------------------------------------------
# Cohort Runners
# ---------------------------------------------------------------------------

COMP_ARMS = ("upstream", "QuantileMatch", "AffineGamma", "S_dice", "S_bce")


def run_hemit(args: argparse.Namespace) -> None:
    from . import run_hemit_multibackbone_cell as rhmc

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(args.bank_cache_dir).resolve() / "hemit_bank_train_596.pkl", "rb") as fh:
        tr_b = pickle.load(fh)
    with open(Path(args.bank_cache_dir).resolve() / "hemit_bank_test_945.pkl", "rb") as fh:
        ev_b = pickle.load(fh)

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    for b_key in [b.strip() for b in args.backbones.split(",") if b.strip()]:
        out_json = out_dir / f"comp20_hemit_{b_key}.json"
        if out_json.exists() and not args.overwrite:
            print(json.dumps(dict(event="skip", cohort="hemit", backbone=b_key)), flush=True)
            continue
        t0 = time.monotonic()
        with np.load(Path(args.upstream_cache_dir).resolve() / f"{b_key}_train_596.npz") as z:
            up_tr = z["pred_u8"]
        with np.load(Path(args.upstream_cache_dir).resolve() / f"{b_key}_test_945.npz") as z:
            up_ev = z["pred_u8"]

        arm_tr = {a: (up_tr if a == "upstream" else np.empty_like(up_tr)) for a in COMP_ARMS}
        arm_ev = {a: (up_ev if a == "upstream" else np.empty_like(up_ev)) for a in COMP_ARMS}
        meta_all = {}

        for ch, (m_name, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
            q_tr, q_ev, q_m = apply_quantile_match_u8(up_tr[:, ch], tr_b["targ_nchw"][:, ch], up_ev[:, ch])
            a_tr, a_ev, a_m = tune_affine_gamma_u8(up_tr[:, ch], tr_b["targ_nchw"][:, ch], up_ev[:, ch], cutoff, device)
            arm_tr["QuantileMatch"][:, ch], arm_ev["QuantileMatch"][:, ch] = q_tr, q_ev
            arm_tr["AffineGamma"][:, ch], arm_ev["AffineGamma"][:, ch] = a_tr, a_ev
            meta_all[m_name] = dict(QuantileMatch=q_m, AffineGamma=a_m)

            c12_tr = rhmc.build_common12_for_3ch(tr_b["he_nhwc"], up_tr, ch, device=device)
            d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr[:, ch]), torch.from_numpy(tr_b["targ_nchw"][:, ch]), torch.from_numpy(up_ev[:, ch])
            fitted = {}
            for mode in ("S_dice", "S_bce"):
                tr_l = []
                for sd in seeds:
                    m, fm = fit_alt_loss_marker(
                        mode=mode, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                        target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    )
                    meta_all[f"{m_name}:{mode}:{sd}"] = fm
                    pt, _ = suh.predict_universal_marker(m, c12_tr, d_tr_t, device=device, batch_size=16)
                    tr_l.append(pt.astype(np.float32))
                    fitted[(mode, sd)] = m.cpu()
                arm_tr[mode][:, ch] = np.round(np.mean(tr_l, axis=0)).astype(np.uint8)
            del c12_tr
            gc.collect()
            c12_ev = rhmc.build_common12_for_3ch(ev_b["he_nhwc"], up_ev, ch, device=device)
            for mode in ("S_dice", "S_bce"):
                ev_l = []
                for sd in seeds:
                    m = fitted[(mode, sd)].to(device)
                    pe, _ = suh.predict_universal_marker(m, c12_ev, d_ev_t, device=device, batch_size=16)
                    ev_l.append(pe.astype(np.float32))
                    del m
                arm_ev[mode][:, ch] = np.round(np.mean(ev_l, axis=0)).astype(np.uint8)
            del c12_ev, fitted
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        arms_eval = {}
        for a in COMP_ARMS:
            per_m = {}
            for ch, (m_name, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
                ch_r = jce.evaluate_channel_predictions(
                    train_pred_u8=arm_tr[a][:, ch], train_targ_u8=tr_b["targ_nchw"][:, ch],
                    eval_pred_u8=arm_ev[a][:, ch], eval_targ_u8=ev_b["targ_nchw"][:, ch],
                    eval_clusters=ev_b["clusters"], fixed_cutoff=cutoff, device=device,
                )
                per_m[m_name] = ch_r["pooled"]
            macro = {mk: float(np.mean([per_m[m][mk] for m in rhmc.HEMIT_MARKERS])) for mk in ("ODS_F", "AP", "Fixed_tol_F1", "Cal_F1", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed")}
            tr_c = rhmc.extract_pred_cell_means_for_records(arm_tr[a], tr_b["nuclei_records"])
            ev_c = rhmc.extract_pred_cell_means_for_records(arm_ev[a], ev_b["nuclei_records"])
            cell_res = rhmc.evaluate_hemit_arm_cells_and_infiltration(tr_c, ev_c, ev_b["nuclei_records"], arm_ev[a])
            arms_eval[a] = dict(macro_pixel=macro, per_marker_pixel=per_m, single_cell=cell_res)

        rec = dict(schema="nm-comp20-v1", cohort="HEMIT", backbone=b_key, wall_seconds=time.monotonic() - t0, arms=arms_eval, meta=meta_all)
        _atomic_write_json(out_json, rec)
        print(json.dumps(dict(event="done", cohort="hemit", backbone=b_key, **{a: round(arms_eval[a]["macro_pixel"]["Fixed_tol_F1"], 4) for a in COMP_ARMS}, sec=round(rec["wall_seconds"], 1))), flush=True)


def run_orion(args: argparse.Namespace) -> None:
    from . import run_orion_multibackbone_survival as roms

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_manifest = json.loads(Path(args.cv_manifest).read_text())
    with open(Path(args.bank_cache_dir).resolve() / "orion_bank_all40_2804_crop256.pkl", "rb") as fh:
        bank = pickle.load(fh)
    fold0 = cv_manifest["folds"][0]
    tr_idx = np.nonzero(np.isin(bank["patients"], list(set(roms._fold_patients(fold0, "inner_train")))))[0][::2]
    ev_idx = np.nonzero(np.isin(bank["patients"], list(set(roms._fold_patients(fold0, "outer_test")))))[0]
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    for b_key in [b.strip() for b in args.backbones.split(",") if b.strip()]:
        out_json = out_dir / f"comp20_orion_{b_key}.json"
        if out_json.exists() and not args.overwrite:
            print(json.dumps(dict(event="skip", cohort="orion", backbone=b_key)), flush=True)
            continue
        t0 = time.monotonic()
        with np.load(Path(args.upstream_cache_dir).resolve() / f"{b_key}_all40_2804.npz") as z:
            up_all = z["pred_u8"]
        c_doc = json.loads((Path(args.frozen_state_dir).resolve() / b_key / "fold0" / "fold0_physical_cutoffs.json").read_text())
        cutoffs_u8, tau_phys_float = list(c_doc["cutoffs_u8"]), dict(c_doc["tau_phys_float"])
        up_tr, up_ev = up_all[tr_idx], up_all[ev_idx]
        arm_tr = {a: (up_tr if a == "upstream" else np.empty_like(up_tr)) for a in COMP_ARMS}
        arm_ev = {a: (up_ev if a == "upstream" else np.empty_like(up_ev)) for a in COMP_ARMS}
        meta_all = {}

        for ch16, m_name in enumerate(roms.ORION_MARKERS):
            cutoff = int(cutoffs_u8[ch16])
            q_tr, q_ev, q_m = apply_quantile_match_u8(up_tr[:, ch16], bank["targ_nchw"][tr_idx, ch16], up_ev[:, ch16])
            a_tr, a_ev, a_m = tune_affine_gamma_u8(up_tr[:, ch16], bank["targ_nchw"][tr_idx, ch16], up_ev[:, ch16], cutoff, device)
            arm_tr["QuantileMatch"][:, ch16], arm_ev["QuantileMatch"][:, ch16] = q_tr, q_ev
            arm_tr["AffineGamma"][:, ch16], arm_ev["AffineGamma"][:, ch16] = a_tr, a_ev
            meta_all[m_name] = dict(QuantileMatch=q_m, AffineGamma=a_m)

            c12_tr = roms.build_common12_orion(bank["he_nhwc"][tr_idx], up_tr, ch16, device=device)
            d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr[:, ch16]), torch.from_numpy(bank["targ_nchw"][tr_idx, ch16]), torch.from_numpy(up_ev[:, ch16])
            fitted = {}
            for mode in ("S_dice", "S_bce"):
                tr_l = []
                for sd in seeds:
                    m, fm = fit_alt_loss_marker(
                        mode=mode, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                        target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    )
                    meta_all[f"{m_name}:{mode}:{sd}"] = fm
                    pt, _ = suh.predict_universal_marker(m, c12_tr, d_tr_t, device=device, batch_size=32)
                    tr_l.append(pt.astype(np.float32))
                    fitted[(mode, sd)] = m.cpu()
                arm_tr[mode][:, ch16] = np.round(np.mean(tr_l, axis=0)).astype(np.uint8)
            del c12_tr
            gc.collect()
            c12_ev = roms.build_common12_orion(bank["he_nhwc"][ev_idx], up_ev, ch16, device=device)
            for mode in ("S_dice", "S_bce"):
                ev_l = []
                for sd in seeds:
                    m = fitted[(mode, sd)].to(device)
                    pe, _ = suh.predict_universal_marker(m, c12_ev, d_ev_t, device=device, batch_size=32)
                    ev_l.append(pe.astype(np.float32))
                    del m
                arm_ev[mode][:, ch16] = np.round(np.mean(ev_l, axis=0)).astype(np.uint8)
            del c12_ev, fitted
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        arms_eval = {}
        eval_pats = bank["patients"][ev_idx]
        for a in COMP_ARMS:
            per_m = {}
            for ch16, m_name in enumerate(roms.ORION_MARKERS):
                ch_r = jce.evaluate_channel_predictions(
                    train_pred_u8=arm_tr[a][:, ch16], train_targ_u8=bank["targ_nchw"][tr_idx, ch16],
                    eval_pred_u8=arm_ev[a][:, ch16], eval_targ_u8=bank["targ_nchw"][ev_idx, ch16],
                    eval_clusters=eval_pats, fixed_cutoff=int(cutoffs_u8[ch16]), device=device,
                )
                per_m[m_name] = ch_r["pooled"]
            macro = {mk: float(np.mean([per_m[m][mk] for m in roms.ORION_MARKERS])) for mk in ("ODS_F", "AP", "Fixed_tol_F1", "Cal_F1", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed")}
            tr_cells = roms.gather_subset_cells(arm_tr[a], tr_idx, bank["nuclei_records"])
            ev_cells = roms.gather_subset_cells(arm_ev[a], ev_idx, bank["nuclei_records"])
            dep_f1s = []
            for ch16 in range(1, 16):
                m_nm = roms.ORION_MARKERS[ch16]
                rc = roms.ccs.evaluate_single_cell_marker(
                    train_gt_scores=tr_cells["gt_means16"][:, ch16], train_pred_scores=tr_cells["pred_means"][:, ch16],
                    train_labels=tr_cells["labels15"][:, ch16 - 1], eval_pred_scores=ev_cells["pred_means"][:, ch16],
                    eval_labels=ev_cells["labels15"][:, ch16 - 1], eval_tile_ids=ev_cells["tile_ids"],
                    physical_cutoff=float(tau_phys_float[m_nm]),
                )
                dep_f1s.append(float(rc["deployable_cell_f1"]))
            arms_eval[a] = dict(macro_pixel_16ch=macro, per_marker_pixel=per_m, single_cell_15class=dict(macro_deployable_cell_f1=float(np.mean(dep_f1s))))

        rec = dict(schema="nm-comp20-v1", cohort="Orion-CRC", backbone=b_key, wall_seconds=time.monotonic() - t0, arms=arms_eval, meta=meta_all)
        _atomic_write_json(out_json, rec)
        print(json.dumps(dict(event="done", cohort="orion", backbone=b_key, **{a: round(arms_eval[a]["macro_pixel_16ch"]["Fixed_tol_F1"], 4) for a in COMP_ARMS}, sec=round(rec["wall_seconds"], 1))), flush=True)


def run_her2(args: argparse.Namespace) -> None:
    from . import run_her2match_dab_campaign as rhd

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_doc = json.loads(Path(args.calibration_json).read_text())
    cutoff_u8 = int(round(float(cal_doc["threshold"]) / float(cal_doc["histogram"]["range"][1]) * 255.0))
    with open(Path(args.bank_cache_dir).resolve() / "her2_bank_train_480_512.pkl", "rb") as fh:
        tr_b = pickle.load(fh)
    with open(Path(args.bank_cache_dir).resolve() / "her2_bank_test_500_512.pkl", "rb") as fh:
        ev_b = pickle.load(fh)
    hopt_path = Path(args.hoptimus0_ckpt).resolve() if args.hoptimus0_ckpt else None
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    for b_key in [b.strip() for b in args.backbones.split(",") if b.strip()]:
        out_json = out_dir / f"comp20_her2_{b_key}.json"
        if out_json.exists() and not args.overwrite:
            print(json.dumps(dict(event="skip", cohort="her2", backbone=b_key)), flush=True)
            continue
        t0 = time.monotonic()
        _, up_tr, up_ev = rhd.train_or_load_her2_backbone(
            b_key, tr_b, ev_b, cutoff_u8, Path(args.frozen_state_dir).resolve(), device,
            hoptimus0_ckpt=hopt_path, upstream_epochs=8, allow_train=False,
        )
        arm_tr = {"upstream": up_tr}
        arm_ev = {"upstream": up_ev}
        meta_all = {}
        arm_tr["QuantileMatch"], arm_ev["QuantileMatch"], meta_all["QuantileMatch"] = apply_quantile_match_u8(up_tr, tr_b["dab_u8"], up_ev)
        arm_tr["AffineGamma"], arm_ev["AffineGamma"], meta_all["AffineGamma"] = tune_affine_gamma_u8(up_tr, tr_b["dab_u8"], up_ev, cutoff_u8, device)

        c12_tr = rhd.build_common12_her2(tr_b["he_nhwc"], up_tr, tr_b["hem_u8"], device=device)
        d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr), torch.from_numpy(tr_b["dab_u8"]), torch.from_numpy(up_ev)
        fitted = {}
        for mode in ("S_dice", "S_bce"):
            tr_l = []
            for sd in seeds:
                m, fm = fit_alt_loss_marker(
                    mode=mode, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                    target_u8_train=y_tr_t, cutoff=cutoff_u8, device=device,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                )
                meta_all[f"{mode}:{sd}"] = fm
                pt, _ = suh.predict_universal_marker(m, c12_tr, d_tr_t, device=device, batch_size=16)
                tr_l.append(pt.astype(np.float32))
                fitted[(mode, sd)] = m.cpu()
            arm_tr[mode] = np.round(np.mean(tr_l, axis=0)).astype(np.uint8)
        del c12_tr
        gc.collect()
        c12_ev = rhd.build_common12_her2(ev_b["he_nhwc"], up_ev, ev_b["hem_u8"], device=device)
        for mode in ("S_dice", "S_bce"):
            ev_l = []
            for sd in seeds:
                m = fitted[(mode, sd)].to(device)
                pe, _ = suh.predict_universal_marker(m, c12_ev, d_ev_t, device=device, batch_size=16)
                ev_l.append(pe.astype(np.float32))
                del m
            arm_ev[mode] = np.round(np.mean(ev_l, axis=0)).astype(np.uint8)
        del c12_ev, fitted
        gc.collect()

        arms_eval = {}
        wsis = np.asarray(ev_b["wsis"])
        for a in COMP_ARMS:
            ch_r = jce.evaluate_channel_predictions(
                train_pred_u8=arm_tr[a], train_targ_u8=tr_b["dab_u8"],
                eval_pred_u8=arm_ev[a], eval_targ_u8=ev_b["dab_u8"],
                eval_clusters=wsis, fixed_cutoff=cutoff_u8, device=device,
            )
            burden = rhd.evaluate_dab_burden_concordance(arm_ev[a], ev_b["dab_u8"], wsis, cutoff_u8)
            arms_eval[a] = dict(dab_pixel=ch_r["pooled"], dab_burden_concordance=burden)

        rec = dict(schema="nm-comp20-v1", cohort="HER2match", backbone=b_key, wall_seconds=time.monotonic() - t0, arms=arms_eval, meta=meta_all)
        _atomic_write_json(out_json, rec)
        print(json.dumps(dict(event="done", cohort="her2", backbone=b_key, **{a: round(arms_eval[a]["dab_pixel"]["Fixed_tol_F1"], 4) for a in COMP_ARMS}, sec=round(rec["wall_seconds"], 1))), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=("hemit", "orion", "her2"), required=True)
    ap.add_argument("--backbones", required=True)
    ap.add_argument("--bank-cache-dir", required=True)
    ap.add_argument("--upstream-cache-dir", default="")
    ap.add_argument("--frozen-state-dir", default="")
    ap.add_argument("--cv-manifest", default="")
    ap.add_argument("--calibration-json", default="")
    ap.add_argument("--hoptimus0-ckpt", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="202610270,202610271")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.cohort == "hemit":
        run_hemit(args)
    elif args.cohort == "orion":
        run_orion(args)
    else:
        run_her2(args)


if __name__ == "__main__":
    main()
