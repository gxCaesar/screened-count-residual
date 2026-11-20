#!/usr/bin/env python3
"""PUSH-3: 100% Strictly Unseen-Patient 5-Fold End-to-End Orion-CRC (N = 40 Patients, Killing B1).

Eliminates upstream pretraining slide overlap completely:
  - For each outer fold f in {0..4}, trains 3 diverse 16-channel virtual-staining generators
    FROM SCRATCH strictly on the 32 non-test patients (26 inner_train + 6 inner_val;
    asserting ZERO overlap with the 8 outer_test patients).
  - Architectures:
      1. unseen_unet16         : 7.77M Double-Convolution U-Net (MSE + 0.5 L1)
      2. unseen_resnet34_unet16: 24.4M ResNet-34 Encoder U-Net (MSE + 0.5 L1)
      3. unseen_pix2pix16      : U-Net Generator + 70x70 PatchGAN cGAN (L1 + adversarial BCE)
  - Freezes each fold's unseen-patient generator, calibrates physical Otsu cutoffs strictly
    on the 26 inner_train patients, and evaluates 5 post-hoc arms strictly on the 8 unseen
    outer_test patients:
      upstream | S_recon (2-seed) | QuantileMatch (256-bin LUT) | E_count (2-seed unscreened) | S_count (2-seed screened)
  - Measures 16-channel pixel F1/SSIM/PSNR/ODS-F/exact-r0-F1, 15-class single-cell nuclear F1,
    15x15 marker co-occurrence MAE, 25-um Pan-CK->CD8a infiltration r, per-patient macro F1,
    AND fold-isolated ImmunoScore compartment densities for 40-patient strictly unseen PFS survival!
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from s2 import clinical_cell_survival as ccs
from s2 import journal_cohort_eval as jce
from s2 import query_relation_reconstruction as core
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


class DoubleConv16(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet16(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 16, base: int = 32):
        super().__init__()
        self.e1 = DoubleConv16(in_ch, base)
        self.e2 = DoubleConv16(base, base * 2)
        self.e3 = DoubleConv16(base * 2, base * 4)
        self.e4 = DoubleConv16(base * 4, base * 8)
        self.bot = DoubleConv16(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = DoubleConv16(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = DoubleConv16(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = DoubleConv16(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = DoubleConv16(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.e1(x)
        x2 = self.e2(F.max_pool2d(x1, 2))
        x3 = self.e3(F.max_pool2d(x2, 2))
        x4 = self.e4(F.max_pool2d(x3, 2))
        b = self.bot(F.max_pool2d(x4, 2))
        y4 = self.d4(torch.cat([self.u4(b), x4], dim=1))
        y3 = self.d3(torch.cat([self.u3(y4), x3], dim=1))
        y2 = self.d2(torch.cat([self.u2(y3), x2], dim=1))
        y1 = self.d1(torch.cat([self.u1(y2), x1], dim=1))
        return torch.sigmoid(self.out(y1))


class PatchDiscriminator16(nn.Module):
    def __init__(self, in_ch: int = 19, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_upstream_16ch(arch: str) -> nn.Module:
    if arch == "unseen_unet16":
        return UNet16(3, 16, base=32)
    if arch == "unseen_resnet34_unet16":
        return rp1.ResNet34UNetVirtualStainer(3, 16)
    if arch == "unseen_pix2pix16":
        return UNet16(3, 16, base=32)
    raise ValueError(f"Unknown arch: {arch}")


@torch.no_grad()
def predict_16ch_u8(net: nn.Module, he_nhwc: np.ndarray, device: torch.device, batch_size: int = 32) -> np.ndarray:
    net.eval()
    n, h, w, _ = he_nhwc.shape
    out = np.empty((n, 16, h, w), dtype=np.uint8)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        xb = torch.from_numpy(he_nhwc[s:e]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
        pr = net(xb).clamp(0.0, 1.0)
        out[s:e] = torch.round(pr * 255.0).to(torch.uint8).cpu().numpy()
    return out


def train_fold_isolated_generator(
    arch: str,
    fold_idx: int,
    he_tr: np.ndarray,
    targ_tr: np.ndarray,
    he_val: np.ndarray,
    targ_val: np.ndarray,
    device: torch.device,
    epochs: int = 22,
    batch_size: int = 16,
) -> tuple[nn.Module, dict]:
    """Train 16-channel generator FROM SCRATCH strictly on 26 inner_train + 6 inner_val patients."""
    t0 = time.monotonic()
    seed = 20261120 + fold_idx * 101 + (0 if arch == "unseen_unet16" else (17 if arch == "unseen_resnet34_unet16" else 37))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    gen = build_upstream_16ch(arch).to(device)
    n_params = int(sum(p.numel() for p in gen.parameters()))
    is_gan = arch == "unseen_pix2pix16"
    opt_g = torch.optim.AdamW(gen.parameters(), lr=1e-3 if not is_gan else 6e-4, weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=5e-5)

    disc = PatchDiscriminator16(19, 64).to(device) if is_gan else None
    opt_d = torch.optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.5, 0.999)) if is_gan else None

    n_tr = len(he_tr)
    best_val_l1 = 1e9
    best_sd = None
    best_ep = 0

    for ep in range(epochs):
        gen.train()
        if disc is not None:
            disc.train()
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, batch_size):
            idx = order[s : s + batch_size]
            xb = torch.from_numpy(he_tr[idx]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            yb = torch.from_numpy(targ_tr[idx]).to(device, dtype=torch.float32) / 255.0
            k_rot = int(rng.integers(0, 4))
            if k_rot:
                xb = torch.rot90(xb, k_rot, dims=(-2, -1))
                yb = torch.rot90(yb, k_rot, dims=(-2, -1))
            if int(rng.integers(0, 2)):
                xb = torch.flip(xb, dims=(-1,))
                yb = torch.flip(yb, dims=(-1,))

            fake = gen(xb)
            if is_gan and disc is not None and opt_d is not None:
                # Discriminator step
                opt_d.zero_grad(set_to_none=True)
                d_real = disc(torch.cat([xb, yb], dim=1))
                d_fake = disc(torch.cat([xb, fake.detach()], dim=1))
                loss_d = 0.5 * (
                    F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                    + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                )
                loss_d.backward()
                opt_d.step()

                # Generator step
                opt_g.zero_grad(set_to_none=True)
                g_adv = F.binary_cross_entropy_with_logits(
                    disc(torch.cat([xb, fake], dim=1)), torch.ones_like(d_real)
                )
                loss_g = F.l1_loss(fake, yb) + 0.5 * F.mse_loss(fake, yb) + 0.015 * g_adv
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
                opt_g.step()
            else:
                opt_g.zero_grad(set_to_none=True)
                loss_g = F.mse_loss(fake, yb) + 0.5 * F.l1_loss(fake, yb)
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
                opt_g.step()

        sched_g.step()
        if (ep + 1) % 3 == 0 or (ep + 1) == epochs:
            gen.eval()
            val_l1_sum, val_cnt = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(he_val), 32):
                    xb = torch.from_numpy(he_val[s : s + 32]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                    yb = torch.from_numpy(targ_val[s : s + 32]).to(device, dtype=torch.float32) / 255.0
                    val_l1_sum += float(F.l1_loss(gen(xb), yb, reduction="sum").item())
                    val_cnt += int(yb.numel())
            v_l1 = val_l1_sum / max(val_cnt, 1)
            if v_l1 < best_val_l1:
                best_val_l1 = v_l1
                best_ep = ep + 1
                best_sd = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}

    if best_sd is not None:
        gen.load_state_dict(best_sd)
    if disc is not None:
        del disc, opt_d
    torch.cuda.empty_cache()
    return gen, dict(params=n_params, best_epoch=best_ep, best_val_l1=round(best_val_l1, 6), train_sec=round(time.monotonic() - t0, 1))


def exact_r0_f1_u8(pred_u8: np.ndarray, targ_u8: np.ndarray, cutoff: int) -> float:
    p = pred_u8 > int(cutoff)
    g = targ_u8 > int(cutoff)
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & (~g)))
    fn = int(np.sum((~p) & g))
    denom = 2 * tp + fp + fn
    return float(2.0 * tp / denom) if denom > 0 else 0.0


def run_unseen_fold(
    *,
    arch: str,
    fold_idx: int,
    bank: dict,
    fold_spec: dict,
    out_dir: Path,
    device: torch.device,
    seeds: tuple[int, ...],
    up_epochs: int,
    head_epochs: int,
    overwrite: bool,
) -> None:
    out_json = out_dir / f"unseen5fold_{arch}_fold{fold_idx}.json"
    if out_json.exists() and not overwrite:
        print(json.dumps(dict(event="skip_existing", arch=arch, fold=fold_idx)), flush=True)
        return

    t0 = time.monotonic()
    inner_train_pats = set(roms._fold_patients(fold_spec, "inner_train"))
    inner_val_pats = set(roms._fold_patients(fold_spec, "inner_val"))
    outer_test_pats = set(roms._fold_patients(fold_spec, "outer_test"))

    # STRICT ZERO-LEAKAGE ASSERTION
    assert len((inner_train_pats | inner_val_pats) & outer_test_pats) == 0, "Patient leakage detected!"
    assert len(inner_train_pats & inner_val_pats) == 0, "Inner train/val leakage detected!"

    tr_idx_all = np.nonzero(np.isin(bank["patients"], list(inner_train_pats)))[0]
    va_idx_all = np.nonzero(np.isin(bank["patients"], list(inner_val_pats)))[0]
    ev_idx = np.nonzero(np.isin(bank["patients"], list(outer_test_pats)))[0]
    tr_idx = tr_idx_all[::2]  # fit_stride=2 matching S6

    # 1. Train fresh 16-ch upstream generator strictly on inner_train (validated on inner_val)
    gen, gen_info = train_fold_isolated_generator(
        arch=arch,
        fold_idx=fold_idx,
        he_tr=bank["he_nhwc"][tr_idx_all],
        targ_tr=bank["targ_nchw"][tr_idx_all],
        he_val=bank["he_nhwc"][va_idx_all],
        targ_val=bank["targ_nchw"][va_idx_all],
        device=device,
        epochs=up_epochs,
        batch_size=16 if arch != "unseen_resnet34_unet16" else 12,
    )
    print(json.dumps(dict(event="upstream_trained_zero_leakage", arch=arch, fold=fold_idx, **gen_info)), flush=True)

    # Freeze upstream generator and emit predictions
    up_tr_all = predict_16ch_u8(gen, bank["he_nhwc"][tr_idx_all], device)
    up_tr = up_tr_all[::2]
    up_ev = predict_16ch_u8(gen, bank["he_nhwc"][ev_idx], device)
    del gen
    torch.cuda.empty_cache()

    # 2. Calibrate physical cutoffs strictly on inner_train patients
    cutoffs_u8, tau_phys_float = roms.calibrate_orion_physical_cutoffs_from_train(
        bank["targ_nchw"][tr_idx_all], tr_idx_all, bank["nuclei_records"]
    )

    arms = ("upstream", "S_recon", "QuantileMatch", "E_count", "S_count")
    arm_tr = {a: (up_tr if a == "upstream" else np.empty_like(up_tr)) for a in arms}
    arm_ev = {a: (up_ev if a == "upstream" else np.empty_like(up_ev)) for a in arms}
    # Full inner_train (all tiles, not stride-2) for exact survival compartment percentile reference on triad [14=Pan-CK, 3=CD3e, 6=CD8a]
    triad_chs = (14, 3, 6)
    arm_tr_all_triad = {a: (up_tr_all[:, triad_chs] if a == "upstream" else np.empty((len(tr_idx_all), 3, 256, 256), dtype=np.uint8)) for a in arms}
    dc_drifts = {"E_count": [], "S_count": []}

    for ch16, m_name in enumerate(roms.ORION_MARKERS):
        cutoff = int(cutoffs_u8[ch16])
        c12_tr = roms.build_common12_orion(bank["he_nhwc"][tr_idx], up_tr, ch16, device=device)
        d_tr_t = torch.from_numpy(up_tr[:, ch16])
        y_tr_t = torch.from_numpy(bank["targ_nchw"][tr_idx, ch16])
        d_ev_t = torch.from_numpy(up_ev[:, ch16])

        # QuantileMatch (256-bin empirical CDF LUT strictly on inner_train)
        qm_tr, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr[:, ch16], bank["targ_nchw"][tr_idx, ch16], up_ev[:, ch16])
        arm_tr["QuantileMatch"][:, ch16] = qm_tr
        arm_ev["QuantileMatch"][:, ch16] = qm_ev
        if ch16 in triad_chs:
            ti = triad_chs.index(ch16)
            _, qm_tra, _ = rcomp.apply_quantile_match_u8(up_tr[:, ch16], bank["targ_nchw"][tr_idx, ch16], up_tr_all[:, ch16])
            arm_tr_all_triad["QuantileMatch"][:, ti] = qm_tra

        fitted: dict[tuple[str, int], object] = {}
        for arm in ("S_recon", "E_count", "S_count"):
            tr_list = []
            for sd in seeds:
                if arm == "E_count":
                    model, _ = rp2.fit_ablation_marker(
                        mode="E_count",
                        seed=sd,
                        common_train=c12_tr,
                        donor_u8_train=d_tr_t,
                        target_u8_train=y_tr_t,
                        cutoff=cutoff,
                        device=device,
                        epochs=head_epochs,
                        batch_size=16,
                        lr=1.5e-3,
                    )
                    p_tr, _ = rp2.predict_ablation_marker(model, "E_count", c12_tr, d_tr_t, device=device, batch_size=32)
                else:
                    model, _ = suh.fit_universal_marker(
                        arm=arm,
                        seed=sd,
                        common_train=c12_tr,
                        donor_u8_train=d_tr_t,
                        target_u8_train=y_tr_t,
                        cutoff=cutoff,
                        device=device,
                        epochs=head_epochs,
                        batch_size=16,
                        lr=1.5e-3,
                        deploy_anchor_weight=1.25,
                    )
                    p_tr, _ = suh.predict_universal_marker(model, c12_tr, d_tr_t, device=device, batch_size=32)
                tr_list.append(p_tr.astype(np.float32))
                fitted[(arm, sd)] = model.cpu()
            arm_tr[arm][:, ch16] = np.round(np.mean(tr_list, axis=0)).astype(np.uint8)

        del c12_tr
        gc.collect()

        # Predict on held-out unseen test patients
        c12_ev = roms.build_common12_orion(bank["he_nhwc"][ev_idx], up_ev, ch16, device=device)
        for arm in ("S_recon", "E_count", "S_count"):
            ev_list = []
            dc_vals = []
            for sd in seeds:
                model = fitted[(arm, sd)].to(device)
                if arm == "E_count":
                    p_ev, d_ev = rp2.predict_ablation_marker(model, "E_count", c12_ev, d_ev_t, device=device, batch_size=32)
                    dc_vals.append(float(d_ev["residual_mean_max"]))
                else:
                    p_ev, d_ev = suh.predict_universal_marker(model, c12_ev, d_ev_t, device=device, batch_size=32)
                    dc_vals.append(float(d_ev["residual_mean_max"]))
                ev_list.append(p_ev.astype(np.float32))
                del model
            arm_ev[arm][:, ch16] = np.round(np.mean(ev_list, axis=0)).astype(np.uint8)
            if arm in dc_drifts:
                dc_drifts[arm].append(max(dc_vals))
        del c12_ev

        # Also predict on full inner_train for survival triad percentile reference
        if ch16 in triad_chs:
            ti = triad_chs.index(ch16)
            c12_tr_all = roms.build_common12_orion(bank["he_nhwc"][tr_idx_all], up_tr_all, ch16, device=device)
            d_tra_t = torch.from_numpy(up_tr_all[:, ch16])
            for arm in ("S_recon", "E_count", "S_count"):
                tra_list = []
                for sd in seeds:
                    model = fitted[(arm, sd)].to(device)
                    if arm == "E_count":
                        p_a, _ = rp2.predict_ablation_marker(model, "E_count", c12_tr_all, d_tra_t, device=device, batch_size=32)
                    else:
                        p_a, _ = suh.predict_universal_marker(model, c12_tr_all, d_tra_t, device=device, batch_size=32)
                    tra_list.append(p_a.astype(np.float32))
                    del model
                arm_tr_all_triad[arm][:, ti] = np.round(np.mean(tra_list, axis=0)).astype(np.uint8)
            del c12_tr_all

        del fitted
        gc.collect()
        torch.cuda.empty_cache()

        if (ch16 + 1) % 4 == 0 or ch16 == 15:
            print(json.dumps(dict(event="channel_done", arch=arch, fold=fold_idx, ch=ch16 + 1, marker=m_name, sec=round(time.monotonic() - t0, 1))), flush=True)

    # 3. Full evaluation on 8 strictly unseen outer_test patients
    eval_pats = bank["patients"][ev_idx]
    uniq_pats = sorted(set(str(p) for p in eval_pats))
    ev_cells_ref = roms.gather_subset_cells(up_ev, ev_idx, bank["nuclei_records"])
    gt_cooc = np.zeros((15, 15), dtype=np.float64)
    lbl_gt = ev_cells_ref["labels15"].astype(np.float64)
    for i in range(15):
        si = float(np.std(lbl_gt[:, i]))
        for j in range(15):
            sj = float(np.std(lbl_gt[:, j]))
            if i == j:
                gt_cooc[i, j] = 1.0
            elif si > 1e-12 and sj > 1e-12:
                gt_cooc[i, j] = float(np.corrcoef(lbl_gt[:, i], lbl_gt[:, j])[0, 1])

    arms_eval = {}
    per_patient_macro = {p: {} for p in uniq_pats}
    oof_immunoscore = {a: {} for a in arms}
    tau_triad = {m: float(tau_phys_float[m]) for m in ("Pan-CK", "CD3e", "CD8a")}

    for arm in arms:
        per_m = {}
        pat_f1_accum = {p: [] for p in uniq_pats}
        r0_list = []
        for ch16, m_name in enumerate(roms.ORION_MARKERS):
            ch_r = jce.evaluate_channel_predictions(
                train_pred_u8=arm_tr[arm][:, ch16],
                train_targ_u8=bank["targ_nchw"][tr_idx, ch16],
                eval_pred_u8=arm_ev[arm][:, ch16],
                eval_targ_u8=bank["targ_nchw"][ev_idx, ch16],
                eval_clusters=eval_pats,
                fixed_cutoff=int(cutoffs_u8[ch16]),
                device=device,
            )
            r0_f1 = exact_r0_f1_u8(arm_ev[arm][:, ch16], bank["targ_nchw"][ev_idx, ch16], int(cutoffs_u8[ch16]))
            r0_list.append(r0_f1)
            pooled = dict(ch_r["pooled"])
            pooled["Exact_r0_F1"] = r0_f1
            per_m[m_name] = pooled
            for p_id in uniq_pats:
                if p_id in ch_r.get("per_cluster", {}):
                    pat_f1_accum[p_id].append(float(ch_r["per_cluster"][p_id]["Fixed_tol_F1"]))

        for p_id in uniq_pats:
            per_patient_macro[p_id][arm] = float(np.mean(pat_f1_accum[p_id])) if pat_f1_accum[p_id] else 0.0

        macro = {
            mk: float(np.mean([per_m[m][mk] for m in roms.ORION_MARKERS]))
            for mk in ("ODS_F", "AP", "Fixed_tol_F1", "Cal_F1", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed", "Exact_r0_F1")
        }

        tr_cells = roms.gather_subset_cells(arm_tr[arm], tr_idx, bank["nuclei_records"])
        ev_cells = roms.gather_subset_cells(arm_ev[arm], ev_idx, bank["nuclei_records"])
        dep_f1s, auprcs, aurocs = [], [], []
        for ch16 in range(1, 16):
            cls_idx = ch16 - 1
            m_nm = roms.ORION_MARKERS[ch16]
            res_c = roms.ccs.evaluate_single_cell_marker(
                train_gt_scores=tr_cells["gt_means16"][:, ch16],
                train_pred_scores=tr_cells["pred_means"][:, ch16],
                train_labels=tr_cells["labels15"][:, cls_idx],
                eval_pred_scores=ev_cells["pred_means"][:, ch16],
                eval_labels=ev_cells["labels15"][:, cls_idx],
                eval_tile_ids=ev_cells["tile_ids"],
                physical_cutoff=float(tau_phys_float[m_nm]),
            )
            dep_f1s.append(float(res_c["deployable_cell_f1"]))
            if np.isfinite(res_c.get("cell_auprc", np.nan)):
                auprcs.append(float(res_c["cell_auprc"]))
            if np.isfinite(res_c.get("cell_auroc", np.nan)):
                aurocs.append(float(res_c["cell_auroc"]))

        cooc_mat = ro5._compute_cooccurrence_corr(ev_cells["pred_means"], tau_phys_float)
        cooc_mae = float(np.mean(np.abs(cooc_mat - gt_cooc)))

        # Fold-isolated ImmunoScore out-of-fold scoring for survival
        tr_comp = roms.compute_patient_compartments_for_subset(
            tr_idx_all, bank["patients"][tr_idx_all], bank["nuclei_records"], arm_tr_all_triad[arm], tau_triad, use_ground_truth_labels=False
        )
        ev_comp = roms.compute_patient_compartments_for_subset(
            ev_idx, bank["patients"][ev_idx], bank["nuclei_records"], arm_ev[arm][:, triad_chs], tau_triad, use_ground_truth_labels=False
        )
        tr_p_sorted = sorted(p for p in inner_train_pats if p in tr_comp)
        ev_p_sorted = sorted(p for p in outer_test_pats if p in ev_comp)
        tr_vecs = {k: np.array([tr_comp[p][k] for p in tr_p_sorted]) for k in ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2")}
        ev_vecs = {k: np.array([ev_comp[p][k] for p in ev_p_sorted]) for k in ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2")}
        sc_iso = ccs.score_immunoscore_from_training_ref(tr_vecs, ev_vecs)
        for pi, pid in enumerate(ev_p_sorted):
            oof_immunoscore[arm][pid] = dict(
                continuous=float(sc_iso["immunoscore_continuous"][pi]),
                discrete=float(sc_iso["immunoscore_discrete"][pi]),
            )

        arms_eval[arm] = dict(
            macro_pixel_16ch=macro,
            per_marker_pixel=per_m,
            single_cell_15class=dict(
                macro_deployable_cell_f1=float(np.mean(dep_f1s)),
                macro_cell_auprc=float(np.mean(auprcs)) if auprcs else float("nan"),
                macro_cell_auroc=float(np.mean(aurocs)) if aurocs else float("nan"),
                cooccurrence_mae=cooc_mae,
            ),
            max_dc_drift=max(dc_drifts[arm]) if arm in dc_drifts and dc_drifts[arm] else 0.0,
        )

    rec = dict(
        schema="push3-orion-strictly-unseen-5fold-v1",
        cohort="Orion-CRC-Strictly-Unseen-Patients",
        architecture=arch,
        upstream_info=gen_info,
        zero_patient_overlap_verified=True,
        fold=fold_idx,
        n_inner_train_tiles=int(len(tr_idx_all)),
        n_inner_val_tiles=int(len(va_idx_all)),
        n_head_fit_tiles=int(len(tr_idx)),
        n_eval_tiles=int(len(ev_idx)),
        eval_patients=uniq_pats,
        wall_seconds=round(time.monotonic() - t0, 1),
        cutoffs_u8=[int(x) for x in cutoffs_u8],
        per_patient_macro_fixed_f1=per_patient_macro,
        oof_immunoscore=oof_immunoscore,
        arms=arms_eval,
    )
    _atomic_write_json(out_json, rec)
    print(
        json.dumps(
            dict(
                event="unseen_fold_done",
                arch=arch,
                fold=fold_idx,
                up_f1=round(arms_eval["upstream"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                sr_f1=round(arms_eval["S_recon"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                qm_f1=round(arms_eval["QuantileMatch"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                ec_f1=round(arms_eval["E_count"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                sc_f1=round(arms_eval["S_count"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                up_ssim=round(arms_eval["upstream"]["macro_pixel_16ch"]["SSIM"], 4),
                qm_ssim=round(arms_eval["QuantileMatch"]["macro_pixel_16ch"]["SSIM"], 4),
                sc_ssim=round(arms_eval["S_count"]["macro_pixel_16ch"]["SSIM"], 4),
                up_cell=round(arms_eval["upstream"]["single_cell_15class"]["macro_deployable_cell_f1"], 4),
                qm_cell=round(arms_eval["QuantileMatch"]["single_cell_15class"]["macro_deployable_cell_f1"], 4),
                sc_cell=round(arms_eval["S_count"]["single_cell_15class"]["macro_deployable_cell_f1"], 4),
                sec=rec["wall_seconds"],
            )
        ),
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", required=True, help="Comma-separated: unseen_unet16,unseen_resnet34_unet16,unseen_pix2pix16")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--bank-pkl", required=True)
    ap.add_argument("--cv-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="202610270,202610271")
    ap.add_argument("--up-epochs", type=int, default=22)
    ap.add_argument("--head-epochs", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_manifest = json.loads(Path(args.cv_manifest).read_text())
    with open(Path(args.bank_pkl).resolve(), "rb") as fh:
        bank = pickle.load(fh)

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    for arch in archs:
        for fi in folds:
            run_unseen_fold(
                arch=arch,
                fold_idx=fi,
                bank=bank,
                fold_spec=cv_manifest["folds"][fi],
                out_dir=out_dir,
                device=device,
                seeds=seeds,
                up_epochs=int(args.up_epochs),
                head_epochs=int(args.head_epochs),
                overwrite=bool(args.overwrite),
            )
    print("PUSH3_ORION_UNSEEN_BATCH_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
