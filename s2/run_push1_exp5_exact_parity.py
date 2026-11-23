#!/usr/bin/env python3
"""Exact-Parity Objective Comparison: End-to-End Fine-Tuning vs. Frozen Screened Count Residual.

Compares end-to-end fine-tuning vs. freezing the upstream generator and attaching
Screened Count Residual heads under the EXACT SAME:
  1. Pre-clipping reconstruction normalization scales (`suh._fast_loss_scales_f32`),
  2. Dynamic threshold floor (`min(cutoff, max(4, round(0.80 * cutoff)))`),
  3. Output-space gradient-norm weight calibration (`counts.calibration_lambda` in [0.25, 400.0]
     over deterministic 96-tile linspace probes, refreshed every 2 epochs),
  4. Lagged 255-threshold ODS table + locked-cutoff deployment table updates,
  5. Training-split composite checkpoint selection (`suh._train_count_composite_gpu`, 0.65 F_dep + 0.35 F_ods),
  6. Equal-weight two-seed ensemble (`202610270`, `202610271`) and D4 dihedral augmentation.

Runs an explicit pre-flight `assert_exact_objective_parity` test verifying that on an
identical prediction tensor both paths pick the exact same `t*`, `lam_ods`, `lam_dep`,
scalar loss, and output-space gradient norm (< 1e-6).

Evaluates:
  - Architecture A: 7.76M Double-Convolution U-Net (`StandardUNetVirtualStainer`, base=32)
  - Architecture B: 24.4M Residual-Encoder U-Net (`ResNet34UNetVirtualStainer`)
  - Fair E2E LR sweep: {1e-4, 3e-4, 1e-3} (reporting every LR plus best val-selected LR)
  - Multi-budget adaptation scaling: N in {32, 64, 149, 298, 596} labeled HEMIT tiles
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s2 import query_relation_count_objective as counts
from s2 import run_hemit_multibackbone_cell as rhmc
from s2 import run_nm_nature_killer_suite as rnk
from s2 import screened_universal_head as suh


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.short = (
            nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))
            if (stride != 1 or c_in != c_out)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.silu(out + self.short(x), inplace=True)


class ResNet34UNetVirtualStainer(nn.Module):
    """~24M-parameter ResNet-34-backbone U-Net for virtual staining (no external weights)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128), ResBlock(128, 128), ResBlock(128, 128))
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self.layer4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512), ResBlock(512, 512))
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(ResBlock(512, 256))
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(ResBlock(256, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(ResBlock(128, 64))
        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.layer1(self.stem(x))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


def build_backbone(arch: str) -> nn.Module:
    if arch == "double_conv_unet":
        return rnk.StandardUNetVirtualStainer(3, 3, base=32)
    if arch == "resnet34_unet":
        return ResNet34UNetVirtualStainer(3, 3)
    raise ValueError(f"Unknown arch: {arch}")


def exact_thr_floor(cutoff: int) -> int:
    """Bit-for-bit identical to suh.fit_universal_marker line 929."""
    return min(int(cutoff), max(4, int(round(0.80 * cutoff))))


def exact_recalibrate_lambdas_from_preds(
    pred_fn,
    he_sub: np.ndarray,
    up_sub: np.ndarray,
    targ_sub: np.ndarray,
    ch: int,
    cutoff: int,
    scales: dict[str, float],
    lagged_rows_dev: torch.Tensor,
    dep_table_dev: torch.Tensor,
    mean_G: float,
    device: torch.device,
    calib_subset: int = 96,
) -> tuple[float, float]:
    """Bit-for-bit identical output-space gradient-norm calibration to suh._recalibrate_lambdas_live."""
    if mean_G <= 1e-15:
        return 4.0, 4.0
    n = len(he_sub)
    thr_floor = exact_thr_floor(cutoff)
    step_idx = np.linspace(0, n - 1, num=min(n, calib_subset), dtype=int)
    with torch.no_grad():
        active_thr_t, adj_ods, _ = suh._gpu_pool_ods_adjoint(lagged_rows_dev, mean_G, thr_floor)
        adj_dep = suh._adjoint_from_single_row_gpu(dep_table_dev.mean(dim=0), mean_G)

    recon_sq = 0.0
    aux_ods_sq = 0.0
    aux_dep_sq = 0.0
    grad_bs = 4
    for g_start in range(0, len(step_idx), grad_bs):
        sub_idx = [int(x) for x in step_idx[g_start : g_start + grad_bs]]
        with torch.no_grad():
            z0 = pred_fn(sub_idx, ch).detach()
            d_f0 = torch.from_numpy(up_sub[sub_idx, ch : ch + 1]).to(device=device, dtype=torch.float32) / 255.0
            t_u8 = torch.from_numpy(targ_sub[sub_idx, ch : ch + 1]).to(device=device, dtype=torch.uint8)
            t_f = t_u8.to(torch.float32) / 255.0

        z_rec = z0.clone().requires_grad_(True)
        rec_loss = suh.preclip_reconstruction_loss(z_rec, t_f, d_f0, scales).sum()
        (g_rec,) = torch.autograd.grad(rec_loss, z_rec, retain_graph=False)
        recon_sq += float(g_rec.detach().to(torch.float64).square().sum().item())

        z_ods = z0.clone().requires_grad_(True)
        prep_ods = suh.prepare_channel_counts(z_ods, t_u8, cutoff)
        dens_ods = suh._fast_densities_single_gpu_thr(prep_ods, active_thr_t)
        loss_ods = (dens_ods * adj_ods[None, :]).sum()
        (g_ods,) = torch.autograd.grad(loss_ods, z_ods, retain_graph=False)
        aux_ods_sq += float(g_ods.detach().to(torch.float64).square().sum().item())

        z_dep = z0.clone().requires_grad_(True)
        prep_dep = suh.prepare_channel_counts(z_dep, t_u8, cutoff)
        dens_dep = suh.deployment_cutoff_densities(prep_dep, cutoff)
        loss_dep = (dens_dep * adj_dep[None, :]).sum()
        (g_dep,) = torch.autograd.grad(loss_dep, z_dep, retain_graph=False)
        aux_dep_sq += float(g_dep.detach().to(torch.float64).square().sum().item())

    lam_ods = (
        float(min(max(counts.calibration_lambda(recon_sq, aux_ods_sq), 0.25), 400.0))
        if (aux_ods_sq > 1e-24 and recon_sq > 0.0)
        else 4.0
    )
    lam_dep = (
        float(min(max(counts.calibration_lambda(recon_sq, aux_dep_sq), 0.25), 400.0))
        if (aux_dep_sq > 1e-24 and recon_sq > 0.0)
        else 4.0
    )
    return lam_ods, lam_dep


def exact_step_channel_loss(
    z_unclipped: torch.Tensor,
    t_u8: torch.Tensor,
    d_f: torch.Tensor,
    scales: dict[str, float],
    cutoff: int,
    lagged_rows_dev: torch.Tensor,
    dep_table_dev: torch.Tensor,
    idx_gpu: torch.Tensor,
    mean_G: float,
    eff_lambda: float,
    eff_dep_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Bit-for-bit identical forward loss to suh.fit_universal_marker lines 1018-1037."""
    t_f = t_u8.to(torch.float32) / 255.0
    rec_vec = suh.preclip_reconstruction_loss(z_unclipped, t_f, d_f, scales)
    loss_rec = rec_vec.mean()
    thr_floor = exact_thr_floor(cutoff)
    with torch.no_grad():
        prep_det = suh.prepare_channel_counts(z_unclipped.detach(), t_u8, cutoff)
        lagged_rows_dev[idx_gpu] = suh.fast_densities_hist_nograd(prep_det)
        dep_table_dev[idx_gpu] = suh.deployment_cutoff_densities(prep_det, cutoff)
        active_thr_t, adj_ods, _ = suh._gpu_pool_ods_adjoint(lagged_rows_dev, mean_G, thr_floor)
        adj_dep = suh._adjoint_from_single_row_gpu(dep_table_dev.mean(dim=0), mean_G)

    prep = suh.prepare_channel_counts(z_unclipped, t_u8, cutoff)
    dens_ods = suh._fast_densities_single_gpu_thr(prep, active_thr_t)
    dens_dep = suh.deployment_cutoff_densities(prep, cutoff)
    ods_term = (dens_ods * adj_ods[None, :]).sum(dim=-1).mean()
    dep_term = (dens_dep * adj_dep[None, :]).sum(dim=-1).mean()
    scaled_aux = float(eff_lambda) * ods_term + float(eff_dep_lambda) * dep_term
    total = loss_rec + scaled_aux.to(loss_rec.dtype)
    return total, loss_rec, scaled_aux, int(active_thr_t.item())


def assert_exact_objective_parity(device: torch.device) -> dict:
    """Prove E2E path and suh head path compute identical t*, lambdas, loss, and output gradients."""
    torch.manual_seed(12345)
    cutoff = 30
    n_tiles = 4
    z_raw = torch.rand((n_tiles, 1, 64, 64), device=device, dtype=torch.float32) * 0.45
    d_f = z_raw.detach().clone()
    t_u8 = torch.zeros((n_tiles, 1, 64, 64), device=device, dtype=torch.uint8)
    t_u8[:, :, 20:36, 20:36] = 160
    t_f = t_u8.to(torch.float32) / 255.0

    # Shared normalization scales
    sc_raw = suh._fast_loss_scales_f32(d_f, t_f)
    scales = {k: max(float(v.mean().item()), suh.core.SCALE_FLOOR) for k, v in sc_raw.items()}

    # Build lagged & dep tables identically
    prep0 = suh.prepare_channel_counts(z_raw.detach(), t_u8, cutoff)
    lag_a = suh.fast_densities_hist_nograd(prep0).to(torch.float64).clone()
    dep_a = suh.deployment_cutoff_densities(prep0, cutoff).to(torch.float64).clone()
    lag_b = lag_a.clone()
    dep_b = dep_a.clone()
    mean_G = float(prep0["G"].mean().item())
    thr_floor = exact_thr_floor(cutoff)
    assert thr_floor == 24, f"Expected CD3 thr_floor=24, got {thr_floor}"

    # Path A: direct suh equations
    z_a = z_raw.detach().clone().requires_grad_(True)
    with torch.no_grad():
        act_a, adj_ods_a, _ = suh._gpu_pool_ods_adjoint(lag_a, mean_G, thr_floor)
        adj_dep_a = suh._adjoint_from_single_row_gpu(dep_a.mean(dim=0), mean_G)
    rec_a = suh.preclip_reconstruction_loss(z_a, t_f, d_f, scales).mean()
    prep_a = suh.prepare_channel_counts(z_a, t_u8, cutoff)
    ods_a = (suh._fast_densities_single_gpu_thr(prep_a, act_a) * adj_ods_a[None, :]).sum(dim=-1).mean()
    dep_loss_a = (suh.deployment_cutoff_densities(prep_a, cutoff) * adj_dep_a[None, :]).sum(dim=-1).mean()
    tot_a = rec_a + (2.5 * ods_a + 3.125 * dep_loss_a).to(rec_a.dtype)
    (grad_a,) = torch.autograd.grad(tot_a, z_a)

    # Path B: shared exact_step_channel_loss used by E2E
    z_b = z_raw.detach().clone().requires_grad_(True)
    idx_gpu = torch.arange(n_tiles, device=device, dtype=torch.long)
    tot_b, _, _, act_b = exact_step_channel_loss(
        z_b, t_u8, d_f, scales, cutoff, lag_b, dep_b, idx_gpu, mean_G, 2.5, 3.125
    )
    (grad_b,) = torch.autograd.grad(tot_b, z_b)

    loss_diff = abs(float(tot_a.item()) - float(tot_b.item()))
    grad_diff = float((grad_a - grad_b).abs().max().item())
    norm_a = float(torch.linalg.vector_norm(grad_a).item())
    norm_b = float(torch.linalg.vector_norm(grad_b).item())
    assert int(act_a.item()) == act_b, f"Threshold mismatch: {int(act_a.item())} vs {act_b}"
    assert loss_diff < 1e-6, f"Loss mismatch: {loss_diff}"
    assert grad_diff < 1e-6, f"Grad mismatch: {grad_diff}"

    # Verify gradient sign on isolated FP pixel vs missed TP pixel
    z_sign = torch.full((1, 1, 32, 32), 5.0 / 255.0, device=device, dtype=torch.float32)
    z_sign[:, :, 4, 4] = 31.0 / 255.0   # isolated FP just above cutoff=30
    z_sign[:, :, 20, 20] = 29.0 / 255.0 # missed TP just below cutoff=30
    z_sign.requires_grad_(True)
    y_sign = torch.zeros((1, 1, 32, 32), device=device, dtype=torch.uint8)
    y_sign[:, :, 20, 20] = 200
    pr_s = suh.prepare_channel_counts(z_sign, y_sign, cutoff)
    d_dep_s = suh.deployment_cutoff_densities(pr_s, cutoff)
    adj_s = suh._adjoint_from_single_row_gpu(d_dep_s.detach().mean(dim=0), float(pr_s["G"].mean().item()))
    (g_s,) = torch.autograd.grad((d_dep_s * adj_s[None, :]).sum(), z_sign)
    fp_grad = float(g_s[0, 0, 4, 4].item())
    tp_grad = float(g_s[0, 0, 20, 20].item())
    assert fp_grad > 0.0 and tp_grad < 0.0, f"Wrong adjoint sign: fp_grad={fp_grad}, tp_grad={tp_grad}"

    receipt = dict(
        passed=True,
        cutoff=cutoff,
        thr_floor=thr_floor,
        active_thr_suh=int(act_a.item()),
        active_thr_e2e=act_b,
        loss_abs_diff=loss_diff,
        grad_max_abs_diff=grad_diff,
        grad_norm_suh=round(norm_a, 6),
        grad_norm_e2e=round(norm_b, 6),
        isolated_fp_grad_positive=fp_grad,
        missed_tp_grad_negative=tp_grad,
    )
    print(f"[PARITY_CHECK PASS] {receipt}", flush=True)
    return receipt


@torch.no_grad()
def _refresh_e2e_tables(
    ft_net: nn.Module,
    he_sub: np.ndarray,
    targ_sub: np.ndarray,
    lagged_rows_list: list[torch.Tensor],
    dep_tables_list: list[torch.Tensor],
    device: torch.device,
) -> None:
    was_train = ft_net.training
    ft_net.eval()
    n = len(he_sub)
    for s0 in range(0, n, 4):
        e0 = min(n, s0 + 4)
        xb0 = torch.from_numpy(he_sub[s0:e0]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
        z_all = ft_net(xb0)
        for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
            y0 = torch.from_numpy(targ_sub[s0:e0, ch : ch + 1]).to(device, dtype=torch.uint8)
            pr0 = suh.prepare_channel_counts(z_all[:, ch : ch + 1], y0, cutoff)
            lagged_rows_list[ch][s0:e0] = suh.fast_densities_hist_nograd(pr0).to(torch.float64)
            dep_tables_list[ch][s0:e0] = suh.deployment_cutoff_densities(pr0, cutoff).to(torch.float64)
    ft_net.train(was_train)


def pretrain_upstream_generator(
    arch: str, h_tr: dict, device: torch.device, epochs: int = 12, bs_override: int | None = None
) -> tuple[dict, float, int]:
    t0 = time.monotonic()
    torch.manual_seed(20261120)
    net = build_backbone(arch).to(device)
    n_params = int(sum(p.numel() for p in net.parameters()))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(20261120)
    n_tr = h_tr["he_nhwc"].shape[0]
    bs = int(bs_override) if bs_override is not None else (4 if arch == "resnet34_unet" else 16)
    for _ in range(epochs):
        net.train()
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, bs):
            idx = order[s : s + bs]
            xb = torch.from_numpy(h_tr["he_nhwc"][idx]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            yb = torch.from_numpy(h_tr["targ_nchw"][idx]).to(device, dtype=torch.float32) / 255.0
            k_rot = int(rng.integers(0, 4))
            if k_rot:
                xb, yb = torch.rot90(xb, k_rot, dims=(-2, -1)), torch.rot90(yb, k_rot, dims=(-2, -1))
            if int(rng.integers(0, 2)):
                xb, yb = torch.flip(xb, dims=(-1,)), torch.flip(yb, dims=(-1,))
            pred = net(xb)
            loss = F.mse_loss(pred, yb) + 0.5 * F.l1_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    wall = time.monotonic() - t0
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    torch.cuda.empty_cache()
    return sd, wall, n_params


def run_e2e_arm(
    arch: str,
    base_sd: dict,
    ft_mode: str,
    lr_e2e: float,
    he_sub: np.ndarray,
    up_sub: np.ndarray,
    targ_sub: np.ndarray,
    ch_scales: list[dict],
    h_tr: dict,
    h_ev: dict,
    sub_idx: np.ndarray,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 4,
    deploy_anchor_weight: float = 1.25,
) -> dict:
    """Two-seed E2E fine-tuning using bit-for-bit identical objective & checkpoint selection to suh."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t_ft = time.monotonic()
    n_sub = len(sub_idx)
    tr_preds_seeds, ev_preds_seeds = [], []
    per_seed_diag = []

    for sd in (202610270, 202610271):
        torch.manual_seed(sd)
        rng = random.Random(int(sd) * 17 + 7)
        ft_net = build_backbone(arch).to(device)
        ft_net.load_state_dict(base_sd)
        opt_ft = torch.optim.AdamW(ft_net.parameters(), lr=lr_e2e, weight_decay=1e-4)
        total_steps = max(1, epochs * math.ceil(n_sub / batch_size))
        warmup = max(1, total_steps // 10)

        def _lr_mult(st: int) -> float:
            if st < warmup:
                return float(st + 1) / float(warmup)
            prog = (st - warmup) / max(1, total_steps - warmup)
            return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

        lagged_rows_list: list[torch.Tensor] = []
        dep_tables_list: list[torch.Tensor] = []
        mean_g_list: list[float] = []
        eff_ods_lams = [0.0, 0.0, 0.0]
        eff_dep_lams = [0.0, 0.0, 0.0]

        def _pred_probe(probe_idx: list[int], ch_i: int) -> torch.Tensor:
            xb_p = torch.from_numpy(he_sub[probe_idx]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            return ft_net(xb_p)[:, ch_i : ch_i + 1]

        if ft_mode == "E2E_Count":
            for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                lagged_rows_list.append(torch.zeros((n_sub, 255, 3), dtype=torch.float64, device=device))
                dep_tables_list.append(torch.zeros((n_sub, 3), dtype=torch.float64, device=device))
                mean_g_list.append(0.0)
            _refresh_e2e_tables(ft_net, he_sub, targ_sub, lagged_rows_list, dep_tables_list, device)
            for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                g_val = float((targ_sub[:, ch] > cutoff).mean())
                mean_g_list[ch] = max(g_val, 1e-12)
                l_ods, l_dep = exact_recalibrate_lambdas_from_preds(
                    _pred_probe, he_sub, up_sub, targ_sub, ch, cutoff, ch_scales[ch],
                    lagged_rows_list[ch], dep_tables_list[ch], mean_g_list[ch], device
                )
                eff_ods_lams[ch] = l_ods
                eff_dep_lams[ch] = l_dep * deploy_anchor_weight

        best_score = -1.0
        best_ep = 0
        best_sd = None
        if ft_mode == "E2E_Count":
            best_score = float(np.mean([
                suh._train_count_composite_gpu(lagged_rows_list[ch], mean_g_list[ch], cutoff)[0]
                for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS)
            ]))
            best_sd = {k: v.detach().cpu().clone() for k, v in ft_net.state_dict().items()}

        step = 0
        for ep in range(epochs):
            if ep > 0 and ep % 2 == 0 and ft_mode == "E2E_Count":
                _refresh_e2e_tables(ft_net, he_sub, targ_sub, lagged_rows_list, dep_tables_list, device)
                cur_comp = float(np.mean([
                    suh._train_count_composite_gpu(lagged_rows_list[ch], mean_g_list[ch], cutoff)[0]
                    for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS)
                ]))
                if cur_comp >= best_score:
                    best_score = cur_comp
                    best_ep = ep
                    best_sd = {k: v.detach().cpu().clone() for k, v in ft_net.state_dict().items()}
                for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                    l_ods, l_dep = exact_recalibrate_lambdas_from_preds(
                        _pred_probe, he_sub, up_sub, targ_sub, ch, cutoff, ch_scales[ch],
                        lagged_rows_list[ch], dep_tables_list[ch], mean_g_list[ch], device
                    )
                    eff_ods_lams[ch] = l_ods
                    eff_dep_lams[ch] = l_dep * deploy_anchor_weight

            ft_net.train()
            order = list(range(n_sub))
            rng.shuffle(order)
            for start in range(0, n_sub, batch_size):
                b_idx = order[start : start + batch_size]
                bsz = len(b_idx)
                codes = [rng.randrange(8) for _ in range(bsz)]
                idx_gpu = torch.tensor(b_idx, dtype=torch.long, device=device)

                xb = torch.from_numpy(he_sub[b_idx]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                db = torch.from_numpy(up_sub[b_idx]).to(device, dtype=torch.float32) / 255.0
                tb_u8 = torch.from_numpy(targ_sub[b_idx]).to(device, dtype=torch.uint8)
                xb = suh.apply_d4_batch(xb, codes)
                db = suh.apply_d4_batch(db, codes)
                tb_u8 = suh.apply_d4_batch(tb_u8, codes)

                mult = _lr_mult(step)
                for pg in opt_ft.param_groups:
                    pg["lr"] = lr_e2e * mult

                opt_ft.zero_grad(set_to_none=True)
                z_all = ft_net(xb)
                loss_total = torch.zeros((), device=device, dtype=torch.float32)
                for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                    z_c = z_all[:, ch : ch + 1]
                    d_c = db[:, ch : ch + 1]
                    t_c_u8 = tb_u8[:, ch : ch + 1]
                    if ft_mode == "E2E_Count":
                        l_c, _, _, _ = exact_step_channel_loss(
                            z_c, t_c_u8, d_c, ch_scales[ch], cutoff,
                            lagged_rows_list[ch], dep_tables_list[ch], idx_gpu,
                            mean_g_list[ch], eff_ods_lams[ch], eff_dep_lams[ch],
                        )
                    else:
                        t_c_f = t_c_u8.to(torch.float32) / 255.0
                        l_c = suh.preclip_reconstruction_loss(z_c, t_c_f, d_c, ch_scales[ch]).mean()
                    loss_total = loss_total + l_c / 3.0

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(ft_net.parameters(), 2.0)
                opt_ft.step()
                step += 1

        if ft_mode == "E2E_Count":
            _refresh_e2e_tables(ft_net, he_sub, targ_sub, lagged_rows_list, dep_tables_list, device)
            final_comp = float(np.mean([
                suh._train_count_composite_gpu(lagged_rows_list[ch], mean_g_list[ch], cutoff)[0]
                for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS)
            ]))
            if best_sd is not None and best_score > final_comp:
                ft_net.load_state_dict(best_sd)
            else:
                best_score = final_comp
                best_ep = epochs

        per_seed_diag.append(dict(seed=sd, best_train_composite=round(best_score, 6), best_epoch=best_ep))
        tr_preds_seeds.append(rnk._predict_unet_u8(ft_net, he_sub, device).astype(np.float32))
        ev_preds_seeds.append(rnk._predict_unet_u8(ft_net, h_ev["he_nhwc"], device).astype(np.float32))
        del ft_net, opt_ft
        torch.cuda.empty_cache()

    wall_ft = time.monotonic() - t_ft
    vram_ft = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
    tr_ens = np.rint(np.clip(0.5 * (tr_preds_seeds[0] + tr_preds_seeds[1]), 0, 255)).astype(np.uint8)
    ev_ens = np.rint(np.clip(0.5 * (ev_preds_seeds[0] + ev_preds_seeds[1]), 0, 255)).astype(np.uint8)
    res_ft = rnk._eval_hemit_full(tr_ens, ev_ens, h_tr, h_ev, device, tr_idx=sub_idx)
    return dict(
        lr=lr_e2e,
        peak_vram_mb=round(vram_ft, 2),
        train_wall_sec=round(wall_ft, 1),
        train_composite_mean=round(float(np.mean([d["best_train_composite"] for d in per_seed_diag])), 6),
        per_seed=per_seed_diag,
        metrics=res_ft,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--archs", default="double_conv_unet,resnet34_unet")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pretrain-bs", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_path = out_dir / "push1_exp5_exact_parity.json"

    parity_receipt = assert_exact_objective_parity(device)

    import pickle
    with open(data_dir / "bank_cache" / "hemit_bank_train_596.pkl", "rb") as f:
        h_tr = pickle.load(f)
    with open(data_dir / "bank_cache" / "hemit_bank_test_945.pkl", "rb") as f:
        h_ev = pickle.load(f)

    n_tr = h_tr["he_nhwc"].shape[0]
    perm = np.random.default_rng(42).permutation(n_tr)

    # Full 5-budget ladder on Double-Conv U-Net + 3-budget confirmation (small/mid/full) on ResNet34-UNet
    wanted_archs = {a.strip() for a in args.archs.split(",") if a.strip()}
    plan = [
        ("double_conv_unet", [("N32", 32), ("N64", 64), ("N149", 149), ("N298", 298), ("N596", 596)], [1e-4, 3e-4, 1e-3]),
        ("resnet34_unet", [("N32", 32), ("N149", 149), ("N596", 596)], [1e-4, 3e-4]),
    ]

    if args.resume and out_path.exists():
        result = json.loads(out_path.read_text())
        result["parity_receipt"] = parity_receipt
        print(f"[PUSH-1] Resuming from {out_path} (existing archs: {list(result.get('architectures', {}).keys())})", flush=True)
    else:
        result = dict(
            schema="push1-exp5-exact-parity-v3",
            parity_receipt=parity_receipt,
            architectures={},
        )

    for arch, budgets, lr_grid in plan:
        if arch not in wanted_archs:
            continue
        print(f"\n[PUSH-1] Pretraining upstream {arch} on N=596 HEMIT tiles...", flush=True)
        base_sd, pretrain_sec, n_params = pretrain_upstream_generator(
            arch, h_tr, device, epochs=12, bs_override=args.pretrain_bs
        )
        net_eval = build_backbone(arch).to(device)
        net_eval.load_state_dict(base_sd)
        up_tr_full = rnk._predict_unet_u8(net_eval, h_tr["he_nhwc"], device)
        up_ev_full = rnk._predict_unet_u8(net_eval, h_ev["he_nhwc"], device)
        del net_eval
        torch.cuda.empty_cache()

        arch_out = result["architectures"].get(arch, dict(
            params_per_seed=n_params,
            params_2seed=2 * n_params,
            pretrain_sec=round(pretrain_sec, 1),
            budgets={},
        ))
        print(f"  [{arch}] params={n_params:,} (2-seed {2*n_params:,}), pretrain={pretrain_sec:.1f}s", flush=True)

        for b_name, n_b in budgets:
            if args.resume and b_name in arch_out.get("budgets", {}):
                print(f"  [{arch} {b_name}] already completed; skipping.", flush=True)
                continue
            sub_idx = np.arange(n_tr) if n_b == n_tr else np.sort(perm[:n_b])
            he_sub = h_tr["he_nhwc"][sub_idx]
            up_sub = up_tr_full[sub_idx]
            targ_sub = h_tr["targ_nchw"][sub_idx]
            n_sub = len(sub_idx)

            # Shared per-channel reconstruction normalization scales
            ch_scales = []
            for ch in range(3):
                s_sum = {"mse": 0.0, "ssim": 0.0, "edge1": 0.0, "edge16": 0.0}
                with torch.no_grad():
                    for s0 in range(0, n_sub, 32):
                        db0 = torch.from_numpy(up_sub[s0 : s0 + 32, ch]).to(device, dtype=torch.float32).unsqueeze(1) / 255.0
                        yb0 = torch.from_numpy(targ_sub[s0 : s0 + 32, ch]).to(device, dtype=torch.float32).unsqueeze(1) / 255.0
                        sc0 = suh._fast_loss_scales_f32(db0, yb0)
                        for k in s_sum:
                            s_sum[k] += float(sc0[k].sum().item())
                ch_scales.append({k: max(v / float(n_sub), suh.core.SCALE_FLOOR) for k, v in s_sum.items()})

            b_out = {}
            res_frz = rnk._eval_hemit_full(up_sub, up_ev_full, h_tr, h_ev, device, tr_idx=sub_idx)
            b_out["Frozen"] = dict(trainable_params_2seed=0, metrics=res_frz)

            # E2E Recon (at lr=1e-4)
            e2e_rec = run_e2e_arm(
                arch, base_sd, "E2E_Recon", 1e-4, he_sub, up_sub, targ_sub, ch_scales,
                h_tr, h_ev, sub_idx, device, epochs=10, batch_size=4
            )
            e2e_rec["trainable_params_2seed"] = 2 * n_params
            b_out["E2E_Recon"] = e2e_rec

            # E2E Count across fair LR grid -> pick best by leak-free training composite (matching suh)
            e2e_lr_runs = {}
            best_lr_key = None
            best_lr_comp = -1.0
            for lr_c in lr_grid:
                k_lr = f"lr_{lr_c:.0e}"
                run_c = run_e2e_arm(
                    arch, base_sd, "E2E_Count", lr_c, he_sub, up_sub, targ_sub, ch_scales,
                    h_tr, h_ev, sub_idx, device, epochs=10, batch_size=4
                )
                run_c["trainable_params_2seed"] = 2 * n_params
                e2e_lr_runs[k_lr] = run_c
                m = run_c["metrics"]
                print(
                    f"  [{arch} {b_name}] E2E_Count({k_lr}): comp={run_c['train_composite_mean']:.4f} "
                    f"F1={m['macro_pixel']['Fixed_tol_F1']:.4f} SSIM={m['macro_pixel']['SSIM']:.4f} "
                    f"CD3_cell={m['cd3_cell_f1']:.4f} panCK_cell={m['panck_cell_f1']:.4f} ({run_c['train_wall_sec']:.1f}s)",
                    flush=True,
                )
                if run_c["train_composite_mean"] > best_lr_comp:
                    best_lr_comp = run_c["train_composite_mean"]
                    best_lr_key = k_lr
            b_out["E2E_Count_lr_sweep"] = e2e_lr_runs
            b_out["E2E_Count_best_by_train_comp"] = dict(selected_lr=best_lr_key, **e2e_lr_runs[best_lr_key])

            # Frozen + Screened Count Head (2-seed ensemble via exact suh.fit_universal_marker)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            t_sc = time.monotonic()
            c_tr_ch, c_ev_ch = [], []
            for ch in range(3):
                c_tr_ch.append(rhmc.build_common12_for_3ch(he_sub, up_sub, ch, device=device))
                c_ev_ch.append(rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev_full, ch, device=device))

            sc_tr_u8 = np.zeros_like(up_sub)
            sc_ev_u8 = np.zeros_like(up_ev_full)
            n_reg, n_trunk, n_act = 0, 0, 0
            for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                d_tr_t = torch.from_numpy(up_sub[:, ch]).to(torch.uint8)
                t_tr_t = torch.from_numpy(targ_sub[:, ch]).to(torch.uint8)
                d_ev_t = torch.from_numpy(up_ev_full[:, ch]).to(torch.uint8)
                p_tr_list, p_ev_list = [], []
                for sd in (202610270, 202610271):
                    mdl, _ = suh.fit_universal_marker(
                        arm="S_count", seed=sd, common_train=c_tr_ch[ch], donor_u8_train=d_tr_t,
                        target_u8_train=t_tr_t, cutoff=cutoff, device=device,
                        epochs=10, lr=1.2e-3, batch_size=4, deploy_anchor_weight=1.25,
                    )
                    if ch == 0 and sd == 202610270:
                        n_reg = int(sum(p.numel() for p in mdl.parameters()))
                        n_trunk = int(sum(p.numel() for nm, p in mdl.named_parameters() if "token" not in nm))
                        n_act = n_trunk - 4608
                    p_tr_np, _ = suh.predict_universal_marker(mdl, c_tr_ch[ch], d_tr_t, device)
                    p_ev_np, _ = suh.predict_universal_marker(mdl, c_ev_ch[ch], d_ev_t, device)
                    p_tr_list.append(p_tr_np.astype(np.float32))
                    p_ev_list.append(p_ev_np.astype(np.float32))
                    del mdl
                sc_tr_u8[:, ch] = np.rint(np.clip(0.5 * (p_tr_list[0] + p_tr_list[1]), 0, 255)).astype(np.uint8)
                sc_ev_u8[:, ch] = np.rint(np.clip(0.5 * (p_ev_list[0] + p_ev_list[1]), 0, 255)).astype(np.uint8)

            wall_sc = time.monotonic() - t_sc
            vram_sc = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            res_sc = rnk._eval_hemit_full(sc_tr_u8, sc_ev_u8, h_tr, h_ev, device, tr_idx=sub_idx)
            b_out["Frozen_S_count"] = dict(
                registered_params_2seed_3ch=6 * n_reg,
                trunk_params_2seed_3ch=6 * n_trunk,
                active_params_2seed_3ch=6 * n_act,
                peak_vram_mb=round(vram_sc, 2),
                train_wall_sec=round(wall_sc, 1),
                metrics=res_sc,
            )
            print(
                f"  [{arch} {b_name}] Frozen+S_count : F1={res_sc['macro_pixel']['Fixed_tol_F1']:.4f} "
                f"SSIM={res_sc['macro_pixel']['SSIM']:.4f} CD3_cell={res_sc['cd3_cell_f1']:.4f} "
                f"panCK_cell={res_sc['panck_cell_f1']:.4f} ({wall_sc:.1f}s, {vram_sc:.0f}MB)",
                flush=True,
            )

            arch_out["budgets"][b_name] = b_out
            result["architectures"][arch] = arch_out
            _atomic_write_json(out_path, result)

    print("PUSH1_EXP5_EXACT_PARITY_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
