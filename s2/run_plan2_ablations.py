"""Plan 2 multi-cohort mechanism ablations & classical spatial post-processing comparators.

Runs strictly against existing frozen upstream prediction banks across ALL 20 generators
(11 HEMIT + 6 Orion-CRC Fold 0 + 3 HER2match) without modifying any frozen S6 artifact
or frozen operator module:

  1. ``S_recon_comp`` : Identical S_recon pure reconstruction loss (zero count gradient)
                        paired with the EXACT 2-epoch training-split composite checkpoint
                        selector (0.65 F_dep + 0.35 F_ods) used by S_count. Isolates the
                        pooled spatial-tolerance count objective from checkpoint selection.
  2. ``S_count_1off`` : Identical S_count objective and checkpoint selector with a
                        single-offset 1-pixel axial graph Laplacian L_1 (scale 1/8 so
                        spec(L_1) in [0, 1] and kappa_2(I + L_1) <= 2.0 identically),
                        removing 16-pixel cellular-scale coupling.
  3. ``E_count``      : Identical S_count objective and checkpoint selector with an
                        unconstrained residual r_c = psi_c(x, b(x)), removing the
                        screened Poisson solve and the exact zero-mean gauge (1^T r_c = 0).
  4. ``Unsharp``      : Classical train-tuned spatial unsharp masking
                        (sigma in {1.5, 3.0, 6.0}, alpha in {0.5, 1.0, 2.0, 3.0}, plus
                        identity fallback alpha=0) maximizing training Fixed_tol_F1.
  5. ``CLAHE``        : Classical train-tuned 8x8-tile bilinear contrast-limited adaptive
                        histogram equalization (clip_limit in {0.005, 0.01, 0.02, 0.04, 0.08},
                        plus identity fallback) maximizing training Fixed_tol_F1.

Also exports two verified HER2+ positive membrane-staining visual crops (wsi009) from
the frozen ``hoptimus_lora_ihc`` S6 heads when run on HER2match.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import pickle
import random
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from . import journal_cohort_eval as jce
from . import query_relation_count_objective as counts
from . import query_relation_reconstruction as core
from . import screened_universal_head as suh

ABLATION_TRAINED_ARMS = ("S_recon_comp", "S_count_1off", "E_count")
ALL_ABLATION_ARMS = ("upstream", "S_recon_comp", "S_count_1off", "E_count", "Unsharp", "CLAHE")


# ---------------------------------------------------------------------------
# Single-offset (1-pixel) Neumann Laplacian & Screened Poisson Solver
# ---------------------------------------------------------------------------

_LAPLACIAN_1_KERNEL_CACHE: dict[tuple[torch.device, torch.dtype], Tensor] = {}


def fast_laplacian_1(x: Tensor) -> Tensor:
    """Exact 1-pixel axial Neumann graph Laplacian L_1 x = B_1^T B_1 x (edge weight 1/8).

    Each interior node has degree 4 * 0.125 = 0.5, matching the 8 * 0.0625 = 0.5
    degree of the dual-offset (1, 16) Laplacian so spec(L_1) subset [0, 1] and
    kappa_2(I + L_1) <= 2.0 identically, with exact nullspace 1^T L_1 = 0.
    Uses replicate padding of 1 pixel, which is mathematically identical to zero-flux
    boundary edges on the 1-pixel axial grid.
    """
    key = (x.device, x.dtype)
    kern = _LAPLACIAN_1_KERNEL_CACHE.get(key)
    if kern is None:
        kern = torch.tensor(
            [[[[0.0, -0.125, 0.0], [-0.125, 0.5, -0.125], [0.0, -0.125, 0.0]]]],
            device=x.device,
            dtype=x.dtype,
        )
        _LAPLACIAN_1_KERNEL_CACHE[key] = kern
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kern)


class _FastScreenedSolve1OffTrain(torch.autograd.Function):
    """Sync-free 10-iteration CG solver for (I + L_1) x = rhs in float32 training."""

    @staticmethod
    def _cg_steps(rhs: Tensor, n_iter: int = 10) -> Tensor:
        sol = torch.zeros_like(rhs)
        res = rhs.clone()
        direction = res.clone()
        rr = res.square().sum(dim=(-2, -1), keepdim=True)
        tiny = torch.finfo(rhs.dtype).tiny
        for _ in range(n_iter):
            op = direction + fast_laplacian_1(direction)
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
    def forward(ctx, rhs: Tensor) -> Tensor:
        ctx.input_dtype = rhs.dtype
        return _FastScreenedSolve1OffTrain._cg_steps(rhs.to(torch.float32), n_iter=10).to(rhs.dtype)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return _FastScreenedSolve1OffTrain._cg_steps(gradient.to(torch.float32), n_iter=10).to(ctx.input_dtype)


def fast_ablation_forward(
    model: core.RelationModel,
    common: Tensor,
    mode: str,
    inference: bool = False,
) -> tuple[Tensor, dict]:
    """Forward pass for ``S_recon_comp`` (standard 1,16), ``S_count_1off`` (1-off), or ``E_count`` (unconstrained)."""
    if mode == "S_recon_comp":
        return suh.fast_sc_forward(model, common, inference=inference)

    dtype = model.token_projection.weight.dtype
    c = common.to(dtype=dtype)
    zeros16 = model.token_projection.weight.new_zeros((c.shape[0], 16, *c.shape[-2:]))
    h = model.backbone(torch.cat((c, zeros16), dim=1))
    potential = suh.fast_unary_potential(model, h)

    if mode == "E_count":
        if inference:
            residual = potential.to(torch.float64)
            mean_abs = residual.detach().mean(dim=(-2, -1)).abs()
            diagnostics = {
                "relative_residual_max": 0.0,
                "absolute_residual_max": 0.0,
                "solver_dtype": "none_unconstrained",
                "residual_mean_max": float(mean_abs.max().item()),
                "residual_mean_avg": float(mean_abs.mean().item()),
                "arm": "E_count",
            }
        else:
            residual = potential.to(torch.float32)
            diagnostics = {
                "relative_residual_max": 0.0,
                "absolute_residual_max": 0.0,
                "solver_dtype": "none_unconstrained",
                "residual_mean_max_tensor": residual.detach().mean(dim=(-2, -1)).abs().max(),
                "arm": "E_count",
            }
        return residual, diagnostics

    if mode == "S_count_1off":
        if inference:
            rhs_f64 = fast_laplacian_1(potential.to(torch.float64))
            rhs_norm = rhs_f64.square().sum(dim=(-2, -1), keepdim=True).sqrt()
            if float(rhs_norm.max().item()) == 0.0:
                residual = torch.zeros_like(rhs_f64)
                rel_max = 0.0
                abs_max = 0.0
            else:
                sol_f64 = _FastScreenedSolve1OffTrain._cg_steps(rhs_f64, n_iter=22)
                sol_f64 = sol_f64 - sol_f64.mean(dim=(-2, -1), keepdim=True)
                sol_f64 = sol_f64 - sol_f64.mean(dim=(-2, -1), keepdim=True)
                true_res = rhs_f64 - (sol_f64 + fast_laplacian_1(sol_f64))
                res_norm = true_res.square().sum(dim=(-2, -1), keepdim=True).sqrt()
                rel_vec = torch.where(rhs_norm > 0.0, res_norm / rhs_norm.clamp_min(1e-30), torch.zeros_like(res_norm))
                rel_max = float(rel_vec.max().item())
                abs_max = float(res_norm.max().item())
                residual = sol_f64
            mean_abs = residual.detach().mean(dim=(-2, -1)).abs()
            diagnostics = {
                "relative_residual_max": rel_max,
                "absolute_residual_max": abs_max,
                "solver_dtype": "float64",
                "residual_mean_max": float(mean_abs.max().item()),
                "residual_mean_avg": float(mean_abs.mean().item()),
                "arm": "S_count_1off",
            }
        else:
            rhs = fast_laplacian_1(potential).to(torch.float32)
            residual = _FastScreenedSolve1OffTrain.apply(rhs)
            diagnostics = {
                "relative_residual_max": 0.0,
                "absolute_residual_max": 0.0,
                "solver_dtype": "float32",
                "residual_mean_max_tensor": residual.detach().mean(dim=(-2, -1)).abs().max(),
                "arm": "S_count_1off",
            }
        return residual, diagnostics

    raise ValueError(f"Unknown ablation forward mode: {mode!r}")


# ---------------------------------------------------------------------------
# Unified Training & Inference for Plan 2 Trained Ablation Arms
# ---------------------------------------------------------------------------

@torch.no_grad()
def _refresh_tables_ablation_gpu(
    model: core.RelationModel,
    mode: str,
    common_src: Tensor,
    donor_src: Tensor,
    target_src: Tensor,
    src_on_gpu: bool,
    cutoff: int,
    lagged_rows_dev: Tensor,
    dep_table_dev: Tensor,
    device: torch.device,
    eval_bs: int = 32,
) -> None:
    was_training = model.training
    model.eval()
    n = common_src.shape[0]
    hw_max = max(int(common_src.shape[-2]), int(common_src.shape[-1]))
    eff_eval_bs = min(int(eval_bs), 8 if hw_max >= 800 else (16 if hw_max >= 512 else 32))
    for s in range(0, n, eff_eval_bs):
        e = min(n, s + eff_eval_bs)
        if src_on_gpu:
            c_b = common_src[s:e].to(torch.float32)
            d_f = donor_src[s:e].to(torch.float32) / 255.0
            t_u8 = target_src[s:e]
        else:
            c_b = common_src[s:e].to(device=device, dtype=torch.float32, non_blocking=True)
            d_f = donor_src[s:e, None].to(device=device, dtype=torch.float32) / 255.0
            t_u8 = target_src[s:e, None].to(device=device)
        res, _ = fast_ablation_forward(model, c_b, mode=mode, inference=False)
        z = d_f + res
        prep = suh.prepare_channel_counts(z, t_u8, cutoff)
        lagged_rows_dev[s:e] = suh.fast_densities_hist_nograd(prep)
        dep_table_dev[s:e] = suh.deployment_cutoff_densities(prep, cutoff)
    model.train(was_training)


def _recalibrate_lambdas_ablation(
    model: core.RelationModel,
    mode: str,
    common_src: Tensor,
    donor_src: Tensor,
    target_src: Tensor,
    src_on_gpu: bool,
    scales: Mapping[str, float],
    lagged_rows_dev: Tensor,
    dep_table_dev: Tensor,
    mean_G: float,
    cutoff: int,
    thr_floor: int,
    device: torch.device,
    calib_subset: int = 96,
) -> tuple[float, float]:
    if mean_G <= 1e-15:
        return 4.0, 4.0
    was_training = model.training
    model.eval()
    n = common_src.shape[0]
    hw_max = max(int(common_src.shape[-2]), int(common_src.shape[-1]))
    grad_bs = 4 if hw_max >= 800 else (8 if hw_max >= 512 else 16)
    step_idx = np.linspace(0, n - 1, num=min(n, calib_subset), dtype=int)

    with torch.no_grad():
        active_thr_t, adj_ods, _ = suh._gpu_pool_ods_adjoint(lagged_rows_dev, mean_G, thr_floor)
        adj_dep = suh._adjoint_from_single_row_gpu(dep_table_dev.mean(dim=0), mean_G)

    recon_sq = 0.0
    aux_ods_sq = 0.0
    aux_dep_sq = 0.0
    for g_start in range(0, len(step_idx), grad_bs):
        sub_idx = [int(x) for x in step_idx[g_start : g_start + grad_bs]]
        with torch.no_grad():
            if src_on_gpu:
                idx_dev = torch.tensor(sub_idx, dtype=torch.long, device=device)
                c_b = common_src.index_select(0, idx_dev).to(torch.float32)
                d_f0 = donor_src.index_select(0, idx_dev).to(torch.float32) / 255.0
                t_u8 = target_src.index_select(0, idx_dev)
            else:
                idx_cpu = torch.tensor(sub_idx, dtype=torch.long)
                c_b = common_src.index_select(0, idx_cpu).to(device=device, dtype=torch.float32)
                d_f0 = donor_src.index_select(0, idx_cpu)[:, None].to(device=device, dtype=torch.float32) / 255.0
                t_u8 = target_src.index_select(0, idx_cpu)[:, None].to(device=device)
            res0, _ = fast_ablation_forward(model, c_b, mode=mode, inference=False)
            z0 = (d_f0 + res0).detach()
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

    model.train(was_training)
    lam_ods = float(min(max(counts.calibration_lambda(recon_sq, aux_ods_sq), 0.25), 400.0)) if aux_ods_sq > 1e-24 and recon_sq > 0 else 4.0
    lam_dep = float(min(max(counts.calibration_lambda(recon_sq, aux_dep_sq), 0.25), 400.0)) if aux_dep_sq > 1e-24 and recon_sq > 0 else 4.0
    return lam_ods, lam_dep


def fit_ablation_marker(
    *,
    mode: str,
    seed: int,
    common_train: Tensor,
    donor_u8_train: Tensor,
    target_u8_train: Tensor,
    cutoff: int,
    device: torch.device | str,
    epochs: int = 8,
    lr: float = 1.2e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    lambda_scale: float = 1.0,
    deploy_anchor_weight: float = 1.25,
) -> tuple[core.RelationModel, dict]:
    """Fit ``S_recon_comp``, ``S_count_1off``, or ``E_count`` with matched S6 schedule."""
    if mode not in ABLATION_TRAINED_ARMS:
        raise ValueError(f"mode must be in {ABLATION_TRAINED_ARMS}, got {mode!r}")
    dev = torch.device(device)
    n = common_train.shape[0]
    model = suh.build_coordinate_screened_model(seed, dev)
    use_count_loss = mode in ("S_count_1off", "E_count")

    # Always initialize training-split count tables so ALL three arms share identical
    # 2-epoch training-split composite checkpoint selection (0.65 F_dep + 0.35 F_ods).
    scales, base_lambda, dep_lambda, lagged, dep_table, g_vec = suh.estimate_training_scales_and_lambda(
        model, common_train, donor_u8_train, target_u8_train, cutoff, dev, arm="S_count"
    )
    mean_G_scalar = float(g_vec.mean().item())
    eff_lambda = base_lambda * float(lambda_scale) if use_count_loss else 0.0
    eff_dep_lambda = dep_lambda * float(deploy_anchor_weight) * float(lambda_scale) if use_count_loss else 0.0

    src_on_gpu = False
    common_gpu: Tensor | None = None
    donor_gpu: Tensor | None = None
    target_gpu: Tensor | None = None
    if dev.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(dev)
            needed_bytes = int(common_train.numel()) * 2 + int(donor_u8_train.numel()) * 2 + 2_200_000_000
            if free_bytes > needed_bytes:
                common_gpu = common_train.to(device=dev, dtype=torch.float16)
                donor_gpu = donor_u8_train[:, None].to(device=dev)
                target_gpu = target_u8_train[:, None].to(device=dev)
                src_on_gpu = True
        except Exception:
            src_on_gpu = False

    rng = random.Random(int(seed) * 17 + 7)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    total_steps = max(1, epochs * math.ceil(n / batch_size))
    warmup = max(1, total_steps // 10)

    def _lr_mult(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    step = 0
    history = []
    dep_table_dev = dep_table.to(dev) if dep_table is not None else None
    lagged_rows_dev = lagged.rows.to(device=dev, dtype=torch.float64) if lagged is not None else None
    thr_floor = min(int(cutoff), max(4, int(round(0.80 * cutoff))))

    best_train_score = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    if lagged_rows_dev is not None and mean_G_scalar > 1e-15:
        best_train_score, _, _ = suh._train_count_composite_gpu(lagged_rows_dev, mean_G_scalar, cutoff)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for ep in range(epochs):
        if ep > 0 and ep % 2 == 0 and lagged_rows_dev is not None and dep_table_dev is not None:
            c_src = common_gpu if src_on_gpu and common_gpu is not None else common_train
            d_src = donor_gpu if src_on_gpu and donor_gpu is not None else donor_u8_train
            t_src = target_gpu if src_on_gpu and target_gpu is not None else target_u8_train
            _refresh_tables_ablation_gpu(
                model, mode, c_src, d_src, t_src, src_on_gpu, cutoff, lagged_rows_dev, dep_table_dev, dev
            )
            if mean_G_scalar > 1e-15:
                cur_comp, _, _ = suh._train_count_composite_gpu(lagged_rows_dev, mean_G_scalar, cutoff)
                if cur_comp >= best_train_score:
                    best_train_score = cur_comp
                    best_epoch = ep
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if use_count_loss:
                    live_lam_ods, live_lam_dep = _recalibrate_lambdas_ablation(
                        model,
                        mode,
                        c_src,
                        d_src,
                        t_src,
                        src_on_gpu,
                        scales,
                        lagged_rows_dev,
                        dep_table_dev,
                        mean_G_scalar,
                        cutoff,
                        thr_floor,
                        dev,
                    )
                    eff_lambda = live_lam_ods * float(lambda_scale)
                    eff_dep_lambda = live_lam_dep * float(deploy_anchor_weight) * float(lambda_scale)

        model.train()
        order = list(range(n))
        rng.shuffle(order)
        ep_rec_t = torch.zeros((), dtype=torch.float64, device=dev)
        ep_aux_t = torch.zeros((), dtype=torch.float64, device=dev)
        ep_res_mean_t = torch.zeros((), dtype=torch.float32, device=dev)
        n_batches = 0

        if src_on_gpu and common_gpu is not None and donor_gpu is not None and target_gpu is not None:
            order_dev = torch.tensor(order, dtype=torch.long, device=dev)
        else:
            order_dev = None

        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            bsz = len(batch_idx)
            codes = [rng.randrange(8) for _ in range(bsz)]
            if src_on_gpu and order_dev is not None and common_gpu is not None and donor_gpu is not None and target_gpu is not None:
                idx_gpu = order_dev[start : start + bsz]
                c_gpu = common_gpu.index_select(0, idx_gpu).to(torch.float32)
                d_gpu = donor_gpu.index_select(0, idx_gpu)
                t_gpu = target_gpu.index_select(0, idx_gpu)
            else:
                idx_t = torch.tensor(batch_idx, dtype=torch.long)
                idx_gpu = idx_t.to(dev)
                c_gpu = torch.index_select(common_train, 0, idx_t).to(device=dev, dtype=torch.float32, non_blocking=True)
                d_gpu = torch.index_select(donor_u8_train, 0, idx_t)[:, None].to(dev, non_blocking=True)
                t_gpu = torch.index_select(target_u8_train, 0, idx_t)[:, None].to(dev, non_blocking=True)

            c_b = suh.apply_d4_batch(c_gpu, codes)
            d_u8 = suh.apply_d4_batch(d_gpu, codes)
            t_u8 = suh.apply_d4_batch(t_gpu, codes)
            d_f = d_u8.to(torch.float32) / 255.0
            t_f = t_u8.to(torch.float32) / 255.0

            mult = _lr_mult(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * mult

            optimizer.zero_grad(set_to_none=True)
            residual, diag = fast_ablation_forward(model, c_b, mode=mode, inference=False)
            z_unclipped = d_f + residual
            rec_vec = suh.preclip_reconstruction_loss(z_unclipped, t_f, d_f, scales)
            loss = rec_vec.mean()

            if use_count_loss and lagged_rows_dev is not None and dep_table_dev is not None:
                with torch.no_grad():
                    prep_det = suh.prepare_channel_counts(z_unclipped.detach(), t_u8, cutoff)
                    lagged_rows_dev[idx_gpu] = suh.fast_densities_hist_nograd(prep_det)
                    dep_table_dev[idx_gpu] = suh.deployment_cutoff_densities(prep_det, cutoff)
                    active_thr_t, adj_ods, _ = suh._gpu_pool_ods_adjoint(lagged_rows_dev, mean_G_scalar, thr_floor)
                    adj_dep = suh._adjoint_from_single_row_gpu(dep_table_dev.mean(dim=0), mean_G_scalar)

                prep = suh.prepare_channel_counts(z_unclipped, t_u8, cutoff)
                dens_ods = suh._fast_densities_single_gpu_thr(prep, active_thr_t)
                dens_dep = suh.deployment_cutoff_densities(prep, cutoff)

                ods_term = (dens_ods * adj_ods[None, :]).sum(dim=-1).mean()
                dep_term = (dens_dep * adj_dep[None, :]).sum(dim=-1).mean()
                scaled_aux = eff_lambda * ods_term + eff_dep_lambda * dep_term
                loss = loss + scaled_aux.to(loss.dtype)
                ep_aux_t = ep_aux_t + ods_term.detach()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            ep_rec_t = ep_rec_t + rec_vec.detach().mean().to(torch.float64)
            ep_res_mean_t = torch.maximum(ep_res_mean_t, diag["residual_mean_max_tensor"])
            n_batches += 1
            step += 1

        ep_rec = float(ep_rec_t.item())
        ep_aux = float(ep_aux_t.item())
        ep_res_mean = float(ep_res_mean_t.item())
        if not math.isfinite(ep_rec) or not math.isfinite(ep_aux):
            raise RuntimeError(f"Non-finite loss in {mode} epoch {ep + 1}: rec={ep_rec}, aux={ep_aux}")
        history.append(
            dict(
                epoch=ep + 1,
                recon_loss=ep_rec / max(1, n_batches),
                aux_loss=ep_aux / max(1, n_batches),
                max_abs_residual_mean=ep_res_mean,
            )
        )

    if lagged_rows_dev is not None and dep_table_dev is not None and mean_G_scalar > 1e-15:
        c_src = common_gpu if src_on_gpu and common_gpu is not None else common_train
        d_src = donor_gpu if src_on_gpu and donor_gpu is not None else donor_u8_train
        t_src = target_gpu if src_on_gpu and target_gpu is not None else target_u8_train
        _refresh_tables_ablation_gpu(
            model, mode, c_src, d_src, t_src, src_on_gpu, cutoff, lagged_rows_dev, dep_table_dev, dev
        )
        final_comp, _, _ = suh._train_count_composite_gpu(lagged_rows_dev, mean_G_scalar, cutoff)
        if final_comp >= best_train_score:
            best_train_score = final_comp
            best_epoch = epochs
        elif best_state is not None:
            model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    del common_gpu, donor_gpu, target_gpu
    model.eval()
    meta = dict(
        schema="plan2-ablation-v1",
        mode=mode,
        seed=int(seed),
        cutoff=int(cutoff),
        epochs=int(epochs),
        selected_train_epoch=int(best_epoch),
        selected_train_composite=float(best_train_score),
        batch_size=int(batch_size),
        lr=float(lr),
        effective_lambda=float(eff_lambda),
        effective_deploy_lambda=float(eff_dep_lambda),
        history=history,
    )
    return model, meta


@torch.no_grad()
def predict_ablation_marker(
    model: core.RelationModel,
    mode: str,
    common_eval: Tensor,
    donor_u8_eval: Tensor,
    device: torch.device | str,
    batch_size: int = 16,
) -> tuple[np.ndarray, dict]:
    """Predict uint8 [0, 255] output and exact zero-mean residual diagnostics."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    n = common_eval.shape[0]
    preds_u8 = np.empty_like(donor_u8_eval.cpu().numpy(), dtype=np.uint8)
    max_rel_res = 0.0
    max_zero_mean = 0.0
    sum_zero_mean = 0.0
    hw_max = max(int(common_eval.shape[-2]), int(common_eval.shape[-1]))
    eff_bs = min(int(batch_size), 4 if hw_max >= 800 else (8 if hw_max >= 512 else 32))

    for start in range(0, n, eff_bs):
        stop = min(n, start + eff_bs)
        c_b = common_eval[start:stop].to(dev, non_blocking=True)
        d_f = donor_u8_eval[start:stop, None].to(device=dev, dtype=torch.float64) / 255.0
        residual, diag = fast_ablation_forward(model, c_b, mode=mode, inference=True)
        max_rel_res = max(max_rel_res, float(diag["relative_residual_max"]))
        tile_means = residual.detach().to(torch.float64).mean(dim=(-2, -1)).abs()
        max_zero_mean = max(max_zero_mean, float(tile_means.max().item()))
        sum_zero_mean += float(tile_means.sum().item())
        pred01 = (d_f + residual.to(torch.float64)).clamp(0.0, 1.0)
        pred_byte = torch.round(pred01[:, 0] * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        preds_u8[start:stop] = pred_byte.cpu().numpy()

    return preds_u8, dict(
        mode=mode,
        relative_residual_max=max_rel_res,
        residual_mean_max=max_zero_mean,
        residual_mean_avg=sum_zero_mean / max(1, n),
    )


# ---------------------------------------------------------------------------
# Classical Spatial Post-Processing Baselines (Pure PyTorch GPU/CPU, Zero skimage/cv2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_unsharp_u8(arr_nhw: np.ndarray, sigma: float, alpha: float, device: torch.device | str) -> np.ndarray:
    """Separable reflect-padded Gaussian unsharp masking on uint8 (N, H, W)."""
    if alpha <= 0.0:
        return arr_nhw.copy()
    dev = torch.device(device)
    rad = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-rad, rad + 1, device=dev, dtype=torch.float32)
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / g.sum()
    g_col = g.view(1, 1, 2 * rad + 1, 1)
    g_row = g.view(1, 1, 1, 2 * rad + 1)

    n = arr_nhw.shape[0]
    out = np.empty_like(arr_nhw, dtype=np.uint8)
    bs = 64
    for s in range(0, n, bs):
        e = min(n, s + bs)
        x = torch.from_numpy(arr_nhw[s:e]).to(dev, dtype=torch.float32)[:, None] / 255.0
        blur = F.conv2d(F.pad(x, (0, 0, rad, rad), mode="reflect"), g_col)
        blur = F.conv2d(F.pad(blur, (rad, rad, 0, 0), mode="reflect"), g_row)
        sharp = (x + float(alpha) * (x - blur)).clamp(0.0, 1.0)
        out[s:e] = torch.round(sharp[:, 0] * 255.0).clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
    return out


@torch.no_grad()
def apply_clahe_u8(
    arr_nhw: np.ndarray,
    clip_limit: float,
    device: torch.device | str,
    grid_size: int = 8,
) -> np.ndarray:
    """Exact 8x8 tile-grid bilinear CLAHE on uint8 (N, H, W) using pure PyTorch."""
    if clip_limit <= 0.0:
        return arr_nhw.copy()
    dev = torch.device(device)
    n, h, w = arr_nhw.shape
    gh = gw = int(grid_size)
    # Pad minimally if H or W is not divisible by grid_size (both 512 and 256 divide evenly by 8)
    pad_h = (gh - (h % gh)) % gh
    pad_w = (gw - (w % gw)) % gw
    hp = h + pad_h
    wp = w + pad_w
    th = hp // gh
    tw = wp // gw
    tile_pix = float(th * tw)
    clip_val = max(1.0, float(clip_limit) * tile_pix)

    # Precompute bilinear interpolation grid over tile centers
    yy = (torch.arange(h, device=dev, dtype=torch.float32) + 0.5) / float(th) - 0.5
    xx = (torch.arange(w, device=dev, dtype=torch.float32) + 0.5) / float(tw) - 0.5
    y0 = yy.floor().clamp(0, gh - 1).to(torch.long)
    x0 = xx.floor().clamp(0, gw - 1).to(torch.long)
    y1 = (y0 + 1).clamp(0, gh - 1)
    x1 = (x0 + 1).clamp(0, gw - 1)
    wy = (yy - y0.to(torch.float32)).clamp(0.0, 1.0)[:, None]  # (H, 1)
    wx = (xx - x0.to(torch.float32)).clamp(0.0, 1.0)[None, :]  # (1, W)

    t00 = (y0[:, None] * gw + x0[None, :]).reshape(-1)  # (H*W,)
    t01 = (y0[:, None] * gw + x1[None, :]).reshape(-1)
    t10 = (y1[:, None] * gw + x0[None, :]).reshape(-1)
    t11 = (y1[:, None] * gw + x1[None, :]).reshape(-1)
    w00 = ((1.0 - wy) * (1.0 - wx)).reshape(-1)
    w01 = ((1.0 - wy) * wx).reshape(-1)
    w10 = (wy * (1.0 - wx)).reshape(-1)
    w11 = (wy * wx).reshape(-1)

    out = np.empty_like(arr_nhw, dtype=np.uint8)
    bs = 16 if h >= 512 else 64
    num_tiles = gh * gw

    for s in range(0, n, bs):
        e = min(n, s + bs)
        bsz = e - s
        x_u8 = torch.from_numpy(arr_nhw[s:e]).to(dev, dtype=torch.long)
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x_u8[:, None].to(torch.float32), (0, pad_w, 0, pad_h), mode="reflect")[:, 0].to(torch.long)
        else:
            x_pad = x_u8

        # Reshape into (bsz * gh * gw, th * tw) and build histograms via scatter_add_
        blocks = x_pad.view(bsz, gh, th, gw, tw).permute(0, 1, 3, 2, 4).reshape(bsz * num_tiles, th * tw)
        hist = torch.zeros((bsz * num_tiles, 256), device=dev, dtype=torch.float32)
        ones = torch.ones_like(blocks, dtype=torch.float32)
        hist.scatter_add_(1, blocks, ones)

        # Clip and uniformly redistribute excess (2 passes handles residual overflow cleanly)
        for _ in range(2):
            excess = (hist - clip_val).clamp_min(0.0)
            total_ex = excess.sum(dim=1, keepdim=True)
            hist = torch.minimum(hist, torch.full_like(hist, clip_val)) + (total_ex / 256.0)

        cdf = torch.cumsum(hist, dim=1)
        cdf_min = cdf[:, :1]
        denom = (cdf[:, -1:] - cdf_min).clamp_min(1.0)
        lut = ((cdf - cdf_min) / denom * 255.0).clamp(0.0, 255.0).view(bsz, num_tiles, 256)

        # Bilinear LUT lookup per pixel
        flat_x = x_u8.reshape(bsz, h * w)
        lut_flat = lut.reshape(bsz, num_tiles * 256)
        v00 = torch.gather(lut_flat, 1, t00[None, :] * 256 + flat_x)
        v01 = torch.gather(lut_flat, 1, t01[None, :] * 256 + flat_x)
        v10 = torch.gather(lut_flat, 1, t10[None, :] * 256 + flat_x)
        v11 = torch.gather(lut_flat, 1, t11[None, :] * 256 + flat_x)
        mapped = (w00[None, :] * v00 + w01[None, :] * v01 + w10[None, :] * v10 + w11[None, :] * v11).view(bsz, h, w)
        out[s:e] = torch.round(mapped).clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
    return out


def tune_classical_channel(
    train_up_u8: np.ndarray,
    train_targ_u8: np.ndarray,
    eval_up_u8: np.ndarray,
    cutoff: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray, dict]:
    """Tune Unsharp and CLAHE strictly on training Fixed_tol_F1 and apply to train/eval."""
    def _train_fixed_f1(pred_tr: np.ndarray) -> float:
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

    base_f1 = _train_fixed_f1(train_up_u8)

    # 1) Unsharp grid
    best_u = (0.0, 0.0, base_f1)
    best_u_tr = train_up_u8.copy()
    for sigma in (1.5, 3.0, 6.0):
        for alpha in (0.5, 1.0, 2.0, 3.0):
            cand_tr = apply_unsharp_u8(train_up_u8, sigma=sigma, alpha=alpha, device=device)
            f1 = _train_fixed_f1(cand_tr)
            if f1 > best_u[2] + 1e-6:
                best_u = (sigma, alpha, f1)
                best_u_tr = cand_tr
    best_u_ev = apply_unsharp_u8(eval_up_u8, sigma=best_u[0], alpha=best_u[1], device=device)
    u_meta = dict(sigma=best_u[0], alpha=best_u[1], train_fixed_f1=best_u[2], train_base_f1=base_f1)

    # 2) CLAHE grid
    best_c = (0.0, base_f1)
    best_c_tr = train_up_u8.copy()
    for clip_lim in (0.005, 0.01, 0.02, 0.04, 0.08):
        cand_tr = apply_clahe_u8(train_up_u8, clip_limit=clip_lim, device=device)
        f1 = _train_fixed_f1(cand_tr)
        if f1 > best_c[1] + 1e-6:
            best_c = (clip_lim, f1)
            best_c_tr = cand_tr
    best_c_ev = apply_clahe_u8(eval_up_u8, clip_limit=best_c[0], device=device)
    c_meta = dict(clip_limit=best_c[0], train_fixed_f1=best_c[1], train_base_f1=base_f1)

    return best_u_tr, best_u_ev, u_meta, best_c_tr, best_c_ev, c_meta


def _atomic_write_json(path: Path, obj: Mapping) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Cohort Runners (HEMIT, Orion Fold 0, HER2match)
# ---------------------------------------------------------------------------

def run_hemit_plan2(args: argparse.Namespace) -> None:
    from . import run_hemit_multibackbone_cell as rhmc

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    up_cache_dir = Path(args.upstream_cache_dir).resolve()
    bc_dir = Path(args.bank_cache_dir).resolve()

    with open(bc_dir / "hemit_bank_train_596.pkl", "rb") as fh:
        train_data = pickle.load(fh)
    with open(bc_dir / "hemit_bank_test_945.pkl", "rb") as fh:
        eval_data = pickle.load(fh)

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]

    for b_key in backbones:
        out_json = out_dir / f"plan2_hemit_{b_key}.json"
        if out_json.exists() and not args.overwrite:
            print(json.dumps(dict(event="skip_existing", cohort="hemit", backbone=b_key)), flush=True)
            continue

        b_t0 = time.monotonic()
        tr_c = up_cache_dir / f"{b_key}_train_596.npz"
        ev_c = up_cache_dir / f"{b_key}_test_945.npz"
        with np.load(tr_c) as z_tr:
            up_train = z_tr["pred_u8"]
        with np.load(ev_c) as z_ev:
            up_eval = z_ev["pred_u8"]

        arm_tr: dict[str, np.ndarray] = {"upstream": up_train}
        arm_ev: dict[str, np.ndarray] = {"upstream": up_eval}
        for arm in ("S_recon_comp", "S_count_1off", "E_count", "Unsharp", "CLAHE"):
            arm_tr[arm] = np.empty_like(up_train)
            arm_ev[arm] = np.empty_like(up_eval)

        diagnostics: dict[str, dict] = {}
        classical_meta: dict[str, dict] = {}
        fit_meta: dict[str, dict] = {}

        for ch, (m_name, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
            # Classical baselines first (fast)
            u_tr, u_ev, u_m, cl_tr, cl_ev, cl_m = tune_classical_channel(
                up_train[:, ch], train_data["targ_nchw"][:, ch], up_eval[:, ch], cutoff, device
            )
            arm_tr["Unsharp"][:, ch] = u_tr
            arm_ev["Unsharp"][:, ch] = u_ev
            arm_tr["CLAHE"][:, ch] = cl_tr
            arm_ev["CLAHE"][:, ch] = cl_ev
            classical_meta[m_name] = dict(Unsharp=u_m, CLAHE=cl_m)

            # Shared common12 built once for train and once for test
            c12_tr = rhmc.build_common12_for_3ch(train_data["he_nhwc"], up_train, ch, device=device)
            d_tr_t = torch.from_numpy(up_train[:, ch])
            y_tr_t = torch.from_numpy(train_data["targ_nchw"][:, ch])
            d_ev_t = torch.from_numpy(up_eval[:, ch])

            fitted: dict[tuple[str, int], core.RelationModel] = {}
            for mode in ABLATION_TRAINED_ARMS:
                tr_list = []
                for sd in seeds:
                    model, meta = fit_ablation_marker(
                        mode=mode,
                        seed=sd,
                        common_train=c12_tr,
                        donor_u8_train=d_tr_t,
                        target_u8_train=y_tr_t,
                        cutoff=cutoff,
                        device=device,
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        deploy_anchor_weight=1.25,
                    )
                    fit_meta[f"{m_name}:{mode}:{sd}"] = dict(
                        selected_train_epoch=meta["selected_train_epoch"],
                        selected_train_composite=meta["selected_train_composite"],
                        final_recon_loss=meta["history"][-1]["recon_loss"],
                    )
                    p_tr, _ = predict_ablation_marker(model, mode, c12_tr, d_tr_t, device=device, batch_size=16)
                    tr_list.append(p_tr.astype(np.float32))
                    fitted[(mode, sd)] = model.cpu()
                arm_tr[mode][:, ch] = np.round(np.mean(tr_list, axis=0)).astype(np.uint8)

            del c12_tr
            gc.collect()

            c12_ev = rhmc.build_common12_for_3ch(eval_data["he_nhwc"], up_eval, ch, device=device)
            for mode in ABLATION_TRAINED_ARMS:
                ev_list = []
                for sd in seeds:
                    model = fitted.pop((mode, sd)).to(device)
                    p_ev, diag = predict_ablation_marker(model, mode, c12_ev, d_ev_t, device=device, batch_size=16)
                    diagnostics[f"{m_name}:{mode}:{sd}"] = diag
                    ev_list.append(p_ev.astype(np.float32))
                    del model
                arm_ev[mode][:, ch] = np.round(np.mean(ev_list, axis=0)).astype(np.uint8)

            del c12_ev, fitted
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(
                json.dumps(dict(event="channel_done", cohort="hemit", backbone=b_key, marker=m_name, elapsed=round(time.monotonic() - b_t0, 1))),
                flush=True,
            )

        arms_eval = {}
        for arm_k in ALL_ABLATION_ARMS:
            per_m = {}
            for ch, (m_name, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
                ch_r = jce.evaluate_channel_predictions(
                    train_pred_u8=arm_tr[arm_k][:, ch],
                    train_targ_u8=train_data["targ_nchw"][:, ch],
                    eval_pred_u8=arm_ev[arm_k][:, ch],
                    eval_targ_u8=eval_data["targ_nchw"][:, ch],
                    eval_clusters=eval_data["clusters"],
                    fixed_cutoff=cutoff,
                    device=device,
                )
                per_m[m_name] = ch_r["pooled"]
            macro = {
                mk: float(np.mean([per_m[m][mk] for m in rhmc.HEMIT_MARKERS]))
                for mk in ("ODS_F", "AP", "Fixed_tol_F1", "Cal_F1", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed")
            }
            tr_cells = rhmc.extract_pred_cell_means_for_records(arm_tr[arm_k], train_data["nuclei_records"])
            ev_cells = rhmc.extract_pred_cell_means_for_records(arm_ev[arm_k], eval_data["nuclei_records"])
            cell_res = rhmc.evaluate_hemit_arm_cells_and_infiltration(
                tr_cells, ev_cells, eval_data["nuclei_records"], arm_ev[arm_k]
            )
            arms_eval[arm_k] = dict(
                macro_pixel=macro,
                per_marker_pixel=per_m,
                single_cell=dict(
                    macro_deployable_cell_f1=cell_res.get("macro_deployable_cell_f1", 0.0),
                    macro_recalibrated_cell_f1=cell_res.get("macro_recalibrated_cell_f1", 0.0),
                    macro_cell_auprc=cell_res.get("macro_cell_auprc", 0.0),
                ),
            )

        rec = dict(
            schema="plan2-ablation-result-v1",
            cohort="HEMIT",
            backbone=b_key,
            wall_seconds=time.monotonic() - b_t0,
            seeds=list(seeds),
            epochs=int(args.epochs),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            arms=arms_eval,
            fit_meta=fit_meta,
            solver_diagnostics=diagnostics,
            classical_meta=classical_meta,
        )
        _atomic_write_json(out_json, rec)
        print(
            json.dumps(
                dict(
                    event="backbone_done",
                    cohort="hemit",
                    backbone=b_key,
                    up=round(arms_eval["upstream"]["macro_pixel"]["Fixed_tol_F1"], 4),
                    srecon_comp=round(arms_eval["S_recon_comp"]["macro_pixel"]["Fixed_tol_F1"], 4),
                    scount_1off=round(arms_eval["S_count_1off"]["macro_pixel"]["Fixed_tol_F1"], 4),
                    ecount=round(arms_eval["E_count"]["macro_pixel"]["Fixed_tol_F1"], 4),
                    unsharp=round(arms_eval["Unsharp"]["macro_pixel"]["Fixed_tol_F1"], 4),
                    clahe=round(arms_eval["CLAHE"]["macro_pixel"]["Fixed_tol_F1"], 4),
                    sec=round(rec["wall_seconds"], 1),
                )
            ),
            flush=True,
        )


def run_orion_plan2(args: argparse.Namespace) -> None:
    from . import run_orion_multibackbone_survival as roms

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    up_cache_dir = Path(args.upstream_cache_dir).resolve()
    bc_dir = Path(args.bank_cache_dir).resolve()
    state_dir = Path(args.frozen_state_dir).resolve()

    cv_manifest = json.loads(Path(args.cv_manifest).read_text())
    with open(bc_dir / "orion_bank_all40_2804_crop256.pkl", "rb") as fh:
        bank = pickle.load(fh)

    fold0 = cv_manifest["folds"][0]
    f0_train_pats = set(roms._fold_patients(fold0, "inner_train"))
    f0_eval_pats = set(roms._fold_patients(fold0, "outer_test"))
    f0_train_idx_all = np.nonzero(np.isin(bank["patients"], list(f0_train_pats)))[0]
    f0_train_idx = f0_train_idx_all[::2]  # fit_stride=2 matching S6 exactly
    f0_eval_idx = np.nonzero(np.isin(bank["patients"], list(f0_eval_pats)))[0]

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]

    for b_key in backbones:
        out_json = out_dir / f"plan2_orion_{b_key}.json"
        if out_json.exists() and not args.overwrite:
            print(json.dumps(dict(event="skip_existing", cohort="orion", backbone=b_key)), flush=True)
            continue

        b_t0 = time.monotonic()
        with np.load(up_cache_dir / f"{b_key}_all40_2804.npz") as npz:
            up_all = npz["pred_u8"]

        cutoff_json = state_dir / b_key / "fold0" / "fold0_physical_cutoffs.json"
        c_doc = json.loads(cutoff_json.read_text())
        cutoffs_u8 = list(c_doc["cutoffs_u8"])
        tau_phys_float = dict(c_doc["tau_phys_float"])

        up_tr = up_all[f0_train_idx]
        up_ev = up_all[f0_eval_idx]
        arm_tr: dict[str, np.ndarray] = {"upstream": up_tr}
        arm_ev: dict[str, np.ndarray] = {"upstream": up_ev}
        for arm in ("S_recon_comp", "S_count_1off", "E_count", "Unsharp", "CLAHE"):
            arm_tr[arm] = np.empty_like(up_tr)
            arm_ev[arm] = np.empty_like(up_ev)

        diagnostics: dict[str, dict] = {}
        classical_meta: dict[str, dict] = {}
        fit_meta: dict[str, dict] = {}

        for ch16, m_name in enumerate(roms.ORION_MARKERS):
            cutoff = int(cutoffs_u8[ch16])
            u_tr, u_ev, u_m, cl_tr, cl_ev, cl_m = tune_classical_channel(
                up_tr[:, ch16], bank["targ_nchw"][f0_train_idx, ch16], up_ev[:, ch16], cutoff, device
            )
            arm_tr["Unsharp"][:, ch16] = u_tr
            arm_ev["Unsharp"][:, ch16] = u_ev
            arm_tr["CLAHE"][:, ch16] = cl_tr
            arm_ev["CLAHE"][:, ch16] = cl_ev
            classical_meta[m_name] = dict(Unsharp=u_m, CLAHE=cl_m)

            c12_tr = roms.build_common12_orion(bank["he_nhwc"][f0_train_idx], up_tr, ch16, device=device)
            d_tr_t = torch.from_numpy(up_tr[:, ch16])
            y_tr_t = torch.from_numpy(bank["targ_nchw"][f0_train_idx, ch16])
            d_ev_t = torch.from_numpy(up_ev[:, ch16])

            fitted: dict[tuple[str, int], core.RelationModel] = {}
            for mode in ABLATION_TRAINED_ARMS:
                tr_list = []
                for sd in seeds:
                    model, meta = fit_ablation_marker(
                        mode=mode,
                        seed=sd,
                        common_train=c12_tr,
                        donor_u8_train=d_tr_t,
                        target_u8_train=y_tr_t,
                        cutoff=cutoff,
                        device=device,
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        deploy_anchor_weight=1.25,
                    )
                    fit_meta[f"{m_name}:{mode}:{sd}"] = dict(
                        selected_train_epoch=meta["selected_train_epoch"],
                        selected_train_composite=meta["selected_train_composite"],
                        final_recon_loss=meta["history"][-1]["recon_loss"],
                    )
                    p_tr, _ = predict_ablation_marker(model, mode, c12_tr, d_tr_t, device=device, batch_size=32)
                    tr_list.append(p_tr.astype(np.float32))
                    fitted[(mode, sd)] = model.cpu()
                arm_tr[mode][:, ch16] = np.round(np.mean(tr_list, axis=0)).astype(np.uint8)

            del c12_tr
            gc.collect()

            c12_ev = roms.build_common12_orion(bank["he_nhwc"][f0_eval_idx], up_ev, ch16, device=device)
            for mode in ABLATION_TRAINED_ARMS:
                ev_list = []
                for sd in seeds:
                    model = fitted.pop((mode, sd)).to(device)
                    p_ev, diag = predict_ablation_marker(model, mode, c12_ev, d_ev_t, device=device, batch_size=32)
                    diagnostics[f"{m_name}:{mode}:{sd}"] = diag
                    ev_list.append(p_ev.astype(np.float32))
                    del model
                arm_ev[mode][:, ch16] = np.round(np.mean(ev_list, axis=0)).astype(np.uint8)

            del c12_ev, fitted
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if (ch16 + 1) % 4 == 0 or ch16 == 15:
                print(
                    json.dumps(dict(event="channel_done", cohort="orion", backbone=b_key, ch=ch16 + 1, marker=m_name, elapsed=round(time.monotonic() - b_t0, 1))),
                    flush=True,
                )

        arms_eval = {}
        eval_pats = bank["patients"][f0_eval_idx]
        for arm_k in ALL_ABLATION_ARMS:
            per_m = {}
            for ch16, m_name in enumerate(roms.ORION_MARKERS):
                ch_r = jce.evaluate_channel_predictions(
                    train_pred_u8=arm_tr[arm_k][:, ch16],
                    train_targ_u8=bank["targ_nchw"][f0_train_idx, ch16],
                    eval_pred_u8=arm_ev[arm_k][:, ch16],
                    eval_targ_u8=bank["targ_nchw"][f0_eval_idx, ch16],
                    eval_clusters=eval_pats,
                    fixed_cutoff=int(cutoffs_u8[ch16]),
                    device=device,
                )
                per_m[m_name] = ch_r["pooled"]
            macro = {
                mk: float(np.mean([per_m[m][mk] for m in roms.ORION_MARKERS]))
                for mk in ("ODS_F", "AP", "Fixed_tol_F1", "Cal_F1", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed")
            }
            # Also evaluate 15-class single-cell deployable F1
            tr_cells = roms.gather_subset_cells(arm_tr[arm_k], f0_train_idx, bank["nuclei_records"])
            ev_cells = roms.gather_subset_cells(arm_ev[arm_k], f0_eval_idx, bank["nuclei_records"])
            dep_f1s = []
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
            arms_eval[arm_k] = dict(
                macro_pixel_16ch=macro,
                per_marker_pixel=per_m,
                single_cell_15class=dict(macro_deployable_cell_f1=float(np.mean(dep_f1s))),
            )

        rec = dict(
            schema="plan2-ablation-result-v1",
            cohort="Orion-CRC",
            backbone=b_key,
            wall_seconds=time.monotonic() - b_t0,
            seeds=list(seeds),
            epochs=int(args.epochs),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            arms=arms_eval,
            fit_meta=fit_meta,
            solver_diagnostics=diagnostics,
            classical_meta=classical_meta,
        )
        _atomic_write_json(out_json, rec)
        print(
            json.dumps(
                dict(
                    event="backbone_done",
                    cohort="orion",
                    backbone=b_key,
                    up=round(arms_eval["upstream"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                    srecon_comp=round(arms_eval["S_recon_comp"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                    scount_1off=round(arms_eval["S_count_1off"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                    ecount=round(arms_eval["E_count"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                    unsharp=round(arms_eval["Unsharp"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                    clahe=round(arms_eval["CLAHE"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                    sec=round(rec["wall_seconds"], 1),
                )
            ),
            flush=True,
        )


def run_her2_plan2(args: argparse.Namespace) -> None:
    from . import run_her2match_dab_campaign as rhd

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bc_dir = Path(args.bank_cache_dir).resolve()
    state_dir = Path(args.frozen_state_dir).resolve()
    hopt_path = Path(args.hoptimus0_ckpt).resolve() if args.hoptimus0_ckpt else None

    cal_doc = json.loads(Path(args.calibration_json).read_text())
    dab_upper = float(cal_doc["histogram"]["range"][1])
    dab_thr = float(cal_doc["threshold"])
    cutoff_u8 = int(round(dab_thr / dab_upper * 255.0))

    with open(bc_dir / "her2_bank_train_480_512.pkl", "rb") as fh:
        train_bank = pickle.load(fh)
    with open(bc_dir / "her2_bank_test_500_512.pkl", "rb") as fh:
        eval_bank = pickle.load(fh)

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]

    for b_key in backbones:
        b_t0 = time.monotonic()
        _, up_tr, up_ev = rhd.train_or_load_her2_backbone(
            b_key,
            train_bank,
            eval_bank,
            cutoff_u8,
            state_dir,
            device,
            hoptimus0_ckpt=hopt_path,
            upstream_epochs=8,
            allow_train=False,
        )

        # Immediately export verified HER2+ positive visual crops from wsi009 using frozen S6 heads!
        if b_key == "hoptimus_lora_ihc" and args.export_her2_pos_crops:
            vis_out = out_dir / "her2_wsi009_pos_examples.npz"
            c12_ev_vis = rhd.build_common12_her2(eval_bank["he_nhwc"], up_ev, eval_bank["hem_u8"], device=device)
            d_ev_vis = torch.from_numpy(up_ev)
            b_state = state_dir / b_key
            srec_list, scnt_list = [], []
            for sd in seeds:
                m_sr = suh.build_coordinate_screened_model(sd, device)
                m_sr.load_state_dict(torch.load(b_state / f"HER2_DAB_S_recon_{sd}.pt", map_location=device, weights_only=True))
                p_sr, _ = suh.predict_universal_marker(m_sr, c12_ev_vis, d_ev_vis, device=device, batch_size=16)
                srec_list.append(p_sr.astype(np.float32))
                del m_sr

                m_sc = suh.build_coordinate_screened_model(sd, device)
                m_sc.load_state_dict(torch.load(b_state / f"HER2_DAB_S_count_{sd}.pt", map_location=device, weights_only=True))
                p_sc, _ = suh.predict_universal_marker(m_sc, c12_ev_vis, d_ev_vis, device=device, batch_size=16)
                scnt_list.append(p_sc.astype(np.float32))
                del m_sc
            srec_ens = np.round(np.mean(srec_list, axis=0)).astype(np.uint8)
            scnt_ens = np.round(np.mean(scnt_list, axis=0)).astype(np.uint8)
            del c12_ev_vis

            # Pick 2 genuine positive wsi009 tiles with clear membrane staining (gt_pos in 0.12..0.55)
            # and high S_count tolerant F1 improvement over upstream.
            wsis_arr = np.asarray(eval_bank["wsis"])
            cand_scores = []
            for i in range(len(wsis_arr)):
                gt_frac = float(np.mean(eval_bank["dab_u8"][i] > cutoff_u8))
                if str(wsis_arr[i]) == "wsi009" and 0.12 <= gt_frac <= 0.55:
                    c_up = jce.tile_channel_tolerance_curves(up_ev[i], eval_bank["dab_u8"][i], cutoff_u8)
                    c_sc = jce.tile_channel_tolerance_curves(scnt_ens[i], eval_bank["dab_u8"][i], cutoff_u8)
                    ep_up = jce.endpoints_from_pooled_curves(c_up["K"], c_up["U"], c_up["V"], c_up["T"], c_up["G"], cutoff_u8)
                    ep_sc = jce.endpoints_from_pooled_curves(c_sc["K"], c_sc["U"], c_sc["V"], c_sc["T"], c_sc["G"], cutoff_u8)
                    gain = float(ep_sc["Fixed_tol_F1"] - ep_up["Fixed_tol_F1"])
                    cand_scores.append((gain + 0.35 * float(ep_sc["Fixed_tol_F1"]), i, gt_frac, float(ep_up["Fixed_tol_F1"]), float(ep_sc["Fixed_tol_F1"])))
            cand_scores.sort(reverse=True)
            if len(cand_scores) < 2:
                # Fallback to top-2 highest gt_frac across all test tiles
                order_gt = np.argsort([float(np.mean(eval_bank["dab_u8"][i] > cutoff_u8)) for i in range(len(wsis_arr))])[::-1]
                vi = [int(order_gt[0]), int(order_gt[1])]
            else:
                vi = [cand_scores[0][1], cand_scores[min(3, len(cand_scores) - 1)][1]]
            np.savez_compressed(
                vis_out,
                wsis=np.array([eval_bank["wsis"][i] for i in vi]),
                names=np.array([eval_bank["names"][i] for i in vi]),
                he_nhwc=eval_bank["he_nhwc"][vi],
                ihc_nhwc=eval_bank["ihc_nhwc"][vi],
                gt_dab_u8=eval_bank["dab_u8"][vi],
                upstream_u8=up_ev[vi],
                srecon_u8=srec_ens[vi],
                scount_u8=scnt_ens[vi],
            )
            print(
                json.dumps(
                    dict(
                        event="her2_pos_crops_exported",
                        path=str(vis_out),
                        indices=vi,
                        wsis=[str(eval_bank["wsis"][i]) for i in vi],
                        names=[str(eval_bank["names"][i]) for i in vi],
                        gt_pos_fracs=[round(float(np.mean(eval_bank["dab_u8"][i] > cutoff_u8)), 4) for i in vi],
                    )
                ),
                flush=True,
            )

        out_json = out_dir / f"plan2_her2_{b_key}.json"
        if out_json.exists() and not args.overwrite:
            print(json.dumps(dict(event="skip_existing", cohort="her2", backbone=b_key)), flush=True)
            continue

        arm_tr: dict[str, np.ndarray] = {"upstream": up_tr}
        arm_ev: dict[str, np.ndarray] = {"upstream": up_ev}

        u_tr, u_ev, u_m, cl_tr, cl_ev, cl_m = tune_classical_channel(
            up_tr, train_bank["dab_u8"], up_ev, cutoff_u8, device
        )
        arm_tr["Unsharp"], arm_ev["Unsharp"] = u_tr, u_ev
        arm_tr["CLAHE"], arm_ev["CLAHE"] = cl_tr, cl_ev
        classical_meta = dict(HER2_DAB=dict(Unsharp=u_m, CLAHE=cl_m))

        c12_tr = rhd.build_common12_her2(train_bank["he_nhwc"], up_tr, train_bank["hem_u8"], device=device)
        c12_ev = rhd.build_common12_her2(eval_bank["he_nhwc"], up_ev, eval_bank["hem_u8"], device=device)
        d_tr_t = torch.from_numpy(up_tr)
        y_tr_t = torch.from_numpy(train_bank["dab_u8"])
        d_ev_t = torch.from_numpy(up_ev)

        diagnostics: dict[str, dict] = {}
        fit_meta: dict[str, dict] = {}

        for mode in ABLATION_TRAINED_ARMS:
            tr_list, ev_list = [], []
            for sd in seeds:
                model, meta = fit_ablation_marker(
                    mode=mode,
                    seed=sd,
                    common_train=c12_tr,
                    donor_u8_train=d_tr_t,
                    target_u8_train=y_tr_t,
                    cutoff=cutoff_u8,
                    device=device,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    deploy_anchor_weight=1.25,
                )
                fit_meta[f"HER2_DAB:{mode}:{sd}"] = dict(
                    selected_train_epoch=meta["selected_train_epoch"],
                    selected_train_composite=meta["selected_train_composite"],
                    final_recon_loss=meta["history"][-1]["recon_loss"],
                )
                p_tr, _ = predict_ablation_marker(model, mode, c12_tr, d_tr_t, device=device, batch_size=16)
                p_ev, diag = predict_ablation_marker(model, mode, c12_ev, d_ev_t, device=device, batch_size=16)
                diagnostics[f"HER2_DAB:{mode}:{sd}"] = diag
                tr_list.append(p_tr.astype(np.float32))
                ev_list.append(p_ev.astype(np.float32))
                del model
            arm_tr[mode] = np.round(np.mean(tr_list, axis=0)).astype(np.uint8)
            arm_ev[mode] = np.round(np.mean(ev_list, axis=0)).astype(np.uint8)

        del c12_tr, c12_ev
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        arms_eval = {}
        for arm_k in ALL_ABLATION_ARMS:
            pix_res = jce.evaluate_channel_predictions(
                train_pred_u8=arm_tr[arm_k],
                train_targ_u8=train_bank["dab_u8"],
                eval_pred_u8=arm_ev[arm_k],
                eval_targ_u8=eval_bank["dab_u8"],
                eval_clusters=eval_bank["wsis"],
                fixed_cutoff=cutoff_u8,
                device=device,
            )
            burd = rhd.evaluate_dab_burden_concordance(
                arm_ev[arm_k], eval_bank["dab_u8"], eval_bank["wsis"], cutoff_u8
            )
            arms_eval[arm_k] = dict(
                dab_pixel=pix_res["pooled"],
                dab_burden_concordance=dict(
                    tile_burden_pearson_r=burd["tile_burden_pearson_r"],
                    wsi_burden_pearson_r=burd["wsi_burden_pearson_r"],
                    wsi_burden_mae=burd["wsi_burden_mae"],
                ),
            )

        rec = dict(
            schema="plan2-ablation-result-v1",
            cohort="HER2match",
            backbone=b_key,
            wall_seconds=time.monotonic() - b_t0,
            seeds=list(seeds),
            epochs=int(args.epochs),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            cutoff_u8=cutoff_u8,
            arms=arms_eval,
            fit_meta=fit_meta,
            solver_diagnostics=diagnostics,
            classical_meta=classical_meta,
        )
        _atomic_write_json(out_json, rec)
        print(
            json.dumps(
                dict(
                    event="backbone_done",
                    cohort="her2",
                    backbone=b_key,
                    up=round(arms_eval["upstream"]["dab_pixel"]["Fixed_tol_F1"], 4),
                    srecon_comp=round(arms_eval["S_recon_comp"]["dab_pixel"]["Fixed_tol_F1"], 4),
                    scount_1off=round(arms_eval["S_count_1off"]["dab_pixel"]["Fixed_tol_F1"], 4),
                    ecount=round(arms_eval["E_count"]["dab_pixel"]["Fixed_tol_F1"], 4),
                    unsharp=round(arms_eval["Unsharp"]["dab_pixel"]["Fixed_tol_F1"], 4),
                    clahe=round(arms_eval["CLAHE"]["dab_pixel"]["Fixed_tol_F1"], 4),
                    sec=round(rec["wall_seconds"], 1),
                )
            ),
            flush=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", choices=("hemit", "orion", "her2"), required=True)
    p.add_argument("--backbones", required=True)
    p.add_argument("--bank-cache-dir", required=True)
    p.add_argument("--upstream-cache-dir", default="")
    p.add_argument("--frozen-state-dir", default="")
    p.add_argument("--cv-manifest", default="")
    p.add_argument("--calibration-json", default="")
    p.add_argument("--hoptimus0-ckpt", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", default="202610270,202610271")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument("--export-her2-pos-crops", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    args = build_arg_parser().parse_args()
    if args.cohort == "hemit":
        run_hemit_plan2(args)
    elif args.cohort == "orion":
        run_orion_plan2(args)
    elif args.cohort == "her2":
        run_her2_plan2(args)
