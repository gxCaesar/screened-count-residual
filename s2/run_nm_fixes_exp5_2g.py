#!/usr/bin/env python3
"""Exact-fix rerun for EXP-2G (fitted 2-seed 22-step CG timing) and EXP-5 (sign-verified identical-objective E2E U-Net vs Frozen+S_count)."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from s2 import journal_cohort_eval as jce
from s2 import query_relation_reconstruction as qrr
from s2 import run_hemit_multibackbone_cell as rhmc
from s2 import run_nm_nature_killer_suite as rnk
from s2 import screened_universal_head as suh


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _assert_loss_sign_matches_suh(device: torch.device) -> None:
    """Verify exact sign and gradient parity of the shared count objective before training."""
    z = torch.full((1, 1, 32, 32), 0.08, dtype=torch.float32, device=device, requires_grad=True)
    targ = torch.zeros((1, 1, 32, 32), dtype=torch.uint8, device=device)
    targ[:, :, 4:10, 4:10] = 60
    prep = suh.prepare_channel_counts(z, targ, cutoff=23)
    dens_dep = suh.deployment_cutoff_densities(prep, cutoff=23)
    mean_g = max(float(prep["G"].mean().item()), 1e-6)
    adj_dep = suh._adjoint_from_single_row_gpu(dens_dep.detach().mean(dim=0), mean_g)
    loss = (dens_dep * adj_dep[None, :]).sum(dim=-1).mean()
    loss.backward()
    fp_grad = float(z.grad[0, 0, 26, 26].item())
    tp_grad = float(z.grad[0, 0, 6, 6].item())
    assert float(adj_dep[0].item()) > 0.0, f"Expected positive K adjoint, got {adj_dep}"
    assert fp_grad > 0.0, f"Expected positive grad on isolated FP (pushing down), got {fp_grad}"
    assert tp_grad < 0.0, f"Expected negative grad on TP (pushing up), got {tp_grad}"
    print(f"[SIGN_CHECK PASS] adj_K={float(adj_dep[0]):+.4f}, fp_grad={fp_grad:+.4e}, tp_grad={tp_grad:+.4e}", flush=True)


def run_true_2g_fitted_timing(h_tr: dict, up_tr: np.ndarray, h_ev: dict, up_ev: np.ndarray, out_dir: Path, device: torch.device) -> None:
    print("[2G-FIX] Fitting two real S_count seeds (202610270, 202610271) on HEMIT panCK for true 22-step CG timing...", flush=True)
    c12_tr = rhmc.build_common12_for_3ch(h_tr["he_nhwc"], up_tr, 0, device=device)
    c12_ev = rhmc.build_common12_for_3ch(h_ev["he_nhwc"][:8], up_ev[:8], 0, device=device)
    models_gpu = []
    for sd in (202610270, 202610271):
        m, _ = suh.fit_universal_marker(
            arm="S_count", seed=sd, common_train=c12_tr,
            donor_u8_train=torch.from_numpy(up_tr[:, 0]),
            target_u8_train=torch.from_numpy(h_tr["targ_nchw"][:, 0]),
            cutoff=rhmc.HEMIT_CUTOFFS[0], device=device, epochs=4, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
        )
        m.eval()
        models_gpu.append(m)
    del c12_tr
    torch.cuda.empty_cache()

    prof_rows = []
    for hw in (256, 512, 1024, 2048):
        # Build real histological 12-ch input tile at target resolution (crop or tile real HEMIT crop)
        base_x = c12_ev[0:1].to(device=device, dtype=torch.float32)  # (1, 12, 512, 512)
        if hw == 256:
            x_gpu = base_x[:, :, 128:384, 128:384].contiguous()
        elif hw == 512:
            x_gpu = base_x.contiguous()
        else:
            rep = hw // 512
            x_gpu = base_x.repeat(1, 1, rep, rep).contiguous()

        # Verify BOTH fitted seeds enter the real 22-step FP64 CG solver (nonzero RHS!)
        rhs_norms = []
        rel_res_list = []
        dc_list = []
        with torch.no_grad():
            for m in models_gpu:
                res, diag = suh.fast_sc_forward(m, x_gpu, inference=True)
                # Log L2 norm of the solved residual (strictly nonzero confirms non-trivial CG solve)
                rhs_n = float(torch.linalg.vector_norm(res).item())
                rel_r = float(diag["relative_residual_max"])
                dc_v = float(res.mean(dim=(-2, -1)).abs().max().item())
                assert rhs_n > 1e-3, f"Zero residual detected at {hw}x{hw}!"
                assert 0.0 < rel_r < 1e-10, f"Unexpected CG relative residual {rel_r} (zero-RHS short-circuit?)"
                assert dc_v < 1e-14, f"Unexpected DC drift {dc_v}"
                rhs_norms.append(rhs_n)
                rel_res_list.append(rel_r)
                dc_list.append(dc_v)

        # Warmup BOTH seeds
        with torch.no_grad():
            for _ in range(3):
                for m in models_gpu:
                    suh.fast_sc_forward(m, x_gpu, inference=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        n_rep = 12 if hw <= 1024 else 5
        ts = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_rep):
                for m in models_gpu:
                    suh.fast_sc_forward(m, x_gpu, inference=True)
        torch.cuda.synchronize(device)
        gpu_ms_2seed = (time.perf_counter() - ts) * 1000.0 / n_rep
        gpu_vram_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)

        # CPU timing executing BOTH real fitted seeds
        cpu_ms_2seed = None
        if hw <= 1024:
            models_cpu = [suh.build_coordinate_screened_model(sd, torch.device("cpu")).eval() for sd in (202610270, 202610271)]
            for mc, mg in zip(models_cpu, models_gpu):
                mc.load_state_dict({k: v.cpu() for k, v in mg.state_dict().items()})
            x_cpu = x_gpu.cpu()
            with torch.no_grad():
                for mc in models_cpu:
                    suh.fast_sc_forward(mc, x_cpu, inference=True)
            n_c = 3 if hw <= 512 else 1
            tc = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_c):
                    for mc in models_cpu:
                        suh.fast_sc_forward(mc, x_cpu, inference=True)
            cpu_ms_2seed = round((time.perf_counter() - tc) * 1000.0 / n_c, 2)
            del models_cpu, x_cpu

        prof_rows.append(dict(
            resolution=f"{hw}x{hw}",
            hw=hw,
            gpu_name=torch.cuda.get_device_name(device),
            scope="2-seed per-channel fitted checkpoint inference (22 FP64 CG steps, excluding upstream generator)",
            nonzero_rhs_frac=1.0,
            min_rhs_norm=round(min(rhs_norms), 6),
            max_cg_rel_residual=float(max(rel_res_list)),
            max_dc_drift=float(max(dc_list)),
            gpu_ms_2seed=round(gpu_ms_2seed, 3),
            gpu_tiles_per_sec_2seed=round(1000.0 / gpu_ms_2seed, 2),
            gpu_wsi_10k_sec_2seed=round(10000.0 * gpu_ms_2seed / 1000.0, 1),
            gpu_peak_vram_mb=round(gpu_vram_mb, 2),
            cpu_ms_2seed=cpu_ms_2seed,
        ))
        print(f"  [2G-FIX {hw}x{hw}] GPU 2-seed={gpu_ms_2seed:.3f}ms ({gpu_vram_mb:.1f}MB), CPU 2-seed={cpu_ms_2seed}ms, cg_rel={max(rel_res_list):.2e}", flush=True)
        del x_gpu
        torch.cuda.empty_cache()

    _atomic_write_json(out_dir / "mech" / "mech_2g_throughput_scaling.json", dict(schema="mech-2g-v2-fitted-cg", profiles=prof_rows))
    del models_gpu, c12_ev
    torch.cuda.empty_cache()


def _recalibrate_unet_channel_lambdas(
    ft_net: torch.nn.Module,
    he_sub: np.ndarray,
    up_sub: np.ndarray,
    targ_sub: np.ndarray,
    ch_scales: list[dict],
    lagged_rows_list: list[torch.Tensor],
    dep_tables_list: list[torch.Tensor],
    mean_g_list: list[float],
    thr_floor_list: list[int],
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """Output-space gradient-norm balancing identical to suh._recalibrate_lambdas_live."""
    ft_net.eval()
    n_probe = min(16, len(he_sub))
    xb = torch.from_numpy(he_sub[:n_probe]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
    yb = torch.from_numpy(targ_sub[:n_probe]).to(device, dtype=torch.float32) / 255.0
    db = torch.from_numpy(up_sub[:n_probe]).to(device, dtype=torch.float32) / 255.0
    y_u8_all = torch.from_numpy(targ_sub[:n_probe]).to(device, dtype=torch.uint8)
    with torch.no_grad():
        pred0 = ft_net(xb).detach()
    lam_ods_out, lam_dep_out = [], []
    for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
        z_var = pred0[:, ch : ch + 1].detach().clone().requires_grad_(True)
        l_rec = suh.preclip_reconstruction_loss(z_var, yb[:, ch : ch + 1], db[:, ch : ch + 1], ch_scales[ch]).mean()
        g_rec = torch.autograd.grad(l_rec, z_var, retain_graph=False)[0]
        rec_norm = float(torch.linalg.vector_norm(g_rec).item())

        z_var2 = pred0[:, ch : ch + 1].detach().clone().requires_grad_(True)
        prep = suh.prepare_channel_counts(z_var2, y_u8_all[:, ch : ch + 1], cutoff)
        active_thr_t, adj_ods, _ = suh._gpu_pool_ods_adjoint(lagged_rows_list[ch], mean_g_list[ch], thr_floor_list[ch])
        adj_dep = suh._adjoint_from_single_row_gpu(dep_tables_list[ch].mean(dim=0), mean_g_list[ch])
        dens_ods = suh._fast_densities_single_gpu_thr(prep, active_thr_t)
        dens_dep = suh.deployment_cutoff_densities(prep, cutoff)
        ods_term = (dens_ods * adj_ods.to(dens_ods.dtype)[None, :]).sum(dim=-1).mean()
        dep_term = (dens_dep * adj_dep.to(dens_dep.dtype)[None, :]).sum(dim=-1).mean()
        g_ods = torch.autograd.grad(ods_term, z_var2, retain_graph=True)[0]
        g_dep = torch.autograd.grad(dep_term, z_var2, retain_graph=False)[0]
        ods_norm = max(float(torch.linalg.vector_norm(g_ods).item()), 1e-8)
        dep_norm = max(float(torch.linalg.vector_norm(g_dep).item()), 1e-8)
        lam_ods_out.append(float(np.clip(rec_norm / ods_norm, 0.05, 50.0)))
        lam_dep_out.append(float(np.clip(rec_norm / dep_norm, 0.05, 50.0)))
    return lam_ods_out, lam_dep_out


def run_true_exp5_fixed(h_tr: dict, h_ev: dict, out_dir: Path, device: torch.device) -> None:
    print("[EXP-5-FIX] Running sign-verified, identical two-threshold count objective E2E U-Net vs Frozen+S_count (2-seed ensembles)...", flush=True)
    _assert_loss_sign_matches_suh(device)

    t0 = time.monotonic()
    base_unet = rnk.StandardUNetVirtualStainer(3, 3, base=32).to(device)
    n_unet_params = int(sum(p.numel() for p in base_unet.parameters()))
    opt = torch.optim.AdamW(base_unet.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(20261120)
    n_tr = h_tr["he_nhwc"].shape[0]
    for _ in range(10):
        base_unet.train()
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, 16):
            idx = order[s : s + 16]
            xb = torch.from_numpy(h_tr["he_nhwc"][idx]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            yb = torch.from_numpy(h_tr["targ_nchw"][idx]).to(device, dtype=torch.float32) / 255.0
            k_rot = int(rng.integers(0, 4))
            if k_rot:
                xb, yb = torch.rot90(xb, k_rot, dims=(-2, -1)), torch.rot90(yb, k_rot, dims=(-2, -1))
            if int(rng.integers(0, 2)):
                xb, yb = torch.flip(xb, dims=(-1,)), torch.flip(yb, dims=(-1,))
            pred = base_unet(xb)
            loss = F.mse_loss(pred, yb) + 0.5 * F.l1_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    pretrain_sec = time.monotonic() - t0
    base_sd = {k: v.detach().cpu().clone() for k, v in base_unet.state_dict().items()}
    up_tr_full = rnk._predict_unet_u8(base_unet, h_tr["he_nhwc"], device)
    up_ev_full = rnk._predict_unet_u8(base_unet, h_ev["he_nhwc"], device)
    del base_unet
    torch.cuda.empty_cache()

    perm_596 = np.random.default_rng(42).permutation(n_tr)
    regimes = [("full_N596", np.arange(n_tr)), ("lowdata_N149", np.sort(perm_596[:149]))]
    out_exp5 = dict(
        schema="exp5-e2e-vs-frozen-v2-sign-verified",
        architecture="Standard double-convolution U-Net (base=32)",
        unet_params_per_seed=n_unet_params,
        unet_params_2seed=2 * n_unet_params,
        pretrain_sec=round(pretrain_sec, 1),
        regimes={},
    )

    for reg_name, sub_idx in regimes:
        reg_out = {}
        res_frozen = rnk._eval_hemit_full(up_tr_full[sub_idx], up_ev_full, h_tr, h_ev, device, tr_idx=sub_idx)
        reg_out["UNet_Frozen"] = dict(trainable_params_2seed=0, peak_vram_mb=0.0, train_wall_sec=0.0, metrics=res_frozen)
        print(f"  [EXP-5-FIX {reg_name}] UNet_Frozen    : F1={res_frozen['macro_pixel']['Fixed_tol_F1']:.4f} SSIM={res_frozen['macro_pixel']['SSIM']:.4f} CD3_px={res_frozen['per_marker_pixel']['CD3']['Fixed_tol_F1']:.4f} CD3_cell={res_frozen['cd3_cell_f1']:.4f}", flush=True)

        he_sub = h_tr["he_nhwc"][sub_idx]
        up_sub = up_tr_full[sub_idx]
        targ_sub = h_tr["targ_nchw"][sub_idx]
        n_sub = len(sub_idx)

        # Precompute identical per-channel reconstruction normalization scales
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

        for ft_mode in ("UNet_E2E_Recon", "UNet_E2E_Count"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            t_ft = time.monotonic()
            tr_preds_seeds, ev_preds_seeds = [], []
            for sd in (202610270, 202610271):
                torch.manual_seed(sd)
                rng_ft = np.random.default_rng(sd)
                ft_net = rnk.StandardUNetVirtualStainer(3, 3, base=32).to(device)
                ft_net.load_state_dict(base_sd)
                base_lr = 1e-4
                opt_ft = torch.optim.AdamW(ft_net.parameters(), lr=base_lr, weight_decay=1e-4)
                total_steps = max(1, 8 * math.ceil(n_sub / 8))
                warm_steps = max(1, int(0.10 * total_steps))

                # Initialize exact suh lagged tables (255-thr ODS histogram + locked-cutoff dep table)
                lagged_rows_list, dep_tables_list, mean_g_list, thr_floor_list = [], [], [], []
                if ft_mode == "UNet_E2E_Count":
                    with torch.no_grad():
                        for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                            lr_t = torch.zeros((n_sub, 255, 3), dtype=torch.float64, device=device)
                            dt_t = torch.zeros((n_sub, 3), dtype=torch.float64, device=device)
                            g_acc = 0.0
                            for s0 in range(0, n_sub, 16):
                                xb0 = torch.from_numpy(he_sub[s0 : s0 + 16]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                                z0 = ft_net(xb0)[:, ch : ch + 1]
                                y0 = torch.from_numpy(targ_sub[s0 : s0 + 16, ch : ch + 1]).to(device, dtype=torch.uint8)
                                pr0 = suh.prepare_channel_counts(z0, y0, cutoff)
                                lr_t[s0 : s0 + len(xb0)] = suh.fast_densities_hist_nograd(pr0).to(torch.float64)
                                dt_t[s0 : s0 + len(xb0)] = suh.deployment_cutoff_densities(pr0, cutoff).to(torch.float64)
                                g_acc += float(pr0["G"].sum().item())
                            lagged_rows_list.append(lr_t)
                            dep_tables_list.append(dt_t)
                            mean_g_list.append(max(g_acc / float(n_sub), 1e-6))
                            thr_floor_list.append(max(1, min(cutoff, 254)))

                best_sc = -1.0
                best_ft_sd = None
                step = 0
                eff_ods_lams = [1.0, 1.0, 1.0]
                eff_dep_lams = [1.25, 1.25, 1.25]

                for ep in range(8):
                    if ft_mode == "UNet_E2E_Count" and ep % 2 == 0:
                        if ep > 0:
                            ft_net.eval()
                            with torch.no_grad():
                                for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                                    for s0 in range(0, n_sub, 16):
                                        xb0 = torch.from_numpy(he_sub[s0 : s0 + 16]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                                        z0 = ft_net(xb0)[:, ch : ch + 1]
                                        y0 = torch.from_numpy(targ_sub[s0 : s0 + 16, ch : ch + 1]).to(device, dtype=torch.uint8)
                                        pr0 = suh.prepare_channel_counts(z0, y0, cutoff)
                                        lagged_rows_list[ch][s0 : s0 + len(xb0)] = suh.fast_densities_hist_nograd(pr0).to(torch.float64)
                                        dep_tables_list[ch][s0 : s0 + len(xb0)] = suh.deployment_cutoff_densities(pr0, cutoff).to(torch.float64)
                        l_ods, l_dep = _recalibrate_unet_channel_lambdas(
                            ft_net, he_sub, up_sub, targ_sub, ch_scales,
                            lagged_rows_list, dep_tables_list, mean_g_list, thr_floor_list, device,
                        )
                        eff_ods_lams = l_ods
                        eff_dep_lams = [1.25 * v for v in l_dep]

                    ft_net.train()
                    order = rng_ft.permutation(n_sub)
                    for s in range(0, n_sub, 8):
                        ii = order[s : s + 8]
                        idx_gpu = torch.tensor(ii, dtype=torch.long, device=device)
                        xb = torch.from_numpy(he_sub[ii]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                        yb = torch.from_numpy(targ_sub[ii]).to(device, dtype=torch.float32) / 255.0
                        db = torch.from_numpy(up_sub[ii]).to(device, dtype=torch.float32) / 255.0
                        y_u8 = torch.from_numpy(targ_sub[ii]).to(device, dtype=torch.uint8)
                        k_rot = int(rng_ft.integers(0, 4))
                        if k_rot:
                            xb = torch.rot90(xb, k_rot, dims=(-2, -1))
                            yb = torch.rot90(yb, k_rot, dims=(-2, -1))
                            db = torch.rot90(db, k_rot, dims=(-2, -1))
                            y_u8 = torch.rot90(y_u8, k_rot, dims=(-2, -1))
                        if int(rng_ft.integers(0, 2)):
                            xb = torch.flip(xb, dims=(-1,))
                            yb = torch.flip(yb, dims=(-1,))
                            db = torch.flip(db, dims=(-1,))
                            y_u8 = torch.flip(y_u8, dims=(-1,))

                        mult = (step + 1) / warm_steps if step < warm_steps else 0.5 * (1.0 + math.cos(math.pi * (step - warm_steps) / max(1, total_steps - warm_steps)))
                        for pg in opt_ft.param_groups:
                            pg["lr"] = base_lr * mult

                        pred = ft_net(xb)
                        loss_tot = pred.new_tensor(0.0)
                        for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                            z_c = pred[:, ch : ch + 1]
                            y_c = yb[:, ch : ch + 1]
                            d_c = db[:, ch : ch + 1]
                            yu_c = y_u8[:, ch : ch + 1]
                            l_rec = suh.preclip_reconstruction_loss(z_c, y_c, d_c, ch_scales[ch]).mean()
                            if ft_mode == "UNet_E2E_Count":
                                with torch.no_grad():
                                    prep_det = suh.prepare_channel_counts(z_c.detach(), yu_c, cutoff)
                                    lagged_rows_list[ch][idx_gpu] = suh.fast_densities_hist_nograd(prep_det).to(torch.float64)
                                    dep_tables_list[ch][idx_gpu] = suh.deployment_cutoff_densities(prep_det, cutoff).to(torch.float64)
                                    active_thr_t, adj_ods, _ = suh._gpu_pool_ods_adjoint(lagged_rows_list[ch], mean_g_list[ch], thr_floor_list[ch])
                                    adj_dep = suh._adjoint_from_single_row_gpu(dep_tables_list[ch].mean(dim=0), mean_g_list[ch])
                                prep = suh.prepare_channel_counts(z_c, yu_c, cutoff)
                                dens_ods = suh._fast_densities_single_gpu_thr(prep, active_thr_t)
                                dens_dep = suh.deployment_cutoff_densities(prep, cutoff)
                                # EXACT POSITIVE SIGN MATCHING screened_universal_head.py:1033-1036!
                                ods_term = (dens_ods * adj_ods.to(dens_ods.dtype)[None, :]).sum(dim=-1).mean()
                                dep_term = (dens_dep * adj_dep.to(dens_dep.dtype)[None, :]).sum(dim=-1).mean()
                                loss_tot = loss_tot + l_rec + eff_ods_lams[ch] * ods_term + eff_dep_lams[ch] * dep_term
                            else:
                                loss_tot = loss_tot + l_rec
                        loss_tot = loss_tot / 3.0
                        opt_ft.zero_grad(set_to_none=True)
                        loss_tot.backward()
                        torch.nn.utils.clip_grad_norm_(ft_net.parameters(), 2.0)
                        opt_ft.step()
                        step += 1

                    # Composite checkpoint selection at every 2nd epoch matching S_count
                    if (ep + 1) % 2 == 0:
                        p_chk = rnk._predict_unet_u8(ft_net, he_sub, device)
                        sc_sum = 0.0
                        for ch, cutoff in enumerate(rhmc.HEMIT_CUTOFFS):
                            r_ev = jce.batch_curves_and_quality_torch(p_chk[:, ch], targ_sub[:, ch], cutoff, device=device, compute_quality=False)
                            ep_d = jce.endpoints_from_pooled_curves(r_ev["K"].sum(0), r_ev["U"].sum(0), r_ev["V"].sum(0), r_ev["T"].sum(0), int(r_ev["G"].sum()), cutoff)
                            sc_sum += 0.65 * float(ep_d["Fixed_tol_F1"]) + 0.35 * float(ep_d["ODS_F"])
                        if ft_mode == "UNet_E2E_Count":
                            if sc_sum > best_sc:
                                best_sc = sc_sum
                                best_ft_sd = {k: v.detach().cpu().clone() for k, v in ft_net.state_dict().items()}
                        else:
                            best_ft_sd = {k: v.detach().cpu().clone() for k, v in ft_net.state_dict().items()}

                if best_ft_sd is not None:
                    ft_net.load_state_dict(best_ft_sd)
                tr_preds_seeds.append(rnk._predict_unet_u8(ft_net, he_sub, device).astype(np.float32))
                ev_preds_seeds.append(rnk._predict_unet_u8(ft_net, h_ev["he_nhwc"], device).astype(np.float32))
                del ft_net
                torch.cuda.empty_cache()

            ft_sec = time.monotonic() - t_ft
            ft_vram = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            p_tr_ens = np.round(np.mean(tr_preds_seeds, axis=0)).astype(np.uint8)
            p_ev_ens = np.round(np.mean(ev_preds_seeds, axis=0)).astype(np.uint8)
            m_ft = rnk._eval_hemit_full(p_tr_ens, p_ev_ens, h_tr, h_ev, device, tr_idx=sub_idx)
            reg_out[ft_mode] = dict(
                trainable_params_per_seed=n_unet_params,
                trainable_params_2seed=2 * n_unet_params,
                peak_vram_mb=round(ft_vram, 1),
                train_wall_sec=round(ft_sec, 1),
                metrics=m_ft,
            )
            print(f"  [EXP-5-FIX {reg_name}] {ft_mode:15s}: F1={m_ft['macro_pixel']['Fixed_tol_F1']:.4f} SSIM={m_ft['macro_pixel']['SSIM']:.4f} CD3_px={m_ft['per_marker_pixel']['CD3']['Fixed_tol_F1']:.4f} CD3_cell={m_ft['cd3_cell_f1']:.4f} VRAM={ft_vram:.0f}MB sec={ft_sec:.1f}s", flush=True)

        # D. Frozen 7.76M U-Net + 2-seed 3-channel Screened Count Residual head
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        t_sc = time.monotonic()
        ptr_sc = np.empty((n_sub, 3, 512, 512), dtype=np.uint8)
        pev_sc = np.empty_like(up_ev_full)
        n_sc_reg = 0
        for ch, (_, cutoff) in enumerate(zip(rhmc.HEMIT_MARKERS, rhmc.HEMIT_CUTOFFS)):
            c12_tr = rhmc.build_common12_for_3ch(he_sub, up_sub, ch, device=device)
            c12_ev = rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev_full, ch, device=device)
            tr_l, ev_l = [], []
            for sd in (202610270, 202610271):
                m, _ = suh.fit_universal_marker(
                    arm="S_count", seed=sd, common_train=c12_tr,
                    donor_u8_train=torch.from_numpy(up_sub[:, ch]),
                    target_u8_train=torch.from_numpy(targ_sub[:, ch]),
                    cutoff=cutoff, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
                )
                n_sc_reg = int(sum(p.numel() for p in m.parameters()))
                pt, _ = suh.predict_universal_marker(m, c12_tr, torch.from_numpy(up_sub[:, ch]), device=device, batch_size=16)
                pe, _ = suh.predict_universal_marker(m, c12_ev, torch.from_numpy(up_ev_full[:, ch]), device=device, batch_size=16)
                tr_l.append(pt.astype(np.float32))
                ev_l.append(pe.astype(np.float32))
                del m
            ptr_sc[:, ch] = np.round(np.mean(tr_l, axis=0)).astype(np.uint8)
            pev_sc[:, ch] = np.round(np.mean(ev_l, axis=0)).astype(np.uint8)
            del c12_tr, c12_ev
        sc_sec = time.monotonic() - t_sc
        sc_vram = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        m_sc = rnk._eval_hemit_full(ptr_sc, pev_sc, h_tr, h_ev, device, tr_idx=sub_idx)
        reg_out["UNet_Frozen_plus_S_count_64k"] = dict(
            trainable_params_per_head_registered=n_sc_reg,
            trainable_params_3ch_1seed_registered=3 * n_sc_reg,
            trainable_params_3ch_2seed_registered=6 * n_sc_reg,
            trainable_params_3ch_2seed_trunk_readout=6 * 64161,
            trainable_params_3ch_2seed_active_planes=6 * 59553,
            peak_vram_mb=round(sc_vram, 1),
            train_wall_sec=round(sc_sec, 1),
            metrics=m_sc,
        )
        print(f"  [EXP-5-FIX {reg_name}] Frozen+S_count : F1={m_sc['macro_pixel']['Fixed_tol_F1']:.4f} SSIM={m_sc['macro_pixel']['SSIM']:.4f} CD3_px={m_sc['per_marker_pixel']['CD3']['Fixed_tol_F1']:.4f} CD3_cell={m_sc['cd3_cell_f1']:.4f} VRAM={sc_vram:.0f}MB sec={sc_sec:.1f}s", flush=True)
        out_exp5["regimes"][reg_name] = reg_out

    _atomic_write_json(out_dir / "killer" / "exp5_e2e_vs_frozen.json", out_exp5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemit-bank-dir", required=True)
    ap.add_argument("--hemit-up-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-2g", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    with open(Path(args.hemit_bank_dir) / "hemit_bank_train_596.pkl", "rb") as fh:
        h_tr = pickle.load(fh)
    with open(Path(args.hemit_bank_dir) / "hemit_bank_test_945.pkl", "rb") as fh:
        h_ev = pickle.load(fh)
    with np.load(Path(args.hemit_up_dir) / "sanofi_miphei_convnext_train_596.npz") as z:
        up_tr = z["pred_u8"]
    with np.load(Path(args.hemit_up_dir) / "sanofi_miphei_convnext_test_945.npz") as z:
        up_ev = z["pred_u8"]

    if not args.skip_2g:
        run_true_2g_fitted_timing(h_tr, up_tr, h_ev, up_ev, out_dir, device)
    run_true_exp5_fixed(h_tr, h_ev, out_dir, device)
    print("ALL_FIXES_EXP5_2G_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
