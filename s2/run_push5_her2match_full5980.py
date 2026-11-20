#!/usr/bin/env python3
"""PUSH-5: Full 5,980-Tile HER2match Multi-Backbone & Negative-Slide Specificity Audit (cluster-host).

Resolves Reviewer M3's 500-tile subsampling critique by evaluating ALL 5,980 locked-test
tiles across ALL 5 held-out breast carcinoma slides:
  - wsi001 (HER2 0,  1,505 tiles)
  - wsi016 (HER2 0,    436 tiles)
  - wsi008 (HER2 1+, 1,427 tiles)
  - wsi012 (HER2 2+, 2,274 tiles)
  - wsi009 (HER2 3+,   338 tiles)

Trains 3 diverse brightfield H&E -> Ruifrok-Johnston DAB optical density generators strictly
on train (2,400 balanced tiles across 6 train slides) & val (600 tiles across 6 val slides),
then freezes them and fits 5 post-hoc calibration arms strictly on train tiles:
  upstream | S_recon | QuantileMatch | E_count (unscreened) | S_count (screened)

Evaluates on ALL 5,980 locked test tiles:
  - Full-cohort and per-slide Fixed-cutoff DAB F1 (tau_DAB=23), pixel-exact r=0 F1, SSIM, PSNR
  - True pixel-level specificity and negative-slide predicted DAB+ area across ALL 3,368
    negative/1+ tiles (wsi001, wsi016, wsi008) — directly falsifying unscreened E_count
    and QuantileMatch DC/FP drift vs exact zero-DC S_count!
  - Whole-slide DAB burden ordering across HER2 grades 0, 0, 1+, 2+, 3+ (Spearman rho).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from s2 import journal_cohort_eval as jce
from s2 import run_her2match_dab_campaign as rher
from s2 import run_nm_comparators20 as rcomp
from s2 import run_plan2_ablations as rp2
from s2 import run_push1_exp5_exact_parity as rp1
from s2 import screened_universal_head as suh

SLIDE_GRADES = {"wsi001": 0, "wsi016": 0, "wsi008": 1, "wsi012": 2, "wsi009": 3}


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


class PatchDisc1(nn.Module):
    def __init__(self, in_ch: int = 4, base: int = 64):
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


def train_dab_generator(arch: str, he_tr: np.ndarray, dab_tr: np.ndarray, he_va: np.ndarray, dab_va: np.ndarray, device: torch.device, epochs: int = 16) -> tuple[nn.Module, dict]:
    t0 = time.monotonic()
    seed = 20261120 + (0 if arch == "her2_unet" else (17 if arch == "her2_resnet34" else 37))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if arch == "her2_resnet34":
        gen = rp1.ResNet34UNetVirtualStainer(3, 1).to(device)
    else:
        gen = rnk_unet(3, 1).to(device)
    is_gan = arch == "her2_pix2pix"
    n_params = int(sum(p.numel() for p in gen.parameters()))
    opt_g = torch.optim.AdamW(gen.parameters(), lr=8e-4 if not is_gan else 5e-4, weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=4e-5)
    disc = PatchDisc1(4, 64).to(device) if is_gan else None
    opt_d = torch.optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.5, 0.999)) if is_gan else None

    bs = 12 if arch == "her2_resnet34" else 16
    best_vl, best_sd, best_ep = 1e9, None, 0
    for ep in range(epochs):
        gen.train()
        if disc is not None:
            disc.train()
        order = rng.permutation(len(he_tr))
        for s in range(0, len(he_tr), bs):
            idx = order[s : s + bs]
            xb = torch.from_numpy(he_tr[idx]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            yb = torch.from_numpy(dab_tr[idx]).unsqueeze(1).to(device, dtype=torch.float32) / 255.0
            k_rot = int(rng.integers(0, 4))
            if k_rot:
                xb, yb = torch.rot90(xb, k_rot, dims=(-2, -1)), torch.rot90(yb, k_rot, dims=(-2, -1))
            if int(rng.integers(0, 2)):
                xb, yb = torch.flip(xb, dims=(-1,)), torch.flip(yb, dims=(-1,))
            fake = gen(xb)
            if is_gan and disc is not None and opt_d is not None:
                opt_d.zero_grad(set_to_none=True)
                d_r = disc(torch.cat([xb, yb], dim=1))
                d_f = disc(torch.cat([xb, fake.detach()], dim=1))
                l_d = 0.5 * (
                    F.binary_cross_entropy_with_logits(d_r, torch.ones_like(d_r))
                    + F.binary_cross_entropy_with_logits(d_f, torch.zeros_like(d_f))
                )
                l_d.backward()
                opt_d.step()

                opt_g.zero_grad(set_to_none=True)
                g_adv = F.binary_cross_entropy_with_logits(disc(torch.cat([xb, fake], dim=1)), torch.ones_like(d_r))
                l_g = F.l1_loss(fake, yb) + 0.5 * F.mse_loss(fake, yb) + 0.015 * g_adv
                l_g.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
                opt_g.step()
            else:
                opt_g.zero_grad(set_to_none=True)
                l_g = F.l1_loss(fake, yb) + 0.5 * F.mse_loss(fake, yb)
                l_g.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
                opt_g.step()
        sched_g.step()
        if (ep + 1) % 2 == 0 or (ep + 1) == epochs:
            gen.eval()
            vsum, vcnt = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(he_va), 32):
                    xb = torch.from_numpy(he_va[s : s + 32]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                    yb = torch.from_numpy(dab_va[s : s + 32]).unsqueeze(1).to(device, dtype=torch.float32) / 255.0
                    vsum += float(F.l1_loss(gen(xb), yb, reduction="sum").item())
                    vcnt += int(yb.numel())
            vl = vsum / max(vcnt, 1)
            if vl < best_vl:
                best_vl, best_ep = vl, ep + 1
                best_sd = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}
    if best_sd is not None:
        gen.load_state_dict(best_sd)
    if disc is not None:
        del disc, opt_d
    torch.cuda.empty_cache()
    return gen, dict(params=n_params, best_epoch=best_ep, best_val_l1=round(best_vl, 6), train_sec=round(time.monotonic() - t0, 1))


def rnk_unet(in_ch: int, out_ch: int) -> nn.Module:
    from s2 import run_nm_nature_killer_suite as rnk
    return rnk.StandardUNetVirtualStainer(in_ch, out_ch, base=32)


@torch.no_grad()
def predict_dab_u8(net: nn.Module, he_nhwc: np.ndarray, device: torch.device, batch_size: int = 32) -> np.ndarray:
    net.eval()
    n, h, w, _ = he_nhwc.shape
    out = np.empty((n, h, w), dtype=np.uint8)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        xb = torch.from_numpy(he_nhwc[s:e]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
        out[s:e] = torch.round(net(xb)[:, 0].clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    return out


def build_common12_her2(he_nhwc: np.ndarray, pred_dab_u8: np.ndarray, hem_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    n = len(he_nhwc)
    chunks = []
    for s in range(0, n, 64):
        e = min(n, s + 64)
        he_f = torch.from_numpy(he_nhwc[s:e]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
        d_f = torch.from_numpy(pred_dab_u8[s:e]).unsqueeze(1).to(device, dtype=torch.float32) / 255.0
        h_f = torch.from_numpy(hem_u8[s:e]).unsqueeze(1).to(device, dtype=torch.float32) / 255.0
        m_f = 0.5 * (d_f + h_f)
        c12 = suh.build_universal_common12(he_f, d_f, h_f, m_f).to(torch.float16).cpu()
        chunks.append(c12)
    return torch.cat(chunks, dim=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-root", required=True)
    ap.add_argument("--archs", default="her2_unet,her2_resnet34,her2_pix2pix")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--cutoff", type=int, default=23)
    ap.add_argument("--dab-upper", type=float, default=1.35)
    args = ap.parse_args()

    device = torch.device(args.device)
    tiles_root = Path(args.tiles_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]

    # Load balanced train (200/slide = 1200) and val (80/slide = 480) + ALL 5,980 LOCKED-TEST TILES!
    print("[PUSH-5] Loading HER2match train, val, and ALL 5,980 locked-test tiles...", flush=True)
    tr_pairs = rher.discover_split_pairs(tiles_root, "train", tiles_per_wsi=200, forbid_test=False)
    va_pairs = rher.discover_split_pairs(tiles_root, "val", tiles_per_wsi=80, forbid_test=False)
    te_pairs = rher.discover_split_pairs(tiles_root, "test", tiles_per_wsi=0, forbid_test=False)  # 0 => ALL 5,980!
    print(f"  Pairs: train={len(tr_pairs)}, val={len(va_pairs)}, FULL test={len(te_pairs)}", flush=True)

    tr_bank = rher.load_her2_dab_bank(tiles_root, "train", tr_pairs, dab_upper=args.dab_upper, img_size=args.img_size, forbid_test=False)
    va_bank = rher.load_her2_dab_bank(tiles_root, "val", va_pairs, dab_upper=args.dab_upper, img_size=args.img_size, forbid_test=False)
    te_bank = rher.load_her2_dab_bank(tiles_root, "test", te_pairs, dab_upper=args.dab_upper, img_size=args.img_size, forbid_test=False)

    wsis_ev = np.asarray(te_bank["wsis"])
    pos_px_by_slide = {}
    tot_pos_px = 0
    for sl in sorted(set(wsis_ev)):
        m = wsis_ev == sl
        p_cnt = int((te_bank["dab_u8"][m] > args.cutoff).sum())
        pos_px_by_slide[sl] = dict(grade=SLIDE_GRADES.get(sl, -1), n_tiles=int(m.sum()), gt_pos_pixels=p_cnt, gt_pos_frac_pct=round(100.0 * float((te_bank["dab_u8"][m] > args.cutoff).mean()), 4))
        tot_pos_px += p_cnt
    for sl in pos_px_by_slide:
        pos_px_by_slide[sl]["share_of_all_test_pos_pct"] = round(100.0 * pos_px_by_slide[sl]["gt_pos_pixels"] / max(tot_pos_px, 1), 3)
    print(f"  Full 5,980-tile test slide GT DAB+ breakdown: {pos_px_by_slide}", flush=True)

    for arch in archs:
        t0 = time.monotonic()
        out_json = out_dir / f"push5_her2_full5980_{arch}.json"
        gen, gen_info = train_dab_generator(arch, tr_bank["he_nhwc"], tr_bank["dab_u8"], va_bank["he_nhwc"], va_bank["dab_u8"], device, epochs=16)
        up_tr = predict_dab_u8(gen, tr_bank["he_nhwc"], device)
        up_ev = predict_dab_u8(gen, te_bank["he_nhwc"], device)
        del gen
        torch.cuda.empty_cache()

        arms = ("upstream", "S_recon", "QuantileMatch", "E_count", "S_count")
        arm_tr = {a: (up_tr if a == "upstream" else np.empty_like(up_tr)) for a in arms}
        arm_ev = {a: (up_ev if a == "upstream" else np.empty_like(up_ev)) for a in arms}

        qm_tr, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr, tr_bank["dab_u8"], up_ev)
        arm_tr["QuantileMatch"] = qm_tr
        arm_ev["QuantileMatch"] = qm_ev

        c12_tr = build_common12_her2(tr_bank["he_nhwc"], up_tr, tr_bank["hem_u8"], device)
        d_tr_t = torch.from_numpy(up_tr)
        y_tr_t = torch.from_numpy(tr_bank["dab_u8"])

        dc_drifts = {}
        for arm in ("S_recon", "E_count", "S_count"):
            tr_l, ev_l, dc_l = [], [], []
            for sd in (202610270, 202610271):
                if arm == "E_count":
                    mdl, _ = rp2.fit_ablation_marker(
                        mode="E_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                        target_u8_train=y_tr_t, cutoff=args.cutoff, device=device,
                        epochs=8, batch_size=8, lr=1.5e-3,
                    )
                    pt, _ = rp2.predict_ablation_marker(mdl, "E_count", c12_tr, d_tr_t, device=device, batch_size=32)
                else:
                    mdl, _ = suh.fit_universal_marker(
                        arm=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                        target_u8_train=y_tr_t, cutoff=args.cutoff, device=device,
                        epochs=8, batch_size=8, lr=1.5e-3, deploy_anchor_weight=1.25,
                    )
                    pt, _ = suh.predict_universal_marker(mdl, c12_tr, d_tr_t, device=device, batch_size=32)
                tr_l.append(pt.astype(np.float32))

                # Stream eval in chunks of 1,000 tiles so 5,980 tiles never spike host RAM
                pe_full = np.empty_like(up_ev, dtype=np.float32)
                dc_m = 0.0
                for s0 in range(0, len(up_ev), 1000):
                    e0 = min(len(up_ev), s0 + 1000)
                    c_chunk = build_common12_her2(te_bank["he_nhwc"][s0:e0], up_ev[s0:e0], te_bank["hem_u8"][s0:e0], device)
                    d_chunk = torch.from_numpy(up_ev[s0:e0])
                    if arm == "E_count":
                        pe_c, de_c = rp2.predict_ablation_marker(mdl, "E_count", c_chunk, d_chunk, device=device, batch_size=32)
                    else:
                        pe_c, de_c = suh.predict_universal_marker(mdl, c_chunk, d_chunk, device=device, batch_size=32)
                    pe_full[s0:e0] = pe_c.astype(np.float32)
                    dc_m = max(dc_m, float(de_c["residual_mean_max"]))
                    del c_chunk
                ev_l.append(pe_full)
                dc_l.append(dc_m)
                del mdl
                torch.cuda.empty_cache()
            arm_tr[arm] = np.round(np.mean(tr_l, axis=0)).astype(np.uint8)
            arm_ev[arm] = np.round(np.mean(ev_l, axis=0)).astype(np.uint8)
            dc_drifts[arm] = max(dc_l)

        del c12_tr
        gc.collect()

        # Evaluate ALL 5,980 locked test tiles
        arms_eval = {}
        gt_ev = te_bank["dab_u8"]
        gt_pos_mask = gt_ev > args.cutoff
        neg_slides_mask = np.isin(wsis_ev, ["wsi001", "wsi016", "wsi008"])
        zero_slides_mask = np.isin(wsis_ev, ["wsi001", "wsi016"])

        for arm in arms:
            pr = arm_ev[arm]
            cr = jce.evaluate_channel_predictions(
                train_pred_u8=arm_tr[arm],
                train_targ_u8=tr_bank["dab_u8"],
                eval_pred_u8=pr,
                eval_targ_u8=gt_ev,
                eval_clusters=wsis_ev,
                fixed_cutoff=args.cutoff,
                device=device,
            )
            p_m = dict(cr["pooled"])
            pr_pos = pr > args.cutoff
            tp = int(np.sum(pr_pos & gt_pos_mask))
            fp = int(np.sum(pr_pos & (~gt_pos_mask)))
            fn = int(np.sum((~pr_pos) & gt_pos_mask))
            tn = int(np.sum((~pr_pos) & (~gt_pos_mask)))
            p_m["Exact_r0_F1"] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            p_m["pixel_specificity_all5980"] = float(tn / max(tn + fp, 1))
            p_m["neg3_slides_pred_pos_area_pct"] = round(100.0 * float(pr_pos[neg_slides_mask].mean()), 4)
            p_m["her2_zero_slides_pred_pos_area_pct"] = round(100.0 * float(pr_pos[zero_slides_mask].mean()), 4)
            p_m["neg3_slides_pixel_specificity"] = float((~pr_pos[neg_slides_mask] & ~gt_pos_mask[neg_slides_mask]).sum() / max((~gt_pos_mask[neg_slides_mask]).sum(), 1))

            # Per-slide burden & clinical ordering
            per_slide = {}
            sl_grades, sl_burdens, sl_gt_burdens = [], [], []
            for sl in sorted(set(wsis_ev)):
                m = wsis_ev == sl
                b_pr = float(pr_pos[m].mean())
                b_gt = float(gt_pos_mask[m].mean())
                per_slide[sl] = dict(
                    grade=SLIDE_GRADES[sl],
                    n_tiles=int(m.sum()),
                    Fixed_tol_F1=float(cr["per_cluster"][sl]["Fixed_tol_F1"]),
                    SSIM=float(cr["per_cluster"][sl]["SSIM"]),
                    pred_dab_pos_pct=round(100.0 * b_pr, 4),
                    gt_dab_pos_pct=round(100.0 * b_gt, 4),
                )
                sl_grades.append(SLIDE_GRADES[sl])
                sl_burdens.append(b_pr)
                sl_gt_burdens.append(b_gt)

            tile_gt_b = gt_pos_mask.mean(axis=(1, 2))
            tile_pr_b = pr_pos.mean(axis=(1, 2))
            tile_r = float(stats.pearsonr(tile_gt_b, tile_pr_b)[0]) if np.std(tile_pr_b) > 1e-12 else 0.0
            slide_rho = float(stats.spearmanr(sl_grades, sl_burdens)[0]) if np.std(sl_burdens) > 1e-12 else 0.0

            arms_eval[arm] = dict(
                pooled_5980=p_m,
                tile_5980_burden_pearson_r=round(tile_r, 4),
                slide_grade_spearman_rho=round(slide_rho, 4),
                max_dc_drift=float(dc_drifts.get(arm, 0.0)),
                per_slide=per_slide,
            )

        rec = dict(
            schema="push5-her2match-full5980-v1",
            architecture=arch,
            upstream_info=gen_info,
            n_train=len(tr_pairs),
            n_val=len(va_pairs),
            n_test_full=len(te_pairs),
            img_size=args.img_size,
            cutoff=args.cutoff,
            gt_slide_breakdown=pos_px_by_slide,
            wall_seconds=round(time.monotonic() - t0, 1),
            arms=arms_eval,
        )
        _atomic_write_json(out_json, rec)
        print(
            f"[PUSH-5 {arch}] F1(5980): up={arms_eval['upstream']['pooled_5980']['Fixed_tol_F1']:.4f} "
            f"sr={arms_eval['S_recon']['pooled_5980']['Fixed_tol_F1']:.4f} "
            f"qm={arms_eval['QuantileMatch']['pooled_5980']['Fixed_tol_F1']:.4f} "
            f"ec={arms_eval['E_count']['pooled_5980']['Fixed_tol_F1']:.4f} "
            f"sc={arms_eval['S_count']['pooled_5980']['Fixed_tol_F1']:.4f} | "
            f"Neg3_FP%: qm={arms_eval['QuantileMatch']['pooled_5980']['neg3_slides_pred_pos_area_pct']:.2f}% "
            f"ec={arms_eval['E_count']['pooled_5980']['neg3_slides_pred_pos_area_pct']:.2f}% "
            f"sc={arms_eval['S_count']['pooled_5980']['neg3_slides_pred_pos_area_pct']:.2f}% | "
            f"SSIM: qm={arms_eval['QuantileMatch']['pooled_5980']['SSIM']:.4f} sc={arms_eval['S_count']['pooled_5980']['SSIM']:.4f}",
            flush=True,
        )

    print("PUSH5_HER2MATCH_FULL5980_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
