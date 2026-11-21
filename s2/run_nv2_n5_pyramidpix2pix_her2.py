#!/usr/bin/env python3
"""NV2-N5: Second Independent Cross-Hospital Beijing-to-Dutch Rescue (Official PyramidPix2Pix).

Deploys the official published 108.13M-parameter Beijing BCI PyramidPix2Pix generator
(Liu et al., CVPR 2022; `bupt-ai-cz/BCI` v1.0 `trained_on_BCI/latest_net_G.pth`,
`attention_unet_32` trained on Beijing Hamamatsu WSI slides) completely UNMODIFIED onto
ALL 5,980 locked-test Dutch HER2match tiles (3DHistech P1000 scanner, 5 held-out slides:
wsi001 [0], wsi016 [0], wsi008 [1+], wsi012 [2+], wsi009 [3+]).

Benchmarks 5 post-hoc calibration arms fitted strictly on the 6 Dutch training slides:
  upstream | S_recon (2-seed) | QuantileMatch (256-bin) | E_count (2-seed) | S_count (2-seed)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from skimage.color import hdx_from_rgb, separate_stains

from s2 import journal_cohort_eval as jce
from s2 import run_her2match_dab_campaign as rher
from s2 import run_nm_comparators20 as rcomp
from s2 import run_plan2_ablations as rp2
from s2 import run_push5_her2match_full5980 as rp5
from s2 import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def load_pyramid_pix2pix(bci_repo_root: Path, ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    sys.path.insert(0, str(bci_repo_root / "PyramidPix2pix"))
    from models import networks

    # Match official BCI training defaults (input_nc=3, output_nc=3, ngf=64, netG='attention_unet_32', norm='batch', use_dropout=True)
    sd = torch.load(ckpt_path, map_location="cpu")
    clean_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    last_err = None
    for net_g in ("attention_unet_32", "resnet_9blocks", "unet_256"):
        for norm in ("batch", "instance"):
            for drop in (True, False):
                try:
                    gen = networks.define_G(3, 3, 64, net_g, norm=norm, use_dropout=drop, gpu_ids=[])
                    missing, unexpected = gen.load_state_dict(clean_sd, strict=False)
                    if len(missing) == 0 and len(unexpected) == 0:
                        print(f"[NV2-N5] Exact match loaded: netG={net_g} norm={norm} dropout={drop}", flush=True)
                        gen = gen.to(device).eval()
                        return gen, int(sum(p.numel() for p in gen.parameters()))
                    last_err = f"netG={net_g} norm={norm} drop={drop}: missing={len(missing)} unexp={len(unexpected)}"
                except Exception as exc:
                    last_err = str(exc)
    raise RuntimeError(f"Could not match PyramidPix2Pix architecture exactly: {last_err}")


@torch.no_grad()
def predict_p2p_dab_u8(
    gen: torch.nn.Module, he_nhwc: np.ndarray, dab_upper: float, device: torch.device, batch_size: int = 8
) -> np.ndarray:
    mat = np.asarray(hdx_from_rgb, dtype=np.float64)
    n, h, w, _ = he_nhwc.shape
    out = np.empty((n, h, w), dtype=np.uint8)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        xb = torch.from_numpy(he_nhwc[s:e]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 127.5 - 1.0
        yb = gen(xb)
        if isinstance(yb, (tuple, list)):
            yb = yb[0]
        rgb01 = ((yb.clamp(-1.0, 1.0) + 1.0) * 0.5).permute(0, 2, 3, 1).cpu().numpy()
        for j in range(e - s):
            st = separate_stains(rgb01[j].astype(np.float64), mat)
            dab_od = np.maximum(st[:, :, 1], 0.0)
            dab_norm = np.clip(dab_od / float(dab_upper), 0.0, 1.0)
            out[s + j] = np.rint(dab_norm * 255.0).astype(np.uint8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--her2-root", required=True)
    ap.add_argument("--bci-repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--cutoff", type=int, default=23)
    ap.add_argument("--dab-upper", type=float, default=1.35)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_root = Path(args.her2_root) / "tiles"

    gen, n_params = load_pyramid_pix2pix(Path(args.bci_repo), Path(args.ckpt), device)

    print("[NV2-N5] Loading HER2match train (1,200 tiles) and ALL 5,980 locked-test tiles...", flush=True)
    tr_pairs = rher.discover_split_pairs(tiles_root, "train", tiles_per_wsi=200, forbid_test=False)
    te_pairs = rher.discover_split_pairs(tiles_root, "test", tiles_per_wsi=0, forbid_test=False)
    tr_bank = rher.load_her2_dab_bank(tiles_root, "train", tr_pairs, dab_upper=args.dab_upper, img_size=args.img_size, forbid_test=False)
    te_bank = rher.load_her2_dab_bank(tiles_root, "test", te_pairs, dab_upper=args.dab_upper, img_size=args.img_size, forbid_test=False)
    print(f"[NV2-N5] Loaded train={len(tr_bank['wsis'])} test={len(te_bank['wsis'])}", flush=True)

    print("[NV2-N5] Running unmodified Beijing PyramidPix2Pix forward pass...", flush=True)
    up_tr = predict_p2p_dab_u8(gen, tr_bank["he_nhwc"], args.dab_upper, device, batch_size=8)
    up_ev = predict_p2p_dab_u8(gen, te_bank["he_nhwc"], args.dab_upper, device, batch_size=8)
    del gen
    gc.collect()
    torch.cuda.empty_cache()

    c12_tr = rp5.build_common12_her2(tr_bank["he_nhwc"], up_tr, tr_bank["hem_u8"], device)
    d_tr_t = torch.from_numpy(up_tr)
    y_tr_t = torch.from_numpy(tr_bank["dab_u8"])

    arms_ev = {"upstream": up_ev}
    _, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr, tr_bank["dab_u8"], up_ev)
    arms_ev["QuantileMatch"] = qm_ev

    for arm in ("S_recon", "E_count", "S_count"):
        preds_s = []
        for sd in (202610270, 202610271):
            if arm == "E_count":
                mdl, _ = rp2.fit_ablation_marker(
                    mode="E_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                    cutoff=args.cutoff, device=device, epochs=8, batch_size=8, lr=1.5e-3,
                )
            else:
                mdl, _ = suh.fit_universal_marker(
                    arm=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                    cutoff=args.cutoff, device=device, epochs=8, batch_size=8, lr=1.5e-3, deploy_anchor_weight=1.25,
                )
            ev_f = np.empty((len(up_ev), args.img_size, args.img_size), dtype=np.float32)
            for s0 in range(0, len(up_ev), 600):
                e0 = min(len(up_ev), s0 + 600)
                c_ch = rp5.build_common12_her2(te_bank["he_nhwc"][s0:e0], up_ev[s0:e0], te_bank["hem_u8"][s0:e0], device)
                d_ch = torch.from_numpy(up_ev[s0:e0])
                if arm == "E_count":
                    pc, _ = rp2.predict_ablation_marker(mdl, "E_count", c_ch, d_ch, device=device, batch_size=24)
                else:
                    pc, _ = suh.predict_universal_marker(mdl, c_ch, d_ch, device=device, batch_size=24)
                ev_f[s0:e0] = pc.astype(np.float32)
                del c_ch
            preds_s.append(ev_f)
            del mdl
            gc.collect()
            torch.cuda.empty_cache()
        arms_ev[arm] = np.rint(np.clip(0.5 * (preds_s[0] + preds_s[1]), 0.0, 255.0)).astype(np.uint8)

    del c12_tr
    gc.collect()

    slides_arr = np.asarray(te_bank["wsis"])
    gt_u8 = te_bank["dab_u8"]
    gt_pos = (gt_u8 > args.cutoff)
    gt_tile_burden = gt_pos.mean(axis=(1, 2))
    neg_mask = np.isin(slides_arr, ["wsi001", "wsi008", "wsi016"])
    zero_mask = np.isin(slides_arr, ["wsi001", "wsi016"])
    slide_grades = {"wsi001": 0, "wsi016": 0, "wsi008": 1, "wsi012": 2, "wsi009": 3}

    summary = dict(
        schema="nv2-n5-cross-hospital-pyramidpix2pix-her2match-v1",
        source_model="bupt-ai-cz/BCI v1.0 PyramidPix2Pix (trained_on_BCI/latest_net_G.pth, Hamamatsu Beijing)",
        n_upstream_params=n_params,
        target_cohort="HER2match (Dutch 3DHistech P1000, all 5,980 locked-test tiles across 5 slides)",
        n_test_tiles=int(len(slides_arr)),
        arms={},
    )

    for arm, pred_u8 in arms_ev.items():
        # Pooled chunked evaluation
        K = np.zeros(256, dtype=np.float64)
        U = np.zeros(256, dtype=np.float64)
        V = np.zeros(256, dtype=np.float64)
        T = np.zeros(256, dtype=np.float64)
        G_tot = 0
        ssims, psnrs = [], []
        for s0 in range(0, len(pred_u8), 600):
            e0 = min(len(pred_u8), s0 + 600)
            ev = jce.batch_curves_and_quality_torch(pred_u8[s0:e0], gt_u8[s0:e0], args.cutoff, device=device, compute_quality=True)
            K += ev["K"].sum(0); U += ev["U"].sum(0); V += ev["V"].sum(0); T += ev["T"].sum(0)
            G_tot += int(ev["G"].sum())
            ssims.append(ev["ssims"]); psnrs.append(ev["psnrs"])
        ep = jce.endpoints_from_pooled_curves(K, U, V, T, G_tot, args.cutoff)
        pr_pos = (pred_u8 > args.cutoff)
        pr_tile_burden = pr_pos.mean(axis=(1, 2))
        r_burd = float(stats.pearsonr(gt_tile_burden, pr_tile_burden)[0]) if np.std(pr_tile_burden) > 1e-12 else 0.0

        per_slide = {}
        sl_gt_b, sl_pr_b = [], []
        for sl in sorted(slide_grades.keys()):
            sm = slides_arr == sl
            ev_s = jce.batch_curves_and_quality_torch(pred_u8[sm], gt_u8[sm], args.cutoff, device=device, compute_quality=True)
            ep_s = jce.endpoints_from_pooled_curves(ev_s["K"].sum(0), ev_s["U"].sum(0), ev_s["V"].sum(0), ev_s["T"].sum(0), int(ev_s["G"].sum()), args.cutoff)
            per_slide[sl] = dict(
                grade=slide_grades[sl],
                n_tiles=int(sm.sum()),
                Fixed_tol_F1=round(float(ep_s["Fixed_tol_F1"]), 4),
                Exact_r0_F1=round(float(ep_s["Fixed_exact_F1"]), 4),
                SSIM=round(float(ev_s["ssims"].mean()), 4),
                pred_pos_area_pct=round(100.0 * float(pr_pos[sm].mean()), 4),
                gt_pos_area_pct=round(100.0 * float(gt_pos[sm].mean()), 4),
            )
            sl_gt_b.append(slide_grades[sl])
            sl_pr_b.append(float(pr_pos[sm].mean()))
        rho_grade = float(stats.spearmanr(sl_gt_b, sl_pr_b)[0]) if np.std(sl_pr_b) > 1e-12 else 0.0

        summary["arms"][arm] = dict(
            Fixed_tol_F1=round(float(ep["Fixed_tol_F1"]), 4),
            Exact_r0_F1=round(float(ep["Fixed_exact_F1"]), 4),
            ODS_F=round(float(ep["ODS_F"]), 4),
            AP=round(float(ep["AP"]), 4),
            SSIM=round(float(np.concatenate(ssims).mean()), 4),
            PSNR_dB=round(float(np.concatenate(psnrs).mean()), 3),
            tile_burden_pearson_r=round(r_burd, 4),
            slide_grade_spearman_rho=round(rho_grade, 4),
            neg_slides_pred_pos_area_pct=round(100.0 * float(pr_pos[neg_mask].mean()), 4),
            her2_zero_slides_pred_pos_area_pct=round(100.0 * float(pr_pos[zero_mask].mean()), 4),
            per_slide=per_slide,
        )
        print(
            f"[NV2-N5] {arm:14s}: F1={summary['arms'][arm]['Fixed_tol_F1']:.4f} "
            f"r0={summary['arms'][arm]['Exact_r0_F1']:.4f} SSIM={summary['arms'][arm]['SSIM']:.4f} "
            f"wsi009_F1={per_slide['wsi009']['Fixed_tol_F1']:.4f} negFP={summary['arms'][arm]['neg_slides_pred_pos_area_pct']:.2f}%",
            flush=True,
        )

    summary["wall_seconds"] = round(time.time() - t0, 1)
    _atomic_write_json(out_dir / "nv2_n5_cross_hospital_pyramidpix2pix_her2match_dab1.json", summary)
    print("NV2_N5_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
