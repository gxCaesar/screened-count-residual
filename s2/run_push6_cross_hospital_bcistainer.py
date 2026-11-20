#!/usr/bin/env python3
"""PUSH-6: Real Cross-Hospital, Cross-Scanner Off-the-Shelf Checkpoint Calibration
(Beijing BCIStainer -> Dutch HER2match, ALL 5,980 Locked-Test Tiles).

Deploys the published third-party BCIStainer generator (pretrained on Beijing BCI
Hamamatsu slides, `model_best_psnr.pth`) completely UNMODIFIED onto the independent
European HER2match cohort (3DHistech P1000 scanner, 5,980 locked-test tiles across
5 held-out breast carcinoma slides: wsi001 [0], wsi016 [0], wsi008 [1+], wsi012 [2+],
wsi009 [3+]).

Deconvolves BCIStainer's predicted RGB virtual IHC via Ruifrok-Johnston into DAB optical
density, then benchmarks 5 post-hoc calibration arms fitted strictly on HER2match train
tiles (200/slide = 1,200 tiles):
  upstream | S_recon (2-seed) | QuantileMatch (256-bin LUT) | E_count (2-seed) | S_count (2-seed)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
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


def load_bcistainer_generator(source_root: Path, ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    sys.path.insert(0, str(source_root))
    from omegaconf import OmegaConf
    from libs.train_basic import BCITrainerBasic

    cfg = OmegaConf.load(source_root / "configs/stainer_basic_cmp/exp3.yaml")
    with tempfile.TemporaryDirectory(prefix="bci_load_") as td:
        trainer = BCITrainerBasic(cfg, td, None)
        gen = trainer.Gema
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck
    if isinstance(ck, dict):
        for k in ("Gema", "G_ema", "netG", "G", "state_dict", "model"):
            if k in ck and isinstance(ck[k], dict):
                sd = ck[k]
                break
    # Strip optional module. prefix
    clean_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = gen.load_state_dict(clean_sd, strict=False)
    print(f"[PUSH-6] Loaded BCIStainer Gema: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    assert len(missing) == 0, f"Unexpected missing keys in BCIStainer load: {missing[:10]}"
    gen = gen.to(device).eval()
    n_params = int(sum(p.numel() for p in gen.parameters()))
    return gen, n_params


@torch.no_grad()
def predict_bcistainer_dab_u8(
    gen: torch.nn.Module, he_nhwc: np.ndarray, dab_upper: float, device: torch.device, batch_size: int = 8
) -> np.ndarray:
    """Run BCIStainer RGB H&E -> RGB virtual IHC ([-1, 1]) -> Ruifrok-Johnston DAB OD uint8."""
    mat = np.asarray(hdx_from_rgb, dtype=np.float64)
    n, h, w, _ = he_nhwc.shape
    out = np.empty((n, h, w), dtype=np.uint8)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        xb = torch.from_numpy(he_nhwc[s:e]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 127.5 - 1.0
        # Upsample to 512 if needed, or run native resolution if divisible by 32
        yb = gen(xb)
        if isinstance(yb, (tuple, list)):
            yb = yb[0]
        rgb01 = ((yb.clamp(-1.0, 1.0) + 1.0) * 0.5).permute(0, 2, 3, 1).cpu().numpy()
        for j in range(e - s):
            st = separate_stains(rgb01[j].astype(np.float64), mat)
            dab_01 = np.clip(st[..., 1] / max(dab_upper, 1e-6), 0.0, 1.0)
            out[s + j] = np.rint(dab_01 * 255.0).astype(np.uint8)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-root", required=True)
    ap.add_argument("--bci-src", required=True)
    ap.add_argument("--bci-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--cutoff", type=int, default=23)
    ap.add_argument("--dab-upper", type=float, default=1.35)
    args = ap.parse_args()

    t0 = time.monotonic()
    device = torch.device(args.device)
    tiles_root = Path(args.tiles_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen, n_params = load_bcistainer_generator(Path(args.bci_src), Path(args.bci_ckpt), device)

    print("[PUSH-6] Loading HER2match train (1,200) and FULL locked test (5,980 tiles)...", flush=True)
    tr_pairs = rher.discover_split_pairs(tiles_root, "train", tiles_per_wsi=200, forbid_test=False)
    te_pairs = rher.discover_split_pairs(tiles_root, "test", tiles_per_wsi=0, forbid_test=False)
    tr_bank = rher.load_her2_dab_bank(tiles_root, "train", tr_pairs, dab_upper=args.dab_upper, img_size=args.img_size, forbid_test=False)
    te_bank = rher.load_her2_dab_bank(tiles_root, "test", te_pairs, dab_upper=args.dab_upper, img_size=args.img_size, forbid_test=False)

    print("[PUSH-6] Running zero-shot cross-hospital BCIStainer inference on 7,180 Dutch HER2match tiles...", flush=True)
    up_tr = predict_bcistainer_dab_u8(gen, tr_bank["he_nhwc"], args.dab_upper, device, batch_size=8)
    up_ev = predict_bcistainer_dab_u8(gen, te_bank["he_nhwc"], args.dab_upper, device, batch_size=8)
    del gen
    torch.cuda.empty_cache()

    arms = ("upstream", "S_recon", "QuantileMatch", "E_count", "S_count")
    arm_tr = {a: (up_tr if a == "upstream" else np.empty_like(up_tr)) for a in arms}
    arm_ev = {a: (up_ev if a == "upstream" else np.empty_like(up_ev)) for a in arms}
    qm_tr, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr, tr_bank["dab_u8"], up_ev)
    arm_tr["QuantileMatch"], arm_ev["QuantileMatch"] = qm_tr, qm_ev

    c12_tr = rp5.build_common12_her2(tr_bank["he_nhwc"], up_tr, tr_bank["hem_u8"], device)
    d_tr_t, y_tr_t = torch.from_numpy(up_tr), torch.from_numpy(tr_bank["dab_u8"])
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

            pe_full = np.empty_like(up_ev, dtype=np.float32)
            dc_m = 0.0
            for s0 in range(0, len(up_ev), 1000):
                e0 = min(len(up_ev), s0 + 1000)
                c_chunk = rp5.build_common12_her2(te_bank["he_nhwc"][s0:e0], up_ev[s0:e0], te_bank["hem_u8"][s0:e0], device)
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

    wsis_ev = np.asarray(te_bank["wsis"])
    gt_ev = te_bank["dab_u8"]
    gt_pos = gt_ev > args.cutoff
    neg_m = np.isin(wsis_ev, ["wsi001", "wsi016", "wsi008"])
    zero_m = np.isin(wsis_ev, ["wsi001", "wsi016"])
    arms_eval = {}

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
        pm = dict(cr["pooled"])
        pr_pos = pr > args.cutoff
        tp = int(np.sum(pr_pos & gt_pos))
        fp = int(np.sum(pr_pos & (~gt_pos)))
        fn = int(np.sum((~pr_pos) & gt_pos))
        tn = int(np.sum((~pr_pos) & (~gt_pos)))
        pm["Exact_r0_F1"] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        pm["pixel_specificity_all5980"] = float(tn / max(tn + fp, 1))
        pm["neg3_slides_pred_pos_area_pct"] = round(100.0 * float(pr_pos[neg_m].mean()), 4)
        pm["her2_zero_slides_pred_pos_area_pct"] = round(100.0 * float(pr_pos[zero_m].mean()), 4)
        per_slide = {}
        sl_g, sl_b = [], []
        for sl in sorted(set(wsis_ev)):
            m = wsis_ev == sl
            b_pr = float(pr_pos[m].mean())
            per_slide[sl] = dict(
                grade=rp5.SLIDE_GRADES[sl],
                n_tiles=int(m.sum()),
                Fixed_tol_F1=float(cr["per_cluster"][sl]["Fixed_tol_F1"]),
                SSIM=float(cr["per_cluster"][sl]["SSIM"]),
                pred_dab_pos_pct=round(100.0 * b_pr, 4),
            )
            sl_g.append(rp5.SLIDE_GRADES[sl])
            sl_b.append(b_pr)
        tile_r = float(stats.pearsonr(gt_pos.mean(axis=(1, 2)), pr_pos.mean(axis=(1, 2)))[0]) if np.std(pr_pos.mean(axis=(1, 2))) > 1e-12 else 0.0
        slide_rho = float(stats.spearmanr(sl_g, sl_b)[0]) if np.std(sl_b) > 1e-12 else 0.0
        arms_eval[arm] = dict(
            pooled_5980=pm,
            tile_5980_burden_pearson_r=round(tile_r, 4),
            slide_grade_spearman_rho=round(slide_rho, 4),
            max_dc_drift=float(dc_drifts.get(arm, 0.0)),
            per_slide=per_slide,
        )

    rec = dict(
        schema="push6-cross-hospital-bcistainer-v1",
        source_checkpoint="BCIStainer (Beijing BCI Hamamatsu, model_best_psnr.pth)",
        target_cohort="HER2match (Netherlands 3DHistech P1000, ALL 5,980 locked test tiles)",
        upstream_params=n_params,
        n_train=len(tr_pairs),
        n_test_full=len(te_pairs),
        wall_seconds=round(time.monotonic() - t0, 1),
        arms=arms_eval,
    )
    _atomic_write_json(out_dir / "push6_cross_hospital_bcistainer_her2match.json", rec)
    print(
        f"[PUSH-6 BCIStainer->HER2match 5980] F1: up={arms_eval['upstream']['pooled_5980']['Fixed_tol_F1']:.4f} "
        f"sr={arms_eval['S_recon']['pooled_5980']['Fixed_tol_F1']:.4f} "
        f"qm={arms_eval['QuantileMatch']['pooled_5980']['Fixed_tol_F1']:.4f} "
        f"ec={arms_eval['E_count']['pooled_5980']['Fixed_tol_F1']:.4f} "
        f"sc={arms_eval['S_count']['pooled_5980']['Fixed_tol_F1']:.4f} | "
        f"Neg3_FP%: up={arms_eval['upstream']['pooled_5980']['neg3_slides_pred_pos_area_pct']:.2f}% "
        f"qm={arms_eval['QuantileMatch']['pooled_5980']['neg3_slides_pred_pos_area_pct']:.2f}% "
        f"ec={arms_eval['E_count']['pooled_5980']['neg3_slides_pred_pos_area_pct']:.2f}% "
        f"sc={arms_eval['S_count']['pooled_5980']['neg3_slides_pred_pos_area_pct']:.2f}% | "
        f"SSIM: up={arms_eval['upstream']['pooled_5980']['SSIM']:.4f} qm={arms_eval['QuantileMatch']['pooled_5980']['SSIM']:.4f} sc={arms_eval['S_count']['pooled_5980']['SSIM']:.4f}",
        flush=True,
    )
    print("PUSH6_CROSS_HOSPITAL_BCISTAINER_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
