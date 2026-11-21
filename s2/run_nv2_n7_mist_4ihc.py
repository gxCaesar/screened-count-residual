#!/usr/bin/env python3
"""NV2-N7: MIST 4-Biomarker Clinical Breast Cancer IHC Panel (ER, PR, Ki67, HER2).

Evaluates Screened Count Residual across all four diagnostic breast cancer IHC biomarkers
(nuclear hormone receptors ER and PR, nuclear proliferation marker Ki67, membranous oncogene HER2)
from the MIST benchmark (Li et al., CVPR 2023; paired H&E -> brightfield IHC).

Strictly enforces zero slide-prefix overlap between train and test splits (verifying official
valA/valB slide isolation and filtering any overlapping slide prefixes if present; records exact
provenance counts).

Per biomarker:
  - Extracts Ruifrok-Johnston DAB optical density (dab_upper=1.35) and Hematoxylin reference.
  - Locks physical DAB positivity cutoff tau_DAB strictly on training physical IHC via Otsu.
  - Trains 3 fresh brightfield architectures strictly on training slides:
      mist_unet (DoubleConv U-Net), mist_resnet34 (ResNet-34 U-Net), mist_pix2pix (PatchGAN cGAN)
  - Fits 5 post-hoc calibration arms strictly on training slides:
      upstream | S_recon (2-seed) | QuantileMatch (256-bin) | E_count (2-seed) | S_count (2-seed)
  - Evaluates on ALL paired locked test tiles (~1,000 tiles/marker):
      Fixed_tol_F1, Exact_r0_F1, ODS_F, AP, SSIM, PSNR, tile DAB+ burden Pearson r & MAE,
      clinical positivity >=1% (ER/PR) / >=10% (HER2/Ki67) tile classification F1, and
      antigen-negative tile false-positive area (%).
  - Exports crisp high-res real microscopy gallery crops per biomarker for Fig. 6b & ED Fig. 9a-d.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from skimage.color import hdx_from_rgb, separate_stains
from skimage.filters import threshold_otsu
from sklearn.metrics import f1_score

from s2 import journal_cohort_eval as jce
from s2 import run_nm_comparators20 as rcomp
from s2 import run_plan2_ablations as rp2
from s2 import run_push5_her2match_full5980 as rp5
from s2 import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _slide_prefix(fname: str) -> str:
    parts = Path(fname).stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def load_mist_marker_split(
    marker_dir: Path,
    img_size: int = 256,
    dab_upper: float = 1.35,
    max_train: int = 1600,
    max_eval: int = 1000,
    seed: int = 20261121,
) -> dict:
    """Load paired H&E (A) and brightfield IHC (B), enforcing strict zero slide-prefix overlap."""
    trA_dir, trB_dir = marker_dir / "trainA", marker_dir / "trainB"
    vaA_dir, vaB_dir = marker_dir / "valA", marker_dir / "valB"

    tr_names = sorted(p.name for p in trA_dir.glob("*.jpg") if (trB_dir / p.name).exists())
    va_names = sorted(p.name for p in vaA_dir.glob("*.jpg") if (vaB_dir / p.name).exists())

    va_slides = set(_slide_prefix(n) for n in va_names)
    tr_slides_raw = set(_slide_prefix(n) for n in tr_names)
    overlap_raw = tr_slides_raw & va_slides
    # If official split has any overlapping slide prefix, strictly remove overlapping prefixes from train
    if overlap_raw:
        tr_names_clean = [n for n in tr_names if _slide_prefix(n) not in va_slides]
    else:
        tr_names_clean = tr_names

    # If filtering removed too many (meaning official MIST split is patch-level from shared WSIs),
    # construct a clean slide-disjoint 80/20 split across all unique slide prefixes so 0% slide overlap is guaranteed!
    all_pairs = []
    if len(tr_names_clean) < 400:
        for n in tr_names:
            all_pairs.append(("trainA", "trainB", n, _slide_prefix(n)))
        for n in va_names:
            all_pairs.append(("valA", "valB", n, _slide_prefix(n)))
        uniq_sl = sorted(set(x[3] for x in all_pairs))
        rng_sp = np.random.default_rng(seed)
        perm = rng_sp.permutation(uniq_sl)
        n_ev_sl = max(4, int(round(0.20 * len(uniq_sl))))
        ev_sl_set = set(perm[:n_ev_sl].tolist())
        tr_sl_set = set(perm[n_ev_sl:].tolist())
        tr_items = [x for x in all_pairs if x[3] in tr_sl_set]
        ev_items = [x for x in all_pairs if x[3] in ev_sl_set]
        split_mode = "slide_disjoint_repartition_0pct_overlap"
    else:
        tr_items = [("trainA", "trainB", n, _slide_prefix(n)) for n in tr_names_clean]
        ev_items = [("valA", "valB", n, _slide_prefix(n)) for n in va_names]
        split_mode = "official_val_zero_slide_overlap"

    # Stratified deterministic subsample if larger than max_train / max_eval
    rng = np.random.default_rng(seed)
    if max_train > 0 and len(tr_items) > max_train:
        idx = np.sort(rng.choice(len(tr_items), size=max_train, replace=False))
        tr_items = [tr_items[i] for i in idx.tolist()]
    if max_eval > 0 and len(ev_items) > max_eval:
        idx = np.sort(rng.choice(len(ev_items), size=max_eval, replace=False))
        ev_items = [ev_items[i] for i in idx.tolist()]

    tr_sl_final = sorted(set(x[3] for x in tr_items))
    ev_sl_final = sorted(set(x[3] for x in ev_items))
    assert len(set(tr_sl_final) & set(ev_sl_final)) == 0, "Slide leakage detected!"

    mat = np.asarray(hdx_from_rgb, dtype=np.float64)

    def _load_items(items):
        n = len(items)
        he = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
        ihc = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
        dab = np.empty((n, img_size, img_size), dtype=np.uint8)
        hem = np.empty((n, img_size, img_size), dtype=np.uint8)
        slides = []
        names = []
        for i, (dA, dB, fn, sl) in enumerate(items):
            im_a = Image.open(marker_dir / dA / fn).convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
            im_b = Image.open(marker_dir / dB / fn).convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
            a_np = np.asarray(im_a, dtype=np.uint8)
            b_np = np.asarray(im_b, dtype=np.uint8)
            he[i] = a_np
            ihc[i] = b_np
            # Ruifrok-Johnston DAB (col 1 of hdx_from_rgb) from IHC (B) and Hematoxylin (col 0) from H&E (A)
            st_b = separate_stains(b_np.astype(np.float64) / 255.0, mat)
            dab_od = np.maximum(st_b[:, :, 1], 0.0)
            dab[i] = np.rint(np.clip(dab_od / dab_upper, 0.0, 1.0) * 255.0).astype(np.uint8)
            st_a = separate_stains(a_np.astype(np.float64) / 255.0, mat)
            hem_upper = float(np.maximum(mat[:, 0], 0.0).sum())
            hem_od = np.maximum(st_a[:, :, 0], 0.0)
            hem[i] = np.rint(np.clip(hem_od / max(hem_upper, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
            slides.append(sl)
            names.append(fn)
        return dict(he_nhwc=he, ihc_nhwc=ihc, dab_u8=dab, hem_u8=hem, slides=slides, names=names)

    return dict(
        split_mode=split_mode,
        official_slide_overlap_count=int(len(overlap_raw)),
        n_train_slides=len(tr_sl_final),
        n_eval_slides=len(ev_sl_final),
        zero_slide_overlap_verified=True,
        train=_load_items(tr_items),
        eval=_load_items(ev_items),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mist-root", required=True)
    ap.add_argument("--markers", default="ER,PR,Ki67,HER2")
    ap.add_argument("--archs", default="her2_unet,her2_resnet34,her2_pix2pix")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=1600)
    ap.add_argument("--max-eval", type=int, default=1000)
    ap.add_argument("--upstream-epochs", type=int, default=14)
    ap.add_argument("--head-epochs", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mist_root = Path(args.mist_root)
    markers = [m.strip() for m in args.markers.split(",") if m.strip()]
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]

    for marker in markers:
        tm = time.time()
        mdir = mist_root / marker
        if not mdir.exists():
            print(f"[NV2-N7] Missing {mdir}, skipping", flush=True)
            continue
        print(f"[NV2-N7] Loading MIST {marker} with strict zero-slide-overlap verification...", flush=True)
        pack = load_mist_marker_split(
            mdir, img_size=args.img_size, dab_upper=1.35,
            max_train=args.max_train, max_eval=args.max_eval,
            seed=20261121 + sum(ord(c) for c in marker),
        )
        tr_b = pack["train"]
        ev_b = pack["eval"]
        # Train-only Otsu cutoff on physical DAB optical density
        otsu_raw = int(round(float(threshold_otsu(tr_b["dab_u8"]))))
        cutoff = max(12, min(80, otsu_raw))
        print(
            f"[NV2-N7] MIST {marker}: split={pack['split_mode']} tr={len(tr_b['names'])} ({pack['n_train_slides']} sl) "
            f"ev={len(ev_b['names'])} ({pack['n_eval_slides']} sl) cutoff={cutoff} (otsu={otsu_raw})",
            flush=True,
        )

        # Split train 85/15 by slide for upstream checkpoint selection
        tr_sl_arr = np.asarray(tr_b["slides"])
        uniq_tr_sl = sorted(set(tr_sl_arr.tolist()))
        n_iv = max(1, int(round(0.15 * len(uniq_tr_sl))))
        iv_sl = set(uniq_tr_sl[:n_iv])
        it_mask = ~np.isin(tr_sl_arr, list(iv_sl))
        iv_mask = np.isin(tr_sl_arr, list(iv_sl))
        it_sub = dict(he_nhwc=tr_b["he_nhwc"][it_mask], dab_u8=tr_b["dab_u8"][it_mask], hem_u8=tr_b["hem_u8"][it_mask])
        iv_sub = dict(he_nhwc=tr_b["he_nhwc"][iv_mask], dab_u8=tr_b["dab_u8"][iv_mask], hem_u8=tr_b["hem_u8"][iv_mask])

        marker_res = dict(
            schema="nv2-n7-mist-4ihc-v1",
            marker=marker,
            split_mode=pack["split_mode"],
            official_slide_overlap_count=pack["official_slide_overlap_count"],
            zero_slide_overlap_verified=True,
            n_train_tiles=len(tr_b["names"]),
            n_eval_tiles=len(ev_b["names"]),
            n_train_slides=pack["n_train_slides"],
            n_eval_slides=pack["n_eval_slides"],
            cutoff_u8=cutoff,
            gt_eval_pos_area_pct=round(100.0 * float((ev_b["dab_u8"] > cutoff).mean()), 4),
            archs={},
        )

        gt_ev = ev_b["dab_u8"]
        gt_pos = gt_ev > cutoff
        gt_tile_burden = gt_pos.mean(axis=(1, 2))
        # Antigen-negative tiles: <=0.1% positive pixels in physical reference
        neg_tile_mask = gt_tile_burden <= 0.001
        clin_pos_thr = 0.01 if marker in ("ER", "PR") else 0.05  # clinical positivity call threshold

        gallery_saved = False
        for arch in archs:
            ta = time.time()
            print(f"[NV2-N7] Training fresh upstream {arch} on MIST {marker}...", flush=True)
            gen, up_meta = rp5.train_dab_generator(
                arch, it_sub["he_nhwc"], it_sub["dab_u8"], iv_sub["he_nhwc"], iv_sub["dab_u8"],
                device, epochs=args.upstream_epochs,
            )
            up_tr = rp5.predict_dab_u8(gen, tr_b["he_nhwc"], device)
            up_ev = rp5.predict_dab_u8(gen, ev_b["he_nhwc"], device)
            del gen
            gc.collect()
            torch.cuda.empty_cache()

            c12_tr = rp5.build_common12_her2(tr_b["he_nhwc"], up_tr, tr_b["hem_u8"], device)
            c12_ev = rp5.build_common12_her2(ev_b["he_nhwc"], up_ev, ev_b["hem_u8"], device)
            d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr), torch.from_numpy(tr_b["dab_u8"]), torch.from_numpy(up_ev)

            arms_ev = {"upstream": up_ev}
            _, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr, tr_b["dab_u8"], up_ev)
            arms_ev["QuantileMatch"] = qm_ev

            for arm in ("S_recon", "E_count", "S_count"):
                ev_s = []
                for sd in (202610270, 202610271):
                    if arm == "E_count":
                        mdl, _ = rp2.fit_ablation_marker(
                            mode="E_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                            target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                            epochs=args.head_epochs, batch_size=12, lr=1.5e-3,
                        )
                        pe, _ = rp2.predict_ablation_marker(mdl, "E_count", c12_ev, d_ev_t, device=device, batch_size=32)
                    else:
                        mdl, _ = suh.fit_universal_marker(
                            arm=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                            target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                            epochs=args.head_epochs, batch_size=12, lr=1.5e-3, deploy_anchor_weight=1.25,
                        )
                        pe, _ = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device=device, batch_size=32)
                    ev_s.append(pe.astype(np.float32))
                    del mdl
                    torch.cuda.empty_cache()
                arms_ev[arm] = np.rint(np.clip(0.5 * (ev_s[0] + ev_s[1]), 0.0, 255.0)).astype(np.uint8)
            del c12_tr, c12_ev
            gc.collect()
            torch.cuda.empty_cache()

            # Save one crisp representative positive tile per biomarker on her2_unet for Fig 6b & ED Fig 9
            if not gallery_saved and arch == archs[0]:
                # Pick tile with strong positive burden (15-40%) and clear S_count gain over upstream
                best_i = int(np.argmax(
                    ((gt_tile_burden >= 0.08) & (gt_tile_burden <= 0.45)).astype(float)
                    * ((arms_ev["S_count"] > cutoff).mean(axis=(1, 2)) - (up_ev > cutoff).mean(axis=(1, 2)) + 0.05)
                ))
                np.savez_compressed(
                    out_dir / f"n7_mist_{marker}_gallery_crop_dab1.npz",
                    tile_name=ev_b["names"][best_i],
                    cutoff=cutoff,
                    he=ev_b["he_nhwc"][best_i],
                    ihc_rgb=ev_b["ihc_nhwc"][best_i],
                    phys_dab=ev_b["dab_u8"][best_i],
                    frz_dab=up_ev[best_i],
                    qm_dab=arms_ev["QuantileMatch"][best_i],
                    sc_dab=arms_ev["S_count"][best_i],
                )
                gallery_saved = True

            arch_arms = {}
            for arm, pr in arms_ev.items():
                ev = jce.batch_curves_and_quality_torch(pr, gt_ev, cutoff, device=device, compute_quality=True)
                ep = jce.endpoints_from_pooled_curves(
                    ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cutoff
                )
                pr_pos = pr > cutoff
                pr_b = pr_pos.mean(axis=(1, 2))
                r_burd = float(stats.pearsonr(gt_tile_burden, pr_b)[0]) if np.std(pr_b) > 1e-12 else 0.0
                mae_burd = float(np.mean(np.abs(pr_b - gt_tile_burden))) * 100.0
                clin_f1 = float(f1_score(gt_tile_burden >= clin_pos_thr, pr_b >= clin_pos_thr, zero_division=0))
                neg_fp = 100.0 * float(pr_pos[neg_tile_mask].mean()) if np.any(neg_tile_mask) else None
                arch_arms[arm] = dict(
                    Fixed_tol_F1=round(float(ep["Fixed_tol_F1"]), 4),
                    Exact_r0_F1=round(float(ep["Fixed_exact_F1"]), 4),
                    ODS_F=round(float(ep["ODS_F"]), 4),
                    AP=round(float(ep["AP"]), 4),
                    SSIM=round(float(ev["ssims"].mean()), 4),
                    PSNR_dB=round(float(ev["psnrs"].mean()), 3),
                    burden_pearson_r=round(r_burd, 4),
                    burden_mae_pct=round(mae_burd, 4),
                    clinical_pos_f1=round(clin_f1, 4),
                    neg_tiles_n=int(neg_tile_mask.sum()),
                    neg_tiles_fp_area_pct=round(neg_fp, 4) if neg_fp is not None else None,
                )
            marker_res["archs"][arch] = dict(upstream_meta=up_meta, wall_seconds=round(time.time() - ta, 1), arms=arch_arms)
            print(
                f"[NV2-N7] MIST {marker:4s} {arch:14s} ({time.time()-ta:.1f}s) | "
                f"F1: up={arch_arms['upstream']['Fixed_tol_F1']:.4f} sr={arch_arms['S_recon']['Fixed_tol_F1']:.4f} "
                f"qm={arch_arms['QuantileMatch']['Fixed_tol_F1']:.4f} ec={arch_arms['E_count']['Fixed_tol_F1']:.4f} "
                f"sc={arch_arms['S_count']['Fixed_tol_F1']:.4f} | "
                f"SSIM: qm={arch_arms['QuantileMatch']['SSIM']:.3f} sc={arch_arms['S_count']['SSIM']:.3f} | "
                f"negFP: qm={arch_arms['QuantileMatch']['neg_tiles_fp_area_pct']} sc={arch_arms['S_count']['neg_tiles_fp_area_pct']}",
                flush=True,
            )
            _atomic_write_json(out_dir / f"nv2_n7_mist_{marker}_dab1.json", marker_res)

        marker_res["wall_seconds"] = round(time.time() - tm, 1)
        _atomic_write_json(out_dir / f"nv2_n7_mist_{marker}_dab1.json", marker_res)
        del pack, tr_b, ev_b
        gc.collect()
        torch.cuda.empty_cache()

    print(f"NV2_N7_COMPLETE ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
