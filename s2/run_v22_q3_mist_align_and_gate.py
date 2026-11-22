#!/usr/bin/env python3
"""v2.2 Q3: Independent Hematoxylin Alignment + Strict 1:1 Hungarian Object F1 on MIST 4-IHC
AND Exhaustive Calibration-Slide Pre-Deployment Accept/Abstain Clinical Gate.

Two self-contained benchmark tasks for serial-section localization and pre-deployment gating:
  1. ``--task mist_align_hungarian``:
     - Runs MIST 4-IHC (ER, PR, HER2, Ki67 x 3 strictly 0%-slide-overlap architectures = 12 models)
       across all 6 arms: upstream, QuantileMatch, E_count, E_count_zm, L_count, S_count.
     - Computes independent hematoxylin-only consecutive-section rigid translation vector
       (H&E hematoxylin OD <-> IHC hematoxylin OD via phase_cross_correlation, NEVER inspecting
       virtual stain or DAB!). Accepts shift only if ||shift|| <= 24 px AND hematoxylin Pearson r
       improves by > 0.01 on the valid overlap crop.
     - Evaluates Exact r=0 pixel F1, 5-px tolerant F1, strict 1:1 bipartite Hungarian object-foci F1
       (5 px & 10 px, centroid median error, unmatched FP > 10 px), AND per-slide metrics BOTH
       BEFORE and AFTER independent hematoxylin rigid alignment on the exact valid interior box.
     - Also runs repeated 1-, 2-, and 4-slide calibration draws on ``her2_unet`` per biomarker with
       a prespecified zero-test-peeking pre-deployment Accept/Abstain clinical gate.
  2. ``--task xdom_slide_gate``:
     - On Dutch HER2match (6 training slides -> 1,200 tiles; ALL 5,980 locked-test tiles across
       5 held-out slides, 3,368 negative-slide tiles) for BOTH frozen Beijing generators
       (BCIStainer 14.96M EMA generator and official PyramidPix2Pix 17.44M):
     - Runs ALL C(6,1) = 6 single-slide draws AND ALL C(6,2) = 15 two-slide draws = 21 exhaustive
       calibration-slide subsets (+ full 6-slide reference), comparing Frozen, QuantileMatch,
       E_count_zm, L_count, and S_count (2-seed ensemble).
     - Evaluates the EXACT prespecified zero-test-peeking Pre-Deployment Accept/Abstain Gate
       using ONLY an internal 75%/25% stratified split of the calibration slide(s):
         Accept iff: (i) calib_pos_area >= 0.50% AND calib_n_pos_tiles >= 5,
                     (ii) calib_internal_val_f1_sc >= 0.40,
                     (iii) calib_internal_neg_fp_sc <= 2.0%.
       Reports exact acceptance rate, locked-test sensitivity/F1 and 3,368-tile negative-slide FP%
       stratified by Accepted vs Abstained draws.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage, stats
from skimage.color import hdx_from_rgb, separate_stains
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation

from s2 import journal_cohort_eval as jce
from s2 import run_her2match_dab_campaign as rher
from s2 import run_nm_comparators20 as rcomp
from s2 import run_nv2_n5_pyramidpix2pix_her2 as rn5
from s2 import run_nv2_n7_mist_4ihc as rn7
from s2 import run_plan2_ablations as rp2
from s2 import run_push5_her2match_full5980 as rp5
from s2 import run_push6_cross_hospital_bcistainer as rp6
from s2 import run_v22_q1_lcount_all20 as rq1
from s2 import screened_universal_head as suh

MIST_ARMS = ("upstream", "QuantileMatch", "E_count", "E_count_zm", "L_count", "S_count")
XDOM_ARMS = ("QuantileMatch", "E_count_zm", "L_count", "S_count")

# Prespecified zero-test-peeking Pre-Deployment Clinical Accept/Abstain Gate criteria
GATE_MIN_POS_PCT = 0.50
GATE_MIN_POS_TILES = 5
GATE_MIN_INTERNAL_F1 = 0.40
GATE_MAX_INTERNAL_NEG_FP_PCT = 2.00


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _valid_crop_slice(h: int, w: int, dy: float, dx: float) -> tuple[int, int, int, int]:
    y0 = int(math.ceil(dy)) if dy > 0 else 0
    y1 = h - int(math.ceil(-dy)) if dy < 0 else h
    x0 = int(math.ceil(dx)) if dx > 0 else 0
    x1 = w - int(math.ceil(-dx)) if dx < 0 else w
    return y0, y1, x0, x1


def compute_independent_hematoxylin_shifts(
    he_nhwc: np.ndarray,
    ihc_nhwc: np.ndarray,
    gt_dab_u8: np.ndarray,
    max_shift_px: float = 24.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute tile-wise H&E<->IHC rigid translation shifts STRICTLY from hematoxylin channels.

    Returns:
      shifts_yx      : (N, 2) float32 accepted rigid shift applied to IHC -> H&E frame
      gt_aligned_u8  : (N, H, W) uint8 physical GT DAB shifted into H&E frame (0 outside valid crop)
      valid_masks    : (N, H, W) bool exact valid interior overlap masks
      diag           : summary dict (acceptance rate, shift distribution, hematoxylin r before/after)
    """
    mat = np.asarray(hdx_from_rgb, dtype=np.float64)
    n, h, w = gt_dab_u8.shape
    shifts_yx = np.zeros((n, 2), dtype=np.float32)
    gt_aligned = gt_dab_u8.copy()
    valid_masks = np.ones((n, h, w), dtype=bool)

    r_before_list, r_after_list, accepted_mags = [], [], []
    n_accepted = 0

    for i in range(n):
        he01 = he_nhwc[i].astype(np.float64) / 255.0
        ihc01 = ihc_nhwc[i].astype(np.float64) / 255.0
        h_he = np.maximum(separate_stains(he01, mat)[..., 0], 0.0)
        h_ihc = np.maximum(separate_stains(ihc01, mat)[..., 0], 0.0)
        if np.std(h_he) < 1e-6 or np.std(h_ihc) < 1e-6:
            continue
        h_he_n = (h_he - h_he.mean()) / (h_he.std() + 1e-8)
        h_ihc_n = (h_ihc - h_ihc.mean()) / (h_ihc.std() + 1e-8)

        # phase_cross_correlation(reference, moving) returns shift such that ndimage.shift(moving, shift) ~ reference
        shift, _, _ = phase_cross_correlation(h_he_n, h_ihc_n, upsample_factor=4, normalization=None)
        dy, dx = float(shift[0]), float(shift[1])
        mag = float(math.hypot(dy, dx))
        if mag > max_shift_px or mag < 0.25:
            continue

        y0, y1, x0, x1 = _valid_crop_slice(h, w, dy, dx)
        if (y1 - y0) < int(0.5 * h) or (x1 - x0) < int(0.5 * w):
            continue

        h_ihc_sh = ndimage.shift(h_ihc, shift=(dy, dx), order=1, mode="constant", cval=0.0)
        a_ref = h_he[y0:y1, x0:x1].ravel()
        b_raw = h_ihc[y0:y1, x0:x1].ravel()
        b_sh = h_ihc_sh[y0:y1, x0:x1].ravel()
        if np.std(a_ref) < 1e-6 or np.std(b_raw) < 1e-6 or np.std(b_sh) < 1e-6:
            continue
        r_raw = float(np.corrcoef(a_ref, b_raw)[0, 1])
        r_sh = float(np.corrcoef(a_ref, b_sh)[0, 1])

        if r_sh > r_raw + 0.01:
            shifts_yx[i] = (dy, dx)
            sh_gt = ndimage.shift(gt_dab_u8[i].astype(np.float32), shift=(dy, dx), order=1, mode="constant", cval=0.0)
            sh_u8 = np.rint(np.clip(sh_gt, 0.0, 255.0)).astype(np.uint8)
            vm = np.zeros((h, w), dtype=bool)
            vm[y0:y1, x0:x1] = True
            sh_u8[~vm] = 0
            gt_aligned[i] = sh_u8
            valid_masks[i] = vm
            n_accepted += 1
            r_before_list.append(r_raw)
            r_after_list.append(r_sh)
            accepted_mags.append(mag)

    diag = dict(
        n_total_tiles=int(n),
        n_tiles_aligned=int(n_accepted),
        pct_tiles_aligned=round(100.0 * n_accepted / max(1, n), 2),
        accepted_shift_median_px=round(float(np.median(accepted_mags)), 3) if accepted_mags else 0.0,
        accepted_shift_iqr_25_px=round(float(np.percentile(accepted_mags, 25)), 3) if accepted_mags else 0.0,
        accepted_shift_iqr_75_px=round(float(np.percentile(accepted_mags, 75)), 3) if accepted_mags else 0.0,
        hematoxylin_r_before_mean=round(float(np.mean(r_before_list)), 4) if r_before_list else None,
        hematoxylin_r_after_mean=round(float(np.mean(r_after_list)), 4) if r_after_list else None,
        valid_overlap_area_pct=round(100.0 * float(valid_masks.mean()), 2),
    )
    return shifts_yx, gt_aligned, valid_masks, diag


def _mask_preds(pred_u8: np.ndarray, valid_masks: np.ndarray) -> np.ndarray:
    out = pred_u8.copy()
    out[~valid_masks] = 0
    return out


def _eval_mist_view(
    pred_u8: np.ndarray,
    gt_u8: np.ndarray,
    slides: list[str],
    cutoff: int,
    device: torch.device,
) -> dict:
    """Evaluate pooled endpoints, strict 1:1 Hungarian object F1, and per-slide breakdown."""
    ev = jce.batch_curves_and_quality_torch(pred_u8, gt_u8, cutoff, device=device, compute_quality=True)
    ep = jce.endpoints_from_pooled_curves(
        ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cutoff
    )
    hung = rq1.compute_hungarian_object_f1(pred_u8, gt_u8, cutoff, radii=(5.0, 10.0), min_area=8, max_tiles=400)
    gt_b = (gt_u8 > cutoff).mean(axis=(1, 2))
    pr_b = (pred_u8 > cutoff).mean(axis=(1, 2))
    r_burd = float(stats.pearsonr(gt_b, pr_b)[0]) if np.std(pr_b) > 1e-12 and np.std(gt_b) > 1e-12 else 0.0
    neg_m = gt_b <= 0.001
    neg_fp = 100.0 * float((pred_u8[neg_m] > cutoff).mean()) if np.any(neg_m) else None

    sl_arr = np.asarray(slides)
    per_sl = {}
    for sl in sorted(set(slides)):
        idx = np.nonzero(sl_arr == sl)[0]
        evs = jce.batch_curves_and_quality_torch(pred_u8[idx], gt_u8[idx], cutoff, device=device, compute_quality=True)
        eps = jce.endpoints_from_pooled_curves(
            evs["K"].sum(0), evs["U"].sum(0), evs["V"].sum(0), evs["T"].sum(0), int(evs["G"].sum()), cutoff
        )
        per_sl[sl] = dict(
            n_tiles=int(len(idx)),
            gt_pos_pct=round(100.0 * float((gt_u8[idx] > cutoff).mean()), 4),
            Fixed_tol_F1=round(float(eps["Fixed_tol_F1"]), 5),
            Exact_r0_F1=round(float(eps["Fixed_exact_F1"]), 5),
            SSIM=round(float(evs["ssims"].mean()), 5),
        )
    return dict(
        Fixed_tol_F1=round(float(ep["Fixed_tol_F1"]), 5),
        Exact_r0_F1=round(float(ep["Fixed_exact_F1"]), 5),
        ODS_F=round(float(ep["ODS_F"]), 5),
        AP=round(float(ep["AP"]), 5),
        SSIM=round(float(ev["ssims"].mean()), 5),
        PSNR_dB=round(float(ev["psnrs"].mean()), 3),
        burden_pearson_r=round(r_burd, 5),
        neg_tiles_n=int(neg_m.sum()),
        neg_tiles_fp_area_pct=round(neg_fp, 4) if neg_fp is not None else None,
        hungarian_obj=hung,
        per_slide=per_sl,
    )


def evaluate_prespecified_calib_gate(
    c12_sub: torch.Tensor,
    up_sub: np.ndarray,
    gt_sub: np.ndarray,
    cutoff: int,
    device: torch.device,
    seed_offset: int = 0,
    head_epochs: int = 6,
) -> dict:
    """Strictly calibration-split-only (ZERO test peeking!) pre-deployment Accept/Abstain check.

    Splits the calibration tiles 75%/25% stratified by positive vs negative tiles, fits a 1-seed
    S_count diagnostic head strictly on the 75% calibration-train split, evaluates on the 25%
    calibration-internal-val split, and applies the prespecified Accept/Abstain gate:
      Accept iff:
        1. calib_pos_pct >= 0.50% AND calib_n_pos_tiles >= 5
        2. internal_val_f1_sc >= 0.40
        3. internal_val_neg_fp_sc <= 2.0% (if negative tiles exist in internal val)
    """
    n = len(gt_sub)
    tile_pos_frac = (gt_sub > cutoff).mean(axis=(1, 2))
    pos_idx = np.nonzero(tile_pos_frac > 0.005)[0]
    neg_idx = np.nonzero(tile_pos_frac <= 0.001)[0]
    calib_pos_pct = 100.0 * float((gt_sub > cutoff).mean())

    reasons = []
    if calib_pos_pct < GATE_MIN_POS_PCT:
        reasons.append(f"low_calib_pos_area({calib_pos_pct:.3f}%<{GATE_MIN_POS_PCT}%)")
    if len(pos_idx) < GATE_MIN_POS_TILES:
        reasons.append(f"few_pos_tiles({len(pos_idx)}<{GATE_MIN_POS_TILES})")

    # Build deterministic 75/25 stratified internal split within the calibration slide(s)
    rng = np.random.default_rng(20261122 + int(seed_offset))
    order = np.argsort(tile_pos_frac)  # stratify across burden quantiles
    val_mask = np.zeros(n, dtype=bool)
    val_mask[order[::4]] = True  # exact deterministic 25% spanning full positivity spectrum
    tr_mask = ~val_mask
    if int(tr_mask.sum()) < 4 or int(val_mask.sum()) < 2:
        reasons.append("insufficient_tiles_for_internal_val")
        return dict(
            decision="ABSTAIN",
            reasons=reasons,
            calib_n_tiles=int(n),
            calib_pos_pct=round(calib_pos_pct, 4),
            calib_n_pos_tiles=int(len(pos_idx)),
            calib_n_neg_tiles=int(len(neg_idx)),
            internal_val_f1_frz=None,
            internal_val_f1_sc=None,
            internal_val_neg_fp_sc=None,
        )

    c_it, c_iv = c12_sub[tr_mask], c12_sub[val_mask]
    d_it, y_it = torch.from_numpy(up_sub[tr_mask]), torch.from_numpy(gt_sub[tr_mask])
    d_iv, y_iv = torch.from_numpy(up_sub[val_mask]), gt_sub[val_mask]

    bs = min(12, max(2, int(tr_mask.sum()) // 4))
    mdl, _ = suh.fit_universal_marker(
        arm="S_count", seed=202610270, common_train=c_it, donor_u8_train=d_it,
        target_u8_train=y_it, cutoff=cutoff, device=device,
        epochs=head_epochs, batch_size=bs, lr=1.5e-3, deploy_anchor_weight=1.25,
    )
    pr_iv, _ = suh.predict_universal_marker(mdl, c_iv, d_iv, device=device, batch_size=32)
    del mdl
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ev_frz = jce.batch_curves_and_quality_torch(up_sub[val_mask], y_iv, cutoff, device=device, compute_quality=False)
    ep_frz = jce.endpoints_from_pooled_curves(
        ev_frz["K"].sum(0), ev_frz["U"].sum(0), ev_frz["V"].sum(0), ev_frz["T"].sum(0), int(ev_frz["G"].sum()), cutoff
    )
    ev_sc = jce.batch_curves_and_quality_torch(pr_iv, y_iv, cutoff, device=device, compute_quality=False)
    ep_sc = jce.endpoints_from_pooled_curves(
        ev_sc["K"].sum(0), ev_sc["U"].sum(0), ev_sc["V"].sum(0), ev_sc["T"].sum(0), int(ev_sc["G"].sum()), cutoff
    )
    iv_f1_frz = float(ep_frz["Fixed_tol_F1"])
    iv_f1_sc = float(ep_sc["Fixed_tol_F1"])

    iv_neg_m = (y_iv > cutoff).mean(axis=(1, 2)) <= 0.001
    iv_neg_fp = 100.0 * float((pr_iv[iv_neg_m] > cutoff).mean()) if np.any(iv_neg_m) else 0.0

    if iv_f1_sc < GATE_MIN_INTERNAL_F1:
        reasons.append(f"low_internal_val_f1({iv_f1_sc:.3f}<{GATE_MIN_INTERNAL_F1})")
    if np.any(iv_neg_m) and iv_neg_fp > GATE_MAX_INTERNAL_NEG_FP_PCT:
        reasons.append(f"high_internal_neg_fp({iv_neg_fp:.2f}%>{GATE_MAX_INTERNAL_NEG_FP_PCT}%)")

    decision = "ACCEPT" if len(reasons) == 0 else "ABSTAIN"
    return dict(
        decision=decision,
        reasons=reasons,
        calib_n_tiles=int(n),
        calib_pos_pct=round(calib_pos_pct, 4),
        calib_n_pos_tiles=int(len(pos_idx)),
        calib_n_neg_tiles=int(len(neg_idx)),
        internal_val_f1_frz=round(iv_f1_frz, 5),
        internal_val_f1_sc=round(iv_f1_sc, 5),
        internal_val_neg_fp_sc=round(iv_neg_fp, 4) if np.any(iv_neg_m) else None,
    )


def run_mist_align_hungarian(args: argparse.Namespace, device: torch.device) -> None:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mist_root = Path(args.mist_root)
    markers = [m.strip() for m in args.markers.split(",") if m.strip()]
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]

    for marker in markers:
        tm = time.time()
        mdir = mist_root / marker
        if not mdir.exists():
            print(f"[V22-Q3 MIST] Missing {mdir}, skipping", flush=True)
            continue
        out_p = out_dir / f"v22_q3_mist_{marker}_align_hungarian.json"
        print(f"[V22-Q3 MIST] Loading MIST {marker} (strict 0%-slide-overlap)...", flush=True)
        pack = rn7.load_mist_marker_split(
            mdir, img_size=args.img_size, dab_upper=1.35,
            max_train=args.max_train, max_eval=args.max_eval,
            seed=20261121 + sum(ord(c) for c in marker),
        )
        tr_b, ev_b = pack["train"], pack["eval"]
        cutoff = max(12, min(80, int(round(float(threshold_otsu(tr_b["dab_u8"]))))))

        # Compute independent hematoxylin-only rigid shifts ONCE per marker (identical for all archs & arms)
        shifts_yx, gt_ev_aligned, valid_masks, align_diag = compute_independent_hematoxylin_shifts(
            ev_b["he_nhwc"], ev_b["ihc_nhwc"], ev_b["dab_u8"], max_shift_px=24.0
        )
        print(
            f"[V22-Q3 MIST {marker}] Independent H-only align: {align_diag['n_tiles_aligned']}/{align_diag['n_total_tiles']} "
            f"({align_diag['pct_tiles_aligned']}%) aligned | median shift={align_diag['accepted_shift_median_px']} px | "
            f"H-corr {align_diag['hematoxylin_r_before_mean']} -> {align_diag['hematoxylin_r_after_mean']}",
            flush=True,
        )

        tr_sl_arr = np.asarray(tr_b["slides"])
        uniq_tr_sl = sorted(set(tr_sl_arr.tolist()))
        n_iv = max(1, int(round(0.15 * len(uniq_tr_sl))))
        iv_sl = set(uniq_tr_sl[:n_iv])
        it_mask = ~np.isin(tr_sl_arr, list(iv_sl))
        iv_mask = np.isin(tr_sl_arr, list(iv_sl))

        marker_res = dict(
            schema="v22-q3-mist-align-hungarian-gate-v1",
            marker=marker,
            split_mode=pack["split_mode"],
            n_train_tiles=len(tr_b["names"]),
            n_eval_tiles=len(ev_b["names"]),
            n_train_slides=pack["n_train_slides"],
            n_eval_slides=pack["n_eval_slides"],
            cutoff_u8=cutoff,
            gt_eval_pos_area_pct=round(100.0 * float((ev_b["dab_u8"] > cutoff).mean()), 4),
            independent_hematoxylin_alignment=align_diag,
            archs={},
            repeated_slide_gate={},
        )
        if out_p.exists():
            try:
                prev = json.loads(out_p.read_text())
                marker_res["archs"].update(prev.get("archs", {}))
                marker_res["repeated_slide_gate"].update(prev.get("repeated_slide_gate", {}))
            except Exception:
                pass

        gt_ev = ev_b["dab_u8"]
        for arch in archs:
            if arch in marker_res["archs"] and (arch != "her2_unet" or marker_res["repeated_slide_gate"]):
                print(f"[V22-Q3 MIST {marker}] Skipping completed {arch}", flush=True)
                continue
            ta = time.time()
            gen, up_meta = rp5.train_dab_generator(
                arch, tr_b["he_nhwc"][it_mask], tr_b["dab_u8"][it_mask],
                tr_b["he_nhwc"][iv_mask], tr_b["dab_u8"][iv_mask],
                device, epochs=args.upstream_epochs,
            )
            up_tr = rp5.predict_dab_u8(gen, tr_b["he_nhwc"], device)
            up_ev = rp5.predict_dab_u8(gen, ev_b["he_nhwc"], device)
            del gen
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            c12_tr = rp5.build_common12_her2(tr_b["he_nhwc"], up_tr, tr_b["hem_u8"], device)
            c12_ev = rp5.build_common12_her2(ev_b["he_nhwc"], up_ev, ev_b["hem_u8"], device)
            d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr), torch.from_numpy(tr_b["dab_u8"]), torch.from_numpy(up_ev)

            if arch not in marker_res["archs"]:
                arms_ev = {"upstream": up_ev}
                _, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr, tr_b["dab_u8"], up_ev)
                arms_ev["QuantileMatch"] = qm_ev
                dc_map = {"upstream": 0.0, "QuantileMatch": 0.0}

                for arm in ("E_count", "E_count_zm", "L_count", "S_count"):
                    _, pe, dc, _ = rq1.fit_and_predict_2seed_arm_timed(
                        arm, c12_tr, d_tr_t, y_tr_t, c12_ev, d_ev_t, cutoff, device,
                        epochs=args.head_epochs, batch_size=12, lr=1.5e-3,
                    )
                    arms_ev[arm] = pe
                    dc_map[arm] = dc

                arch_arms = {}
                for arm, pr in arms_ev.items():
                    unaligned_m = _eval_mist_view(pr, gt_ev, ev_b["slides"], cutoff, device)
                    aligned_m = _eval_mist_view(_mask_preds(pr, valid_masks), gt_ev_aligned, ev_b["slides"], cutoff, device)
                    arch_arms[arm] = dict(
                        max_abs_zero_mean=float(dc_map[arm]),
                        unaligned=unaligned_m,
                        hematoxylin_aligned=aligned_m,
                    )

                marker_res["archs"][arch] = dict(upstream_meta=up_meta, wall_seconds=round(time.time() - ta, 1), arms=arch_arms)
                sc_u = arch_arms["S_count"]["unaligned"]
                sc_a = arch_arms["S_count"]["hematoxylin_aligned"]
                lc_a = arch_arms["L_count"]["hematoxylin_aligned"]
                zm_a = arch_arms["E_count_zm"]["hematoxylin_aligned"]
                print(
                    f"[V22-Q3 MIST {marker:4s}] {arch:12s} ({time.time()-ta:.1f}s) | "
                    f"SC r0: {sc_u['Exact_r0_F1']:.3f}->{sc_a['Exact_r0_F1']:.3f} | "
                    f"SC Hung@5: {sc_u['hungarian_obj']['r5']['f1']:.3f}->{sc_a['hungarian_obj']['r5']['f1']:.3f} | "
                    f"Aligned F1: ZM={zm_a['Fixed_tol_F1']:.3f} LC={lc_a['Fixed_tol_F1']:.3f} SC={sc_a['Fixed_tol_F1']:.3f}",
                    flush=True,
                )

            # Repeated 1-, 2-, and 4-slide calibration draws + Accept/Abstain gate on her2_unet
            if arch == "her2_unet" and not marker_res["repeated_slide_gate"]:
                tg = time.time()
                rng = np.random.default_rng(20261122 + sum(ord(c) for c in marker))
                slide_gate_runs = {}
                for k_sl, max_draws in ((1, 10), (2, 10), (4, 5)):
                    if len(uniq_tr_sl) < k_sl:
                        continue
                    all_combos = list(itertools.combinations(uniq_tr_sl, k_sl))
                    if len(all_combos) > max_draws:
                        sel_idx = rng.choice(len(all_combos), size=max_draws, replace=False)
                        combos = [all_combos[int(i)] for i in sorted(sel_idx)]
                    else:
                        combos = all_combos
                    draws_out = []
                    for d_i, sl_tuple in enumerate(combos):
                        sub = np.nonzero(np.isin(tr_sl_arr, list(sl_tuple)))[0]
                        if len(sub) < 8:
                            continue
                        gate_res = evaluate_prespecified_calib_gate(
                            c12_tr[sub], up_tr[sub], tr_b["dab_u8"][sub], cutoff, device,
                            seed_offset=k_sl * 100 + d_i, head_epochs=6,
                        )
                        # Fit 1-seed fast calibration on selected slide(s) and evaluate on locked test
                        bs_s = min(12, max(2, len(sub) // 4))
                        mdl, _ = suh.fit_universal_marker(
                            arm="S_count", seed=202610270, common_train=c12_tr[sub],
                            donor_u8_train=torch.from_numpy(up_tr[sub]),
                            target_u8_train=torch.from_numpy(tr_b["dab_u8"][sub]),
                            cutoff=cutoff, device=device, epochs=6, batch_size=bs_s, lr=1.5e-3, deploy_anchor_weight=1.25,
                        )
                        pe_sc, _ = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device=device, batch_size=32)
                        del mdl
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                        ev_sc = jce.batch_curves_and_quality_torch(pe_sc, gt_ev, cutoff, device=device, compute_quality=True)
                        ep_sc = jce.endpoints_from_pooled_curves(
                            ev_sc["K"].sum(0), ev_sc["U"].sum(0), ev_sc["V"].sum(0), ev_sc["T"].sum(0), int(ev_sc["G"].sum()), cutoff
                        )
                        neg_m = (gt_ev > cutoff).mean(axis=(1, 2)) <= 0.001
                        te_neg_fp = 100.0 * float((pe_sc[neg_m] > cutoff).mean()) if np.any(neg_m) else None
                        draws_out.append(dict(
                            slides=list(sl_tuple),
                            n_calib_tiles=int(len(sub)),
                            gate=gate_res,
                            test_S_count=dict(
                                Fixed_tol_F1=round(float(ep_sc["Fixed_tol_F1"]), 5),
                                Exact_r0_F1=round(float(ep_sc["Fixed_exact_F1"]), 5),
                                SSIM=round(float(ev_sc["ssims"].mean()), 5),
                                neg_tiles_fp_pct=round(te_neg_fp, 4) if te_neg_fp is not None else None,
                            ),
                        ))
                    n_acc = sum(1 for x in draws_out if x["gate"]["decision"] == "ACCEPT")
                    acc_f1 = [x["test_S_count"]["Fixed_tol_F1"] for x in draws_out if x["gate"]["decision"] == "ACCEPT"]
                    abs_f1 = [x["test_S_count"]["Fixed_tol_F1"] for x in draws_out if x["gate"]["decision"] == "ABSTAIN"]
                    slide_gate_runs[f"{k_sl}_slide"] = dict(
                        n_draws=len(draws_out),
                        n_accepted=n_acc,
                        n_abstained=len(draws_out) - n_acc,
                        accept_rate_pct=round(100.0 * n_acc / max(1, len(draws_out)), 1),
                        accepted_test_f1_mean=round(float(np.mean(acc_f1)), 5) if acc_f1 else None,
                        abstained_test_f1_mean=round(float(np.mean(abs_f1)), 5) if abs_f1 else None,
                        draws=draws_out,
                    )
                    print(
                        f"[V22-Q3 MIST GATE {marker:4s}] {k_sl}-slide: Accept {n_acc}/{len(draws_out)} | "
                        f"AccF1={slide_gate_runs[f'{k_sl}_slide']['accepted_test_f1_mean']} vs AbsF1={slide_gate_runs[f'{k_sl}_slide']['abstained_test_f1_mean']}",
                        flush=True,
                    )
                marker_res["repeated_slide_gate"] = dict(arch="her2_unet", wall_seconds=round(time.time() - tg, 1), budgets=slide_gate_runs)

            del c12_tr, c12_ev
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _atomic_write_json(out_p, marker_res)

        marker_res["wall_seconds"] = round(time.time() - tm, 1)
        _atomic_write_json(out_p, marker_res)
        del pack, tr_b, ev_b
        gc.collect()

    print(f"V22_Q3_MIST_COMPLETE ({time.time()-t0:.1f}s)", flush=True)


def run_xdom_slide_gate(args: argparse.Namespace, device: torch.device) -> None:
    t0 = time.time()
    her2_root = Path(args.her2_root)
    tiles_root = her2_root / "tiles"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else (out_dir / "v22_q3_xdom_21slides_accept_abstain.json")

    print("[V22-Q3 XDOM] Loading Dutch HER2match train (1,200 tiles, 6 slides) & ALL 5,980 test tiles...", flush=True)
    tr_pairs = rher.discover_split_pairs(tiles_root, "train", tiles_per_wsi=200, forbid_test=False)
    te_pairs = rher.discover_split_pairs(tiles_root, "test", tiles_per_wsi=0, forbid_test=False)
    tr_bank = rher.load_her2_dab_bank(tiles_root, "train", tr_pairs, dab_upper=1.35, img_size=256, forbid_test=False)
    te_bank = rher.load_her2_dab_bank(tiles_root, "test", te_pairs, dab_upper=1.35, img_size=256, forbid_test=False)

    tr_wsis = np.asarray(tr_bank["wsis"])
    te_wsis = np.asarray(te_bank["wsis"])
    uniq_tr_sl = sorted(set(tr_wsis.tolist()))
    neg_sl_mask = np.isin(te_wsis, ["wsi001", "wsi008", "wsi016"])
    gt_ev = te_bank["dab_u8"]
    cutoff = 23

    tr_slide_stats = {
        sl: dict(
            n_tiles=int((tr_wsis == sl).sum()),
            gt_pos_pct=round(100.0 * float((tr_bank["dab_u8"][tr_wsis == sl] > cutoff).mean()), 4),
            n_pos_tiles=int(((tr_bank["dab_u8"][tr_wsis == sl] > cutoff).mean(axis=(1, 2)) > 0.005).sum()),
        )
        for sl in uniq_tr_sl
    }

    res = dict(
        schema="v22-q3-xdom-21slides-accept-abstain-v1",
        n_train_total=len(tr_wsis),
        n_test_total=len(te_wsis),
        n_neg_slide_tiles=int(neg_sl_mask.sum()),
        gate_thresholds=dict(
            min_calib_pos_pct=GATE_MIN_POS_PCT,
            min_calib_pos_tiles=GATE_MIN_POS_TILES,
            min_internal_val_f1=GATE_MIN_INTERNAL_F1,
            max_internal_neg_fp_pct=GATE_MAX_INTERNAL_NEG_FP_PCT,
        ),
        train_slides=tr_slide_stats,
        generators={},
    )
    if out_json.exists():
        try:
            res["generators"].update(json.loads(out_json.read_text()).get("generators", {}))
        except Exception:
            pass

    all_subsets = (
        [(1, (sl,)) for sl in uniq_tr_sl]
        + [(2, pair) for pair in itertools.combinations(uniq_tr_sl, 2)]
        + [(len(uniq_tr_sl), tuple(uniq_tr_sl))]
    )

    want_gens = set(g.strip() for g in args.xdom_gens.split(",") if g.strip())
    for gname, gkind in (("BCIStainer", "bci"), ("PyramidPix2Pix", "p2p")):
        if want_gens and gname not in want_gens:
            continue
        if gname in res["generators"]:
            print(f"[V22-Q3 XDOM] Skipping completed {gname}", flush=True)
            continue
        tg = time.time()
        if gkind == "bci":
            gen, n_par = rp6.load_bcistainer_generator(Path(args.bci_src), Path(args.bci_ckpt), device)
            up_tr = rp6.predict_bcistainer_dab_u8(gen, tr_bank["he_nhwc"], 1.35, device, batch_size=8)
            up_ev = rp6.predict_bcistainer_dab_u8(gen, te_bank["he_nhwc"], 1.35, device, batch_size=8)
        else:
            gen, n_par = rn5.load_pyramid_pix2pix(Path(args.p2p_repo), Path(args.p2p_ckpt), device)
            up_tr = rn5.predict_p2p_dab_u8(gen, tr_bank["he_nhwc"], 1.35, device, batch_size=8)
            up_ev = rn5.predict_p2p_dab_u8(gen, te_bank["he_nhwc"], 1.35, device, batch_size=8)
        del gen
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        c12_tr_all = rp5.build_common12_her2(tr_bank["he_nhwc"], up_tr, tr_bank["hem_u8"], device).to(torch.float16)
        ev_c12_fn = lambda s, e: rp5.build_common12_her2(te_bank["he_nhwc"][s:e], up_ev[s:e], te_bank["hem_u8"][s:e], device)
        d_ev_t = torch.from_numpy(up_ev)

        def _score_eval(pr: np.ndarray) -> dict:
            ev = jce.batch_curves_and_quality_torch(pr, gt_ev, cutoff, device=device, compute_quality=True)
            ep = jce.endpoints_from_pooled_curves(
                ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cutoff
            )
            per_sl = {}
            for sl in sorted(set(te_wsis.tolist())):
                sm = te_wsis == sl
                evs = jce.batch_curves_and_quality_torch(pr[sm], gt_ev[sm], cutoff, device=device, compute_quality=False)
                eps = jce.endpoints_from_pooled_curves(
                    evs["K"].sum(0), evs["U"].sum(0), evs["V"].sum(0), evs["T"].sum(0), int(evs["G"].sum()), cutoff
                )
                per_sl[sl] = dict(
                    Fixed_tol_F1=round(float(eps["Fixed_tol_F1"]), 5),
                    fp_area_pct=round(100.0 * float((pr[sm] > cutoff).mean()), 4),
                )
            return dict(
                Fixed_tol_F1=round(float(ep["Fixed_tol_F1"]), 5),
                Exact_r0_F1=round(float(ep["Fixed_exact_F1"]), 5),
                ODS_F=round(float(ep["ODS_F"]), 5),
                SSIM=round(float(ev["ssims"].mean()), 5),
                neg_3368_fp_pct=round(100.0 * float((pr[neg_sl_mask] > cutoff).mean()), 5),
                per_slide=per_sl,
            )

        g_out = dict(n_params=n_par, upstream=_score_eval(up_ev), subsets=[])

        for idx_s, (k_sl, sl_tuple) in enumerate(all_subsets):
            sub = np.nonzero(np.isin(tr_wsis, list(sl_tuple)))[0]
            c_s = c12_tr_all[sub]
            d_s = torch.from_numpy(up_tr[sub])
            y_s = torch.from_numpy(tr_bank["dab_u8"][sub])

            gate_res = evaluate_prespecified_calib_gate(
                c_s, up_tr[sub], tr_bank["dab_u8"][sub], cutoff, device,
                seed_offset=idx_s * 17, head_epochs=6,
            )

            arms_res = {}
            _, qm_e, _ = rcomp.apply_quantile_match_u8(up_tr[sub], tr_bank["dab_u8"][sub], up_ev)
            arms_res["QuantileMatch"] = _score_eval(qm_e)

            for arm in ("E_count_zm", "L_count", "S_count"):
                _, pe, dc, _ = rq1.fit_and_predict_2seed_arm_timed(
                    arm, c_s, d_s, y_s, None, d_ev_t, cutoff, device,
                    epochs=8, batch_size=12, lr=1.5e-3,
                    ev_c12_fn=ev_c12_fn, d_ev_np=up_ev,
                )
                sc_e = _score_eval(pe)
                sc_e["max_abs_zero_mean"] = float(dc)
                arms_res[arm] = sc_e

            rec = dict(
                k_slides=int(k_sl),
                slides=list(sl_tuple),
                n_calib_tiles=int(len(sub)),
                gate=gate_res,
                arms=arms_res,
            )
            g_out["subsets"].append(rec)
            print(
                f"[V22-Q3 XDOM {gname:14s}] k={k_sl} {'+'.join(sl_tuple):15s} | {gate_res['decision']:7s} "
                f"(pos={gate_res['calib_pos_pct']:.2f}%, ivF1={gate_res['internal_val_f1_sc']}) | "
                f"Test F1: ZM={arms_res['E_count_zm']['Fixed_tol_F1']:.4f} LC={arms_res['L_count']['Fixed_tol_F1']:.4f} "
                f"SC={arms_res['S_count']['Fixed_tol_F1']:.4f} | negFP SC={arms_res['S_count']['neg_3368_fp_pct']:.3f}%",
                flush=True,
            )

        # Summarize Accepted vs Abstained stratified by k=1 (6 draws) and k=2 (15 draws)
        summary_k = {}
        for k_val in (1, 2):
            rows_k = [x for x in g_out["subsets"] if x["k_slides"] == k_val]
            acc_k = [x for x in rows_k if x["gate"]["decision"] == "ACCEPT"]
            abs_k = [x for x in rows_k if x["gate"]["decision"] == "ABSTAIN"]
            summary_k[f"{k_val}_slide"] = dict(
                n_total=len(rows_k),
                n_accepted=len(acc_k),
                n_abstained=len(abs_k),
                accept_rate_pct=round(100.0 * len(acc_k) / max(1, len(rows_k)), 1),
                all_sc_f1_mean=round(float(np.mean([x["arms"]["S_count"]["Fixed_tol_F1"] for x in rows_k])), 5),
                accepted_sc_f1_mean=round(float(np.mean([x["arms"]["S_count"]["Fixed_tol_F1"] for x in acc_k])), 5) if acc_k else None,
                accepted_sc_f1_min=round(float(np.min([x["arms"]["S_count"]["Fixed_tol_F1"] for x in acc_k])), 5) if acc_k else None,
                accepted_sc_neg_fp_mean=round(float(np.mean([x["arms"]["S_count"]["neg_3368_fp_pct"] for x in acc_k])), 5) if acc_k else None,
                accepted_sc_neg_fp_max=round(float(np.max([x["arms"]["S_count"]["neg_3368_fp_pct"] for x in acc_k])), 5) if acc_k else None,
                abstained_sc_f1_mean=round(float(np.mean([x["arms"]["S_count"]["Fixed_tol_F1"] for x in abs_k])), 5) if abs_k else None,
                accepted_lc_f1_mean=round(float(np.mean([x["arms"]["L_count"]["Fixed_tol_F1"] for x in acc_k])), 5) if acc_k else None,
                accepted_zm_f1_mean=round(float(np.mean([x["arms"]["E_count_zm"]["Fixed_tol_F1"] for x in acc_k])), 5) if acc_k else None,
            )
        g_out["gate_summary"] = summary_k
        g_out["wall_seconds"] = round(time.time() - tg, 1)
        res["generators"][gname] = g_out
        _atomic_write_json(out_json, res)
        del c12_tr_all
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"V22_Q3_XDOM_COMPLETE ({time.time()-t0:.1f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mist_align_hungarian", "xdom_slide_gate"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mist-root", default="")
    ap.add_argument("--markers", default="ER,PR,HER2,Ki67")
    ap.add_argument("--archs", default="her2_unet,her2_resnet34,her2_pix2pix")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=1600)
    ap.add_argument("--max-eval", type=int, default=1000)
    ap.add_argument("--upstream-epochs", type=int, default=14)
    ap.add_argument("--head-epochs", type=int, default=8)
    ap.add_argument("--her2-root", default="")
    ap.add_argument("--bci-src", default="")
    ap.add_argument("--bci-ckpt", default="")
    ap.add_argument("--p2p-repo", default="")
    ap.add_argument("--p2p-ckpt", default="")
    ap.add_argument("--xdom-gens", default="BCIStainer,PyramidPix2Pix")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.task == "mist_align_hungarian":
        run_mist_align_hungarian(args, device)
    else:
        run_xdom_slide_gate(args, device)


if __name__ == "__main__":
    main()
