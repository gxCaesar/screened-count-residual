#!/usr/bin/env python3
"""v2.1 P0 + P1 + P2: MIST 4-IHC Zero-Mean Unscreened Control (E_count_zm),
Physical Consecutive-Section Registration Diagnostic, Object/Nuclear Localization Audit,
and Per-Slide Independent Statistical Inference across 44+ Held-Out Slides.

Directly addresses the Handling Editor's #1 and #2 review questions:
  1. P0: Compares upstream, QuantileMatch, S_recon, E_count, E_count_zm (r_c = psi_c - mean(psi_c)),
     and S_count across all 4 diagnostic breast IHC markers (ER, PR, HER2, Ki67) x 3 architectures.
  2. P1: Explains the sharp gap between 5-px tolerant F1 and exact r=0 F1 (e.g. MIST Ki67 U-Net
     0.652 vs 0.075) by measuring:
       (a) Empirical 2D sub-pixel phase cross-correlation registration shift between paired H&E
           hematoxylin OD and physical consecutive-section IHC hematoxylin OD on every positive tile.
       (b) Multi-radius tolerance F1 sweep r in {0, 1, 2, 3, 5, 7} px.
       (c) Independent connected-object / nuclear-foci localization F1 (matching within 5 px and 10 px),
           matched object centroid error (px), and off-target FP hallucination rate (>10 px from GT).
  3. P2: Exports exact per-slide metrics for EVERY held-out test slide (10-12 independent slides
     per biomarker, ~44 slides pooled) to support independent slide-level Wilcoxon inference.
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
from scipy import ndimage, spatial, stats
from skimage.color import hdx_from_rgb, separate_stains
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation
from sklearn.metrics import f1_score

from s2 import journal_cohort_eval as jce
from s2 import run_nm_comparators20 as rcomp
from s2 import run_nv2_n7_mist_4ihc as rn7
from s2 import run_plan2_ablations as rp2
from s2 import run_push5_her2match_full5980 as rp5
from s2 import run_v21_p0_ecount_zm_and_loc as rv0
from s2 import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def compute_physical_registration_shift_diagnostic(
    he_nhwc: np.ndarray,
    ihc_nhwc: np.ndarray,
    gt_dab_u8: np.ndarray,
    cutoff: int,
    max_tiles: int = 300,
) -> dict:
    """Measure intrinsic physical consecutive-section misalignment between H&E and IHC hematoxylin channels."""
    mat = np.asarray(hdx_from_rgb, dtype=np.float64)
    pos_idx = [i for i in range(len(gt_dab_u8)) if (gt_dab_u8[i] > cutoff).mean() >= 0.005]
    if len(pos_idx) > max_tiles:
        sel = np.linspace(0, len(pos_idx) - 1, num=max_tiles, dtype=int)
        pos_idx = [pos_idx[k] for k in sel]

    shifts_px = []
    corr_shifts = []
    for i in pos_idx:
        he01 = he_nhwc[i].astype(np.float64) / 255.0
        ihc01 = ihc_nhwc[i].astype(np.float64) / 255.0
        h_he = np.maximum(separate_stains(he01, mat)[..., 0], 0.0)
        h_ihc = np.maximum(separate_stains(ihc01, mat)[..., 0], 0.0)
        if np.std(h_he) < 1e-6 or np.std(h_ihc) < 1e-6:
            continue
        # High-pass standardize to isolate nuclear tissue architecture
        h_he_n = (h_he - h_he.mean()) / (h_he.std() + 1e-8)
        h_ihc_n = (h_ihc - h_ihc.mean()) / (h_ihc.std() + 1e-8)
        shift, _, _ = phase_cross_correlation(h_he_n, h_ihc_n, upsample_factor=4, normalization=None)
        mag = float(np.hypot(shift[0], shift[1]))
        if mag <= 32.0:  # exclude degenerate wraparound artifacts
            shifts_px.append(mag)
            corr_shifts.append((float(shift[0]), float(shift[1])))

    if not shifts_px:
        return dict(n_tiles=0, median_shift_px=None)
    arr = np.asarray(shifts_px, dtype=np.float64)
    return dict(
        n_tiles=int(len(arr)),
        median_shift_px=round(float(np.median(arr)), 3),
        iqr_25_px=round(float(np.percentile(arr, 25)), 3),
        iqr_75_px=round(float(np.percentile(arr, 75)), 3),
        mean_shift_px=round(float(np.mean(arr)), 3),
        pct_gt_1px=round(100.0 * float((arr > 1.0).mean()), 2),
        pct_gt_2px=round(100.0 * float((arr > 2.0).mean()), 2),
        pct_within_5px=round(100.0 * float((arr <= 5.0).mean()), 2),
        pct_within_10px=round(100.0 * float((arr <= 10.0).mean()), 2),
    )


def compute_object_foci_localization(
    pred_u8: np.ndarray,
    gt_u8: np.ndarray,
    cutoff: int,
    min_area: int = 8,
    max_tiles: int = 400,
) -> dict:
    """Object-level positive cellular foci localization (connected components >= min_area px)."""
    n = len(pred_u8)
    idx_all = list(range(n))
    if n > max_tiles:
        idx_all = np.linspace(0, n - 1, num=max_tiles, dtype=int).tolist()

    tp5 = fp5 = fn5 = 0
    tp10 = fp10 = fn10 = 0
    matched_dists_5 = []
    matched_dists_10 = []
    fp_nearest_gt_dists = []

    for i in idx_all:
        p_lbl, np_c = ndimage.label(pred_u8[i] > cutoff)
        g_lbl, ng_c = ndimage.label(gt_u8[i] > cutoff)

        def _centroids(lbl: np.ndarray, k_tot: int) -> np.ndarray:
            if k_tot == 0:
                return np.zeros((0, 2), dtype=np.float64)
            counts = np.bincount(lbl.ravel())
            valid = np.nonzero(counts[1:] >= min_area)[0] + 1
            if len(valid) == 0:
                return np.zeros((0, 2), dtype=np.float64)
            coms = ndimage.center_of_mass(lbl > 0, lbl, valid)
            return np.asarray(coms, dtype=np.float64).reshape(-1, 2)

        p_xy = _centroids(p_lbl, np_c)
        g_xy = _centroids(g_lbl, ng_c)
        if len(p_xy) == 0 and len(g_xy) == 0:
            continue
        if len(g_xy) == 0:
            fp5 += len(p_xy)
            fp10 += len(p_xy)
            continue
        if len(p_xy) == 0:
            fn5 += len(g_xy)
            fn10 += len(g_xy)
            continue

        tree_g = spatial.cKDTree(g_xy)
        d_p2g, _ = tree_g.query(p_xy, k=1)
        tree_p = spatial.cKDTree(p_xy)
        d_g2p, _ = tree_p.query(g_xy, k=1)

        # Within 5 px (matches tolerant 5-px cellular localization radius)
        tp_p5 = int((d_p2g <= 5.0).sum())
        tp_g5 = int((d_g2p <= 5.0).sum())
        tp5 += min(tp_p5, tp_g5)
        fp5 += int((d_p2g > 5.0).sum())
        fn5 += int((d_g2p > 5.0).sum())
        if tp_p5 > 0:
            matched_dists_5.append(d_p2g[d_p2g <= 5.0])

        # Within 10 px (full nuclear diameter ~10 px at 256x256 downsampled scale)
        tp_p10 = int((d_p2g <= 10.0).sum())
        tp_g10 = int((d_g2p <= 10.0).sum())
        tp10 += min(tp_p10, tp_g10)
        fp10 += int((d_p2g > 10.0).sum())
        fn10 += int((d_g2p > 10.0).sum())
        if tp_p10 > 0:
            matched_dists_10.append(d_p2g[d_p2g <= 10.0])
        if (d_p2g > 5.0).any():
            fp_nearest_gt_dists.append(d_p2g[d_p2g > 5.0])

    f1_5 = (2.0 * tp5 / (2.0 * tp5 + fp5 + fn5)) if (2 * tp5 + fp5 + fn5) > 0 else 0.0
    f1_10 = (2.0 * tp10 / (2.0 * tp10 + fp10 + fn10)) if (2 * tp10 + fp10 + fn10) > 0 else 0.0
    m5 = np.concatenate(matched_dists_5) if matched_dists_5 else np.array([])
    m10 = np.concatenate(matched_dists_10) if matched_dists_10 else np.array([])
    fp_d = np.concatenate(fp_nearest_gt_dists) if fp_nearest_gt_dists else np.array([])
    return dict(
        object_f1_5px=round(float(f1_5), 5),
        object_f1_10px=round(float(f1_10), 5),
        object_tp5=int(tp5),
        object_fp5=int(fp5),
        object_fn5=int(fn5),
        matched_5px_centroid_median_px=round(float(np.median(m5)), 3) if len(m5) else None,
        matched_10px_centroid_median_px=round(float(np.median(m10)), 3) if len(m10) else None,
        unmatched_fp_median_dist_px=round(float(np.median(fp_d)), 3) if len(fp_d) else None,
    )


def compute_per_slide_metrics(
    pred_u8: np.ndarray,
    gt_u8: np.ndarray,
    slides: list[str],
    cutoff: int,
    device: torch.device,
) -> dict[str, dict]:
    """Compute exact per-slide metrics across all independent held-out slides."""
    sl_arr = np.asarray(slides)
    out = {}
    for sl in sorted(set(slides)):
        idx = np.nonzero(sl_arr == sl)[0]
        pr_s = pred_u8[idx]
        gt_s = gt_u8[idx]
        ev = jce.batch_curves_and_quality_torch(pr_s, gt_s, cutoff, device=device, compute_quality=True)
        ep = jce.endpoints_from_pooled_curves(
            ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cutoff
        )
        gt_pos_pct = 100.0 * float((gt_s > cutoff).mean())
        pr_pos_pct = 100.0 * float((pr_s > cutoff).mean())
        out[sl] = dict(
            n_tiles=int(len(idx)),
            gt_pos_pct=round(gt_pos_pct, 4),
            pr_pos_pct=round(pr_pos_pct, 4),
            Fixed_tol_F1=round(float(ep["Fixed_tol_F1"]), 5),
            Exact_r0_F1=round(float(ep["Fixed_exact_F1"]), 5),
            SSIM=round(float(ev["ssims"].mean()), 5),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mist-root", required=True)
    ap.add_argument("--markers", default="ER,PR,HER2,Ki67")
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
            print(f"[V21-P1 MIST] Missing {mdir}, skipping", flush=True)
            continue
        out_p = out_dir / f"v21_mist_{marker}_zm_loc.json"
        print(f"[V21-P1 MIST] Loading MIST {marker} (strict 0%-slide-overlap)...", flush=True)
        pack = rn7.load_mist_marker_split(
            mdir, img_size=args.img_size, dab_upper=1.35,
            max_train=args.max_train, max_eval=args.max_eval,
            seed=20261121 + sum(ord(c) for c in marker),
        )
        tr_b = pack["train"]
        ev_b = pack["eval"]
        otsu_raw = int(round(float(threshold_otsu(tr_b["dab_u8"]))))
        cutoff = max(12, min(80, otsu_raw))

        # P1a: Intrinsic physical H&E-to-IHC consecutive-section registration shift diagnostic
        reg_diag = compute_physical_registration_shift_diagnostic(
            ev_b["he_nhwc"], ev_b["ihc_nhwc"], ev_b["dab_u8"], cutoff
        )
        print(
            f"[V21-P1 MIST {marker}] Physical H&E<->IHC shift: median={reg_diag.get('median_shift_px')} px "
            f"(IQR {reg_diag.get('iqr_25_px')}-{reg_diag.get('iqr_75_px')}, >1px={reg_diag.get('pct_gt_1px')}%, <=5px={reg_diag.get('pct_within_5px')}%)",
            flush=True,
        )

        tr_sl_arr = np.asarray(tr_b["slides"])
        uniq_tr_sl = sorted(set(tr_sl_arr.tolist()))
        n_iv = max(1, int(round(0.15 * len(uniq_tr_sl))))
        iv_sl = set(uniq_tr_sl[:n_iv])
        it_mask = ~np.isin(tr_sl_arr, list(iv_sl))
        iv_mask = np.isin(tr_sl_arr, list(iv_sl))

        marker_res = dict(
            schema="v21-p1-mist-zm-loc-v1",
            marker=marker,
            split_mode=pack["split_mode"],
            n_train_tiles=len(tr_b["names"]),
            n_eval_tiles=len(ev_b["names"]),
            n_train_slides=pack["n_train_slides"],
            n_eval_slides=pack["n_eval_slides"],
            cutoff_u8=cutoff,
            gt_eval_pos_area_pct=round(100.0 * float((ev_b["dab_u8"] > cutoff).mean()), 4),
            physical_registration_shift=reg_diag,
            archs={},
        )
        if out_p.exists():
            try:
                prev = json.loads(out_p.read_text())
                marker_res["archs"].update(prev.get("archs", {}))
            except Exception:
                pass

        gt_ev = ev_b["dab_u8"]
        gt_tile_burden = (gt_ev > cutoff).mean(axis=(1, 2))
        neg_tile_mask = gt_tile_burden <= 0.001

        for arch in archs:
            if arch in marker_res["archs"]:
                print(f"[V21-P1 MIST {marker}] Skipping completed {arch}", flush=True)
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
            torch.cuda.empty_cache()

            c12_tr = rp5.build_common12_her2(tr_b["he_nhwc"], up_tr, tr_b["hem_u8"], device)
            c12_ev = rp5.build_common12_her2(ev_b["he_nhwc"], up_ev, ev_b["hem_u8"], device)
            d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr), torch.from_numpy(tr_b["dab_u8"]), torch.from_numpy(up_ev)

            arms_ev = {"upstream": up_ev}
            _, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr, tr_b["dab_u8"], up_ev)
            arms_ev["QuantileMatch"] = qm_ev
            dc_map = {"upstream": 0.0, "QuantileMatch": 0.0}

            for arm in ("S_recon", "E_count", "E_count_zm", "S_count"):
                ev_s = []
                mx_dc = 0.0
                for sd in (202610270, 202610271):
                    if arm in ("E_count", "E_count_zm"):
                        mdl, _ = rp2.fit_ablation_marker(
                            mode=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                            target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                            epochs=args.head_epochs, batch_size=12, lr=1.5e-3,
                        )
                        pe, de = rp2.predict_ablation_marker(mdl, arm, c12_ev, d_ev_t, device=device, batch_size=32)
                    else:
                        mdl, _ = suh.fit_universal_marker(
                            arm=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                            target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                            epochs=args.head_epochs, batch_size=12, lr=1.5e-3, deploy_anchor_weight=1.25,
                        )
                        pe, de = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device=device, batch_size=32)
                    mx_dc = max(mx_dc, float(de.get("max_abs_zero_mean", de.get("residual_mean_max", 0.0))))
                    ev_s.append(pe.astype(np.float32))
                    del mdl
                    torch.cuda.empty_cache()
                arms_ev[arm] = np.rint(np.clip(0.5 * (ev_s[0] + ev_s[1]), 0.0, 255.0)).astype(np.uint8)
                dc_map[arm] = mx_dc
            del c12_tr, c12_ev
            gc.collect()
            torch.cuda.empty_cache()

            arch_arms = {}
            for arm, pr in arms_ev.items():
                ev = jce.batch_curves_and_quality_torch(pr, gt_ev, cutoff, device=device, compute_quality=True)
                ep = jce.endpoints_from_pooled_curves(
                    ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cutoff
                )
                pr_pos = pr > cutoff
                pr_b = pr_pos.mean(axis=(1, 2))
                r_burd = float(stats.pearsonr(gt_tile_burden, pr_b)[0]) if np.std(pr_b) > 1e-12 else 0.0
                neg_fp = 100.0 * float(pr_pos[neg_tile_mask].mean()) if np.any(neg_tile_mask) else None
                rad_loc = rv0.compute_radius_sweep_and_fp_localization(pr, gt_ev, cutoff, max_tiles_dist=250)
                obj_loc = compute_object_foci_localization(pr, gt_ev, cutoff, min_area=8, max_tiles=350)
                per_sl = compute_per_slide_metrics(pr, gt_ev, ev_b["slides"], cutoff, device)
                arch_arms[arm] = dict(
                    Fixed_tol_F1=round(float(ep["Fixed_tol_F1"]), 5),
                    Exact_r0_F1=round(float(ep["Fixed_exact_F1"]), 5),
                    ODS_F=round(float(ep["ODS_F"]), 5),
                    AP=round(float(ep["AP"]), 5),
                    SSIM=round(float(ev["ssims"].mean()), 5),
                    PSNR_dB=round(float(ev["psnrs"].mean()), 3),
                    burden_pearson_r=round(r_burd, 5),
                    neg_tiles_n=int(neg_tile_mask.sum()),
                    neg_tiles_fp_area_pct=round(neg_fp, 4) if neg_fp is not None else None,
                    max_abs_zero_mean=float(dc_map[arm]),
                    radius_f1=rad_loc["radius_f1"],
                    fp_pixel_localization=rad_loc["fp_pixel_localization"],
                    object_foci_localization=obj_loc,
                    per_slide=per_sl,
                )

            marker_res["archs"][arch] = dict(upstream_meta=up_meta, wall_seconds=round(time.time() - ta, 1), arms=arch_arms)
            ec = arch_arms["E_count"]
            zm = arch_arms["E_count_zm"]
            sc = arch_arms["S_count"]
            print(
                f"[V21-P1 MIST {marker:4s}] {arch:12s} ({time.time()-ta:.1f}s) | "
                f"F1: ec={ec['Fixed_tol_F1']:.4f} zm={zm['Fixed_tol_F1']:.4f} sc={sc['Fixed_tol_F1']:.4f} | "
                f"ObjF1@5: zm={zm['object_foci_localization']['object_f1_5px']:.3f} sc={sc['object_foci_localization']['object_f1_5px']:.3f} | "
                f"negFP: zm={zm['neg_tiles_fp_area_pct']} sc={sc['neg_tiles_fp_area_pct']} | "
                f"FP>10px: zm={zm['fp_pixel_localization']['fp_offtarget_gt10px_pct']}% sc={sc['fp_pixel_localization']['fp_offtarget_gt10px_pct']}%",
                flush=True,
            )
            _atomic_write_json(out_p, marker_res)

        marker_res["wall_seconds"] = round(time.time() - tm, 1)
        _atomic_write_json(out_p, marker_res)
        del pack, tr_b, ev_b
        gc.collect()
        torch.cuda.empty_cache()

    print(f"V21_P1_MIST_COMPLETE ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
