#!/usr/bin/env python3
"""v2.1 P0 + P1: Matched Zero-Mean Subtracted Unscreened Control (E_count_zm) &
Cellular Spatial Localization Audit across All 20 Primary Generators.

Isolates what the two-offset open-lattice screened Poisson spatial operator
``(I + L)^{-1} L`` adds beyond simple per-tile zero-mean subtraction
(``r_c = psi_c - mean(psi_c)``) under identical 12-channel inputs, 64k head,
detached tolerant positive-count objective, composite checkpoint selector,
and paired random seeds (202610270, 202610271).

Also quantifies P1 spatial localization fidelity across Frozen, QuantileMatch,
E_count, E_count_zm, and S_count:
  - Multi-radius tolerance sweep r in {0, 1, 2, 3, 5, 7} px
  - False-positive pixel distance-to-nearest-true-positive-pixel CDF (<=1,2,3,5,10, >10 px)
  - 4-band radial Fourier SMHI (0.50 um/px common grid)
  - HEMIT Cellpose single-cell nuclear F1, CD3+ nuclear F1, 25-um infiltration r,
    56-tile Cellpose CD3-absent FP%, positive-cell centroid offset & FP-to-nearest-true-cell distance
  - Orion-CRC Fold 0 16-ch pixel F1/SSIM, 15-class nuclear F1, CD8a nuclear localization & SMHI
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage, spatial, stats

from s2 import clinical_cell_survival as ccs
from s2 import journal_cohort_eval as jce
from s2 import run_hemit_multibackbone_cell as rhmc
from s2 import run_her2match_dab_campaign as rher
from s2 import run_nm_comparators20 as rcomp
from s2 import run_nv2_n2_mechanistic_xai as rn2
from s2 import run_orion_multibackbone_survival as ror
from s2 import run_plan2_ablations as rp2
from s2 import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _disk_kernel(radius: int) -> np.ndarray:
    r = int(radius)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (yy * yy + xx * xx) <= (r * r)


def compute_radius_sweep_and_fp_localization(
    pred_u8: np.ndarray,
    gt_u8: np.ndarray,
    cutoff: int,
    radii: tuple[int, ...] = (0, 1, 2, 3, 5, 7),
    max_tiles_dist: int = 300,
) -> dict:
    """Compute exact multi-radius tolerant F1 sweep AND FP-pixel distance-to-nearest-GT-pixel CDF."""
    n = len(pred_u8)
    pred_pos = pred_u8 > int(cutoff)
    gt_pos = gt_u8 > int(cutoff)

    rad_f1 = {}
    for r in radii:
        tp_p = 0
        tp_g = 0
        n_p = int(pred_pos.sum())
        n_g = int(gt_pos.sum())
        if r == 0:
            inter = int((pred_pos & gt_pos).sum())
            tp_p = inter
            tp_g = inter
        else:
            ker = _disk_kernel(r)
            for i in range(n):
                p_i = pred_pos[i]
                g_i = gt_pos[i]
                if not p_i.any() and not g_i.any():
                    continue
                if p_i.any() and g_i.any():
                    g_dil = ndimage.binary_dilation(g_i, structure=ker)
                    p_dil = ndimage.binary_dilation(p_i, structure=ker)
                    tp_p += int((p_i & g_dil).sum())
                    tp_g += int((g_i & p_dil).sum())
        prec = tp_p / max(1, n_p)
        rec = tp_g / max(1, n_g)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rad_f1[f"r{r}"] = round(float(f1), 6)

    # Distance from predicted positive pixels to nearest GT positive pixel (on positive GT tiles)
    pos_tiles = [i for i in range(n) if gt_pos[i].any()]
    if len(pos_tiles) > max_tiles_dist:
        idx_sel = np.linspace(0, len(pos_tiles) - 1, num=max_tiles_dist, dtype=int)
        pos_tiles = [pos_tiles[k] for k in idx_sel]

    all_fp_dists = []
    tot_fp = 0
    tot_pred_on_pos = 0
    for i in pos_tiles:
        p_i = pred_pos[i]
        g_i = gt_pos[i]
        if not p_i.any():
            continue
        tot_pred_on_pos += int(p_i.sum())
        fp_mask = p_i & (~g_i)
        n_fp_i = int(fp_mask.sum())
        if n_fp_i == 0:
            continue
        tot_fp += n_fp_i
        edt = ndimage.distance_transform_edt(~g_i)
        d_vals = edt[fp_mask]
        if len(d_vals) > 2000:
            step = max(1, len(d_vals) // 2000)
            d_vals = d_vals[::step]
        all_fp_dists.append(d_vals.astype(np.float32))

    if all_fp_dists:
        d_cat = np.concatenate(all_fp_dists)
        fp_loc = dict(
            n_sampled_fp_pixels=int(len(d_cat)),
            fp_median_dist_to_gt_px=round(float(np.median(d_cat)), 3),
            fp_p90_dist_to_gt_px=round(float(np.percentile(d_cat, 90)), 3),
            fp_within_2px_pct=round(100.0 * float((d_cat <= 2.0).mean()), 3),
            fp_within_5px_pct=round(100.0 * float((d_cat <= 5.0).mean()), 3),
            fp_within_10px_pct=round(100.0 * float((d_cat <= 10.0).mean()), 3),
            fp_offtarget_gt10px_pct=round(100.0 * float((d_cat > 10.0).mean()), 3),
            fp_offtarget_gt25px_pct=round(100.0 * float((d_cat > 25.0).mean()), 3),
        )
    else:
        fp_loc = dict(
            n_sampled_fp_pixels=0,
            fp_median_dist_to_gt_px=None,
            fp_p90_dist_to_gt_px=None,
            fp_within_2px_pct=None,
            fp_within_5px_pct=None,
            fp_within_10px_pct=None,
            fp_offtarget_gt10px_pct=None,
            fp_offtarget_gt25px_pct=None,
        )

    return dict(radius_f1=rad_f1, fp_pixel_localization=fp_loc)


def compute_hemit_cell_localization_metrics(
    eval_nuclei_records: list,
    pred_nchw: np.ndarray,
    tau_cd3_cell: float,
    pixel_size_um: float = 0.46,
) -> dict:
    """Compute cellular-level CD3+ FP nearest-true-CD3-cell distance and 56-tile Cellpose absent FP%."""
    fp_cell_dists_px = []
    n_tp_cells = 0
    n_fp_cells = 0
    n_fn_cells = 0
    absent_tile_fp_rates = []
    n_absent_tiles = 0

    for rec in eval_nuclei_records:
        if rec is None:
            continue
        ti = int(rec["tile_idx"])
        gt_cd3 = np.asarray(rec["cd3_pos"], dtype=np.int32) > 0
        if "idx_spec" in rec:
            pm = ccs.extract_means_with_index(pred_nchw[ti], rec["idx_spec"])
        else:
            nuc = ccs.extract_tile_nuclei(pred_nchw[ti], rec["mask_hw"], min_area=10)
            id_to_pos = {int(cid): idx for idx, cid in enumerate(nuc["cell_ids"].tolist())}
            rows = [id_to_pos[int(cid)] for cid in rec["cell_ids"].tolist()]
            pm = nuc["means"][rows]
        pr_cd3 = pm[:, 1] > float(tau_cd3_cell)
        if not gt_cd3.any():
            n_absent_tiles += 1
            absent_tile_fp_rates.append(float((pred_nchw[ti, 1] > 30).mean()))
        n_tp_cells += int((pr_cd3 & gt_cd3).sum())
        n_fp_cells += int((pr_cd3 & (~gt_cd3)).sum())
        n_fn_cells += int(((~pr_cd3) & gt_cd3).sum())
        if gt_cd3.any() and (pr_cd3 & (~gt_cd3)).any():
            gt_xy = rec["centroids_yx"][gt_cd3]
            fp_xy = rec["centroids_yx"][pr_cd3 & (~gt_cd3)]
            tree = spatial.cKDTree(gt_xy)
            dists, _ = tree.query(fp_xy, k=1)
            fp_cell_dists_px.append(dists.astype(np.float64))

    if fp_cell_dists_px:
        dc = np.concatenate(fp_cell_dists_px)
        cell_fp_loc = dict(
            n_fp_cells_on_pos_tiles=int(len(dc)),
            fp_cell_median_dist_px=round(float(np.median(dc)), 3),
            fp_cell_median_dist_um=round(float(np.median(dc) * pixel_size_um), 3),
            fp_cell_within_15px_pct=round(100.0 * float((dc <= 15.0).mean()), 3),
            fp_cell_gt25um_pct=round(100.0 * float((dc * pixel_size_um > 25.0).mean()), 3),
        )
    else:
        cell_fp_loc = dict(
            n_fp_cells_on_pos_tiles=0,
            fp_cell_median_dist_px=None,
            fp_cell_median_dist_um=None,
            fp_cell_within_15px_pct=None,
            fp_cell_gt25um_pct=None,
        )

    prec = n_tp_cells / max(1, n_tp_cells + n_fp_cells)
    rec_v = n_tp_cells / max(1, n_tp_cells + n_fn_cells)
    f1 = (2.0 * prec * rec_v / (prec + rec_v)) if (prec + rec_v) > 0 else 0.0
    return dict(
        cd3_cell_tp=n_tp_cells,
        cd3_cell_fp=n_fp_cells,
        cd3_cell_fn=n_fn_cells,
        cd3_cell_precision=round(float(prec), 6),
        cd3_cell_recall=round(float(rec_v), 6),
        cd3_cell_f1=round(float(f1), 6),
        cellpose_cd3_absent_tiles_n=n_absent_tiles,
        cellpose_cd3_absent_pixel_fp_pct=round(100.0 * float(np.mean(absent_tile_fp_rates)), 5) if absent_tile_fp_rates else None,
        fp_cell_localization=cell_fp_loc,
    )


def fit_and_predict_2seed_arm(
    arm: str,
    c12_tr: torch.Tensor,
    d_tr: torch.Tensor,
    y_tr: torch.Tensor,
    c12_ev: torch.Tensor,
    d_ev: torch.Tensor,
    cutoff: int,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 1.2e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit exact 2-seed ensemble for S_count, E_count, or E_count_zm; returns (tr_ens, ev_ens, max_dc)."""
    tr_preds = []
    ev_preds = []
    max_dc = 0.0
    for sd in (202610270, 202610271):
        if arm == "S_count":
            mdl, _ = suh.fit_universal_marker(
                arm="S_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr,
                target_u8_train=y_tr, cutoff=cutoff, device=device,
                epochs=epochs, batch_size=batch_size, lr=lr, deploy_anchor_weight=1.25,
            )
            pt, dt = suh.predict_universal_marker(mdl, c12_tr, d_tr, device=device, batch_size=max(8, batch_size))
            pe, de = suh.predict_universal_marker(mdl, c12_ev, d_ev, device=device, batch_size=max(8, batch_size))
        else:
            mdl, _ = rp2.fit_ablation_marker(
                mode=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr,
                target_u8_train=y_tr, cutoff=cutoff, device=device,
                epochs=epochs, batch_size=batch_size, lr=lr, deploy_anchor_weight=1.25,
            )
            pt, dt = rp2.predict_ablation_marker(mdl, arm, c12_tr, d_tr, device=device, batch_size=max(8, batch_size))
            pe, de = rp2.predict_ablation_marker(mdl, arm, c12_ev, d_ev, device=device, batch_size=max(8, batch_size))
        max_dc = max(max_dc, float(de.get("max_abs_zero_mean", de.get("residual_mean_max", 0.0))))
        tr_preds.append(pt.astype(np.float32))
        ev_preds.append(pe.astype(np.float32))
        del mdl
        torch.cuda.empty_cache()
    tr_ens = np.rint(np.clip(0.5 * (tr_preds[0] + tr_preds[1]), 0.0, 255.0)).astype(np.uint8)
    ev_ens = np.rint(np.clip(0.5 * (ev_preds[0] + ev_preds[1]), 0.0, 255.0)).astype(np.uint8)
    return tr_ens, ev_ens, max_dc


def _eval_channel_endpoints(
    tr_pr: np.ndarray, tr_gt: np.ndarray, ev_pr: np.ndarray, ev_gt: np.ndarray, cutoff: int, device: torch.device
) -> dict:
    ev_tr = jce.batch_curves_and_quality_torch(tr_pr, tr_gt, cutoff, device=device, compute_quality=False)
    ep_tr = jce.endpoints_from_pooled_curves(
        ev_tr["K"].sum(0), ev_tr["U"].sum(0), ev_tr["V"].sum(0), ev_tr["T"].sum(0), int(ev_tr["G"].sum()), cutoff
    )
    tau_cal = int(ep_tr["ODS_threshold"])
    ev_te = jce.batch_curves_and_quality_torch(ev_pr, ev_gt, cutoff, device=device, compute_quality=True)
    ep_te = jce.endpoints_from_pooled_curves(
        ev_te["K"].sum(0), ev_te["U"].sum(0), ev_te["V"].sum(0), ev_te["T"].sum(0), int(ev_te["G"].sum()), cutoff
    )
    ep_cal = jce.endpoints_from_pooled_curves(
        ev_te["K"].sum(0), ev_te["U"].sum(0), ev_te["V"].sum(0), ev_te["T"].sum(0), int(ev_te["G"].sum()), tau_cal
    )
    neg_t = (ev_gt > cutoff).sum(axis=(1, 2)) == 0
    neg_fp = 100.0 * float((ev_pr[neg_t] > cutoff).mean()) if np.any(neg_t) else None
    return dict(
        Fixed_tol_F1=round(float(ep_te["Fixed_tol_F1"]), 6),
        Exact_r0_F1=round(float(ep_te["Fixed_exact_F1"]), 6),
        Cal_F1=round(float(ep_cal["Fixed_tol_F1"]), 6),
        ODS_F=round(float(ep_te["ODS_F"]), 6),
        AP=round(float(ep_te["AP"]), 6),
        SSIM=round(float(ev_te["ssims"].mean()), 6),
        PSNR_dB=round(float(ev_te["psnrs"].mean()), 4),
        mismatch_gap=round(float(ep_te["ODS_F"] - ep_te["Fixed_tol_F1"]), 6),
        neg_tile_fp_pct=round(neg_fp, 4) if neg_fp is not None else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hemit", "orion", "her2"], required=True)
    ap.add_argument("--train-pkl", default="")
    ap.add_argument("--test-pkl", default="")
    ap.add_argument("--upstream-dir", required=True)
    ap.add_argument("--cv-manifest", default="")
    ap.add_argument("--her2-root", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", default="")
    ap.add_argument("--full-loc-models", default="", help="Comma list of models to run full 5-arm localization audit (upstream,QM,E_count,E_count_zm,S_count)")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out_json = Path(args.out_json)
    up_dir = Path(args.upstream_dir)
    want = set(m.strip() for m in args.models.split(",") if m.strip()) if args.models else None
    full_loc_set = set(m.strip() for m in args.full_loc_models.split(",") if m.strip()) if args.full_loc_models else set()

    results = dict(schema="v21-p0-ecount-zm-and-loc-v1", cohort=args.cohort, models={})
    if out_json.exists():
        try:
            prev = json.loads(out_json.read_text())
            results["models"].update(prev.get("models", {}))
        except Exception:
            pass

    if args.cohort == "hemit":
        h_tr = pickle.load(open(args.train_pkl, "rb"))
        h_ev = pickle.load(open(args.test_pkl, "rb"))
        cuts = (22, 30, 18)
        ch_names = ("panCK", "CD3", "DAPI")
        has_nuclei = any(r is not None for r in h_ev.get("nuclei_records", [])) and any(r is not None for r in h_tr.get("nuclei_records", []))
        print(f"[V21-P0 HEMIT] Loaded banks tr={len(h_tr['he_nhwc'])} ev={len(h_ev['he_nhwc'])} has_nuclei={has_nuclei}", flush=True)

        for tr_f in sorted(up_dir.glob("*_train_596.npz")):
            mkey = tr_f.name.replace("_train_596.npz", "")
            ev_f = up_dir / f"{mkey}_test_945.npz"
            if not ev_f.exists():
                continue
            if want and mkey not in want:
                continue
            if mkey in results["models"]:
                print(f"[V21-P0 HEMIT] Skipping completed {mkey}", flush=True)
                continue
            tm = time.time()
            up_tr = np.load(tr_f)["pred_u8"]
            up_ev = np.load(ev_f)["pred_u8"]

            arms_to_run = ["E_count_zm"]
            if mkey in full_loc_set:
                arms_to_run = ["upstream", "QuantileMatch", "E_count", "E_count_zm", "S_count"]

            arm_tr_nchw = {a: np.zeros_like(up_tr) for a in arms_to_run}
            arm_ev_nchw = {a: np.zeros_like(up_ev) for a in arms_to_run}
            per_ch = {cn: {} for cn in ch_names}
            max_dc_by_arm = {a: 0.0 for a in arms_to_run}

            for ch, (cn, cut) in enumerate(zip(ch_names, cuts)):
                c12_tr = rhmc.build_common12_for_3ch(h_tr["he_nhwc"], up_tr, ch, device=device)
                c12_ev = rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev, ch, device=device)
                d_tr_t = torch.from_numpy(up_tr[:, ch])
                y_tr_t = torch.from_numpy(h_tr["targ_nchw"][:, ch])
                d_ev_t = torch.from_numpy(up_ev[:, ch])

                for arm in arms_to_run:
                    if arm == "upstream":
                        pt, pe, dc = up_tr[:, ch], up_ev[:, ch], 0.0
                    elif arm == "QuantileMatch":
                        pt, pe, _ = rcomp.apply_quantile_match_u8(up_tr[:, ch], h_tr["targ_nchw"][:, ch], up_ev[:, ch])
                        dc = 0.0
                    else:
                        pt, pe, dc = fit_and_predict_2seed_arm(
                            arm, c12_tr, d_tr_t, y_tr_t, c12_ev, d_ev_t, cut, device,
                            epochs=8, batch_size=8, lr=1.2e-3,
                        )
                    arm_tr_nchw[arm][:, ch] = pt
                    arm_ev_nchw[arm][:, ch] = pe
                    max_dc_by_arm[arm] = max(max_dc_by_arm[arm], dc)
                    per_ch[cn][arm] = _eval_channel_endpoints(pt, h_tr["targ_nchw"][:, ch], pe, h_ev["targ_nchw"][:, ch], cut, device)
                    if cn == "CD3":
                        per_ch[cn][arm]["psd_4band"] = rn2.compute_4band_radial_psd(pe, um_per_px=0.50)
                        per_ch[cn][arm]["localization"] = compute_radius_sweep_and_fp_localization(pe, h_ev["targ_nchw"][:, ch], cut)
                del c12_tr, c12_ev
                gc.collect()
                torch.cuda.empty_cache()

            macro = {}
            cells_out = {}
            for arm in arms_to_run:
                macro[arm] = {
                    k: round(float(np.mean([per_ch[cn][arm][k] for cn in ch_names])), 6)
                    for k in ("Fixed_tol_F1", "Exact_r0_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB", "mismatch_gap")
                }
                macro[arm]["max_abs_zero_mean"] = float(max_dc_by_arm[arm])
                macro[arm]["cd3_neg_tile_fp_pct"] = per_ch["CD3"][arm]["neg_tile_fp_pct"]
                macro[arm]["cd3_smhi"] = per_ch["CD3"][arm]["psd_4band"]["smhi_macro_to_cell_ratio"]
                if has_nuclei:
                    tr_c = rhmc.extract_pred_cell_means_for_records(arm_tr_nchw[arm], h_tr["nuclei_records"])
                    ev_c = rhmc.extract_pred_cell_means_for_records(arm_ev_nchw[arm], h_ev["nuclei_records"])
                    cell_res = rhmc.evaluate_hemit_arm_cells_and_infiltration(tr_c, ev_c, h_ev["nuclei_records"], arm_ev_nchw[arm])
                    tau_cd3_c = float(cell_res["per_marker"]["CD3"]["tau_physical"])
                    loc_c = compute_hemit_cell_localization_metrics(h_ev["nuclei_records"], arm_ev_nchw[arm], tau_cd3_c)
                    inf_d = cell_res.get("spatial_infiltration", {})
                    cells_out[arm] = dict(
                        deployable_cell_f1=round(float(cell_res["macro_deployable_cell_f1"]), 6),
                        panck_cell_f1=round(float(cell_res["per_marker"]["panCK"]["deployable_cell_f1"]), 6),
                        cd3_cell_f1=round(float(cell_res["per_marker"]["CD3"]["deployable_cell_f1"]), 6),
                        cd3_cell_auprc=round(float(cell_res["per_marker"]["CD3"]["cell_auprc"]), 6),
                        infiltration_25um_r=round(float(inf_d["within_25um_pearson_r"]), 6) if inf_d.get("within_25um_pearson_r") is not None else None,
                        infiltration_25um_mae=round(float(inf_d["within_25um_mae"]), 6) if inf_d.get("within_25um_mae") is not None else None,
                        localization=loc_c,
                    )

            results["models"][mkey] = dict(
                wall_seconds=round(time.time() - tm, 1),
                macro=macro,
                per_channel=per_ch,
                cells=cells_out,
            )
            zm = macro["E_count_zm"]
            cd3_c_s = f" CD3_cellF1={cells_out['E_count_zm']['cd3_cell_f1']:.4f}" if "E_count_zm" in cells_out else ""
            print(
                f"[V21-P0 HEMIT] {mkey:26s} ({time.time()-tm:.1f}s) | E_count_zm F1={zm['Fixed_tol_F1']:.4f} "
                f"SSIM={zm['SSIM']:.4f} SMHI={zm['cd3_smhi']:.2f} dc={zm['max_abs_zero_mean']:.1e}{cd3_c_s}",
                flush=True,
            )
            _atomic_write_json(out_json, results)

    elif args.cohort == "orion":
        ob = pickle.load(open(args.train_pkl, "rb"))
        om = json.loads(Path(args.cv_manifest).read_text())
        f0 = om["folds"][0]
        pats_arr = np.asarray(ob["patients"])
        tr_idx = np.nonzero(np.isin(pats_arr, f0["inner_train_slides"]))[0]
        ev_idx = np.nonzero(np.isin(pats_arr, f0["outer_test_slides"]))[0]
        cuts, cell_cuts_arr = ror.calibrate_orion_physical_cutoffs_from_train(ob["targ_nchw"][tr_idx], tr_idx, ob["nuclei_records"])
        cuts = [int(c) for c in cuts]
        print(f"[V21-P0 ORION] Loaded bank tr={len(tr_idx)} ev={len(ev_idx)}", flush=True)

        for fnpz in sorted(up_dir.glob("*_all40_2804.npz")):
            mkey = fnpz.name.replace("_all40_2804.npz", "")
            if want and mkey not in want:
                continue
            if mkey in results["models"]:
                print(f"[V21-P0 ORION] Skipping completed {mkey}", flush=True)
                continue
            tm = time.time()
            up_all = np.load(fnpz)["pred_u8"]
            up_tr = up_all[tr_idx]
            up_ev = up_all[ev_idx]
            he_tr = ob["he_nhwc"][tr_idx]
            he_ev = ob["he_nhwc"][ev_idx]
            gt_tr = ob["targ_nchw"][tr_idx]
            gt_ev = ob["targ_nchw"][ev_idx]

            arms_to_run = ["E_count_zm"]
            if mkey in full_loc_set:
                arms_to_run = ["upstream", "QuantileMatch", "E_count", "E_count_zm", "S_count"]

            arm_ev_nchw = {a: np.zeros_like(up_ev) for a in arms_to_run}
            per_ch = {ror.ORION_MARKERS[c]: {} for c in range(16)}
            max_dc_by_arm = {a: 0.0 for a in arms_to_run}

            for ch in range(16):
                cn = ror.ORION_MARKERS[ch]
                cut = cuts[ch]
                c12_tr = ror.build_common12_orion(he_tr, up_tr, ch, device=device)
                c12_ev = ror.build_common12_orion(he_ev, up_ev, ch, device=device)
                d_tr_t = torch.from_numpy(up_tr[:, ch])
                y_tr_t = torch.from_numpy(gt_tr[:, ch])
                d_ev_t = torch.from_numpy(up_ev[:, ch])

                for arm in arms_to_run:
                    if arm == "upstream":
                        pt, pe, dc = up_tr[:, ch], up_ev[:, ch], 0.0
                    elif arm == "QuantileMatch":
                        pt, pe, _ = rcomp.apply_quantile_match_u8(up_tr[:, ch], gt_tr[:, ch], up_ev[:, ch])
                        dc = 0.0
                    else:
                        pt, pe, dc = fit_and_predict_2seed_arm(
                            arm, c12_tr, d_tr_t, y_tr_t, c12_ev, d_ev_t, cut, device,
                            epochs=8, batch_size=16, lr=1.5e-3,
                        )
                    arm_ev_nchw[arm][:, ch] = pe
                    max_dc_by_arm[arm] = max(max_dc_by_arm[arm], dc)
                    per_ch[cn][arm] = _eval_channel_endpoints(pt, gt_tr[:, ch], pe, gt_ev[:, ch], cut, device)
                    if cn == "CD8a":
                        per_ch[cn][arm]["psd_4band"] = rn2.compute_4band_radial_psd(pe, um_per_px=0.50)
                        per_ch[cn][arm]["localization"] = compute_radius_sweep_and_fp_localization(pe, gt_ev[:, ch], cut)
                del c12_tr, c12_ev
                gc.collect()
                torch.cuda.empty_cache()

            macro = {}
            cells15 = {}
            for arm in arms_to_run:
                macro[arm] = {
                    k: round(float(np.mean([per_ch[ror.ORION_MARKERS[c]][arm][k] for c in range(16)])), 6)
                    for k in ("Fixed_tol_F1", "Exact_r0_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB", "mismatch_gap")
                }
                macro[arm]["max_abs_zero_mean"] = float(max_dc_by_arm[arm])
                macro[arm]["cd8a_smhi"] = per_ch["CD8a"][arm]["psd_4band"]["smhi_macro_to_cell_ratio"]

                # Evaluate 15-class nuclear F1 on Fold 0 held-out patients
                pr_m_list, gt_l_list = [], []
                for k_i, idx in enumerate(ev_idx.tolist()):
                    rec = ob["nuclei_records"][idx]
                    if rec is None:
                        continue
                    pr_m_list.append(ccs.extract_means_with_index(arm_ev_nchw[arm][k_i], rec["idx_spec"]))
                    gt_l_list.append(rec["labels15"])
                if pr_m_list:
                    pm_all = np.concatenate(pr_m_list, axis=0)
                    gl_all = np.concatenate(gt_l_list, axis=0)
                    f1_15 = []
                    for j in range(15):
                        m_nm = ror.ORION_MARKERS[j + 1]
                        tau_j = float(cell_cuts_arr[m_nm]) if isinstance(cell_cuts_arr, dict) else float(cell_cuts_arr[j])
                        pred_b = (pm_all[:, j + 1] > tau_j).astype(np.int32)
                        gt_b = (gl_all[:, j] > 0).astype(np.int32)
                        tp = int(((pred_b == 1) & (gt_b == 1)).sum())
                        fp = int(((pred_b == 1) & (gt_b == 0)).sum())
                        fn = int(((pred_b == 0) & (gt_b == 1)).sum())
                        f1_j = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
                        f1_15.append(float(f1_j))
                    cells15[arm] = dict(
                        macro_15class_f1=round(float(np.mean(f1_15)), 6),
                        cd8a_cell_f1=round(float(f1_15[5]), 6),  # CD8a is index 5 in ORION_CLASSES_15 (ch 6)
                        per_class_f1={ror.ORION_CLASSES_15[j]: round(f1_15[j], 5) for j in range(15)},
                    )

            results["models"][mkey] = dict(
                wall_seconds=round(time.time() - tm, 1),
                macro=macro,
                per_channel=per_ch,
                cells15=cells15,
            )
            zm = macro["E_count_zm"]
            print(
                f"[V21-P0 ORION] {mkey:20s} ({time.time()-tm:.1f}s) | E_count_zm F1={zm['Fixed_tol_F1']:.4f} "
                f"SSIM={zm['SSIM']:.4f} Cell15={cells15['E_count_zm']['macro_15class_f1']:.4f} SMHI={zm['cd8a_smhi']:.2f}",
                flush=True,
            )
            _atomic_write_json(out_json, results)

    elif args.cohort == "her2":
        tr_bank = pickle.load(open(args.train_pkl, "rb"))
        te_bank = pickle.load(open(args.test_pkl, "rb"))
        hopt_ckpt = None
        env_ckpt = os.environ.get("HOPTIMUS0_CKPT")
        cands = [
            Path(args.her2_root) / "h_optimus_0.pth",
            Path("/data2/cluster-host/topic2opus/checkpoints_sanofi/UNETR-hoptimus/h_optimus_0.pth"),
            Path("/data2/cluster-host/topic2opus/checkpoints_sanofi/MIPHEI-vit/h_optimus_0.pth"),
        ]
        if env_ckpt:
            cands.insert(0, Path(env_ckpt))
        for cand in cands:
            if cand.exists():
                hopt_ckpt = cand
                break
        for b_key in ("unet_l1_ssim", "pix2pix_cgan_ihc", "hoptimus_lora_ihc"):
            if want and b_key not in want:
                continue
            if b_key in results["models"]:
                continue
            tm = time.time()
            _, up_tr, up_ev = rher.train_or_load_her2_backbone(
                b_key, tr_bank, te_bank, 23, up_dir, device,
                hoptimus0_ckpt=hopt_ckpt, upstream_epochs=8, allow_train=False,
            )
            c12_tr = rher.build_common12_her2(tr_bank["he_nhwc"], up_tr, tr_bank["hem_u8"], device=device)
            c12_ev = rher.build_common12_her2(te_bank["he_nhwc"], up_ev, te_bank["hem_u8"], device=device)
            d_tr_t = torch.from_numpy(up_tr)
            y_tr_t = torch.from_numpy(tr_bank["dab_u8"])
            d_ev_t = torch.from_numpy(up_ev)

            arms_out = {}
            for arm in ("upstream", "QuantileMatch", "E_count", "E_count_zm", "S_count"):
                if arm == "upstream":
                    pt, pe, dc = up_tr, up_ev, 0.0
                elif arm == "QuantileMatch":
                    pt, pe, _ = rcomp.apply_quantile_match_u8(up_tr, tr_bank["dab_u8"], up_ev)
                    dc = 0.0
                else:
                    pt, pe, dc = fit_and_predict_2seed_arm(
                        arm, c12_tr, d_tr_t, y_tr_t, c12_ev, d_ev_t, 23, device,
                        epochs=8, batch_size=8, lr=1.2e-3,
                    )
                ep = _eval_channel_endpoints(pt, tr_bank["dab_u8"], pe, te_bank["dab_u8"], 23, device)
                ep["max_abs_zero_mean"] = float(dc)
                ep["localization"] = compute_radius_sweep_and_fp_localization(pe, te_bank["dab_u8"], 23)
                # Negative slides (wsi001, wsi008, wsi016) FP%
                wsis = np.asarray(te_bank["wsis"])
                neg_sl = np.isin(wsis, ["wsi001", "wsi008", "wsi016"])
                ep["neg_slides_fp_area_pct"] = round(100.0 * float((pe[neg_sl] > 23).mean()), 5) if np.any(neg_sl) else None
                arms_out[arm] = ep
            del c12_tr, c12_ev
            gc.collect()
            torch.cuda.empty_cache()

            results["models"][b_key] = dict(wall_seconds=round(time.time() - tm, 1), macro=arms_out)
            zm = arms_out["E_count_zm"]
            sc = arms_out["S_count"]
            print(
                f"[V21-P0 HER2] {b_key:18s} ({time.time()-tm:.1f}s) | E_count_zm F1={zm['Fixed_tol_F1']:.4f} (negFP={zm['neg_slides_fp_area_pct']:.3f}%) "
                f"vs S_count F1={sc['Fixed_tol_F1']:.4f} (negFP={sc['neg_slides_fp_area_pct']:.3f}%)",
                flush=True,
            )
            _atomic_write_json(out_json, results)

    print(f"V21_P0_COMPLETE cohort={args.cohort} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
