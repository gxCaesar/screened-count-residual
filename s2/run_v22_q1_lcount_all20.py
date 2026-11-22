#!/usr/bin/env python3
"""v2.2 Q1: Gain-Matched Local Laplacian Flux Control (L_count) &
100% Single-Run Matched 6-Arm Parity across All 20 Pretrained Generators.

Runs ALL 6 arms inside ONE unified single-run receipt per generator across all 20
primary generators (11 HEMIT + 6 Orion-CRC Fold 0 + 3 HER2match):
  1. upstream      : Frozen pretrained virtual-staining backbone
  2. QuantileMatch : 256-bin training-calibrated monotone LUT
  3. E_count       : Unconstrained residual (r_c = psi_c), no zero-mean, no operator
  4. E_count_zm    : Per-tile zero-mean unscreened (r_c = psi_c - mean(psi_c))
  5. L_count       : Gain-matched pure local Laplacian flux (r_c = 0.5 * L psi_c),
                     exact algebraic 1^T r_c == 0, exact spectral range [0, 0.5], ZERO CG solve
  6. S_count       : Screened Count Residual (r_c = (I + L)^{-1} L psi_c)

Co-reports pixel endpoints (Fixed_tol_F1, Exact_r0_F1, Cal_F1, ODS_F, AP, SSIM, PSNR),
4-band radial Fourier SMHI, wall inference latency (ms/tile), exact float64 zero-mean
conservation, biological single-cell metrics (HEMIT Cellpose CD3 cell F1 + 34-tile zero-nuc FP
+ 56-tile Cellpose absent FP + 25-um infiltration r; Orion 15-class nuclear F1 + CD8a F1;
HER2match 3-negative-slide FP%), AND strict 1:1 bipartite Hungarian cell/object F1.
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
from scipy import ndimage, optimize, spatial

from s2 import clinical_cell_survival as ccs
from s2 import journal_cohort_eval as jce
from s2 import run_hemit_multibackbone_cell as rhmc
from s2 import run_her2match_dab_campaign as rher
from s2 import run_nm_comparators20 as rcomp
from s2 import run_nv2_n2_mechanistic_xai as rn2
from s2 import run_orion_multibackbone_survival as ror
from s2 import run_plan2_ablations as rp2
from s2 import run_v21_p0_ecount_zm_and_loc as rv0
from s2 import screened_universal_head as suh

ALL_SIX_ARMS = ("upstream", "QuantileMatch", "E_count", "E_count_zm", "L_count", "S_count")


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _component_centroids(binary_hw: np.ndarray, min_area: int = 8, max_objs: int = 1000) -> np.ndarray:
    lbl, k_tot = ndimage.label(binary_hw)
    if k_tot == 0:
        return np.zeros((0, 2), dtype=np.float64)
    counts = np.bincount(lbl.ravel())
    valid = np.nonzero(counts[1:] >= int(min_area))[0] + 1
    if len(valid) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(valid) > max_objs:
        order = np.argsort(counts[valid])[::-1][:max_objs]
        valid = valid[order]
    coms = ndimage.center_of_mass(lbl > 0, lbl, valid)
    return np.asarray(coms, dtype=np.float64).reshape(-1, 2)


def hungarian_match_points(
    p_xy: np.ndarray,
    g_xy: np.ndarray,
    radius: float,
) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    """Exact 1:1 bipartite maximum-cardinality / minimum-distance matching within ``radius``.

    Returns (tp, fp, fn, matched_dists, unmatched_fp_nearest_gt_dists).
    """
    n_p = len(p_xy)
    n_g = len(g_xy)
    if n_p == 0 and n_g == 0:
        return 0, 0, 0, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if n_g == 0:
        return 0, n_p, 0, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if n_p == 0:
        return 0, 0, n_g, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    D = spatial.distance.cdist(p_xy, g_xy, metric="euclidean")
    big = 1e6 + float(radius)
    cost = np.where(D <= float(radius), D, big)
    row_ind, col_ind = optimize.linear_sum_assignment(cost)
    matched_mask = D[row_ind, col_ind] <= float(radius)
    matched_rows = set(row_ind[matched_mask].tolist())
    matched_dists = D[row_ind[matched_mask], col_ind[matched_mask]]

    tp = int(matched_mask.sum())
    fp = int(n_p - tp)
    fn = int(n_g - tp)
    if fp > 0:
        unmatched_rows = [r for r in range(n_p) if r not in matched_rows]
        unmatched_nearest = D[unmatched_rows].min(axis=1)
    else:
        unmatched_nearest = np.empty(0, dtype=np.float64)
    return tp, fp, fn, matched_dists, unmatched_nearest


def compute_hungarian_object_f1(
    pred_u8: np.ndarray,
    gt_u8: np.ndarray,
    cutoff: int,
    radii: tuple[float, ...] = (5.0, 10.0),
    min_area: int = 8,
    max_tiles: int = 400,
) -> dict:
    """Strict 1:1 Hungarian connected-component object-foci F1 across tiles."""
    n = len(pred_u8)
    idx_all = list(range(n)) if n <= max_tiles else np.linspace(0, n - 1, num=max_tiles, dtype=int).tolist()
    acc = {r: dict(tp=0, fp=0, fn=0, md=[], uf=[]) for r in radii}
    for i in idx_all:
        p_xy = _component_centroids(pred_u8[i] > cutoff, min_area=min_area)
        g_xy = _component_centroids(gt_u8[i] > cutoff, min_area=min_area)
        for r in radii:
            tp, fp, fn, md, uf = hungarian_match_points(p_xy, g_xy, r)
            acc[r]["tp"] += tp
            acc[r]["fp"] += fp
            acc[r]["fn"] += fn
            if len(md):
                acc[r]["md"].append(md)
            if len(uf):
                acc[r]["uf"].append(uf)

    out = {}
    for r in radii:
        d = acc[r]
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        md_all = np.concatenate(d["md"]) if d["md"] else np.empty(0)
        uf_all = np.concatenate(d["uf"]) if d["uf"] else np.empty(0)
        rk = f"r{int(r)}"
        out[rk] = dict(
            f1=round(float(f1), 5),
            precision=round(float(prec), 5),
            recall=round(float(rec), 5),
            tp=int(tp),
            fp=int(fp),
            fn=int(fn),
            matched_median_px=round(float(np.median(md_all)), 3) if len(md_all) else None,
            unmatched_fp_gt10px_pct=round(100.0 * float((uf_all > 10.0).mean()), 2) if len(uf_all) else None,
        )
    return out


def compute_hemit_hungarian_cell_f1(
    eval_nuclei_records: list,
    pred_nchw: np.ndarray,
    tau_cd3_cell: float,
    radii: tuple[float, ...] = (5.0, 15.0),
) -> dict:
    """Strict 1:1 Hungarian matching between predicted CD3+ nuclear centroids and true CD3+ nuclear centroids."""
    acc = {r: dict(tp=0, fp=0, fn=0, md=[]) for r in radii}
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
        p_xy = rec["centroids_yx"][pr_cd3]
        g_xy = rec["centroids_yx"][gt_cd3]
        for r in radii:
            tp, fp, fn, md, _ = hungarian_match_points(p_xy, g_xy, r)
            acc[r]["tp"] += tp
            acc[r]["fp"] += fp
            acc[r]["fn"] += fn
            if len(md):
                acc[r]["md"].append(md)
    out = {}
    for r in radii:
        d = acc[r]
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        md_all = np.concatenate(d["md"]) if d["md"] else np.empty(0)
        out[f"r{int(r)}"] = dict(
            f1=round(float(f1), 5),
            tp=int(tp),
            fp=int(fp),
            fn=int(fn),
            matched_median_px=round(float(np.median(md_all)), 3) if len(md_all) else None,
        )
    return out


def _predict_chunked(
    mdl: torch.nn.Module,
    arm: str,
    build_c12_slice,
    d_u8_np: np.ndarray,
    device: torch.device,
    chunk: int = 128,
    batch_size: int = 16,
) -> tuple[np.ndarray, float, float]:
    n = len(d_u8_np)
    out = np.empty_like(d_u8_np, dtype=np.uint8)
    max_dc = 0.0
    t_tot = 0.0
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        c_sub = build_c12_slice(s, e)
        d_sub = torch.from_numpy(d_u8_np[s:e])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        if arm == "S_count":
            p_sub, de = suh.predict_universal_marker(mdl, c_sub, d_sub, device=device, batch_size=batch_size)
        else:
            p_sub, de = rp2.predict_ablation_marker(mdl, arm, c_sub, d_sub, device=device, batch_size=batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_tot += time.perf_counter() - t0
        out[s:e] = p_sub
        max_dc = max(max_dc, float(de.get("max_abs_zero_mean", de.get("residual_mean_max", 0.0))))
        del c_sub, d_sub
    return out, max_dc, 1000.0 * t_tot / max(1, n)


def fit_and_predict_2seed_arm_timed(
    arm: str,
    c12_tr: torch.Tensor,
    d_tr: torch.Tensor,
    y_tr: torch.Tensor,
    c12_ev: torch.Tensor | None,
    d_ev: torch.Tensor,
    cutoff: int,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 1.2e-3,
    ev_c12_fn=None,
    d_ev_np: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit exact 2-seed ensemble for S_count, L_count, E_count_zm, or E_count; returns (tr_ens, ev_ens, max_dc, infer_ms_per_tile)."""
    tr_preds = []
    ev_preds = []
    max_dc = 0.0
    infer_ms_list = []
    n_ev = int(d_ev.shape[0])
    for sd in (202610270, 202610271):
        if arm == "S_count":
            mdl, _ = suh.fit_universal_marker(
                arm="S_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr,
                target_u8_train=y_tr, cutoff=cutoff, device=device,
                epochs=epochs, batch_size=batch_size, lr=lr, deploy_anchor_weight=1.25,
            )
            pt, _ = suh.predict_universal_marker(mdl, c12_tr, d_tr, device=device, batch_size=max(8, batch_size))
        else:
            mdl, _ = rp2.fit_ablation_marker(
                mode=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr,
                target_u8_train=y_tr, cutoff=cutoff, device=device,
                epochs=epochs, batch_size=batch_size, lr=lr, deploy_anchor_weight=1.25,
            )
            pt, _ = rp2.predict_ablation_marker(mdl, arm, c12_tr, d_tr, device=device, batch_size=max(8, batch_size))
        if ev_c12_fn is not None and d_ev_np is not None:
            pe, dc_s, ms_s = _predict_chunked(mdl, arm, ev_c12_fn, d_ev_np, device, chunk=96, batch_size=max(8, batch_size))
            max_dc = max(max_dc, dc_s)
            infer_ms_list.append(ms_s)
        else:
            assert c12_ev is not None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_inf = time.perf_counter()
            if arm == "S_count":
                pe, de = suh.predict_universal_marker(mdl, c12_ev, d_ev, device=device, batch_size=max(8, batch_size))
            else:
                pe, de = rp2.predict_ablation_marker(mdl, arm, c12_ev, d_ev, device=device, batch_size=max(8, batch_size))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_ms_list.append(1000.0 * (time.perf_counter() - t_inf) / max(1, n_ev))
            max_dc = max(max_dc, float(de.get("max_abs_zero_mean", de.get("residual_mean_max", 0.0))))
        tr_preds.append(pt.astype(np.float32))
        ev_preds.append(pe.astype(np.float32))
        del mdl
        if device.type == "cuda":
            torch.cuda.empty_cache()
    tr_ens = np.rint(np.clip(0.5 * (tr_preds[0] + tr_preds[1]), 0.0, 255.0)).astype(np.uint8)
    ev_ens = np.rint(np.clip(0.5 * (ev_preds[0] + ev_preds[1]), 0.0, 255.0)).astype(np.uint8)
    return tr_ens, ev_ens, max_dc, float(np.mean(infer_ms_list))


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
    ap.add_argument("--batch-size", type=int, default=0, help="Override batch size (0 = default 8 HEMIT/HER2, 16 Orion)")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out_json = Path(args.out_json)
    up_dir = Path(args.upstream_dir)
    want = set(m.strip() for m in args.models.split(",") if m.strip()) if args.models else None

    results = dict(schema="v22-q1-lcount-single-run-6arm-v1", cohort=args.cohort, arms=list(ALL_SIX_ARMS), models={})
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
        bs = args.batch_size if args.batch_size > 0 else 8
        print(f"[V22-Q1 HEMIT] Loaded tr={len(h_tr['he_nhwc'])} ev={len(h_ev['he_nhwc'])} has_nuclei={has_nuclei} bs={bs}", flush=True)

        for tr_f in sorted(up_dir.glob("*_train_596.npz")):
            mkey = tr_f.name.replace("_train_596.npz", "")
            ev_f = up_dir / f"{mkey}_test_945.npz"
            if not ev_f.exists():
                continue
            if want and mkey not in want:
                continue
            if mkey in results["models"]:
                print(f"[V22-Q1 HEMIT] Skipping completed {mkey}", flush=True)
                continue
            tm = time.time()
            up_tr = np.load(tr_f)["pred_u8"]
            up_ev = np.load(ev_f)["pred_u8"]

            per_ch = {cn: {} for cn in ch_names}
            macro = {}
            cells_out = {}

            # Arm-by-arm low-RAM loop: holds only 1 arm's 3-ch prediction cube + float16 train c12 + 96-tile chunked eval c12
            for arm in ALL_SIX_ARMS:
                tr_cube = np.zeros_like(up_tr)
                ev_cube = np.zeros_like(up_ev)
                max_dc = 0.0
                ms_list = []
                for ch, (cn, cut) in enumerate(zip(ch_names, cuts)):
                    if arm == "upstream":
                        pt, pe, dc, ms = up_tr[:, ch], up_ev[:, ch], 0.0, 0.0
                    elif arm == "QuantileMatch":
                        t_qm = time.perf_counter()
                        pt, pe, _ = rcomp.apply_quantile_match_u8(up_tr[:, ch], h_tr["targ_nchw"][:, ch], up_ev[:, ch])
                        ms = 1000.0 * (time.perf_counter() - t_qm) / max(1, len(up_ev))
                        dc = 0.0
                    else:
                        c12_tr = rhmc.build_common12_for_3ch(h_tr["he_nhwc"], up_tr, ch, device=device).to(torch.float16)
                        d_tr_t = torch.from_numpy(up_tr[:, ch])
                        y_tr_t = torch.from_numpy(h_tr["targ_nchw"][:, ch])
                        d_ev_t = torch.from_numpy(up_ev[:, ch])
                        ev_fn = lambda s, e, _ch=ch: rhmc.build_common12_for_3ch(h_ev["he_nhwc"][s:e], up_ev[s:e], _ch, device=device)
                        pt, pe, dc, ms = fit_and_predict_2seed_arm_timed(
                            arm, c12_tr, d_tr_t, y_tr_t, None, d_ev_t, cut, device,
                            epochs=8, batch_size=bs, lr=1.2e-3,
                            ev_c12_fn=ev_fn, d_ev_np=up_ev[:, ch],
                        )
                        del c12_tr
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    tr_cube[:, ch] = pt
                    ev_cube[:, ch] = pe
                    max_dc = max(max_dc, dc)
                    ms_list.append(ms)
                    per_ch[cn][arm] = rv0._eval_channel_endpoints(pt, h_tr["targ_nchw"][:, ch], pe, h_ev["targ_nchw"][:, ch], cut, device)
                    if cn == "CD3":
                        per_ch[cn][arm]["psd_4band"] = rn2.compute_4band_radial_psd(pe, um_per_px=0.50)
                        per_ch[cn][arm]["localization"] = rv0.compute_radius_sweep_and_fp_localization(pe, h_ev["targ_nchw"][:, ch], cut)
                        per_ch[cn][arm]["hungarian_obj"] = compute_hungarian_object_f1(pe, h_ev["targ_nchw"][:, ch], cut, radii=(5.0, 10.0), max_tiles=300)

                macro[arm] = {
                    k: round(float(np.mean([per_ch[cn][arm][k] for cn in ch_names])), 6)
                    for k in ("Fixed_tol_F1", "Exact_r0_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB", "mismatch_gap")
                }
                macro[arm]["max_abs_zero_mean"] = float(max_dc)
                macro[arm]["infer_ms_per_tile_per_ch"] = round(float(np.mean(ms_list)), 4)
                macro[arm]["cd3_neg_tile_fp_pct"] = per_ch["CD3"][arm]["neg_tile_fp_pct"]
                macro[arm]["cd3_smhi"] = per_ch["CD3"][arm]["psd_4band"]["smhi_macro_to_cell_ratio"]
                macro[arm]["cd3_hungarian_obj_f1_5px"] = per_ch["CD3"][arm]["hungarian_obj"]["r5"]["f1"]
                if has_nuclei:
                    tr_c = rhmc.extract_pred_cell_means_for_records(tr_cube, h_tr["nuclei_records"])
                    ev_c = rhmc.extract_pred_cell_means_for_records(ev_cube, h_ev["nuclei_records"])
                    cell_res = rhmc.evaluate_hemit_arm_cells_and_infiltration(tr_c, ev_c, h_ev["nuclei_records"], ev_cube)
                    tau_cd3_c = float(cell_res["per_marker"]["CD3"]["tau_physical"])
                    loc_c = rv0.compute_hemit_cell_localization_metrics(h_ev["nuclei_records"], ev_cube, tau_cd3_c)
                    hung_c = compute_hemit_hungarian_cell_f1(h_ev["nuclei_records"], ev_cube, tau_cd3_c)
                    inf_d = cell_res.get("spatial_infiltration", {})
                    cells_out[arm] = dict(
                        deployable_cell_f1=round(float(cell_res["macro_deployable_cell_f1"]), 6),
                        panck_cell_f1=round(float(cell_res["per_marker"]["panCK"]["deployable_cell_f1"]), 6),
                        cd3_cell_f1=round(float(cell_res["per_marker"]["CD3"]["deployable_cell_f1"]), 6),
                        cd3_cell_auprc=round(float(cell_res["per_marker"]["CD3"]["cell_auprc"]), 6),
                        infiltration_25um_r=round(float(inf_d["within_25um_pearson_r"]), 6) if inf_d.get("within_25um_pearson_r") is not None else None,
                        infiltration_25um_mae=round(float(inf_d["within_25um_mae"]), 6) if inf_d.get("within_25um_mae") is not None else None,
                        localization=loc_c,
                        hungarian_cd3_cells=hung_c,
                    )
                del tr_cube, ev_cube
                gc.collect()

            results["models"][mkey] = dict(
                single_run_matched=True,
                wall_seconds=round(time.time() - tm, 1),
                macro=macro,
                per_channel=per_ch,
                cells=cells_out,
            )
            zm, lc, sc = macro["E_count_zm"], macro["L_count"], macro["S_count"]
            print(
                f"[V22-Q1 HEMIT] {mkey:26s} ({time.time()-tm:.1f}s) | "
                f"F1: ZM={zm['Fixed_tol_F1']:.4f} LC={lc['Fixed_tol_F1']:.4f} SC={sc['Fixed_tol_F1']:.4f} | "
                f"SSIM: ZM={zm['SSIM']:.4f} LC={lc['SSIM']:.4f} SC={sc['SSIM']:.4f} | "
                f"SMHI: ZM={zm['cd3_smhi']:.2f} LC={lc['cd3_smhi']:.2f} SC={sc['cd3_smhi']:.2f} | "
                f"dc(LC)={lc['max_abs_zero_mean']:.1e}",
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
        bs = args.batch_size if args.batch_size > 0 else 16
        print(f"[V22-Q1 ORION] Loaded bank tr={len(tr_idx)} ev={len(ev_idx)} bs={bs}", flush=True)

        for fnpz in sorted(up_dir.glob("*_all40_2804.npz")):
            mkey = fnpz.name.replace("_all40_2804.npz", "")
            if want and mkey not in want:
                continue
            if mkey in results["models"]:
                print(f"[V22-Q1 ORION] Skipping completed {mkey}", flush=True)
                continue
            tm = time.time()
            up_all = np.load(fnpz)["pred_u8"]
            up_tr, up_ev = up_all[tr_idx], up_all[ev_idx]
            he_tr, he_ev = ob["he_nhwc"][tr_idx], ob["he_nhwc"][ev_idx]
            gt_tr, gt_ev = ob["targ_nchw"][tr_idx], ob["targ_nchw"][ev_idx]

            arm_ev_nchw = {a: np.zeros_like(up_ev) for a in ALL_SIX_ARMS}
            per_ch = {ror.ORION_MARKERS[c]: {} for c in range(16)}
            max_dc_by_arm = {a: 0.0 for a in ALL_SIX_ARMS}
            ms_by_arm = {a: [] for a in ALL_SIX_ARMS}

            for ch in range(16):
                cn = ror.ORION_MARKERS[ch]
                cut = cuts[ch]
                c12_tr = ror.build_common12_orion(he_tr, up_tr, ch, device=device).to(torch.float16)
                c12_ev = ror.build_common12_orion(he_ev, up_ev, ch, device=device).to(torch.float16)
                d_tr_t = torch.from_numpy(up_tr[:, ch])
                y_tr_t = torch.from_numpy(gt_tr[:, ch])
                d_ev_t = torch.from_numpy(up_ev[:, ch])

                for arm in ALL_SIX_ARMS:
                    if arm == "upstream":
                        pt, pe, dc, ms = up_tr[:, ch], up_ev[:, ch], 0.0, 0.0
                    elif arm == "QuantileMatch":
                        t_qm = time.perf_counter()
                        pt, pe, _ = rcomp.apply_quantile_match_u8(up_tr[:, ch], gt_tr[:, ch], up_ev[:, ch])
                        ms = 1000.0 * (time.perf_counter() - t_qm) / max(1, len(up_ev))
                        dc = 0.0
                    else:
                        pt, pe, dc, ms = fit_and_predict_2seed_arm_timed(
                            arm, c12_tr, d_tr_t, y_tr_t, c12_ev, d_ev_t, cut, device,
                            epochs=8, batch_size=bs, lr=1.5e-3,
                        )
                    arm_ev_nchw[arm][:, ch] = pe
                    max_dc_by_arm[arm] = max(max_dc_by_arm[arm], dc)
                    ms_by_arm[arm].append(ms)
                    per_ch[cn][arm] = rv0._eval_channel_endpoints(pt, gt_tr[:, ch], pe, gt_ev[:, ch], cut, device)
                    if cn == "CD8a":
                        per_ch[cn][arm]["psd_4band"] = rn2.compute_4band_radial_psd(pe, um_per_px=0.50)
                        per_ch[cn][arm]["localization"] = rv0.compute_radius_sweep_and_fp_localization(pe, gt_ev[:, ch], cut)
                        per_ch[cn][arm]["hungarian_obj"] = compute_hungarian_object_f1(pe, gt_ev[:, ch], cut, radii=(5.0, 10.0), max_tiles=300)
                del c12_tr, c12_ev
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            macro = {}
            cells15 = {}
            for arm in ALL_SIX_ARMS:
                macro[arm] = {
                    k: round(float(np.mean([per_ch[ror.ORION_MARKERS[c]][arm][k] for c in range(16)])), 6)
                    for k in ("Fixed_tol_F1", "Exact_r0_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB", "mismatch_gap")
                }
                macro[arm]["max_abs_zero_mean"] = float(max_dc_by_arm[arm])
                macro[arm]["infer_ms_per_tile_per_ch"] = round(float(np.mean(ms_by_arm[arm])), 4)
                macro[arm]["cd8a_smhi"] = per_ch["CD8a"][arm]["psd_4band"]["smhi_macro_to_cell_ratio"]
                macro[arm]["cd8a_hungarian_obj_f1_5px"] = per_ch["CD8a"][arm]["hungarian_obj"]["r5"]["f1"]

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
                        cd8a_cell_f1=round(float(f1_15[5]), 6),
                        per_class_f1={ror.ORION_CLASSES_15[j]: round(f1_15[j], 5) for j in range(15)},
                    )

            results["models"][mkey] = dict(
                single_run_matched=True,
                wall_seconds=round(time.time() - tm, 1),
                macro=macro,
                per_channel=per_ch,
                cells15=cells15,
            )
            zm, lc, sc = macro["E_count_zm"], macro["L_count"], macro["S_count"]
            print(
                f"[V22-Q1 ORION] {mkey:20s} ({time.time()-tm:.1f}s) | "
                f"F1: ZM={zm['Fixed_tol_F1']:.4f} LC={lc['Fixed_tol_F1']:.4f} SC={sc['Fixed_tol_F1']:.4f} | "
                f"Cell15: ZM={cells15['E_count_zm']['macro_15class_f1']:.4f} LC={cells15['L_count']['macro_15class_f1']:.4f} SC={cells15['S_count']['macro_15class_f1']:.4f} | "
                f"SSIM: ZM={zm['SSIM']:.4f} LC={lc['SSIM']:.4f} SC={sc['SSIM']:.4f}",
                flush=True,
            )
            _atomic_write_json(out_json, results)

    elif args.cohort == "her2":
        tr_bank = pickle.load(open(args.train_pkl, "rb"))
        te_bank = pickle.load(open(args.test_pkl, "rb"))
        bs = args.batch_size if args.batch_size > 0 else 8
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
            c12_tr = rher.build_common12_her2(tr_bank["he_nhwc"], up_tr, tr_bank["hem_u8"], device=device).to(torch.float16)
            ev_fn_h2 = lambda s, e: rher.build_common12_her2(te_bank["he_nhwc"][s:e], up_ev[s:e], te_bank["hem_u8"][s:e], device=device)
            d_tr_t = torch.from_numpy(up_tr)
            y_tr_t = torch.from_numpy(tr_bank["dab_u8"])
            d_ev_t = torch.from_numpy(up_ev)

            arms_out = {}
            wsis = np.asarray(te_bank["wsis"])
            neg_sl = np.isin(wsis, ["wsi001", "wsi008", "wsi016"])
            for arm in ALL_SIX_ARMS:
                if arm == "upstream":
                    pt, pe, dc, ms = up_tr, up_ev, 0.0, 0.0
                elif arm == "QuantileMatch":
                    t_qm = time.perf_counter()
                    pt, pe, _ = rcomp.apply_quantile_match_u8(up_tr, tr_bank["dab_u8"], up_ev)
                    ms = 1000.0 * (time.perf_counter() - t_qm) / max(1, len(up_ev))
                    dc = 0.0
                else:
                    pt, pe, dc, ms = fit_and_predict_2seed_arm_timed(
                        arm, c12_tr, d_tr_t, y_tr_t, None, d_ev_t, 23, device,
                        epochs=8, batch_size=bs, lr=1.2e-3,
                        ev_c12_fn=ev_fn_h2, d_ev_np=up_ev,
                    )
                ep = rv0._eval_channel_endpoints(pt, tr_bank["dab_u8"], pe, te_bank["dab_u8"], 23, device)
                ep["max_abs_zero_mean"] = float(dc)
                ep["infer_ms_per_tile_per_ch"] = round(float(ms), 4)
                ep["psd_4band"] = rn2.compute_4band_radial_psd(pe, um_per_px=0.50)
                ep["dab_smhi"] = ep["psd_4band"]["smhi_macro_to_cell_ratio"]
                ep["localization"] = rv0.compute_radius_sweep_and_fp_localization(pe, te_bank["dab_u8"], 23)
                ep["hungarian_obj"] = compute_hungarian_object_f1(pe, te_bank["dab_u8"], 23, radii=(5.0, 10.0), max_tiles=300)
                ep["neg_slides_fp_area_pct"] = round(100.0 * float((pe[neg_sl] > 23).mean()), 5) if np.any(neg_sl) else None
                arms_out[arm] = ep
            del c12_tr
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            results["models"][b_key] = dict(single_run_matched=True, wall_seconds=round(time.time() - tm, 1), macro=arms_out)
            zm, lc, sc = arms_out["E_count_zm"], arms_out["L_count"], arms_out["S_count"]
            print(
                f"[V22-Q1 HER2] {b_key:18s} ({time.time()-tm:.1f}s) | "
                f"F1: ZM={zm['Fixed_tol_F1']:.4f} LC={lc['Fixed_tol_F1']:.4f} SC={sc['Fixed_tol_F1']:.4f} | "
                f"negFP: ZM={zm['neg_slides_fp_area_pct']:.3f}% LC={lc['neg_slides_fp_area_pct']:.3f}% SC={sc['neg_slides_fp_area_pct']:.3f}%",
                flush=True,
            )
            _atomic_write_json(out_json, results)

    print(f"V22_Q1_COMPLETE cohort={args.cohort} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
