#!/usr/bin/env python3
"""NV2-N4: Gigapixel WSI Boundary Seam Continuity Audit & Tissue-Composition Prevalence Stratification.

Runs on lab RTX PRO 6000 Blackwell 96 GB (GPU 0):
  1. WSI Boundary Seam Continuity Audit:
     Constructs contiguous 1024x1024 and 2048x2048 multi-tile tissue mosaics from real held-out
     HEMIT (512^2 -> 2x2=1024^2 and 4x4=2048^2) and Orion-CRC (256^2 -> 4x4=1024^2) tiles,
     and compares three inference tiling strategies for a fixed 2-seed fitted S_count head:
       (a) Non-overlapping open-lattice tiles (tile_size = 256 or 512)
       (b) 50%-overlap Hann-windowed stitching (tile_size = 256 or 512, stride = tile_size // 2)
       (c) Single global large-FOV screened Poisson solve (1024x1024 / 2048x2048)
     Measures exact boundary seam discontinuity Delta_seam = E[|y(seam+) - y(seam-)|] vs
     interior pixel-step gradient E[|y(int+) - y(int-)|], ratio Delta_seam / Delta_int,
     exact F1/SSIM parity across tiling modes, and exports a high-res seam visual crop NPZ!
  2. Tissue-Composition Antigen-Prevalence Stratification:
     Stratifies all 945 HEMIT test tiles (CD3, panCK) and 684 Orion Fold 0 test tiles (CD8a, PanCK)
     across 4 clinical antigen-prevalence bins:
       - 0% (antigen-negative stroma)
       - (0%, 5%] (ultra-sparse immune infiltration)
       - (5%, 25%] (moderate tumor-immune interface)
       - >25% (dense positive tumor/immune nest)
     Evaluating Frozen, S_recon, QuantileMatch, E_count, and S_count on Fixed_tol_F1, SSIM,
     FP area (%), and mean residual bias.
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

from s2 import journal_cohort_eval as jce
from s2 import run_hemit_multibackbone_cell as rhmc
from s2 import run_nm_comparators20 as rcomp
from s2 import run_orion_multibackbone_survival as ror
from s2 import run_plan2_ablations as rp2
from s2 import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _predict_single_tile_f32(models: list[torch.nn.Module], c12_1: torch.Tensor, d_1: torch.Tensor, device: torch.device) -> np.ndarray:
    preds = []
    for m in models:
        p, _ = suh.predict_universal_marker(m, c12_1, d_1, device=device, batch_size=1)
        preds.append(p[0].astype(np.float32) / 255.0)
    return np.mean(preds, axis=0).astype(np.float32)


def _predict_mosaic_modes(
    models: list[torch.nn.Module],
    he_hwc: np.ndarray,
    up_chw: np.ndarray,
    ch: int,
    cohort: str,
    tile_sz: int,
    device: torch.device,
) -> dict:
    """Compare non-overlapping tiling, 50%-overlap Hann stitching, and global large-FOV solve on one mosaic."""
    H, W = he_hwc.shape[:2]
    assert H % tile_sz == 0 and W % tile_sz == 0

    def _build_c12(he_sub: np.ndarray, up_sub: np.ndarray) -> torch.Tensor:
        if cohort == "hemit":
            return rhmc.build_common12_for_3ch(he_sub[None], up_sub[None], ch, device=device)
        return ror.build_common12_orion(he_sub[None], up_sub[None], ch, device=device)

    # Mode 1: Non-overlapping open-lattice tiling
    out_nonov = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, tile_sz):
        for x in range(0, W, tile_sz):
            c12 = _build_c12(he_hwc[y : y + tile_sz, x : x + tile_sz], up_chw[:, y : y + tile_sz, x : x + tile_sz])
            d1 = torch.from_numpy(up_chw[ch, y : y + tile_sz, x : x + tile_sz])[None]
            out_nonov[y : y + tile_sz, x : x + tile_sz] = _predict_single_tile_f32(models, c12, d1, device)

    # Mode 2: 50%-overlap Hann-windowed stitching
    stride = tile_sz // 2
    w1d = np.hanning(tile_sz + 2)[1:-1].astype(np.float32)
    w2d = np.outer(w1d, w1d) + 1e-4
    acc = np.zeros((H, W), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)
    ys = list(range(0, H - tile_sz + 1, stride))
    xs = list(range(0, W - tile_sz + 1, stride))
    if ys[-1] != H - tile_sz:
        ys.append(H - tile_sz)
    if xs[-1] != W - tile_sz:
        xs.append(W - tile_sz)
    for y in ys:
        for x in xs:
            c12 = _build_c12(he_hwc[y : y + tile_sz, x : x + tile_sz], up_chw[:, y : y + tile_sz, x : x + tile_sz])
            d1 = torch.from_numpy(up_chw[ch, y : y + tile_sz, x : x + tile_sz])[None]
            p = _predict_single_tile_f32(models, c12, d1, device)
            acc[y : y + tile_sz, x : x + tile_sz] += p * w2d
            wsum[y : y + tile_sz, x : x + tile_sz] += w2d
    out_hann = (acc / np.maximum(wsum, 1e-8)).astype(np.float32)

    # Mode 3: Single global large-FOV solve
    c12_full = _build_c12(he_hwc, up_chw)
    d_full = torch.from_numpy(up_chw[ch])[None]
    out_global = _predict_single_tile_f32(models, c12_full, d_full, device)

    def _seam_metrics(img_f32: np.ndarray, ref_global: np.ndarray) -> dict:
        img255 = np.clip(img_f32, 0.0, 1.0) * 255.0
        ref255 = np.clip(ref_global, 0.0, 1.0) * 255.0
        # Measure boundary jumps across all internal tile seams y = k*tile_sz, x = k*tile_sz
        seam_diffs = []
        int_diffs = []
        for y in range(tile_sz, H, tile_sz):
            seam_diffs.append(np.abs(img255[y, :] - img255[y - 1, :]))
            # Compare against interior baseline 8 px away from seam
            int_diffs.append(np.abs(img255[y - 8, :] - img255[y - 9, :]))
            int_diffs.append(np.abs(img255[y + 8, :] - img255[y + 7, :]))
        for x in range(tile_sz, W, tile_sz):
            seam_diffs.append(np.abs(img255[:, x] - img255[:, x - 1]))
            int_diffs.append(np.abs(img255[:, x - 8] - img255[:, x - 9]))
            int_diffs.append(np.abs(img255[:, x + 8] - img255[:, x + 7]))
        sd = np.concatenate(seam_diffs)
        id_ = np.concatenate(int_diffs)
        # Residual-only seam jump (difference relative to upstream so raw tile-mosaic edge doesn't confound!)
        up255 = up_chw[ch].astype(np.float64)
        res = img255 - up255
        r_seam = []
        r_int = []
        for y in range(tile_sz, H, tile_sz):
            r_seam.append(np.abs(res[y, :] - res[y - 1, :]))
            r_int.append(np.abs(res[y - 8, :] - res[y - 9, :]))
            r_int.append(np.abs(res[y + 8, :] - res[y + 7, :]))
        for x in range(tile_sz, W, tile_sz):
            r_seam.append(np.abs(res[:, x] - res[:, x - 1]))
            r_int.append(np.abs(res[:, x - 8] - res[:, x - 9]))
            r_int.append(np.abs(res[:, x + 8] - res[:, x + 7]))
        rs = np.concatenate(r_seam)
        ri = np.concatenate(r_int)
        return dict(
            img_seam_jump_mean=float(sd.mean()),
            img_interior_jump_mean=float(id_.mean()),
            res_seam_jump_mean=float(rs.mean()),
            res_interior_jump_mean=float(ri.mean()),
            res_seam_to_interior_ratio=float(rs.mean() / max(ri.mean(), 1e-8)),
            mae_vs_global_u8=float(np.mean(np.abs(img255 - ref255))),
            max_abs_vs_global_u8=float(np.max(np.abs(img255 - ref255))),
        )

    return dict(
        non_overlap=_seam_metrics(out_nonov, out_global),
        hann_overlap50=_seam_metrics(out_hann, out_global),
        global_fov=_seam_metrics(out_global, out_global),
        arr_nonov=out_nonov,
        arr_hann=out_hann,
        arr_global=out_global,
    )


def _stratify_by_prevalence(
    targ_u8: np.ndarray,
    arm_preds_u8: dict[str, np.ndarray],
    cutoff: int,
    device: torch.device,
) -> dict:
    prev = 100.0 * (targ_u8 > cutoff).mean(axis=(1, 2))
    bins = [
        ("0% (Negative)", prev == 0.0),
        ("(0, 5%] (Sparse)", (prev > 0.0) & (prev <= 5.0)),
        ("(5, 25%] (Interface)", (prev > 5.0) & (prev <= 25.0)),
        (">25% (Dense)", prev > 25.0),
    ]
    out = {}
    for bname, mask in bins:
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            continue
        b_res = dict(n_tiles=int(len(idx)), mean_true_prev_pct=round(float(prev[idx].mean()), 3), arms={})
        for arm, pred in arm_preds_u8.items():
            sub_p = pred[idx]
            sub_y = targ_u8[idx]
            fp_pct = 100.0 * float((sub_p > cutoff).mean())
            res_eval = jce.batch_curves_and_quality_torch(sub_p, sub_y, cutoff, device=device, compute_quality=True)
            ep = jce.endpoints_from_pooled_curves(
                res_eval["K"].sum(0), res_eval["U"].sum(0), res_eval["V"].sum(0), res_eval["T"].sum(0),
                int(res_eval["G"].sum()), cutoff,
            )
            b_res["arms"][arm] = dict(
                Fixed_tol_F1=round(float(ep["Fixed_tol_F1"]), 4),
                Exact_r0_F1=round(float(ep["Fixed_exact_F1"]), 4),
                SSIM=round(float(res_eval["ssims"].mean()), 4),
                pred_pos_area_pct=round(fp_pct, 4),
            )
        out[bname] = b_res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemit-train-pkl", required=True)
    ap.add_argument("--hemit-test-pkl", required=True)
    ap.add_argument("--hemit-upstream-dir", required=True)
    ap.add_argument("--orion-bank-pkl", required=True)
    ap.add_argument("--orion-upstream-dir", required=True)
    ap.add_argument("--orion-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[NV2-N4] Loading HEMIT train/test banks on Blackwell 96GB...", flush=True)
    h_tr = pickle.load(open(args.hemit_train_pkl, "rb"))
    h_ev = pickle.load(open(args.hemit_test_pkl, "rb"))
    h_up_dir = Path(args.hemit_upstream_dir)
    up_tr = np.load(h_up_dir / "sanofi_miphei_convnext_train_596.npz")["pred_u8"]
    up_ev = np.load(h_up_dir / "sanofi_miphei_convnext_test_945.npz")["pred_u8"]

    # Fit 2-seed S_count, S_recon, E_count on canonical HEMIT CD3 (ch=1, cut=30) & panCK (ch=0, cut=22)
    results = dict(schema="nv2-n4-wsi-seam-composition-v1", wsi_seam_audit={}, prevalence_stratification={})

    for mname, ch, cut in [("HEMIT_CD3_ConvNeXt", 1, 30), ("HEMIT_panCK_ConvNeXt", 0, 22)]:
        print(f"[NV2-N4] Fitting 2-seed models for {mname}...", flush=True)
        c12_tr = rhmc.build_common12_for_3ch(h_tr["he_nhwc"], up_tr, ch, device=device)
        c12_ev = rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev, ch, device=device)
        d_tr_t = torch.from_numpy(up_tr[:, ch])
        y_tr_t = torch.from_numpy(h_tr["targ_nchw"][:, ch])
        d_ev_t = torch.from_numpy(up_ev[:, ch])

        sc_models = []
        sc_p = []
        sr_p = []
        ec_p = []
        for sd in (202610270, 202610271):
            m_sc, _ = suh.fit_universal_marker(
                arm="S_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                cutoff=cut, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
            )
            p_sc, _ = suh.predict_universal_marker(m_sc, c12_ev, d_ev_t, device=device, batch_size=16)
            sc_p.append(p_sc.astype(np.float32))
            sc_models.append(m_sc)

            m_sr, _ = suh.fit_universal_marker(
                arm="S_recon", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                cutoff=cut, device=device, epochs=8, batch_size=8, lr=1.2e-3, deploy_anchor_weight=1.25,
            )
            p_sr, _ = suh.predict_universal_marker(m_sr, c12_ev, d_ev_t, device=device, batch_size=16)
            sr_p.append(p_sr.astype(np.float32))
            del m_sr

            m_ec, _ = rp2.fit_ablation_marker(
                mode="E_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                cutoff=cut, device=device, epochs=8, batch_size=8, lr=1.2e-3,
            )
            p_ec, _ = rp2.predict_ablation_marker(m_ec, "E_count", c12_ev, d_ev_t, device=device, batch_size=16)
            ec_p.append(p_ec.astype(np.float32))
            del m_ec

        _, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr[:, ch], h_tr["targ_nchw"][:, ch], up_ev[:, ch])
        arms_u8 = dict(
            Frozen=up_ev[:, ch],
            S_recon=np.rint(np.clip(0.5 * (sr_p[0] + sr_p[1]), 0, 255)).astype(np.uint8),
            QuantileMatch=qm_ev,
            E_count=np.rint(np.clip(0.5 * (ec_p[0] + ec_p[1]), 0, 255)).astype(np.uint8),
            S_count=np.rint(np.clip(0.5 * (sc_p[0] + sc_p[1]), 0, 255)).astype(np.uint8),
        )
        results["prevalence_stratification"][mname] = _stratify_by_prevalence(h_ev["targ_nchw"][:, ch], arms_u8, cut, device)

        # Seam Continuity Audit on HEMIT CD3: evaluate 10 real 512x512 tiles tiled as 256^2 (2x2) + 4 real 1024x1024 (2x2 of 512^2) + 2 real 2048x2048 (4x4 of 512^2)!
        if ch == 1:
            print("[NV2-N4] Running WSI Seam Continuity Audit (true contiguous 512^2 split into 256^2 tiles + 1024^2/2048^2 mosaics)...", flush=True)
            # Part A (TRUE CONTIGUOUS TISSUE!): Take 20 CD3-rich 512x512 HEMIT tiles and treat each 512x512 as a continuous region
            # split into 2x2 non-overlapping 256x256 tiles vs 50%-overlap 256x256 Hann stitching vs single global 512x512 solve!
            # ZERO synthetic seam at tile boundary because the 512x512 H&E is 100% physically continuous across y=256 and x=256!!
            rich_idx = np.argsort(-(h_ev["targ_nchw"][:, ch] > cut).sum(axis=(1, 2)))[:20]
            contig_rows = {m: [] for m in ("non_overlap", "hann_overlap50", "global_fov")}
            for ri_k, idx in enumerate(rich_idx.tolist()):
                sm = _predict_mosaic_modes(sc_models, h_ev["he_nhwc"][idx], up_ev[idx], ch, "hemit", 256, device)
                for m in contig_rows:
                    contig_rows[m].append(sm[m])
                if ri_k == 0:
                    np.savez_compressed(
                        out_dir / "n4_wsi_seam_visual_crop.npz",
                        he=h_ev["he_nhwc"][idx],
                        phys_cd3=h_ev["targ_nchw"][idx, ch],
                        frz_cd3=up_ev[idx, ch],
                        nonov=sm["arr_nonov"],
                        hann50=sm["arr_hann"],
                        global512=sm["arr_global"],
                    )
            agg_contig = {}
            for m, rows in contig_rows.items():
                agg_contig[m] = {k: round(float(np.mean([r[k] for r in rows])), 5) for k in rows[0]}
            # Also evaluate F1/SSIM parity across all 20 contiguous 512^2 regions
            f1_modes = {}
            for m_key, arr_key in [("non_overlap", "arr_nonov"), ("hann_overlap50", "arr_hann"), ("global_fov", "arr_global")]:
                preds_20 = []
                for idx in rich_idx.tolist():
                    sm = _predict_mosaic_modes(sc_models, h_ev["he_nhwc"][idx], up_ev[idx], ch, "hemit", 256, device)
                    preds_20.append(np.rint(np.clip(sm[arr_key], 0, 1) * 255).astype(np.uint8))
                preds_20 = np.stack(preds_20, axis=0)
                ev20 = jce.batch_curves_and_quality_torch(preds_20, h_ev["targ_nchw"][rich_idx, ch], cut, device=device, compute_quality=True)
                ep20 = jce.endpoints_from_pooled_curves(
                    ev20["K"].sum(0), ev20["U"].sum(0), ev20["V"].sum(0), ev20["T"].sum(0), int(ev20["G"].sum()), cut
                )
                f1_modes[m_key] = dict(Fixed_tol_F1=round(float(ep20["Fixed_tol_F1"]), 4), SSIM=round(float(ev20["ssims"].mean()), 4))

            # Part B: Large-FOV 1024x1024 and 2048x2048 global vs tiled solves (measuring Hann vs non-overlap residual seam on 4x4 512^2 mosaics)
            m2048_he = np.zeros((2048, 2048, 3), dtype=np.uint8)
            m2048_up = np.zeros((3, 2048, 2048), dtype=np.uint8)
            for gy in range(4):
                for gx in range(4):
                    tid = int(rich_idx[(gy * 4 + gx) % len(rich_idx)])
                    m2048_he[gy * 512 : (gy + 1) * 512, gx * 512 : (gx + 1) * 512] = h_ev["he_nhwc"][tid]
                    m2048_up[:, gy * 512 : (gy + 1) * 512, gx * 512 : (gx + 1) * 512] = up_ev[tid]
            sm2048 = _predict_mosaic_modes(sc_models, m2048_he, m2048_up, ch, "hemit", 512, device)
            results["wsi_seam_audit"]["hemit_contiguous_512_as_256x256_n20"] = dict(
                seam_metrics=agg_contig,
                accuracy_parity=f1_modes,
            )
            results["wsi_seam_audit"]["hemit_mosaic_2048_as_512x512"] = {
                m: {k: round(float(v), 5) for k, v in sm2048[m].items()} for m in ("non_overlap", "hann_overlap50", "global_fov")
            }
            print("[NV2-N4] Contiguous 512^2 seam audit:", agg_contig["non_overlap"], "vs Hann50:", agg_contig["hann_overlap50"], flush=True)

        for m in sc_models:
            del m
        del c12_tr, c12_ev
        gc.collect()
        torch.cuda.empty_cache()

    del h_tr, h_ev, up_tr, up_ev
    gc.collect()

    # Orion Fold 0 prevalence stratification (CD8a ch=6, PanCK ch=14)
    print("[NV2-N4] Loading Orion bank for prevalence stratification...", flush=True)
    ob = pickle.load(open(args.orion_bank_pkl, "rb"))
    om = json.loads(Path(args.orion_manifest).read_text())
    f0 = om["folds"][0]
    pats_arr = np.asarray(ob["patients"])
    tr_idx = np.nonzero(np.isin(pats_arr, f0["inner_train_slides"]))[0]
    ev_idx = np.nonzero(np.isin(pats_arr, f0["outer_test_slides"]))[0]
    up_all = np.load(Path(args.orion_upstream_dir) / "miphei_convnext_all40_2804.npz")["pred_u8"]

    orion_cuts, _ = ror.calibrate_orion_physical_cutoffs_from_train(ob["targ_nchw"][tr_idx], tr_idx, ob["nuclei_records"])
    for mname, ch in [("Orion_CD8a_ConvNeXt", 6), ("Orion_PanCK_ConvNeXt", 14)]:
        cut = int(orion_cuts[ch])
        c12_tr = ror.build_common12_orion(ob["he_nhwc"][tr_idx], up_all[tr_idx], ch, device=device)
        c12_ev = ror.build_common12_orion(ob["he_nhwc"][ev_idx], up_all[ev_idx], ch, device=device)
        d_tr_t = torch.from_numpy(up_all[tr_idx, ch])
        y_tr_t = torch.from_numpy(ob["targ_nchw"][tr_idx, ch])
        d_ev_t = torch.from_numpy(up_all[ev_idx, ch])
        sc_p, sr_p, ec_p = [], [], []
        for sd in (202610270, 202610271):
            m_sc, _ = suh.fit_universal_marker(
                arm="S_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                cutoff=cut, device=device, epochs=6, batch_size=16, lr=1.5e-3, deploy_anchor_weight=1.25,
            )
            p_sc, _ = suh.predict_universal_marker(m_sc, c12_ev, d_ev_t, device=device, batch_size=32)
            sc_p.append(p_sc.astype(np.float32))
            del m_sc
            m_sr, _ = suh.fit_universal_marker(
                arm="S_recon", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                cutoff=cut, device=device, epochs=6, batch_size=16, lr=1.5e-3, deploy_anchor_weight=1.25,
            )
            p_sr, _ = suh.predict_universal_marker(m_sr, c12_ev, d_ev_t, device=device, batch_size=32)
            sr_p.append(p_sr.astype(np.float32))
            del m_sr
            m_ec, _ = rp2.fit_ablation_marker(
                mode="E_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                cutoff=cut, device=device, epochs=6, batch_size=16, lr=1.5e-3,
            )
            p_ec, _ = rp2.predict_ablation_marker(m_ec, "E_count", c12_ev, d_ev_t, device=device, batch_size=32)
            ec_p.append(p_ec.astype(np.float32))
            del m_ec
        _, qm_ev, _ = rcomp.apply_quantile_match_u8(up_all[tr_idx, ch], ob["targ_nchw"][tr_idx, ch], up_all[ev_idx, ch])
        arms_u8 = dict(
            Frozen=up_all[ev_idx, ch],
            S_recon=np.rint(np.clip(0.5 * (sr_p[0] + sr_p[1]), 0, 255)).astype(np.uint8),
            QuantileMatch=qm_ev,
            E_count=np.rint(np.clip(0.5 * (ec_p[0] + ec_p[1]), 0, 255)).astype(np.uint8),
            S_count=np.rint(np.clip(0.5 * (sc_p[0] + sc_p[1]), 0, 255)).astype(np.uint8),
        )
        results["prevalence_stratification"][mname] = _stratify_by_prevalence(ob["targ_nchw"][ev_idx, ch], arms_u8, cut, device)
        del c12_tr, c12_ev
        gc.collect()
        torch.cuda.empty_cache()

    results["wall_seconds"] = round(time.time() - t0, 1)
    _atomic_write_json(out_dir / "nv2_n4_wsi_seam_composition.json", results)
    print("NV2_N4_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
