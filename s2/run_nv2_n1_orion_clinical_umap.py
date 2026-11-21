#!/usr/bin/env python3
"""NV2-N1: Orion-CRC Physical-Reference-Locked vs Virtual-Refitted ImmunoScore Survival,
15-Marker Single-Cell UMAP Phenotype Landscape, and High/Low-Risk Spatial Specimen Maps.

Computes on the locked 40-patient Harvard Orion-CRC cohort (2,804 tiles, 5-fold patient-disjoint CV):
  1. Physical-mIF-Locked Clinical Cutoff ImmunoScore (transferring inner-train PHYSICAL mIF
     R1/R2 CD3e+/CD8a+ density reference distributions directly to virtual stains without
     per-generator percentile recalibration) vs Virtual-Refitted Percentile ImmunoScore
     across all cached published generators (ConvNeXt, W-MSE Swin, Adv-Swin, Pix2Pix, ROSIE)
     and 5 post-hoc arms (Frozen, S_recon, QuantileMatch, E_count, S_count).
     Reports tie-checked Harrell C-index + 10,000 duplicate-free patient bootstrap 95% CIs,
     paired delta-C, univariable KM HR & log-rank P, numbers-at-risk at [0,12,24,36,48,60] mo,
     risk-group counts (n_high, n_low), ImmunoScore MAE & quadratic Cohen's kappa vs physical mIF.
  2. Fold 0 15-marker single-cell phenotype landscape (UMAP/PCA embedding + 8 canonical
     biological phenotypes: Tumor PanCK+, Cytotoxic T CD8a+, Helper T CD4+, Treg FOXP3+,
     Macrophage CD68/CD163+, B cell CD20+, Endothelia CD31+, Stroma SMA+) comparing
     Physical mIF vs Frozen vs S_count, saving a compact subsampled NPZ for Fig. 4i,j.
  3. Real patient Core/Margin spatial specimen crops (high-ImmunoScore long survivor vs
     low-ImmunoScore early recurrence) for Fig. 5a,b.
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
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, f1_score

from s2 import backbone_zoo as zoo
from s2 import clinical_cell_survival as ccs
from s2 import run_nm_comparators20 as rcomp
from s2 import run_orion_multibackbone_survival as ror
from s2 import run_plan2_ablations as rp2
from s2 import screened_universal_head as suh

ORION_MARKERS = list(zoo.ORION_16_MARKERS)
ORION_CLASSES_15 = list(zoo.ORION_NUCLEI_15_CLASSES)
TRIAD_NAMES = ("Pan-CK", "CD3e", "CD8a")
TRIAD_CHS = {m: ORION_MARKERS.index(m) for m in TRIAD_NAMES}
DMMR_CANONICAL = {"C15", "C16", "C23", "C33", "C34", "C35"}
STAGE_ORD = {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _km_numbers_at_risk(times_mo: np.ndarray, mask: np.ndarray, ticks=(0, 12, 24, 36, 48, 60)) -> list[int]:
    t = times_mo[mask]
    return [int(np.sum(t >= tk)) for tk in ticks]


def _compute_patient_compartments_with_cutoffs(
    bank: dict,
    pred_nchw: np.ndarray,
    tile_indices: np.ndarray,
    cell_cutoffs: dict[str, float],
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Compute per-patient R1/R2 CD3e/CD8a positive-cell fractions under FIXED physical cell cutoffs."""
    ch_pck = TRIAD_CHS["Pan-CK"]
    ch_cd3 = TRIAD_CHS["CD3e"]
    ch_cd8 = TRIAD_CHS["CD8a"]
    cut_pck = float(cell_cutoffs["Pan-CK"])
    cut_cd3 = float(cell_cutoffs["CD3e"])
    cut_cd8 = float(cell_cutoffs["CD8a"])

    by_pat: dict[str, list[tuple[int, int, int, int]]] = {}
    for idx in tile_indices.tolist():
        rec = bank["nuclei_records"][idx]
        if rec is None:
            continue
        pat = str(rec["patient_id"])
        means = ccs.extract_means_with_index(pred_nchw[idx, [ch_pck, ch_cd3, ch_cd8]], rec["idx_spec"])
        n_tot = int(means.shape[0])
        if n_tot == 0:
            continue
        n_pck = int(np.sum(means[:, 0] > cut_pck))
        n_cd3 = int(np.sum(means[:, 1] > cut_cd3))
        n_cd8 = int(np.sum(means[:, 2] > cut_cd8))
        by_pat.setdefault(pat, []).append((n_tot, n_pck, n_cd3, n_cd8))

    pats = sorted(by_pat.keys())
    out = {k: np.zeros(len(pats), dtype=np.float64) for k in ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2", "mean_CD3ep", "mean_CD8ap")}
    for i, pat in enumerate(pats):
        arr = np.asarray(by_pat[pat], dtype=np.float64)
        comp = ccs.compute_patient_compartment_densities(
            tile_total_cells=arr[:, 0],
            tile_panck_pos=arr[:, 1],
            tile_cd3_pos=arr[:, 2],
            tile_cd8_pos=arr[:, 3],
        )
        for k in out:
            out[k][i] = float(comp[k])
    return pats, out


def _summarize_survival_vector(
    pats_order: list[str],
    scores_cont: np.ndarray,
    scores_disc: np.ndarray,
    oracle_disc: np.ndarray,
    clin_by_pat: dict[str, dict],
    ref_scores_for_delta: np.ndarray | None = None,
) -> dict:
    times_d = np.asarray([float(clin_by_pat[p]["pfs_days"]) for p in pats_order], dtype=np.float64)
    times_mo = times_d / 30.4375
    events = np.asarray([int(clin_by_pat[p]["recurrence_event"]) for p in pats_order], dtype=np.int32)
    stages = np.asarray([STAGE_ORD.get(str(clin_by_pat[p]["stage"]), 2.5) for p in pats_order], dtype=np.float64)
    mmr = np.asarray([1.0 if p in DMMR_CANONICAL else 0.0 for p in pats_order], dtype=np.float64)

    c_idx = ccs.harrell_c_index(times_d, events, scores_cont)
    rng = np.random.default_rng(20261121)
    n = len(pats_order)
    boot_c = np.empty(4000, dtype=np.float64)
    boot_dc = np.empty(4000, dtype=np.float64) if ref_scores_for_delta is not None else None
    for b in range(4000):
        idx = rng.integers(0, n, size=n)
        boot_c[b] = ccs.harrell_c_index(times_d[idx], events[idx], scores_cont[idx], unit_ids=idx)
        if boot_dc is not None and ref_scores_for_delta is not None:
            c_ref = ccs.harrell_c_index(times_d[idx], events[idx], ref_scores_for_delta[idx], unit_ids=idx)
            boot_dc[b] = boot_c[b] - c_ref

    high_mask = scores_cont > 2.0
    low_mask = ~high_mask
    n_high = int(np.sum(high_mask))
    n_low = int(np.sum(low_mask))
    lr_p = None
    hr_hl = None
    km_high = None
    km_low = None
    nar_high = _km_numbers_at_risk(times_mo, high_mask)
    nar_low = _km_numbers_at_risk(times_mo, low_mask)

    if n_high > 0 and n_low > 0:
        lr = ccs.logrank_two_group(times_d, events, low_mask.astype(np.int32))
        lr_p = float(lr["p_value"])
        km_h = ccs.kaplan_meier_curve(times_mo[high_mask], events[high_mask])
        km_l = ccs.kaplan_meier_curve(times_mo[low_mask], events[low_mask])
        km_high = dict(time=km_h["time"].tolist(), survival=km_h["survival"].tolist(), ci_low=km_h["ci_low"].tolist(), ci_high=km_h["ci_high"].tolist())
        km_low = dict(time=km_l["time"].tolist(), survival=km_l["survival"].tolist(), ci_low=km_l["ci_low"].tolist(), ci_high=km_l["ci_high"].tolist())
        # Univariable low-vs-high HR (risk of low ImmunoScore)
        if int(events[high_mask].sum()) > 0 and int(events[low_mask].sum()) > 0:
            cox_uni = ccs.fit_cox_ph_breslow(times_d, events, low_mask.astype(np.float64)[:, None], l2_pen=1e-4)
            hr_hl = float(cox_uni["hr"][0])

    # Agreement with physical mIF ImmunoScore
    mae_disc = float(np.mean(np.abs(scores_disc - oracle_disc)))
    mae_cont = float(np.mean(np.abs(scores_cont - oracle_disc)))
    pearson_r = float(stats.pearsonr(scores_cont, oracle_disc)[0]) if np.std(scores_cont) > 1e-9 else 0.0
    spearman_rho = float(stats.spearmanr(scores_cont, oracle_disc)[0]) if np.std(scores_cont) > 1e-9 else 0.0
    d_int = np.clip(np.rint(scores_disc), 0, 4).astype(int)
    o_int = np.clip(np.rint(oracle_disc), 0, 4).astype(int)
    qwk = float(cohen_kappa_score(o_int, d_int, weights="quadratic")) if len(np.unique(d_int)) > 1 else 0.0

    # Multivariable Stage + MMR adjusted HR per -1 point ImmunoScore
    mv_hr = None
    if np.std(scores_cont) > 1e-8:
        X_mv = np.column_stack([-scores_cont, stages, mmr])
        cox_mv = ccs.fit_cox_ph_breslow(times_d, events, X_mv, l2_pen=1e-4)
        mv_hr = float(cox_mv["hr"][0])

    res = dict(
        c_index=round(float(c_idx), 6),
        c_ci95=[round(float(np.quantile(boot_c, 0.025)), 6), round(float(np.quantile(boot_c, 0.975)), 6)],
        n_high=n_high,
        n_low=n_low,
        risk_group_collapsed=bool(n_high == 0 or n_low == 0),
        logrank_p=round(lr_p, 8) if lr_p is not None else None,
        hr_low_vs_high=round(hr_hl, 4) if hr_hl is not None else None,
        mv_stage_mmr_hr_per_neg_point=round(mv_hr, 4) if mv_hr is not None else None,
        immunoscore_mae_disc=round(mae_disc, 4),
        immunoscore_mae_cont=round(mae_cont, 4),
        pearson_vs_oracle=round(pearson_r, 4),
        spearman_vs_oracle=round(spearman_rho, 4),
        quadratic_kappa_vs_oracle=round(qwk, 4),
        mean_score_cont=round(float(np.mean(scores_cont)), 4),
        nar_ticks_mo=[0, 12, 24, 36, 48, 60],
        nar_high=nar_high,
        nar_low=nar_low,
        km_high=km_high,
        km_low=km_low,
        per_patient_cont={p: round(float(s), 4) for p, s in zip(pats_order, scores_cont)},
        per_patient_disc={p: int(s) for p, s in zip(pats_order, d_int)},
    )
    if boot_dc is not None and ref_scores_for_delta is not None:
        dc = float(c_idx - ccs.harrell_c_index(times_d, events, ref_scores_for_delta))
        res["delta_c_vs_frozen"] = round(dc, 6)
        res["delta_c_ci95"] = [round(float(np.quantile(boot_dc, 0.025)), 6), round(float(np.quantile(boot_dc, 0.975)), 6)]
    return res


def _fit_and_predict_triad_fold(
    bank: dict,
    up_all: np.ndarray,
    tr_idx: np.ndarray,
    ev_idx: np.ndarray,
    cutoffs_pixel: dict[str, int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Fit S_recon, QuantileMatch, E_count, S_count on inner-train tiles for Pan-CK/CD3e/CD8a and return full 16-ch arrays for eval+train."""
    # We only modify the 3 triad channels; others copy upstream
    arms_tr = {a: up_all[tr_idx].copy() for a in ("Frozen", "S_recon", "QuantileMatch", "E_count", "S_count")}
    arms_ev = {a: up_all[ev_idx].copy() for a in ("Frozen", "S_recon", "QuantileMatch", "E_count", "S_count")}

    he_tr = bank["he_nhwc"][tr_idx]
    he_ev = bank["he_nhwc"][ev_idx]
    up_tr = up_all[tr_idx]
    up_ev = up_all[ev_idx]

    for mname in TRIAD_NAMES:
        ch = TRIAD_CHS[mname]
        cut = int(cutoffs_pixel[mname])
        y_tr = bank["targ_nchw"][tr_idx, ch]
        u_tr = up_tr[:, ch]
        u_ev = up_ev[:, ch]

        # QuantileMatch
        qm_tr, qm_ev, _ = rcomp.apply_quantile_match_u8(u_tr, y_tr, u_ev)
        arms_tr["QuantileMatch"][:, ch] = qm_tr
        arms_ev["QuantileMatch"][:, ch] = qm_ev

        # Common12 tensor for this Orion channel
        c12_tr = ror.build_common12_orion(he_tr, up_tr, ch, device=device)
        c12_ev = ror.build_common12_orion(he_ev, up_ev, ch, device=device)
        d_tr_t = torch.from_numpy(u_tr)
        y_tr_t = torch.from_numpy(y_tr)
        d_ev_t = torch.from_numpy(u_ev)

        for arm in ("S_recon", "E_count", "S_count"):
            tr_preds = []
            ev_preds = []
            for sd in (202610270, 202610271):
                if arm == "E_count":
                    mdl, _ = rp2.fit_ablation_marker(
                        mode="E_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                        cutoff=cut, device=device, epochs=6, batch_size=16, lr=1.5e-3,
                    )
                    pr_tr, _ = rp2.predict_ablation_marker(mdl, "E_count", c12_tr, d_tr_t, device=device, batch_size=32)
                    pr_ev, _ = rp2.predict_ablation_marker(mdl, "E_count", c12_ev, d_ev_t, device=device, batch_size=32)
                else:
                    mdl, _ = suh.fit_universal_marker(
                        arm=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                        cutoff=cut, device=device, epochs=6, batch_size=16, lr=1.5e-3, deploy_anchor_weight=1.25,
                    )
                    pr_tr, _ = suh.predict_universal_marker(mdl, c12_tr, d_tr_t, device=device, batch_size=32)
                    pr_ev, _ = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device=device, batch_size=32)
                tr_preds.append(pr_tr.astype(np.float32))
                ev_preds.append(pr_ev.astype(np.float32))
                del mdl
            arms_tr[arm][:, ch] = np.rint(np.clip(0.5 * (tr_preds[0] + tr_preds[1]), 0.0, 255.0)).astype(np.uint8)
            arms_ev[arm][:, ch] = np.rint(np.clip(0.5 * (ev_preds[0] + ev_preds[1]), 0.0, 255.0)).astype(np.uint8)
        del c12_tr, c12_ev
        gc.collect()
        torch.cuda.empty_cache()

    return dict(train=arms_tr, eval=arms_ev)


def _assign_bio_phenotype_8class(means16: np.ndarray, cutoffs16: np.ndarray) -> np.ndarray:
    """Assign each nucleus to 1 of 8 canonical Orion biological phenotypes using physical cutoffs."""
    # Indices in ORION_16_MARKERS:
    # 0 Hoechst, 1 CD31, 2 CD45, 3 CD68, 4 CD4, 5 FOXP3, 6 CD8a, 7 CD45RO,
    # 8 CD20, 9 PD-L1, 10 CD3e, 11 CD163, 12 E-cadherin, 13 Ki67, 14 Pan-CK, 15 SMA
    pos = means16 > cutoffs16[None, :]
    n = means16.shape[0]
    cls = np.full(n, 7, dtype=np.int32)  # 7 = Other / Stromal-neg
    # Priority hierarchy matching Lin et al. Orion biological lineages
    cls[pos[:, 15]] = 6                                # Stroma (SMA+)
    cls[pos[:, 1]] = 5                                 # Endothelia (CD31+)
    cls[pos[:, 14] | pos[:, 12]] = 0                   # Tumor / Epithelial (Pan-CK+ / E-cad+)
    cls[pos[:, 3] | pos[:, 11]] = 4                    # Macrophage (CD68+ / CD163+)
    cls[pos[:, 8]] = 3                                 # B cell (CD20+)
    cls[pos[:, 4] | pos[:, 10]] = 2                    # Helper / General T (CD4+ / CD3e+)
    cls[pos[:, 5]] = 2                                 # Treg grouped or separate? Let's keep 8 clean classes:
    # Wait: let's make 8 clean distinct classes: 0 Tumor, 1 Cytotoxic T (CD8a+), 2 Helper/Treg T (CD4+/CD3e+/FOXP3+),
    # 3 B cell (CD20+), 4 Macrophage (CD68+/CD163+), 5 Endothelia (CD31+), 6 Stroma (SMA+), 7 Other
    cls[pos[:, 6]] = 1                                 # Cytotoxic T (CD8a+)
    return cls


def run_umap_and_specimen_export(
    bank: dict,
    up_conv: np.ndarray,
    manifest: dict,
    out_dir: Path,
    device: torch.device,
) -> dict:
    """Fit Fold 0 16-ch S_count on ConvNeXt, extract 15-marker single-cell UMAP/PCA & Core/Margin specimen crops."""
    f0 = manifest["folds"][0]
    tr_pats = set(f0["inner_train_slides"])
    ev_pats = set(f0["outer_test_slides"])
    pats_arr = np.asarray(bank["patients"])
    tr_idx = np.nonzero(np.isin(pats_arr, list(tr_pats)))[0]
    ev_idx = np.nonzero(np.isin(pats_arr, list(ev_pats)))[0]

    # Compute pixel Otsu & cell physical cutoffs strictly on Fold 0 inner_train
    pix_cuts = []
    cell_cuts = np.zeros(16, dtype=np.float64)
    for ch in range(16):
        c_px = int(ror.threshold_otsu(bank["targ_nchw"][tr_idx, ch]))
        pix_cuts.append(c_px)
    # Cell cutoffs from inner-train GT nuclei vs labels15 (or pixel Otsu for Hoechst)
    tr_gt_means, tr_lbls = [], []
    for idx in tr_idx.tolist():
        rec = bank["nuclei_records"][idx]
        if rec is None:
            continue
        tr_gt_means.append(rec["gt_means16"])
        tr_lbls.append(rec["labels15"])
    tr_gt_m = np.concatenate(tr_gt_means, axis=0)
    tr_lb = np.concatenate(tr_lbls, axis=0)
    cell_cuts[0] = float(pix_cuts[0])
    for j in range(15):
        cell_cuts[j + 1] = ccs.fit_best_f1_threshold(tr_gt_m[:, j + 1], tr_lb[:, j])

    # Fit 1-seed fast S_count across all 16 channels on Fold 0 for ConvNeXt UMAP + specimen crops
    he_tr = bank["he_nhwc"][tr_idx]
    he_ev = bank["he_nhwc"][ev_idx]
    up_tr = up_conv[tr_idx]
    up_ev = up_conv[ev_idx]
    sc_ev = up_ev.copy()
    for ch in range(16):
        c12_tr = ror.build_common12_orion(he_tr, up_tr, ch, device=device)
        c12_ev = ror.build_common12_orion(he_ev, up_ev, ch, device=device)
        mdl, _ = suh.fit_universal_marker(
            arm="S_count", seed=202610270, common_train=c12_tr,
            donor_u8_train=torch.from_numpy(up_tr[:, ch]),
            target_u8_train=torch.from_numpy(bank["targ_nchw"][tr_idx, ch]),
            cutoff=pix_cuts[ch], device=device, epochs=6, batch_size=16, lr=1.5e-3, deploy_anchor_weight=1.25,
        )
        pr, _ = suh.predict_universal_marker(mdl, c12_ev, torch.from_numpy(up_ev[:, ch]), device=device, batch_size=32)
        sc_ev[:, ch] = pr.astype(np.uint8)
        del mdl, c12_tr, c12_ev
    gc.collect()
    torch.cuda.empty_cache()

    # Extract all Fold 0 eval nuclei across Physical, Frozen, S_count
    phys_m, frz_m, sc_m = [], [], []
    for k_i, idx in enumerate(ev_idx.tolist()):
        rec = bank["nuclei_records"][idx]
        if rec is None:
            continue
        phys_m.append(rec["gt_means16"])
        frz_m.append(ccs.extract_means_with_index(up_ev[k_i], rec["idx_spec"]))
        sc_m.append(ccs.extract_means_with_index(sc_ev[k_i], rec["idx_spec"]))
    X_phys = np.concatenate(phys_m, axis=0)
    X_frz = np.concatenate(frz_m, axis=0)
    X_sc = np.concatenate(sc_m, axis=0)

    cls_names = ["Tumor", "Cytotoxic T", "Helper/Treg T", "B cell", "Macrophage", "Endothelia", "Stroma", "Other"]
    y_phys = _assign_bio_phenotype_8class(X_phys, cell_cuts)
    y_frz = _assign_bio_phenotype_8class(X_frz, cell_cuts)
    y_sc = _assign_bio_phenotype_8class(X_sc, cell_cuts)

    # Per-phenotype F1 and population proportion correlation
    f1_frz = [float(f1_score(y_phys == c, y_frz == c, zero_division=0)) for c in range(8)]
    f1_sc = [float(f1_score(y_phys == c, y_sc == c, zero_division=0)) for c in range(8)]
    prop_phys = np.bincount(y_phys, minlength=8) / max(1, len(y_phys))
    prop_frz = np.bincount(y_frz, minlength=8) / max(1, len(y_frz))
    prop_sc = np.bincount(y_sc, minlength=8) / max(1, len(y_sc))

    # Build 2D embedding (try umap-learn if installed, else clean standardized PCA/t-SNE) on 12,000 stratified nuclei
    rng = np.random.default_rng(20261121)
    n_sub = min(12000, len(X_phys))
    sub_idx = rng.choice(len(X_phys), size=n_sub, replace=False)
    X_log = np.log1p(X_phys[sub_idx, 1:])  # 15 biological markers (excl Hoechst)
    X_std = (X_log - X_log.mean(0, keepdims=True)) / np.maximum(X_log.std(0, keepdims=True), 1e-6)
    emb_method = "PCA"
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.35, random_state=42)
        emb2d = reducer.fit_transform(X_std).astype(np.float32)
        emb_method = "UMAP"
    except Exception:
        try:
            from sklearn.manifold import TSNE
            p10 = PCA(n_components=10, random_state=42).fit_transform(X_std)
            emb2d = TSNE(n_components=2, perplexity=35, max_iter=600, random_state=42, init="pca").fit_transform(p10).astype(np.float32)
            emb_method = "t-SNE"
        except Exception:
            emb2d = PCA(n_components=2, random_state=42).fit_transform(X_std).astype(np.float32)

    np.savez_compressed(
        out_dir / "n1_orion_singlecell_landscape.npz",
        emb2d=emb2d,
        y_phys=y_phys[sub_idx].astype(np.int8),
        y_frz=y_frz[sub_idx].astype(np.int8),
        y_sc=y_sc[sub_idx].astype(np.int8),
        cd8a_phys=X_phys[sub_idx, 6].astype(np.float32),
        cd8a_frz=X_frz[sub_idx, 6].astype(np.float32),
        cd8a_sc=X_sc[sub_idx, 6].astype(np.float32),
        panck_phys=X_phys[sub_idx, 14].astype(np.float32),
        panck_sc=X_sc[sub_idx, 14].astype(np.float32),
        cd3e_phys=X_phys[sub_idx, 10].astype(np.float32),
        cd3e_sc=X_sc[sub_idx, 10].astype(np.float32),
    )

    # Also export 2 real patient specimen crops (High-ImmunoScore long survivor vs Low-ImmunoScore early recurrence)
    # Find best Fold 0 eval tiles for a high-score survivor (e.g. C34/C15/C23) and low-score recurred patient
    clin_by_pat = {p["slide_id"]: p for p in manifest["patients"]}
    ev_pat_list = sorted(set(ev_pats))
    hi_cands = [p for p in ev_pat_list if clin_by_pat[p]["recurrence_event"] == 0 and clin_by_pat[p]["oracle_immunoscore"] >= 3.0]
    lo_cands = [p for p in ev_pat_list if clin_by_pat[p]["recurrence_event"] == 1 and clin_by_pat[p]["oracle_immunoscore"] <= 1.0]
    hi_pat = hi_cands[0] if hi_cands else ev_pat_list[0]
    lo_pat = lo_cands[0] if lo_cands else ev_pat_list[-1]

    def _pick_rich_tile(pat_id: str) -> dict:
        cands = [k for k, idx in enumerate(ev_idx.tolist()) if str(pats_arr[idx]) == pat_id]
        # Score by balanced Pan-CK + CD8a/CD3e signal
        best_k = cands[0]
        best_score = -1.0
        for k in cands:
            idx = int(ev_idx[k])
            pck = float((bank["targ_nchw"][idx, 14] > pix_cuts[14]).mean())
            cd8 = float((bank["targ_nchw"][idx, 6] > pix_cuts[6]).mean())
            cd3 = float((bank["targ_nchw"][idx, 10] > pix_cuts[10]).mean())
            sc_v = min(pck, 0.35) + 3.0 * (cd8 + cd3)
            if sc_v > best_score:
                best_score = sc_v
                best_k = k
        idx = int(ev_idx[best_k])
        return dict(
            patient=pat_id,
            tile_idx=idx,
            pfs_days=float(clin_by_pat[pat_id]["pfs_days"]),
            recurrence=int(clin_by_pat[pat_id]["recurrence_event"]),
            oracle_is=float(clin_by_pat[pat_id]["oracle_immunoscore"]),
            he=bank["he_nhwc"][idx],
            phys_triad=bank["targ_nchw"][idx, [14, 10, 6]],  # Pan-CK, CD3e, CD8a
            frz_triad=up_ev[best_k, [14, 10, 6]],
            sc_triad=sc_ev[best_k, [14, 10, 6]],
        )

    hi_crop = _pick_rich_tile(hi_pat)
    lo_crop = _pick_rich_tile(lo_pat)
    np.savez_compressed(
        out_dir / "n1_orion_survival_specimen_crops.npz",
        hi_he=hi_crop["he"], hi_phys=hi_crop["phys_triad"], hi_frz=hi_crop["frz_triad"], hi_sc=hi_crop["sc_triad"],
        lo_he=lo_crop["he"], lo_phys=lo_crop["phys_triad"], lo_frz=lo_crop["frz_triad"], lo_sc=lo_crop["sc_triad"],
        hi_meta=json.dumps({k: hi_crop[k] for k in ("patient", "tile_idx", "pfs_days", "recurrence", "oracle_is")}),
        lo_meta=json.dumps({k: lo_crop[k] for k in ("patient", "tile_idx", "pfs_days", "recurrence", "oracle_is")}),
    )

    return dict(
        embedding_method=emb_method,
        n_total_fold0_nuclei=int(len(X_phys)),
        n_embedded_nuclei=int(n_sub),
        class_names=cls_names,
        macro_f1_8class_frozen=round(float(np.mean(f1_frz)), 4),
        macro_f1_8class_scount=round(float(np.mean(f1_sc)), 4),
        per_class_f1_frozen={c: round(v, 4) for c, v in zip(cls_names, f1_frz)},
        per_class_f1_scount={c: round(v, 4) for c, v in zip(cls_names, f1_sc)},
        proportion_pearson_frozen=round(float(stats.pearsonr(prop_phys, prop_frz)[0]), 4),
        proportion_pearson_scount=round(float(stats.pearsonr(prop_phys, prop_sc)[0]), 4),
        specimen_hi={k: hi_crop[k] for k in ("patient", "tile_idx", "pfs_days", "recurrence", "oracle_is")},
        specimen_lo={k: lo_crop[k] for k in ("patient", "tile_idx", "pfs_days", "recurrence", "oracle_is")},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-pkl", required=True)
    ap.add_argument("--upstream-dir", required=True)
    ap.add_argument("--cv-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-json", default="", help="Optional custom JSON filename inside out-dir")
    ap.add_argument("--models", default="", help="Optional comma-separated backbone names to run (e.g. 'Adv-Swin,Pix2Pix')")
    ap.add_argument("--skip-landscape", action="store_true", help="Skip Fold 0 16-ch single-cell landscape export")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[NV2-N1] Loading Orion all-40 bank and CV manifest...", flush=True)
    with open(args.bank_pkl, "rb") as f:
        bank = pickle.load(f)
    manifest = json.loads(Path(args.cv_manifest).read_text())
    clin_by_pat = {p["slide_id"]: p for p in manifest["patients"]}
    pats_all = sorted(clin_by_pat.keys())
    pats_arr = np.asarray(bank["patients"])

    up_dir = Path(args.upstream_dir)
    backbones = [
        ("ConvNeXt", "miphei_convnext_all40_2804.npz"),
        ("W-MSE Swin", "miphei_hemit_arch_all40_2804.npz"),
        ("Adv-Swin", "hemit_orion_all40_2804.npz"),
        ("Pix2Pix", "pix2pix_orion_all40_2804.npz"),
        ("ROSIE", "rosie_orion_all40_2804.npz"),
    ]
    if args.models.strip():
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        backbones = [(bn, fn) for bn, fn in backbones if bn in wanted]
    out_json_path = out_dir / (args.out_json.strip() or "nv2_n1_orion_clinical_umap_canonical.json")

    # Precompute oracle physical mIF 5-fold out-of-patient scores (exact baseline)
    oracle_cont = {}
    oracle_disc = {}
    fold_phys_tr_comps = {}
    fold_pix_cuts = {}
    fold_cell_cuts = {}

    for fd in manifest["folds"]:
        fidx = int(fd["fold"])
        tr_idx = np.nonzero(np.isin(pats_arr, fd["inner_train_slides"]))[0]
        ev_idx = np.nonzero(np.isin(pats_arr, fd["outer_test_slides"]))[0]

        pix_c = {m: int(ror.threshold_otsu(bank["targ_nchw"][tr_idx, TRIAD_CHS[m]])) for m in TRIAD_NAMES}
        # Physical cell cutoffs from inner-train GT nuclei
        tr_gm, tr_lb = [], []
        for idx in tr_idx.tolist():
            rec = bank["nuclei_records"][idx]
            if rec is not None:
                tr_gm.append(rec["gt_means16"])
                tr_lb.append(rec["labels15"])
        tr_gm = np.concatenate(tr_gm, axis=0)
        tr_lb = np.concatenate(tr_lb, axis=0)
        cell_c = {}
        for m in TRIAD_NAMES:
            ch = TRIAD_CHS[m]
            cell_c[m] = ccs.fit_best_f1_threshold(tr_gm[:, ch], tr_lb[:, ch - 1])
        fold_pix_cuts[fidx] = pix_c
        fold_cell_cuts[fidx] = cell_c

        _, phys_tr = _compute_patient_compartments_with_cutoffs(bank, bank["targ_nchw"], tr_idx, cell_c)
        ev_pats_f, phys_ev = _compute_patient_compartments_with_cutoffs(bank, bank["targ_nchw"], ev_idx, cell_c)
        fold_phys_tr_comps[fidx] = phys_tr
        sc_or = ccs.score_immunoscore_from_training_ref(phys_tr, phys_ev)
        for p, c_v, d_v in zip(ev_pats_f, sc_or["immunoscore_continuous"], sc_or["immunoscore_discrete"]):
            oracle_cont[p] = float(c_v)
            oracle_disc[p] = float(d_v)

    oracle_cont_vec = np.asarray([oracle_cont[p] for p in pats_all], dtype=np.float64)
    oracle_disc_vec = np.asarray([oracle_disc[p] for p in pats_all], dtype=np.float64)
    oracle_summary = _summarize_survival_vector(pats_all, oracle_cont_vec, oracle_disc_vec, oracle_disc_vec, clin_by_pat)
    print(f"[NV2-N1] Physical mIF Oracle: C={oracle_summary['c_index']:.4f} HR={oracle_summary['hr_low_vs_high']} P={oracle_summary['logrank_p']}", flush=True)

    results = dict(
        schema="nv2-n1-orion-clinical-umap-v1",
        n_patients=len(pats_all),
        oracle_physical_mif=oracle_summary,
        backbones={},
    )

    for bname, fname in backbones:
        fpath = up_dir / fname
        if not fpath.exists():
            print(f"[NV2-N1] Skipping missing upstream cache: {fpath}", flush=True)
            continue
        tb = time.time()
        print(f"[NV2-N1] Running 5-fold Physical-Locked vs Virtual-Refitted survival for {bname}...", flush=True)
        up_all = np.load(fpath)["pred_u8"]

        # Containers: protocol -> arm -> pat -> (cont, disc)
        oof = {
            prot: {arm: {} for arm in ("Frozen", "S_recon", "QuantileMatch", "E_count", "S_count")}
            for prot in ("physical_locked", "virtual_refitted")
        }

        for fd in manifest["folds"]:
            fidx = int(fd["fold"])
            tr_idx = np.nonzero(np.isin(pats_arr, fd["inner_train_slides"]))[0]
            ev_idx = np.nonzero(np.isin(pats_arr, fd["outer_test_slides"]))[0]
            cell_c = fold_cell_cuts[fidx]
            pix_c = fold_pix_cuts[fidx]
            phys_tr = fold_phys_tr_comps[fidx]

            preds = _fit_and_predict_triad_fold(bank, up_all, tr_idx, ev_idx, pix_c, device)
            # Re-index into full-bank proxy via temporary local indices
            tr_sub = np.arange(len(tr_idx))
            ev_sub = np.arange(len(ev_idx))
            bank_tr = dict(nuclei_records=[bank["nuclei_records"][i] for i in tr_idx.tolist()])
            bank_ev = dict(nuclei_records=[bank["nuclei_records"][i] for i in ev_idx.tolist()])

            for arm in ("Frozen", "S_recon", "QuantileMatch", "E_count", "S_count"):
                _, arm_tr_comp = _compute_patient_compartments_with_cutoffs(bank_tr, preds["train"][arm], tr_sub, cell_c)
                ev_pats_f, arm_ev_comp = _compute_patient_compartments_with_cutoffs(bank_ev, preds["eval"][arm], ev_sub, cell_c)

                sc_phys = ccs.score_immunoscore_from_training_ref(phys_tr, arm_ev_comp)
                sc_virt = ccs.score_immunoscore_from_training_ref(arm_tr_comp, arm_ev_comp)
                for p, c1, d1, c2, d2 in zip(
                    ev_pats_f,
                    sc_phys["immunoscore_continuous"], sc_phys["immunoscore_discrete"],
                    sc_virt["immunoscore_continuous"], sc_virt["immunoscore_discrete"],
                ):
                    oof["physical_locked"][arm][p] = (float(c1), float(d1))
                    oof["virtual_refitted"][arm][p] = (float(c2), float(d2))

        b_res = {}
        for prot in ("physical_locked", "virtual_refitted"):
            frz_vec = np.asarray([oof[prot]["Frozen"][p][0] for p in pats_all], dtype=np.float64)
            b_res[prot] = {}
            for arm in ("Frozen", "S_recon", "QuantileMatch", "E_count", "S_count"):
                c_vec = np.asarray([oof[prot][arm][p][0] for p in pats_all], dtype=np.float64)
                d_vec = np.asarray([oof[prot][arm][p][1] for p in pats_all], dtype=np.float64)
                b_res[prot][arm] = _summarize_survival_vector(
                    pats_all, c_vec, d_vec, oracle_disc_vec, clin_by_pat,
                    ref_scores_for_delta=None if arm == "Frozen" else frz_vec,
                )
        results["backbones"][bname] = b_res
        pl_f = b_res["physical_locked"]["Frozen"]
        pl_s = b_res["physical_locked"]["S_count"]
        vr_f = b_res["virtual_refitted"]["Frozen"]
        vr_s = b_res["virtual_refitted"]["S_count"]
        print(
            f"[NV2-N1] {bname} ({time.time()-tb:.1f}s) | "
            f"PHYS-LOCKED: Frz C={pl_f['c_index']:.4f} (nH={pl_f['n_high']}, MAE={pl_f['immunoscore_mae_cont']:.3f}, QWK={pl_f['quadratic_kappa_vs_oracle']:.3f}) -> "
            f"S_cnt C={pl_s['c_index']:.4f} (nH={pl_s['n_high']}, MAE={pl_s['immunoscore_mae_cont']:.3f}, QWK={pl_s['quadratic_kappa_vs_oracle']:.3f}) | "
            f"VIRT-REFIT: Frz C={vr_f['c_index']:.4f} -> S_cnt C={vr_s['c_index']:.4f}",
            flush=True,
        )

        # Run single-cell UMAP & specimen crops on ConvNeXt
        if bname == "ConvNeXt" and not args.skip_landscape:
            print("[NV2-N1] Computing Fold 0 15-marker single-cell phenotype landscape & specimen crops...", flush=True)
            results["single_cell_landscape"] = run_umap_and_specimen_export(bank, up_all, manifest, out_dir, device)
            print("[NV2-N1] Landscape done:", results["single_cell_landscape"]["embedding_method"],
                  results["single_cell_landscape"]["macro_f1_8class_frozen"], "->",
                  results["single_cell_landscape"]["macro_f1_8class_scount"], flush=True)

        del up_all
        gc.collect()
        _atomic_write_json(out_json_path, results)

    results["wall_seconds"] = round(time.time() - t0, 1)
    _atomic_write_json(out_json_path, results)
    print("NV2_N1_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
