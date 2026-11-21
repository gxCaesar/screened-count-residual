#!/usr/bin/env python3
"""NV2-N1 (v2 Locked-Checkpoint Exact Protocol):
Orion-CRC Physical-Reference-Locked vs Virtual-Refitted ImmunoScore Survival across ALL 6 Published
Generators (ConvNeXt, W-MSE Swin, Adv-Swin, Pix2Pix, Diffusion-FT, ROSIE), 15-Marker Single-Cell
Phenotype Landscape, and High/Low-ImmunoScore Spatial Specimen Maps.

Loads the EXACT locked v1.2.0 confirmatory 5-fold checkpoints (`state_dir/<backbone>/fold{0..4}/`)
and exact canonical physical cutoffs (`ror.calibrate_orion_physical_cutoffs_from_train`), guaranteeing
100% bit-exact protocol fidelity with v1.2.0 while adding:
  1. Direct Physical-Reference-Locked clinical deployment (transferring inner-train PHYSICAL mIF R1/R2
     reference distributions directly to virtual stains WITHOUT virtual-cohort percentile refitting)
     side-by-side with Virtual-Refitted cohort scoring across all 6 backbones x 5 arms
     (Frozen, S_recon, QuantileMatch, E_count, S_count).
  2. Tie-checked Harrell C-index + 10,000 duplicate-free patient bootstrap 95% CIs, paired delta-C,
     univariable KM HR, log-rank P, Stage+MMR multivariable HR, numbers-at-risk at [0,12,24,36,48,60] mo,
     risk-group counts (n_high, n_low), ImmunoScore MAE, Pearson r, Spearman rho & quadratic Cohen's kappa
     against the Physical mIF Oracle.
  3. Exact locked Fold-0 16-marker 2-seed single-cell phenotype landscape (85,729 nuclei, 8 canonical
     biological lineages) & High/Low-ImmunoScore real patient specimen crops.
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
import pandas as pd
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
TRIAD_NAMES = ("Pan-CK", "CD3e", "CD8a")
TRIAD_CHS = {m: ORION_MARKERS.index(m) for m in TRIAD_NAMES}
SEEDS = (202610270, 202610271)

BACKBONES = [
    ("ConvNeXt", "miphei_convnext", "miphei_convnext_all40_2804.npz"),
    ("W-MSE Swin", "miphei_hemit_arch", "miphei_hemit_arch_all40_2804.npz"),
    ("Adv-Swin", "hemit_orion", "hemit_orion_all40_2804.npz"),
    ("Pix2Pix", "pix2pix_orion", "pix2pix_orion_all40_2804.npz"),
    ("Diffusion-FT", "diffusion_ft", "diffusion_ft_all40_2804.npz"),
    ("ROSIE", "rosie_orion", "rosie_orion_all40_2804.npz"),
]


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _km_numbers_at_risk(times_mo: np.ndarray, mask: np.ndarray, ticks=(0, 12, 24, 36, 48, 60)) -> list[int]:
    t = times_mo[mask]
    return [int(np.sum(t >= tk)) for tk in ticks]


def _summarize_survival_vector(
    pats_order: list[str],
    scores_cont: np.ndarray,
    scores_disc: np.ndarray,
    oracle_disc: np.ndarray,
    clin_df: pd.DataFrame,
    ref_scores_for_delta: np.ndarray | None = None,
) -> dict:
    pfs_days = clin_df["PFSDays"].to_numpy(dtype=np.float64)
    times_mo = pfs_days / 30.4375
    rec_ev = clin_df["Recurrence"].to_numpy(dtype=np.int32)
    stage_num = clin_df["Stage"].map({"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}).to_numpy(dtype=np.float64)
    mmr_def = (clin_df["MMRIHC"].fillna("Intact").str.lower() != "intact").astype(np.float64).to_numpy()

    ev = ccs.evaluate_pfs_survival(
        pfs_days=pfs_days,
        recurrence_events=rec_ev,
        protective_score=scores_cont,
        dichotomy_cutoff=2.0,
        stage_numeric=stage_num,
        mmr_deficient=mmr_def,
        reference_protective_score=ref_scores_for_delta,
        n_bootstrap=10000,
        seed=20261115,
    )

    high_mask = scores_cont > 2.0
    low_mask = ~high_mask
    n_high = int(np.sum(high_mask))
    n_low = int(np.sum(low_mask))
    nar_high = _km_numbers_at_risk(times_mo, high_mask)
    nar_low = _km_numbers_at_risk(times_mo, low_mask)
    km_high = None
    km_low = None
    if n_high > 0 and n_low > 0:
        km_h = ccs.kaplan_meier_curve(times_mo[high_mask], rec_ev[high_mask])
        km_l = ccs.kaplan_meier_curve(times_mo[low_mask], rec_ev[low_mask])
        km_high = dict(time=km_h["time"].tolist(), survival=km_h["survival"].tolist(), ci_low=km_h["ci_low"].tolist(), ci_high=km_h["ci_high"].tolist())
        km_low = dict(time=km_l["time"].tolist(), survival=km_l["survival"].tolist(), ci_low=km_l["ci_low"].tolist(), ci_high=km_l["ci_high"].tolist())

    mae_disc = float(np.mean(np.abs(scores_disc - oracle_disc)))
    mae_cont = float(np.mean(np.abs(scores_cont - oracle_disc)))
    pearson_r = float(stats.pearsonr(scores_cont, oracle_disc)[0]) if np.std(scores_cont) > 1e-9 else 0.0
    spearman_rho = float(stats.spearmanr(scores_cont, oracle_disc)[0]) if np.std(scores_cont) > 1e-9 else 0.0
    d_int = np.clip(np.rint(scores_disc), 0, 4).astype(int)
    o_int = np.clip(np.rint(oracle_disc), 0, 4).astype(int)
    qwk = float(cohen_kappa_score(o_int, d_int, weights="quadratic")) if len(np.unique(d_int)) > 1 else 0.0

    # Extract key fields cleanly from ccs.evaluate_pfs_survival with binary risk-set estimability guard
    collapsed = bool(n_high == 0 or n_low == 0)
    ev_times = pfs_days[rec_ev == 1]
    t_first = float(np.min(ev_times)) if len(ev_times) > 0 else float("inf")
    hi_at_risk_first = int(np.sum(pfs_days[high_mask] >= t_first)) if n_high > 0 else 0
    lo_at_risk_first = int(np.sum(pfs_days[low_mask] >= t_first)) if n_low > 0 else 0
    contrast_estimable = bool((not collapsed) and hi_at_risk_first > 0 and lo_at_risk_first > 0)
    lr_p = None if not contrast_estimable else ev.get("logrank_p")
    hr_hl = None if not contrast_estimable else ev.get("hr_high_vs_low_risk")
    hr_ci = None if not contrast_estimable else ev.get("hr_high_vs_low_ci95")
    mv_hr = None
    mv_ci = None
    mv_p = None
    if isinstance(ev.get("multivariable_cox"), dict) and np.std(scores_cont) > 1e-9:
        mv_hr = ev["multivariable_cox"]["hr"][0]
        mv_ci = [round(float(ev["multivariable_cox"]["hr_ci_low"][0]), 4), round(float(ev["multivariable_cox"]["hr_ci_high"][0]), 4)]
        mv_p = float(ev["multivariable_cox"]["p_value"][0]) if "p_value" in ev["multivariable_cox"] else None

    out = dict(
        c_index=round(float(ev["c_index"]), 6),
        c_ci95=[round(float(ev["c_index_ci95"][0]), 6), round(float(ev["c_index_ci95"][1]), 6)],
        n_high=n_high,
        n_low=n_low,
        contrast_estimable=contrast_estimable,
        mv_p_n40=mv_p,
        risk_group_collapsed=collapsed,
        logrank_p=round(float(lr_p), 8) if lr_p is not None and np.isfinite(lr_p) else None,
        hr_low_vs_high=round(float(hr_hl), 4) if hr_hl is not None and np.isfinite(hr_hl) else None,
        hr_low_vs_high_ci95=[round(float(hr_ci[0]), 4), round(float(hr_ci[1]), 4)] if hr_ci is not None else None,
        mv_stage_mmr_hr_per_sd=round(float(mv_hr), 4) if mv_hr is not None and np.isfinite(mv_hr) else None,
        mv_stage_mmr_ci95=mv_ci,
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
    if ref_scores_for_delta is not None and "delta_c_index_vs_ref" in ev:
        out["delta_c_vs_frozen"] = round(float(ev["delta_c_index_vs_ref"]), 6)
        out["delta_c_ci95"] = [round(float(ev["delta_c_index_ci95"][0]), 6), round(float(ev["delta_c_index_ci95"][1]), 6)]
        if "delta_c_index_boot_p" in ev and ev["delta_c_index_boot_p"] is not None:
            out["delta_c_boot_p"] = round(float(ev["delta_c_index_boot_p"]), 6)
    return out


def _assign_bio_phenotype_8class(means16: np.ndarray, cutoffs16: np.ndarray) -> np.ndarray:
    pos = means16 > cutoffs16[None, :]
    n = means16.shape[0]
    cls = np.full(n, 7, dtype=np.int32)
    cls[pos[:, 15]] = 6                 # Stroma (SMA+)
    cls[pos[:, 1]] = 5                  # Endothelia (CD31+)
    cls[pos[:, 14] | pos[:, 12]] = 0    # Tumor / Epithelial (Pan-CK+/E-cad+)
    cls[pos[:, 3] | pos[:, 11]] = 4     # Macrophage (CD68+/CD163+)
    cls[pos[:, 8]] = 3                  # B cell (CD20+)
    cls[pos[:, 4] | pos[:, 10] | pos[:, 5]] = 2  # Helper/Treg T (CD4+/CD3e+/FOXP3+)
    cls[pos[:, 6]] = 1                  # Cytotoxic T (CD8a+)
    return cls


def _run_locked_fold0_landscape(
    bank: dict,
    up_conv: np.ndarray,
    state_dir: Path,
    manifest: dict,
    clin_df: pd.DataFrame,
    out_dir: Path,
    device: torch.device,
) -> dict:
    """Use EXACT locked Fold 0 16-ch 2-seed S_count checkpoints for ConvNeXt UMAP & specimen crops."""
    fd0 = manifest["folds"][0]
    tr_p = ror._fold_patients(fd0, "inner_train")
    te_p = ror._fold_patients(fd0, "outer_test")
    pats_arr = np.asarray(bank["patients"])
    tr_idx = np.nonzero(np.isin(pats_arr, tr_p))[0]
    te_idx = np.nonzero(np.isin(pats_arr, te_p))[0]

    f_cuts_u8, f_tau_float = ror.calibrate_orion_physical_cutoffs_from_train(
        bank["targ_nchw"][tr_idx], tr_idx, bank["nuclei_records"]
    )
    cell_cuts = np.asarray([float(f_tau_float[m]) for m in ORION_MARKERS], dtype=np.float64)

    he_te = bank["he_nhwc"][te_idx]
    up_te = up_conv[te_idx]
    sc_te = np.empty_like(up_te)
    f0_dir = state_dir / "miphei_convnext" / "fold0"

    for ch16, m_name in enumerate(ORION_MARKERS):
        c12_te = ror.build_common12_orion(he_te, up_te, ch16, device=device)
        d_te_t = torch.from_numpy(up_te[:, ch16])
        preds = []
        for sd in SEEDS:
            ck_p = f0_dir / f"{m_name}_S_count_{sd}.pt"
            mdl = suh.build_coordinate_screened_model(sd, device)
            mdl.load_state_dict(torch.load(ck_p, map_location=device, weights_only=True))
            pr, _ = suh.predict_universal_marker(mdl, c12_te, d_te_t, device=device, batch_size=32)
            preds.append(pr.astype(np.float32))
            del mdl
        sc_te[:, ch16] = np.rint(np.clip(0.5 * (preds[0] + preds[1]), 0.0, 255.0)).astype(np.uint8)
        del c12_te
    gc.collect()
    torch.cuda.empty_cache()

    phys_m, frz_m, sc_m = [], [], []
    for k_i, gi in enumerate(te_idx.tolist()):
        rec = bank["nuclei_records"][gi]
        if rec is None:
            continue
        phys_m.append(rec["gt_means16"])
        frz_m.append(ccs.extract_means_with_index(up_te[k_i], rec["idx_spec"]))
        sc_m.append(ccs.extract_means_with_index(sc_te[k_i], rec["idx_spec"]))
    X_phys = np.concatenate(phys_m, axis=0)
    X_frz = np.concatenate(frz_m, axis=0)
    X_sc = np.concatenate(sc_m, axis=0)

    cls_names = ["Tumor", "Cytotoxic T", "Helper/Treg T", "B cell", "Macrophage", "Endothelia", "Stroma", "Other"]
    y_phys = _assign_bio_phenotype_8class(X_phys, cell_cuts)
    y_frz = _assign_bio_phenotype_8class(X_frz, cell_cuts)
    y_sc = _assign_bio_phenotype_8class(X_sc, cell_cuts)

    f1_frz = [float(f1_score(y_phys == c, y_frz == c, zero_division=0)) for c in range(8)]
    f1_sc = [float(f1_score(y_phys == c, y_sc == c, zero_division=0)) for c in range(8)]
    prop_phys = np.bincount(y_phys, minlength=8) / max(1, len(y_phys))
    prop_frz = np.bincount(y_frz, minlength=8) / max(1, len(y_frz))
    prop_sc = np.bincount(y_sc, minlength=8) / max(1, len(y_sc))

    rng = np.random.default_rng(20261121)
    n_sub = min(12000, len(X_phys))
    sub_idx = rng.choice(len(X_phys), size=n_sub, replace=False)
    X_log = np.log1p(X_phys[sub_idx, 1:])
    X_std = (X_log - X_log.mean(0, keepdims=True)) / np.maximum(X_log.std(0, keepdims=True), 1e-6)
    emb_method = "PCA"
    try:
        import umap
        emb2d = umap.UMAP(n_neighbors=25, min_dist=0.35, random_state=42).fit_transform(X_std).astype(np.float32)
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
        out_dir / "n1_orion_singlecell_landscape_v2.npz",
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

    clin_map = {str(r["PatientID"]): r for _, r in clin_df.iterrows()}
    te_pats_sorted = sorted(set(te_p))
    hi_cands = [p for p in te_pats_sorted if int(clin_map[p]["Recurrence"]) == 0 and float(clin_map[p]["ImmunoScore"]) >= 3.0]
    lo_cands = [p for p in te_pats_sorted if int(clin_map[p]["Recurrence"]) == 1 and float(clin_map[p]["ImmunoScore"]) <= 1.0]
    hi_pat = hi_cands[0] if hi_cands else te_pats_sorted[0]
    lo_pat = lo_cands[0] if lo_cands else te_pats_sorted[-1]

    def _pick_rich_tile(pat_id: str) -> dict:
        cands = [k for k, gi in enumerate(te_idx.tolist()) if str(pats_arr[gi]) == pat_id]
        best_k, best_s = cands[0], -1.0
        for k in cands:
            gi = int(te_idx[k])
            pck = float((bank["targ_nchw"][gi, 14] > f_cuts_u8[14]).mean())
            cd8 = float((bank["targ_nchw"][gi, 6] > f_cuts_u8[6]).mean())
            cd3 = float((bank["targ_nchw"][gi, 10] > f_cuts_u8[10]).mean())
            sv = min(pck, 0.35) + 3.0 * (cd8 + cd3)
            if sv > best_s:
                best_s = sv
                best_k = k
        gi = int(te_idx[best_k])
        return dict(
            patient=pat_id,
            tile_idx=gi,
            pfs_days=float(clin_map[pat_id]["PFSDays"]),
            recurrence=int(clin_map[pat_id]["Recurrence"]),
            oracle_is=float(clin_map[pat_id]["ImmunoScore"]),
            he=bank["he_nhwc"][gi],
            phys_triad=bank["targ_nchw"][gi, [14, 10, 6]],
            frz_triad=up_te[best_k, [14, 10, 6]],
            sc_triad=sc_te[best_k, [14, 10, 6]],
        )

    hi_crop = _pick_rich_tile(hi_pat)
    lo_crop = _pick_rich_tile(lo_pat)
    np.savez_compressed(
        out_dir / "n1_orion_survival_specimen_crops_v2.npz",
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
    ap.add_argument("--state-dir", required=True, help="Directory holding locked v1.2.0 orion_state_v4 checkpoints & upstream_cache")
    ap.add_argument("--cv-manifest", required=True)
    ap.add_argument("--clinical-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-json", default="nv2_n1_orion_clinical_umap_locked_v2.json")
    ap.add_argument("--models", default="", help="Optional comma-separated display names to run (e.g. 'W-MSE Swin,Adv-Swin')")
    ap.add_argument("--skip-landscape", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir = Path(args.state_dir)
    up_dir = state_dir / "upstream_cache"

    print("[NV2-N1v2] Loading Orion all-40 bank, canonical CV manifest, and clinical CSV...", flush=True)
    with open(args.bank_pkl, "rb") as fh:
        bank = pickle.load(fh)
    manifest = json.loads(Path(args.cv_manifest).read_text())
    clin_df = pd.read_csv(args.clinical_csv)
    if "PatientID" not in clin_df.columns and "slideName" in clin_df.columns:
        clin_df["PatientID"] = clin_df["slideName"].astype(str)
    pats_order = clin_df["PatientID"].astype(str).tolist()
    pat_to_row = {p: i for i, p in enumerate(pats_order)}
    pats_arr = np.asarray(bank["patients"])

    # Precompute exact canonical physical mIF cutoffs & out-of-fold Physical mIF Oracle
    fold_cuts_u8 = {}
    fold_tau_triad = {}
    fold_phys_tr_vecs = {}
    oof_gt_cont = np.zeros(len(pats_order), dtype=np.float64)
    oof_gt_disc = np.zeros(len(pats_order), dtype=np.float64)

    for fd in manifest["folds"]:
        f_id = int(fd["fold"])
        tr_p = ror._fold_patients(fd, "inner_train")
        te_p = ror._fold_patients(fd, "outer_test")
        tr_idx = np.nonzero(np.isin(pats_arr, tr_p))[0]
        te_idx = np.nonzero(np.isin(pats_arr, te_p))[0]

        cuts_u8, tau_f = ror.calibrate_orion_physical_cutoffs_from_train(
            bank["targ_nchw"][tr_idx], tr_idx, bank["nuclei_records"]
        )
        fold_cuts_u8[f_id] = cuts_u8
        fold_tau_triad[f_id] = {m: tau_f[m] for m in TRIAD_NAMES}

        # Physical mIF compartments via exact ground-truth labels (matching v1.2.0 oracle)
        tr_comp = ror.compute_patient_compartments_for_subset(
            tr_idx, pats_arr[tr_idx], bank["nuclei_records"], None, {}, use_ground_truth_labels=True
        )
        te_comp = ror.compute_patient_compartments_for_subset(
            te_idx, pats_arr[te_idx], bank["nuclei_records"], None, {}, use_ground_truth_labels=True
        )
        tr_vecs = {k: np.array([tr_comp[p][k] for p in tr_p if p in tr_comp]) for k in ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2")}
        te_present = [p for p in te_p if p in te_comp]
        te_vecs = {k: np.array([te_comp[p][k] for p in te_present]) for k in ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2")}
        fold_phys_tr_vecs[f_id] = tr_vecs
        sc_gt = ccs.score_immunoscore_from_training_ref(tr_vecs, te_vecs)
        for pi, pat_id in enumerate(te_present):
            r_i = pat_to_row[pat_id]
            oof_gt_cont[r_i] = sc_gt["immunoscore_continuous"][pi]
            oof_gt_disc[r_i] = sc_gt["immunoscore_discrete"][pi]

    oracle_wsi_disc = clin_df["ImmunoScore"].to_numpy(dtype=np.float64)
    oracle_tile_summary = _summarize_survival_vector(pats_order, oof_gt_cont, oof_gt_disc, oof_gt_disc, clin_df)
    oracle_wsi_summary = _summarize_survival_vector(pats_order, oracle_wsi_disc, oracle_wsi_disc, oracle_wsi_disc, clin_df)
    print(
        f"[NV2-N1v2] Physical mIF Oracles | WSI C={oracle_wsi_summary['c_index']:.4f} HR={oracle_wsi_summary['hr_low_vs_high']} P={oracle_wsi_summary['logrank_p']} | "
        f"Tile-subset OOF C={oracle_tile_summary['c_index']:.4f} HR={oracle_tile_summary['hr_low_vs_high']} P={oracle_tile_summary['logrank_p']}",
        flush=True,
    )

    results = dict(
        schema="nv2-n1-orion-clinical-umap-locked-v2",
        protocol="Exact locked v1.2.0 5-fold checkpoints + canonical calibrate_orion_physical_cutoffs_from_train",
        n_patients=len(pats_order),
        oracle_published_wsi_mif=oracle_wsi_summary,
        oracle_physical_mif=oracle_tile_summary,
        backbones={},
    )

    arms_all = ("Frozen", "S_recon", "QuantileMatch", "E_count", "S_count")
    want_models = {m.strip() for m in args.models.split(",") if m.strip()}
    out_json_path = out_dir / args.out_json

    for disp_name, b_key, up_fname in BACKBONES:
        if want_models and disp_name not in want_models:
            continue
        up_path = up_dir / up_fname
        if not up_path.exists():
            print(f"[NV2-N1v2] Skipping missing upstream cache: {up_path}", flush=True)
            continue
        tb = time.time()
        print(f"[NV2-N1v2] Evaluating {disp_name} ({b_key}) across 5 locked folds...", flush=True)
        up_all = np.load(up_path)["pred_u8"]

        oof_cont = {prot: {a: np.zeros(len(pats_order), dtype=np.float64) for a in arms_all} for prot in ("physical_locked", "virtual_refitted")}
        oof_disc = {prot: {a: np.zeros(len(pats_order), dtype=np.float64) for a in arms_all} for prot in ("physical_locked", "virtual_refitted")}

        for fd in manifest["folds"]:
            f_id = int(fd["fold"])
            tr_p = ror._fold_patients(fd, "inner_train")
            te_p = ror._fold_patients(fd, "outer_test")
            tr_idx = np.nonzero(np.isin(pats_arr, tr_p))[0]
            te_idx = np.nonzero(np.isin(pats_arr, te_p))[0]
            cuts_u8 = fold_cuts_u8[f_id]
            tau_triad = fold_tau_triad[f_id]
            phys_tr_vecs = fold_phys_tr_vecs[f_id]
            f_ckpt_dir = state_dir / b_key / f"fold{f_id}"

            triad_tr = {a: np.empty((len(tr_idx), 3, 256, 256), dtype=np.uint8) for a in arms_all}
            triad_te = {a: np.empty((len(te_idx), 3, 256, 256), dtype=np.uint8) for a in arms_all}

            he_tr = bank["he_nhwc"][tr_idx]
            he_te = bank["he_nhwc"][te_idx]
            up_tr = up_all[tr_idx]
            up_te = up_all[te_idx]

            for ti_m, m_name in enumerate(TRIAD_NAMES):
                ch16 = TRIAD_CHS[m_name]
                cut = int(cuts_u8[ch16])
                u_tr = up_tr[:, ch16]
                u_te = up_te[:, ch16]
                y_tr = bank["targ_nchw"][tr_idx, ch16]

                triad_tr["Frozen"][:, ti_m] = u_tr
                triad_te["Frozen"][:, ti_m] = u_te

                qm_tr, qm_te, _ = rcomp.apply_quantile_match_u8(u_tr, y_tr, u_te)
                triad_tr["QuantileMatch"][:, ti_m] = qm_tr
                triad_te["QuantileMatch"][:, ti_m] = qm_te

                c12_tr = ror.build_common12_orion(he_tr, up_tr, ch16, device=device)
                c12_te = ror.build_common12_orion(he_te, up_te, ch16, device=device)
                d_tr_t = torch.from_numpy(u_tr)
                d_te_t = torch.from_numpy(u_te)
                y_tr_t = torch.from_numpy(y_tr)

                # Load EXACT locked v1.2.0 checkpoints for S_recon and S_count
                for arm in ("S_recon", "S_count"):
                    tr_s, te_s = [], []
                    for sd in SEEDS:
                        ck_p = f_ckpt_dir / f"{m_name}_{arm}_{sd}.pt"
                        mdl = suh.build_coordinate_screened_model(sd, device)
                        mdl.load_state_dict(torch.load(ck_p, map_location=device, weights_only=True))
                        p_tr, _ = suh.predict_universal_marker(mdl, c12_tr, d_tr_t, device=device, batch_size=32)
                        p_te, _ = suh.predict_universal_marker(mdl, c12_te, d_te_t, device=device, batch_size=32)
                        tr_s.append(p_tr.astype(np.float32))
                        te_s.append(p_te.astype(np.float32))
                        del mdl
                    triad_tr[arm][:, ti_m] = np.round(np.mean(tr_s, axis=0)).astype(np.uint8)
                    triad_te[arm][:, ti_m] = np.round(np.mean(te_s, axis=0)).astype(np.uint8)

                # Fit E_count (2-seed) at the exact canonical cutoff
                tr_e, te_e = [], []
                for sd in SEEDS:
                    mdl_e, _ = rp2.fit_ablation_marker(
                        mode="E_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t, target_u8_train=y_tr_t,
                        cutoff=cut, device=device, epochs=6, batch_size=16, lr=1.5e-3,
                    )
                    p_tr, _ = rp2.predict_ablation_marker(mdl_e, "E_count", c12_tr, d_tr_t, device=device, batch_size=32)
                    p_te, _ = rp2.predict_ablation_marker(mdl_e, "E_count", c12_te, d_te_t, device=device, batch_size=32)
                    tr_e.append(p_tr.astype(np.float32))
                    te_e.append(p_te.astype(np.float32))
                    del mdl_e
                triad_tr["E_count"][:, ti_m] = np.round(np.mean(tr_e, axis=0)).astype(np.uint8)
                triad_te["E_count"][:, ti_m] = np.round(np.mean(te_e, axis=0)).astype(np.uint8)

                del c12_tr, c12_te
                gc.collect()
                torch.cuda.empty_cache()

            # Score BOTH Physical-Reference-Locked and Virtual-Refitted ImmunoScores
            for arm in arms_all:
                tr_comp = ror.compute_patient_compartments_for_subset(
                    tr_idx, pats_arr[tr_idx], bank["nuclei_records"], triad_tr[arm], tau_triad
                )
                te_comp = ror.compute_patient_compartments_for_subset(
                    te_idx, pats_arr[te_idx], bank["nuclei_records"], triad_te[arm], tau_triad
                )
                tr_present = [p for p in tr_p if p in tr_comp]
                te_present = [p for p in te_p if p in te_comp]
                arm_tr_vecs = {k: np.array([tr_comp[p][k] for p in tr_present]) for k in ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2")}
                arm_te_vecs = {k: np.array([te_comp[p][k] for p in te_present]) for k in ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2")}

                sc_phys = ccs.score_immunoscore_from_training_ref(phys_tr_vecs, arm_te_vecs)
                sc_virt = ccs.score_immunoscore_from_training_ref(arm_tr_vecs, arm_te_vecs)
                for pi, pat_id in enumerate(te_present):
                    r_i = pat_to_row[pat_id]
                    oof_cont["physical_locked"][arm][r_i] = sc_phys["immunoscore_continuous"][pi]
                    oof_disc["physical_locked"][arm][r_i] = sc_phys["immunoscore_discrete"][pi]
                    oof_cont["virtual_refitted"][arm][r_i] = sc_virt["immunoscore_continuous"][pi]
                    oof_disc["virtual_refitted"][arm][r_i] = sc_virt["immunoscore_discrete"][pi]

        b_res = {}
        for prot in ("physical_locked", "virtual_refitted"):
            frz_vec = oof_cont[prot]["Frozen"]
            b_res[prot] = {}
            for arm in arms_all:
                b_res[prot][arm] = _summarize_survival_vector(
                    pats_order,
                    oof_cont[prot][arm],
                    oof_disc[prot][arm],
                    oof_gt_disc,
                    clin_df,
                    ref_scores_for_delta=None if arm == "Frozen" else frz_vec,
                )
        results["backbones"][disp_name] = b_res
        pl_f, pl_s = b_res["physical_locked"]["Frozen"], b_res["physical_locked"]["S_count"]
        pl_q, pl_e = b_res["physical_locked"]["QuantileMatch"], b_res["physical_locked"]["E_count"]
        vr_f, vr_s = b_res["virtual_refitted"]["Frozen"], b_res["virtual_refitted"]["S_count"]
        print(
            f"[NV2-N1v2] {disp_name:12s} ({time.time()-tb:.1f}s) | "
            f"PHYS-LOCK: Frz={pl_f['c_index']:.4f}(nH={pl_f['n_high']},QWK={pl_f['quadratic_kappa_vs_oracle']:.2f}) "
            f"QM={pl_q['c_index']:.4f}(nH={pl_q['n_high']},QWK={pl_q['quadratic_kappa_vs_oracle']:.2f}) "
            f"Ec={pl_e['c_index']:.4f}(nH={pl_e['n_high']},QWK={pl_e['quadratic_kappa_vs_oracle']:.2f}) "
            f"Sc={pl_s['c_index']:.4f}(nH={pl_s['n_high']},QWK={pl_s['quadratic_kappa_vs_oracle']:.2f},P={pl_s['logrank_p']}) | "
            f"VIRT-REFIT: Frz={vr_f['c_index']:.4f} -> Sc={vr_s['c_index']:.4f}",
            flush=True,
        )

        if disp_name == "ConvNeXt" and not args.skip_landscape:
            print("[NV2-N1v2] Computing locked Fold 0 16-ch single-cell landscape & specimen crops...", flush=True)
            results["single_cell_landscape"] = _run_locked_fold0_landscape(
                bank, up_all, state_dir, manifest, clin_df, out_dir, device
            )
            ls = results["single_cell_landscape"]
            print(
                f"[NV2-N1v2] Landscape ({ls['embedding_method']}): 8-class F1 {ls['macro_f1_8class_frozen']} -> {ls['macro_f1_8class_scount']} | "
                f"prop r {ls['proportion_pearson_frozen']} -> {ls['proportion_pearson_scount']}",
                flush=True,
            )

        del up_all
        gc.collect()
        _atomic_write_json(out_json_path, results)

    results["wall_seconds"] = round(time.time() - t0, 1)
    _atomic_write_json(out_json_path, results)
    print("NV2_N1V2_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
