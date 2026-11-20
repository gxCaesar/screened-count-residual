#!/usr/bin/env python3
"""Aggregate all five Nature Methods / Nature-caliber expansion suites (PUSH-1..PUSH-5).

Produces verified summary JSONs and Source Data CSVs under experiments/20261120-nm-nature-push/:
  1. PUSH-1: 100% Bit-for-Bit Identical-Objective EXP-5v3 (2 archs x 3 LRs x 5 budgets)
  2. PUSH-2: True 100-tile/1000-tile streaming CG timing + Spectral Decomposition +
             12-Arm Biological Audit + Exploratory GlobalAffine+S_count + HEMIT TME Ecology
  3. PUSH-3: 100% Strictly Unseen-Patient 5-Fold End-to-End Orion-CRC (N=40, 3 fresh archs,
             16-ch pixel + 15-class nuclear + 15x15 co-occurrence + 40-patient PFS survival!)
  4. PUSH-4: Multi-Organ Clinical Ki67 Proliferation Index & Cell Segmentation (DeepLIIF:
             Lung n=473, Bladder n=125, Breast n=66; IHC->mIF and Hematoxylin->mIF)
  5. PUSH-5: Full 5,980-Tile HER2match Multi-Backbone & Negative-Slide Specificity Audit
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from s2 import clinical_cell_survival as ccs

PUSH_DIR = ROOT / "experiments" / "20261120-nm-nature-push"
CLIN_CSV = ROOT / "development" / "orion_crc40_clinical_and_summary.csv"
CV_MANIFEST = ROOT / "development" / "orion-patient-cv-manifest.json"


def _patient_bootstrap_ci(vals: np.ndarray, n_boot: int = 10000, seed: int = 20261120) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(vals)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_c_bootstrap(pfs_days: np.ndarray, events: np.ndarray, s_new: np.ndarray, s_ref: np.ndarray, n_boot: int = 10000, seed: int = 20261120) -> dict:
    """Duplicate-free patient bootstrap (excluding identical self-pairs) for paired Delta C."""
    rng = np.random.default_rng(seed)
    n = len(pfs_days)
    c_new = float(ccs.harrell_c_index(pfs_days, events, s_new))
    c_ref = float(ccs.harrell_c_index(pfs_days, events, s_ref))
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Duplicate-free C-index by passing original patient IDs to exclude self-pairs
        cn = float(ccs.harrell_c_index(pfs_days[idx], events[idx], s_new[idx], unit_ids=idx))
        cr = float(ccs.harrell_c_index(pfs_days[idx], events[idx], s_ref[idx], unit_ids=idx))
        if np.isfinite(cn) and np.isfinite(cr):
            deltas.append(cn - cr)
    arr = np.asarray(deltas, dtype=np.float64)
    return dict(
        c_new=round(c_new, 6),
        c_ref=round(c_ref, 6),
        delta_c=round(c_new - c_ref, 6),
        ci_low=round(float(np.quantile(arr, 0.025)), 6) if len(arr) else float("nan"),
        ci_high=round(float(np.quantile(arr, 0.975)), 6) if len(arr) else float("nan"),
    )


def aggregate_push3_orion_unseen() -> dict | None:
    p3_dir = PUSH_DIR / "push3_orion_unseen"
    if not p3_dir.is_dir():
        return None
    archs = ["unseen_unet16", "unseen_resnet34_unet16", "unseen_pix2pix16"]
    arms = ["upstream", "S_recon", "QuantileMatch", "E_count", "S_count"]

    fold_rows = []
    pat_rows = []
    prov_rows = []
    oof_scores = {a: {arm: {} for arm in arms} for a in archs}
    present_archs = []

    cv_man = json.loads(CV_MANIFEST.read_text()) if CV_MANIFEST.is_file() else {}
    folds_by_idx = {int(f["fold"]): f for f in cv_man.get("folds", [])}
    pat_to_fold = {}

    for arch in archs:
        files = [p3_dir / f"unseen5fold_{arch}_fold{f}.json" for f in range(5)]
        if not all(fp.is_file() for fp in files):
            continue
        present_archs.append(arch)
        for f_idx, fp in enumerate(files):
            rec = json.loads(fp.read_text())
            assert rec.get("zero_patient_overlap_verified") is True
            f_spec = folds_by_idx.get(f_idx, {})
            in_tr = sorted(f_spec.get("inner_train_slides", []))
            in_va = sorted(f_spec.get("inner_val_slides", []))
            out_te = sorted(rec["eval_patients"])
            if f_spec:
                assert sorted(f_spec["outer_test_slides"]) == out_te
            tr_ov = len(set(in_tr) & set(out_te))
            va_ov = len(set(in_va) & set(out_te))
            assert tr_ov == 0 and va_ov == 0
            for pid in out_te:
                pat_to_fold[pid] = f_idx
            uinfo = rec.get("upstream_info", {})
            prov_rows.append(dict(
                architecture=arch,
                fold=f_idx,
                upstream_params=int(uinfo.get("params", 0)),
                upstream_best_epoch=int(uinfo.get("best_epoch", 0)),
                n_inner_train_patients=len(in_tr),
                n_inner_val_patients=len(in_va),
                n_head_fit_patients=len(in_tr),
                n_outer_test_patients=len(out_te),
                n_inner_train_tiles=int(rec.get("n_inner_train_tiles", 0)),
                n_inner_val_tiles=int(rec.get("n_inner_val_tiles", 0)),
                n_head_fit_tiles=int(rec.get("n_head_fit_tiles", 0)),
                n_eval_tiles=int(rec.get("n_eval_tiles", 0)),
                train_test_patient_overlap=tr_ov,
                val_test_patient_overlap=va_ov,
                inner_train_patients=";".join(in_tr),
                inner_val_patients=";".join(in_va),
                head_fit_patients=";".join(in_tr),
                outer_test_patients=";".join(out_te),
            ))
            for arm in arms:
                ar = rec["arms"][arm]
                mp = ar["macro_pixel_16ch"]
                sc = ar["single_cell_15class"]
                fold_rows.append(dict(
                    architecture=arch,
                    fold=f_idx,
                    arm=arm,
                    Fixed_tol_F1=mp["Fixed_tol_F1"],
                    Exact_r0_F1=mp["Exact_r0_F1"],
                    ODS_F=mp["ODS_F"],
                    SSIM=mp["SSIM"],
                    PSNR_dB=mp["PSNR_dB"],
                    Cal_F1=mp["Cal_F1"],
                    Cell_15cls_F1=sc["macro_deployable_cell_f1"],
                    Cell_AUROC=sc["macro_cell_auroc"],
                    Cooc_MAE=sc["cooccurrence_mae"],
                    Max_DC_drift=ar.get("max_dc_drift", 0.0),
                ))
                for pid, sc_d in rec["oof_immunoscore"][arm].items():
                    oof_scores[arch][arm][pid] = sc_d["continuous"]
            for pid, p_arms in rec["per_patient_macro_fixed_f1"].items():
                pat_rows.append(dict(architecture=arch, fold=f_idx, patient=pid, **p_arms))

    if not fold_rows:
        return None

    df_f = pd.DataFrame(fold_rows)
    df_p = pd.DataFrame(pat_rows)
    df_f.to_csv(PUSH_DIR / "push3_orion_unseen_folds.csv", index=False)
    df_p.to_csv(PUSH_DIR / "push3_orion_unseen_per_patient.csv", index=False)
    pd.DataFrame(prov_rows).to_csv(PUSH_DIR / "push3_orion_unseen_fold_provenance.csv", index=False)

    # Grand summary per arm
    arm_summary = {}
    for arm in arms:
        sub = df_f[df_f["arm"] == arm]
        arm_summary[arm] = {
            c: round(float(sub[c].mean()), 6)
            for c in ("Fixed_tol_F1", "Exact_r0_F1", "ODS_F", "SSIM", "PSNR_dB", "Cal_F1", "Cell_15cls_F1", "Cooc_MAE")
        }
        arm_summary[arm]["Max_DC_drift_median"] = float(sub["Max_DC_drift"].median())

    # Fold win counts & per-patient 40/40 strictly unseen test
    piv_f = df_f.pivot(index=["architecture", "fold"], columns="arm", values="Fixed_tol_F1")
    n_folds_tot = len(piv_f)
    wins_vs_up = int((piv_f["S_count"] > piv_f["upstream"]).sum())
    wins_vs_sr = int((piv_f["S_count"] > piv_f["S_recon"]).sum())
    wins_vs_qm_f1 = int((piv_f["S_count"] > piv_f["QuantileMatch"]).sum())
    piv_ssim = df_f.pivot(index=["architecture", "fold"], columns="arm", values="SSIM")
    wins_vs_qm_ssim = int((piv_ssim["S_count"] > piv_ssim["QuantileMatch"]).sum())

    pat_avg = df_p.groupby("patient")[arms].mean()
    deltas_pat = (pat_avg["S_count"] - pat_avg["upstream"]).to_numpy()
    pm, pci_l, pci_h = _patient_bootstrap_ci(deltas_pat)
    w_stat, w_p = stats.wilcoxon(deltas_pat, alternative="two-sided")

    # 40-patient strictly unseen-patient clinical PFS survival evaluation!
    clin = pd.read_csv(CLIN_CSV)
    id_col = "PatientID" if "PatientID" in clin.columns else "slideName"
    clin = clin.sort_values(id_col).reset_index(drop=True)
    pfs_days = clin["PFSDays"].to_numpy(dtype=np.float64)
    rec_ev = clin["Recurrence"].to_numpy(dtype=np.int32)
    stage_num = clin["Stage"].map({"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}).to_numpy(dtype=np.float64)
    # Canonical MMR encoding matching s2/recompute_journal_claims.py:627 (missing coded intact for N=40 primary; N=36 complete-case also audited)
    mmr_def = (clin["MMRIHC"].fillna("Intact").astype(str).str.lower() != "intact").astype(np.float64).to_numpy()
    assert int(mmr_def.sum()) == 6, f"Expected 6 dMMR patients, got {int(mmr_def.sum())}"
    cc_mask = clin["MMRIHC"].notna().to_numpy()
    pats_ordered = clin[id_col].astype(str).tolist()

    # Export complete 40-patient continuous out-of-fold ImmunoScore table for independent verification
    score_df = pd.DataFrame(dict(
        PatientID=pats_ordered,
        outer_fold=[pat_to_fold.get(p, -1) for p in pats_ordered],
        PFSDays=pfs_days,
        Recurrence=rec_ev,
        Stage=clin["Stage"].astype(str).tolist(),
        MMRIHC=clin["MMRIHC"].fillna("Missing").astype(str).tolist(),
        mmr_deficient=mmr_def.astype(int),
    ))
    for arch in present_archs:
        for arm in arms:
            score_df[f"{arch}__{arm}_cont"] = [float(oof_scores[arch][arm][p]) for p in pats_ordered]
    score_df.to_csv(PUSH_DIR / "push3_orion_unseen_patient_scores.csv", index=False)

    surv_out = {}
    surv_rows = []
    for arch in present_archs:
        surv_out[arch] = {}
        up_vec = np.array([oof_scores[arch]["upstream"][p] for p in pats_ordered], dtype=np.float64)
        for arm in arms:
            vec = np.array([oof_scores[arch][arm][p] for p in pats_ordered], dtype=np.float64)
            sv = ccs.evaluate_pfs_survival(
                pfs_days=pfs_days, recurrence_events=rec_ev, protective_score=vec,
                dichotomy_cutoff=2.0, stage_numeric=stage_num, mmr_deficient=mmr_def,
                reference_protective_score=(up_vec if arm != "upstream" else None),
                n_bootstrap=10000,
            )
            # Also compute N=36 complete-case MMR sensitivity
            sv_cc = ccs.evaluate_pfs_survival(
                pfs_days=pfs_days[cc_mask], recurrence_events=rec_ev[cc_mask], protective_score=vec[cc_mask],
                dichotomy_cutoff=2.0, stage_numeric=stage_num[cc_mask], mmr_deficient=mmr_def[cc_mask],
                reference_protective_score=None, n_bootstrap=2000,
            )
            n_high = int(sv.get("n_high_risk", (vec <= 2.0).sum()))
            n_low = int(sv.get("n_low_risk", (vec > 2.0).sum()))
            km_estimable = (n_high > 0) and (n_low > 0)
            cont_estimable = float(np.std(vec)) > 1e-12
            sv["km_estimable"] = km_estimable
            sv["continuous_estimable"] = cont_estimable
            if not km_estimable:
                sv["hr_high_vs_low_risk"] = None
                sv["hr_high_vs_low_ci95"] = [None, None]
                sv["hr_high_vs_low_wald_p"] = None
                sv["logrank_p"] = None
                sv["logrank_chisq"] = None
            dc_val = float(sv.get("delta_c_index_vs_ref", 0.0)) if arm != "upstream" else 0.0
            dc_ci = sv.get("delta_c_index_ci95", [0.0, 0.0]) if arm != "upstream" else [0.0, 0.0]
            if isinstance(sv.get("multivariable_cox"), dict):
                sv["multivariable_cox"]["l2_penalty"] = 1e-4
                sv["multivariable_cox"]["mmr_covariate_separated_inestimable"] = True
                # Clear ridge-dependent inference on the zero-event separated MMR covariate (index 2)
                for ck in ("hr", "hr_ci_low", "hr_ci_high", "p_value"):
                    if isinstance(sv["multivariable_cox"].get(ck), list) and len(sv["multivariable_cox"][ck]) > 2:
                        sv["multivariable_cox"][ck][2] = None
            if not cont_estimable:
                sv["hr_per_sd_deficit"] = None
                sv["hr_per_sd_ci95"] = [None, None]
                sv["hr_per_sd_wald_p"] = None
                if isinstance(sv.get("multivariable_cox"), dict):
                    sv["multivariable_cox"]["score_estimable"] = False
                    for ck in ("beta", "se", "hr", "hr_ci_low", "hr_ci_high", "p_value"):
                        if isinstance(sv["multivariable_cox"].get(ck), list) and len(sv["multivariable_cox"][ck]) > 0:
                            sv["multivariable_cox"][ck][0] = None
                mv_hr = float("nan")
                mv_hr_cc = float("nan")
                sv["multivariable_cox_complete_case_n36_hr"] = None
            else:
                if isinstance(sv.get("multivariable_cox"), dict):
                    sv["multivariable_cox"]["score_estimable"] = True
                mv_hr = float(sv["multivariable_cox"]["hr"][0]) if isinstance(sv.get("multivariable_cox"), dict) and sv["multivariable_cox"].get("hr") else float("nan")
                mv_hr_cc = float(sv_cc["multivariable_cox"]["hr"][0]) if isinstance(sv_cc.get("multivariable_cox"), dict) and sv_cc["multivariable_cox"].get("hr") else float("nan")
                sv["multivariable_cox_complete_case_n36_hr"] = mv_hr_cc
            surv_out[arch][arm] = dict(survival=sv)
            surv_rows.append(dict(
                architecture=arch,
                arm=arm,
                n_high_risk=n_high,
                n_low_risk=n_low,
                km_estimable=km_estimable,
                continuous_estimable=cont_estimable,
                c_index=sv["c_index"],
                c_ci_low=sv["c_index_ci95"][0],
                c_ci_high=sv["c_index_ci95"][1],
                km_hr=(float(sv["hr_high_vs_low_risk"]) if km_estimable else float("nan")),
                km_logrank_p=(float(sv["logrank_p"]) if km_estimable else float("nan")),
                mv_cox_hr_n40=mv_hr,
                mv_cox_hr_n36=mv_hr_cc,
                delta_c_vs_up=dc_val,
                delta_c_ci_low=float(dc_ci[0]),
                delta_c_ci_high=float(dc_ci[1]),
            ))
    pd.DataFrame(surv_rows).to_csv(PUSH_DIR / "push3_orion_unseen_survival.csv", index=False)

    summary = dict(
        schema="push3-orion-unseen5fold-summary-v1",
        zero_upstream_patient_overlap=True,
        n_architectures=len(present_archs),
        architectures=present_archs,
        n_fold_evaluations=n_folds_tot,
        wins_S_count_vs_upstream=f"{wins_vs_up}/{n_folds_tot}",
        wins_S_count_vs_S_recon=f"{wins_vs_sr}/{n_folds_tot}",
        wins_S_count_vs_QuantileMatch_F1=f"{wins_vs_qm_f1}/{n_folds_tot}",
        wins_S_count_vs_QuantileMatch_SSIM=f"{wins_vs_qm_ssim}/{n_folds_tot}",
        n_unseen_patients=len(pat_avg),
        patients_positive=f"{int((deltas_pat > 0).sum())}/{len(deltas_pat)}",
        patient_mean_delta=round(pm, 6),
        patient_boot_ci95=[round(pci_l, 6), round(pci_h, 6)],
        patient_wilcoxon_p=float(w_p),
        arm_means=arm_summary,
        survival_N40_strictly_unseen=surv_out,
    )
    (PUSH_DIR / "push3_orion_unseen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def aggregate_push4_deepliif() -> dict | None:
    p = PUSH_DIR / "push4_deepliif_multiorgan.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    rows = []
    for s_name, s_obj in d.get("suites", {}).items():
        for bk, bk_obj in s_obj.get("backbones", {}).items():
            for arm, ar in bk_obj.get("arms", {}).items():
                mp = ar["macro_3ch"]
                sc = ar["single_cell_ki67"]
                ci = ar["clinical_ki67_index"]
                rows.append(dict(
                    suite=s_name,
                    backbone=bk,
                    arm=arm,
                    Fixed_tol_F1=mp["Fixed_tol_F1"],
                    Exact_r0_F1=mp["Exact_r0_F1"],
                    SSIM=mp["SSIM"],
                    PSNR_dB=mp["PSNR_dB"],
                    Ki67_px_F1=ar["per_channel"]["Ki67"]["Fixed_tol_F1"],
                    Ki67_cell_F1=sc["ki67_cell_f1"],
                    Ki67_cell_AUROC=sc["ki67_cell_auroc"],
                    Ki67_PI_MAE_pct=ci["pi_mae_pct"],
                    Ki67_PI_Pearson_r=ci["pi_pearson_r"],
                    StGallen_14pct_F1=ci["st_gallen"]["ge_14pct"]["f1"],
                    StGallen_20pct_F1=ci["st_gallen"]["ge_20pct"]["f1"],
                    Max_DC_drift=ar.get("max_dc_drift", 0.0),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(PUSH_DIR / "push4_deepliif_summary.csv", index=False)
    return dict(n_rows=len(df))


def aggregate_push5_her2_full5980() -> dict | None:
    files = sorted(PUSH_DIR.glob("push5_her2_full5980_*.json"))
    if not files:
        return None
    rows = []
    for fp in files:
        rec = json.loads(fp.read_text())
        arch = rec["architecture"]
        for arm, ar in rec["arms"].items():
            pm = ar["pooled_5980"]
            rows.append(dict(
                architecture=arch,
                arm=arm,
                Fixed_tol_F1=pm["Fixed_tol_F1"],
                Exact_r0_F1=pm["Exact_r0_F1"],
                SSIM=pm["SSIM"],
                PSNR_dB=pm["PSNR_dB"],
                Neg3_FP_area_pct=pm["neg3_slides_pred_pos_area_pct"],
                HER2_zero_FP_area_pct=pm["her2_zero_slides_pred_pos_area_pct"],
                Neg3_pixel_specificity=pm["neg3_slides_pixel_specificity"],
                Tile5980_burden_r=ar["tile_5980_burden_pearson_r"],
                Slide_grade_rho=ar["slide_grade_spearman_rho"],
                Max_DC_drift=ar["max_dc_drift"],
            ))
    df = pd.DataFrame(rows)
    df.to_csv(PUSH_DIR / "push5_her2_full5980_summary.csv", index=False)
    return dict(n_rows=len(df))


def aggregate_push1_parity() -> dict | None:
    p = PUSH_DIR / "push1_exp5_exact_parity.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    rows = []
    for arch, ad in d.get("architectures", {}).items():
        for b_name, b in sorted(ad.get("budgets", {}).items(), key=lambda kv: int(kv[0][1:])):
            n_b = int(b_name[1:])
            for arm_key, arm_lbl in (
                ("Frozen", "Frozen"),
                ("E2E_Recon", "E2E_Recon"),
                ("E2E_Count_best_by_train_comp", "E2E_Count_best"),
                ("Frozen_S_count", "Frozen_S_count"),
            ):
                if arm_key not in b:
                    continue
                ao = b[arm_key]
                m = ao["metrics"]
                rows.append(dict(
                    architecture=arch,
                    budget=b_name,
                    n_tiles=n_b,
                    arm=arm_lbl,
                    selected_lr=ao.get("selected_lr", ""),
                    trainable_params_2seed=int(ao.get("trainable_params_2seed", ao.get("registered_params_2seed_3ch", 0))),
                    active_params_2seed=int(ao.get("active_params_2seed_3ch", ao.get("trainable_params_2seed", 0))),
                    macro_fixed_f1=float(m["macro_pixel"]["Fixed_tol_F1"]),
                    macro_ods_f=float(m["macro_pixel"]["ODS_F"]),
                    macro_ssim=float(m["macro_pixel"]["SSIM"]),
                    cd3_pixel_f1=float(m["per_marker_pixel"]["CD3"]["Fixed_tol_F1"]),
                    cd3_cell_f1=float(m["cd3_cell_f1"]),
                    panck_cell_f1=float(m["panck_cell_f1"]),
                    peak_vram_mb=float(ao.get("peak_vram_mb", 0.0)),
                    train_wall_sec=float(ao.get("train_wall_sec", 0.0)),
                ))
            for lrk, lr_obj in sorted(b.get("E2E_Count_lr_sweep", {}).items()):
                m = lr_obj["metrics"]
                rows.append(dict(
                    architecture=arch,
                    budget=b_name,
                    n_tiles=n_b,
                    arm=f"E2E_Count_{lrk}",
                    selected_lr=lrk,
                    trainable_params_2seed=int(lr_obj.get("trainable_params_2seed", 0)),
                    active_params_2seed=int(lr_obj.get("trainable_params_2seed", 0)),
                    macro_fixed_f1=float(m["macro_pixel"]["Fixed_tol_F1"]),
                    macro_ods_f=float(m["macro_pixel"]["ODS_F"]),
                    macro_ssim=float(m["macro_pixel"]["SSIM"]),
                    cd3_pixel_f1=float(m["per_marker_pixel"]["CD3"]["Fixed_tol_F1"]),
                    cd3_cell_f1=float(m["cd3_cell_f1"]),
                    panck_cell_f1=float(m["panck_cell_f1"]),
                    peak_vram_mb=float(lr_obj.get("peak_vram_mb", 0.0)),
                    train_wall_sec=float(lr_obj.get("train_wall_sec", 0.0)),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(PUSH_DIR / "push1_exp5_exact_parity_summary.csv", index=False)
    return dict(n_rows=len(df), architectures=list(d.get("architectures", {}).keys()))


def aggregate_push2_all() -> dict | None:
    out = {}
    p_a = PUSH_DIR / "push2_timing_streaming.json"
    if p_a.is_file():
        da = json.loads(p_a.read_text())
        pd.DataFrame(da.get("profiles", [])).to_csv(PUSH_DIR / "push2_timing_streaming_summary.csv", index=False)
        out["part_a"] = True
    p_b = PUSH_DIR / "push2_spectral_and_12arm_bio.json"
    if p_b.is_file():
        db = json.loads(p_b.read_text())
        h_rows = []
        for bk, bobj in db.get("hemit_bio_audit", {}).items():
            for arm, aobj in bobj.get("arms", {}).items():
                h_rows.append(dict(cohort="HEMIT", backbone=bk, arm=arm, n_absent_tiles=bobj.get("n_cd3_absent_tiles", 0), **aobj))
        if h_rows:
            pd.DataFrame(h_rows).to_csv(PUSH_DIR / "push2_hemit_bio_audit.csv", index=False)
        o_rows = []
        for bk, bobj in db.get("orion_fold0_bio_and_hybrid", {}).items():
            for arm, aobj in bobj.get("arms", {}).items():
                o_rows.append(dict(cohort="Orion-CRC", backbone=bk, arm=arm, n_absent_tiles=bobj.get("n_cd8a_absent_tiles", 0), **aobj))
        if o_rows:
            pd.DataFrame(o_rows).to_csv(PUSH_DIR / "push2_orion_bio_hybrid.csv", index=False)
        sp_rows = []
        for k, v in db.get("spectral_decomposition", {}).items():
            sp_rows.append(dict(
                entry=k,
                dc_ratio_up_to_gt=v["dc_ratio_up_to_gt"],
                cell_scale_ratio_up_to_gt=v["cell_scale_ratio_up_to_gt"],
                gt_dc_power=v["gt"]["dc_power"],
                up_dc_power=v["upstream"]["dc_power"],
                gt_cell_power=v["gt"]["cell_scale_power"],
                up_cell_power=v["upstream"]["cell_scale_power"],
            ))
        if sp_rows:
            pd.DataFrame(sp_rows).to_csv(PUSH_DIR / "push2_spectral_decomposition.csv", index=False)
        out["part_b"] = True
    p_c = PUSH_DIR / "push2_hemit_tme_ecology.json"
    if p_c.is_file():
        dc = json.loads(p_c.read_text())
        c_rows = []
        for bk, bobj in dc.get("backbones", {}).items():
            for arm in ("upstream", "S_count"):
                if arm not in bobj:
                    continue
                ao = bobj[arm]
                c_rows.append(dict(
                    backbone=bk,
                    arm=arm,
                    phenotype_acc=ao["immune_phenotype_accuracy"],
                    phenotype_macro_f1=ao["immune_phenotype_macro_f1"],
                    phenotype_kappa=ao["immune_phenotype_kappa"],
                    crossk_10um_r=ao["multiscale_cross_k"]["r10um"]["pearson_r"],
                    crossk_25um_r=ao["multiscale_cross_k"]["r25um"]["pearson_r"],
                    crossk_50um_r=ao["multiscale_cross_k"]["r50um"]["pearson_r"],
                    crossk_25um_rho=ao["multiscale_cross_k"]["r25um"]["spearman_rho"],
                ))
        if c_rows:
            pd.DataFrame(c_rows).to_csv(PUSH_DIR / "push2_hemit_tme_ecology.csv", index=False)
        out["part_c"] = True
    return out or None


def aggregate_push6_cross_hospital() -> dict | None:
    p = PUSH_DIR / "push6_cross_hospital_bcistainer_her2match.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    rows = []
    for arm, ar in d.get("arms", {}).items():
        pm = ar["pooled_5980"]
        rows.append(dict(
            arm=arm,
            Fixed_tol_F1=pm["Fixed_tol_F1"],
            Exact_r0_F1=pm["Exact_r0_F1"],
            ODS_F=pm["ODS_F"],
            SSIM=pm["SSIM"],
            PSNR_dB=pm["PSNR_dB"],
            Neg3_FP_area_pct=pm["neg3_slides_pred_pos_area_pct"],
            HER2_zero_FP_area_pct=pm["her2_zero_slides_pred_pos_area_pct"],
            Tile5980_burden_r=ar["tile_5980_burden_pearson_r"],
            Slide_grade_rho=ar["slide_grade_spearman_rho"],
            Max_DC_drift=ar["max_dc_drift"],
        ))
    pd.DataFrame(rows).to_csv(PUSH_DIR / "push6_cross_hospital_bcistainer_summary.csv", index=False)
    return dict(n_rows=len(rows))


def main() -> int:
    s1 = aggregate_push1_parity()
    s2 = aggregate_push2_all()
    s3 = aggregate_push3_orion_unseen()
    s4 = aggregate_push4_deepliif()
    s5 = aggregate_push5_her2_full5980()
    s6 = aggregate_push6_cross_hospital()
    print("AGGREGATED:", dict(push1=s1, push2=s2, push3=bool(s3), push4=s4, push5=s5, push6=s6))
    if s3:
        print("  PUSH-3 Unseen Orion:", s3["wins_S_count_vs_upstream"], "patients:", s3["patients_positive"], "delta:", s3["patient_mean_delta"], s3["patient_boot_ci95"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
