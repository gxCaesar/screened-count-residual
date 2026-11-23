#!/usr/bin/env python3
"""Recompute Nature Methods multi-cohort confirmatory statistics from archived per-unit CSV tables."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from s2 import clinical_cell_survival as ccs


def _exact_wilcoxon_nonzero(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    wins = int(np.sum(diff > 0.0))
    nz = diff[diff != 0.0]
    p_val = float(stats.wilcoxon(nz, method="exact").pvalue) if len(nz) > 0 else 1.0
    return wins, int(len(diff)), p_val


def main() -> int:
    tables = ROOT / "results" / "tables"
    summary_path = ROOT / "results" / "confirmatory_summary.json"
    summary = json.loads(summary_path.read_text())

    df20 = pd.read_csv(tables / "all20_universal_backbones.csv")
    df_hem = pd.read_csv(tables / "hemit_per_backbone_summary.csv")
    df_ori = pd.read_csv(tables / "orion_per_backbone_summary.csv")
    df_her = pd.read_csv(tables / "her2_per_backbone_summary.csv")
    df_pat = pd.read_csv(tables / "orion_n40_patient_survival.csv")
    df_det = pd.read_csv(tables / "determinism_480_parity.csv")
    df_surv_corr = pd.read_csv(tables / "orion_n40_corrected_survival_bootstrap.csv")

    assert len(df20) == 20, f"Expected 20 pretrained generator instances, got {len(df20)}"
    assert len(df_hem) == 11 and len(df_ori) == 6 and len(df_her) == 3
    assert len(df_pat) == 40
    assert len(df_det) == 480 and float(df_det["abs_diff"].max()) == 0.0
    assert len(df_surv_corr) == 6

    d_fix_up = (df20["sc_Fixed_tol_F1"] - df20["up_Fixed_tol_F1"]).to_numpy(dtype=np.float64)
    wins_up = int(np.sum(d_fix_up > 0.0))
    mean_gain_up = float(np.mean(d_fix_up))
    p_wilcox = float(stats.wilcoxon(d_fix_up, method="exact").pvalue)

    cb = summary["cross_backbone_20"]
    assert wins_up == int(cb["wins_sc_vs_up_Fixed_tol_F1"])
    assert abs(mean_gain_up - float(cb["mean_delta_sc_vs_up_Fixed_tol_F1"])) < 1e-6
    assert abs(p_wilcox - float(cb["wilcoxon_p_sc_vs_up_Fixed_tol_F1"])) < 1e-9

    # Recompute Oracle PFS C-index directly from N = 40 patient rows (duplicate-free bootstrap)
    ora_res = ccs.evaluate_pfs_survival(
        pfs_days=df_pat["PFSDays"].to_numpy(dtype=np.float64),
        recurrence_events=df_pat["Recurrence"].to_numpy(dtype=np.int32),
        protective_score=df_pat["ImmunoScore"].to_numpy(dtype=np.float64),
        dichotomy_cutoff=2.0,
        n_bootstrap=500,
    )
    assert 0.5 < float(ora_res["c_index"]) <= 1.0

    # Verify full 5-fold Orion-CRC (30/30 wins) and 20-generator x 12-arm comparator table
    exp_dir = ROOT / "results" / "expansion"
    df_o5 = pd.read_csv(exp_dir / "orion_5fold_n40_16ch.csv")
    df_c12 = pd.read_csv(exp_dir / "comp12_all20.csv")
    assert len(df_o5) == 30, f"Expected 30 Orion 5-fold rows, got {len(df_o5)}"
    assert int((df_o5["sc_f1"] > df_o5["up_f1"]).sum()) == 30
    assert int((df_o5["sc_f1"] > df_o5["sr_f1"]).sum()) == 30
    assert len(df_c12) == 20 and int(df_c12["s_dice_f1"].notna().sum()) == 20
    assert int((df_c12["s_count_f1"] > df_c12["s_dice_f1"]).sum()) == 18
    assert int((df_c12["s_count_f1"] > df_c12["s_bce_f1"]).sum()) == 18

    # Verify zero-leakage 15-fold unseen-patient Orion provenance and 40/40 patient wins
    push_dir = ROOT / "results" / "nm_nature_push"
    df_prov = pd.read_csv(push_dir / "push3_orion_unseen_fold_provenance.csv")
    df_up = pd.read_csv(push_dir / "push3_orion_unseen_per_patient.csv")
    df_scores = pd.read_csv(push_dir / "push3_orion_unseen_patient_scores.csv")
    assert len(df_prov) == 15
    assert int(df_prov["train_test_patient_overlap"].max()) == 0
    assert int(df_prov["val_test_patient_overlap"].max()) == 0
    assert len(df_scores) == 40
    pat_mean = (df_up["S_count"] - df_up["upstream"]).groupby(df_up["patient"]).mean()
    assert len(pat_mean) == 40 and int((pat_mean > 0.0).sum()) == 40

    # Verify 100% file resolution across all 271 mapped Source Data panels inside results/source_data/
    sd_dir = ROOT / "results" / "source_data"
    df_pmap = pd.read_csv(sd_dir / "PANEL_TO_SOURCE_DATA.csv")
    assert len(df_pmap) == 271
    referenced_csvs: set[str] = set()
    for _, row in df_pmap.iterrows():
        referenced_csvs.add(str(row["primary_source_data_csv"]).strip())
        aux_raw = str(row["auxiliary_source_data_csvs"])
        if aux_raw and aux_raw != "nan":
            for tok in aux_raw.split(";"):
                tok = tok.strip()
                if tok:
                    referenced_csvs.add(tok)
    for csv_name in sorted(referenced_csvs):
        csv_path = sd_dir / csv_name
        assert csv_path.is_file(), f"Missing Source Data CSV referenced in PANEL_TO_SOURCE_DATA.csv: {csv_name}"
        assert len(pd.read_csv(csv_path)) > 0, f"Empty Source Data CSV: {csv_name}"

    # Verify v2.0.0 unified full-tile Orion survival & 18-arm frontier
    nv2_dir = ROOT / "results" / "nature_v2"
    v21_dir = ROOT / "results" / "nature_v21"
    df_surv_u = pd.read_csv(nv2_dir / "nv2_n1_orion_survival_unified.csv")
    df_c18 = pd.read_csv(nv2_dir / "nv2_n3_comp18_all20.csv")
    assert len(df_c18) == 20
    assert int((df_c18["s_count_f1"] > df_c18["isotonic_f1"]).sum()) == 20
    assert int((df_c18["s_count_f1"] > df_c18["s_dice_f1"]).sum()) == 18
    assert int((df_c18["s_count_f1"] > df_c18["s_bce_f1"]).sum()) == 18
    assert int(df_surv_u["mv_p_n40"].notna().sum()) == 56
    p2p_lock = df_surv_u[
        (df_surv_u["protocol"] == "physical_locked")
        & (df_surv_u["backbone_slug"] == "pix2pix_orion")
        & (df_surv_u["arm"] == "S_count")
    ]
    assert len(p2p_lock) == 1 and not bool(p2p_lock.iloc[0]["contrast_estimable"])
    vr_sc = df_surv_u[(df_surv_u["protocol"] == "virtual_refitted") & (df_surv_u["arm"] == "S_count")]
    assert len(vr_sc) == 6 and abs(float(vr_sc["c_index"].mean()) - 0.5982) < 5e-4

    # Recompute v2.1.0 P0 paired exact Wilcoxon stats directly from v21_comp19_all20.csv
    df_c19 = pd.read_csv(sd_dir / "v21_comp19_all20.csv")
    assert len(df_c19) == 20
    w20_zm, n20_zm, p20_zm = _exact_wilcoxon_nonzero(df_c19["s_count_matched_f1"], df_c19["e_count_zm_f1"])
    assert (w20_zm, n20_zm) == (9, 20) and abs(p20_zm - 0.32998) < 1e-4
    sr10 = df_c19[df_c19["s_count_matched_source"] == "v21_same_run"]
    assert len(sr10) == 10
    w10_ec, n10_ec, p10_ec = _exact_wilcoxon_nonzero(sr10["e_count_zm_f1"], sr10["e_count_v21_f1"])
    assert (w10_ec, n10_ec) == (7, 10) and abs(p10_ec - 0.375) < 1e-4
    sr14 = df_c19[df_c19["s_count_matched_source"].isin(("v21_same_run", "v21_separate_sc_run"))]
    w14_zm, n14_zm, p14_zm = _exact_wilcoxon_nonzero(sr14["s_count_matched_f1"], sr14["e_count_zm_f1"])
    assert (w14_zm, n14_zm) == (6, 14) and abs(p14_zm - 0.35754) < 1e-4

    # Recompute v2.1.0 P1 MIST 45 marker-slide AND collapsed 43 distinct-slide exact Wilcoxon tests directly from CSV
    df_m45 = pd.read_csv(sd_dir / "v21_p1_mist_perslide_44.csv")
    assert len(df_m45) == 810 and int(df_m45["pred_pos_pct"].notna().sum()) == 810
    assert int(df_m45["slide"].nunique()) == 45 and int(df_m45["raw_slide"].nunique()) == 43
    piv_f1 = df_m45.groupby(["slide", "arm"])["Fixed_tol_F1"].mean().unstack()
    piv_ss = df_m45.groupby(["slide", "arm"])["SSIM"].mean().unstack()
    gt_pos = df_m45.groupby("slide")["gt_pos_pct"].mean()
    pos34 = gt_pos[gt_pos > 0.05].index
    w34_frz, n34_frz, p34_frz = _exact_wilcoxon_nonzero(piv_f1.loc[pos34, "S_count"], piv_f1.loc[pos34, "upstream"])
    assert (w34_frz, n34_frz) == (31, 34) and abs(p34_frz - 1.537e-8) < 1e-10
    w45_ss, n45_ss, p45_ss = _exact_wilcoxon_nonzero(piv_ss["S_count"], piv_ss["E_count_zm"])
    assert (w45_ss, n45_ss) == (33, 45) and abs(p45_ss - 9.547e-5) < 1e-7
    # Collapsed 43 distinct unprefixed slides sensitivity
    piv_f1_43 = df_m45.groupby(["raw_slide", "arm"])["Fixed_tol_F1"].mean().unstack()
    piv_ss_43 = df_m45.groupby(["raw_slide", "arm"])["SSIM"].mean().unstack()
    gt_pos_43 = df_m45.groupby("raw_slide")["gt_pos_pct"].mean()
    pos32 = gt_pos_43[gt_pos_43 > 0.05].index
    w32_frz, n32_frz, p32_frz = _exact_wilcoxon_nonzero(piv_f1_43.loc[pos32, "S_count"], piv_f1_43.loc[pos32, "upstream"])
    w43_ss, n43_ss, p43_ss = _exact_wilcoxon_nonzero(piv_ss_43["S_count"], piv_ss_43["E_count_zm"])
    assert (w32_frz, n32_frz) == (29, 32) and abs(p32_frz - 3.539e-8) < 1e-10
    assert (w43_ss, n43_ss) == (33, 43) and abs(p43_ss - 4.655e-5) < 1e-7

    # Recompute v2.1.0 P2 Orion 40-patient (335,597 nuclei) exact Wilcoxon tests directly from per-patient JSON
    atlas_obj = json.loads((v21_dir / "v21_p2_orion40_5fold_atlas.json").read_text())["models"]["miphei_convnext"]
    assert int(atlas_obj["n_total_nuclei"]) == 335597 and len(atlas_obj["per_patient"]) == 40
    pats = sorted(atlas_obj["per_patient"].keys())
    sc_r = np.array([atlas_obj["per_patient"][p]["S_count"]["prop_r"] for p in pats])
    fr_r = np.array([atlas_obj["per_patient"][p]["Frozen"]["prop_r"] for p in pats])
    zm_r = np.array([atlas_obj["per_patient"][p]["E_count_zm"]["prop_r"] for p in pats])
    sc_cd8 = np.array([atlas_obj["per_patient"][p]["S_count"]["cd8_f1"] for p in pats])
    zm_cd8 = np.array([atlas_obj["per_patient"][p]["E_count_zm"]["cd8_f1"] for p in pats])
    sc_mf1 = np.array([atlas_obj["per_patient"][p]["S_count"]["macro_f1"] for p in pats])
    zm_mf1 = np.array([atlas_obj["per_patient"][p]["E_count_zm"]["macro_f1"] for p in pats])
    sc_htr = np.array([atlas_obj["per_patient"][p]["S_count"]["helper_treg_f1"] for p in pats])
    zm_htr = np.array([atlas_obj["per_patient"][p]["E_count_zm"]["helper_treg_f1"] for p in pats])
    w40_fr, _, p40_fr = _exact_wilcoxon_nonzero(sc_r, fr_r)
    w40_zmr, _, p40_zmr = _exact_wilcoxon_nonzero(sc_r, zm_r)
    w40_cd8, _, p40_cd8 = _exact_wilcoxon_nonzero(sc_cd8, zm_cd8)
    w40_mf1, _, p40_mf1 = _exact_wilcoxon_nonzero(sc_mf1, zm_mf1)
    w40_htr, _, p40_htr = _exact_wilcoxon_nonzero(sc_htr, zm_htr)
    assert w40_fr == 40 and abs(p40_fr - 1.819e-12) < 1e-14
    assert w40_zmr == 29 and abs(p40_zmr - 1.518e-3) < 1e-5
    assert w40_cd8 == 31 and abs(p40_cd8 - 9.447e-5) < 1e-7
    assert w40_mf1 == 12 and abs(p40_mf1 - 1.202e-2) < 1e-4
    assert w40_htr == 11 and abs(p40_htr - 3.520e-3) < 1e-5

    df_xd = pd.read_csv(sd_dir / "v21_p2_xdom_budget_curves.csv")
    assert len(df_xd) == 58

    # Recompute v2.2.0 Q1 (20/20 single-run 6-arm incl. L_count), Q2 (5-arch Orion-40), Q3 (MIST aligned + 21-slide gate)
    v22_dir = ROOT / "results" / "nature_v22"
    df_c20 = pd.read_csv(sd_dir / "v22_comp20_all20.csv")
    assert len(df_c20) == 20 and int(df_c20["v22_l_count_f1"].notna().sum()) == 20
    w20_lc, n20_lc, p20_lc = _exact_wilcoxon_nonzero(df_c20["v22_s_count_f1"], df_c20["v22_l_count_f1"])
    assert (w20_lc, n20_lc) == (10, 20) and abs(p20_lc - 0.78413) < 1e-4
    w20_sm_lc, _, p20_sm_lc = _exact_wilcoxon_nonzero(df_c20["v22_e_count_zm_smhi"], df_c20["v22_l_count_smhi"])
    w20_sm_sc, _, p20_sm_sc = _exact_wilcoxon_nonzero(df_c20["v22_e_count_zm_smhi"], df_c20["v22_s_count_smhi"])
    assert w20_sm_lc == 16 and abs(p20_sm_lc - 0.00486) < 1e-4
    assert w20_sm_sc == 14 and abs(p20_sm_sc - 0.02664) < 1e-4

    df_q2 = pd.read_csv(sd_dir / "v22_q2_orion40_5arch_atlas.csv")
    assert int(df_q2["model"].nunique()) == 5 and len(df_q2) == 20
    df_q2_pp = pd.read_csv(sd_dir / "v22_q2_orion40_5arch_per_patient.csv")
    assert len(df_q2_pp) == 200 and int(df_q2_pp["model"].nunique()) == 5
    exp_q2_sl = {
        "miphei_convnext": (30, 10, 0, 0.001367483206),
        "miphei_hemit_arch": (33, 7, 0, 5.723268259e-8),
        "unseen_unet16": (26, 14, 0, 0.0146776428),
        "unseen_resnet34_unet16": (19, 21, 0, 0.8471556244),
        "unseen_pix2pix16": (26, 14, 0, 0.002166251263),
    }
    for arch_k, (ew, el, et, ep) in exp_q2_sl.items():
        sa = df_q2_pp[df_q2_pp["model"] == arch_k]
        assert len(sa) == 40
        d_a = sa["S_count_macro_f1"].to_numpy(dtype=np.float64) - sa["L_count_macro_f1"].to_numpy(dtype=np.float64)
        w_a, l_a, t_a = int((d_a > 0).sum()), int((d_a < 0).sum()), int((d_a == 0).sum())
        w_nz, n_nz, p_a = _exact_wilcoxon_nonzero(sa["S_count_macro_f1"], sa["L_count_macro_f1"])
        assert (w_a, l_a, t_a) == (ew, el, et) and (w_nz, n_nz) == (ew, ew + el), f"{arch_k}: {(w_a, l_a, t_a)}"
        assert abs(p_a - ep) <= 1e-3 * ep, f"{arch_k}: p={p_a} vs {ep}"
    sa_r34 = df_q2_pp[df_q2_pp["model"] == "unseen_resnet34_unet16"]
    d_r34_t = sa_r34["S_count_tumor_f1"].to_numpy(dtype=np.float64) - sa_r34["Frozen_tumor_f1"].to_numpy(dtype=np.float64)
    w_r34_t, l_r34_t, t_r34_t = int((d_r34_t > 0).sum()), int((d_r34_t < 0).sum()), int((d_r34_t == 0).sum())
    _, n_r34_t, p_r34_t = _exact_wilcoxon_nonzero(sa_r34["S_count_tumor_f1"], sa_r34["Frozen_tumor_f1"])
    assert (w_r34_t, l_r34_t, t_r34_t, n_r34_t) == (2, 38, 0, 40), f"r34_tumor: {(w_r34_t, l_r34_t, t_r34_t)}"
    assert abs(p_r34_t - 1.386069926e-9) <= 1e-3 * 1.386069926e-9, f"r34_tumor p={p_r34_t}"

    df_m22_sl = pd.read_csv(sd_dir / "v22_q3_mist_perslide_aligned.csv")
    sv_al = df_m22_sl[df_m22_sl["view"] == "hematoxylin_aligned"]
    s43_al = sv_al.groupby(["raw_slide", "arm"], as_index=False)[["Fixed_tol_F1", "gt_pos_pct"]].mean()
    p43_al = s43_al.pivot(index="raw_slide", columns="arm", values="Fixed_tol_F1")
    pos31 = s43_al[s43_al["arm"] == "S_count"].set_index("raw_slide")["gt_pos_pct"] > 0.05
    w31_lc, n31_lc, p31_lc = _exact_wilcoxon_nonzero(p43_al.loc[pos31, "S_count"], p43_al.loc[pos31, "L_count"])
    w31_zm, n31_zm, p31_zm = _exact_wilcoxon_nonzero(p43_al.loc[pos31, "S_count"], p43_al.loc[pos31, "E_count_zm"])
    assert (w31_lc, n31_lc) == (24, 31) and abs(p31_lc - 0.004436) < 1e-4
    assert (w31_zm, n31_zm) == (21, 31) and abs(p31_zm - 0.022097) < 1e-4

    df_mg22 = pd.read_csv(sd_dir / "v22_q3_mist_slide_gate.csv")
    assert len(df_mg22) == 95 and "calib_n_pos_tiles" in df_mg22.columns
    mg_base = (df_mg22["calib_pos_pct"] >= 0.5) & (df_mg22["calib_n_pos_tiles"] >= 5) & (df_mg22["internal_val_neg_fp_sc"].notna()) & (df_mg22["internal_val_neg_fp_sc"] <= 2.0)
    mg_meas_25 = df_mg22[mg_base & (df_mg22["internal_val_f1_sc"] >= 0.25)]
    mg_meas_30 = df_mg22[mg_base & (df_mg22["internal_val_f1_sc"] >= 0.30)]
    mg_meas_40 = df_mg22[mg_base & (df_mg22["internal_val_f1_sc"] >= 0.40)]
    assert len(mg_meas_25) == 1 and len(mg_meas_30) == 1 and str(mg_meas_30.iloc[0]["marker"]) == "ER" and len(mg_meas_40) == 0

    df_x22 = pd.read_csv(sd_dir / "v22_q3_xdom_21slides_gate.csv")
    assert "calib_n_pos_tiles" in df_x22.columns
    exp_dutch = {
        "BCIStainer": {0.25: 3, 0.30: 2, 0.35: 2, 0.40: 0},
        "PyramidPix2Pix": {0.25: 3, 0.30: 3, 0.35: 2, 0.40: 1},
    }
    for g_k, thr_map in exp_dutch.items():
        sub_g = df_x22[(df_x22["generator"] == g_k) & (df_x22["arm"] == "S_count") & (df_x22["k_slides"].isin([1, 2]))]
        base_ok = (sub_g["calib_pos_pct"] >= 0.5) & (sub_g["calib_n_pos_tiles"] >= 5) & (sub_g["calib_internal_neg_fp_sc"].notna()) & (sub_g["calib_internal_neg_fp_sc"] <= 2.0)
        for thr, exp_cnt in thr_map.items():
            acc_t = sub_g[base_ok & (sub_g["calib_internal_val_f1_sc"] >= thr)]
            assert len(acc_t) == exp_cnt, f"{g_k} thr={thr}: got {len(acc_t)} != {exp_cnt}"
            if thr == 0.30:
                assert float(acc_t["test_Fixed_tol_F1"].mean()) > 0.14

    df_pre_mist = pd.read_csv(sd_dir / "v21_p1_mist_prealign_ods_burden.csv")
    pr_qm_pre = df_pre_mist[(df_pre_mist["marker"] == "PR") & (df_pre_mist["arm"] == "QuantileMatch")]
    assert len(pr_qm_pre) == 3 and abs(float(pr_qm_pre["burden_pearson_r"].mean()) - (-0.011857)) < 1e-4

    print(
        f"Verified 20 frozen generator instances across 3 primary + 3 external cohorts (v2.2.2): "
        f"wins={wins_up}/20, mean_delta={mean_gain_up:+.6f}, wilcoxon_p={p_wilcox:.3e}, "
        f"v22_single_run_20x6 (S_vs_L {w20_lc}/20 p={p20_lc:.3f}; SMHI<E_zm L={w20_sm_lc}/20 p={p20_sm_lc:.4f}, S={w20_sm_sc}/20 p={p20_sm_sc:.4f}), "
        f"orion_5fold_wins=30/30, unseen_orion_wins=40/40 (0% overlap across 15 folds), "
        f"orion40_5arch_atlas_nuclei=335597 (ConvNeXt vs L 30/40, W-Swin vs L 33/40, CD8a vs E_zm {w40_cd8}/40 p={p40_cd8:.2e}), "
        f"mist_aligned_43slides (S_vs_L {w31_lc}/31 p={p31_lc:.4f}; S_vs_E_zm {w31_zm}/31 p={p31_zm:.4f}), "
        f"xdom_21slide_gate (BCI 2/21 & P2P 3/21 accepted at ivF1>=0.30, F1>0.142), "
        f"determinism_480_max_diff={float(df_det['abs_diff'].max()):.1f}, "
        f"Orion N=40 recurrence oracle C-index={ora_res['c_index']:.4f}, "
        f"unified_virt_refit_S_count_mean_C={float(vr_sc['c_index'].mean()):.4f}, "
        f"resolved_source_data_csvs={len(referenced_csvs)}, mapped_panels={len(df_pmap)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
