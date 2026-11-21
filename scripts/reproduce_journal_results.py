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
    p_wilcox = float(stats.wilcoxon(d_fix_up).pvalue)

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

    # Verify v2.0.0 unified full-tile Orion survival, 18-arm comparator frontier, and 6-cohort summary
    nv2_dir = ROOT / "results" / "nature_v2"
    df_surv_u = pd.read_csv(nv2_dir / "nv2_n1_orion_survival_unified.csv")
    df_c18 = pd.read_csv(nv2_dir / "nv2_n3_comp18_all20.csv")
    nv2_master = json.loads((nv2_dir / "NV2_MASTER_SUMMARY.json").read_text())
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
    assert int(nv2_master["n_mapped_panels"]) == 268

    print(
        f"Verified 20 frozen generator instances across 3 primary + 3 external cohorts (v2.0.0): "
        f"wins={wins_up}/20, mean_delta={mean_gain_up:+.6f}, wilcoxon_p={p_wilcox:.3e}, "
        f"orion_5fold_wins=30/30, unseen_orion_wins=40/40 (0% overlap across 15 folds), "
        f"comp18_vs_dice=18/20, comp18_vs_bce=18/20, comp18_vs_isotonic=20/20, "
        f"determinism_480_max_diff={float(df_det['abs_diff'].max()):.1f}, "
        f"Orion N=40 recurrence oracle C-index={ora_res['c_index']:.4f}, "
        f"unified_virt_refit_S_count_mean_C={float(vr_sc['c_index'].mean()):.4f}, "
        f"mapped_panels={int(nv2_master['n_mapped_panels'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
