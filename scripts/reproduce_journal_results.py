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

    print(
        f"Verified 20 frozen generator instances across 3 cohorts: "
        f"wins={wins_up}/20, mean_delta={mean_gain_up:+.6f}, wilcoxon_p={p_wilcox:.3e}, "
        f"determinism_480_max_diff={float(df_det['abs_diff'].max()):.1f}, "
        f"Orion N=40 oracle C-index={ora_res['c_index']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
