#!/usr/bin/env python3
"""Aggregate all Nature Methods expansion experiments (EXP-1..EXP-4) without touching immutable S6 files.

Reads:
  - Immutable S6 summary tables & Fold-0 Orion JSONs under experiments/20261116-s6-journal-confirmatory/pulled/
  - Plan 2 ablation CSV under experiments/20261119-plan2-ablations/plan2_ablations_all20.csv
  - EXP-1 Orion 5-fold CV JSONs under experiments/20261120-nm-expansion/orion5fold/
  - EXP-2 Mechanistic suite JSONs under experiments/20261120-nm-expansion/mech/
  - EXP-3 4-comparator JSONs across 20 generators under experiments/20261120-nm-expansion/comp20/
  - EXP-4 Biophysical/spectral/microscopy audit under experiments/20261120-nm-expansion/biophysical/

Writes clean verified CSV/JSON tables under experiments/20261120-nm-expansion/.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
S6_DIR = ROOT / "experiments" / "20261116-s6-journal-confirmatory" / "pulled"
PLAN2_CSV = ROOT / "experiments" / "20261119-plan2-ablations" / "plan2_ablations_all20.csv"
EXP_DIR = ROOT / "experiments" / "20261120-nm-expansion"


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    nz = diff[np.abs(diff) > 1e-12]
    if len(nz) < 1:
        return 1.0
    try:
        return float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return 1.0


def aggregate_comp12() -> None:
    """Merge immutable S6 + Plan 2 + EXP-3 into complete 12-comparator x 20-generator master table."""
    t1 = pd.read_csv(S6_DIR / "tables" / "all20_universal_backbones.csv")
    p2 = pd.read_csv(PLAN2_CSV)
    comp_dir = EXP_DIR / "comp20"
    if not comp_dir.exists():
        print("[comp12] comp20 dir not yet present; skipping.")
        return

    comp_map: dict[tuple[str, str], dict] = {}
    for jf in sorted(comp_dir.glob("comp20_*.json")):
        doc = json.loads(jf.read_text())
        comp_map[(doc["cohort"], doc["backbone"])] = doc

    rows = []
    for _, r in t1.iterrows():
        coh, bk = str(r["cohort"]), str(r["backbone"])
        p2_row = p2[(p2["cohort"] == coh) & (p2["backbone"] == bk)]
        p2_r = p2_row.iloc[0] if len(p2_row) else None
        c3 = comp_map.get((coh, bk))

        def _get_c3(arm: str, metric: str) -> float:
            if c3 is None or arm not in c3["arms"]:
                return float("nan")
            a_doc = c3["arms"][arm]
            if coh == "HEMIT":
                if metric == "cell_f1":
                    return float(a_doc["single_cell"]["per_marker"]["CD3"]["deployable_cell_f1"])
                return float(a_doc["macro_pixel"][metric])
            elif coh == "Orion-CRC":
                if metric == "cell_f1":
                    return float(a_doc["single_cell_15class"]["macro_deployable_cell_f1"])
                return float(a_doc["macro_pixel_16ch"][metric])
            else:
                if metric == "cell_f1":
                    return float(a_doc["dab_burden_concordance"]["wsi_burden_pearson_r"])
                return float(a_doc["dab_pixel"][metric])

        rec = dict(
            cohort=coh,
            backbone=bk,
            family=str(r["family"]),
            # Fixed_tol_F1 across ALL 12 arms
            frozen_f1=float(r["up_Fixed_tol_F1"]),
            unsharp_f1=float(p2_r["unsharp_Fixed_tol_F1"]) if p2_r is not None else float("nan"),
            clahe_f1=float(p2_r["clahe_Fixed_tol_F1"]) if p2_r is not None else float("nan"),
            qmatch_f1=_get_c3("QuantileMatch", "Fixed_tol_F1"),
            affgamma_f1=_get_c3("AffineGamma", "Fixed_tol_F1"),
            s_recon_f1=float(r["sr_Fixed_tol_F1"]),
            s_recon_comp_f1=float(p2_r["srcomp_Fixed_tol_F1"]) if p2_r is not None else float("nan"),
            s_dice_f1=_get_c3("S_dice", "Fixed_tol_F1"),
            s_bce_f1=_get_c3("S_bce", "Fixed_tol_F1"),
            e_count_f1=float(p2_r["ecount_Fixed_tol_F1"]) if p2_r is not None else float("nan"),
            s_count_1off_f1=float(p2_r["sc1off_Fixed_tol_F1"]) if p2_r is not None else float("nan"),
            s_count_f1=float(r["sc_Fixed_tol_F1"]),
            # SSIM across ALL 12 arms
            frozen_ssim=float(r["up_SSIM"]),
            unsharp_ssim=float(p2_r["unsharp_SSIM"]) if p2_r is not None else float("nan"),
            clahe_ssim=float(p2_r["clahe_SSIM"]) if p2_r is not None else float("nan"),
            qmatch_ssim=_get_c3("QuantileMatch", "SSIM"),
            affgamma_ssim=_get_c3("AffineGamma", "SSIM"),
            s_recon_ssim=float(r["sr_SSIM"]),
            s_recon_comp_ssim=float(p2_r["srcomp_SSIM"]) if p2_r is not None else float("nan"),
            s_dice_ssim=_get_c3("S_dice", "SSIM"),
            s_bce_ssim=_get_c3("S_bce", "SSIM"),
            e_count_ssim=float(p2_r["ecount_SSIM"]) if p2_r is not None else float("nan"),
            s_count_1off_ssim=float(p2_r["sc1off_SSIM"]) if p2_r is not None else float("nan"),
            s_count_ssim=float(r["sc_SSIM"]),
            # PSNR across key arms
            frozen_psnr=float(r["up_PSNR_dB"]),
            qmatch_psnr=_get_c3("QuantileMatch", "PSNR_dB"),
            affgamma_psnr=_get_c3("AffineGamma", "PSNR_dB"),
            s_recon_psnr=float(r["sr_PSNR_dB"]),
            s_dice_psnr=_get_c3("S_dice", "PSNR_dB"),
            s_bce_psnr=_get_c3("S_bce", "PSNR_dB"),
            e_count_psnr=float(p2_r["ecount_PSNR_dB"]) if p2_r is not None else float("nan"),
            s_count_psnr=float(r["sc_PSNR_dB"]),
            # ODS_F across key arms
            frozen_ods=float(r["up_ODS_F"]),
            qmatch_ods=_get_c3("QuantileMatch", "ODS_F"),
            affgamma_ods=_get_c3("AffineGamma", "ODS_F"),
            s_recon_ods=float(r["sr_ODS_F"]),
            s_dice_ods=_get_c3("S_dice", "ODS_F"),
            s_bce_ods=_get_c3("S_bce", "ODS_F"),
            e_count_ods=float(p2_r["ecount_ODS_F"]) if p2_r is not None else float("nan"),
            s_count_ods=float(r["sc_ODS_F"]),
            # Downstream cell/burden metric
            qmatch_down=_get_c3("QuantileMatch", "cell_f1"),
            affgamma_down=_get_c3("AffineGamma", "cell_f1"),
            s_dice_down=_get_c3("S_dice", "cell_f1"),
            s_bce_down=_get_c3("S_bce", "cell_f1"),
        )
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(EXP_DIR / "comp12_all20.csv", index=False)

    arms_order = [
        ("frozen", "Frozen generator"),
        ("unsharp", "Unsharp mask"),
        ("clahe", "CLAHE"),
        ("qmatch", "QuantileMatch (CDF)"),
        ("affgamma", "AffineGamma (monotone)"),
        ("s_recon", "Matched recon (S_recon)"),
        ("s_recon_comp", "Composite-selected recon"),
        ("s_dice", "Soft-Dice head (S_dice)"),
        ("s_bce", "Focal-BCE head (S_bce)"),
        ("e_count", "Unscreened count (E_count)"),
        ("s_count_1off", "1-px Laplacian (S_count-1px)"),
        ("s_count", "Screened count (S_count)"),
    ]
    summary = {"n_generators": int(len(df)), "n_comp20_loaded": int(len(comp_map)), "arms": {}}
    sc_f1 = df["s_count_f1"].to_numpy()
    sc_ssim = df["s_count_ssim"].to_numpy()
    for key, label in arms_order:
        f1 = df[f"{key}_f1"].to_numpy()
        ss = df[f"{key}_ssim"].to_numpy()
        valid = np.isfinite(f1)
        wins = int(np.sum(sc_f1[valid] > f1[valid]))
        summary["arms"][key] = dict(
            label=label,
            n_valid=int(valid.sum()),
            mean_f1=float(np.nanmean(f1)),
            sd_f1=float(np.nanstd(f1, ddof=1)) if valid.sum() > 1 else 0.0,
            mean_ssim=float(np.nanmean(ss)),
            mean_delta_sc_minus_arm_f1=float(np.nanmean(sc_f1[valid] - f1[valid])) if valid.sum() else float("nan"),
            mean_delta_sc_minus_arm_ssim=float(np.nanmean(sc_ssim[valid] - ss[valid])) if valid.sum() else float("nan"),
            sc_wins_f1=wins,
            wilcoxon_p_f1=_wilcoxon_p(sc_f1[valid], f1[valid]) if valid.sum() >= 3 and key != "s_count" else None,
            wilcoxon_p_ssim=_wilcoxon_p(sc_ssim[valid], ss[valid]) if valid.sum() >= 3 and key != "s_count" else None,
        )
    (EXP_DIR / "comp12_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[comp12] Wrote {len(df)} rows ({len(comp_map)}/20 new comp20 JSONs loaded).")


def aggregate_orion_5fold() -> None:
    """Combine immutable S6 Fold 0 + EXP-1 Folds 1..4 across all 6 Orion backbones and all 40 patients."""
    o5_dir = EXP_DIR / "orion5fold"
    if not o5_dir.exists():
        print("[orion5fold] dir not yet present; skipping.")
        return

    bks = ["miphei_convnext", "miphei_hemit_arch", "hemit_orion", "pix2pix_orion", "diffusion_ft", "rosie_orion"]
    fold_rows = []
    pat_rows = []
    per_marker_accum: dict[str, dict[str, list[float]]] = {}

    # Fold 0 per-patient directly from immutable S6 orion_per_patient_fold0.csv
    f0_pat_csv = S6_DIR / "tables" / "orion_per_patient_fold0.csv"
    if f0_pat_csv.exists():
        df_f0p = pd.read_csv(f0_pat_csv)
        for _, pr in df_f0p.iterrows():
            pat_rows.append(dict(
                backbone=str(pr["backbone"]),
                fold=0,
                patient=str(pr["patient_id"]),
                up_f1=float(pr["up_Fixed_tol_F1"]),
                sr_f1=float(pr["sr_Fixed_tol_F1"]),
                sc_f1=float(pr["sc_Fixed_tol_F1"]),
            ))

    for bk in bks:
        # Fold 0 from immutable S6 JSON
        f0_path = S6_DIR / "orion" / f"orion_backbone_{bk}_s6_confirmatory.json"
        if f0_path.exists():
            f0 = json.loads(f0_path.read_text())
            arms0 = f0["fold0_16ch_arms"]
            sr_k0 = "S_recon:ens2" if "S_recon:ens2" in arms0 else "S_recon"
            sc_k0 = "S_count:ens2" if "S_count:ens2" in arms0 else "S_count"
            fold_rows.append(dict(
                backbone=bk, fold=0, source="S6_frozen",
                n_eval_tiles=int(f0.get("n_eval_tiles", 684)),
                up_f1=float(arms0["upstream"]["macro_pixel_16ch"]["Fixed_tol_F1"]),
                sr_f1=float(arms0[sr_k0]["macro_pixel_16ch"]["Fixed_tol_F1"]),
                sc_f1=float(arms0[sc_k0]["macro_pixel_16ch"]["Fixed_tol_F1"]),
                up_ods=float(arms0["upstream"]["macro_pixel_16ch"]["ODS_F"]),
                sr_ods=float(arms0[sr_k0]["macro_pixel_16ch"]["ODS_F"]),
                sc_ods=float(arms0[sc_k0]["macro_pixel_16ch"]["ODS_F"]),
                up_ssim=float(arms0["upstream"]["macro_pixel_16ch"]["SSIM"]),
                sr_ssim=float(arms0[sr_k0]["macro_pixel_16ch"]["SSIM"]),
                sc_ssim=float(arms0[sc_k0]["macro_pixel_16ch"]["SSIM"]),
                up_cell=float(arms0["upstream"]["single_cell_15class"]["macro_deployable_cell_f1"]),
                sr_cell=float(arms0[sr_k0]["single_cell_15class"]["macro_deployable_cell_f1"]),
                sc_cell=float(arms0[sc_k0]["single_cell_15class"]["macro_deployable_cell_f1"]),
                up_cooc_mae=float("nan"), sr_cooc_mae=float("nan"), sc_cooc_mae=float("nan"),
            ))
            pm0 = arms0[sc_k0].get("per_marker_pixel", {})
            for m_nm, m_doc in pm0.items():
                per_marker_accum.setdefault(m_nm, {"up": [], "sr": [], "sc": []})
                per_marker_accum[m_nm]["up"].append(float(arms0["upstream"]["per_marker_pixel"][m_nm]["Fixed_tol_F1"]))
                per_marker_accum[m_nm]["sr"].append(float(arms0[sr_k0]["per_marker_pixel"][m_nm]["Fixed_tol_F1"]))
                per_marker_accum[m_nm]["sc"].append(float(m_doc["Fixed_tol_F1"]))

        for fi in (1, 2, 3, 4):
            fp = o5_dir / f"orion5fold_{bk}_fold{fi}.json"
            if not fp.exists():
                continue
            fd = json.loads(fp.read_text())
            ar = fd["arms"]
            fold_rows.append(dict(
                backbone=bk, fold=fi, source="EXP1_5fold",
                n_eval_tiles=int(fd["n_eval_tiles"]),
                up_f1=float(ar["upstream"]["macro_pixel_16ch"]["Fixed_tol_F1"]),
                sr_f1=float(ar["S_recon"]["macro_pixel_16ch"]["Fixed_tol_F1"]),
                sc_f1=float(ar["S_count"]["macro_pixel_16ch"]["Fixed_tol_F1"]),
                up_ods=float(ar["upstream"]["macro_pixel_16ch"]["ODS_F"]),
                sr_ods=float(ar["S_recon"]["macro_pixel_16ch"]["ODS_F"]),
                sc_ods=float(ar["S_count"]["macro_pixel_16ch"]["ODS_F"]),
                up_ssim=float(ar["upstream"]["macro_pixel_16ch"]["SSIM"]),
                sr_ssim=float(ar["S_recon"]["macro_pixel_16ch"]["SSIM"]),
                sc_ssim=float(ar["S_count"]["macro_pixel_16ch"]["SSIM"]),
                up_cell=float(ar["upstream"]["single_cell_15class"]["macro_deployable_cell_f1"]),
                sr_cell=float(ar["S_recon"]["single_cell_15class"]["macro_deployable_cell_f1"]),
                sc_cell=float(ar["S_count"]["single_cell_15class"]["macro_deployable_cell_f1"]),
                up_cooc_mae=float(ar["upstream"]["single_cell_15class"]["cooccurrence_mae"]),
                sr_cooc_mae=float(ar["S_recon"]["single_cell_15class"]["cooccurrence_mae"]),
                sc_cooc_mae=float(ar["S_count"]["single_cell_15class"]["cooccurrence_mae"]),
            ))
            for pid, pvals in sorted(fd.get("per_patient_macro_fixed_f1", {}).items()):
                pat_rows.append(dict(
                    backbone=bk, fold=fi, patient=pid,
                    up_f1=float(pvals["upstream"]),
                    sr_f1=float(pvals["S_recon"]),
                    sc_f1=float(pvals["S_count"]),
                ))
            for m_nm, m_doc in ar["S_count"]["per_marker_pixel"].items():
                per_marker_accum.setdefault(m_nm, {"up": [], "sr": [], "sc": []})
                per_marker_accum[m_nm]["up"].append(float(ar["upstream"]["per_marker_pixel"][m_nm]["Fixed_tol_F1"]))
                per_marker_accum[m_nm]["sr"].append(float(ar["S_recon"]["per_marker_pixel"][m_nm]["Fixed_tol_F1"]))
                per_marker_accum[m_nm]["sc"].append(float(m_doc["Fixed_tol_F1"]))

    df_f = pd.DataFrame(fold_rows)
    df_f.to_csv(EXP_DIR / "orion_5fold_n40_16ch.csv", index=False)
    df_p = pd.DataFrame(pat_rows)
    df_p.to_csv(EXP_DIR / "orion_5fold_per_patient.csv", index=False)

    per_bk = {}
    for bk, grp in df_f.groupby("backbone", sort=False):
        per_bk[bk] = dict(
            n_folds=int(len(grp)),
            total_eval_tiles=int(grp["n_eval_tiles"].sum()),
            up_f1_mean=float(grp["up_f1"].mean()),
            up_f1_sd=float(grp["up_f1"].std(ddof=1)) if len(grp) > 1 else 0.0,
            sr_f1_mean=float(grp["sr_f1"].mean()),
            sr_f1_sd=float(grp["sr_f1"].std(ddof=1)) if len(grp) > 1 else 0.0,
            sc_f1_mean=float(grp["sc_f1"].mean()),
            sc_f1_sd=float(grp["sc_f1"].std(ddof=1)) if len(grp) > 1 else 0.0,
            delta_f1_vs_up=float((grp["sc_f1"] - grp["up_f1"]).mean()),
            delta_f1_vs_sr=float((grp["sc_f1"] - grp["sr_f1"]).mean()),
            up_cell_mean=float(grp["up_cell"].mean()),
            sr_cell_mean=float(grp["sr_cell"].mean()),
            sc_cell_mean=float(grp["sc_cell"].mean()),
            delta_cell_vs_up=float((grp["sc_cell"] - grp["up_cell"]).mean()),
            sc_ssim_mean=float(grp["sc_ssim"].mean()),
            up_ssim_mean=float(grp["up_ssim"].mean()),
            folds_sc_gt_up=int((grp["sc_f1"] > grp["up_f1"]).sum()),
            folds_sc_gt_sr=int((grp["sc_f1"] > grp["sr_f1"]).sum()),
        )

    summary = dict(
        total_fold_evals=int(len(df_f)),
        total_folds_sc_gt_up=int((df_f["sc_f1"] > df_f["up_f1"]).sum()),
        total_folds_sc_gt_sr=int((df_f["sc_f1"] > df_f["sr_f1"]).sum()),
        total_folds_cell_sc_gt_up=int((df_f["sc_cell"] > df_f["up_cell"]).sum()),
        grand_mean_up_f1=float(df_f["up_f1"].mean()),
        grand_mean_sr_f1=float(df_f["sr_f1"].mean()),
        grand_mean_sc_f1=float(df_f["sc_f1"].mean()),
        grand_delta_f1_vs_up=float((df_f["sc_f1"] - df_f["up_f1"]).mean()),
        grand_delta_f1_vs_sr=float((df_f["sc_f1"] - df_f["sr_f1"]).mean()),
        grand_mean_up_cell=float(df_f["up_cell"].mean()),
        grand_mean_sc_cell=float(df_f["sc_cell"].mean()),
        grand_delta_cell_vs_up=float((df_f["sc_cell"] - df_f["up_cell"]).mean()),
        wilcoxon_p_30folds_vs_up=_wilcoxon_p(df_f["sc_f1"], df_f["up_f1"]),
        wilcoxon_p_30folds_vs_sr=_wilcoxon_p(df_f["sc_f1"], df_f["sr_f1"]),
        per_backbone=per_bk,
        per_marker_30fold_mean={
            m: dict(up=float(np.mean(v["up"])), sr=float(np.mean(v["sr"])), sc=float(np.mean(v["sc"])), delta=float(np.mean(v["sc"]) - np.mean(v["up"])))
            for m, v in per_marker_accum.items()
        },
    )
    (EXP_DIR / "orion_5fold_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[orion5fold] Wrote {len(df_f)}/30 fold evaluations across {df_f['backbone'].nunique()} backbones.")


def summarize_extra_suites() -> None:
    """Summarize EXP-2 (mech), EXP-4 (biophysical), and EXP-5/6/7 (killer suite) into a unified digest."""
    digest = {}
    for sub, files in (
        ("biophysical", ["tolerance_radius_sweep.json", "threshold_curves_and_histograms.json", "spectral_and_cg_audit.json", "single_cell_deep_audit.json"]),
        ("mech", ["mech_2a_data_efficiency.json", "mech_2b_feature_ablation.json", "mech_2c_offset_scale_sweep.json", "mech_2d_lambda_pareto.json", "mech_2e_cross_generator_transfer.json", "mech_2f_stain_perturbation.json", "mech_2g_throughput_scaling.json"]),
        ("killer", ["exp5_e2e_vs_frozen.json", "exp6_multiseed_stability.json", "exp7a_cross_cohort_zeroshot.json", "exp7b_quant_calibration_integrity.json"]),
    ):
        d = EXP_DIR / sub
        present = [f for f in files if (d / f).exists()]
        digest[sub] = dict(present=present, missing=[f for f in files if f not in present])
    (EXP_DIR / "expansion_inventory.json").write_text(json.dumps(digest, indent=2) + "\n")
    print("[inventory]", json.dumps(digest))


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_comp12()
    aggregate_orion_5fold()
    summarize_extra_suites()


if __name__ == "__main__":
    main()
