"""Aggregate S6 confirmatory results into per-unit CSV tables and recompute claims from raw units.

Recomputes all confirmatory tables and verification metrics from raw per-unit receipts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import clinical_cell_survival as ccs

HEMIT_ORDER = (
    "native_dgr",
    "native_unistainnet_uni2h",
    "native_hemit_fml",
    "native_hybrid_donor",
    "sanofi_miphei_vit",
    "sanofi_unetr_hoptimus",
    "sanofi_miphei_convnext",
    "sanofi_miphei_hemit",
    "sanofi_hemit_swin",
    "sanofi_pix2pix",
    "sanofi_diffusion_ft",
)

ORION_ORDER = (
    "miphei_convnext",
    "miphei_hemit_arch",
    "hemit_orion",
    "pix2pix_orion",
    "diffusion_ft",
    "rosie_orion",
)

HER2_ORDER = (
    "unet_l1_ssim",
    "pix2pix_cgan_ihc",
    "hoptimus_lora_ihc",
)

PIXEL_METRICS = (
    "Fixed_tol_F1",
    "Cal_F1",
    "ODS_F",
    "AP",
    "SSIM",
    "PSNR_dB",
    "mismatch_gap_ods_minus_fixed",
)


def _ci_bounds(bobj: dict) -> tuple[float, float]:
    if "ci95" in bobj and isinstance(bobj["ci95"], (list, tuple)) and len(bobj["ci95"]) >= 2:
        return float(bobj["ci95"][0]), float(bobj["ci95"][1])
    return float(bobj.get("ci95_low", 0.0)), float(bobj.get("ci95_high", 0.0))


def _cell_metric(c_obj: dict, primary: str, fallback: str) -> float:
    if primary in c_obj:
        return float(c_obj[primary])
    return float(c_obj.get(fallback, 0.0))


def _extract_surv_fields(sobj: dict) -> dict[str, float]:
    c_idx = float(sobj["c_index"])
    c_lo = float(sobj["c_index_ci95"][0])
    c_hi = float(sobj["c_index_ci95"][1])
    km_hr = float(sobj.get("hr_high_vs_low_risk", sobj.get("km_hazard_ratio", float("nan"))))
    km_p = float(sobj.get("logrank_p", sobj.get("km_logrank_p", float("nan"))))
    cont_hr = float(sobj.get("hr_per_sd_deficit", float("nan")))
    cont_p = float(sobj.get("hr_per_sd_wald_p", float("nan")))
    mv_hr = float("nan")
    mv_p = float("nan")
    if "multivariable_cox" in sobj and isinstance(sobj["multivariable_cox"], dict):
        mv = sobj["multivariable_cox"]
        if mv.get("hr"):
            mv_hr = float(mv["hr"][0])
        if mv.get("p_value"):
            mv_p = float(mv["p_value"][0])
    elif "cox_multivariable_stage_mmr" in sobj and isinstance(sobj["cox_multivariable_stage_mmr"], dict):
        mv = sobj["cox_multivariable_stage_mmr"]
        mv_hr = float(mv.get("immunoscore_hr", float("nan")))
        mv_p = float(mv.get("immunoscore_p", float("nan")))
    return dict(
        c_index=c_idx,
        c_lo=c_lo,
        c_hi=c_hi,
        km_hr=km_hr,
        km_p=km_p,
        cont_hr=cont_hr,
        cont_p=cont_p,
        mv_hr=mv_hr,
        mv_p=mv_p,
    )


def _load_backbone_jsons(folder: Path, prefix: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not folder.is_dir():
        return out
    for p in sorted(folder.glob(f"{prefix}*.json")):
        if p.name.endswith("_summary.json"):
            continue
        doc = json.loads(p.read_text())
        if isinstance(doc, dict) and "backbone" in doc:
            out[str(doc["backbone"])] = doc
    return out


def build_confirmatory_tables(
    *,
    hemit_dir: Path,
    orion_dir: Path,
    her2_dir: Path,
    clinical_csv: Path,
    out_dir: Path,
) -> dict:
    """Build per-unit CSV tables, determinism payload, and analysis summary."""
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    hemit_docs = _load_backbone_jsons(hemit_dir, "backbone_")
    orion_docs = _load_backbone_jsons(orion_dir, "orion_backbone_")
    her2_docs = _load_backbone_jsons(her2_dir, "her2_backbone_")

    orion_summary_files = sorted(orion_dir.glob("orion_multibackbone_*_summary.json"))
    oracle_surv = None
    for sf in orion_summary_files:
        sdoc = json.loads(sf.read_text())
        if sdoc.get("oracle_survival_n40"):
            oracle_surv = sdoc["oracle_survival_n40"]
            break

    # 1. HEMIT tables
    hemit_keys = [k for k in HEMIT_ORDER if k in hemit_docs] + [
        k for k in sorted(hemit_docs) if k not in HEMIT_ORDER
    ]
    hemit_summary_rows = []
    hemit_region_rows = []
    hemit_marker_rows = []
    hemit_cell_rows = []
    det_records = []

    for bk in hemit_keys:
        d = hemit_docs[bk]
        fam = str(d["family"])
        arms = d["arms"]
        up = arms["upstream"]
        sr = arms["S_recon:ens2"]
        sc = arms["S_count:ens2"]
        boot_up = d["paired_cluster_bootstrap"]["scount_ens2_vs_upstream"]
        boot_sr = d["paired_cluster_bootstrap"]["scount_ens2_vs_srecon_ens2"]

        row = {
            "cohort": "HEMIT",
            "backbone": bk,
            "family": fam,
            "wall_seconds": float(d.get("wall_seconds", 0.0)),
        }
        for tag, arm_obj in (("up", up), ("sr", sr), ("sc", sc)):
            for mk in PIXEL_METRICS:
                row[f"{tag}_{mk}"] = float(arm_obj["macro_pixel"][mk])
            scell = arm_obj["single_cell"]
            row[f"{tag}_cell_dep_f1"] = float(scell["macro_deployable_cell_f1"])
            row[f"{tag}_cell_rec_f1"] = float(scell["macro_recalibrated_cell_f1"])
            row[f"{tag}_cell_auprc"] = float(scell["macro_cell_auprc"])
            row[f"{tag}_cell_gap"] = float(scell["cell_mismatch_gap"])
            inf = scell.get("spatial_infiltration") or scell.get("spatial_panck_cd3_infiltration") or {}
            row[f"{tag}_w25_r"] = float(inf.get("within_25um_pearson_r", 0.0))
            row[f"{tag}_w25_mae"] = float(inf.get("within_25um_mae", 0.0))

        row["delta_sc_vs_up_Fixed_tol_F1"] = row["sc_Fixed_tol_F1"] - row["up_Fixed_tol_F1"]
        row["delta_sc_vs_sr_Fixed_tol_F1"] = row["sc_Fixed_tol_F1"] - row["sr_Fixed_tol_F1"]
        row["delta_sc_vs_up_ODS_F"] = row["sc_ODS_F"] - row["up_ODS_F"]
        row["delta_sc_vs_sr_ODS_F"] = row["sc_ODS_F"] - row["sr_ODS_F"]
        row["delta_sc_vs_up_cell_dep_f1"] = row["sc_cell_dep_f1"] - row["up_cell_dep_f1"]
        row["delta_sc_vs_sr_cell_dep_f1"] = row["sc_cell_dep_f1"] - row["sr_cell_dep_f1"]
        up_lo, up_hi = _ci_bounds(boot_up["Fixed_tol_F1"])
        sr_lo, sr_hi = _ci_bounds(boot_sr["Fixed_tol_F1"])
        row["boot_sc_vs_up_Fixed_ci_lo"] = up_lo
        row["boot_sc_vs_up_Fixed_ci_hi"] = up_hi
        row["boot_sc_vs_sr_Fixed_ci_lo"] = sr_lo
        row["boot_sc_vs_sr_Fixed_ci_hi"] = sr_hi
        hemit_summary_rows.append(row)

        cl_map = d["per_cluster_macro"]
        for cl in sorted(cl_map["upstream"]):
            r_cl = {"backbone": bk, "family": fam, "region": cl}
            for tag, a_name in (("up", "upstream"), ("sr", "S_recon:ens2"), ("sc", "S_count:ens2")):
                for mk in PIXEL_METRICS:
                    r_cl[f"{tag}_{mk}"] = float(cl_map[a_name][cl][mk])
            hemit_region_rows.append(r_cl)

        for mk_name in up["per_marker_pixel"]:
            r_mk = {"backbone": bk, "family": fam, "marker": mk_name}
            for tag, arm_obj in (("up", up), ("sr", sr), ("sc", sc)):
                for mk in PIXEL_METRICS:
                    r_mk[f"{tag}_{mk}"] = float(arm_obj["per_marker_pixel"][mk_name][mk])
            hemit_marker_rows.append(r_mk)

        hemit_cell_key = "per_marker" if "per_marker" in up["single_cell"] else "per_class"
        for cls_name in up["single_cell"].get(hemit_cell_key, {}):
            r_c = {"backbone": bk, "family": fam, "cell_class": cls_name}
            for tag, arm_obj in (("up", up), ("sr", sr), ("sc", sc)):
                c_obj = arm_obj["single_cell"][hemit_cell_key][cls_name]
                r_c[f"{tag}_dep_f1"] = _cell_metric(c_obj, "deployable_cell_f1", "deployable_f1")
                r_c[f"{tag}_rec_f1"] = _cell_metric(c_obj, "recalibrated_cell_f1", "recalibrated_f1")
                r_c[f"{tag}_auprc"] = _cell_metric(c_obj, "cell_auprc", "auprc")
                r_c[f"{tag}_auroc"] = _cell_metric(c_obj, "cell_roc_auc", "auroc")
                r_c[f"{tag}_tile_density_r"] = _cell_metric(c_obj, "tile_count_pearson_r_deploy", "tile_density_pearson_r")
            hemit_cell_rows.append(r_c)

        det_records.append(
            {
                "fold_id": f"hemit:{bk}",
                "scores": {
                    a_name: {
                        "Fixed_tol_F1": float(arms[a_name]["macro_pixel"]["Fixed_tol_F1"]),
                        "Cal_F1": float(arms[a_name]["macro_pixel"]["Cal_F1"]),
                        "ODS_F": float(arms[a_name]["macro_pixel"]["ODS_F"]),
                        "AP": float(arms[a_name]["macro_pixel"]["AP"]),
                        "SSIM": float(arms[a_name]["macro_pixel"]["SSIM"]),
                        "PSNR_dB": float(arms[a_name]["macro_pixel"]["PSNR_dB"]),
                        "cell_dep_f1": float(arms[a_name]["single_cell"]["macro_deployable_cell_f1"]),
                        "cell_auprc": float(arms[a_name]["single_cell"]["macro_cell_auprc"]),
                        "seconds": float(d.get("wall_seconds", 0.0)),
                    }
                    for a_name in ("upstream", "S_recon:ens2", "S_count:ens2")
                },
            }
        )

    # 2. Orion-CRC tables
    orion_keys = [k for k in ORION_ORDER if k in orion_docs] + [
        k for k in sorted(orion_docs) if k not in ORION_ORDER
    ]
    orion_summary_rows = []
    orion_patient_rows = []
    orion_marker_rows = []
    orion_cell_rows = []

    clin_df = pd.read_csv(clinical_csv)
    if "PatientID" not in clin_df.columns and "slideName" in clin_df.columns:
        clin_df = clin_df.rename(columns={"slideName": "PatientID"})
    clin_df = clin_df.sort_values("PatientID").reset_index(drop=True)
    surv_pat_df = clin_df[["PatientID", "PFSDays", "Recurrence", "Stage", "MMRIHC", "ImmunoScore"]].copy()

    for bk in orion_keys:
        d = orion_docs[bk]
        fam = str(d["family"])
        arms = d["fold0_16ch_arms"]
        up = arms["upstream"]
        sr = arms["S_recon:ens2"]
        sc = arms["S_count:ens2"]
        surv = d.get("survival_5fold_n40") or {}

        row = {
            "cohort": "Orion-CRC",
            "backbone": bk,
            "family": fam,
            "wall_seconds": float(d.get("wall_seconds", 0.0)),
        }
        for tag, arm_obj in (("up", up), ("sr", sr), ("sc", sc)):
            for mk in PIXEL_METRICS:
                row[f"{tag}_{mk}"] = float(arm_obj["macro_pixel_16ch"][mk])
            scell = arm_obj["single_cell_15class"]
            row[f"{tag}_cell_dep_f1"] = float(scell["macro_deployable_cell_f1"])
            row[f"{tag}_cell_rec_f1"] = float(scell["macro_recalibrated_cell_f1"])
            row[f"{tag}_cell_auprc"] = float(scell["macro_cell_auprc"])
            row[f"{tag}_cell_gap"] = float(scell["cell_mismatch_gap"])
            inf = arm_obj.get("spatial_panck_cd8_infiltration", {})
            row[f"{tag}_w25_r"] = float(inf.get("within_25um_pearson_r", 0.0))
            row[f"{tag}_w25_mae"] = float(inf.get("within_25um_mae", 0.0))

        row["delta_sc_vs_up_Fixed_tol_F1"] = row["sc_Fixed_tol_F1"] - row["up_Fixed_tol_F1"]
        row["delta_sc_vs_sr_Fixed_tol_F1"] = row["sc_Fixed_tol_F1"] - row["sr_Fixed_tol_F1"]
        row["delta_sc_vs_up_ODS_F"] = row["sc_ODS_F"] - row["up_ODS_F"]
        row["delta_sc_vs_sr_ODS_F"] = row["sc_ODS_F"] - row["sr_ODS_F"]
        row["delta_sc_vs_up_cell_dep_f1"] = row["sc_cell_dep_f1"] - row["up_cell_dep_f1"]
        row["delta_sc_vs_sr_cell_dep_f1"] = row["sc_cell_dep_f1"] - row["sr_cell_dep_f1"]

        if "fold0_paired_patient_bootstrap" in d:
            ob_up = d["fold0_paired_patient_bootstrap"]["scount_ens2_vs_upstream"]["Fixed_tol_F1"]
            ob_sr = d["fold0_paired_patient_bootstrap"]["scount_ens2_vs_srecon_ens2"]["Fixed_tol_F1"]
            row["boot_sc_vs_up_Fixed_ci_lo"], row["boot_sc_vs_up_Fixed_ci_hi"] = _ci_bounds(ob_up)
            row["boot_sc_vs_sr_Fixed_ci_lo"], row["boot_sc_vs_sr_Fixed_ci_hi"] = _ci_bounds(ob_sr)

        if surv:
            for tag, s_key in (("up", "upstream"), ("sr", "srecon_ens2"), ("sc", "scount_ens2")):
                sf = _extract_surv_fields(surv[s_key])
                row[f"{tag}_pfs_c_index"] = sf["c_index"]
                row[f"{tag}_pfs_c_ci_lo"] = sf["c_lo"]
                row[f"{tag}_pfs_c_ci_hi"] = sf["c_hi"]
                row[f"{tag}_km_hr"] = sf["km_hr"]
                row[f"{tag}_km_logrank_p"] = sf["km_p"]
                row[f"{tag}_cont_hr"] = sf["cont_hr"]
                row[f"{tag}_cont_p"] = sf["cont_p"]
                row[f"{tag}_mv_hr"] = sf["mv_hr"]
                row[f"{tag}_mv_p"] = sf["mv_p"]
            row["delta_sc_vs_up_c_index"] = row["sc_pfs_c_index"] - row["up_pfs_c_index"]
            row["delta_sc_vs_sr_c_index"] = row["sc_pfs_c_index"] - row["sr_pfs_c_index"]

            oof_c = surv["oof_immunoscore_continuous"]
            oof_d = surv["oof_immunoscore_discrete"]
            surv_pat_df[f"{bk}_up_cont"] = oof_c["upstream"]
            surv_pat_df[f"{bk}_sr_cont"] = oof_c["S_recon:ens2"]
            surv_pat_df[f"{bk}_sc_cont"] = oof_c["S_count:ens2"]
            surv_pat_df[f"{bk}_up_disc"] = oof_d["upstream"]
            surv_pat_df[f"{bk}_sr_disc"] = oof_d["S_recon:ens2"]
            surv_pat_df[f"{bk}_sc_disc"] = oof_d["S_count:ens2"]
        orion_summary_rows.append(row)

        pmap = d["fold0_per_patient_macro"]
        for pat in sorted(pmap["upstream"]):
            r_p = {"backbone": bk, "family": fam, "patient_id": pat}
            for tag, a_name in (("up", "upstream"), ("sr", "S_recon:ens2"), ("sc", "S_count:ens2")):
                for mk in PIXEL_METRICS:
                    r_p[f"{tag}_{mk}"] = float(pmap[a_name][pat][mk])
            orion_patient_rows.append(r_p)

        for mk_name, cut_val in d["fold0_physical_cutoffs_u8"].items():
            r_mk = {"backbone": bk, "family": fam, "marker": mk_name, "cutoff_u8": int(cut_val)}
            for tag, arm_obj in (("up", up), ("sr", sr), ("sc", sc)):
                for mk in PIXEL_METRICS:
                    r_mk[f"{tag}_{mk}"] = float(arm_obj["per_marker_pixel"][mk_name][mk])
            orion_marker_rows.append(r_mk)

        for cls_name in up["single_cell_15class"]["per_class"]:
            r_c = {"backbone": bk, "family": fam, "cell_class": cls_name}
            for tag, arm_obj in (("up", up), ("sr", sr), ("sc", sc)):
                c_obj = arm_obj["single_cell_15class"]["per_class"][cls_name]
                r_c[f"{tag}_dep_f1"] = _cell_metric(c_obj, "deployable_cell_f1", "deployable_f1")
                r_c[f"{tag}_rec_f1"] = _cell_metric(c_obj, "recalibrated_cell_f1", "recalibrated_f1")
                r_c[f"{tag}_auprc"] = _cell_metric(c_obj, "cell_auprc", "auprc")
                r_c[f"{tag}_auroc"] = _cell_metric(c_obj, "cell_roc_auc", "auroc")
                r_c[f"{tag}_tile_density_r"] = _cell_metric(c_obj, "tile_count_pearson_r_deploy", "tile_density_pearson_r")
            orion_cell_rows.append(r_c)

        det_scores = {}
        for a_name, s_key in (("upstream", "upstream"), ("S_recon:ens2", "srecon_ens2"), ("S_count:ens2", "scount_ens2")):
            sc_d = {
                "Fixed_tol_F1": float(arms[a_name]["macro_pixel_16ch"]["Fixed_tol_F1"]),
                "Cal_F1": float(arms[a_name]["macro_pixel_16ch"]["Cal_F1"]),
                "ODS_F": float(arms[a_name]["macro_pixel_16ch"]["ODS_F"]),
                "AP": float(arms[a_name]["macro_pixel_16ch"]["AP"]),
                "SSIM": float(arms[a_name]["macro_pixel_16ch"]["SSIM"]),
                "PSNR_dB": float(arms[a_name]["macro_pixel_16ch"]["PSNR_dB"]),
                "cell_dep_f1": float(arms[a_name]["single_cell_15class"]["macro_deployable_cell_f1"]),
                "cell_auprc": float(arms[a_name]["single_cell_15class"]["macro_cell_auprc"]),
                "seconds": float(d.get("wall_seconds", 0.0)),
            }
            if surv and s_key in surv:
                sc_d["pfs_c_index"] = float(surv[s_key]["c_index"])
            det_scores[a_name] = sc_d
        det_records.append({"fold_id": f"orion:{bk}", "scores": det_scores})

    # 3. HER2match tables
    her2_keys = [k for k in HER2_ORDER if k in her2_docs] + [
        k for k in sorted(her2_docs) if k not in HER2_ORDER
    ]
    her2_summary_rows = []
    her2_wsi_rows = []

    for bk in her2_keys:
        d = her2_docs[bk]
        fam = str(d["family"])
        arms = d["arms"]
        up = arms["upstream"]
        sr = arms["S_recon:ens2"]
        sc = arms["S_count:ens2"]

        row = {
            "cohort": "HER2match",
            "backbone": bk,
            "family": fam,
            "cutoff_u8": int(d["cutoff_u8"]),
            "wall_seconds": float(d.get("wall_seconds", 0.0)),
        }
        for tag, arm_obj in (("up", up), ("sr", sr), ("sc", sc)):
            for mk in PIXEL_METRICS:
                row[f"{tag}_{mk}"] = float(arm_obj["dab_pixel"][mk])
            burd = arm_obj["dab_burden_concordance"]
            row[f"{tag}_tile_burden_r"] = float(burd["tile_burden_pearson_r"])
            row[f"{tag}_wsi_burden_r"] = float(burd["wsi_burden_pearson_r"])
            row[f"{tag}_wsi_burden_mae"] = float(burd["wsi_burden_mae"])
            if "clinical_her2_score_spearman_pred" in burd:
                row[f"{tag}_clinical_spearman"] = float(burd["clinical_her2_score_spearman_pred"])

        row["delta_sc_vs_up_Fixed_tol_F1"] = row["sc_Fixed_tol_F1"] - row["up_Fixed_tol_F1"]
        row["delta_sc_vs_sr_Fixed_tol_F1"] = row["sc_Fixed_tol_F1"] - row["sr_Fixed_tol_F1"]
        row["delta_sc_vs_up_ODS_F"] = row["sc_ODS_F"] - row["up_ODS_F"]
        row["delta_sc_vs_sr_ODS_F"] = row["sc_ODS_F"] - row["sr_ODS_F"]
        her2_summary_rows.append(row)

        for wsi_id in sorted(up["per_wsi_pixel"]):
            r_w = {
                "backbone": bk,
                "family": fam,
                "wsi_id": wsi_id,
                "gt_dab_burden": float(up["dab_burden_concordance"]["per_wsi_gt_burden"][wsi_id]),
            }
            for tag, arm_obj in (("up", up), ("sr", sr), ("sc", sc)):
                for mk in PIXEL_METRICS:
                    r_w[f"{tag}_{mk}"] = float(arm_obj["per_wsi_pixel"][wsi_id][mk])
                r_w[f"{tag}_pred_dab_burden"] = float(arm_obj["dab_burden_concordance"]["per_wsi_pred_burden"][wsi_id])
            her2_wsi_rows.append(r_w)

        det_records.append(
            {
                "fold_id": f"her2:{bk}",
                "scores": {
                    a_name: {
                        "Fixed_tol_F1": float(arms[a_name]["dab_pixel"]["Fixed_tol_F1"]),
                        "Cal_F1": float(arms[a_name]["dab_pixel"]["Cal_F1"]),
                        "ODS_F": float(arms[a_name]["dab_pixel"]["ODS_F"]),
                        "AP": float(arms[a_name]["dab_pixel"]["AP"]),
                        "SSIM": float(arms[a_name]["dab_pixel"]["SSIM"]),
                        "PSNR_dB": float(arms[a_name]["dab_pixel"]["PSNR_dB"]),
                        "cell_dep_f1": float(arms[a_name]["dab_burden_concordance"]["tile_burden_pearson_r"]),
                        "cell_auprc": float(arms[a_name]["dab_burden_concordance"]["wsi_burden_pearson_r"]),
                        "seconds": float(d.get("wall_seconds", 0.0)),
                    }
                    for a_name in ("upstream", "S_recon:ens2", "S_count:ens2")
                },
            }
        )

    # 4. Unified 20-backbone cross-cohort table
    all20_rows = []
    for src_rows in (hemit_summary_rows, orion_summary_rows, her2_summary_rows):
        for r in src_rows:
            all20_rows.append(
                {
                    "cohort": r["cohort"],
                    "backbone": r["backbone"],
                    "family": r["family"],
                    "up_Fixed_tol_F1": r["up_Fixed_tol_F1"],
                    "sr_Fixed_tol_F1": r["sr_Fixed_tol_F1"],
                    "sc_Fixed_tol_F1": r["sc_Fixed_tol_F1"],
                    "up_Cal_F1": r["up_Cal_F1"],
                    "sr_Cal_F1": r["sr_Cal_F1"],
                    "sc_Cal_F1": r["sc_Cal_F1"],
                    "up_ODS_F": r["up_ODS_F"],
                    "sr_ODS_F": r["sr_ODS_F"],
                    "sc_ODS_F": r["sc_ODS_F"],
                    "up_AP": r["up_AP"],
                    "sr_AP": r["sr_AP"],
                    "sc_AP": r["sc_AP"],
                    "up_SSIM": r["up_SSIM"],
                    "sr_SSIM": r["sr_SSIM"],
                    "sc_SSIM": r["sc_SSIM"],
                    "up_PSNR_dB": r["up_PSNR_dB"],
                    "sr_PSNR_dB": r["sr_PSNR_dB"],
                    "sc_PSNR_dB": r["sc_PSNR_dB"],
                    "up_mismatch_gap": r["up_mismatch_gap_ods_minus_fixed"],
                    "sr_mismatch_gap": r["sr_mismatch_gap_ods_minus_fixed"],
                    "sc_mismatch_gap": r["sc_mismatch_gap_ods_minus_fixed"],
                    "delta_sc_vs_up_Fixed_tol_F1": r["delta_sc_vs_up_Fixed_tol_F1"],
                    "delta_sc_vs_sr_Fixed_tol_F1": r["delta_sc_vs_sr_Fixed_tol_F1"],
                    "delta_sc_vs_up_ODS_F": r["delta_sc_vs_up_ODS_F"],
                    "delta_sc_vs_sr_ODS_F": r["delta_sc_vs_sr_ODS_F"],
                }
            )

    def _write_csv(name: str, rows: list[dict]) -> Path:
        p = tables_dir / name
        if rows:
            pd.DataFrame(rows).to_csv(p, index=False)
        else:
            p.write_text("")
        return p

    _write_csv("hemit_per_backbone_summary.csv", hemit_summary_rows)
    _write_csv("hemit_per_region_macro.csv", hemit_region_rows)
    _write_csv("hemit_per_marker_detail.csv", hemit_marker_rows)
    _write_csv("hemit_per_cell_class.csv", hemit_cell_rows)
    _write_csv("orion_per_backbone_summary.csv", orion_summary_rows)
    _write_csv("orion_per_patient_fold0.csv", orion_patient_rows)
    _write_csv("orion_per_marker_16ch.csv", orion_marker_rows)
    _write_csv("orion_per_cell_15class.csv", orion_cell_rows)
    surv_pat_df.to_csv(tables_dir / "orion_n40_patient_survival.csv", index=False)
    _write_csv("her2_per_backbone_summary.csv", her2_summary_rows)
    _write_csv("her2_per_wsi_test.csv", her2_wsi_rows)
    _write_csv("all20_universal_backbones.csv", all20_rows)

    # Normalize determinism records so every fold_id shares identical score keys
    common_det_keys = (
        "Fixed_tol_F1",
        "Cal_F1",
        "ODS_F",
        "AP",
        "SSIM",
        "PSNR_dB",
        "cell_dep_f1",
        "cell_auprc",
        "seconds",
    )
    norm_det_records = [
        {
            "fold_id": r["fold_id"],
            "scores": {
                m: {k: float(r["scores"][m][k]) for k in common_det_keys}
                for m in ("upstream", "S_recon:ens2", "S_count:ens2")
            },
        }
        for r in det_records
    ]
    det_payload = {
        "methods": ["upstream", "S_recon:ens2", "S_count:ens2"],
        "records": norm_det_records,
    }
    (out_dir / "determinism_records.json").write_text(json.dumps(det_payload, indent=2) + "\n")

    # Cross-backbone aggregate statistics
    all_df = pd.DataFrame(all20_rows)
    cross_stats = {}
    if len(all_df) > 0:
        d_fix_up = all_df["delta_sc_vs_up_Fixed_tol_F1"].to_numpy(dtype=np.float64)
        d_fix_sr = all_df["delta_sc_vs_sr_Fixed_tol_F1"].to_numpy(dtype=np.float64)
        d_ods_up = all_df["delta_sc_vs_up_ODS_F"].to_numpy(dtype=np.float64)
        d_ods_sr = all_df["delta_sc_vs_sr_ODS_F"].to_numpy(dtype=np.float64)
        w_p_up = float(stats.wilcoxon(d_fix_up).pvalue) if len(d_fix_up) >= 6 and np.any(d_fix_up != 0) else 1.0
        w_p_sr = float(stats.wilcoxon(d_fix_sr).pvalue) if len(d_fix_sr) >= 6 and np.any(d_fix_sr != 0) else 1.0
        cross_stats = {
            "n_backbones_total": int(len(all_df)),
            "n_hemit_backbones": int(len(hemit_summary_rows)),
            "n_orion_backbones": int(len(orion_summary_rows)),
            "n_her2_backbones": int(len(her2_summary_rows)),
            "wins_sc_vs_up_Fixed_tol_F1": int(np.sum(d_fix_up > 0)),
            "wins_sc_vs_sr_Fixed_tol_F1": int(np.sum(d_fix_sr > 0)),
            "wins_sc_vs_up_ODS_F": int(np.sum(d_ods_up > 0)),
            "wins_sc_vs_sr_ODS_F": int(np.sum(d_ods_sr > 0)),
            "mean_up_Fixed_tol_F1": float(np.mean(all_df["up_Fixed_tol_F1"])),
            "mean_sr_Fixed_tol_F1": float(np.mean(all_df["sr_Fixed_tol_F1"])),
            "mean_sc_Fixed_tol_F1": float(np.mean(all_df["sc_Fixed_tol_F1"])),
            "mean_delta_sc_vs_up_Fixed_tol_F1": float(np.mean(d_fix_up)),
            "mean_delta_sc_vs_sr_Fixed_tol_F1": float(np.mean(d_fix_sr)),
            "mean_up_ODS_F": float(np.mean(all_df["up_ODS_F"])),
            "mean_sr_ODS_F": float(np.mean(all_df["sr_ODS_F"])),
            "mean_sc_ODS_F": float(np.mean(all_df["sc_ODS_F"])),
            "mean_delta_sc_vs_up_ODS_F": float(np.mean(d_ods_up)),
            "mean_delta_sc_vs_sr_ODS_F": float(np.mean(d_ods_sr)),
            "wilcoxon_p_sc_vs_up_Fixed_tol_F1": w_p_up,
            "wilcoxon_p_sc_vs_sr_Fixed_tol_F1": w_p_sr,
        }

    analysis_doc = {
        "schema": "journal-confirmatory-analysis-v1",
        "status": "completed",
        "cross_backbone_20": cross_stats,
        "oracle_survival_n40": oracle_surv,
        "hemit_backbones": hemit_keys,
        "orion_backbones": orion_keys,
        "her2_backbones": her2_keys,
    }
    (out_dir / "analysis.json").write_text(json.dumps(analysis_doc, indent=2) + "\n")
    (out_dir / "accepted.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "n_backbones_total": len(all20_rows),
                "n_hemit": len(hemit_summary_rows),
                "n_orion": len(orion_summary_rows),
                "n_her2": len(her2_summary_rows),
            },
            indent=2,
        )
        + "\n"
    )
    return analysis_doc


def emit_from_table(
    *,
    csv_path: Path,
    op: str,
    column: str = "",
    col_b: str = "",
    where: Sequence[str] = (),
    backbone: str = "",
    arm_tag: str = "sc",
    digits: int = 6,
) -> str:
    """Recompute a claim value directly from a per-unit CSV table."""
    df = pd.read_csv(csv_path)
    for cond in where:
        k, _, v = cond.partition("=")
        df = df[df[k.strip()].astype(str) == v.strip()]
    if len(df) == 0:
        raise ValueError(f"Filter {where!r} matched 0 rows in {csv_path}")

    if op == "mean":
        val = float(df[column].astype(float).mean())
    elif op == "median":
        val = float(df[column].astype(float).median())
    elif op == "sum":
        val = float(df[column].astype(float).sum())
    elif op == "count":
        val = int(len(df))
        return str(val)
    elif op == "wins_gt":
        diff = df[column].astype(float) - (df[col_b].astype(float) if col_b else 0.0)
        return str(int((diff > 0.0).sum()))
    elif op == "diff_mean":
        val = float((df[column].astype(float) - df[col_b].astype(float)).mean())
    elif op == "single":
        if len(df) != 1:
            raise ValueError(f"Expected 1 row for op=single, got {len(df)}")
        val = float(df[column].iloc[0])
    elif op == "pearson_r":
        val = float(stats.pearsonr(df[column].astype(float), df[col_b].astype(float))[0])
    elif op == "wilcoxon_p":
        d = df[column].astype(float).to_numpy() - (df[col_b].astype(float).to_numpy() if col_b else 0.0)
        val = float(stats.wilcoxon(d).pvalue)
    elif op in ("survival_c_index", "survival_km_hr", "survival_km_p", "survival_mv_hr", "survival_mv_p", "survival_oracle_c_index"):
        pfs = df["PFSDays"].to_numpy(dtype=np.float64)
        rec = df["Recurrence"].to_numpy(dtype=np.int32)
        stage_num = df["Stage"].map({"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}).to_numpy(dtype=np.float64)
        mmr_def = (df["MMRIHC"].fillna("Intact").str.lower() != "intact").astype(np.float64).to_numpy()
        if op == "survival_oracle_c_index":
            score = df["ImmunoScore"].to_numpy(dtype=np.float64)
        else:
            score_col = f"{backbone}_{arm_tag}_cont"
            score = df[score_col].to_numpy(dtype=np.float64)
        res = ccs.evaluate_pfs_survival(
            pfs_days=pfs,
            recurrence_events=rec,
            protective_score=score,
            dichotomy_cutoff=2.0,
            stage_numeric=stage_num,
            mmr_deficient=mmr_def,
            n_bootstrap=2000,
        )
        sf = _extract_surv_fields(res)
        if op in ("survival_c_index", "survival_oracle_c_index"):
            val = sf["c_index"]
        elif op == "survival_km_hr":
            val = sf["km_hr"]
        elif op == "survival_km_p":
            val = sf["km_p"]
        elif op == "survival_mv_hr":
            val = sf["mv_hr"]
        else:
            val = sf["mv_p"]
    else:
        raise ValueError(f"Unsupported op: {op!r}")

    return f"{val:.{digits}f}"


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b_p = sub.add_parser("build")
    b_p.add_argument("--hemit-dir", required=True)
    b_p.add_argument("--orion-dir", required=True)
    b_p.add_argument("--her2-dir", required=True)
    b_p.add_argument("--clinical-csv", required=True)
    b_p.add_argument("--out-dir", required=True)

    e_p = sub.add_parser("emit")
    e_p.add_argument("--csv", required=True)
    e_p.add_argument("--op", required=True)
    e_p.add_argument("--col", default="")
    e_p.add_argument("--col-b", default="")
    e_p.add_argument("--where", action="append", default=[])
    e_p.add_argument("--backbone", default="")
    e_p.add_argument("--arm-tag", default="sc")
    e_p.add_argument("--digits", type=int, default=6)

    args = ap.parse_args(argv)
    if args.cmd == "build":
        res = build_confirmatory_tables(
            hemit_dir=Path(args.hemit_dir),
            orion_dir=Path(args.orion_dir),
            her2_dir=Path(args.her2_dir),
            clinical_csv=Path(args.clinical_csv),
            out_dir=Path(args.out_dir),
        )
        print(json.dumps(res["cross_backbone_20"], indent=2))
    elif args.cmd == "emit":
        out = emit_from_table(
            csv_path=Path(args.csv),
            op=args.op,
            column=args.col,
            col_b=args.col_b,
            where=args.where,
            backbone=args.backbone,
            arm_tag=args.arm_tag,
            digits=args.digits,
        )
        print(f"VALUE: {out}")


if __name__ == "__main__":
    main()
