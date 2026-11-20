#!/usr/bin/env python3
"""EXP-1: Full 5-fold patient-disjoint Orion-CRC 16-channel & 15-marker nuclear CV (N = 40 patients).

Runs Folds 1, 2, 3, 4 (matching the exact Fold 0 S6 protocol: training-only physical cutoffs
fitted strictly on the 26 inner-training patients, stride-2 spatial fit split, 2-seed ensembles
for S_recon and S_count, and held-out outer-test evaluation across 8 unseen patients per fold).
Combined with immutable S6 Fold 0, this covers all N = 40 primary colorectal cancer patients
and all n = 2,804 tiles across 16 validated channels and 15 overlapping nuclear classes.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import time
from typing import Mapping

import numpy as np
import torch

from . import journal_cohort_eval as jce
from . import run_orion_multibackbone_survival as roms
from . import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: Mapping) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _compute_cooccurrence_corr(pred_means16: np.ndarray, tau_float: dict[str, float]) -> np.ndarray:
    """Compute 15x15 binary marker co-occurrence Pearson correlation matrix across nuclei."""
    cols = []
    for ch16 in range(1, 16):
        m_nm = roms.ORION_MARKERS[ch16]
        cols.append((pred_means16[:, ch16] > float(tau_float[m_nm])).astype(np.float64))
    mat = np.stack(cols, axis=1)  # (K, 15)
    c = np.zeros((15, 15), dtype=np.float64)
    for i in range(15):
        si = float(np.std(mat[:, i]))
        for j in range(15):
            sj = float(np.std(mat[:, j]))
            if i == j:
                c[i, j] = 1.0
            elif si > 1e-12 and sj > 1e-12:
                c[i, j] = float(np.corrcoef(mat[:, i], mat[:, j])[0, 1])
    return c


def run_fold(
    *,
    b_key: str,
    fold_idx: int,
    bank: dict,
    up_all: np.ndarray,
    fold_spec: dict,
    out_dir: Path,
    device: torch.device,
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
    lr: float,
    overwrite: bool,
) -> None:
    out_json = out_dir / f"orion5fold_{b_key}_fold{fold_idx}.json"
    if out_json.exists() and not overwrite:
        print(json.dumps(dict(event="skip_existing", backbone=b_key, fold=fold_idx)), flush=True)
        return

    t0 = time.monotonic()
    train_pats = set(roms._fold_patients(fold_spec, "inner_train"))
    eval_pats_set = set(roms._fold_patients(fold_spec, "outer_test"))
    tr_idx_all = np.nonzero(np.isin(bank["patients"], list(train_pats)))[0]
    tr_idx = tr_idx_all[::2]  # fit_stride=2 matching S6 Fold 0 exactly
    ev_idx = np.nonzero(np.isin(bank["patients"], list(eval_pats_set)))[0]

    # Calibrate physical cutoffs strictly on this fold's 26 inner-training patients
    cutoffs_u8, tau_phys_float = roms.calibrate_orion_physical_cutoffs_from_train(
        bank["targ_nchw"][tr_idx_all], tr_idx_all, bank["nuclei_records"]
    )

    up_tr = up_all[tr_idx]
    up_ev = up_all[ev_idx]
    arm_tr = {"upstream": up_tr, "S_recon": np.empty_like(up_tr), "S_count": np.empty_like(up_tr)}
    arm_ev = {"upstream": up_ev, "S_recon": np.empty_like(up_ev), "S_count": np.empty_like(up_ev)}

    for ch16, m_name in enumerate(roms.ORION_MARKERS):
        cutoff = int(cutoffs_u8[ch16])
        c12_tr = roms.build_common12_orion(bank["he_nhwc"][tr_idx], up_tr, ch16, device=device)
        d_tr_t = torch.from_numpy(up_tr[:, ch16])
        y_tr_t = torch.from_numpy(bank["targ_nchw"][tr_idx, ch16])
        d_ev_t = torch.from_numpy(up_ev[:, ch16])

        fitted: dict[tuple[str, int], object] = {}
        for arm in ("S_recon", "S_count"):
            tr_list = []
            for sd in seeds:
                model, _ = suh.fit_universal_marker(
                    arm=arm,
                    seed=sd,
                    common_train=c12_tr,
                    donor_u8_train=d_tr_t,
                    target_u8_train=y_tr_t,
                    cutoff=cutoff,
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    deploy_anchor_weight=1.25,
                )
                p_tr, _ = suh.predict_universal_marker(model, c12_tr, d_tr_t, device=device, batch_size=32)
                tr_list.append(p_tr.astype(np.float32))
                fitted[(arm, sd)] = model.cpu()
            arm_tr[arm][:, ch16] = np.round(np.mean(tr_list, axis=0)).astype(np.uint8)

        del c12_tr
        gc.collect()

        c12_ev = roms.build_common12_orion(bank["he_nhwc"][ev_idx], up_ev, ch16, device=device)
        for arm in ("S_recon", "S_count"):
            ev_list = []
            for sd in seeds:
                model = fitted[(arm, sd)].to(device)
                p_ev, _ = suh.predict_universal_marker(model, c12_ev, d_ev_t, device=device, batch_size=32)
                ev_list.append(p_ev.astype(np.float32))
                del model
            arm_ev[arm][:, ch16] = np.round(np.mean(ev_list, axis=0)).astype(np.uint8)
        del c12_ev, fitted
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (ch16 + 1) % 4 == 0 or ch16 == 15:
            print(
                json.dumps(dict(event="channel_done", backbone=b_key, fold=fold_idx, ch=ch16 + 1, marker=m_name, elapsed=round(time.monotonic() - t0, 1))),
                flush=True,
            )

    eval_pats = bank["patients"][ev_idx]
    uniq_pats = sorted(set(str(p) for p in eval_pats))
    arms_eval: dict[str, dict] = {}
    per_patient_macro: dict[str, dict] = {p: {} for p in uniq_pats}

    # Physical 15x15 co-occurrence reference
    ev_cells_ref = roms.gather_subset_cells(up_ev, ev_idx, bank["nuclei_records"])
    gt_cooc = np.zeros((15, 15), dtype=np.float64)
    lbl_gt = ev_cells_ref["labels15"].astype(np.float64)
    for i in range(15):
        si = float(np.std(lbl_gt[:, i]))
        for j in range(15):
            sj = float(np.std(lbl_gt[:, j]))
            if i == j:
                gt_cooc[i, j] = 1.0
            elif si > 1e-12 and sj > 1e-12:
                gt_cooc[i, j] = float(np.corrcoef(lbl_gt[:, i], lbl_gt[:, j])[0, 1])

    for arm in ("upstream", "S_recon", "S_count"):
        per_m = {}
        pat_f1_accum = {p: [] for p in uniq_pats}
        for ch16, m_name in enumerate(roms.ORION_MARKERS):
            ch_r = jce.evaluate_channel_predictions(
                train_pred_u8=arm_tr[arm][:, ch16],
                train_targ_u8=bank["targ_nchw"][tr_idx, ch16],
                eval_pred_u8=arm_ev[arm][:, ch16],
                eval_targ_u8=bank["targ_nchw"][ev_idx, ch16],
                eval_clusters=eval_pats,
                fixed_cutoff=int(cutoffs_u8[ch16]),
                device=device,
            )
            per_m[m_name] = ch_r["pooled"]
            for p_id in uniq_pats:
                if p_id in ch_r.get("per_cluster", {}):
                    pat_f1_accum[p_id].append(float(ch_r["per_cluster"][p_id]["Fixed_tol_F1"]))

        for p_id in uniq_pats:
            per_patient_macro[p_id][arm] = float(np.mean(pat_f1_accum[p_id])) if pat_f1_accum[p_id] else 0.0

        macro = {
            mk: float(np.mean([per_m[m][mk] for m in roms.ORION_MARKERS]))
            for mk in ("ODS_F", "AP", "Fixed_tol_F1", "Cal_F1", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed")
        }

        tr_cells = roms.gather_subset_cells(arm_tr[arm], tr_idx, bank["nuclei_records"])
        ev_cells = roms.gather_subset_cells(arm_ev[arm], ev_idx, bank["nuclei_records"])
        per_cls = {}
        dep_f1s, auprcs, aurocs = [], [], []
        for ch16 in range(1, 16):
            cls_idx = ch16 - 1
            m_nm = roms.ORION_MARKERS[ch16]
            res_c = roms.ccs.evaluate_single_cell_marker(
                train_gt_scores=tr_cells["gt_means16"][:, ch16],
                train_pred_scores=tr_cells["pred_means"][:, ch16],
                train_labels=tr_cells["labels15"][:, cls_idx],
                eval_pred_scores=ev_cells["pred_means"][:, ch16],
                eval_labels=ev_cells["labels15"][:, cls_idx],
                eval_tile_ids=ev_cells["tile_ids"],
                physical_cutoff=float(tau_phys_float[m_nm]),
            )
            per_cls[m_nm] = res_c
            dep_f1s.append(float(res_c["deployable_cell_f1"]))
            if np.isfinite(res_c.get("cell_auprc", np.nan)):
                auprcs.append(float(res_c["cell_auprc"]))
            if np.isfinite(res_c.get("cell_auroc", np.nan)):
                aurocs.append(float(res_c["cell_auroc"]))

        cooc_mat = _compute_cooccurrence_corr(ev_cells["pred_means"], tau_phys_float)
        cooc_mae = float(np.mean(np.abs(cooc_mat - gt_cooc)))
        cooc_frob = float(np.linalg.norm(cooc_mat - gt_cooc, ord="fro"))

        arms_eval[arm] = dict(
            macro_pixel_16ch=macro,
            per_marker_pixel=per_m,
            single_cell_15class=dict(
                macro_deployable_cell_f1=float(np.mean(dep_f1s)),
                macro_cell_auprc=float(np.mean(auprcs)) if auprcs else float("nan"),
                macro_cell_auroc=float(np.mean(aurocs)) if aurocs else float("nan"),
                per_class=per_cls,
                cooccurrence_mae=cooc_mae,
                cooccurrence_frobenius=cooc_frob,
                cooccurrence_matrix=cooc_mat.tolist(),
            ),
        )

    rec = dict(
        schema="orion-5fold-16ch-v1",
        cohort="Orion-CRC",
        backbone=b_key,
        fold=fold_idx,
        n_train_tiles=int(len(tr_idx)),
        n_eval_tiles=int(len(ev_idx)),
        eval_patients=uniq_pats,
        wall_seconds=time.monotonic() - t0,
        cutoffs_u8=[int(x) for x in cutoffs_u8],
        tau_phys_float={k: float(v) for k, v in tau_phys_float.items()},
        gt_cooccurrence_matrix=gt_cooc.tolist(),
        per_patient_macro_fixed_f1=per_patient_macro,
        arms=arms_eval,
    )
    _atomic_write_json(out_json, rec)
    print(
        json.dumps(
            dict(
                event="fold_done",
                backbone=b_key,
                fold=fold_idx,
                up=round(arms_eval["upstream"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                sr=round(arms_eval["S_recon"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                sc=round(arms_eval["S_count"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                up_cell=round(arms_eval["upstream"]["single_cell_15class"]["macro_deployable_cell_f1"], 4),
                sc_cell=round(arms_eval["S_count"]["single_cell_15class"]["macro_deployable_cell_f1"], 4),
                sec=round(rec["wall_seconds"], 1),
            )
        ),
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", required=True)
    ap.add_argument("--folds", default="1,2,3,4")
    ap.add_argument("--bank-cache-dir", required=True)
    ap.add_argument("--upstream-cache-dir", required=True)
    ap.add_argument("--cv-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="202610270,202610271")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_manifest = json.loads(Path(args.cv_manifest).read_text())
    with open(Path(args.bank_cache_dir).resolve() / "orion_bank_all40_2804_crop256.pkl", "rb") as fh:
        bank = pickle.load(fh)

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    for b_key in backbones:
        with np.load(Path(args.upstream_cache_dir).resolve() / f"{b_key}_all40_2804.npz") as npz:
            up_all = npz["pred_u8"]
        for fi in folds:
            run_fold(
                b_key=b_key,
                fold_idx=fi,
                bank=bank,
                up_all=up_all,
                fold_spec=cv_manifest["folds"][fi],
                out_dir=out_dir,
                device=device,
                seeds=seeds,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                overwrite=bool(args.overwrite),
            )


if __name__ == "__main__":
    main()
