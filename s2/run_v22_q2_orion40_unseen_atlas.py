#!/usr/bin/env python3
"""v2.2 Q2: Multi-Architecture Zero-Leakage 40-Patient (335,597-Nucleus) Orion-CRC
8-Lineage Single-Cell Phenotyping Atlas (Frozen, E_count_zm, L_count, S_count).

Evaluates 40-patient 8-lineage single-cell phenotyping across strictly unseen-patient
and published pretrained architectures:
  - ``--mode unseen``: For each strictly 0%-test-patient-overlap architecture
      (``unseen_unet16``, ``unseen_resnet34_unet16``, ``unseen_pix2pix16``),
      trains a fresh 16-channel virtual-staining generator FROM SCRATCH per outer fold
      strictly on the 26 inner_train + 6 inner_val patients (asserting 0 overlap with
      the 8 outer_test patients), freezes it, fits 2-seed post-hoc heads for
      ``E_count_zm``, ``L_count``, and ``S_count``, and scores all 8 biological lineages
      across all 40 patients (335,597 nuclei).
  - ``--mode cached``: Evaluates published pretrained architectures
      (``miphei_convnext``, ``miphei_hemit_arch``) across all 5 folds & 40 patients
      with ``Frozen``, ``E_count_zm``, ``L_count``, and ``S_count`` (or subset via
      ``--arms`` to augment existing v2.1 ConvNeXt/Swin caches cleanly).

Supports per-fold atomic NPZ caching under ``<out_dir>/fold_cache/`` so folds or
architectures can run in parallel across GPUs and merge cleanly via exact SciPy 1.16.0
``method="exact"`` patient-paired Wilcoxon tests (N = 40 independent patients).
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
from sklearn.metrics import f1_score

from s2 import clinical_cell_survival as ccs
from s2 import run_nv2_n1_orion_clinical_umap as rn1
from s2 import run_orion_multibackbone_survival as ror
from s2 import run_plan2_ablations as rp2
from s2 import run_push3_orion_unseen5fold as rp3
from s2 import screened_universal_head as suh

CLS_NAMES = ["Tumor", "Cytotoxic T", "Helper/Treg T", "B cell", "Macrophage", "Endothelia", "Stroma", "Other"]
ALL_ATLAS_ARMS = ("Frozen", "E_count_zm", "L_count", "S_count")


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _exact_wilcoxon_pair(va: np.ndarray, vb: np.ndarray) -> dict:
    """Exact two-sided Wilcoxon signed-rank test omitting exact-zero ties (SciPy 1.16 method='exact')."""
    diff = np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    nz = diff[diff != 0.0]
    if len(nz) == 0:
        p_val = 1.0
    else:
        try:
            p_val = float(stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided", method="exact").pvalue)
        except TypeError:
            p_val = float(stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided")[1])
    return dict(
        mean_a=round(float(np.mean(va)), 5),
        mean_b=round(float(np.mean(vb)), 5),
        mean_delta=round(float(np.mean(diff)), 5),
        median_delta=round(float(np.median(diff)), 5),
        wins_of_n=wins,
        losses_of_n=losses,
        ties_of_n=ties,
        n_nonzero=int(len(nz)),
        wilcoxon_exact_p=float(f"{p_val:.4e}"),
    )


def run_single_fold(
    *,
    mode: str,
    arch: str,
    fold_idx: int,
    ob: dict,
    fold: dict,
    upstream_all: np.ndarray | None,
    arms: tuple[str, ...],
    fit_stride: int,
    up_epochs: int,
    head_epochs: int,
    batch_size: int,
    fold_cache_path: Path,
    device: torch.device,
) -> dict:
    """Execute one outer fold (8 held-out patients) and save atomic compact nuclear labels/means NPZ."""
    t0 = time.time()
    pats_arr = np.asarray(ob["patients"])
    inner_tr_pats = set(fold["inner_train_slides"])
    inner_va_pats = set(fold.get("inner_val_slides", []))
    outer_te_pats = set(fold["outer_test_slides"])

    assert len((inner_tr_pats | inner_va_pats) & outer_te_pats) == 0, "Patient leakage detected!"
    assert len(inner_tr_pats & inner_va_pats) == 0, "Inner train/val leakage detected!"

    tr_idx_full = np.nonzero(np.isin(pats_arr, list(inner_tr_pats)))[0]
    va_idx_full = np.nonzero(np.isin(pats_arr, list(inner_va_pats)))[0]
    ev_idx = np.nonzero(np.isin(pats_arr, list(outer_te_pats)))[0]
    tr_idx = tr_idx_full[:: max(1, int(fit_stride))]

    # Calibrate physical cutoffs strictly on inner_train patients
    pix_cuts, cell_cuts_dict = ror.calibrate_orion_physical_cutoffs_from_train(
        ob["targ_nchw"][tr_idx_full], tr_idx_full, ob["nuclei_records"]
    )
    pix_cuts = [int(c) for c in pix_cuts]
    if isinstance(cell_cuts_dict, dict):
        cell_cuts16 = np.asarray([float(cell_cuts_dict[m]) for m in ror.ORION_MARKERS], dtype=np.float64)
    else:
        cell_cuts16 = np.zeros(16, dtype=np.float64)
        cell_cuts16[0] = float(pix_cuts[0])
        cell_cuts16[1:] = np.asarray(cell_cuts_dict, dtype=np.float64)

    gen_info = {}
    if mode == "unseen":
        gen, gen_info = rp3.train_fold_isolated_generator(
            arch=arch,
            fold_idx=fold_idx,
            he_tr=ob["he_nhwc"][tr_idx_full],
            targ_tr=ob["targ_nchw"][tr_idx_full],
            he_val=ob["he_nhwc"][va_idx_full],
            targ_val=ob["targ_nchw"][va_idx_full],
            device=device,
            epochs=up_epochs,
            batch_size=16 if arch != "unseen_resnet34_unet16" else 12,
        )
        up_tr = rp3.predict_16ch_u8(gen, ob["he_nhwc"][tr_idx], device=device, batch_size=32)
        up_ev = rp3.predict_16ch_u8(gen, ob["he_nhwc"][ev_idx], device=device, batch_size=32)
        del gen
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[V22-Q2 {arch} f{fold_idx}] Upstream zero-leakage trained: best_ep={gen_info['best_epoch']} "
            f"val_l1={gen_info['best_val_l1']} ({gen_info['train_sec']}s)",
            flush=True,
        )
    else:
        assert upstream_all is not None
        up_tr = upstream_all[tr_idx]
        up_ev = upstream_all[ev_idx]

    he_tr, he_ev = ob["he_nhwc"][tr_idx], ob["he_nhwc"][ev_idx]
    gt_tr = ob["targ_nchw"][tr_idx]

    arm_ev = {}
    if "Frozen" in arms:
        arm_ev["Frozen"] = up_ev
    trained_arms = [a for a in arms if a != "Frozen"]
    for a in trained_arms:
        arm_ev[a] = up_ev.copy()

    for ch in range(16):
        c12_tr = ror.build_common12_orion(he_tr, up_tr, ch, device=device)
        c12_ev = ror.build_common12_orion(he_ev, up_ev, ch, device=device)
        d_tr_t = torch.from_numpy(up_tr[:, ch])
        y_tr_t = torch.from_numpy(gt_tr[:, ch])
        d_ev_t = torch.from_numpy(up_ev[:, ch])

        for arm_nm in trained_arms:
            preds_s = []
            for sd in (202610270, 202610271):
                if arm_nm == "S_count":
                    mdl, _ = suh.fit_universal_marker(
                        arm="S_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                        target_u8_train=y_tr_t, cutoff=pix_cuts[ch], device=device,
                        epochs=head_epochs, batch_size=batch_size, lr=1.5e-3, deploy_anchor_weight=1.25,
                    )
                    pe, _ = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device=device, batch_size=32)
                else:
                    mdl, _ = rp2.fit_ablation_marker(
                        mode=arm_nm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                        target_u8_train=y_tr_t, cutoff=pix_cuts[ch], device=device,
                        epochs=head_epochs, batch_size=batch_size, lr=1.5e-3, deploy_anchor_weight=1.25,
                    )
                    pe, _ = rp2.predict_ablation_marker(mdl, arm_nm, c12_ev, d_ev_t, device=device, batch_size=32)
                preds_s.append(pe.astype(np.float32))
                del mdl
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            arm_ev[arm_nm][:, ch] = np.rint(np.clip(0.5 * (preds_s[0] + preds_s[1]), 0.0, 255.0)).astype(np.uint8)
        del c12_tr, c12_ev
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Extract out-of-fold 16-ch nuclear intensities and classify 8 biological lineages
    f_phys, f_pat = [], []
    f_arm_m = {a: [] for a in arms}
    f_gt_l15 = []
    for k_i, idx in enumerate(ev_idx.tolist()):
        rec = ob["nuclei_records"][idx]
        if rec is None:
            continue
        pm_gt = rec["gt_means16"]
        f_phys.append(pm_gt)
        f_gt_l15.append(rec["labels15"])
        f_pat.extend([str(ob["patients"][idx])] * len(pm_gt))
        for a in arms:
            f_arm_m[a].append(ccs.extract_means_with_index(arm_ev[a][k_i], rec["idx_spec"]))

    Xp = np.concatenate(f_phys, axis=0)
    gl15 = np.concatenate(f_gt_l15, axis=0)
    y_phys = rn1._assign_bio_phenotype_8class(Xp, cell_cuts16).astype(np.int8)

    payload = dict(
        y_phys=y_phys,
        patients=np.asarray(f_pat),
        cell_cuts16=cell_cuts16,
        gt_labels15=gl15.astype(np.int8),
        wall_seconds=np.array([round(time.time() - t0, 1)], dtype=np.float64),
    )
    for a in arms:
        Xa = np.concatenate(f_arm_m[a], axis=0)
        ya = rn1._assign_bio_phenotype_8class(Xa, cell_cuts16).astype(np.int8)
        # Also compute 15-class binary predictions using fold-calibrated cell_cuts16[1:]
        pred_l15 = (Xa[:, 1:] > cell_cuts16[None, 1:]).astype(np.int8)
        payload[f"y_{a}"] = ya
        payload[f"l15_{a}"] = pred_l15

    fold_cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = fold_cache_path.with_name(f".tmp_{os.getpid()}_{fold_cache_path.name}")
    np.savez_compressed(tmp_npz, **payload)
    os.replace(tmp_npz, fold_cache_path)
    print(
        f"[V22-Q2 {arch} f{fold_idx}] Saved {len(y_phys)} nuclei across 8 held-out patients ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return dict(n_nuclei=int(len(y_phys)), wall_seconds=round(time.time() - t0, 1), gen_info=gen_info)


def aggregate_arch_5fold(arch: str, cache_dir: Path, arms: tuple[str, ...]) -> dict | None:
    """Merge 5 fold NPZ caches into complete N=40 patient 335,597-nucleus report with exact Wilcoxon tests."""
    fold_files = [cache_dir / f"{arch}_fold{fi}.npz" for fi in range(5)]
    if not all(f.exists() for f in fold_files):
        missing = [fi for fi, f in enumerate(fold_files) if not f.exists()]
        print(f"[V22-Q2 AGG] {arch} missing folds {missing}, skipping aggregate", flush=True)
        return None

    y_phys_list, pat_list, gl15_list = [], [], []
    arm_y = {a: [] for a in arms}
    arm_l15 = {a: [] for a in arms}
    tot_wall = 0.0
    for f in fold_files:
        z = np.load(f, allow_pickle=False)
        y_phys_list.append(z["y_phys"])
        pat_list.append(z["patients"])
        gl15_list.append(z["gt_labels15"])
        tot_wall += float(z["wall_seconds"][0]) if "wall_seconds" in z else 0.0
        for a in arms:
            arm_y[a].append(z[f"y_{a}"])
            arm_l15[a].append(z[f"l15_{a}"])

    y_phys = np.concatenate(y_phys_list, axis=0)
    pat_vec = np.concatenate(pat_list, axis=0)
    gl15 = np.concatenate(gl15_list, axis=0)
    uniq_pats = sorted(set(pat_vec.tolist()))
    assert len(uniq_pats) == 40, f"Expected 40 patients, got {len(uniq_pats)}"

    y_arms = {a: np.concatenate(arm_y[a], axis=0) for a in arms}
    l15_arms = {a: np.concatenate(arm_l15[a], axis=0) for a in arms}

    pooled = {}
    for a in arms:
        yh = y_arms[a]
        f1s = [float(f1_score(y_phys == c, yh == c, zero_division=0)) for c in range(8)]
        pp = np.bincount(y_phys, minlength=8) / len(y_phys)
        ph = np.bincount(yh, minlength=8) / len(yh)
        r_v = float(stats.pearsonr(pp, ph)[0]) if np.std(ph) > 1e-12 else 0.0
        # 15-class nuclear macro F1 across all 335,597 nuclei
        pl = l15_arms[a]
        c15 = []
        for j in range(15):
            tp = int(((pl[:, j] == 1) & (gl15[:, j] > 0)).sum())
            fp = int(((pl[:, j] == 1) & (gl15[:, j] == 0)).sum())
            fn = int(((pl[:, j] == 0) & (gl15[:, j] > 0)).sum())
            c15.append((2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0)
        pooled[a] = dict(
            macro_f1=round(float(np.mean(f1s)), 5),
            per_lineage_f1={CLS_NAMES[c]: round(f1s[c], 5) for c in range(8)},
            prop_pearson_r=round(r_v, 5),
            macro_15class_nuclear_f1=round(float(np.mean(c15)), 5),
            cd8a_nuclear_f1=round(float(c15[5]), 5),
        )

    per_pat = {}
    for p in uniq_pats:
        m = pat_vec == p
        yp = y_phys[m]
        pp = np.bincount(yp, minlength=8) / max(1, len(yp))
        p_rec = dict(n_nuclei=int(m.sum()))
        for a in arms:
            yh = y_arms[a][m]
            f1s = [float(f1_score(yp == c, yh == c, zero_division=0)) for c in range(8)]
            ph = np.bincount(yh, minlength=8) / max(1, len(yh))
            r_v = float(stats.pearsonr(pp, ph)[0]) if np.std(ph) > 1e-12 and np.std(pp) > 1e-12 else 0.0
            p_rec[a] = dict(
                macro_f1=round(float(np.mean(f1s)), 5),
                tumor_f1=round(f1s[0], 5),
                cd8_f1=round(f1s[1], 5),
                helper_treg_f1=round(f1s[2], 5),
                prop_r=round(r_v, 5),
            )
        per_pat[p] = p_rec

    paired_tests = {}
    pairs = [
        ("S_count", "Frozen", "sc_vs_frozen"),
        ("S_count", "E_count_zm", "sc_vs_zm"),
        ("S_count", "L_count", "sc_vs_lcount"),
        ("L_count", "Frozen", "lc_vs_frozen"),
        ("L_count", "E_count_zm", "lc_vs_zm"),
    ]
    for a_k, b_k, prefix in pairs:
        if a_k in arms and b_k in arms:
            for metric in ("macro_f1", "tumor_f1", "cd8_f1", "helper_treg_f1", "prop_r"):
                va = np.array([per_pat[p][a_k][metric] for p in uniq_pats], dtype=np.float64)
                vb = np.array([per_pat[p][b_k][metric] for p in uniq_pats], dtype=np.float64)
                paired_tests[f"{prefix}_{metric}"] = _exact_wilcoxon_pair(va, vb)

    return dict(
        arch=arch,
        n_total_nuclei=int(len(y_phys)),
        n_patients=40,
        arms=list(arms),
        wall_seconds_sum=round(tot_wall, 1),
        pooled=pooled,
        patient_paired_n40=paired_tests,
        per_patient=per_pat,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unseen", "cached"], required=True)
    ap.add_argument("--bank-pkl", required=True)
    ap.add_argument("--cv-manifest", required=True)
    ap.add_argument("--upstream-dir", default="")
    ap.add_argument("--archs", required=True, help="Comma list e.g. unseen_unet16,unseen_resnet34_unet16,unseen_pix2pix16")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--arms", default="Frozen,E_count_zm,L_count,S_count")
    ap.add_argument("--fit-stride", type=int, default=2, help="2 for unseen S6 parity, 1 for cached v2.1 parity")
    ap.add_argument("--up-epochs", type=int, default=22)
    ap.add_argument("--head-epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "fold_cache"
    out_json = Path(args.out_json) if args.out_json else (out_dir / f"v22_q2_orion40_{args.mode}_atlas.json")
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    if not args.aggregate_only:
        device = torch.device(args.device)
        ob = pickle.load(open(args.bank_pkl, "rb"))
        om = json.loads(Path(args.cv_manifest).read_text())

        for arch in archs:
            up_all = None
            if args.mode == "cached":
                fnpz = Path(args.upstream_dir) / f"{arch}_all40_2804.npz"
                up_all = np.load(fnpz)["pred_u8"]
            for fi in folds:
                f_path = cache_dir / f"{arch}_fold{fi}.npz"
                if f_path.exists():
                    print(f"[V22-Q2] Skipping existing {f_path.name}", flush=True)
                    continue
                run_single_fold(
                    mode=args.mode,
                    arch=arch,
                    fold_idx=fi,
                    ob=ob,
                    fold=om["folds"][fi],
                    upstream_all=up_all,
                    arms=arms,
                    fit_stride=args.fit_stride,
                    up_epochs=args.up_epochs,
                    head_epochs=args.head_epochs,
                    batch_size=args.batch_size,
                    fold_cache_path=f_path,
                    device=device,
                )
            del up_all
            gc.collect()

    summary = dict(schema="v22-q2-orion40-multiarch-atlas-v1", mode=args.mode, fit_stride=args.fit_stride, models={})
    if out_json.exists():
        try:
            summary["models"].update(json.loads(out_json.read_text()).get("models", {}))
        except Exception:
            pass
    for arch in archs:
        agg = aggregate_arch_5fold(arch, cache_dir, arms)
        if agg is not None:
            summary["models"][arch] = agg
            p = agg["pooled"]
            parts = " | ".join(f"{a}: M={p[a]['macro_f1']:.4f} T={p[a]['per_lineage_f1']['Tumor']:.4f} CD8={p[a]['per_lineage_f1']['Cytotoxic T']:.4f} r={p[a]['prop_pearson_r']:.4f}" for a in arms)
            print(f"[V22-Q2 AGG {arch}] {agg['n_total_nuclei']} nuc, 40 pts | {parts}", flush=True)
    _atomic_write_json(out_json, summary)
    print(f"V22_Q2_COMPLETE mode={args.mode} json={out_json}", flush=True)


if __name__ == "__main__":
    main()
