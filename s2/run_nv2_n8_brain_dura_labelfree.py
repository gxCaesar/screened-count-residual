#!/usr/bin/env python3
"""NV2-N8: Non-Cancer Neuroscience Label-Free Brightfield-to-Fluorescence Virtual Staining
and Mouse-Level PBS vs LPS Neuroinflammation Discovery (Mouse Brain & Dura Mater, CC-BY-4.0).

Uses the open CC-BY-4.0 dataset `YibinLiu/mouse-brain-dura-virtual-staining` (19 male C57BL/6J
mice treated with PBS vehicle vs LPS neuroinflammation; 70,906 paired 256x256 brightfield ->
fluorescence patches across 5 neuro-immune targets: DAPI, IBA1 microglia, CD206 macrophages,
LYVE1 meningeal lymphatics, and Neutrophil — ZERO H&E stain!).

Two-part scientific benchmark:
  Part A — Official Frozen Animal-Disjoint Test Split (13 Train / 3 Val / 3 Test mice, 0% overlap):
    Evaluates ALL 5 neuro-immune targets across 3 fresh label-free architectures
    (U-Net, ResNet-34 U-Net, Pix2Pix) and 5 post-hoc calibration arms
    (upstream, S_recon, QuantileMatch, E_count, S_count) using a zero-dye brightfield 12-ch
    coordinate interface (brightfield RGB + multi-scale brightfield luminance gradients +
    upstream prediction + brightfield grayscale structural reference + peak-halo geometry).
    Exports 5 high-res label-free microscopy gallery crops for Fig. 6d & ED Fig. 10a-e.
  Part B — All-Animal Disjoint Cross-Validation for PBS vs LPS Biological Perturbation Recovery:
    Runs animal-disjoint 3-fold CV across ALL mice per neuroinflammatory target (IBA1 6 mice,
    CD206 13 mice, LYVE1 6 mice, Neutrophil 6 mice; 0% mouse overlap verified per fold),
    measuring mouse-level mean positive fluorescence area under fixed physical cutoffs and
    quantifying recovery of the true biological LPS vs PBS neuroinflammation effect
    (Cohen's d, fold-change, Mann-Whitney AUROC, and mouse-level Pearson r vs physical FL).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_auc_score

from s2 import journal_cohort_eval as jce
from s2 import run_nm_comparators20 as rcomp
from s2 import run_plan2_ablations as rp2
from s2 import run_push5_her2match_full5980 as rp5
from s2 import screened_universal_head as suh

TASKS = [
    ("DAPI", "BFtoDAPI"),
    ("IBA1", "BFtoIBA1"),
    ("CD206", "BFtoCD206"),
    ("LYVE1", "BFtoLYVE1"),
    ("Neutrophil", "BFtoNeutrophil"),
]


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _find_patches_root(base: Path) -> Path:
    for cand in (
        base / "deep learning" / "dataset_patches",
        base / "dataset_patches",
        base,
    ):
        if (cand / "BFtoDAPI").exists() or (cand / "BFtoIBA1").exists():
            return cand
    # Recursive search fallback
    for p in base.rglob("BFtoIBA1"):
        if p.is_dir():
            return p.parent
    raise FileNotFoundError(f"Could not locate BFto* patch directories under {base}")


def _parse_treatment_and_animal(row: dict) -> tuple[str, str]:
    animal = ""
    treat = ""
    for k, v in row.items():
        kl = k.lower()
        if any(t in kl for t in ("animal", "mouse", "subject")) and not animal:
            animal = str(v).strip()
        if any(t in kl for t in ("treat", "group", "condition", "stimulus", "drug")) and not treat:
            treat = str(v).strip().upper()
    # Fallback parse from path string if needed
    raw_s = " ".join(str(v) for v in row.values())
    if not treat:
        if "LPS" in raw_s.upper():
            treat = "LPS"
        elif "PBS" in raw_s.upper():
            treat = "PBS"
        else:
            treat = "UNKNOWN"
    if not animal:
        import re
        m = re.search(r"(?:EXP01_)?(M\d{3})", raw_s)
        animal = m.group(1) if m else "UNK"
    return animal, ("LPS" if "LPS" in treat else ("PBS" if "PBS" in treat else treat))


def load_task_subset(task_dir: Path, subset: str, max_patches: int = 0, seed: int = 20261121) -> dict:
    sub_dir = task_dir / subset
    meta_csv = sub_dir / "metadata.csv"
    with meta_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if max_patches > 0 and len(rows) > max_patches:
        # Stratified across animals so every mouse is balanced
        rng = np.random.default_rng(seed)
        by_a: dict[str, list[dict]] = {}
        for r in rows:
            a, _ = _parse_treatment_and_animal(r)
            by_a.setdefault(a, []).append(r)
        per_a = max(1, max_patches // max(1, len(by_a)))
        picked = []
        for a in sorted(by_a.keys()):
            lst = by_a[a]
            if len(lst) <= per_a:
                picked.extend(lst)
            else:
                idx = np.sort(rng.choice(len(lst), size=per_a, replace=False))
                picked.extend(lst[i] for i in idx.tolist())
        rows = picked

    n = len(rows)
    bf_nhwc = np.empty((n, 256, 256, 3), dtype=np.uint8)
    fl_u8 = np.empty((n, 256, 256), dtype=np.uint8)
    gray_u8 = np.empty((n, 256, 256), dtype=np.uint8)
    animals = []
    treats = []
    for i, r in enumerate(rows):
        in_p = sub_dir / r["input_path"]
        tg_p = sub_dir / r["target_path"]
        x = np.load(in_p, allow_pickle=False)
        y = np.load(tg_p, allow_pickle=False)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        elif x.shape[0] == 3 and x.shape[-1] != 3:
            x = np.transpose(x, (1, 2, 0))
        if y.ndim == 3:
            y = y[..., 0] if y.shape[-1] <= 4 else y[0]
        x_u8 = np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8) if x.dtype != np.uint8 else x
        y_u8 = np.rint(np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8) if y.dtype != np.uint8 else y
        bf_nhwc[i] = x_u8
        fl_u8[i] = y_u8
        g = 0.299 * x_u8[:, :, 0] + 0.587 * x_u8[:, :, 1] + 0.114 * x_u8[:, :, 2]
        gray_u8[i] = np.rint(np.clip(g, 0.0, 255.0)).astype(np.uint8)
        a, tr = _parse_treatment_and_animal(r)
        animals.append(a)
        treats.append(tr)
    return dict(he_nhwc=bf_nhwc, dab_u8=fl_u8, hem_u8=gray_u8, animals=animals, treats=treats)


def _eval_arms_on_task(
    tr_b: dict,
    ev_b: dict,
    up_tr: np.ndarray,
    up_ev: np.ndarray,
    cutoff: int,
    device: torch.device,
    head_epochs: int = 8,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    c12_tr = rp5.build_common12_her2(tr_b["he_nhwc"], up_tr, tr_b["hem_u8"], device)
    c12_ev = rp5.build_common12_her2(ev_b["he_nhwc"], up_ev, ev_b["hem_u8"], device)
    d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr), torch.from_numpy(tr_b["dab_u8"]), torch.from_numpy(up_ev)

    arms_ev = {"upstream": up_ev}
    _, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr, tr_b["dab_u8"], up_ev)
    arms_ev["QuantileMatch"] = qm_ev

    for arm in ("S_recon", "E_count", "S_count"):
        ev_s = []
        for sd in (202610270, 202610271):
            if arm == "E_count":
                mdl, _ = rp2.fit_ablation_marker(
                    mode="E_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                    target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                    epochs=head_epochs, batch_size=12, lr=1.5e-3,
                )
                pe, _ = rp2.predict_ablation_marker(mdl, "E_count", c12_ev, d_ev_t, device=device, batch_size=32)
            else:
                mdl, _ = suh.fit_universal_marker(
                    arm=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                    target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                    epochs=head_epochs, batch_size=12, lr=1.5e-3, deploy_anchor_weight=1.25,
                )
                pe, _ = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device=device, batch_size=32)
            ev_s.append(pe.astype(np.float32))
            del mdl
            torch.cuda.empty_cache()
        arms_ev[arm] = np.rint(np.clip(0.5 * (ev_s[0] + ev_s[1]), 0.0, 255.0)).astype(np.uint8)
    del c12_tr, c12_ev
    gc.collect()
    torch.cuda.empty_cache()

    gt_ev = ev_b["dab_u8"]
    gt_pos = gt_ev > cutoff
    gt_b = gt_pos.mean(axis=(1, 2))
    neg_m = gt_b <= 0.0005

    metrics = {}
    for arm, pr in arms_ev.items():
        ev = jce.batch_curves_and_quality_torch(pr, gt_ev, cutoff, device=device, compute_quality=True)
        ep = jce.endpoints_from_pooled_curves(
            ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cutoff
        )
        pr_pos = pr > cutoff
        pr_b = pr_pos.mean(axis=(1, 2))
        r_burd = float(stats.pearsonr(gt_b, pr_b)[0]) if np.std(pr_b) > 1e-12 else 0.0
        neg_fp = 100.0 * float(pr_pos[neg_m].mean()) if np.any(neg_m) else None
        metrics[arm] = dict(
            Fixed_tol_F1=round(float(ep["Fixed_tol_F1"]), 4),
            Exact_r0_F1=round(float(ep["Fixed_exact_F1"]), 4),
            ODS_F=round(float(ep["ODS_F"]), 4),
            AP=round(float(ep["AP"]), 4),
            SSIM=round(float(ev["ssims"].mean()), 4),
            PSNR_dB=round(float(ev["psnrs"].mean()), 3),
            burden_pearson_r=round(r_burd, 4),
            neg_patches_n=int(neg_m.sum()),
            neg_patches_fp_area_pct=round(neg_fp, 4) if neg_fp is not None else None,
        )
    return arms_ev, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-train", type=int, default=1600)
    ap.add_argument("--max-eval", type=int, default=1200)
    ap.add_argument("--upstream-epochs", type=int, default=14)
    ap.add_argument("--head-epochs", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proot = _find_patches_root(Path(args.data_root))
    print(f"[NV2-N8] Located BrainDura patches root: {proot}", flush=True)

    results = dict(
        schema="nv2-n8-brain-dura-labelfree-v1",
        dataset="YibinLiu/mouse-brain-dura-virtual-staining (CC-BY-4.0, 19 C57BL/6J mice, PBS vs LPS)",
        part_a_official_animal_split={},
        part_b_pbs_vs_lps_discovery={},
    )

    archs = ("her2_unet", "her2_resnet34", "her2_pix2pix")

    # PART A: Official Frozen Animal-Disjoint Split across ALL 5 targets
    for mname, tdir_name in TASKS:
        tdir = proot / tdir_name
        if not tdir.exists():
            continue
        tm = time.time()
        print(f"[NV2-N8 Part A] Loading {mname} ({tdir_name})...", flush=True)
        tr_b = load_task_subset(tdir, "train", max_patches=args.max_train, seed=20261121)
        va_b = load_task_subset(tdir, "val", max_patches=600, seed=20261122)
        te_b = load_task_subset(tdir, "test", max_patches=args.max_eval, seed=20261123)
        tr_a, va_a, te_a = set(tr_b["animals"]), set(va_b["animals"]), set(te_b["animals"])
        assert len(tr_a & te_a) == 0 and len(va_a & te_a) == 0, f"Animal overlap in {mname}!"

        otsu_raw = int(round(float(threshold_otsu(tr_b["dab_u8"]))))
        cutoff = max(10, min(100, otsu_raw))
        print(
            f"[NV2-N8 Part A] {mname}: tr={len(tr_b['animals'])} ({sorted(tr_a)}) "
            f"va={len(va_b['animals'])} te={len(te_b['animals'])} ({sorted(te_a)}) cutoff={cutoff}",
            flush=True,
        )

        task_res = dict(
            n_train=len(tr_b["animals"]),
            n_val=len(va_b["animals"]),
            n_test=len(te_b["animals"]),
            train_animals=sorted(tr_a),
            val_animals=sorted(va_a),
            test_animals=sorted(te_a),
            zero_animal_overlap_verified=True,
            cutoff_u8=cutoff,
            archs={},
        )

        saved_crop = False
        for arch in archs:
            ta = time.time()
            gen, up_meta = rp5.train_dab_generator(
                arch, tr_b["he_nhwc"], tr_b["dab_u8"], va_b["he_nhwc"], va_b["dab_u8"],
                device, epochs=args.upstream_epochs,
            )
            up_tr = rp5.predict_dab_u8(gen, tr_b["he_nhwc"], device)
            up_ev = rp5.predict_dab_u8(gen, te_b["he_nhwc"], device)
            del gen
            gc.collect()
            torch.cuda.empty_cache()

            arms_ev, metrics = _eval_arms_on_task(tr_b, te_b, up_tr, up_ev, cutoff, device, head_epochs=args.head_epochs)
            task_res["archs"][arch] = dict(upstream_meta=up_meta, wall_seconds=round(time.time() - ta, 1), arms=metrics)
            print(
                f"[NV2-N8 Part A] {mname:10s} {arch:14s} ({time.time()-ta:.1f}s) | "
                f"F1: up={metrics['upstream']['Fixed_tol_F1']:.4f} sr={metrics['S_recon']['Fixed_tol_F1']:.4f} "
                f"qm={metrics['QuantileMatch']['Fixed_tol_F1']:.4f} ec={metrics['E_count']['Fixed_tol_F1']:.4f} "
                f"sc={metrics['S_count']['Fixed_tol_F1']:.4f} | SSIM sc={metrics['S_count']['SSIM']:.3f} qm={metrics['QuantileMatch']['SSIM']:.3f}",
                flush=True,
            )

            if not saved_crop and arch == "her2_unet":
                gt_b = (te_b["dab_u8"] > cutoff).mean(axis=(1, 2))
                best_i = int(np.argmax(((gt_b >= 0.04) & (gt_b <= 0.45)).astype(float) * (gt_b + 0.01)))
                np.savez_compressed(
                    out_dir / f"n8_brain_{mname}_gallery_crop.npz",
                    animal=te_b["animals"][best_i],
                    treat=te_b["treats"][best_i],
                    cutoff=cutoff,
                    bf=te_b["he_nhwc"][best_i],
                    phys_fl=te_b["dab_u8"][best_i],
                    frz_fl=up_ev[best_i],
                    qm_fl=arms_ev["QuantileMatch"][best_i],
                    sc_fl=arms_ev["S_count"][best_i],
                )
                saved_crop = True

        results["part_a_official_animal_split"][mname] = task_res
        _atomic_write_json(out_dir / "nv2_n8_brain_dura_labelfree.json", results)
        del tr_b, va_b, te_b
        gc.collect()

    # PART B: All-Animal Disjoint 3-Fold CV for PBS vs LPS Biological Perturbation Discovery
    for mname, tdir_name in [("IBA1", "BFtoIBA1"), ("CD206", "BFtoCD206"), ("LYVE1", "BFtoLYVE1"), ("Neutrophil", "BFtoNeutrophil")]:
        tdir = proot / tdir_name
        if not tdir.exists():
            continue
        print(f"[NV2-N8 Part B] All-animal PBS vs LPS 3-fold animal-disjoint CV on {mname}...", flush=True)
        # Pool train+val+test metadata to obtain every animal, then split strictly by animal into 3 stratified folds
        all_subs = [load_task_subset(tdir, s, max_patches=1500 if s == "train" else 800, seed=20261121) for s in ("train", "val", "test")]
        he_all = np.concatenate([d["he_nhwc"] for d in all_subs], axis=0)
        fl_all = np.concatenate([d["dab_u8"] for d in all_subs], axis=0)
        gr_all = np.concatenate([d["hem_u8"] for d in all_subs], axis=0)
        an_all = np.asarray(sum([d["animals"] for d in all_subs], []))
        tr_all = np.asarray(sum([d["treats"] for d in all_subs], []))

        pat_treat = {a: str(tr_all[an_all == a][0]) for a in sorted(set(an_all.tolist()))}
        pbs_a = sorted([a for a, t in pat_treat.items() if t == "PBS"])
        lps_a = sorted([a for a, t in pat_treat.items() if t == "LPS"])
        if len(pbs_a) < 2 or len(lps_a) < 2:
            print(f"[NV2-N8 Part B] Skipping {mname}: PBS={len(pbs_a)} LPS={len(lps_a)}", flush=True)
            continue

        # Assign 3 stratified animal-disjoint folds
        n_folds = min(3, min(len(pbs_a), len(lps_a)))
        fold_test_animals = [[] for _ in range(n_folds)]
        for i, a in enumerate(pbs_a):
            fold_test_animals[i % n_folds].append(a)
        for i, a in enumerate(lps_a):
            fold_test_animals[i % n_folds].append(a)

        per_mouse_density = {arm: {} for arm in ("Physical", "upstream", "S_recon", "QuantileMatch", "E_count", "S_count")}
        fold_f1s = {arm: [] for arm in ("upstream", "S_recon", "QuantileMatch", "E_count", "S_count")}

        for fidx in range(n_folds):
            ev_set = set(fold_test_animals[fidx])
            tr_set = set(pat_treat.keys()) - ev_set
            assert len(ev_set & tr_set) == 0
            ev_m = np.isin(an_all, list(ev_set))
            tr_m = np.isin(an_all, list(tr_set))
            # Split tr_set into inner train / inner val by animal
            tr_list = sorted(tr_set)
            iv_set = {tr_list[0]}
            it_m = np.isin(an_all, [a for a in tr_list if a not in iv_set])
            iv_m = np.isin(an_all, list(iv_set))

            tr_b = dict(he_nhwc=he_all[tr_m], dab_u8=fl_all[tr_m], hem_u8=gr_all[tr_m])
            it_b = dict(he_nhwc=he_all[it_m], dab_u8=fl_all[it_m], hem_u8=gr_all[it_m])
            iv_b = dict(he_nhwc=he_all[iv_m], dab_u8=fl_all[iv_m], hem_u8=gr_all[iv_m])
            ev_b = dict(he_nhwc=he_all[ev_m], dab_u8=fl_all[ev_m], hem_u8=gr_all[ev_m])
            ev_an = an_all[ev_m]

            cutoff = max(10, min(100, int(round(float(threshold_otsu(tr_b["dab_u8"]))))))
            gen, _ = rp5.train_dab_generator(
                "her2_unet", it_b["he_nhwc"], it_b["dab_u8"], iv_b["he_nhwc"], iv_b["dab_u8"],
                device, epochs=12,
            )
            up_tr = rp5.predict_dab_u8(gen, tr_b["he_nhwc"], device)
            up_ev = rp5.predict_dab_u8(gen, ev_b["he_nhwc"], device)
            del gen
            gc.collect()
            torch.cuda.empty_cache()

            arms_ev, mets = _eval_arms_on_task(tr_b, ev_b, up_tr, up_ev, cutoff, device, head_epochs=6)
            for arm in fold_f1s:
                fold_f1s[arm].append(mets[arm]["Fixed_tol_F1"])
            # Aggregate per-mouse mean positive-area density (%)
            for a in sorted(ev_set):
                am = ev_an == a
                per_mouse_density["Physical"][a] = round(100.0 * float((ev_b["dab_u8"][am] > cutoff).mean()), 5)
                for arm, pr in arms_ev.items():
                    per_mouse_density[arm][a] = round(100.0 * float((pr[am] > cutoff).mean()), 5)

        # Quantify PBS vs LPS biological effect recovery per arm at the independent mouse unit
        animals_sorted = sorted(pat_treat.keys())
        y_lps = np.asarray([1 if pat_treat[a] == "LPS" else 0 for a in animals_sorted], dtype=np.int32)
        phys_v = np.asarray([per_mouse_density["Physical"][a] for a in animals_sorted], dtype=np.float64)

        bio_arms = {}
        for arm in ("Physical", "upstream", "S_recon", "QuantileMatch", "E_count", "S_count"):
            v = np.asarray([per_mouse_density[arm][a] for a in animals_sorted], dtype=np.float64)
            v_pbs = v[y_lps == 0]
            v_lps = v[y_lps == 1]
            s_pool = np.sqrt(0.5 * (np.var(v_pbs, ddof=1) + np.var(v_lps, ddof=1)))
            cohen_d = float((v_lps.mean() - v_pbs.mean()) / max(s_pool, 1e-12)) if s_pool > 1e-12 else 0.0
            fc = float(v_lps.mean() / max(v_pbs.mean(), 1e-6))
            auc = float(roc_auc_score(y_lps, v)) if np.std(v) > 1e-12 else 0.5
            if np.mean(phys_v[y_lps == 1]) < np.mean(phys_v[y_lps == 0]):
                auc_dir = 1.0 - auc
            else:
                auc_dir = auc
            mwu_p = float(stats.mannwhitneyu(v_lps, v_pbs, alternative="two-sided").pvalue) if np.std(v) > 1e-12 else 1.0
            r_vs_phys = float(stats.pearsonr(phys_v, v)[0]) if np.std(v) > 1e-12 and np.std(phys_v) > 1e-12 else 0.0
            mae_vs_phys = float(np.mean(np.abs(v - phys_v)))
            bio_arms[arm] = dict(
                pbs_mean_pct=round(float(v_pbs.mean()), 4),
                lps_mean_pct=round(float(v_lps.mean()), 4),
                fold_change_lps_over_pbs=round(fc, 4),
                cohen_d_lps_minus_pbs=round(cohen_d, 4),
                directional_auroc=round(auc_dir, 4),
                mannwhitney_p=round(mwu_p, 6),
                mouse_pearson_vs_physical=round(r_vs_phys, 4),
                mouse_density_mae_pct=round(mae_vs_phys, 4),
                mean_cv_f1=round(float(np.mean(fold_f1s[arm])), 4) if arm in fold_f1s else 1.0,
                per_mouse_pct=per_mouse_density[arm],
            )
        results["part_b_pbs_vs_lps_discovery"][mname] = dict(
            n_mice=len(animals_sorted),
            n_pbs=len(pbs_a),
            n_lps=len(lps_a),
            n_folds=n_folds,
            zero_animal_overlap_verified=True,
            mouse_treatments=pat_treat,
            arms=bio_arms,
        )
        print(
            f"[NV2-N8 Part B] {mname:10s} (n={len(animals_sorted)} mice: {len(pbs_a)} PBS, {len(lps_a)} LPS) | "
            f"Phys d={bio_arms['Physical']['cohen_d_lps_minus_pbs']:+.2f} AUC={bio_arms['Physical']['directional_auroc']:.3f} | "
            f"Frz d={bio_arms['upstream']['cohen_d_lps_minus_pbs']:+.2f} r={bio_arms['upstream']['mouse_pearson_vs_physical']:.3f} MAE={bio_arms['upstream']['mouse_density_mae_pct']:.3f}% | "
            f"S_cnt d={bio_arms['S_count']['cohen_d_lps_minus_pbs']:+.2f} r={bio_arms['S_count']['mouse_pearson_vs_physical']:.3f} MAE={bio_arms['S_count']['mouse_density_mae_pct']:.3f}%",
            flush=True,
        )
        _atomic_write_json(out_dir / "nv2_n8_brain_dura_labelfree.json", results)
        del all_subs, he_all, fl_all, gr_all
        gc.collect()

    results["wall_seconds"] = round(time.time() - t0, 1)
    _atomic_write_json(out_dir / "nv2_n8_brain_dura_labelfree.json", results)
    print("NV2_N8_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
