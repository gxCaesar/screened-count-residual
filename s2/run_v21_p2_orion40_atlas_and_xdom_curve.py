#!/usr/bin/env python3
"""v2.1 P2: Independent Specimen Scaling & Clinical Deployment Curves.

Two self-contained modes:
  1. ``--task orion40_atlas``:
       Full 5-fold patient-disjoint all-40-patient (~350,000+ scored nuclei across C01-C40)
       15-marker single-cell phenotyping & 8-lineage spatial atlas comparing Frozen, E_count_zm,
       and S_count on Orion ConvNeXt and W-MSE Swin. Computes N=40 patient-level paired Wilcoxon
       signed-rank tests on 8-lineage macro-F1, Helper/Treg T F1, Cytotoxic T (CD8a+) F1, and
       lineage-proportion Pearson r, and exports a 20,000-nucleus 40-patient UMAP NPZ.
  2. ``--task xdom_budget``:
       Cross-hospital target-site calibration sample-efficiency & slide-budget curves for BOTH
       frozen Beijing generators (BCIStainer 29.91M and official PyramidPix2Pix 17.44M) onto ALL
       5,980 locked-test Dutch HER2match tiles (5 held-out slides, 3,368 negative-slide tiles):
         - Tile budgets: N in {4, 16, 64, 256, 1200} (3 repeats for N <= 64)
         - Slide budgets: 1 positive Dutch slide vs 2 slides vs all 6 Dutch training slides
         - Compares Frozen, QuantileMatch, E_count, E_count_zm, and S_count on Fixed_tol_F1,
           Exact_r0_F1, SSIM, 3,368-tile negative-slide FP%, and per-slide breakdown.
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
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from s2 import clinical_cell_survival as ccs
from s2 import journal_cohort_eval as jce
from s2 import run_her2match_dab_campaign as rher
from s2 import run_nm_comparators20 as rcomp
from s2 import run_nv2_n1_orion_clinical_umap as rn1
from s2 import run_nv2_n5_pyramidpix2pix_her2 as rn5
from s2 import run_orion_multibackbone_survival as ror
from s2 import run_plan2_ablations as rp2
from s2 import run_push5_her2match_full5980 as rp5
from s2 import run_push6_cross_hospital_bcistainer as rp6
from s2 import screened_universal_head as suh


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def run_orion40_5fold_atlas(
    bank_pkl: Path,
    upstream_dir: Path,
    cv_manifest: Path,
    out_dir: Path,
    device: torch.device,
    models: list[str],
) -> None:
    t0 = time.time()
    ob = pickle.load(open(bank_pkl, "rb"))
    om = json.loads(cv_manifest.read_text())
    pats_arr = np.asarray(ob["patients"])
    cls_names = ["Tumor", "Cytotoxic T", "Helper/Treg T", "B cell", "Macrophage", "Endothelia", "Stroma", "Other"]

    out_json = out_dir / "v21_p2_orion40_5fold_atlas.json"
    summary = dict(schema="v21-p2-orion40-5fold-atlas-v1", models={})
    if out_json.exists():
        try:
            summary["models"].update(json.loads(out_json.read_text()).get("models", {}))
        except Exception:
            pass

    for mkey in models:
        if mkey in summary["models"]:
            print(f"[V21-P2 ORION40] Skipping completed {mkey}", flush=True)
            continue
        fnpz = upstream_dir / f"{mkey}_all40_2804.npz"
        if not fnpz.exists():
            print(f"[V21-P2 ORION40] Missing {fnpz}", flush=True)
            continue
        tm = time.time()
        up_all = np.load(fnpz)["pred_u8"]

        all_phys_m, all_frz_m, all_zm_m, all_sc_m, all_pats = [], [], [], [], []
        all_y_phys, all_y_frz, all_y_zm, all_y_sc = [], [], [], []

        for fi, fold in enumerate(om["folds"]):
            tr_idx = np.nonzero(np.isin(pats_arr, fold["inner_train_slides"]))[0]
            ev_idx = np.nonzero(np.isin(pats_arr, fold["outer_test_slides"]))[0]
            pix_cuts, cell_cuts_dict = ror.calibrate_orion_physical_cutoffs_from_train(
                ob["targ_nchw"][tr_idx], tr_idx, ob["nuclei_records"]
            )
            pix_cuts = [int(c) for c in pix_cuts]
            if isinstance(cell_cuts_dict, dict):
                cell_cuts16 = np.asarray([float(cell_cuts_dict[m]) for m in ror.ORION_MARKERS], dtype=np.float64)
            else:
                cell_cuts16 = np.zeros(16, dtype=np.float64)
                cell_cuts16[0] = float(pix_cuts[0])
                cell_cuts16[1:] = np.asarray(cell_cuts_dict, dtype=np.float64)

            he_tr, he_ev = ob["he_nhwc"][tr_idx], ob["he_nhwc"][ev_idx]
            up_tr, up_ev = up_all[tr_idx], up_all[ev_idx]
            gt_tr = ob["targ_nchw"][tr_idx]

            zm_ev = up_ev.copy()
            sc_ev = up_ev.copy()
            for ch in range(16):
                c12_tr = ror.build_common12_orion(he_tr, up_tr, ch, device=device)
                c12_ev = ror.build_common12_orion(he_ev, up_ev, ch, device=device)
                d_tr_t, y_tr_t, d_ev_t = torch.from_numpy(up_tr[:, ch]), torch.from_numpy(gt_tr[:, ch]), torch.from_numpy(up_ev[:, ch])

                # 2-seed ensemble for both E_count_zm and S_count (6 epochs fast 16-ch)
                for arm_nm, dst_arr in (("E_count_zm", zm_ev), ("S_count", sc_ev)):
                    preds_s = []
                    for sd in (202610270, 202610271):
                        if arm_nm == "E_count_zm":
                            mdl, _ = rp2.fit_ablation_marker(
                                mode="E_count_zm", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                                target_u8_train=y_tr_t, cutoff=pix_cuts[ch], device=device,
                                epochs=6, batch_size=16, lr=1.5e-3, deploy_anchor_weight=1.25,
                            )
                            pe, _ = rp2.predict_ablation_marker(mdl, "E_count_zm", c12_ev, d_ev_t, device=device, batch_size=32)
                        else:
                            mdl, _ = suh.fit_universal_marker(
                                arm="S_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                                target_u8_train=y_tr_t, cutoff=pix_cuts[ch], device=device,
                                epochs=6, batch_size=16, lr=1.5e-3, deploy_anchor_weight=1.25,
                            )
                            pe, _ = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device=device, batch_size=32)
                        preds_s.append(pe.astype(np.float32))
                        del mdl
                        torch.cuda.empty_cache()
                    dst_arr[:, ch] = np.rint(np.clip(0.5 * (preds_s[0] + preds_s[1]), 0.0, 255.0)).astype(np.uint8)
                del c12_tr, c12_ev
                gc.collect()
                torch.cuda.empty_cache()

            # Extract out-of-fold nuclei for this fold's 8 held-out patients
            f_phys, f_frz, f_zm, f_sc, f_pat = [], [], [], [], []
            for k_i, idx in enumerate(ev_idx.tolist()):
                rec = ob["nuclei_records"][idx]
                if rec is None:
                    continue
                pm_gt = rec["gt_means16"]
                f_phys.append(pm_gt)
                f_frz.append(ccs.extract_means_with_index(up_ev[k_i], rec["idx_spec"]))
                f_zm.append(ccs.extract_means_with_index(zm_ev[k_i], rec["idx_spec"]))
                f_sc.append(ccs.extract_means_with_index(sc_ev[k_i], rec["idx_spec"]))
                f_pat.extend([str(ob["patients"][idx])] * len(pm_gt))

            Xp = np.concatenate(f_phys, axis=0)
            Xf = np.concatenate(f_frz, axis=0)
            Xz = np.concatenate(f_zm, axis=0)
            Xs = np.concatenate(f_sc, axis=0)
            all_phys_m.append(Xp)
            all_frz_m.append(Xf)
            all_zm_m.append(Xz)
            all_sc_m.append(Xs)
            all_pats.extend(f_pat)
            all_y_phys.append(rn1._assign_bio_phenotype_8class(Xp, cell_cuts16))
            all_y_frz.append(rn1._assign_bio_phenotype_8class(Xf, cell_cuts16))
            all_y_zm.append(rn1._assign_bio_phenotype_8class(Xz, cell_cuts16))
            all_y_sc.append(rn1._assign_bio_phenotype_8class(Xs, cell_cuts16))
            print(f"[V21-P2 ORION40] {mkey} fold {fi}/4 done: {len(Xp)} nuclei ({time.time()-tm:.1f}s)", flush=True)

        X_phys = np.concatenate(all_phys_m, axis=0)
        y_phys = np.concatenate(all_y_phys, axis=0)
        y_frz = np.concatenate(all_y_frz, axis=0)
        y_zm = np.concatenate(all_y_zm, axis=0)
        y_sc = np.concatenate(all_y_sc, axis=0)
        pat_vec = np.asarray(all_pats)
        uniq_pats = sorted(set(all_pats))
        assert len(uniq_pats) == 40, f"Expected 40 patients, got {len(uniq_pats)}"

        # Global pooled metrics across all ~350k+ nuclei
        def _pooled(y_hat: np.ndarray) -> dict:
            f1s = [float(f1_score(y_phys == c, y_hat == c, zero_division=0)) for c in range(8)]
            pp = np.bincount(y_phys, minlength=8) / len(y_phys)
            ph = np.bincount(y_hat, minlength=8) / len(y_hat)
            return dict(
                macro_f1=round(float(np.mean(f1s)), 5),
                per_lineage_f1={cls_names[c]: round(f1s[c], 5) for c in range(8)},
                prop_pearson_r=round(float(stats.pearsonr(pp, ph)[0]), 5),
            )

        # Patient-level (N=40 independent patients!) paired Wilcoxon tests
        per_pat = {}
        for p in uniq_pats:
            m = pat_vec == p
            yp, yf, yz, ys = y_phys[m], y_frz[m], y_zm[m], y_sc[m]
            pp = np.bincount(yp, minlength=8) / max(1, len(yp))

            def _p_stat(yh):
                f1s = [float(f1_score(yp == c, yh == c, zero_division=0)) for c in range(8)]
                ph = np.bincount(yh, minlength=8) / max(1, len(yh))
                r_v = float(stats.pearsonr(pp, ph)[0]) if np.std(ph) > 1e-12 and np.std(pp) > 1e-12 else 0.0
                return dict(
                    macro_f1=round(float(np.mean(f1s)), 5),
                    cd8_f1=round(f1s[1], 5),
                    helper_treg_f1=round(f1s[2], 5),
                    prop_r=round(r_v, 5),
                )

            per_pat[p] = dict(n_nuclei=int(m.sum()), Frozen=_p_stat(yf), E_count_zm=_p_stat(yz), S_count=_p_stat(ys))

        def _wilcox(a_key: str, b_key: str, metric: str) -> dict:
            va = np.array([per_pat[p][a_key][metric] for p in uniq_pats], dtype=np.float64)
            vb = np.array([per_pat[p][b_key][metric] for p in uniq_pats], dtype=np.float64)
            diff = va - vb
            wins = int((diff > 0).sum())
            p_val = float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")[1]) if np.any(diff != 0) else 1.0
            return dict(
                mean_a=round(float(va.mean()), 5),
                mean_b=round(float(vb.mean()), 5),
                mean_delta=round(float(diff.mean()), 5),
                wins_of_40=wins,
                wilcoxon_p=float(f"{p_val:.4e}"),
            )

        paired_tests = dict(
            sc_vs_frozen_macro_f1=_wilcox("S_count", "Frozen", "macro_f1"),
            sc_vs_zm_macro_f1=_wilcox("S_count", "E_count_zm", "macro_f1"),
            sc_vs_frozen_prop_r=_wilcox("S_count", "Frozen", "prop_r"),
            sc_vs_zm_prop_r=_wilcox("S_count", "E_count_zm", "prop_r"),
            sc_vs_frozen_cd8_f1=_wilcox("S_count", "Frozen", "cd8_f1"),
            sc_vs_zm_cd8_f1=_wilcox("S_count", "E_count_zm", "cd8_f1"),
            sc_vs_frozen_helper_f1=_wilcox("S_count", "Frozen", "helper_treg_f1"),
            sc_vs_zm_helper_f1=_wilcox("S_count", "E_count_zm", "helper_treg_f1"),
        )

        # Save 20,000-nucleus 40-patient UMAP for ConvNeXt
        if mkey == "miphei_convnext":
            rng = np.random.default_rng(20261121)
            n_sub = min(20000, len(X_phys))
            sub_idx = rng.choice(len(X_phys), size=n_sub, replace=False)
            X_log = np.log1p(X_phys[sub_idx, 1:])
            X_std = (X_log - X_log.mean(0, keepdims=True)) / np.maximum(X_log.std(0, keepdims=True), 1e-6)
            emb_method = "PCA"
            try:
                import umap
                emb2d = umap.UMAP(n_neighbors=25, min_dist=0.35, random_state=42).fit_transform(X_std).astype(np.float32)
                emb_method = "UMAP"
            except Exception:
                emb2d = PCA(n_components=2, random_state=42).fit_transform(X_std).astype(np.float32)
            np.savez_compressed(
                out_dir / "v21_orion40_5fold_umap.npz",
                emb2d=emb2d,
                emb_method=emb_method,
                y_phys=y_phys[sub_idx].astype(np.int8),
                y_frz=y_frz[sub_idx].astype(np.int8),
                y_zm=y_zm[sub_idx].astype(np.int8),
                y_sc=y_sc[sub_idx].astype(np.int8),
                patients=pat_vec[sub_idx],
            )
            print(f"[V21-P2 ORION40] Saved 40-patient {n_sub}-nucleus {emb_method} NPZ", flush=True)

        summary["models"][mkey] = dict(
            n_total_nuclei=int(len(X_phys)),
            n_patients=40,
            wall_seconds=round(time.time() - tm, 1),
            pooled=dict(Frozen=_pooled(y_frz), E_count_zm=_pooled(y_zm), S_count=_pooled(y_sc)),
            patient_paired_n40=paired_tests,
            per_patient=per_pat,
        )
        print(
            f"[V21-P2 ORION40] {mkey} DONE ({len(X_phys)} nuclei, 40 pts) | "
            f"Macro-F1: Frz={summary['models'][mkey]['pooled']['Frozen']['macro_f1']:.4f} "
            f"ZM={summary['models'][mkey]['pooled']['E_count_zm']['macro_f1']:.4f} "
            f"SC={summary['models'][mkey]['pooled']['S_count']['macro_f1']:.4f} | "
            f"SC vs Frz p={paired_tests['sc_vs_frozen_macro_f1']['wilcoxon_p']} | "
            f"SC vs ZM p={paired_tests['sc_vs_zm_macro_f1']['wilcoxon_p']}",
            flush=True,
        )
        _atomic_write_json(out_json, summary)
    print(f"V21_P2_ORION40_COMPLETE ({time.time()-t0:.1f}s)", flush=True)


def run_xdom_calibration_budget(
    her2_root: Path,
    bci_src: Path,
    bci_ckpt: Path,
    p2p_repo: Path,
    p2p_ckpt: Path,
    out_dir: Path,
    device: torch.device,
) -> None:
    t0 = time.time()
    tiles_root = her2_root / "tiles"
    print("[V21-P2 XDOM] Loading Dutch HER2match train (1,200 tiles, 6 slides) & ALL 5,980 test tiles...", flush=True)
    tr_pairs = rher.discover_split_pairs(tiles_root, "train", tiles_per_wsi=200, forbid_test=False)
    te_pairs = rher.discover_split_pairs(tiles_root, "test", tiles_per_wsi=0, forbid_test=False)
    tr_bank = rher.load_her2_dab_bank(tiles_root, "train", tr_pairs, dab_upper=1.35, img_size=256, forbid_test=False)
    te_bank = rher.load_her2_dab_bank(tiles_root, "test", te_pairs, dab_upper=1.35, img_size=256, forbid_test=False)

    tr_wsis = np.asarray(tr_bank["wsis"])
    te_wsis = np.asarray(te_bank["wsis"])
    neg_sl_mask = np.isin(te_wsis, ["wsi001", "wsi008", "wsi016"])
    gt_ev = te_bank["dab_u8"]
    cutoff = 23

    # Rank training slides by positive pixel rate so 1-slide / 2-slide subsets include positive tissue
    tr_sl_pos = sorted(
        [(float((tr_bank["dab_u8"][tr_wsis == sl] > cutoff).mean()), sl) for sl in sorted(set(tr_wsis.tolist()))],
        reverse=True,
    )
    top1_slide = [tr_sl_pos[0][1]]
    top2_slides = [tr_sl_pos[0][1], tr_sl_pos[1][1]]

    out_json = out_dir / "v21_p2_xdom_budget_and_zm.json"
    res = dict(
        schema="v21-p2-xdom-budget-and-zm-v1",
        n_train_total=len(tr_wsis),
        n_test_total=len(te_wsis),
        n_neg_slide_tiles=int(neg_sl_mask.sum()),
        train_slides_by_positivity=[dict(slide=sl, gt_pos_pct=round(100.0 * v, 3)) for v, sl in tr_sl_pos],
        generators={},
    )
    if out_json.exists():
        try:
            res["generators"].update(json.loads(out_json.read_text()).get("generators", {}))
        except Exception:
            pass

    gen_specs = [
        ("BCIStainer", "bci"),
        ("PyramidPix2Pix", "p2p"),
    ]
    for gname, gkind in gen_specs:
        if gname in res["generators"]:
            print(f"[V21-P2 XDOM] Skipping completed {gname}", flush=True)
            continue
        tg = time.time()
        if gkind == "bci":
            gen, n_par = rp6.load_bcistainer_generator(bci_src, bci_ckpt, device)
            up_tr = rp6.predict_bcistainer_dab_u8(gen, tr_bank["he_nhwc"], 1.35, device, batch_size=8)
            up_ev = rp6.predict_bcistainer_dab_u8(gen, te_bank["he_nhwc"], 1.35, device, batch_size=8)
        else:
            gen, n_par = rn5.load_pyramid_pix2pix(p2p_repo, p2p_ckpt, device)
            up_tr = rn5.predict_p2p_dab_u8(gen, tr_bank["he_nhwc"], 1.35, device, batch_size=8)
            up_ev = rn5.predict_p2p_dab_u8(gen, te_bank["he_nhwc"], 1.35, device, batch_size=8)
        del gen
        gc.collect()
        torch.cuda.empty_cache()

        c12_tr_all = rp5.build_common12_her2(tr_bank["he_nhwc"], up_tr, tr_bank["hem_u8"], device)
        c12_ev = rp5.build_common12_her2(te_bank["he_nhwc"], up_ev, te_bank["hem_u8"], device)
        d_ev_t = torch.from_numpy(up_ev)

        def _score_eval(pr: np.ndarray) -> dict:
            ev = jce.batch_curves_and_quality_torch(pr, gt_ev, cutoff, device=device, compute_quality=True)
            ep = jce.endpoints_from_pooled_curves(
                ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cutoff
            )
            per_sl = {}
            for sl in sorted(set(te_wsis.tolist())):
                sm = te_wsis == sl
                evs = jce.batch_curves_and_quality_torch(pr[sm], gt_ev[sm], cutoff, device=device, compute_quality=False)
                eps = jce.endpoints_from_pooled_curves(
                    evs["K"].sum(0), evs["U"].sum(0), evs["V"].sum(0), evs["T"].sum(0), int(evs["G"].sum()), cutoff
                )
                per_sl[sl] = dict(
                    Fixed_tol_F1=round(float(eps["Fixed_tol_F1"]), 5),
                    fp_area_pct=round(100.0 * float((pr[sm] > cutoff).mean()), 4),
                )
            return dict(
                Fixed_tol_F1=round(float(ep["Fixed_tol_F1"]), 5),
                Exact_r0_F1=round(float(ep["Fixed_exact_F1"]), 5),
                ODS_F=round(float(ep["ODS_F"]), 5),
                SSIM=round(float(ev["ssims"].mean()), 5),
                neg_3368_fp_pct=round(100.0 * float((pr[neg_sl_mask] > cutoff).mean()), 5),
                per_slide=per_sl,
            )

        g_out = dict(n_params=n_par, upstream=_score_eval(up_ev), tile_budgets={}, slide_budgets={})

        # 1. Tile budget curve: N in {4, 16, 64, 256, 1200}
        pos_tr_idx = np.nonzero((tr_bank["dab_u8"] > cutoff).mean(axis=(1, 2)) > 0.005)[0]
        neg_tr_idx = np.nonzero((tr_bank["dab_u8"] > cutoff).mean(axis=(1, 2)) <= 0.005)[0]

        for n_bud in (4, 16, 64, 256, 1200):
            n_reps = 3 if n_bud <= 64 else 1
            rep_arms = {a: [] for a in ("QuantileMatch", "E_count", "E_count_zm", "S_count")}
            for r_i in range(n_reps):
                if n_bud >= len(tr_wsis):
                    sub = np.arange(len(tr_wsis))
                else:
                    rng = np.random.default_rng(20261121 + 100 * n_bud + r_i)
                    # Balanced stratified sampling (ensure both positive and negative target tiles)
                    n_p = min(len(pos_tr_idx), max(2, n_bud // 2))
                    n_n = min(len(neg_tr_idx), n_bud - n_p)
                    sub = np.sort(np.concatenate([
                        rng.choice(pos_tr_idx, size=n_p, replace=False),
                        rng.choice(neg_tr_idx, size=n_n, replace=False),
                    ]))
                c_s = c12_tr_all[sub]
                d_s = torch.from_numpy(up_tr[sub])
                y_s = torch.from_numpy(tr_bank["dab_u8"][sub])
                bs_s = min(12, max(2, len(sub) // 2))
                ep_s = 6 if n_bud <= 64 else 8

                _, qm_e, _ = rcomp.apply_quantile_match_u8(up_tr[sub], tr_bank["dab_u8"][sub], up_ev)
                rep_arms["QuantileMatch"].append(_score_eval(qm_e))

                for arm in ("E_count", "E_count_zm", "S_count"):
                    # 2-seed for full N=1200/256, 1-seed per repeat x 3 repeats for N<=64
                    sds = (202610270, 202610271) if n_bud >= 256 else (202610270 + r_i,)
                    ev_list = []
                    for sd in sds:
                        if arm in ("E_count", "E_count_zm"):
                            mdl, _ = rp2.fit_ablation_marker(
                                mode=arm, seed=sd, common_train=c_s, donor_u8_train=d_s,
                                target_u8_train=y_s, cutoff=cutoff, device=device,
                                epochs=ep_s, batch_size=bs_s, lr=1.5e-3,
                            )
                            pe, _ = rp2.predict_ablation_marker(mdl, arm, c12_ev, d_ev_t, device=device, batch_size=32)
                        else:
                            mdl, _ = suh.fit_universal_marker(
                                arm="S_count", seed=sd, common_train=c_s, donor_u8_train=d_s,
                                target_u8_train=y_s, cutoff=cutoff, device=device,
                                epochs=ep_s, batch_size=bs_s, lr=1.5e-3, deploy_anchor_weight=1.25,
                            )
                            pe, _ = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device=device, batch_size=32)
                        ev_list.append(pe.astype(np.float32))
                        del mdl
                        torch.cuda.empty_cache()
                    ens = np.rint(np.clip(np.mean(ev_list, axis=0), 0.0, 255.0)).astype(np.uint8)
                    rep_arms[arm].append(_score_eval(ens))

            bud_agg = {}
            for arm, lst in rep_arms.items():
                bud_agg[arm] = dict(
                    n_repeats=len(lst),
                    Fixed_tol_F1_mean=round(float(np.mean([x["Fixed_tol_F1"] for x in lst])), 5),
                    Fixed_tol_F1_sd=round(float(np.std([x["Fixed_tol_F1"] for x in lst], ddof=1)) if len(lst) > 1 else 0.0, 5),
                    Exact_r0_F1_mean=round(float(np.mean([x["Exact_r0_F1"] for x in lst])), 5),
                    SSIM_mean=round(float(np.mean([x["SSIM"] for x in lst])), 5),
                    neg_3368_fp_pct_mean=round(float(np.mean([x["neg_3368_fp_pct"] for x in lst])), 5),
                    repeats=lst,
                )
            g_out["tile_budgets"][str(n_bud)] = bud_agg
            print(
                f"[V21-P2 XDOM {gname}] N={n_bud:4d} | F1: QM={bud_agg['QuantileMatch']['Fixed_tol_F1_mean']:.4f} "
                f"EC={bud_agg['E_count']['Fixed_tol_F1_mean']:.4f} ZM={bud_agg['E_count_zm']['Fixed_tol_F1_mean']:.4f} "
                f"SC={bud_agg['S_count']['Fixed_tol_F1_mean']:.4f} | negFP: ZM={bud_agg['E_count_zm']['neg_3368_fp_pct_mean']:.3f}% "
                f"SC={bud_agg['S_count']['neg_3368_fp_pct_mean']:.3f}%",
                flush=True,
            )

        # 2. Slide budget curve: 1 positive slide vs 2 slides
        for sb_key, sl_list in (("1_pos_slide", top1_slide), ("2_slides", top2_slides)):
            sub = np.nonzero(np.isin(tr_wsis, sl_list))[0]
            c_s = c12_tr_all[sub]
            d_s = torch.from_numpy(up_tr[sub])
            y_s = torch.from_numpy(tr_bank["dab_u8"][sub])
            sb_arms = {}
            _, qm_e, _ = rcomp.apply_quantile_match_u8(up_tr[sub], tr_bank["dab_u8"][sub], up_ev)
            sb_arms["QuantileMatch"] = _score_eval(qm_e)
            for arm in ("E_count", "E_count_zm", "S_count"):
                ev_list = []
                for sd in (202610270, 202610271):
                    if arm in ("E_count", "E_count_zm"):
                        mdl, _ = rp2.fit_ablation_marker(
                            mode=arm, seed=sd, common_train=c_s, donor_u8_train=d_s,
                            target_u8_train=y_s, cutoff=cutoff, device=device,
                            epochs=8, batch_size=12, lr=1.5e-3,
                        )
                        pe, _ = rp2.predict_ablation_marker(mdl, arm, c12_ev, d_ev_t, device=device, batch_size=32)
                    else:
                        mdl, _ = suh.fit_universal_marker(
                            arm="S_count", seed=sd, common_train=c_s, donor_u8_train=d_s,
                            target_u8_train=y_s, cutoff=cutoff, device=device,
                            epochs=8, batch_size=12, lr=1.5e-3, deploy_anchor_weight=1.25,
                        )
                        pe, _ = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device=device, batch_size=32)
                    ev_list.append(pe.astype(np.float32))
                    del mdl
                    torch.cuda.empty_cache()
                ens = np.rint(np.clip(0.5 * (ev_list[0] + ev_list[1]), 0.0, 255.0)).astype(np.uint8)
                sb_arms[arm] = _score_eval(ens)
            g_out["slide_budgets"][sb_key] = dict(slides=sl_list, n_tiles=int(len(sub)), arms=sb_arms)
            print(
                f"[V21-P2 XDOM {gname}] {sb_key} ({len(sub)} tiles) | SC F1={sb_arms['S_count']['Fixed_tol_F1']:.4f} "
                f"ZM F1={sb_arms['E_count_zm']['Fixed_tol_F1']:.4f} negFP SC={sb_arms['S_count']['neg_3368_fp_pct']:.3f}%",
                flush=True,
            )

        g_out["wall_seconds"] = round(time.time() - tg, 1)
        res["generators"][gname] = g_out
        _atomic_write_json(out_json, res)
        del c12_tr_all, c12_ev
        gc.collect()
        torch.cuda.empty_cache()

    print(f"V21_P2_XDOM_COMPLETE ({time.time()-t0:.1f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["orion40_atlas", "xdom_budget"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bank-pkl", default="")
    ap.add_argument("--upstream-dir", default="")
    ap.add_argument("--cv-manifest", default="")
    ap.add_argument("--models", default="miphei_convnext,miphei_hemit_arch")
    ap.add_argument("--her2-root", default="")
    ap.add_argument("--bci-src", default="")
    ap.add_argument("--bci-ckpt", default="")
    ap.add_argument("--p2p-repo", default="")
    ap.add_argument("--p2p-ckpt", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.task == "orion40_atlas":
        run_orion40_5fold_atlas(
            Path(args.bank_pkl), Path(args.upstream_dir), Path(args.cv_manifest),
            out_dir, device, [m.strip() for m in args.models.split(",") if m.strip()],
        )
    else:
        run_xdom_calibration_budget(
            Path(args.her2_root), Path(args.bci_src), Path(args.bci_ckpt),
            Path(args.p2p_repo), Path(args.p2p_ckpt), out_dir, device,
        )


if __name__ == "__main__":
    main()
