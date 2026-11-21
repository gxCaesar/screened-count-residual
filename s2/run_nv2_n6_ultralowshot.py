#!/usr/bin/env python3
"""NV2-N6: Ultra-Low-Shot Calibration Frontier (N = 4, 8, 16, 32, 64 labeled tiles).

Evaluates 3 random subsamples x 2 seeds on HEMIT (ConvNeXt, W-MSE Swin, Pix2Pix, DGR)
comparing Frozen, S_recon, QuantileMatch, and S_count at ultra-low budgets N in {4, 8, 16, 32, 64},
mapping the exact minimum-label clinical deployment boundary.
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

from s2 import journal_cohort_eval as jce
from s2 import run_hemit_multibackbone_cell as rhmc
from s2 import run_nm_comparators20 as rcomp
from s2 import screened_universal_head as suh

HEMIT_CUTS = tuple(int(c) for c in rhmc.HEMIT_CUTOFFS)  # (22, 30, 18) panCK, CD3, DAPI


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemit-train-pkl", required=True)
    ap.add_argument("--hemit-test-pkl", required=True)
    ap.add_argument("--upstream-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budgets", default="4,8,16,32,64")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(",")]

    print("[NV2-N6] Loading HEMIT banks...", flush=True)
    h_tr = pickle.load(open(args.hemit_train_pkl, "rb"))
    h_ev = pickle.load(open(args.hemit_test_pkl, "rb"))
    up_dir = Path(args.upstream_dir)
    n_tr_total = len(h_tr["he_nhwc"])

    backbones = [
        ("ConvNeXt", "sanofi_miphei_convnext_train_596.npz", "sanofi_miphei_convnext_test_945.npz"),
        ("W-MSE Swin", "sanofi_miphei_hemit_train_596.npz", "sanofi_miphei_hemit_test_945.npz"),
        ("Pix2Pix", "sanofi_pix2pix_train_596.npz", "sanofi_pix2pix_test_945.npz"),
        ("DGR", "native_dgr_train_596.npz", "native_dgr_test_945.npz"),
    ]
    available = [(b, trf, evf) for b, trf, evf in backbones if (up_dir / trf).exists() and (up_dir / evf).exists()]
    print(f"[NV2-N6] Found {len(available)} HEMIT backbones: {[x[0] for x in available]}", flush=True)

    results = dict(schema="nv2-n6-ultralowshot-v1", budgets=budgets, repeats=args.repeats, backbones={})

    for bname, trf, evf in available:
        tb = time.time()
        up_tr = np.load(up_dir / trf)["pred_u8"]
        up_ev = np.load(up_dir / evf)["pred_u8"]

        # Frozen baseline macro 3-ch F1 & SSIM
        frz_f1s, frz_ssims = [], []
        for ch, cut in enumerate(HEMIT_CUTS):
            ev = jce.batch_curves_and_quality_torch(up_ev[:, ch], h_ev["targ_nchw"][:, ch], cut, device=device, compute_quality=True)
            ep = jce.endpoints_from_pooled_curves(ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cut)
            frz_f1s.append(float(ep["Fixed_tol_F1"]))
            frz_ssims.append(float(ev["ssims"].mean()))
        b_out = dict(
            frozen_macro_f1=round(float(np.mean(frz_f1s)), 4),
            frozen_cd3_f1=round(float(frz_f1s[1]), 4),
            frozen_ssim=round(float(np.mean(frz_ssims)), 4),
            by_budget={},
        )

        # Precompute eval common12 for each channel once
        c12_ev_chs = [rhmc.build_common12_for_3ch(h_ev["he_nhwc"], up_ev, ch, device=device) for ch in range(3)]
        d_ev_chs = [torch.from_numpy(up_ev[:, ch]) for ch in range(3)]

        for N in budgets:
            rep_rows = {arm: [] for arm in ("S_recon", "QuantileMatch", "S_count")}
            for rep in range(args.repeats):
                rng = np.random.default_rng(20261121 + 1000 * N + 37 * rep)
                sub = np.sort(rng.choice(n_tr_total, size=N, replace=False))
                bs = min(8, N)
                # Ensure enough gradient steps at tiny N (e.g. N=4 -> 24 epochs so ~24 steps; N=64 -> 10 epochs)
                ep_n = 24 if N <= 8 else (16 if N <= 16 else (12 if N <= 32 else 10))

                arm_ch_f1 = {arm: [] for arm in rep_rows}
                arm_ch_ssim = {arm: [] for arm in rep_rows}
                arm_ch_negfp = {arm: [] for arm in rep_rows}

                for ch, cut in enumerate(HEMIT_CUTS):
                    c12_sub = rhmc.build_common12_for_3ch(h_tr["he_nhwc"][sub], up_tr[sub], ch, device=device)
                    d_sub = torch.from_numpy(up_tr[sub, ch])
                    y_sub = torch.from_numpy(h_tr["targ_nchw"][sub, ch])

                    # QuantileMatch
                    _, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr[sub, ch], h_tr["targ_nchw"][sub, ch], up_ev[:, ch])
                    arm_preds = {"QuantileMatch": qm_ev}

                    for arm in ("S_recon", "S_count"):
                        ps = []
                        for sd in (202610270, 202610271):
                            mdl, _ = suh.fit_universal_marker(
                                arm=arm, seed=sd, common_train=c12_sub, donor_u8_train=d_sub, target_u8_train=y_sub,
                                cutoff=cut, device=device, epochs=ep_n, batch_size=bs, lr=1.2e-3, deploy_anchor_weight=1.25,
                            )
                            p_u8, _ = suh.predict_universal_marker(mdl, c12_ev_chs[ch], d_ev_chs[ch], device=device, batch_size=24)
                            ps.append(p_u8.astype(np.float32))
                            del mdl
                        arm_preds[arm] = np.rint(np.clip(0.5 * (ps[0] + ps[1]), 0, 255)).astype(np.uint8)
                    del c12_sub

                    neg_t = (h_ev["targ_nchw"][:, ch] > cut).sum(axis=(1, 2)) == 0
                    for arm, pr in arm_preds.items():
                        ev = jce.batch_curves_and_quality_torch(pr, h_ev["targ_nchw"][:, ch], cut, device=device, compute_quality=True)
                        ep = jce.endpoints_from_pooled_curves(ev["K"].sum(0), ev["U"].sum(0), ev["V"].sum(0), ev["T"].sum(0), int(ev["G"].sum()), cut)
                        arm_ch_f1[arm].append(float(ep["Fixed_tol_F1"]))
                        arm_ch_ssim[arm].append(float(ev["ssims"].mean()))
                        arm_ch_negfp[arm].append(100.0 * float((pr[neg_t] > cut).mean()) if np.any(neg_t) else 0.0)

                for arm in rep_rows:
                    rep_rows[arm].append(dict(
                        macro_f1=float(np.mean(arm_ch_f1[arm])),
                        cd3_f1=float(arm_ch_f1[arm][1]),
                        macro_ssim=float(np.mean(arm_ch_ssim[arm])),
                        cd3_neg_fp_pct=float(arm_ch_negfp[arm][1]),
                    ))
                torch.cuda.empty_cache()

            b_out["by_budget"][str(N)] = {}
            for arm, rows in rep_rows.items():
                b_out["by_budget"][str(N)][arm] = dict(
                    macro_f1_mean=round(float(np.mean([r["macro_f1"] for r in rows])), 4),
                    macro_f1_sd=round(float(np.std([r["macro_f1"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0, 4),
                    cd3_f1_mean=round(float(np.mean([r["cd3_f1"] for r in rows])), 4),
                    macro_ssim_mean=round(float(np.mean([r["macro_ssim"] for r in rows])), 4),
                    cd3_neg_fp_pct_mean=round(float(np.mean([r["cd3_neg_fp_pct"] for r in rows])), 4),
                )
            sc_m = b_out["by_budget"][str(N)]["S_count"]
            qm_m = b_out["by_budget"][str(N)]["QuantileMatch"]
            print(
                f"[NV2-N6] {bname:10s} N={N:2d} | Frz={b_out['frozen_macro_f1']:.4f} "
                f"S_cnt={sc_m['macro_f1_mean']:.4f}+/-{sc_m['macro_f1_sd']:.4f} (negFP={sc_m['cd3_neg_fp_pct_mean']:.2f}%) "
                f"QM={qm_m['macro_f1_mean']:.4f} (negFP={qm_m['cd3_neg_fp_pct_mean']:.1f}%)",
                flush=True,
            )

        results["backbones"][bname] = b_out
        del c12_ev_chs, d_ev_chs, up_tr, up_ev
        gc.collect()
        torch.cuda.empty_cache()
        _atomic_write_json(out_dir / "nv2_n6_ultralowshot.json", results)

    results["wall_seconds"] = round(time.time() - t0, 1)
    _atomic_write_json(out_dir / "nv2_n6_ultralowshot.json", results)
    print("NV2_N6_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
