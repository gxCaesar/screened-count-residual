#!/usr/bin/env python3
"""PUSH-4: Multi-Organ Clinical Ki67 Proliferation Index & Cell-Mask Validation (DeepLIIF).

Validates Screened Count Residuals on a fourth completely independent multi-organ cohort
spanning THREE new organs (Lung, Bladder, and Breast carcinoma; DeepLIIF + BC-DeepLIIF,
Zenodo 4751737 / HuggingFace MedOtter/DeepLIIF):
  - Splits (official, strictly disjoint):
      * DeepLIIF Lung + Bladder: train=575, val=91, locked test=598 (Lung=473, Bladder=125)
      * BC-DeepLIIF Breast     : bc_train=385, held-out bc_val=66 (6 slides)
  - Two distinct physical inputs tested on the exact same 3-plex mIF targets (DAPI, Lap2, Ki67):
      1. modality="ihc"        : Brightfield Ki67 IHC -> 3-plex mIF (official DeepLIIF task)
      2. modality="hematoxylin": Pure Hematoxylin morphology (no DAB chromogen) -> 3-plex mIF
  - Three diverse upstream generator families trained strictly on train+val (zero test exposure):
      * deepliif_unet       : 7.76M Double-Convolution U-Net (MSE + 0.5 L1)
      * deepliif_resnet34   : 24.4M ResNet-34 Encoder U-Net (MSE + 0.5 L1)
      * deepliif_pix2pix    : U-Net + 70x70 PatchGAN cGAN (L1 + adversarial BCE)
  - Five post-hoc calibration arms fitted strictly on train split:
      upstream | S_recon (2-seed) | QuantileMatch (256-bin LUT) | E_count (2-seed) | S_count (2-seed)
  - Endpoints:
      * 3-channel pixel fixed-cutoff tolerant F1, pixel-exact r=0 F1, ODS-F, SSIM, PSNR, DC drift
      * Pathologist-verified cell instance masks (seg_mask: Red=Ki67+ nuclei, Blue=Ki67- nuclei):
          - Single-cell Ki67+ classification F1, precision, recall, AUROC
          - Clinical Ki67 Proliferation Index (% Ki67+ nuclei) MAE, Pearson r, Spearman rho
          - St. Gallen high-proliferation clinical threshold (Ki67 >= 14% and >= 20%) AUROC & F1
      * Organ-stratified metrics (Lung, Bladder, Breast) + per-tile CSV for source data.
"""
from __future__ import annotations

import argparse
import gc
import glob
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage, stats
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from s2 import journal_cohort_eval as jce
from s2 import run_hemit_multibackbone_cell as rhmc
from s2 import run_nm_comparators20 as rcomp
from s2 import run_plan2_ablations as rp2
from s2 import run_push1_exp5_exact_parity as rp1
from s2 import screened_universal_head as suh

DEEPLIIF_CHANNELS = ("DAPI", "Lap2", "Ki67")


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _decode_rgb(cell_dict: dict, size: int = 256) -> np.ndarray:
    im = Image.open(io.BytesIO(cell_dict["bytes"])).convert("RGB")
    if size and (im.width != size or im.height != size):
        im = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _extract_gt_nuclei_from_seg(seg_rgb: np.ndarray, min_area: int = 12) -> dict:
    """Parse DeepLIIF pathologist seg_mask RGB: Red=Ki67+ nuclei, Blue=Ki67- nuclei."""
    r = seg_rgb[:, :, 0].astype(np.int16)
    g = seg_rgb[:, :, 1].astype(np.int16)
    b = seg_rgb[:, :, 2].astype(np.int16)
    pos_mask = (r > 128) & (r > b + 40) & (r > g + 30)
    neg_mask = (b > 128) & (b > r + 40) & (b > g + 30)

    inst_map = np.zeros(seg_rgb.shape[:2], dtype=np.int32)
    labels = []
    cur_id = 0
    for bin_m, lbl_val in ((pos_mask, 1), (neg_mask, 0)):
        cc, n_cc = ndimage.label(bin_m)
        if n_cc == 0:
            continue
        areas = np.bincount(cc.ravel())
        for cid in range(1, n_cc + 1):
            if areas[cid] >= min_area:
                cur_id += 1
                inst_map[cc == cid] = cur_id
                labels.append(lbl_val)
    # Precompute compact CSR-like pixel indices per nucleus for ultra-fast intensity extraction
    if cur_id == 0:
        return dict(n_cells=0, labels=np.zeros((0,), dtype=np.int32), flat_idx=np.zeros((0,), dtype=np.int32), cell_ids=np.zeros((0,), dtype=np.int32))
    flat = inst_map.ravel()
    nz = np.nonzero(flat > 0)[0].astype(np.int32)
    cids = (flat[nz] - 1).astype(np.int32)
    counts = np.bincount(cids, minlength=cur_id).astype(np.float64)
    return dict(n_cells=cur_id, labels=np.asarray(labels, dtype=np.int32), flat_idx=nz, cell_ids=cids, counts=counts)


def _tile_cell_means(img_u8_hw: np.ndarray, rec: dict) -> np.ndarray:
    if rec["n_cells"] == 0:
        return np.zeros((0,), dtype=np.float64)
    vals = img_u8_hw.ravel()[rec["flat_idx"]].astype(np.float64)
    sums = np.bincount(rec["cell_ids"], weights=vals, minlength=rec["n_cells"])
    return sums / np.maximum(rec["counts"], 1.0)


def load_deepliif_split(data_dir: Path, split_prefix: str, modality: str, size: int = 256) -> dict:
    files = sorted(glob.glob(str(data_dir / f"{split_prefix}-*.parquet")))
    assert files, f"No parquet shards found for {split_prefix} in {data_dir}"
    in_imgs, targ_list, sids, organs, nuc_recs = [], [], [], [], []
    tissue_map = {0: "Breast", 1: "Bladder", 2: "Lung"}

    for fp in files:
        tbl = pq.read_table(fp, columns=["sample_id", "tissue", modality, "dapi", "lap2", "marker", "seg_mask"])
        n = tbl.num_rows
        for i in range(n):
            sid = str(tbl["sample_id"][i].as_py())
            tis_id = int(tbl["tissue"][i].as_py())
            organ = tissue_map.get(tis_id, f"tissue_{tis_id}")
            x_rgb = _decode_rgb(tbl[modality][i].as_py(), size=size)
            dp_rgb = _decode_rgb(tbl["dapi"][i].as_py(), size=size)
            lp_rgb = _decode_rgb(tbl["lap2"][i].as_py(), size=size)
            mk_rgb = _decode_rgb(tbl["marker"][i].as_py(), size=size)
            seg_rgb = _decode_rgb(tbl["seg_mask"][i].as_py(), size=size)

            # DeepLIIF channel encodings: DAPI grayscale (R=G=B), Lap2 green channel, Ki67 marker grayscale
            dapi_u8 = dp_rgb[:, :, 0]
            lap2_u8 = lp_rgb[:, :, 1]
            ki67_u8 = mk_rgb[:, :, 0]
            in_imgs.append(x_rgb)
            targ_list.append(np.stack([dapi_u8, lap2_u8, ki67_u8], axis=0))
            sids.append(sid)
            organs.append(organ)
            nuc_recs.append(_extract_gt_nuclei_from_seg(seg_rgb, min_area=max(6, int(12 * (size / 512.0) ** 2))))

    return dict(
        sample_ids=np.asarray(sids),
        organs=np.asarray(organs),
        in_nhwc=np.stack(in_imgs, axis=0),
        targ_nchw=np.stack(targ_list, axis=0),
        nuclei=nuc_recs,
    )


def compute_otsu_cutoffs(targ_nchw: np.ndarray) -> list[int]:
    cutoffs = []
    for ch in range(3):
        hist = np.bincount(targ_nchw[:, ch].ravel(), minlength=256).astype(np.float64)
        total = hist.sum()
        bins = np.arange(256, dtype=np.float64)
        w0 = np.cumsum(hist)
        w1 = total - w0
        m0 = np.cumsum(hist * bins)
        m_tot = m0[-1]
        best_t, best_v = 20, -1.0
        for t in range(4, 250):
            if w0[t] <= 0 or w1[t] <= 0:
                continue
            mu0 = m0[t] / w0[t]
            mu1 = (m_tot - m0[t]) / w1[t]
            v = w0[t] * w1[t] * (mu0 - mu1) ** 2
            if v > best_v:
                best_v = v
                best_t = t
        cutoffs.append(int(best_t))
    return cutoffs


class PatchDisc3(nn.Module):
    def __init__(self, in_ch: int = 6, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_upstream_3ch(arch: str, tr_bank: dict, va_bank: dict, device: torch.device, epochs: int = 24) -> tuple[nn.Module, dict]:
    t0 = time.monotonic()
    seed = 20261120 + (0 if arch == "deepliif_unet" else (17 if arch == "deepliif_resnet34" else 37))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if arch == "deepliif_resnet34":
        gen = rp1.ResNet34UNetVirtualStainer(3, 3).to(device)
    else:
        gen = rp1.build_backbone("double_conv_unet").to(device)
    is_gan = arch == "deepliif_pix2pix"
    n_params = int(sum(p.numel() for p in gen.parameters()))
    opt_g = torch.optim.AdamW(gen.parameters(), lr=1e-3 if not is_gan else 6e-4, weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=5e-5)
    disc = PatchDisc3(6, 64).to(device) if is_gan else None
    opt_d = torch.optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.5, 0.999)) if is_gan else None

    x_tr, y_tr = tr_bank["in_nhwc"], tr_bank["targ_nchw"]
    x_va, y_va = va_bank["in_nhwc"], va_bank["targ_nchw"]
    bs = 12 if arch == "deepliif_resnet34" else 16
    best_vl1, best_sd, best_ep = 1e9, None, 0

    for ep in range(epochs):
        gen.train()
        if disc is not None:
            disc.train()
        order = rng.permutation(len(x_tr))
        for s in range(0, len(x_tr), bs):
            idx = order[s : s + bs]
            xb = torch.from_numpy(x_tr[idx]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            yb = torch.from_numpy(y_tr[idx]).to(device, dtype=torch.float32) / 255.0
            k_rot = int(rng.integers(0, 4))
            if k_rot:
                xb, yb = torch.rot90(xb, k_rot, dims=(-2, -1)), torch.rot90(yb, k_rot, dims=(-2, -1))
            if int(rng.integers(0, 2)):
                xb, yb = torch.flip(xb, dims=(-1,)), torch.flip(yb, dims=(-1,))
            fake = gen(xb)
            if is_gan and disc is not None and opt_d is not None:
                opt_d.zero_grad(set_to_none=True)
                d_r = disc(torch.cat([xb, yb], dim=1))
                d_f = disc(torch.cat([xb, fake.detach()], dim=1))
                l_d = 0.5 * (
                    F.binary_cross_entropy_with_logits(d_r, torch.ones_like(d_r))
                    + F.binary_cross_entropy_with_logits(d_f, torch.zeros_like(d_f))
                )
                l_d.backward()
                opt_d.step()

                opt_g.zero_grad(set_to_none=True)
                g_adv = F.binary_cross_entropy_with_logits(disc(torch.cat([xb, fake], dim=1)), torch.ones_like(d_r))
                l_g = F.l1_loss(fake, yb) + 0.5 * F.mse_loss(fake, yb) + 0.015 * g_adv
                l_g.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
                opt_g.step()
            else:
                opt_g.zero_grad(set_to_none=True)
                l_g = F.mse_loss(fake, yb) + 0.5 * F.l1_loss(fake, yb)
                l_g.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
                opt_g.step()
        sched_g.step()
        if (ep + 1) % 3 == 0 or (ep + 1) == epochs:
            gen.eval()
            vsum, vcnt = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(x_va), 32):
                    xb = torch.from_numpy(x_va[s : s + 32]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                    yb = torch.from_numpy(y_va[s : s + 32]).to(device, dtype=torch.float32) / 255.0
                    vsum += float(F.l1_loss(gen(xb), yb, reduction="sum").item())
                    vcnt += int(yb.numel())
            vl1 = vsum / max(vcnt, 1)
            if vl1 < best_vl1:
                best_vl1, best_ep = vl1, ep + 1
                best_sd = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}
    if best_sd is not None:
        gen.load_state_dict(best_sd)
    if disc is not None:
        del disc, opt_d
    torch.cuda.empty_cache()
    return gen, dict(params=n_params, best_epoch=best_ep, best_val_l1=round(best_vl1, 6), train_sec=round(time.monotonic() - t0, 1))


@torch.no_grad()
def predict_3ch_u8(net: nn.Module, x_nhwc: np.ndarray, device: torch.device, batch_size: int = 32) -> np.ndarray:
    net.eval()
    n, h, w, _ = x_nhwc.shape
    out = np.empty((n, 3, h, w), dtype=np.uint8)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        xb = torch.from_numpy(x_nhwc[s:e]).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
        out[s:e] = torch.round(net(xb).clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    return out


def exact_r0_f1(pred_u8: np.ndarray, targ_u8: np.ndarray, cutoff: int) -> float:
    p = pred_u8 > int(cutoff)
    g = targ_u8 > int(cutoff)
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & (~g)))
    fn = int(np.sum((~p) & g))
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0


def evaluate_deepliif_clinical(
    pred_tr: np.ndarray,
    pred_ev: np.ndarray,
    tr_bank: dict,
    ev_bank: dict,
    cutoffs: list[int],
    cell_ki67_cutoff: float,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    """Evaluate 3-plex pixel metrics + pathologist-mask single-cell Ki67 + clinical Ki67 Proliferation Index."""
    per_ch = {}
    for ch, m_name in enumerate(DEEPLIIF_CHANNELS):
        cr = jce.evaluate_channel_predictions(
            train_pred_u8=pred_tr[:, ch],
            train_targ_u8=tr_bank["targ_nchw"][:, ch],
            eval_pred_u8=pred_ev[:, ch],
            eval_targ_u8=ev_bank["targ_nchw"][:, ch],
            eval_clusters=ev_bank["organs"],
            fixed_cutoff=int(cutoffs[ch]),
            device=device,
        )
        p_m = dict(cr["pooled"])
        p_m["Exact_r0_F1"] = exact_r0_f1(pred_ev[:, ch], ev_bank["targ_nchw"][:, ch], int(cutoffs[ch]))
        p_m["per_organ_fixed_f1"] = {org: float(v["Fixed_tol_F1"]) for org, v in cr.get("per_cluster", {}).items()}
        per_ch[m_name] = p_m

    macro = {
        mk: float(np.mean([per_ch[m][mk] for m in DEEPLIIF_CHANNELS]))
        for mk in ("Fixed_tol_F1", "ODS_F", "AP", "Cal_F1", "SSIM", "PSNR_dB", "Exact_r0_F1", "mismatch_gap_ods_minus_fixed")
    }

    # Single-cell Ki67 classification & clinical Ki67 Proliferation Index (% Ki67+ nuclei) using pathologist seg_mask!
    all_lbl, all_pred_score = [], []
    tile_gt_pi, tile_pr_pi, tile_organs = [], [], []
    tile_rows = []
    for i, rec in enumerate(ev_bank["nuclei"]):
        if rec["n_cells"] == 0:
            continue
        scores = _tile_cell_means(pred_ev[i, 2], rec)
        gt_lbl = rec["labels"]
        pr_lbl = (scores > cell_ki67_cutoff).astype(np.int32)
        all_lbl.append(gt_lbl)
        all_pred_score.append(scores)
        gt_pi = float(gt_lbl.mean()) * 100.0
        pr_pi = float(pr_lbl.mean()) * 100.0
        tile_gt_pi.append(gt_pi)
        tile_pr_pi.append(pr_pi)
        tile_organs.append(str(ev_bank["organs"][i]))
        tile_rows.append(dict(
            sample_id=str(ev_bank["sample_ids"][i]),
            organ=str(ev_bank["organs"][i]),
            n_cells=int(rec["n_cells"]),
            gt_ki67_pi_pct=round(gt_pi, 3),
            pred_ki67_pi_pct=round(pr_pi, 3),
        ))

    y_cell = np.concatenate(all_lbl) if all_lbl else np.zeros((0,), dtype=np.int32)
    s_cell = np.concatenate(all_pred_score) if all_pred_score else np.zeros((0,), dtype=np.float64)
    p_cell = (s_cell > cell_ki67_cutoff).astype(np.int32)

    cell_metrics = dict(
        n_total_nuclei=int(len(y_cell)),
        n_gt_ki67_pos=int(y_cell.sum()),
        ki67_cell_f1=float(f1_score(y_cell, p_cell, zero_division=0)),
        ki67_cell_precision=float(precision_score(y_cell, p_cell, zero_division=0)),
        ki67_cell_recall=float(recall_score(y_cell, p_cell, zero_division=0)),
        ki67_cell_auroc=float(roc_auc_score(y_cell, s_cell)) if len(np.unique(y_cell)) > 1 else float("nan"),
    )

    gt_arr = np.asarray(tile_gt_pi, dtype=np.float64)
    pr_arr = np.asarray(tile_pr_pi, dtype=np.float64)
    pi_r = float(stats.pearsonr(gt_arr, pr_arr)[0]) if len(gt_arr) >= 3 and np.std(pr_arr) > 1e-9 else 0.0
    pi_rho = float(stats.spearmanr(gt_arr, pr_arr)[0]) if len(gt_arr) >= 3 and np.std(pr_arr) > 1e-9 else 0.0
    pi_mae = float(np.mean(np.abs(pr_arr - gt_arr))) if len(gt_arr) else float("nan")

    # St. Gallen clinical high-proliferation thresholds (Ki67 >= 14% and >= 20%)
    stg = {}
    for thr in (14.0, 20.0):
        yg = (gt_arr >= thr).astype(np.int32)
        yp = (pr_arr >= thr).astype(np.int32)
        stg[f"ge_{int(thr)}pct"] = dict(
            f1=float(f1_score(yg, yp, zero_division=0)),
            auroc=float(roc_auc_score(yg, pr_arr)) if len(np.unique(yg)) > 1 else float("nan"),
        )

    # Organ-stratified Ki67 PI MAE and cell F1
    by_organ = {}
    org_arr = np.asarray(tile_organs)
    for org in sorted(set(tile_organs)):
        m = org_arr == org
        by_organ[org] = dict(
            n_tiles=int(m.sum()),
            ki67_pi_mae_pct=round(float(np.mean(np.abs(pr_arr[m] - gt_arr[m]))), 3),
            ki67_pi_pearson_r=round(float(stats.pearsonr(gt_arr[m], pr_arr[m])[0]) if m.sum() >= 3 and np.std(pr_arr[m]) > 1e-9 else 0.0, 4),
            ki67_fixed_f1=round(float(per_ch["Ki67"]["per_organ_fixed_f1"].get(org, np.nan)), 4),
        )

    return dict(
        macro_3ch=macro,
        per_channel=per_ch,
        single_cell_ki67=cell_metrics,
        clinical_ki67_index=dict(
            n_tiles_with_cells=int(len(gt_arr)),
            pi_mae_pct=round(pi_mae, 3),
            pi_pearson_r=round(pi_r, 4),
            pi_spearman_rho=round(pi_rho, 4),
            st_gallen=stg,
            by_organ=by_organ,
        ),
    ), tile_rows


def build_common12_deepliif(in_nhwc: np.ndarray, donor_nchw: np.ndarray, ch: int, device: torch.device) -> torch.Tensor:
    """Build (N, 12, H, W) float16 coordinate features using channel 0 (DAPI) as structural reference."""
    n, _, h, w = donor_nchw.shape
    out = torch.empty((n, 12, h, w), dtype=torch.float16)
    with torch.no_grad():
        for s in range(0, n, 64):
            e = min(n, s + 64)
            x_t = torch.from_numpy(in_nhwc[s:e]).to(device, non_blocking=True).permute(0, 3, 1, 2).to(torch.float32) / 255.0
            d_all = torch.from_numpy(donor_nchw[s:e]).to(device, non_blocking=True).to(torch.float32) / 255.0
            bc = d_all[:, ch : ch + 1]
            bs = d_all[:, 0:1]  # DAPI structural reference
            bm = d_all.mean(dim=1, keepdim=True)
            out[s:e] = suh.build_universal_common12(x_t, bc, bs, bm).to(torch.float16).cpu()
    return out


def run_cohort_suite(
    suite_name: str,
    tr_bank: dict,
    va_bank: dict,
    ev_bank: dict,
    archs: list[str],
    device: torch.device,
    out_dir: Path,
    seeds: tuple[int, ...] = (202610270, 202610271),
    head_epochs: int = 8,
) -> dict:
    cutoffs = compute_otsu_cutoffs(tr_bank["targ_nchw"])
    # Lock physical single-cell Ki67 cutoff strictly on training ground-truth mIF + training seg_mask!
    tr_gt_pos, tr_gt_neg = [], []
    for i, rec in enumerate(tr_bank["nuclei"]):
        if rec["n_cells"] == 0:
            continue
        sc = _tile_cell_means(tr_bank["targ_nchw"][i, 2], rec)
        tr_gt_pos.append(sc[rec["labels"] == 1])
        tr_gt_neg.append(sc[rec["labels"] == 0])
    pos_all = np.concatenate(tr_gt_pos) if tr_gt_pos else np.array([40.0])
    neg_all = np.concatenate(tr_gt_neg) if tr_gt_neg else np.array([10.0])
    # Optimal F1 threshold on physical training mIF nuclei (locked before test)
    y_tr_c = np.concatenate([np.ones(len(pos_all), dtype=np.int32), np.zeros(len(neg_all), dtype=np.int32)])
    s_tr_c = np.concatenate([pos_all, neg_all])
    best_cf, best_ct = -1.0, float(cutoffs[2])
    for t_cand in np.linspace(5.0, 120.0, 116):
        f_c = float(f1_score(y_tr_c, (s_tr_c > t_cand).astype(np.int32), zero_division=0))
        if f_c > best_cf:
            best_cf, best_ct = f_c, float(t_cand)
    cell_ki67_cutoff = best_ct

    print(f"\n[PUSH-4 {suite_name}] n_tr={len(tr_bank['in_nhwc'])} n_val={len(va_bank['in_nhwc'])} n_ev={len(ev_bank['in_nhwc'])} cutoffs={cutoffs} cell_ki67_tau={cell_ki67_cutoff:.1f} (gt_train_cell_F1={best_cf:.4f})", flush=True)

    suite_out = dict(
        suite=suite_name,
        n_train=int(len(tr_bank["in_nhwc"])),
        n_val=int(len(va_bank["in_nhwc"])),
        n_eval=int(len(ev_bank["in_nhwc"])),
        organs_eval=dict(pd.Series(ev_bank["organs"]).value_counts().to_dict()),
        cutoffs_u8=cutoffs,
        physical_cell_ki67_cutoff=round(cell_ki67_cutoff, 2),
        physical_train_cell_ki67_f1=round(best_cf, 4),
        backbones={},
    )
    all_tile_rows = []

    for arch in archs:
        gen, gen_info = train_upstream_3ch(arch, tr_bank, va_bank, device, epochs=24)
        up_tr = predict_3ch_u8(gen, tr_bank["in_nhwc"], device)
        up_ev = predict_3ch_u8(gen, ev_bank["in_nhwc"], device)
        del gen
        torch.cuda.empty_cache()

        arms = ("upstream", "S_recon", "QuantileMatch", "E_count", "S_count")
        arm_tr = {a: (up_tr if a == "upstream" else np.empty_like(up_tr)) for a in arms}
        arm_ev = {a: (up_ev if a == "upstream" else np.empty_like(up_ev)) for a in arms}
        dc_drifts = {"E_count": [], "S_count": []}

        for ch, m_name in enumerate(DEEPLIIF_CHANNELS):
            cutoff = int(cutoffs[ch])
            qm_tr, qm_ev, _ = rcomp.apply_quantile_match_u8(up_tr[:, ch], tr_bank["targ_nchw"][:, ch], up_ev[:, ch])
            arm_tr["QuantileMatch"][:, ch] = qm_tr
            arm_ev["QuantileMatch"][:, ch] = qm_ev

            c12_tr = build_common12_deepliif(tr_bank["in_nhwc"], up_tr, ch, device=device)
            c12_ev = build_common12_deepliif(ev_bank["in_nhwc"], up_ev, ch, device=device)
            d_tr_t = torch.from_numpy(up_tr[:, ch])
            y_tr_t = torch.from_numpy(tr_bank["targ_nchw"][:, ch])
            d_ev_t = torch.from_numpy(up_ev[:, ch])

            for arm in ("S_recon", "E_count", "S_count"):
                tr_l, ev_l, dc_l = [], [], []
                for sd in seeds:
                    if arm == "E_count":
                        mdl, _ = rp2.fit_ablation_marker(
                            mode="E_count", seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                            target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                            epochs=head_epochs, batch_size=8, lr=1.5e-3,
                        )
                        pt, _ = rp2.predict_ablation_marker(mdl, "E_count", c12_tr, d_tr_t, device=device, batch_size=32)
                        pe, de = rp2.predict_ablation_marker(mdl, "E_count", c12_ev, d_ev_t, device=device, batch_size=32)
                    else:
                        mdl, _ = suh.fit_universal_marker(
                            arm=arm, seed=sd, common_train=c12_tr, donor_u8_train=d_tr_t,
                            target_u8_train=y_tr_t, cutoff=cutoff, device=device,
                            epochs=head_epochs, batch_size=8, lr=1.5e-3, deploy_anchor_weight=1.25,
                        )
                        pt, _ = suh.predict_universal_marker(mdl, c12_tr, d_tr_t, device=device, batch_size=32)
                        pe, de = suh.predict_universal_marker(mdl, c12_ev, d_ev_t, device=device, batch_size=32)
                    tr_l.append(pt.astype(np.float32))
                    ev_l.append(pe.astype(np.float32))
                    dc_l.append(float(de["residual_mean_max"]))
                    del mdl
                arm_tr[arm][:, ch] = np.round(np.mean(tr_l, axis=0)).astype(np.uint8)
                arm_ev[arm][:, ch] = np.round(np.mean(ev_l, axis=0)).astype(np.uint8)
                if arm in dc_drifts:
                    dc_drifts[arm].append(max(dc_l))
            del c12_tr, c12_ev
            gc.collect()
            torch.cuda.empty_cache()

        arch_arms = {}
        for arm in arms:
            ev_res, t_rows = evaluate_deepliif_clinical(
                arm_tr[arm], arm_ev[arm], tr_bank, ev_bank, cutoffs, cell_ki67_cutoff, device
            )
            ev_res["max_dc_drift"] = max(dc_drifts[arm]) if arm in dc_drifts and dc_drifts[arm] else 0.0
            arch_arms[arm] = ev_res
            for r in t_rows:
                all_tile_rows.append(dict(suite=suite_name, backbone=arch, arm=arm, **r))

        print(
            f"  [{suite_name} | {arch}] F1: up={arch_arms['upstream']['macro_3ch']['Fixed_tol_F1']:.4f} "
            f"sr={arch_arms['S_recon']['macro_3ch']['Fixed_tol_F1']:.4f} "
            f"qm={arch_arms['QuantileMatch']['macro_3ch']['Fixed_tol_F1']:.4f} "
            f"ec={arch_arms['E_count']['macro_3ch']['Fixed_tol_F1']:.4f} "
            f"sc={arch_arms['S_count']['macro_3ch']['Fixed_tol_F1']:.4f} | "
            f"Ki67_cell_F1: up={arch_arms['upstream']['single_cell_ki67']['ki67_cell_f1']:.4f} "
            f"qm={arch_arms['QuantileMatch']['single_cell_ki67']['ki67_cell_f1']:.4f} "
            f"sc={arch_arms['S_count']['single_cell_ki67']['ki67_cell_f1']:.4f} | "
            f"PI_MAE: up={arch_arms['upstream']['clinical_ki67_index']['pi_mae_pct']:.2f}% "
            f"sc={arch_arms['S_count']['clinical_ki67_index']['pi_mae_pct']:.2f}%",
            flush=True,
        )
        suite_out["backbones"][arch] = dict(upstream_info=gen_info, arms=arch_arms)

    if all_tile_rows:
        pd.DataFrame(all_tile_rows).to_csv(out_dir / f"push4_deepliif_tiles_{suite_name}.csv", index=False)
    return suite_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(args.device)
    pdir = Path(args.parquet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "push4_deepliif_multiorgan.json"

    archs = ["deepliif_unet", "deepliif_resnet34", "deepliif_pix2pix"]
    results = dict(schema="push4-deepliif-multiorgan-v1", img_size=args.size, suites={})

    # Suite 1: DeepLIIF Lung + Bladder (official split: train=575, val=91, test=598) — Brightfield IHC -> 3-plex mIF
    tr_ihc = load_deepliif_split(pdir, "train", "ihc", size=args.size)
    va_ihc = load_deepliif_split(pdir, "validation", "ihc", size=args.size)
    te_ihc = load_deepliif_split(pdir, "test", "ihc", size=args.size)
    results["suites"]["lung_bladder_ihc_to_mif"] = run_cohort_suite("lung_bladder_ihc_to_mif", tr_ihc, va_ihc, te_ihc, archs, device, out_dir)
    _atomic_write_json(out_path, results)
    del tr_ihc, va_ihc, te_ihc
    gc.collect()

    # Suite 2: DeepLIIF Lung + Bladder — Pure Hematoxylin morphology (ZERO DAB chromogen) -> 3-plex mIF
    tr_hem = load_deepliif_split(pdir, "train", "hematoxylin", size=args.size)
    va_hem = load_deepliif_split(pdir, "validation", "hematoxylin", size=args.size)
    te_hem = load_deepliif_split(pdir, "test", "hematoxylin", size=args.size)
    results["suites"]["lung_bladder_hem_to_mif"] = run_cohort_suite("lung_bladder_hem_to_mif", tr_hem, va_hem, te_hem, archs, device, out_dir)
    _atomic_write_json(out_path, results)
    del tr_hem, va_hem, te_hem
    gc.collect()

    # Suite 3: BC-DeepLIIF Breast Cancer (bc_train=385 split 320/65 train/val, held-out bc_validation=66 across 6 slides)
    bc_tr_all = load_deepliif_split(pdir, "bc_train", "ihc", size=args.size)
    bc_ev = load_deepliif_split(pdir, "bc_validation", "ihc", size=args.size)
    # Slide-disjoint inner split of bc_train (prefix before _ is slide ID)
    bc_slides = sorted(set(str(s).split("_")[0] for s in bc_tr_all["sample_ids"]))
    n_va_sl = max(1, len(bc_slides) // 6)
    va_sl_set = set(bc_slides[-n_va_sl:])
    tr_m = np.array([str(s).split("_")[0] not in va_sl_set for s in bc_tr_all["sample_ids"]])
    va_m = ~tr_m
    bc_tr = {k: (v[tr_m] if isinstance(v, np.ndarray) else [v[i] for i in np.nonzero(tr_m)[0]]) for k, v in bc_tr_all.items()}
    bc_va = {k: (v[va_m] if isinstance(v, np.ndarray) else [v[i] for i in np.nonzero(va_m)[0]]) for k, v in bc_tr_all.items()}
    results["suites"]["breast_bc_deepliif_ihc_to_mif"] = run_cohort_suite("breast_bc_deepliif_ihc_to_mif", bc_tr, bc_va, bc_ev, archs, device, out_dir)
    _atomic_write_json(out_path, results)

    print("PUSH4_DEEPLIIF_MULTIORGAN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
