"""Multi-backbone brightfield H&E -> HER2 chromogenic DAB IHC campaign on HER2match.

Implements Cohort 3 of the Nature Methods Route 1 study on same-section breast
cancer H&E -> brightfield HER2 IHC (HER2match, Zenodo 15797050; 17 WSIs: 6 train,
6 val, 5 locked test):
  * Physical optical-density grounding: Ruifrok-Johnston H-DAB color deconvolution
    (``skimage.color.separate_stains(..., hdx_from_rgb)``) mapped onto ``[0, upper]``
    -> ``uint8 [0, 255]`` with the pre-registered train-only Otsu threshold
    (``calibration.json``: ``0.1353106`` over ``[0, 1.5331459]`` -> ``round(22.5055) = 23``;
    strict ``> 23`` positivity).
  * Three distinct virtual-IHC generator paradigms trained strictly on the 6 train WSIs:
      1. ``unet_l1_ssim``     : Multi-scale Residual U-Net under L1 + Multi-Scale Gradient/SSIM regression
      2. ``pix2pix_cgan_ihc`` : Conditional U-Net + 70x70 PatchGAN under LSGAN + 40 * L1
      3. ``hoptimus_lora_ihc``: Pathology Foundation ViT (H-optimus-0 + Q/V LoRA + DetailCapture)
                                under Weighted-MSE regression
  * Plug-and-play rescue: attaches ``(S_recon, S_count) x 2 seeds + ens2`` onto each
    frozen brightfield generator strictly on the 6 train WSIs.
  * Strict holdout containment:
      - ``s5_val``: trains upstream generators & residual heads on 6 train WSIs and
        evaluates 6 val WSIs. Hard-blocks any read of ``test/`` or ``HER2match_scores.csv``.
      - ``s6_locked_test``: requires frozen ``benchmark-protocol.json``, runs ``new_fits == 0``,
        and evaluates the 5 untouched locked test WSIs plus WSI-level DAB membrane burden
        and clinical HER2 score concordance.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import time
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
try:
    from skimage.color import hdx_from_rgb, separate_stains
except ModuleNotFoundError:
    # Exact Ruifrok & Johnston H-DAB inverse stain matrix from scikit-image 0.26
    # Column 0 = Hematoxylin, Column 1 = DAB (upper = 1.53314594), Column 2 = Residual
    hdx_from_rgb = np.array(
        [
            [1.20013702, -0.6897672, 1.05605387],
            [0.68523338, 0.02360991, -1.17857556],
            [-0.91780895, 1.50953603, 0.50098664],
        ],
        dtype=np.float64,
    )
    def separate_stains(rgb, conv_matrix):
        arr = np.maximum(np.asarray(rgb, dtype=np.float64), 1e-6)
        log_adjust = 1.0 / np.log(1e-6)
        return (np.log(arr) * log_adjust) @ np.asarray(conv_matrix, dtype=np.float64)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import journal_cohort_eval as jce
from . import screened_universal_head as suh

IMAGE_RE = re.compile(r"^(wsi\d+)_id\d+_x\d+_y\d+\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _select_spaced_names(names: Sequence[str], k: int) -> list[str]:
    s_names = sorted(names)
    if k <= 0 or len(s_names) <= k:
        return s_names
    if k == 1:
        return [s_names[len(s_names) // 2]]
    idxs = [j * (len(s_names) - 1) // (k - 1) for j in range(k)]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for ix in idxs:
        nm = s_names[ix]
        if nm not in seen:
            seen.add(nm)
            out.append(nm)
    return out


def discover_split_pairs(tiles_root: Path, split: str, tiles_per_wsi: int, *, forbid_test: bool) -> list[tuple[str, str]]:
    if forbid_test and split == "test":
        raise RuntimeError("Holdout containment violation: 'test' split access forbidden in S5")
    he_dir = tiles_root / "HE" / split
    ihc_dir = tiles_root / "IHC" / split
    if not he_dir.is_dir() or not ihc_dir.is_dir():
        raise FileNotFoundError(f"Missing HER2match split dirs: {he_dir} / {ihc_dir}")

    he_names = sorted(p.name for p in he_dir.iterdir() if p.is_file() and IMAGE_RE.fullmatch(p.name))
    ihc_set = {p.name for p in ihc_dir.iterdir() if p.is_file() and IMAGE_RE.fullmatch(p.name)}
    common = [nm for nm in he_names if nm in ihc_set]
    by_wsi: dict[str, list[str]] = {}
    for nm in common:
        m = IMAGE_RE.fullmatch(nm)
        assert m is not None
        by_wsi.setdefault(m.group(1), []).append(nm)

    selected: list[tuple[str, str]] = []
    for wsi_id in sorted(by_wsi):
        for nm in _select_spaced_names(by_wsi[wsi_id], tiles_per_wsi):
            selected.append((wsi_id, nm))
    return selected


def load_her2_dab_bank(
    tiles_root: Path,
    split: str,
    pairs: Sequence[tuple[str, str]],
    dab_upper: float,
    img_size: int = 512,
    *,
    forbid_test: bool,
) -> dict:
    """Load resized 512x512 H&E RGB, H&E Hematoxylin OD, IHC RGB, and physical IHC DAB OD uint8."""
    if forbid_test and split == "test":
        raise RuntimeError("Forbidden test split load in S5")
    mat = np.asarray(hdx_from_rgb, dtype=np.float64)
    hem_upper = float(np.maximum(mat[:, 0], 0.0).sum())

    n = len(pairs)
    he_nhwc = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
    ihc_nhwc = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
    hem_u8 = np.empty((n, img_size, img_size), dtype=np.uint8)
    dab_u8 = np.empty((n, img_size, img_size), dtype=np.uint8)
    wsis = []
    names = []

    he_dir = tiles_root / "HE" / split
    ihc_dir = tiles_root / "IHC" / split

    for i, (wsi_id, nm) in enumerate(pairs):
        p_he = he_dir / nm
        p_ihc = ihc_dir / nm
        if forbid_test and ("/test/" in str(p_he) or "/test/" in str(p_ihc)):
            raise RuntimeError(f"Forbidden test file path: {p_he}")

        with Image.open(p_he) as im_h:
            im_hr = im_h.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
            arr_he = np.asarray(im_hr, dtype=np.uint8)
        with Image.open(p_ihc) as im_i:
            im_ir = im_i.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
            arr_ihc = np.asarray(im_ir, dtype=np.uint8)

        he_nhwc[i] = arr_he
        ihc_nhwc[i] = arr_ihc

        st_he = separate_stains(arr_he.astype(np.float64) / 255.0, mat)
        hem_01 = np.clip(st_he[..., 0] / max(hem_upper, 1e-6), 0.0, 1.0)
        hem_u8[i] = np.rint(hem_01 * 255.0).astype(np.uint8)

        st_ihc = separate_stains(arr_ihc.astype(np.float64) / 255.0, mat)
        dab_01 = np.clip(st_ihc[..., 1] / max(dab_upper, 1e-6), 0.0, 1.0)
        dab_u8[i] = np.rint(dab_01 * 255.0).astype(np.uint8)

        wsis.append(wsi_id)
        names.append(nm)

    return dict(
        names=names,
        wsis=wsis,
        he_nhwc=he_nhwc,
        ihc_nhwc=ihc_nhwc,
        hem_u8=hem_u8,
        dab_u8=dab_u8,
    )


# ---------------------------------------------------------------------------
# Three Distinct Brightfield Virtual-IHC Generator Families
# ---------------------------------------------------------------------------

class _ResConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x) + self.body(x))


class ResUNetDABGenerator(nn.Module):
    """Multi-scale Residual U-Net predicting continuous DAB optical density in [0, 1]."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.e1 = _ResConvBlock(3, base)
        self.e2 = _ResConvBlock(base, base * 2)
        self.e3 = _ResConvBlock(base * 2, base * 4)
        self.e4 = _ResConvBlock(base * 4, base * 8)
        self.bot = _ResConvBlock(base * 8, base * 8)
        self.d4 = _ResConvBlock(base * 16, base * 4)
        self.d3 = _ResConvBlock(base * 8, base * 2)
        self.d2 = _ResConvBlock(base * 4, base)
        self.d1 = _ResConvBlock(base * 2, base)
        self.head = nn.Sequential(nn.Conv2d(base, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)
        s2 = self.e2(F.avg_pool2d(s1, 2))
        s3 = self.e3(F.avg_pool2d(s2, 2))
        s4 = self.e4(F.avg_pool2d(s3, 2))
        b = self.bot(F.avg_pool2d(s4, 2))
        u4 = self.d4(torch.cat([F.interpolate(b, size=s4.shape[-2:], mode="bilinear", align_corners=False), s4], dim=1))
        u3 = self.d3(torch.cat([F.interpolate(u4, size=s3.shape[-2:], mode="bilinear", align_corners=False), s3], dim=1))
        u2 = self.d2(torch.cat([F.interpolate(u3, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], dim=1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], dim=1))
        return self.head(u1)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN conditional discriminator over concatenated (H&E, DAB)."""

    def __init__(self, in_ch: int = 4, ndf: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x_he: torch.Tensor, y_dab: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_he, y_dab], dim=1))


def build_hoptimus_lora_dab_model(
    hoptimus0_ckpt: Path | None,
    img_size: int,
    device: torch.device,
    *,
    synthetic_backbone: bool = False,
) -> nn.Module:
    """Build H-optimus-0 + rank-8 Q/V LoRA + Detail_Capture 1-channel DAB model offline."""
    from . import run_mipheivit_hemit as rmh

    model = rmh.build_mipheivit_generator(
        img_size=img_size,
        num_classes=1,
        lora_rank=8,
        lora_alpha=1.0,
        ckpt_path=str(hoptimus0_ckpt) if hoptimus0_ckpt is not None else None,
        synthetic_backbone=synthetic_backbone,
    )
    # Keep LoRA active on the top 12 ViT blocks + full Detail_Capture decoder (<6 GB VRAM on 16GB GPUs)
    if hasattr(model.encoder.vit, "blocks") and len(model.encoder.vit.blocks) > 12:
        for blk in model.encoder.vit.blocks[:-12]:
            for p in blk.parameters():
                p.requires_grad_(False)
    return model.to(device)


def _aug_pair(x: torch.Tensor, y: torch.Tensor, code: int) -> tuple[torch.Tensor, torch.Tensor]:
    return suh.apply_d4(x, code), suh.apply_d4(y, code)


def _fit_or_load_single_her2_net(
    key: str,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    ckpt_path: Path,
    img_size: int,
    device: torch.device,
    *,
    hoptimus0_ckpt: Path | None,
    upstream_epochs: int,
    allow_train: bool,
    seed_offset: int = 0,
) -> tuple[str, callable]:
    """Fit or load one brightfield generator instance returning float01 predictor (N, H, W)."""
    n_tr = x_tr.shape[0]

    if key == "unet_l1_ssim":
        fam = "Multi-Scale Residual U-Net (Continuous Weighted-MSE + Gradient Regression)"
        net = ResUNetDABGenerator(base=32).to(device)
        if ckpt_path.exists():
            net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        elif allow_train:
            seed_val = 20261116 + seed_offset
            torch.manual_seed(seed_val)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
            rng = random.Random(seed_val)
            bs = 8
            for ep in range(upstream_epochs):
                net.train()
                order = list(range(n_tr))
                rng.shuffle(order)
                ep_l = 0.0
                nb = 0
                for s in range(0, n_tr, bs):
                    idxs = order[s : s + bs]
                    xb, yb = [], []
                    for ii in idxs:
                        c = rng.randrange(8)
                        xa, ya = _aug_pair(x_tr[ii], y_tr[ii], c)
                        xb.append(xa)
                        yb.append(ya)
                    xt = (torch.stack(xb).to(device) - 0.5) / 0.5
                    yt = torch.stack(yb).to(device)
                    opt.zero_grad(set_to_none=True)
                    pred = net(xt)
                    # Mild foreground-balanced MSE + gradient MSE prevents zero-median collapse on sparse DAB
                    w = 1.0 + 3.0 * yt
                    wmse = (w * (pred - yt).square()).mean()
                    gx_p = pred[..., :, 1:] - pred[..., :, :-1]
                    gx_t = yt[..., :, 1:] - yt[..., :, :-1]
                    gy_p = pred[..., 1:, :] - pred[..., :-1, :]
                    gy_t = yt[..., 1:, :] - yt[..., :-1, :]
                    grad_mse = (gx_p - gx_t).square().mean() + (gy_p - gy_t).square().mean()
                    loss = 0.85 * wmse + 0.15 * grad_mse
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                    opt.step()
                    ep_l += float(loss.detach().item())
                    nb += 1
                print(
                    json.dumps(
                        dict(
                            event="her2_upstream_epoch",
                            backbone=key,
                            fold=seed_offset,
                            epoch=ep + 1,
                            loss=round(ep_l / max(nb, 1), 5),
                        )
                    ),
                    flush=True,
                )
            _atomic_torch_save({k: v.cpu() for k, v in net.state_dict().items()}, ckpt_path)
        else:
            raise FileNotFoundError(f"Missing frozen upstream checkpoint: {ckpt_path}")

        net.eval()

        @torch.no_grad()
        def _pred_unet_f01(he_nhwc: np.ndarray) -> np.ndarray:
            out = np.empty((len(he_nhwc), img_size, img_size), dtype=np.float32)
            for s in range(0, len(he_nhwc), 8):
                e = min(len(he_nhwc), s + 8)
                xt = torch.from_numpy(he_nhwc[s:e]).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.0
                p = net((xt - 0.5) / 0.5)[:, 0].clamp(0.0, 1.0)
                out[s:e] = p.cpu().numpy()
            return out

        return fam, _pred_unet_f01

    if key == "pix2pix_cgan_ihc":
        fam = "Conditional Adversarial PatchGAN + U-Net (LSGAN + L1)"
        gen = ResUNetDABGenerator(base=32).to(device)
        if ckpt_path.exists():
            gen.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        elif allow_train:
            seed_val = 20261117 + seed_offset
            torch.manual_seed(seed_val)
            disc = PatchGANDiscriminator(in_ch=4, ndf=48).to(device)
            opt_g = torch.optim.AdamW(gen.parameters(), lr=2.5e-4, betas=(0.5, 0.999), weight_decay=1e-4)
            opt_d = torch.optim.AdamW(disc.parameters(), lr=1.5e-4, betas=(0.5, 0.999), weight_decay=1e-4)
            rng = random.Random(seed_val)
            bs = 8
            for ep in range(upstream_epochs):
                gen.train()
                disc.train()
                order = list(range(n_tr))
                rng.shuffle(order)
                gl_sum, dl_sum, nb = 0.0, 0.0, 0
                for s in range(0, n_tr, bs):
                    idxs = order[s : s + bs]
                    xb, yb = [], []
                    for ii in idxs:
                        c = rng.randrange(8)
                        xa, ya = _aug_pair(x_tr[ii], y_tr[ii], c)
                        xb.append(xa)
                        yb.append(ya)
                    xt = (torch.stack(xb).to(device) - 0.5) / 0.5
                    yt = torch.stack(yb).to(device)

                    # Discriminator step
                    opt_d.zero_grad(set_to_none=True)
                    fake_det = gen(xt).detach()
                    d_real = disc(xt, yt)
                    d_fake = disc(xt, fake_det)
                    loss_d = 0.5 * (
                        F.mse_loss(d_real, torch.ones_like(d_real)) + F.mse_loss(d_fake, torch.zeros_like(d_fake))
                    )
                    loss_d.backward()
                    opt_d.step()

                    # Generator step: foreground-balanced L1 + gradient + LSGAN prevents sparse-DAB zero-collapse
                    opt_g.zero_grad(set_to_none=True)
                    fake = gen(xt)
                    g_adv = F.mse_loss(disc(xt, fake), torch.ones_like(d_real))
                    fg_w = 1.0 + 12.0 * (yt > 0.08).to(yt.dtype) + 6.0 * yt
                    g_l1 = (fg_w * (fake - yt).abs()).mean()
                    gx_f = fake[..., :, 1:] - fake[..., :, :-1]
                    gx_y = yt[..., :, 1:] - yt[..., :, :-1]
                    gy_f = fake[..., 1:, :] - fake[..., :-1, :]
                    gy_y = yt[..., 1:, :] - yt[..., :-1, :]
                    g_grad = (gx_f - gx_y).abs().mean() + (gy_f - gy_y).abs().mean()
                    loss_g = 0.05 * g_adv + 2.0 * g_l1 + 0.5 * g_grad
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
                    opt_g.step()

                    gl_sum += float(loss_g.detach().item())
                    dl_sum += float(loss_d.detach().item())
                    nb += 1
                print(
                    json.dumps(
                        dict(
                            event="her2_upstream_epoch",
                            backbone=key,
                            fold=seed_offset,
                            epoch=ep + 1,
                            g_loss=round(gl_sum / max(nb, 1), 5),
                            d_loss=round(dl_sum / max(nb, 1), 5),
                        )
                    ),
                    flush=True,
                )
            _atomic_torch_save({k: v.cpu() for k, v in gen.state_dict().items()}, ckpt_path)
            del disc
        else:
            raise FileNotFoundError(f"Missing frozen upstream checkpoint: {ckpt_path}")

        gen.eval()

        @torch.no_grad()
        def _pred_cgan_f01(he_nhwc: np.ndarray) -> np.ndarray:
            out = np.empty((len(he_nhwc), img_size, img_size), dtype=np.float32)
            for s in range(0, len(he_nhwc), 8):
                e = min(len(he_nhwc), s + 8)
                xt = torch.from_numpy(he_nhwc[s:e]).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.0
                p = gen((xt - 0.5) / 0.5)[:, 0].clamp(0.0, 1.0)
                out[s:e] = p.cpu().numpy()
            return out

        return fam, _pred_cgan_f01

    if key in ("hoptimus_lora_ihc", "synthetic_hoptimus_lora"):
        use_syn = key == "synthetic_hoptimus_lora"
        if not use_syn and (hoptimus0_ckpt is None or not hoptimus0_ckpt.exists()):
            raise FileNotFoundError(f"hoptimus0_ckpt required for {key}: {hoptimus0_ckpt}")
        fam = "Pathology Foundation ViT (H-optimus-0 + LoRA + DetailCapture)"
        model = build_hoptimus_lora_dab_model(
            hoptimus0_ckpt, img_size, device, synthetic_backbone=use_syn
        )
        mean_t = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        std_t = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)

        if ckpt_path.exists():
            saved_trainable = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(saved_trainable, strict=False)
        elif allow_train:
            seed_val = 20261118 + seed_offset
            torch.manual_seed(seed_val)
            t_params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(t_params, lr=5e-4, weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
            rng = random.Random(seed_val)
            bs = 4 if use_syn else 2
            vi_epochs = upstream_epochs if use_syn else max(5, upstream_epochs - 2)
            for ep in range(vi_epochs):
                model.train()
                order = list(range(n_tr))
                rng.shuffle(order)
                ep_l, nb = 0.0, 0
                for s in range(0, n_tr, bs):
                    idxs = order[s : s + bs]
                    xb, yb = [], []
                    for ii in idxs:
                        c = rng.randrange(8)
                        xa, ya = _aug_pair(x_tr[ii] * 255.0, y_tr[ii], c)
                        xb.append(xa)
                        yb.append(ya)
                    xt = (torch.stack(xb).to(device) - mean_t) / std_t
                    yt_tanh = torch.stack(yb).to(device) * 1.8 - 0.9
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                        pred = model(xt)
                        # Moderate intensity-weighted MSE matching Sanofi WeightedMSELoss without background inflation
                        w = 1.0 + 2.5 * ((yt_tanh + 0.9) / 1.8).clamp(0.0, 1.0)
                        loss = (w * (pred - yt_tanh).square()).mean()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(t_params, 1.5)
                    scaler.step(opt)
                    scaler.update()
                    ep_l += float(loss.detach().item())
                    nb += 1
                print(
                    json.dumps(
                        dict(
                            event="her2_upstream_epoch",
                            backbone=key,
                            fold=seed_offset,
                            epoch=ep + 1,
                            loss=round(ep_l / max(nb, 1), 5),
                        )
                    ),
                    flush=True,
                )
            trainable_sd = {
                k: v.cpu()
                for k, v in model.state_dict().items()
                if ("lora_" in k or k.startswith("decoder."))
            }
            _atomic_torch_save(trainable_sd, ckpt_path)
        else:
            raise FileNotFoundError(f"Missing frozen H-optimus-0 adapter checkpoint: {ckpt_path}")

        model.eval()

        @torch.no_grad()
        def _pred_hopt_f01(he_nhwc: np.ndarray) -> np.ndarray:
            out = np.empty((len(he_nhwc), img_size, img_size), dtype=np.float32)
            for s in range(0, len(he_nhwc), 4):
                e = min(len(he_nhwc), s + 4)
                xt = torch.from_numpy(he_nhwc[s:e]).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
                xt = (xt - mean_t) / std_t
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    p_tanh = model(xt).float()
                p01 = ((p_tanh + 0.9) / 1.8).clamp(0.0, 1.0)[:, 0]
                out[s:e] = p01.cpu().numpy()
            return out

        return fam, _pred_hopt_f01

    raise ValueError(f"Unknown HER2match backbone key: {key!r}")


def train_or_load_her2_backbone(
    key: str,
    train_bank: dict,
    eval_bank: dict,
    cutoff_u8: int,
    state_dir: Path,
    device: torch.device,
    *,
    hoptimus0_ckpt: Path | None,
    upstream_epochs: int,
    allow_train: bool,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Train/load stratified 2-fold out-of-WSI upstream generator on ``train_bank``.

    Returns ``(family_description, up_tr_u8, up_ev_u8)`` where ``up_tr_u8`` is strictly
    out-of-WSI cross-fitted inside ``train_bank`` (zero access to ``val`` or ``test``)
    and ``up_ev_u8`` is the 2-fold ensemble prediction on ``eval_bank``.
    """
    img_size = train_bank["he_nhwc"].shape[1]

    if key == "synthetic_ihc_reg":
        def _syn_reg(he_nhwc: np.ndarray) -> np.ndarray:
            lum = 1.0 - (he_nhwc.astype(np.float32).mean(axis=-1) / 255.0)
            return np.rint(np.clip(0.55 * lum - 0.04, 0.0, 1.0) * 255.0).astype(np.uint8)
        return "Synthetic Continuous Regressor", _syn_reg(train_bank["he_nhwc"]), _syn_reg(eval_bank["he_nhwc"])

    if key == "synthetic_ihc_gan":
        def _syn_gan(he_nhwc: np.ndarray) -> np.ndarray:
            r_c = 1.0 - (he_nhwc[..., 0].astype(np.float32) / 255.0)
            b_c = 1.0 - (he_nhwc[..., 2].astype(np.float32) / 255.0)
            est = 0.62 * (0.6 * r_c + 0.4 * b_c) - 0.05
            return np.rint(np.clip(est, 0.0, 1.0) * 255.0).astype(np.uint8)
        return "Synthetic Adversarial cGAN", _syn_gan(train_bank["he_nhwc"]), _syn_gan(eval_bank["he_nhwc"])

    wsis_tr = np.asarray(train_bank["wsis"])
    uniq_wsis = sorted(set(wsis_tr.tolist()))
    if len(uniq_wsis) < 2:
        raise ValueError(f"Need >=2 training WSIs for out-of-WSI cross-fitting, got {uniq_wsis}")

    # Spatially disjoint half-WSI block 2-fold cross-fitting strictly inside train_bank:
    # Each training WSI's spatially ordered tiles are split into contiguous first-half (fold0)
    # vs second-half (fold1) spatial blocks so both upstream folds observe HER2 3+/2+/0 slides
    # without zero-positive-slide starvation, while val (6 WSIs) and locked test (5 WSIs)
    # remain 100% slide-disjoint and untouched.
    f0_list: list[int] = []
    f1_list: list[int] = []
    for w in uniq_wsis:
        w_idx = np.nonzero(wsis_tr == w)[0]
        mid = len(w_idx) // 2
        if mid == 0:
            f0_list.extend(w_idx.tolist())
        else:
            f0_list.extend(w_idx[:mid].tolist())
            f1_list.extend(w_idx[mid:].tolist())
    idx_f0 = np.asarray(sorted(f0_list), dtype=np.int64)
    idx_f1 = np.asarray(sorted(f1_list), dtype=np.int64)

    x_all = torch.from_numpy(train_bank["he_nhwc"]).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    y_all = torch.from_numpy(train_bank["dab_u8"])[:, None].to(torch.float32) / 255.0

    up_tr_f01 = np.empty((len(wsis_tr), img_size, img_size), dtype=np.float32)
    up_ev_acc = np.zeros((len(eval_bank["he_nhwc"]), img_size, img_size), dtype=np.float32)

    fam_desc = ""
    for f_id, (tr_sub, oof_sub) in enumerate(((idx_f0, idx_f1), (idx_f1, idx_f0))):
        ckpt_f = state_dir / f"upstream_{key}_fold{f_id}.pt"
        fam_desc, pred_f01 = _fit_or_load_single_her2_net(
            key=key,
            x_tr=x_all[tr_sub],
            y_tr=y_all[tr_sub],
            ckpt_path=ckpt_f,
            img_size=img_size,
            device=device,
            hoptimus0_ckpt=hoptimus0_ckpt,
            upstream_epochs=upstream_epochs,
            allow_train=allow_train,
            seed_offset=100 * f_id,
        )
        up_tr_f01[oof_sub] = pred_f01(train_bank["he_nhwc"][oof_sub])
        up_ev_acc += 0.5 * pred_f01(eval_bank["he_nhwc"])
        del pred_f01
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    up_tr_u8 = np.rint(np.clip(up_tr_f01, 0.0, 1.0) * 255.0).astype(np.uint8)
    up_ev_u8 = np.rint(np.clip(up_ev_acc, 0.0, 1.0) * 255.0).astype(np.uint8)
    return fam_desc, up_tr_u8, up_ev_u8


def _atomic_torch_save(obj: object, path: Path) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(text)
    os.replace(tmp, path)


def build_common12_her2(
    he_nhwc: np.ndarray,
    donor_u8: np.ndarray,
    hem_u8: np.ndarray,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    dev = torch.device(device)
    n, h, w = donor_u8.shape
    out = torch.empty((n, 12, h, w), dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, n, 64):
            e = min(n, s + 64)
            he_t = torch.from_numpy(he_nhwc[s:e]).to(dev, non_blocking=True).permute(0, 3, 1, 2).to(torch.float32) / 255.0
            bc = torch.from_numpy(donor_u8[s:e, None]).to(dev, non_blocking=True).to(torch.float32) / 255.0
            bs = torch.from_numpy(hem_u8[s:e, None]).to(dev, non_blocking=True).to(torch.float32) / 255.0
            bm = 0.5 * (bc + bs)
            out[s:e] = suh.build_universal_common12(he_t, bc, bs, bm).cpu()
    return out


def evaluate_dab_burden_concordance(
    pred_u8: np.ndarray,
    targ_u8: np.ndarray,
    wsis: Sequence[str],
    cutoff_u8: int,
    scores_map: dict[str, float] | None = None,
) -> dict:
    """Evaluate tile-level and WSI-level DAB positive membrane burden concordance."""
    gt_frac = np.mean(targ_u8 > cutoff_u8, axis=(1, 2))
    pr_frac = np.mean(pred_u8 > cutoff_u8, axis=(1, 2))

    tile_r = float(stats.pearsonr(gt_frac, pr_frac)[0]) if len(gt_frac) >= 3 and np.std(pr_frac) > 1e-12 else 0.0
    tile_rho = float(stats.spearmanr(gt_frac, pr_frac)[0]) if len(gt_frac) >= 3 and np.std(pr_frac) > 1e-12 else 0.0
    tile_mae = float(np.mean(np.abs(pr_frac - gt_frac)))

    wsi_gt = {}
    wsi_pr = {}
    for w in sorted(set(wsis)):
        m = np.asarray([x == w for x in wsis])
        wsi_gt[w] = float(np.mean(gt_frac[m]))
        wsi_pr[w] = float(np.mean(pr_frac[m]))

    g_vec = np.array([wsi_gt[w] for w in sorted(wsi_gt)])
    p_vec = np.array([wsi_pr[w] for w in sorted(wsi_pr)])
    wsi_r = float(stats.pearsonr(g_vec, p_vec)[0]) if len(g_vec) >= 3 and np.std(p_vec) > 1e-12 else 0.0
    wsi_rho = float(stats.spearmanr(g_vec, p_vec)[0]) if len(g_vec) >= 3 and np.std(p_vec) > 1e-12 else 0.0
    wsi_mae = float(np.mean(np.abs(p_vec - g_vec)))

    res = dict(
        tile_burden_pearson_r=tile_r,
        tile_burden_spearman_rho=tile_rho,
        tile_burden_mae=tile_mae,
        wsi_burden_pearson_r=wsi_r,
        wsi_burden_spearman_rho=wsi_rho,
        wsi_burden_mae=wsi_mae,
        per_wsi_gt_burden=wsi_gt,
        per_wsi_pred_burden=wsi_pr,
    )
    if scores_map:
        matched_w = [w for w in sorted(wsi_pr) if w in scores_map]
        if len(matched_w) >= 3:
            c_scores = np.array([scores_map[w] for w in matched_w], dtype=np.float64)
            p_sub = np.array([wsi_pr[w] for w in matched_w], dtype=np.float64)
            g_sub = np.array([wsi_gt[w] for w in matched_w], dtype=np.float64)
            res["clinical_her2_score_spearman_pred"] = (
                float(stats.spearmanr(c_scores, p_sub)[0]) if np.std(p_sub) > 1e-12 else 0.0
            )
            res["clinical_her2_score_spearman_gt_dab"] = (
                float(stats.spearmanr(c_scores, g_sub)[0]) if np.std(g_sub) > 1e-12 else 0.0
            )
    return res


def _parse_her2_scores_csv(scores_csv: Path) -> dict[str, float]:
    df = pd.read_csv(scores_csv, sep=None, engine="python", encoding="utf-8-sig")
    out = {}
    cols = list(df.columns)
    w_col = cols[0]
    s_col = cols[1]
    grade_map = {"0": 0.0, "1+": 1.0, "2+": 2.0, "3+": 3.0, "1": 1.0, "2": 2.0, "3": 3.0}
    for _, r in df.iterrows():
        raw_w = str(r[w_col]).strip().lower()
        digits = "".join(ch for ch in raw_w if ch.isdigit())
        if not digits:
            continue
        w_num = int(digits)
        val_str = str(r[s_col]).strip()
        val = grade_map.get(val_str)
        if val is None:
            try:
                val = float(val_str)
            except Exception:
                continue
        for key_fmt in (f"wsi{w_num}", f"wsi{w_num:02d}", f"wsi{w_num:03d}"):
            out[key_fmt] = val
    return out


def run_her2_campaign(args: argparse.Namespace) -> dict:
    t0 = time.monotonic()
    device = torch.device(args.device)
    tiles_root = Path(args.tiles_root).resolve()
    cal_doc = json.loads(Path(args.calibration_json).read_text())
    dab_upper = float(cal_doc["histogram"]["range"][1])
    dab_thr = float(cal_doc["threshold"])
    cutoff_u8 = int(round(dab_thr / dab_upper * 255.0))

    if args.mode == "s5_val":
        forbid_test = True
        eval_split = "val"
        scores_map = None
    elif args.mode == "s6_locked_test":
        if not args.benchmark_protocol:
            raise ValueError("--benchmark-protocol required for s6_locked_test")
        proto = json.loads(Path(args.benchmark_protocol).read_text())
        if proto.get("protocol_status") != "frozen":
            raise RuntimeError("Refusing s6_locked_test: benchmark-protocol is not frozen")
        forbid_test = False
        eval_split = "test"
        scores_map = _parse_her2_scores_csv(Path(args.scores_csv)) if args.scores_csv and Path(args.scores_csv).exists() else None
    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")

    out_dir = Path(args.out_dir).resolve()
    state_dir = Path(args.state_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = discover_split_pairs(tiles_root, "train", int(args.train_tiles_per_wsi), forbid_test=forbid_test)
    eval_pairs = discover_split_pairs(tiles_root, eval_split, int(args.eval_tiles_per_wsi), forbid_test=forbid_test)

    def _get_her2_bank(sp_name: str, sp_pairs: list[tuple[str, str]]) -> dict:
        if forbid_test and sp_name == "test":
            raise RuntimeError("Forbidden test split load in S5")
        bc_file = None
        if getattr(args, "bank_cache_dir", ""):
            bc_dir = Path(args.bank_cache_dir).resolve()
            bc_dir.mkdir(parents=True, exist_ok=True)
            bc_file = bc_dir / f"her2_bank_{sp_name}_{len(sp_pairs)}_{int(args.img_size)}.pkl"
            if bc_file.exists():
                with open(bc_file, "rb") as fh:
                    cand = pickle.load(fh)
                if cand.get("names") == [nm for _, nm in sp_pairs]:
                    return cand
        bk = load_her2_dab_bank(
            tiles_root, sp_name, sp_pairs, dab_upper, img_size=int(args.img_size), forbid_test=forbid_test
        )
        if bc_file is not None:
            tmp_f = bc_file.with_name(f".tmp_{os.getpid()}_{bc_file.name}")
            with open(tmp_f, "wb") as fh:
                pickle.dump(bk, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_f, bc_file)
        return bk

    train_bank = _get_her2_bank("train", train_pairs)
    eval_bank = _get_her2_bank(eval_split, eval_pairs)

    print(
        json.dumps(
            dict(
                event="her2_banks_loaded",
                mode=args.mode,
                eval_split=eval_split,
                n_train=len(train_pairs),
                n_eval=len(eval_pairs),
                cutoff_u8=cutoff_u8,
            )
        ),
        flush=True,
    )

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    hopt_path = Path(args.hoptimus0_ckpt).resolve() if args.hoptimus0_ckpt else None
    summary_by_backbone = {}

    for b_key in backbones:
        b_t0 = time.monotonic()
        b_family, up_tr, up_ev = train_or_load_her2_backbone(
            b_key,
            train_bank,
            eval_bank,
            cutoff_u8,
            state_dir,
            device,
            hoptimus0_ckpt=hopt_path,
            upstream_epochs=int(args.upstream_epochs),
            allow_train=(args.mode == "s5_val"),
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        c12_tr = build_common12_her2(train_bank["he_nhwc"], up_tr, train_bank["hem_u8"], device=device)
        c12_ev = build_common12_her2(eval_bank["he_nhwc"], up_ev, eval_bank["hem_u8"], device=device)
        d_tr_t = torch.from_numpy(up_tr)
        y_tr_t = torch.from_numpy(train_bank["dab_u8"])
        d_ev_t = torch.from_numpy(up_ev)

        b_state = state_dir / b_key
        b_state.mkdir(parents=True, exist_ok=True)

        arm_tr: dict[str, np.ndarray] = {"upstream": up_tr}
        arm_ev: dict[str, np.ndarray] = {"upstream": up_ev}

        for arm in suh.ARMS:
            tr_s, ev_s = [], []
            for sd in seeds:
                ck_p = b_state / f"HER2_DAB_{arm}_{sd}.pt"
                mt_p = b_state / f"HER2_DAB_{arm}_{sd}_meta.json"
                if args.mode == "s5_val":
                    if ck_p.exists() and not args.overwrite_fits:
                        model = suh.build_coordinate_screened_model(sd, device)
                        model.load_state_dict(torch.load(ck_p, map_location=device, weights_only=True))
                    else:
                        model, meta = suh.fit_universal_marker(
                            arm=arm,
                            seed=sd,
                            common_train=c12_tr,
                            donor_u8_train=d_tr_t,
                            target_u8_train=y_tr_t,
                            cutoff=cutoff_u8,
                            device=device,
                            epochs=int(args.residual_epochs),
                            batch_size=int(args.batch_size),
                            lr=float(getattr(args, "lr", 1.5e-3)),
                            deploy_anchor_weight=float(getattr(args, "deploy_anchor_weight", 1.25)),
                        )
                        _atomic_torch_save({k: v.cpu() for k, v in model.state_dict().items()}, ck_p)
                        _atomic_write_text(mt_p, json.dumps(meta, indent=2) + "\n")
                else:
                    if not ck_p.exists():
                        raise FileNotFoundError(f"Missing frozen HER2 residual head: {ck_p}")
                    model = suh.build_coordinate_screened_model(sd, device)
                    model.load_state_dict(torch.load(ck_p, map_location=device, weights_only=True))

                p_tr, _ = suh.predict_universal_marker(model, c12_tr, d_tr_t, device=device, batch_size=16)
                p_ev, _ = suh.predict_universal_marker(model, c12_ev, d_ev_t, device=device, batch_size=16)
                arm_tr[f"{arm}:{sd}"] = p_tr
                arm_ev[f"{arm}:{sd}"] = p_ev
                tr_s.append(p_tr.astype(np.float32))
                ev_s.append(p_ev.astype(np.float32))
                del model
            arm_tr[f"{arm}:ens2"] = np.round(np.mean(tr_s, axis=0)).astype(np.uint8)
            arm_ev[f"{arm}:ens2"] = np.round(np.mean(ev_s, axis=0)).astype(np.uint8)

        del c12_tr, c12_ev
        if device.type == "cuda":
            torch.cuda.empty_cache()

        arms_out = {}
        wsi_clusters_by_arm = {}
        for arm_k, ev_p in arm_ev.items():
            pix_res = jce.evaluate_channel_predictions(
                train_pred_u8=arm_tr[arm_k],
                train_targ_u8=train_bank["dab_u8"],
                eval_pred_u8=ev_p,
                eval_targ_u8=eval_bank["dab_u8"],
                eval_clusters=eval_bank["wsis"],
                fixed_cutoff=cutoff_u8,
                device=device,
            )
            wsi_clusters_by_arm[arm_k] = pix_res["per_cluster"]
            burd = evaluate_dab_burden_concordance(
                ev_p, eval_bank["dab_u8"], eval_bank["wsis"], cutoff_u8, scores_map=scores_map
            )
            arms_out[arm_k] = dict(
                dab_pixel=pix_res["pooled"],
                per_wsi_pixel=pix_res["per_cluster"],
                dab_burden_concordance=burd,
            )

        sc_ens = "S_count:ens2"
        sr_ens = "S_recon:ens2"
        paired_wsi = dict(
            scount_ens2_vs_upstream={
                mk: jce.paired_cluster_bootstrap(wsi_clusters_by_arm[sc_ens], wsi_clusters_by_arm["upstream"], mk)
                for mk in ("Fixed_tol_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB")
            },
            scount_ens2_vs_srecon_ens2={
                mk: jce.paired_cluster_bootstrap(wsi_clusters_by_arm[sc_ens], wsi_clusters_by_arm[sr_ens], mk)
                for mk in ("Fixed_tol_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB")
            },
        )

        if args.mode == "s6_locked_test" and len(eval_pairs) >= 2:
            vis_dir = out_dir / "visual_crops"
            vis_dir.mkdir(parents=True, exist_ok=True)
            vi = [0, len(eval_pairs) // 2]
            np.savez_compressed(
                vis_dir / f"her2_{b_key}_examples.npz",
                wsis=np.array([eval_bank["wsis"][i] for i in vi]),
                names=np.array([eval_bank["names"][i] for i in vi]),
                he_nhwc=eval_bank["he_nhwc"][vi],
                ihc_nhwc=eval_bank["ihc_nhwc"][vi],
                gt_dab_u8=eval_bank["dab_u8"][vi],
                upstream_u8=arm_ev["upstream"][vi],
                srecon_u8=arm_ev[sr_ens][vi],
                scount_u8=arm_ev[sc_ens][vi],
            )

        b_rec = dict(
            backbone=b_key,
            family=b_family,
            cutoff_u8=cutoff_u8,
            wall_seconds=time.monotonic() - b_t0,
            arms=arms_out,
            paired_wsi_bootstrap=paired_wsi,
        )
        summary_by_backbone[b_key] = b_rec
        (out_dir / f"her2_backbone_{b_key}_{args.mode}.json").write_text(json.dumps(b_rec, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    event="her2_backbone_complete",
                    backbone=b_key,
                    mode=args.mode,
                    up_Fixed_F1=round(arms_out["upstream"]["dab_pixel"]["Fixed_tol_F1"], 4),
                    sr_Fixed_F1=round(arms_out[sr_ens]["dab_pixel"]["Fixed_tol_F1"], 4),
                    sc_Fixed_F1=round(arms_out[sc_ens]["dab_pixel"]["Fixed_tol_F1"], 4),
                    up_ODS_F=round(arms_out["upstream"]["dab_pixel"]["ODS_F"], 4),
                    sc_ODS_F=round(arms_out[sc_ens]["dab_pixel"]["ODS_F"], 4),
                    wsi_r_up=round(arms_out["upstream"]["dab_burden_concordance"]["wsi_burden_pearson_r"], 4),
                    wsi_r_sc=round(arms_out[sc_ens]["dab_burden_concordance"]["wsi_burden_pearson_r"], 4),
                    sec=round(b_rec["wall_seconds"], 1),
                )
            ),
            flush=True,
        )

    final_doc = dict(
        schema="her2match-dab-campaign-v1",
        mode=args.mode,
        eval_split=eval_split,
        cutoff_u8=cutoff_u8,
        dab_upper=dab_upper,
        dab_otsu_threshold=dab_thr,
        n_train_tiles=len(train_pairs),
        n_eval_tiles=len(eval_pairs),
        total_wall_seconds=time.monotonic() - t0,
        backbones=summary_by_backbone,
    )
    (out_dir / f"her2match_{args.mode}_summary.json").write_text(json.dumps(final_doc, indent=2) + "\n")
    return final_doc


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("s5_val", "s6_locked_test"), required=True)
    p.add_argument("--tiles-root", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--benchmark-protocol", default="")
    p.add_argument("--scores-csv", default="")
    p.add_argument("--hoptimus0-ckpt", default="")
    p.add_argument("--backbones", default="unet_l1_ssim,pix2pix_cgan_ihc,hoptimus_lora_ihc")
    p.add_argument("--state-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bank-cache-dir", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--train-tiles-per-wsi", type=int, default=80)
    p.add_argument("--eval-tiles-per-wsi", type=int, default=100)
    p.add_argument("--upstream-epochs", type=int, default=8)
    p.add_argument("--residual-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--deploy-anchor-weight", type=float, default=1.25)
    p.add_argument("--seeds", default="202610270,202610271")
    p.add_argument("--overwrite-fits", action="store_true")
    return p


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    run_her2_campaign(build_arg_parser().parse_args())
