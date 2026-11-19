"""Matched HEMIT 3-marker adaptation runner for MIPHEI-ViT (Balezo et al., 2026).

Implements the official MIPHEI-ViT generator and training recipe from
``Sanofi-Public/MIPHEI-ViT`` (``configs/default_configs/miphei-vit.yaml``,
``src/generators/mipheivit.py``, ``src/loss.py``, ``src/models.py``, ``src/utils.py``)
without external Hydra, PyTorch Lightning, W&B, or PyVIPS dependencies:
  * Encoder: ``bioptimus/H-optimus-0`` (``vit_giant_patch14_reg4_dinov2`` at 512x512)
    with Q/V LoRA adapters (``rank=8, alpha=1.0``) and frozen base ViT parameters.
  * Decoder: ``Detail_Capture`` (``ConvStream [48, 96, 192]`` + ``Fusion_Block [256, 128, 64, 32]``
    + per-marker attention ``SegmentationHead`` with ``Tanh`` and Pix2Pix normal init).
  * Normalization: H&E ``(x - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]``;
    mIF target ``y / 255 * 1.8 - 0.9`` in ``[-0.9, 0.9]``; output byte quantization
    ``round(clip((pred + 0.9) / 1.8, 0, 1) * 255)``.
  * Objective & schedule: ``WeightedMSELoss(lambda_factor=50.0, marker_weights)`` with
    inverse-variance channel weights computed strictly on training-split targets,
    15 epochs, effective batch size 16 (``micro_batch=2, grad_accum=8``),
    ``Adam(lr=2e-4 * sqrt(16)=8e-4, betas=(0.5, 0.999), eps=1e-7)``, fp16 mixed
    precision, gradient norm clip 1.0, and ``pix2pix_lr_scheduler`` step schedule.

Supports two strict split modes (zero ``test`` or ``outer8`` access):
  * ``fit1192``: trains on the exact 1,192 ``FIT1192`` patches, calibrates ``Cal. F1``
    thresholds on centered ``FIT1192``, and scores both ``HOLD (300)`` and ``Val (630)``.
  * ``full3717``: trains on all 3,717 HEMIT ``train`` patches, calibrates ``Cal. F1``
    thresholds on centered ``FIT1192``, and scores ``Val (630)``.
"""
from __future__ import annotations

import argparse
from functools import partial
import gc
import gzip
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import check_query_relation_scaled as scaled_check
from . import check_screened_multimarker_scaled as mm_check
from . import query_relation_scaled_data as scaled_data
from . import query_relation_scoring as scoring
from . import run_query_relation_ensemble as old_eval
from . import run_query_relation_scaled as scaled
from . import run_screened_multimarker_scaled as mm

SCHEMA = 'mipheivit-hemit-comparator-v1'
HE_MEAN = (123.675, 116.28, 103.53)
HE_STD = (58.395, 57.12, 57.375)
HOPTIMUS0_HF_ID = 'hf-hub:bioptimus/H-optimus-0'
MODES = ('fit1192', 'full3717')
require = scaled_data.require
metadata = old_eval.metadata
read_json = old_eval.read_json


# ---------------------------------------------------------------------------
# Official MIPHEI-ViT architecture & LoRA modules
# ---------------------------------------------------------------------------

def initialize_decoder_head(module: nn.Module) -> None:
    """Pix2Pix normal initialization matching ``mipheivit_src/src/generators/smp_unet.py``."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class AttentionBlock(nn.Module):
    def __init__(self, in_chns: int):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Conv2d(in_chns, in_chns // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_chns // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_chns // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(x)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 activation: nn.Module | None = None, use_attention: bool = False):
        attention = AttentionBlock(in_channels) if use_attention else nn.Identity()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        act = activation if activation is not None else nn.Identity()
        super().__init__(attention, conv2d, act)
        initialize_decoder_head(self)


class Basic_Conv3x3(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ConvStream(nn.Module):
    def __init__(self, in_chans: int = 3, out_chans: list[int] | None = None):
        super().__init__()
        if out_chans is None:
            out_chans = [48, 96, 192]
        self.convs = nn.ModuleList()
        self.conv_chans = [in_chans, *out_chans]
        for i in range(len(self.conv_chans) - 1):
            self.convs.append(Basic_Conv3x3(self.conv_chans[i], self.conv_chans[i + 1]))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out_dict = {'D0': x}
        for i, conv in enumerate(self.convs):
            x = conv(x)
            out_dict[f'D{i + 1}'] = x
        return out_dict


class Fusion_Block(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x: torch.Tensor, d_feat: torch.Tensor) -> torch.Tensor:
        f_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(torch.cat([d_feat, f_up], dim=1))


class Encoder(nn.Module):
    def __init__(self, vit: nn.Module):
        super().__init__()
        if not hasattr(vit, 'patch_embed'):
            raise ValueError(f'Expected ViT/Swin backbone with patch_embed, got {type(vit)}')
        self.vit = vit
        self.is_swint = type(vit).__name__ == 'SwinTransformer'
        self.grid_size = self.vit.patch_embed.grid_size
        if self.is_swint:
            self.num_prefix_tokens = 0
            self.embed_dim = self.vit.embed_dim * 2 ** (self.vit.num_layers - 1)
        else:
            self.num_prefix_tokens = self.vit.num_prefix_tokens
            self.embed_dim = self.vit.embed_dim
        patch_size = self.vit.patch_embed.patch_size
        img_size = self.vit.patch_embed.img_size
        require(img_size[0] % 16 == 0 and img_size[1] % 16 == 0, 'Encoder img_size must be divisible by 16')
        if self.is_swint:
            self.scale_factor = (2.0, 2.0)
        elif patch_size != (16, 16):
            target_grid = (img_size[0] / 16, img_size[1] / 16)
            self.scale_factor = (target_grid[0] / self.grid_size[0], target_grid[1] / self.grid_size[1])
        else:
            self.scale_factor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit(x)
        if self.is_swint:
            features = features.permute(0, 3, 1, 2)
        else:
            features = features[:, self.num_prefix_tokens:]
            features = features.permute(0, 2, 1)
            features = features.view((-1, self.embed_dim, *self.grid_size))
        if self.scale_factor is not None:
            features = F.interpolate(features, scale_factor=self.scale_factor, mode='bicubic')
        return features


class Detail_Capture(nn.Module):
    def __init__(self, emb_chans: int, in_chans: int = 3, out_chans: int = 3,
                 convstream_out: list[int] | None = None,
                 fusion_out: list[int] | None = None,
                 use_attention: bool = True,
                 activation: nn.Module | None = None):
        super().__init__()
        if convstream_out is None:
            convstream_out = [48, 96, 192]
        if fusion_out is None:
            fusion_out = [256, 128, 64, 32]
        if activation is None:
            activation = nn.Tanh()
        require(len(fusion_out) == len(convstream_out) + 1, 'fusion_out must have len(convstream_out)+1')
        self.convstream = ConvStream(in_chans=in_chans, out_chans=convstream_out)
        self.conv_chans = self.convstream.conv_chans
        self.num_heads = out_chans
        self.fusion_blks = nn.ModuleList()
        self.fus_channs = [emb_chans, *fusion_out]
        for i in range(len(self.fus_channs) - 1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans=self.fus_channs[i] + self.conv_chans[-(i + 1)],
                    out_chans=self.fus_channs[i + 1],
                )
            )
        for idx in range(self.num_heads):
            setattr(self, f'segmentation_head_{idx}', SegmentationHead(
                in_channels=fusion_out[-1],
                out_channels=1,
                activation=activation,
                kernel_size=3,
                use_attention=use_attention,
            ))

    def forward(self, features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        detail_features = self.convstream(images)
        for i, blk in enumerate(self.fusion_blks):
            d_name = f'D{len(self.fusion_blks) - i - 1}'
            features = blk(features, detail_features[d_name])
        outputs = [getattr(self, f'segmentation_head_{idx}')(features) for idx in range(self.num_heads)]
        return torch.cat(outputs, dim=1)


class MIPHEIViT(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.initialize()

    def initialize(self) -> None:
        initialize_decoder_head(self.decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x), x)


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        std = math.sqrt(float(rank))
        self.A = nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A @ self.B)


class QkvWithLoRA(nn.Module):
    def __init__(self, qkv: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qkv(x).clone()
        out[:, :, :self.dim] = out[:, :, :self.dim] + self.lora_q(x)
        out[:, :, -self.dim:] = out[:, :, -self.dim:] + self.lora_v(x)
        return out


class _SyntheticPatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _SyntheticAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(float(self.dim)), dim=-1)
        return self.proj(attn @ v)


class _SyntheticBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SyntheticAttn(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SyntheticViT(nn.Module):
    """Lightweight offline ViT with exact timm attribute contract for unit tests."""

    def __init__(self, img_size: int = 112, patch_size: int = 14, embed_dim: int = 64,
                 depth: int = 2, num_prefix_tokens: int = 5):
        super().__init__()
        self.patch_embed = _SyntheticPatchEmbed(img_size, patch_size, embed_dim)
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.prefix = nn.Parameter(torch.zeros(1, num_prefix_tokens, embed_dim))
        self.blocks = nn.ModuleList([_SyntheticBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        pref = self.prefix.expand(tokens.shape[0], -1, -1)
        out = torch.cat([pref, tokens], dim=1)
        for blk in self.blocks:
            out = blk(out)
        return self.norm(out)


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 1.0) -> None:
    require(hasattr(model, 'blocks'), 'ViT backbone with .blocks required for LoRA')
    assign_lora = partial(QkvWithLoRA, rank=rank, alpha=alpha)
    for block in model.blocks:
        block.attn.qkv = assign_lora(block.attn.qkv)
    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks:
        for param in block.attn.qkv.lora_q.parameters():
            param.requires_grad = True
        for param in block.attn.qkv.lora_v.parameters():
            param.requires_grad = True


def build_hoptimus0(img_size: int = 512, ckpt_path: str | None = None):
    import timm
    from timm.layers import resample_abs_pos_embed
    from timm.models import load_state_dict_from_hf, parse_model_name

    model = timm.create_model(
        'vit_giant_patch14_reg4_dinov2',
        img_size=img_size,
        drop_path_rate=0.0,
        num_classes=0,
        global_pool='',
        pretrained=False,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    else:
        _, model_id = parse_model_name(HOPTIMUS0_HF_ID)
        state_dict = load_state_dict_from_hf(model_id, weights_only=True)
    if img_size != 224:
        state_dict['pos_embed'] = resample_abs_pos_embed(
            state_dict['pos_embed'],
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=0 if model.no_embed_class else model.num_prefix_tokens,
        )
    model.load_state_dict(state_dict, strict=True)
    return model


def build_mipheivit_generator(*, img_size: int = 512, num_classes: int = 3,
                              lora_rank: int = 8, lora_alpha: float = 1.0,
                              ckpt_path: str | None = None,
                              synthetic_backbone: bool = False) -> MIPHEIViT:
    if synthetic_backbone:
        vit = SyntheticViT(
            img_size=img_size,
            patch_size=14 if img_size >= 112 else 16,
            embed_dim=64,
            depth=2,
            num_prefix_tokens=5,
        )
    else:
        vit = build_hoptimus0(img_size=img_size, ckpt_path=ckpt_path)
    apply_lora(vit, rank=lora_rank, alpha=lora_alpha)
    encoder = Encoder(vit)
    decoder = Detail_Capture(
        emb_chans=encoder.embed_dim,
        in_chans=3,
        out_chans=num_classes,
        use_attention=True,
        activation=nn.Tanh(),
    )
    return MIPHEIViT(encoder=encoder, decoder=decoder)


class WeightedMSELoss(nn.Module):
    def __init__(self, lambda_factor: float, marker_weights: torch.Tensor):
        super().__init__()
        self.lambda_factor = float(lambda_factor)
        self.register_buffer('marker_weights', marker_weights.to(torch.float32))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(input=y_pred.float(), target=y_true.float(), reduction='none')
        loss = loss.mean(dim=(0, 2, 3)) * self.marker_weights
        return loss.mean() * self.lambda_factor


def pix2pix_lr_scheduler(total_iters: int, warmup_iters: int, decay_start_iter: int):
    require(total_iters >= 2 and 0 <= warmup_iters <= decay_start_iter < total_iters,
            f'invalid pix2pix schedule bounds: total={total_iters}, warmup={warmup_iters}, decay={decay_start_iter}')

    def lr_lambda(step: int) -> float:
        if warmup_iters > 0 and step < warmup_iters:
            return step / warmup_iters
        if step < decay_start_iter:
            return 1.0
        decay_steps = total_iters - decay_start_iter
        decay_factor = (step - decay_start_iter) / decay_steps
        return max(0.0, 1.0 - decay_factor)

    return lr_lambda


# ---------------------------------------------------------------------------
# Strict HEMIT manifest dataset & preprocessing
# ---------------------------------------------------------------------------

def normalize_he_batch(he_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """B x H x W x 3 uint8 -> B x 3 x H x W float32 normalized with MIPHEI H&E stats."""
    x = he_u8.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(HE_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(HE_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


def normalize_if_batch(if_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """B x H x W x 3 uint8 -> B x 3 x H x W float32 in [-0.9, 0.9]."""
    y = if_u8.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
    return y * (1.8 / 255.0) - 0.9


def quantize_miphei_prediction(pred_chw: torch.Tensor) -> tuple[np.ndarray, list[dict]]:
    """3 x H x W float tensor -> (H x W x 3 uint8, per-channel clipping receipts)."""
    arr = pred_chw.detach().to(dtype=torch.float64, device='cpu').numpy()
    raw_01 = (arr + 0.9) / 1.8
    clipped_01 = np.clip(raw_01, 0.0, 1.0)
    u8_chw = np.rint(255.0 * clipped_01).astype(np.uint8)
    u8_hwc = np.transpose(u8_chw, (1, 2, 0)).copy()
    pixels = int(u8_hwc.shape[0] * u8_hwc.shape[1])
    clips = []
    for ch in range(3):
        z = raw_01[ch]
        c = clipped_01[ch]
        b = u8_hwc[..., ch]
        clips.append(dict(
            pixels=pixels,
            below_zero=int((z < 0.0).sum()),
            above_one=int((z > 1.0).sum()),
            preclip_mean=float(z.mean()),
            postclip_mean=float(c.mean()),
            rounding_mean_change=float(b.mean() / 255.0 - c.mean()),
        ))
    return u8_hwc, clips


def load_tile_rgb(path: Path, expected_size: int) -> np.ndarray:
    require(path.is_file() and not path.is_symlink(), f'regular tile file required: {path}')
    require('test' not in path.parts, f'locked test split access forbidden: {path}')
    with Image.open(path) as img:
        require(img.size == (expected_size, expected_size),
                f'expected {expected_size}x{expected_size} tile, got {img.size}: {path}')
        arr = np.asarray(img.convert('RGB'), dtype=np.uint8)
    require(arr.shape == (expected_size, expected_size, 3) and arr.dtype == np.uint8,
            f'invalid RGB uint8 array: {path}')
    return arr


def center_crop_rgb(arr: np.ndarray, crop_size: int) -> np.ndarray:
    h, w, _ = arr.shape
    require(h >= crop_size and w >= crop_size and (h - crop_size) % 2 == 0 and (w - crop_size) % 2 == 0,
            f'cannot center-crop {crop_size} from {arr.shape}')
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return np.ascontiguousarray(arr[top:top + crop_size, left:left + crop_size])


class MipheiHemitTrainDataset(Dataset):
    """Paired random-crop + H/V flip dataset strictly over authorized train basenames."""

    def __init__(self, hemit_root: Path, names: list[str], *, tile_size: int = 1024,
                 crop_size: int = 512):
        self.hemit_root = Path(hemit_root).resolve()
        require('test' not in self.hemit_root.parts, 'locked test root forbidden')
        self.names = list(names)
        self.tile_size = int(tile_size)
        self.crop_size = int(crop_size)
        require(len(self.names) >= 2 and len(set(self.names)) == len(self.names),
                'unique nonempty training basenames required')
        self.pairs = []
        for name in self.names:
            in_p = self.hemit_root / 'train' / 'input' / name
            lb_p = self.hemit_root / 'train' / 'label' / name
            require(in_p.is_file() and lb_p.is_file(), f'missing train pair for {name}')
            self.pairs.append((name, in_p, lb_p))
        self.rng = np.random.default_rng(0)

    def set_epoch_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        name, in_p, lb_p = self.pairs[index]
        he = load_tile_rgb(in_p, self.tile_size)
        target = load_tile_rgb(lb_p, self.tile_size)
        max_off = self.tile_size - self.crop_size
        if max_off > 0:
            top = int(self.rng.integers(0, max_off + 1))
            left = int(self.rng.integers(0, max_off + 1))
            he = he[top:top + self.crop_size, left:left + self.crop_size]
            target = target[top:top + self.crop_size, left:left + self.crop_size]
        if float(self.rng.random()) < 0.5:
            he = he[:, ::-1]
            target = target[:, ::-1]
        if float(self.rng.random()) < 0.5:
            he = he[::-1, :]
            target = target[::-1, :]
        return (
            torch.from_numpy(np.ascontiguousarray(he)),
            torch.from_numpy(np.ascontiguousarray(target)),
            name,
        )


def compute_training_channel_stats(hemit_root: Path, names: list[str], *,
                                   tile_size: int, crop_size: int) -> dict:
    """Compute training-only per-marker byte std and inverse-variance weights."""
    sum_c = np.zeros(3, dtype=np.float64)
    sum_sq_c = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    for name in names:
        lb_p = hemit_root / 'train' / 'label' / name
        crop = center_crop_rgb(load_tile_rgb(lb_p, tile_size), crop_size).astype(np.float64)
        sum_c += crop.sum(axis=(0, 1))
        sum_sq_c += np.square(crop).sum(axis=(0, 1))
        total_pixels += int(crop.shape[0] * crop.shape[1])
    require(total_pixels > 0, 'empty training pixel denominator')
    mean_c = sum_c / total_pixels
    var_c = np.maximum(sum_sq_c / total_pixels - np.square(mean_c), 1e-12)
    std_c = np.sqrt(var_c)
    inv_var = 1.0 / var_c
    weights = inv_var / inv_var.min()
    return dict(
        training_tiles=len(names),
        pixels_per_channel=total_pixels,
        channel_names=list(scoring.MARKERS),
        mean_u8=[float(v) for v in mean_c],
        std_u8=[float(v) for v in std_c],
        var_u8=[float(v) for v in var_c],
        marker_weights=[float(v) for v in weights],
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calibrate_training_thresholds(model: nn.Module, hemit_root: Path, cal_names: list[str], *,
                                  device: torch.device, tile_size: int, crop_size: int,
                                  eval_batch_size: int, use_amp: bool) -> dict:
    """Calibrate per-marker ODS thresholds strictly on centered FIT1192 training crops."""
    model.eval()
    pooled_counts = {m: None for m in scoring.MARKERS}
    kernel = scoring.tolerance_kernel()
    for start in range(0, len(cal_names), eval_batch_size):
        batch_names = cal_names[start:start + eval_batch_size]
        he_crops = [
            center_crop_rgb(load_tile_rgb(hemit_root / 'train' / 'input' / n, tile_size), crop_size)
            for n in batch_names
        ]
        he_t = normalize_he_batch(torch.from_numpy(np.stack(he_crops, axis=0)), device)
        with torch.inference_mode():
            if use_amp and device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    preds = model(he_t).float()
            else:
                preds = model(he_t).float()
        for b_idx, name in enumerate(batch_names):
            pred_u8, _ = quantize_miphei_prediction(preds[b_idx])
            target_u8 = center_crop_rgb(
                load_tile_rgb(hemit_root / 'train' / 'label' / name, tile_size), crop_size)
            for ch, (marker, cutoff) in enumerate(zip(scoring.MARKERS, scoring.CUTOFFS)):
                labels = target_u8[..., ch] > cutoff
                raw = scoring.reference.counts(pred_u8[..., ch], labels, kernel)
                cnt = {k: int(v) if k == 'G' else v.tolist() for k, v in raw.items()}
                pooled_counts[marker] = scaled.add_counts(pooled_counts[marker], cnt)

    thresholds = []
    train_ods = {}
    for marker, cutoff in zip(scoring.MARKERS, scoring.CUTOFFS):
        ep = scaled_check.endpoints(pooled_counts[marker], cutoff, cutoff)
        thresholds.append(int(ep['ODS_threshold']))
        train_ods[marker] = dict(
            ODS_F=float(ep['ODS_F']),
            ODS_threshold=int(ep['ODS_threshold']),
            AP=float(ep['AP']),
            cutoff_tolerant_F=float(ep['cutoff_tolerant_F']),
        )
    return dict(
        calibration_tiles=len(cal_names),
        train_thresholds=thresholds,
        per_marker_train_endpoints=train_ods,
        mean_train_ODS_F=float(sum(train_ods[m]['ODS_F'] for m in scoring.MARKERS) / 3.0),
    )


def score_held_population(model: nn.Module, hemit_root: Path, split_name: str,
                          names: list[str], train_thresholds: list[int], *,
                          quality, device: torch.device, tile_size: int,
                          crop_size: int, eval_batch_size: int,
                          use_amp: bool, per_image_path: Path) -> tuple[dict, int]:
    """Generate all batch predictions BEFORE reading any held target tile, then score bytes."""
    require(split_name in ('hold', 'val'), f'unexpected held split: {split_name}')
    disk_split = 'train' if split_name == 'hold' else 'val'
    model.eval()
    rows = []
    target_reads = 0
    with gzip.open(per_image_path, 'wt', encoding='utf-8') as gz_out:
        for start in range(0, len(names), eval_batch_size):
            batch_names = names[start:start + eval_batch_size]
            he_crops = [
                center_crop_rgb(load_tile_rgb(hemit_root / disk_split / 'input' / n, tile_size), crop_size)
                for n in batch_names
            ]
            he_t = normalize_he_batch(torch.from_numpy(np.stack(he_crops, axis=0)), device)
            with torch.inference_mode():
                if use_amp and device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        preds = model(he_t).float()
                else:
                    preds = model(he_t).float()
            frozen_outputs = []
            for b_idx, name in enumerate(batch_names):
                pred_u8, clips = quantize_miphei_prediction(preds[b_idx])
                pred_u8.setflags(write=False)
                frozen_outputs.append((name, pred_u8, clips))

            # Targets are read strictly AFTER predictions for this batch are frozen.
            for name, pred_u8, clips in frozen_outputs:
                target_u8 = center_crop_rgb(
                    load_tile_rgb(hemit_root / disk_split / 'label' / name, tile_size), crop_size)
                target_reads += 1
                # Pass pred_u8 as self-reference donor so count curves are computed once cleanly.
                sc = mm.score_multimarker_bytes(pred_u8, target_u8, pred_u8, quality, clips)
                row = dict(split=split_name, name=name, scores=sc)
                rows.append(row)
                gz_out.write(json.dumps(dict(
                    split=split_name,
                    name=name,
                    quality=sc['quality'],
                    ODS_preview=[
                        scaled_check.endpoints(sc['counts'][m], c, thr)['ODS_F']
                        for m, c, thr in zip(scoring.MARKERS, scoring.CUTOFFS, train_thresholds)
                    ],
                )) + '\n')

    summary = mm_check._summarize_population(rows, train_thresholds)
    return summary, target_reads


def run_one(cfg: dict, *, mode: str, seed: int, output_dir: Path, device_str: str) -> dict:
    require(mode in MODES, f'mode must be one of {MODES}')
    started = time.monotonic()
    device = torch.device(device_str)
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), f'one-shot output directory already exists: {output_dir}')
    output_dir.mkdir(parents=True)

    synthetic = bool(cfg.get('synthetic_fixture', False))
    hemit_root = Path(cfg['hemit_root']).resolve()
    require('test' not in hemit_root.parts, 'locked test path forbidden')
    tile_size = int(cfg.get('tile_size', 1024))
    crop_size = int(cfg.get('crop_size', 512))
    epochs = int(cfg.get('epochs', 15))
    micro_batch = int(cfg.get('micro_batch', 2))
    grad_accum = int(cfg.get('grad_accum', 8))
    effective_batch = micro_batch * grad_accum
    base_lr = float(cfg.get('base_lr', 0.0002))
    lr_g = base_lr * math.sqrt(float(effective_batch))
    lambda_factor = float(cfg.get('lambda_factor', 50.0))
    eval_batch_size = int(cfg.get('eval_batch_size', 4))
    use_amp = bool(cfg.get('use_amp', device.type == 'cuda')) and device.type == 'cuda'
    num_workers = int(cfg.get('dataloader_workers', 2 if not synthetic else 0))
    require(0 <= num_workers <= 4, 'dataloader_workers must be in 0..4')

    if synthetic:
        membership = cfg['synthetic_membership']
        data_manifest = cfg['synthetic_data_manifest']
    else:
        fit_meta = read_json(cfg['fit_metadata'])
        pilot_man = read_json(cfg['pilot_manifest'])
        data_manifest = read_json(cfg['data_manifest'])
        membership = scaled_data.build_membership(fit_meta, pilot_man, data_manifest)
        require(len(data_manifest['train']) == 3717 and len(data_manifest['val']) == 630,
                'canonical HEMIT 3717/630 train/val manifest required')

    fit1192_names = [r['name'] for r in membership['training']]
    excluded_outer8 = set(membership['excluded_outer8'])
    hold_names = list(membership['hold'])
    val_names = list(membership['val'])
    require(not (set(fit1192_names) & excluded_outer8), 'outer8 overlap in fit1192')
    require(not (set(fit1192_names) & set(hold_names)), 'hold overlap in fit1192')
    require(not (set(fit1192_names) & set(val_names)), 'val overlap in fit1192')

    if mode == 'fit1192':
        train_names = fit1192_names
        held_splits = [('hold', hold_names), ('val', val_names)]
    else:
        train_names = list(data_manifest['train'])
        require(not (set(train_names) & set(val_names)), 'full3717 overlaps val')
        held_splits = [('val', val_names)]

    require(len(train_names) >= effective_batch,
            f'training set ({len(train_names)}) smaller than effective_batch ({effective_batch})')

    seed_everything(seed)
    stats = compute_training_channel_stats(
        hemit_root, train_names, tile_size=tile_size, crop_size=crop_size)
    (output_dir / 'channel-stats.json').write_text(json.dumps(stats, indent=2) + '\n')

    model = build_mipheivit_generator(
        img_size=crop_size,
        num_classes=3,
        lora_rank=int(cfg.get('lora_rank', 8)),
        lora_alpha=float(cfg.get('lora_alpha', 1.0)),
        ckpt_path=cfg.get('hoptimus0_ckpt_path'),
        synthetic_backbone=synthetic,
    ).to(device)
    loss_fn = WeightedMSELoss(
        lambda_factor=lambda_factor,
        marker_weights=torch.tensor(stats['marker_weights'], dtype=torch.float32, device=device),
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_numel = sum(p.numel() for p in trainable_params)
    require(trainable_numel > 0, 'MIPHEIViT has zero trainable parameters')

    steps_per_epoch = len(train_names) // effective_batch
    micro_batches_per_epoch = steps_per_epoch * grad_accum
    total_iters = steps_per_epoch * epochs
    warmup_iters = min(400, total_iters // 2)
    decay_start_iter = total_iters // 2

    optimizer = torch.optim.Adam(trainable_params, lr=lr_g, betas=(0.5, 0.999), eps=1e-7)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=pix2pix_lr_scheduler(total_iters, warmup_iters, decay_start_iter),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    dataset = MipheiHemitTrainDataset(
        hemit_root, train_names, tile_size=tile_size, crop_size=crop_size)
    scratch_ckpt_dir = Path(cfg['scratch_root']).resolve() / f'{mode}-seed{seed}'
    scratch_ckpt_dir.mkdir(parents=True, exist_ok=True)

    traj_path = output_dir / 'trajectory.jsonl'
    global_opt_step = 0
    first_finite_logged = False

    print(json.dumps(dict(
        event='mipheivit_initialized',
        mode=mode,
        seed=seed,
        train_tiles=len(train_names),
        epochs=epochs,
        effective_batch=effective_batch,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        steps_per_epoch=steps_per_epoch,
        total_iters=total_iters,
        lr_g=lr_g,
        marker_weights=stats['marker_weights'],
        total_params=total_params,
        trainable_params=trainable_numel,
        use_amp=use_amp,
    )), flush=True)

    with traj_path.open('w', encoding='utf-8') as traj_file:
        for epoch in range(1, epochs + 1):
            ep_start = time.monotonic()
            model.train()
            dataset.set_epoch_seed(seed * 1009 + epoch)
            g_loader = torch.Generator()
            g_loader.manual_seed(seed * 1009 + epoch)
            loader = DataLoader(
                dataset,
                batch_size=micro_batch,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda'),
                generator=g_loader,
            )
            optimizer.zero_grad(set_to_none=True)
            ep_losses = []
            ep_grad_norms = []
            skipped_steps = 0
            accum_loss = 0.0
            micro_seen = 0

            for he_u8, if_u8, batch_names in loader:
                if micro_seen >= micro_batches_per_epoch:
                    break
                micro_seen += 1
                x = normalize_he_batch(he_u8, device)
                y = normalize_if_batch(if_u8, device)
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        pred = model(x)
                        require(bool(torch.isfinite(pred).all()),
                                f'non-finite generator output at epoch={epoch}, micro={micro_seen}')
                        raw_loss = loss_fn(y_true=y, y_pred=pred)
                        scaled_loss = raw_loss / grad_accum
                    scaler.scale(scaled_loss).backward()
                else:
                    pred = model(x)
                    require(bool(torch.isfinite(pred).all()),
                            f'non-finite generator output at epoch={epoch}, micro={micro_seen}')
                    raw_loss = loss_fn(y_true=y, y_pred=pred)
                    (raw_loss / grad_accum).backward()

                accum_loss += float(raw_loss.detach().item()) / grad_accum

                if micro_seen % grad_accum == 0:
                    if use_amp:
                        prev_scale = scaler.get_scale()
                        scaler.unscale_(optimizer)
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item())
                        scaler.step(optimizer)
                        scaler.update()
                        step_skipped = scaler.get_scale() < prev_scale
                    else:
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item())
                        require(math.isfinite(grad_norm), 'non-finite gradient norm in fp32')
                        optimizer.step()
                        step_skipped = False

                    optimizer.zero_grad(set_to_none=True)
                    global_opt_step += 1
                    if step_skipped:
                        skipped_steps += 1
                    else:
                        scheduler.step()
                        ep_grad_norms.append(grad_norm)

                    require(math.isfinite(accum_loss), 'non-finite accumulated loss')
                    ep_losses.append(accum_loss)

                    if not first_finite_logged and not step_skipped and math.isfinite(grad_norm):
                        params_ok = all(bool(torch.isfinite(p).all()) for p in trainable_params)
                        require(params_ok, 'non-finite trainable parameters after first update')
                        first_rec = dict(
                            event='first_finite_update',
                            mode=mode,
                            seed=seed,
                            epoch=epoch,
                            opt_step=global_opt_step,
                            loss=accum_loss,
                            grad_norm=grad_norm,
                            lr=float(optimizer.param_groups[0]['lr']),
                            parameter_finite=params_ok,
                        )
                        print(json.dumps(first_rec), flush=True)
                        traj_file.write(json.dumps(first_rec) + '\n')
                        traj_file.flush()
                        first_finite_logged = True

                    accum_loss = 0.0

            require(micro_seen == micro_batches_per_epoch,
                    f'incomplete epoch micro-batches: {micro_seen} != {micro_batches_per_epoch}')
            ep_rec = dict(
                event='epoch_complete',
                mode=mode,
                seed=seed,
                epoch=epoch,
                opt_step=global_opt_step,
                mean_weighted_mse_loss=float(sum(ep_losses) / len(ep_losses)),
                final_step_loss=float(ep_losses[-1]),
                mean_grad_norm=float(sum(ep_grad_norms) / max(len(ep_grad_norms), 1)),
                skipped_amp_steps=skipped_steps,
                lr=float(optimizer.param_groups[0]['lr']),
                wall_seconds=time.monotonic() - ep_start,
            )
            print(json.dumps(ep_rec), flush=True)
            traj_file.write(json.dumps(ep_rec) + '\n')
            traj_file.flush()

            # Ephemeral recovery state on /tmp only (overwritten each epoch, deleted after eval).
            torch.save(
                dict(epoch=epoch, opt_step=global_opt_step,
                     trainable_state={k: v.cpu() for k, v in model.state_dict().items()
                                      if ('lora_' in k or k.startswith('decoder.'))}),
                scratch_ckpt_dir / 'latest_adapter.pt',
            )

    require(first_finite_logged and global_opt_step == total_iters,
            f'training incomplete: first_finite={first_finite_logged}, steps={global_opt_step}/{total_iters}')
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            'non-finite model parameters at end of 15-epoch training')

    # Calibrate training-only thresholds on centered FIT1192 (zero held-target access)
    cal_summary = calibrate_training_thresholds(
        model,
        hemit_root,
        fit1192_names,
        device=device,
        tile_size=tile_size,
        crop_size=crop_size,
        eval_batch_size=eval_batch_size,
        use_amp=use_amp,
    )
    print(json.dumps(dict(
        event='thresholds_calibrated',
        mode=mode,
        seed=seed,
        train_thresholds=cal_summary['train_thresholds'],
        mean_train_ODS_F=cal_summary['mean_train_ODS_F'],
    )), flush=True)

    quality = old_eval.quality_for(cfg)
    populations_out = {}
    total_held_reads = 0
    for split_name, split_names in held_splits:
        pop_summary, reads = score_held_population(
            model,
            hemit_root,
            split_name,
            split_names,
            cal_summary['train_thresholds'],
            quality=quality,
            device=device,
            tile_size=tile_size,
            crop_size=crop_size,
            eval_batch_size=eval_batch_size,
            use_amp=use_amp,
            per_image_path=output_dir / f'{split_name}-per-image.jsonl.gz',
        )
        populations_out[split_name] = pop_summary
        total_held_reads += reads
        m = pop_summary['metrics']
        print(json.dumps(dict(
            event='held_population_scored',
            mode=mode,
            seed=seed,
            split=split_name,
            n_images=pop_summary['n_images'],
            ODS_F=m['ODS_F'],
            panCK_ODS=pop_summary['per_marker']['panCK']['ODS_F'],
            Mean_AP=m['AP'],
            Cal_F1=m['train_tolerant_F'],
            Fixed_tol_F1=m['cutoff_tolerant_F'],
            SSIM=m['avg_ssim'],
            PSNR=m['avg_psnr'],
        )), flush=True)

    # Clean ephemeral /tmp adapter checkpoint now that scoring succeeded
    if scratch_ckpt_dir.exists():
        shutil.rmtree(scratch_ckpt_dir)

    peak_alloc_mb = (
        float(torch.cuda.max_memory_allocated(device)) / 1e6 if device.type == 'cuda' else 0.0
    )
    peak_res_mb = (
        float(torch.cuda.max_memory_reserved(device)) / 1e6 if device.type == 'cuda' else 0.0
    )

    summary = dict(
        schema=SCHEMA,
        status='complete',
        run_id=cfg['run_id'],
        mode=mode,
        seed=seed,
        locked_test_access=False,
        outer8_access=False,
        checkpoint_selection=False,
        recipe=dict(
            encoder='bioptimus/H-optimus-0' if not synthetic else 'synthetic_vit',
            decoder='Detail_Capture',
            lora_rank=int(cfg.get('lora_rank', 8)),
            lora_alpha=float(cfg.get('lora_alpha', 1.0)),
            epochs=epochs,
            effective_batch_size=effective_batch,
            micro_batch=micro_batch,
            grad_accum=grad_accum,
            lr_g=lr_g,
            lambda_factor=lambda_factor,
            steps_per_epoch=steps_per_epoch,
            total_optimizer_steps=total_iters,
            warmup_iters=warmup_iters,
            decay_start_iter=decay_start_iter,
            precision='16-mixed' if use_amp else 'float32',
            total_params=total_params,
            trainable_params=trainable_numel,
        ),
        channel_stats=stats,
        calibration=cal_summary,
        actual_held_target_reads=total_held_reads,
        populations=populations_out,
        table1_rows={
            sp: dict(
                ODS_F=populations_out[sp]['metrics']['ODS_F'],
                panCK_ODS=populations_out[sp]['per_marker']['panCK']['ODS_F'],
                Mean_AP=populations_out[sp]['metrics']['AP'],
                Cal_F1=populations_out[sp]['metrics']['train_tolerant_F'],
                Fixed_tol_F1=populations_out[sp]['metrics']['cutoff_tolerant_F'],
                SSIM=populations_out[sp]['metrics']['avg_ssim'],
                PSNR_dB=populations_out[sp]['metrics']['avg_psnr'],
                RGB_PSNR_dB=populations_out[sp]['metrics']['rgb_psnr'],
            )
            for sp in populations_out
        },
        resources=dict(
            wall_seconds=time.monotonic() - started,
            peak_cuda_allocated_mb=peak_alloc_mb,
            peak_cuda_reserved_mb=peak_res_mb,
            scratch_cleaned=not scratch_ckpt_dir.exists(),
        ),
        quality_metadata=quality.metadata,
    )
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (output_dir / 'exit-code.txt').write_text('0\n')
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)

    cfg = read_json(args.config)
    cpu_threads = int(cfg.get('cpu_threads', 2))
    require(cpu_threads in (1, 2, 4), 'cpu_threads must be 1, 2, or 4')
    torch.set_num_threads(cpu_threads)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    cv2.setNumThreads(1)

    out_dir = Path(args.output).resolve()
    try:
        run_one(cfg, mode=args.mode, seed=args.seed, output_dir=out_dir, device_str=args.device)
    except Exception:
        err = traceback.format_exc()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'failure.txt').write_text(err)
        (out_dir / 'exit-code.txt').write_text('1\n')
        print(err, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
