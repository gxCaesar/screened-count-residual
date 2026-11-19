"""Shared S2 scoring: per-marker PSNR / SSIM / Pearson R and RGB PSNR on uint8 images.

Label channel order follows the HEMIT README: R = DAPI, G = panCK, B = CD3. Must run with the DTR
checkout on sys.path (uses DTR's metrics.PSNR and metrics.SSIM so numbers match the released
protocol).
"""
import numpy as np
import torch
from metrics import PSNR, SSIM

MARKERS = ["DAPI", "panCK", "CD3"]
_psnr = PSNR()


def to_u8(t):
    """Tensor in [-1, 1], shape (3, H, W) -> uint8 (H, W, 3), as DTR's predict.py saves images."""
    return (((t + 1) / 2).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def psnr_ch(x, y):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    return float(20 * np.log10(255.0 / (np.sqrt(mse) + 1e-8)))


def pearson(x, y):
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def score(pred, gt):
    r = {"rgb_psnr": float(_psnr(torch.from_numpy(np.ascontiguousarray(pred)).float(),
                                 torch.from_numpy(np.ascontiguousarray(gt)).float()))}
    for c, m in enumerate(MARKERS):
        r[f"{m}_psnr"] = psnr_ch(pred[..., c], gt[..., c])
        r[f"{m}_ssim"] = float(SSIM._ssim(pred[..., c], gt[..., c]))
        r[f"{m}_r"] = pearson(pred[..., c], gt[..., c])
    r["avg_psnr"] = float(np.mean([r[f"{m}_psnr"] for m in MARKERS]))
    r["avg_ssim"] = float(np.mean([r[f"{m}_ssim"] for m in MARKERS]))
    r["avg_r"] = float(np.nanmean([r[f"{m}_r"] for m in MARKERS]))
    return r


def self_test():
    g = np.random.default_rng(0).integers(0, 256, (64, 64, 3)).astype(np.uint8)
    s = score(g, g)
    assert s["rgb_psnr"] > 150 and abs(s["avg_ssim"] - 1) < 1e-6 and abs(s["avg_r"] - 1) < 1e-9
    noisy = np.clip(g.astype(int) + 20, 0, 255).astype(np.uint8)
    assert score(noisy, g)["rgb_psnr"] < s["rgb_psnr"]
    return True
