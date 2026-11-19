"""Architecture-agnostic plug-and-play Screened Count Residual head.

Attaches onto any frozen C-channel virtual-staining backbone prediction
``b(x) in [0, 1]^(C x H x W)`` paired with normalized H&E input
``x in [0, 1]^(3 x H x W)``, solving the gauge-fixed screened Poisson equation:

    r_c = (I + L)^{-1} L psi_c(x, b(x)),   1^T r_c == 0

under either:
  * ``S_recon``: seed-paired pre-clip MSE + normalized SSIM + multi-offset edge
                 reconstruction control (identical architecture, init, and schedule).
  * ``S_count``: exact same normalized reconstruction objective plus detached-table
                 pooled spatial-tolerance count calibration across both the dynamic
                 ODS threshold ``t*`` and the physical deployment threshold ``tau_c``
                 with automatic output-space gradient-norm balance ``calibration_lambda``.

Reuses ``query_relation_reconstruction`` (``RelationModel('S_C')``, ``screened_solve``,
``loss_scales``, ``normalize_loss``) and ``query_relation_count_objective``
(``_FootprintMaximum``, ``densities``, ``LaggedCounts``, ``calibration_lambda``)
without modifying any frozen CVPR module.
"""
from __future__ import annotations

import math
import random
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from . import query_relation_count_objective as counts
from . import query_relation_reconstruction as core

UNIVERSAL_SCHEMA = "screened-universal-head-v5"
DEFAULT_SEEDS = (202610270, 202610271)
ARMS = ("S_recon", "S_count")


def fast_footprint_max(x: Tensor) -> Tensor:
    """Exact 81-point chamfer-5 tolerance footprint maximum via 4 centered max_pool2d boxes.

    Bit-for-bit identical (0.0 max abs diff) to ``counts._FootprintMaximum`` over the
    81-pixel L2/5 disk (rows |dy|=0: |dx|<=5; |dy|<=3: |dx|<=4; |dy|<=4: |dx|<=3; |dy|<=5: |dx|==0)
    while executing in 4 native C++/CUDA kernels instead of 81 Python slice loops.
    """
    m1 = F.max_pool2d(F.pad(x, (5, 5, 0, 0), value=-math.inf), (1, 11), stride=1)
    m2 = F.max_pool2d(F.pad(x, (4, 4, 3, 3), value=-math.inf), (7, 9), stride=1)
    m3 = F.max_pool2d(F.pad(x, (3, 3, 4, 4), value=-math.inf), (9, 7), stride=1)
    m4 = F.max_pool2d(F.pad(x, (0, 0, 5, 5), value=-math.inf), (11, 1), stride=1)
    return torch.maximum(torch.maximum(m1, m2), torch.maximum(m3, m4))


_VALID8_CACHE: dict[tuple[torch.device, torch.dtype, int, int], Tensor] = {}


def _get_valid8_flags(device: torch.device, dtype: torch.dtype, height: int, width: int) -> Tensor:
    key = (device, dtype, int(height), int(width))
    cached = _VALID8_CACHE.get(key)
    if cached is not None:
        return cached
    flags = []
    for step in core.OFFSETS:
        for dy, dx in ((0, step), (0, -step), (step, 0), (-step, 0)):
            valid = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            if y1 > y0 and x1 > x0:
                valid[:, :, y0:y1, x0:x1] = 1.0
            flags.append(valid)
    out = torch.cat(flags, dim=1)
    _VALID8_CACHE[key] = out
    return out


def fast_unary_potential(model: core.RelationModel, h: Tensor) -> Tensor:
    """Fused contiguous 3x3 + dilated-16 3x3 conv decomposition of ``RelationModel.unary_potential``."""
    if model.unary_head is None:
        raise ValueError("fast_unary_potential requires a unary RelationModel arm")
    _, _, height, width = h.shape
    w0 = model.unary_head[0].weight  # (64, 296)
    b0 = model.unary_head[0].bias    # (64,)
    w2 = model.unary_head[2].weight  # (1, 64)
    b2 = model.unary_head[2].bias    # (1,)

    # Pack center + 4 step-1 neighbors into one (64, 32, 3, 3) conv kernel:
    # idx 0: (0, +1) -> [1, 2]; idx 1: (0, -1) -> [1, 0]; idx 2: (+1, 0) -> [2, 1]; idx 3: (-1, 0) -> [0, 1]
    k1 = w0.new_zeros((64, 32, 3, 3))
    k1[:, :, 1, 1] = w0[:, :32]
    k1[:, :, 1, 2] = w0[:, 32:64]
    k1[:, :, 1, 0] = w0[:, 64:96]
    k1[:, :, 2, 1] = w0[:, 96:128]
    k1[:, :, 0, 1] = w0[:, 128:160]

    # Pack 4 step-16 neighbors into one (64, 32, 3, 3) dilated-16 conv kernel:
    # idx 4: (0, +16) -> [1, 2]; idx 5: (0, -16) -> [1, 0]; idx 6: (+16, 0) -> [2, 1]; idx 7: (-16, 0) -> [0, 1]
    k16 = w0.new_zeros((64, 32, 3, 3))
    k16[:, :, 1, 2] = w0[:, 160:192]
    k16[:, :, 1, 0] = w0[:, 192:224]
    k16[:, :, 2, 1] = w0[:, 224:256]
    k16[:, :, 0, 1] = w0[:, 256:288]

    valid8 = _get_valid8_flags(h.device, h.dtype, height, width)
    acc = (
        F.conv2d(h, k1, b0, padding=1)
        + F.conv2d(h, k16, padding=16, dilation=16)
        + F.conv2d(valid8, w0[:, 288:, None, None])
    )
    return F.conv2d(F.silu(acc), w2[:, :, None, None], b2)


def _fast_laplacian_1_16(x: Tensor) -> Tensor:
    """Allocation-lean exact matrix-free Lx = B^T B x (scale=0.25, offsets=(1, 16))."""
    height, width = x.shape[-2:]
    out = torch.zeros_like(x)
    for step in (1, 16):
        if width > step:
            dr = (x[..., :, step:] - x[..., :, :-step]) * 0.0625
            out[..., :, :-step] -= dr
            out[..., :, step:] += dr
        if height > step:
            dd = (x[..., step:, :] - x[..., :-step, :]) * 0.0625
            out[..., :-step, :] -= dd
            out[..., step:, :] += dd
    return out


class _FastScreenedSolveTrain(torch.autograd.Function):
    """Sync-free 10-iteration CG solver for (I + L) x = rhs in float32 training.

    Because kappa(I + L) <= 2.0 on the frozen (1, 16) graph, 10 CG iterations
    reduce relative residual below 6.8e-8 (<< 1e-6 tolerance) with zero CPU-GPU
    stream synchronizations. Symmetric implicit backward uses the identical solve.
    """

    @staticmethod
    def _cg_steps(rhs: Tensor, n_iter: int = 10, init_sol: Tensor | None = None) -> Tensor:
        if init_sol is None:
            sol = torch.zeros_like(rhs)
            res = rhs.clone()
        else:
            sol = init_sol.clone()
            res = rhs - (sol + _fast_laplacian_1_16(sol))
        direction = res.clone()
        rr = res.square().sum(dim=(-2, -1), keepdim=True)
        tiny = torch.finfo(rhs.dtype).tiny
        for _ in range(n_iter):
            op = direction + _fast_laplacian_1_16(direction)
            denom = (direction * op).sum(dim=(-2, -1), keepdim=True).clamp_min(tiny)
            alpha = rr / denom
            sol = sol + alpha * direction
            res = res - alpha * op
            new_rr = res.square().sum(dim=(-2, -1), keepdim=True)
            beta = new_rr / rr.clamp_min(tiny)
            direction = res + beta * direction
            rr = new_rr
        return sol

    @staticmethod
    def forward(ctx, rhs: Tensor) -> Tensor:
        ctx.input_dtype = rhs.dtype
        return _FastScreenedSolveTrain._cg_steps(rhs.to(torch.float32), n_iter=10).to(rhs.dtype)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return _FastScreenedSolveTrain._cg_steps(gradient.to(torch.float32), n_iter=10).to(ctx.input_dtype)


def fast_sc_forward(
    model: core.RelationModel,
    common: Tensor,
    inference: bool = False,
) -> tuple[Tensor, dict]:
    """Fast mathematically equivalent forward pass for coordinate-only ``S_C`` RelationModel."""
    dtype = model.token_projection.weight.dtype
    c = common.to(dtype=dtype)
    zeros16 = model.token_projection.weight.new_zeros((c.shape[0], 16, *c.shape[-2:]))
    h = model.backbone(torch.cat((c, zeros16), dim=1))
    potential = fast_unary_potential(model, h)
    if inference:
        rhs_f64 = _fast_laplacian_1_16(potential.to(torch.float64))
        rhs_norm = rhs_f64.square().sum(dim=(-2, -1), keepdim=True).sqrt()
        if float(rhs_norm.max().item()) == 0.0:
            residual = torch.zeros_like(rhs_f64)
            rel_max = 0.0
            abs_max = 0.0
        else:
            # Pure scale-invariant float64 CG from zero (22 sync-free steps on kappa(I+L) <= 2.0):
            # Chebyshev bound 2 * ((sqrt(2)-1)/(sqrt(2)+1))^22 < 2.8e-17 << 1e-10 across all scales
            # without float32 warm-start noise on low-amplitude background tiles.
            sol_f64 = _FastScreenedSolveTrain._cg_steps(rhs_f64, n_iter=22)
            sol_f64 = sol_f64 - sol_f64.mean(dim=(-2, -1), keepdim=True)
            sol_f64 = sol_f64 - sol_f64.mean(dim=(-2, -1), keepdim=True)
            true_res = rhs_f64 - (sol_f64 + _fast_laplacian_1_16(sol_f64))
            res_norm = true_res.square().sum(dim=(-2, -1), keepdim=True).sqrt()
            rel_vec = torch.where(rhs_norm > 0.0, res_norm / rhs_norm.clamp_min(1e-30), torch.zeros_like(res_norm))
            rel_max = float(rel_vec.max().item())
            if rel_max > 1e-10:
                sol_f64 = core.screened_solve(rhs_f64, inference=True)
                true_res = rhs_f64 - (sol_f64 + _fast_laplacian_1_16(sol_f64))
                res_norm = true_res.square().sum(dim=(-2, -1), keepdim=True).sqrt()
                rel_vec = torch.where(rhs_norm > 0.0, res_norm / rhs_norm.clamp_min(1e-30), torch.zeros_like(res_norm))
                rel_max = float(rel_vec.max().item())
            abs_max = float(res_norm.max().item())
            residual = sol_f64
        if not bool(torch.isfinite(residual).all()):
            raise RuntimeError("fast_sc_forward produced a nonfinite residual")
        rhs = rhs_f64
        diagnostics = {
            "relative_residual_max": rel_max,
            "absolute_residual_max": abs_max,
            "zero_rhs_count": int((rhs_norm == 0.0).sum().item()),
            "zero_rhs_absolute_residual_max": 0.0,
            "solver_dtype": "float64",
            "rhs": rhs.detach(),
            "residual_mean_max": float(residual.detach().mean(dim=(-2, -1)).abs().max()),
            "arm": model.arm,
        }
    else:
        rhs = _fast_laplacian_1_16(potential).to(torch.float32)
        residual = _FastScreenedSolveTrain.apply(rhs)
        diagnostics = {
            "relative_residual_max": 0.0,
            "absolute_residual_max": 0.0,
            "zero_rhs_count": 0,
            "zero_rhs_absolute_residual_max": 0.0,
            "solver_dtype": "float32",
            "rhs": rhs.detach(),
            "residual_mean_max_tensor": residual.detach().mean(dim=(-2, -1)).abs().max(),
            "arm": model.arm,
        }
    return residual, diagnostics


def build_universal_common12(
    he_rgb01: Tensor,
    donor_c01: Tensor,
    structural_c01: Tensor,
    panel_mean01: Tensor,
) -> Tensor:
    """Build 12-channel coordinate feature tensor (N, 12, H, W) in float32.

    Channels:
      0..2 : H&E RGB in [0, 1]
      3..5 : Multi-scale H&E spatial contrast (|grad_1|, |grad_8|, |laplacian|)
      6    : Upstream target-marker prediction b_c in [0, 1]
      7    : Upstream nuclear/structural reference channel (e.g. DAPI/Hoechst or Hematoxylin OD)
      8    : Upstream panel mean prediction across channels in [0, 1]
      9    : Local step-1 gradient magnitude of b_c
      10   : Local step-16 gradient magnitude of b_c
      11   : Tolerance-footprint local peak contrast max_footprint(b_c) - b_c
    """
    if he_rgb01.ndim != 4 or he_rgb01.shape[1] != 3:
        raise ValueError(f"he_rgb01 must be (N, 3, H, W), got {tuple(he_rgb01.shape)}")
    n, _, h, w = he_rgb01.shape
    for name, t in (
        ("donor_c01", donor_c01),
        ("structural_c01", structural_c01),
        ("panel_mean01", panel_mean01),
    ):
        if t.shape != (n, 1, h, w):
            raise ValueError(f"{name} must have shape {(n, 1, h, w)}, got {tuple(t.shape)}")

    he = he_rgb01.to(torch.float32).clamp(0.0, 1.0)
    bc = donor_c01.to(torch.float32).clamp(0.0, 1.0)
    bs = structural_c01.to(torch.float32).clamp(0.0, 1.0)
    bm = panel_mean01.to(torch.float32).clamp(0.0, 1.0)

    he_gray = 0.299 * he[:, 0:1] + 0.587 * he[:, 1:2] + 0.114 * he[:, 2:3]

    def _grad_mag(field: Tensor, step: int) -> Tensor:
        pad_r = F.pad(field, (0, step, 0, 0), mode="reflect")
        gx = (pad_r[..., :, step:] - field).abs()
        pad_d = F.pad(field, (0, 0, 0, step), mode="reflect")
        gy = (pad_d[..., step:, :] - field).abs()
        return torch.sqrt(gx.square() + gy.square() + 1e-12).clamp(0.0, 1.5)

    he_g1 = _grad_mag(he_gray, 1)
    he_g8 = _grad_mag(he_gray, 8)
    he_lap = core.laplacian(he_gray).abs().clamp(0.0, 1.5)

    with torch.no_grad():
        fmax = fast_footprint_max(bc).clamp(0.0, 1.0)
        bc_scale = fmax.amax(dim=(-2, -1), keepdim=True).clamp_min(2.0 / 255.0)
    bc_rel = (bc / bc_scale).clamp(0.0, 1.0)
    fmax_rel = (fmax / bc_scale).clamp(0.0, 1.0)
    bc_g1 = _grad_mag(bc_rel, 1)
    bc_g16 = _grad_mag(bc_rel, 16)
    bc_halo = (fmax_rel - bc_rel).clamp(0.0, 1.0)

    common12 = torch.cat(
        [he, he_g1, he_g8, he_lap, bc, bs, bm, bc_g1, bc_g16, bc_halo],
        dim=1,
    )
    if not bool(torch.isfinite(common12).all()):
        raise RuntimeError("Non-finite values in build_universal_common12")
    return common12


def apply_d4(x: Tensor, code: int) -> Tensor:
    """Dihedral D4 symmetry transform (code in 0..7) over spatial dims (-2, -1)."""
    if not (isinstance(code, int) and 0 <= code < 8):
        raise ValueError(f"D4 code must be int in 0..7, got {code!r}")
    out = torch.rot90(x, code % 4, dims=(-2, -1))
    if code >= 4:
        out = torch.flip(out, dims=(-1,))
    return out.contiguous()


def apply_d4_batch(x: Tensor, codes: list[int]) -> Tensor:
    """Vectorized batch dihedral D4 transform grouping indices by code 0..7."""
    if len(codes) == 0:
        return x.clone()
    if len(set(codes)) == 1:
        return apply_d4(x, codes[0])
    out = torch.empty_like(x)
    for code in range(8):
        idxs = [i for i, c in enumerate(codes) if c == code]
        if idxs:
            out[idxs] = apply_d4(x[idxs], code)
    return out


_HIST_LUT_CACHE: dict[torch.device, Tensor] = {}


@torch.no_grad()
def fast_densities_hist_nograd(prepared: Mapping[str, Tensor]) -> Tensor:
    """O(H*W) GPU 1025-bin (0.25-byte) histogram + FP64 LUT matmul for 256-threshold LaggedCounts.

    Replaces 256*H*W sigmoid evaluations inside ``no_grad`` population table refreshes with
    3 scatter_add histograms + a single (1025 x 256) FP64 matrix multiply (>500x faster on 512x512).
    """
    a, maximum, g, near = (prepared[key] for key in ("a", "maximum", "g", "near"))
    dev = a.device
    lut = _HIST_LUT_CACHE.get(dev)
    if lut is None:
        bins_f64 = torch.linspace(0.0, 255.0, 1025, device=dev, dtype=torch.float64)[:, None]
        thr_f64 = torch.arange(counts.THRESHOLDS, device=dev, dtype=torch.float64)[None, :]
        low_f64 = torch.sigmoid(-thr_f64 - 0.5)
        span_f64 = (torch.sigmoid(255.0 - thr_f64 - 0.5) - low_f64).clamp_min(1e-12)
        lut = ((torch.sigmoid(bins_f64 - thr_f64 - 0.5) - low_f64) / span_f64).clamp(0.0, 1.0)
        _HIST_LUT_CACHE[dev] = lut

    bsz = a.shape[0]
    npix = a[0].numel()
    bin_a = torch.round(a.reshape(bsz, npix).clamp(0.0, 255.0) * (1024.0 / 255.0)).to(torch.long)
    bin_m = torch.round(maximum.reshape(bsz, npix).clamp(0.0, 255.0) * (1024.0 / 255.0)).to(torch.long)
    ones = torch.ones((bsz, npix), device=dev, dtype=torch.float32)
    near_flat = near.reshape(bsz, npix).to(torch.float32)
    g_flat = g.reshape(bsz, npix).to(torch.float32)

    h_k = torch.zeros((bsz, 1025), device=dev, dtype=torch.float32).scatter_add_(1, bin_a, ones)
    h_u = torch.zeros((bsz, 1025), device=dev, dtype=torch.float32).scatter_add_(1, bin_a, near_flat)
    h_v = torch.zeros((bsz, 1025), device=dev, dtype=torch.float32).scatter_add_(1, bin_m, g_flat)

    inv_p = 1.0 / float(max(1, npix))
    k_row = (h_k.to(torch.float64) * inv_p) @ lut
    u_row = (h_u.to(torch.float64) * inv_p) @ lut
    v_row = (h_v.to(torch.float64) * inv_p) @ lut
    return torch.stack((k_row, u_row, v_row), dim=-1)


def fast_densities(
    prepared: Mapping[str, Tensor],
    thresholds=None,
    chunk_size: int = 64,
) -> Tensor:
    """Fast FP32-sigmoid -> FP64-mean pooled spatial-tolerance densities (N, T, 3).

    Eliminates the 1:64 FP64 hardware throughput penalty on Ampere GPUs and the 4 GiB
    float64 sweep allocation while matching ``counts.densities`` to < 1e-6 across [0, 255]
    and preserving ``0 <= U <= K <= 1`` and ``0 <= V <= G <= 1`` exactly.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    a, maximum, g, near = (prepared[key] for key in ("a", "maximum", "g", "near"))
    values = counts._thresholds(thresholds, a.device)
    a_f32 = a.to(torch.float32)
    m_f32 = maximum.to(torch.float32)
    near_f32 = near.to(torch.float32)
    g_f32 = g.to(torch.float32)

    rows = []
    for part in values.split(chunk_size):
        thr_f64 = part.to(torch.float64)
        low_f64 = torch.sigmoid(-thr_f64 - 0.5)
        span_f64 = (torch.sigmoid(255.0 - thr_f64 - 0.5) - low_f64).clamp_min(1e-12)

        thr_f32 = part.to(torch.float32)[None, :, None, None]
        low_f32 = low_f64.to(torch.float32)[None, :, None, None]
        inv_span_f32 = (1.0 / span_f64).to(torch.float32)[None, :, None, None]
        valid_f32 = (part < 255).to(torch.float32)[None, :, None, None]

        q = ((torch.sigmoid(a_f32 - thr_f32 - 0.5) - low_f32) * inv_span_f32).clamp(0.0, 1.0) * valid_f32
        qm = ((torch.sigmoid(m_f32 - thr_f32 - 0.5) - low_f32) * inv_span_f32).clamp(0.0, 1.0) * valid_f32

        row = torch.stack(
            (
                q.mean(dim=(-2, -1), dtype=torch.float64),
                (near_f32 * q).mean(dim=(-2, -1), dtype=torch.float64),
                (g_f32 * qm).mean(dim=(-2, -1), dtype=torch.float64),
            ),
            dim=-1,
        )
        rows.append(row)
    return torch.cat(rows, dim=1)


def build_coordinate_screened_model(seed: int, device: torch.device | str) -> core.RelationModel:
    """Construct exact coordinate-only gauge-fixed Screened Residual model ('S_C')."""
    dev = torch.device(device)
    mean = torch.zeros(1536, dtype=torch.float32)
    std = torch.ones(1536, dtype=torch.float32)
    models = core.paired_models(("S_C",), mean, std, int(seed))
    return models["S_C"].to(dev)


def _fast_ssim_f32(pred: Tensor, target: Tensor) -> Tensor:
    """Fast separable 11x11 Gaussian (sigma=1.5) valid-center SSIM in float32."""
    x = pred.to(torch.float32)
    y = target.to(torch.float32)
    coord = torch.arange(-5, 6, dtype=torch.float32, device=x.device)
    g = torch.exp(-coord.square() / (2.0 * 1.5 * 1.5))
    g = g / g.sum()
    gc = g.view(1, 1, 11, 1)
    gr = g.view(1, 1, 1, 11)

    def _sep(z: Tensor) -> Tensor:
        return F.conv2d(F.conv2d(z, gc), gr)

    mu_x, mu_y = _sep(x), _sep(y)
    xx, yy, xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    var_x = (_sep(x.square()) - xx).clamp_min(0.0)
    var_y = (_sep(y.square()) - yy).clamp_min(0.0)
    cov = _sep(x * y) - xy
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    val = ((2.0 * xy + c1) * (2.0 * cov + c2)) / ((xx + yy + c1) * (var_x + var_y + c2)).clamp_min(1e-12)
    return val.mean(dim=(1, 2, 3))


def _fast_loss_scales_f32(donor: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Fast float32 equivalent of ``core.loss_scales`` eliminating 11x11 FP64 conv2d bottlenecks."""
    d = donor.to(torch.float32)
    t = target.to(torch.float32)
    diff = d - t
    res: dict[str, Tensor] = {
        "mse": diff.square().mean(dim=(1, 2, 3)),
        "ssim": 1.0 - _fast_ssim_f32(d, t),
    }
    for step in core.OFFSETS:
        dx = diff[..., :, step:] - diff[..., :, :-step]
        dy = diff[..., step:, :] - diff[..., :-step, :]
        denom = dx[0].numel() + dy[0].numel()
        res[f"edge{step}"] = (
            0.0625 * (dx.square().flatten(1).sum(1) + dy.square().flatten(1).sum(1)) / float(denom)
            if denom > 0
            else d.flatten(1).sum(1) * 0.0
        )
    return res


def preclip_reconstruction_loss(
    z_unclipped: Tensor,
    target: Tensor,
    donor: Tensor,
    scales: Mapping[str, Tensor | float],
) -> Tensor:
    """Pre-clip MSE + clipped normalized SSIM + multi-offset edge objective."""
    clipped = z_unclipped.clamp(0.0, 1.0)
    raw = _fast_loss_scales_f32(clipped, target)
    mse_scale = max(float(scales["mse"]), core.SCALE_FLOOR)
    ssim_scale = max(float(scales["ssim"]), core.SCALE_FLOOR)
    e1_scale = max(float(scales["edge1"]), core.SCALE_FLOOR)
    e16_scale = max(float(scales["edge16"]), core.SCALE_FLOOR)
    preclip_mse = (z_unclipped.to(torch.float32) - target.to(torch.float32)).square().mean(dim=(1, 2, 3)) / mse_scale
    norm_ssim = raw["ssim"] / ssim_scale
    norm_edge = 0.5 * (raw["edge1"] / e1_scale + raw["edge16"] / e16_scale)
    return preclip_mse + norm_ssim + norm_edge


def prepare_channel_counts(z: Tensor, target_u8: Tensor, cutoff: int) -> dict[str, Tensor]:
    """Channel-generalized soft-threshold footprint preparation for strict cutoff in 0..254.

    Matches ``counts.prepare`` (``a = 255 * z.clamp(0, 1)``) so zero-mean screened
    residuals ``1^T r_c == 0`` can place small negative offsets in zero-background
    regions without incurring spurious count gradients below zero, while ``preclip_mse``
    regularizes excursions outside ``[0, 1]``.
    """
    if not (isinstance(cutoff, int) and 0 <= cutoff < 255):
        raise ValueError(f"cutoff must be int in 0..254, got {cutoff!r}")
    a = 255.0 * z.clamp(0.0, 1.0)
    g = target_u8 > cutoff
    near = fast_footprint_max(g.to(z.dtype)) > 0
    return dict(
        a=a,
        maximum=fast_footprint_max(a),
        g=g,
        near=near,
        G=g.to(torch.float64).mean(dim=(1, 2, 3)),
    )


def _soft_step_multiscale_t(value: Tensor, thr_t: Tensor | float | int) -> Tensor:
    """Three-scale normalized soft threshold at byte ``thr_t`` in [0, 1] (FP32).

    Combines boundary precision (temp=1.2) with mid-range (temp=5.0) and wide-capture
    (temp=20.0) tails so severely attenuated upstream predictions (e.g. byte 1..4 vs
    cutoff 18..52) receive smooth, non-vanishing count gradients.
    """
    v = value.to(torch.float32)
    thr = torch.as_tensor(thr_t, device=v.device, dtype=torch.float32).view(1, 1, 1, 1) + 0.5
    acc = torch.zeros_like(v)
    for temp, w in ((1.2, 0.45), (5.0, 0.35), (20.0, 0.20)):
        low = torch.sigmoid((-thr / temp).clamp(min=-60.0))
        high = torch.sigmoid(((255.0 - thr) / temp).clamp(max=60.0))
        span = (high - low).clamp_min(1e-12)
        q = (torch.sigmoid((v - thr) / temp) - low) / span
        acc = acc + w * q.clamp(0.0, 1.0)
    return acc


def _soft_step_multiscale(value: Tensor, threshold: int) -> Tensor:
    """Three-scale normalized soft threshold at integer byte ``threshold`` in [0, 1] (FP32)."""
    return _soft_step_multiscale_t(value, int(threshold))


def deployment_cutoff_densities(prep: Mapping[str, Tensor], cutoff: int) -> Tensor:
    """Return (N, 3) double densities (K_dep, U_dep, V_dep) at physical ``cutoff``."""
    a, maximum, g, near = prep["a"], prep["maximum"], prep["g"], prep["near"]
    q = _soft_step_multiscale_t(a, int(cutoff))
    qm = _soft_step_multiscale_t(maximum, int(cutoff))
    near_f32 = near.to(torch.float32)
    g_f32 = g.to(torch.float32)
    return torch.stack(
        (
            q.mean(dim=(1, 2, 3), dtype=torch.float64),
            (near_f32 * q).mean(dim=(1, 2, 3), dtype=torch.float64),
            (g_f32 * qm).mean(dim=(1, 2, 3), dtype=torch.float64),
        ),
        dim=-1,
    )


def _adjoint_from_single_row_gpu(mean_kuv: Tensor, mean_G: float) -> Tensor:
    """Sync-free GPU 3-vector zero-DC prior-stabilized pooled tolerant F1 adjoint.

    Uses physically realizable uniform activation floor ``q0 = clamp(0.20 * G, 1e-4, 0.05)``
    giving ``(k, u, v) = (K + q0, U + G*q0, V + G*q0)``. When ``K = U = V = 0``,
    ``u - k = -(1 - G)*q0 < 0``, providing a non-vanishing upward gradient on all true
    foreground pixels and a zero-mean spatial force matching ``1^T r_c == 0``.
    """
    row = mean_kuv.to(torch.float64)
    G = float(mean_G)
    if G <= 1e-15:
        return torch.zeros(3, dtype=torch.float64, device=row.device)
    q0 = min(0.05, max(1e-4, 0.20 * G))
    k = row[0] + q0
    u = row[1] + G * q0
    v = row[2] + G * q0
    denom = u * G + v * k
    safe = denom.clamp_min(1e-18)
    adj = torch.stack(
        (
            (2.0 * u * v * v) / (3.0 * safe * safe),
            (-2.0 * v * v * k) / (3.0 * safe * safe),
            (-2.0 * u * u * G) / (3.0 * safe * safe),
        )
    )
    return torch.where(denom > 1e-18, adj, torch.zeros_like(adj))


def _gpu_pool_ods_adjoint(
    lagged_rows_dev: Tensor,
    mean_G: float,
    thr_floor: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sync-free GPU computation of active ODS threshold and zero-DC stabilized 3-vector adjoint."""
    mean_counts = lagged_rows_dev.mean(dim=0)  # (255, 3) float64
    if mean_G <= 1e-15:
        z_thr = torch.tensor(int(thr_floor), dtype=torch.int64, device=lagged_rows_dev.device)
        return z_thr, torch.zeros(3, dtype=torch.float64, device=lagged_rows_dev.device), torch.zeros((), dtype=torch.float64, device=lagged_rows_dev.device)
    K, U, V = mean_counts.unbind(dim=1)
    denom = U * float(mean_G) + V * K
    valid = denom > 0.0
    safe = torch.where(valid, denom, torch.ones_like(denom))
    scores = torch.where(valid, 2.0 * U * V / safe, torch.zeros_like(denom))
    peak, raw_thr_t = scores.max(dim=0)
    floor_t = torch.tensor(int(thr_floor), dtype=torch.int64, device=lagged_rows_dev.device)
    active_thr_t = torch.maximum(raw_thr_t, floor_t)

    row_act = mean_counts.index_select(0, active_thr_t.view(1)).squeeze(0)
    adj_ods = _adjoint_from_single_row_gpu(row_act, mean_G)
    return active_thr_t, adj_ods, peak


def _fast_densities_single_gpu_thr(prepared: Mapping[str, Tensor], thr_scalar_t: Tensor) -> Tensor:
    """Sync-free single-threshold (N, 3) multiscale pooled spatial-tolerance densities."""
    a, maximum, g, near = (prepared[key] for key in ("a", "maximum", "g", "near"))
    valid_f32 = (thr_scalar_t < 255).to(torch.float32).view(1, 1, 1, 1)
    q = _soft_step_multiscale_t(a, thr_scalar_t) * valid_f32
    qm = _soft_step_multiscale_t(maximum, thr_scalar_t) * valid_f32
    near_f32 = near.to(torch.float32)
    g_f32 = g.to(torch.float32)

    return torch.stack(
        (
            q.mean(dim=(1, 2, 3), dtype=torch.float64),
            (near_f32 * q).mean(dim=(1, 2, 3), dtype=torch.float64),
            (g_f32 * qm).mean(dim=(1, 2, 3), dtype=torch.float64),
        ),
        dim=-1,
    )


def adjoint_from_single_row(mean_kuv: Tensor, mean_G: Tensor | float) -> tuple[Tensor, float]:
    """Zero-DC prior-stabilized pooled tolerant F1 adjoint d((1 - F_tol)/3)/d(K, U, V)."""
    row = mean_kuv.to(torch.float64)
    G = float(mean_G)
    if G <= 1e-15:
        return torch.zeros(3, dtype=torch.float64, device=row.device), 0.0
    q0 = min(0.05, max(1e-4, 0.20 * G))
    k = float(row[0]) + q0
    u = float(row[1]) + G * q0
    v = float(row[2]) + G * q0
    denom = u * G + v * k
    if denom <= 1e-18:
        return torch.zeros(3, dtype=torch.float64, device=row.device), 0.0
    raw_k, raw_u, raw_v = float(row[0]), float(row[1]), float(row[2])
    raw_denom = raw_u * G + raw_v * raw_k
    score = (2.0 * raw_u * raw_v / raw_denom) if raw_denom > 1e-18 else 0.0
    adj = torch.tensor(
        [
            (2.0 * u * v * v) / (3.0 * denom * denom),
            (-2.0 * v * v * k) / (3.0 * denom * denom),
            (-2.0 * u * u * G) / (3.0 * denom * denom),
        ],
        dtype=torch.float64,
        device=row.device,
    )
    return adj, float(score)


def estimate_training_scales_and_lambda(
    model: core.RelationModel,
    common_train: Tensor,
    donor_u8_train: Tensor,
    target_u8_train: Tensor,
    cutoff: int,
    device: torch.device,
    calib_subset: int = 96,
    arm: str = "S_count",
) -> tuple[dict[str, float], float, float, counts.LaggedCounts | None, Tensor | None, Tensor]:
    """Compute training-subset loss scales, initial LaggedCounts, deployment table, and decoupled (lambda_ods, lambda_dep)."""
    n = common_train.shape[0]

    scale_sums = {"mse": 0.0, "ssim": 0.0, "edge1": 0.0, "edge16": 0.0}
    rows_list = []
    dep_rows_list = []
    g_list = []

    model.eval()
    hw_max = max(int(common_train.shape[-2]), int(common_train.shape[-1]))
    eval_bs = 8 if hw_max >= 800 else (16 if hw_max >= 512 else 32)
    with torch.no_grad():
        for start in range(0, n, eval_bs):
            stop = min(n, start + eval_bs)
            d_f = donor_u8_train[start:stop, None].to(device=device, dtype=torch.float32) / 255.0
            t_u8 = target_u8_train[start:stop, None].to(device=device)
            t_f = t_u8.to(torch.float32) / 255.0
            sc = _fast_loss_scales_f32(d_f, t_f)
            for k in scale_sums:
                scale_sums[k] += float(sc[k].sum().item())
            if arm == "S_count":
                prep = prepare_channel_counts(d_f, t_u8, cutoff)
                rows_list.append(fast_densities_hist_nograd(prep))
                dep_rows_list.append(deployment_cutoff_densities(prep, cutoff))
                g_list.append(prep["G"])

    scales = {k: max(v / n, core.SCALE_FLOOR) for k, v in scale_sums.items()}
    if arm != "S_count":
        return scales, 0.0, 0.0, None, None, torch.zeros(1, dtype=torch.float64)

    g_vec = torch.cat(g_list, dim=0).cpu()
    mean_G_val = float(g_vec.mean().item())
    dep_table = torch.cat(dep_rows_list, dim=0).to(torch.float64).cpu()
    lagged = counts.LaggedCounts(n)
    all_rows = torch.cat(rows_list, dim=0).cpu()
    lagged.refresh(all_rows, g_vec)

    stats = lagged.statistics()
    sel_thr = int(stats["threshold"])
    thr_floor = min(int(cutoff), max(4, int(round(0.80 * cutoff))))
    calib_thr = max(sel_thr, thr_floor)
    calib_thr_t = torch.tensor(int(calib_thr), dtype=torch.int64, device=device)
    adj_ods_cal, _ = adjoint_from_single_row(all_rows[:, calib_thr, :].mean(dim=0).to(device), mean_G_val)
    adj_dep, _ = adjoint_from_single_row(dep_table.mean(dim=0).to(device), mean_G_val)

    # Output-space gradient norm calibration over up to calib_subset deterministic spaced tiles
    step_idx = np.linspace(0, n - 1, num=min(n, calib_subset), dtype=int)
    recon_sq = 0.0
    aux_ods_sq = 0.0
    aux_dep_sq = 0.0
    grad_bs = 4 if hw_max >= 800 else (8 if hw_max >= 512 else 16)
    for g_start in range(0, len(step_idx), grad_bs):
        sub_idx = [int(x) for x in step_idx[g_start : g_start + grad_bs]]
        d_f = (donor_u8_train[sub_idx, None].to(device=device, dtype=torch.float32) / 255.0).detach().requires_grad_(True)
        t_u8 = target_u8_train[sub_idx, None].to(device=device)
        t_f = t_u8.to(torch.float32) / 255.0
        rec_loss = preclip_reconstruction_loss(d_f, t_f, d_f.detach(), scales).sum()
        (g_rec,) = torch.autograd.grad(rec_loss, d_f, retain_graph=False)
        recon_sq += float(g_rec.detach().to(torch.float64).square().sum().item())

        d_f_ods = (donor_u8_train[sub_idx, None].to(device=device, dtype=torch.float32) / 255.0).detach().requires_grad_(True)
        prep_ods = prepare_channel_counts(d_f_ods, t_u8, cutoff)
        dens_ods = _fast_densities_single_gpu_thr(prep_ods, calib_thr_t)
        loss_ods = (dens_ods * adj_ods_cal[None, :]).sum()
        (g_ods,) = torch.autograd.grad(loss_ods, d_f_ods, retain_graph=False)
        aux_ods_sq += float(g_ods.detach().to(torch.float64).square().sum().item())

        d_f_dep = (donor_u8_train[sub_idx, None].to(device=device, dtype=torch.float32) / 255.0).detach().requires_grad_(True)
        prep_dep = prepare_channel_counts(d_f_dep, t_u8, cutoff)
        dens_dep = deployment_cutoff_densities(prep_dep, cutoff)
        loss_dep = (dens_dep * adj_dep[None, :]).sum()
        (g_dep,) = torch.autograd.grad(loss_dep, d_f_dep, retain_graph=False)
        aux_dep_sq += float(g_dep.detach().to(torch.float64).square().sum().item())

    lam_ods = float(min(max(counts.calibration_lambda(recon_sq, aux_ods_sq), 0.25), 400.0)) if aux_ods_sq > 1e-24 and recon_sq > 0 else 4.0
    lam_dep = float(min(max(counts.calibration_lambda(recon_sq, aux_dep_sq), 0.25), 400.0)) if aux_dep_sq > 1e-24 and recon_sq > 0 else 4.0
    return scales, lam_ods, lam_dep, lagged, dep_table, g_vec


@torch.no_grad()
def _refresh_tables_batched_gpu(
    model: core.RelationModel,
    common_src: Tensor,
    donor_src: Tensor,
    target_src: Tensor,
    src_on_gpu: bool,
    cutoff: int,
    lagged_rows_dev: Tensor,
    dep_table_dev: Tensor,
    device: torch.device,
    eval_bs: int = 32,
) -> None:
    """Sync-free GPU-batched population refresh of lagged_rows_dev and dep_table_dev."""
    was_training = model.training
    model.eval()
    n = common_src.shape[0]
    hw_max = max(int(common_src.shape[-2]), int(common_src.shape[-1]))
    eff_eval_bs = min(int(eval_bs), 8 if hw_max >= 800 else (16 if hw_max >= 512 else 32))
    for s in range(0, n, eff_eval_bs):
        e = min(n, s + eff_eval_bs)
        if src_on_gpu:
            c_b = common_src[s:e].to(torch.float32)
            d_f = donor_src[s:e].to(torch.float32) / 255.0
            t_u8 = target_src[s:e]
        else:
            c_b = common_src[s:e].to(device=device, dtype=torch.float32, non_blocking=True)
            d_f = donor_src[s:e, None].to(device=device, dtype=torch.float32) / 255.0
            t_u8 = target_src[s:e, None].to(device=device)
        res, _ = fast_sc_forward(model, c_b, inference=False)
        z = d_f + res
        prep = prepare_channel_counts(z, t_u8, cutoff)
        lagged_rows_dev[s:e] = fast_densities_hist_nograd(prep)
        dep_table_dev[s:e] = deployment_cutoff_densities(prep, cutoff)
    model.train(was_training)


@torch.no_grad()
def _train_count_composite_gpu(lagged_rows_dev: Tensor, mean_G: float, cutoff: int) -> tuple[float, float, float]:
    """Compute leak-free training-split (composite, crisp_dep_f1, crisp_ods_f1) from lagged_rows_dev."""
    if mean_G <= 1e-15:
        return 0.0, 0.0, 0.0
    mean_c = lagged_rows_dev.mean(dim=0)
    K, U, V = mean_c.unbind(dim=1)
    denom = U * float(mean_G) + V * K
    valid = denom > 0.0
    safe = torch.where(valid, denom, torch.ones_like(denom))
    scores = torch.where(valid, 2.0 * U * V / safe, torch.zeros_like(denom))
    dep_f1 = float(scores[min(max(int(cutoff), 0), scores.shape[0] - 1)].item())
    ods_f1 = float(scores.max(dim=0).values.item())
    return 0.65 * dep_f1 + 0.35 * ods_f1, dep_f1, ods_f1


def _recalibrate_lambdas_live(
    model: core.RelationModel,
    common_src: Tensor,
    donor_src: Tensor,
    target_src: Tensor,
    src_on_gpu: bool,
    scales: Mapping[str, float],
    lagged_rows_dev: Tensor,
    dep_table_dev: Tensor,
    mean_G: float,
    cutoff: int,
    thr_floor: int,
    device: torch.device,
    calib_subset: int = 96,
) -> tuple[float, float]:
    """Fast online output-space gradient-norm recalibration at current z = donor + residual."""
    if mean_G <= 1e-15:
        return 4.0, 4.0
    was_training = model.training
    model.eval()
    n = common_src.shape[0]
    hw_max = max(int(common_src.shape[-2]), int(common_src.shape[-1]))
    grad_bs = 4 if hw_max >= 800 else (8 if hw_max >= 512 else 16)
    step_idx = np.linspace(0, n - 1, num=min(n, calib_subset), dtype=int)

    with torch.no_grad():
        active_thr_t, adj_ods, _ = _gpu_pool_ods_adjoint(lagged_rows_dev, mean_G, thr_floor)
        adj_dep = _adjoint_from_single_row_gpu(dep_table_dev.mean(dim=0), mean_G)

    recon_sq = 0.0
    aux_ods_sq = 0.0
    aux_dep_sq = 0.0
    for g_start in range(0, len(step_idx), grad_bs):
        sub_idx = [int(x) for x in step_idx[g_start : g_start + grad_bs]]
        with torch.no_grad():
            if src_on_gpu:
                idx_dev = torch.tensor(sub_idx, dtype=torch.long, device=device)
                c_b = common_src.index_select(0, idx_dev).to(torch.float32)
                d_f0 = donor_src.index_select(0, idx_dev).to(torch.float32) / 255.0
                t_u8 = target_src.index_select(0, idx_dev)
            else:
                idx_cpu = torch.tensor(sub_idx, dtype=torch.long)
                c_b = common_src.index_select(0, idx_cpu).to(device=device, dtype=torch.float32)
                d_f0 = donor_src.index_select(0, idx_cpu)[:, None].to(device=device, dtype=torch.float32) / 255.0
                t_u8 = target_src.index_select(0, idx_cpu)[:, None].to(device=device)
            res0, _ = fast_sc_forward(model, c_b, inference=False)
            z0 = (d_f0 + res0).detach()
            t_f = t_u8.to(torch.float32) / 255.0

        z_rec = z0.clone().requires_grad_(True)
        rec_loss = preclip_reconstruction_loss(z_rec, t_f, d_f0, scales).sum()
        (g_rec,) = torch.autograd.grad(rec_loss, z_rec, retain_graph=False)
        recon_sq += float(g_rec.detach().to(torch.float64).square().sum().item())

        z_ods = z0.clone().requires_grad_(True)
        prep_ods = prepare_channel_counts(z_ods, t_u8, cutoff)
        dens_ods = _fast_densities_single_gpu_thr(prep_ods, active_thr_t)
        loss_ods = (dens_ods * adj_ods[None, :]).sum()
        (g_ods,) = torch.autograd.grad(loss_ods, z_ods, retain_graph=False)
        aux_ods_sq += float(g_ods.detach().to(torch.float64).square().sum().item())

        z_dep = z0.clone().requires_grad_(True)
        prep_dep = prepare_channel_counts(z_dep, t_u8, cutoff)
        dens_dep = deployment_cutoff_densities(prep_dep, cutoff)
        loss_dep = (dens_dep * adj_dep[None, :]).sum()
        (g_dep,) = torch.autograd.grad(loss_dep, z_dep, retain_graph=False)
        aux_dep_sq += float(g_dep.detach().to(torch.float64).square().sum().item())

    model.train(was_training)
    lam_ods = float(min(max(counts.calibration_lambda(recon_sq, aux_ods_sq), 0.25), 400.0)) if aux_ods_sq > 1e-24 and recon_sq > 0 else 4.0
    lam_dep = float(min(max(counts.calibration_lambda(recon_sq, aux_dep_sq), 0.25), 400.0)) if aux_dep_sq > 1e-24 and recon_sq > 0 else 4.0
    return lam_ods, lam_dep


def fit_universal_marker(
    *,
    arm: str,
    seed: int,
    common_train: Tensor,
    donor_u8_train: Tensor,
    target_u8_train: Tensor,
    cutoff: int,
    device: torch.device | str,
    epochs: int = 10,
    lr: float = 6e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    lambda_scale: float = 1.0,
    deploy_anchor_weight: float = 1.25,
) -> tuple[core.RelationModel, dict]:
    """Fit seed-paired S_recon or S_count universal residual head for one marker channel."""
    if arm not in ARMS:
        raise ValueError(f"arm must be in {ARMS}, got {arm!r}")
    dev = torch.device(device)
    n = common_train.shape[0]
    model = build_coordinate_screened_model(seed, dev)

    scales, base_lambda, dep_lambda, lagged, dep_table, g_vec = estimate_training_scales_and_lambda(
        model, common_train, donor_u8_train, target_u8_train, cutoff, dev, arm=arm
    )
    mean_G_scalar = float(g_vec.mean().item())
    eff_lambda = base_lambda * float(lambda_scale) if arm == "S_count" else 0.0
    eff_dep_lambda = dep_lambda * float(deploy_anchor_weight) * float(lambda_scale) if arm == "S_count" else 0.0

    # Cache dataset in GPU float16 + uint8 when sufficient free VRAM exists (eliminates CPU index_select + PCIe DMA)
    src_on_gpu = False
    common_gpu: Tensor | None = None
    donor_gpu: Tensor | None = None
    target_gpu: Tensor | None = None
    if dev.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(dev)
            needed_bytes = int(common_train.numel()) * 2 + int(donor_u8_train.numel()) * 2 + 2_200_000_000
            if free_bytes > needed_bytes:
                common_gpu = common_train.to(device=dev, dtype=torch.float16)
                donor_gpu = donor_u8_train[:, None].to(device=dev)
                target_gpu = target_u8_train[:, None].to(device=dev)
                src_on_gpu = True
        except Exception:
            src_on_gpu = False

    # Identical RNG seed across S_recon and S_count for strict paired batch/D4 trajectories
    rng = random.Random(int(seed) * 17 + 7)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    total_steps = max(1, epochs * math.ceil(n / batch_size))
    warmup = max(1, total_steps // 10)

    def _lr_mult(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    step = 0
    history = []
    dep_table_dev = dep_table.to(dev) if dep_table is not None else None
    lagged_rows_dev = lagged.rows.to(device=dev, dtype=torch.float64) if (arm == "S_count" and lagged is not None) else None
    thr_floor = min(int(cutoff), max(4, int(round(0.80 * cutoff))))

    best_train_score = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    if arm == "S_count" and lagged_rows_dev is not None and mean_G_scalar > 1e-15:
        best_train_score, _, _ = _train_count_composite_gpu(lagged_rows_dev, mean_G_scalar, cutoff)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for ep in range(epochs):
        if ep > 0 and ep % 2 == 0 and arm == "S_count" and lagged_rows_dev is not None and dep_table_dev is not None:
            c_src = common_gpu if src_on_gpu and common_gpu is not None else common_train
            d_src = donor_gpu if src_on_gpu and donor_gpu is not None else donor_u8_train
            t_src = target_gpu if src_on_gpu and target_gpu is not None else target_u8_train
            _refresh_tables_batched_gpu(
                model,
                c_src,
                d_src,
                t_src,
                src_on_gpu,
                cutoff,
                lagged_rows_dev,
                dep_table_dev,
                dev,
            )
            if mean_G_scalar > 1e-15:
                cur_comp, _, _ = _train_count_composite_gpu(lagged_rows_dev, mean_G_scalar, cutoff)
                if cur_comp >= best_train_score:
                    best_train_score = cur_comp
                    best_epoch = ep
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                live_lam_ods, live_lam_dep = _recalibrate_lambdas_live(
                    model,
                    c_src,
                    d_src,
                    t_src,
                    src_on_gpu,
                    scales,
                    lagged_rows_dev,
                    dep_table_dev,
                    mean_G_scalar,
                    cutoff,
                    thr_floor,
                    dev,
                )
                eff_lambda = live_lam_ods * float(lambda_scale)
                eff_dep_lambda = live_lam_dep * float(deploy_anchor_weight) * float(lambda_scale)
        model.train()
        order = list(range(n))
        rng.shuffle(order)
        ep_rec_t = torch.zeros((), dtype=torch.float64, device=dev)
        ep_aux_t = torch.zeros((), dtype=torch.float64, device=dev)
        ep_res_mean_t = torch.zeros((), dtype=torch.float32, device=dev)
        n_batches = 0

        if src_on_gpu and common_gpu is not None and donor_gpu is not None and target_gpu is not None:
            order_dev = torch.tensor(order, dtype=torch.long, device=dev)
        else:
            order_dev = None

        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            bsz = len(batch_idx)
            codes = [rng.randrange(8) for _ in range(bsz)]
            if src_on_gpu and order_dev is not None and common_gpu is not None and donor_gpu is not None and target_gpu is not None:
                idx_gpu = order_dev[start : start + bsz]
                c_gpu = common_gpu.index_select(0, idx_gpu).to(torch.float32)
                d_gpu = donor_gpu.index_select(0, idx_gpu)
                t_gpu = target_gpu.index_select(0, idx_gpu)
            else:
                idx_t = torch.tensor(batch_idx, dtype=torch.long)
                idx_gpu = idx_t.to(dev)
                c_gpu = torch.index_select(common_train, 0, idx_t).to(device=dev, dtype=torch.float32, non_blocking=True)
                d_gpu = torch.index_select(donor_u8_train, 0, idx_t)[:, None].to(dev, non_blocking=True)
                t_gpu = torch.index_select(target_u8_train, 0, idx_t)[:, None].to(dev, non_blocking=True)

            c_b = apply_d4_batch(c_gpu, codes)
            d_u8 = apply_d4_batch(d_gpu, codes)
            t_u8 = apply_d4_batch(t_gpu, codes)
            d_f = d_u8.to(torch.float32) / 255.0
            t_f = t_u8.to(torch.float32) / 255.0

            mult = _lr_mult(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * mult

            optimizer.zero_grad(set_to_none=True)
            residual, diag = fast_sc_forward(model, c_b, inference=False)
            z_unclipped = d_f + residual
            rec_vec = preclip_reconstruction_loss(z_unclipped, t_f, d_f, scales)
            loss = rec_vec.mean()

            if arm == "S_count" and lagged_rows_dev is not None and dep_table_dev is not None:
                with torch.no_grad():
                    prep_det = prepare_channel_counts(z_unclipped.detach(), t_u8, cutoff)
                    lagged_rows_dev[idx_gpu] = fast_densities_hist_nograd(prep_det)
                    dep_table_dev[idx_gpu] = deployment_cutoff_densities(prep_det, cutoff)
                    active_thr_t, adj_ods, _ = _gpu_pool_ods_adjoint(lagged_rows_dev, mean_G_scalar, thr_floor)
                    adj_dep = _adjoint_from_single_row_gpu(dep_table_dev.mean(dim=0), mean_G_scalar)

                prep = prepare_channel_counts(z_unclipped, t_u8, cutoff)
                dens_ods = _fast_densities_single_gpu_thr(prep, active_thr_t)
                dens_dep = deployment_cutoff_densities(prep, cutoff)

                ods_term = (dens_ods * adj_ods[None, :]).sum(dim=-1).mean()
                dep_term = (dens_dep * adj_dep[None, :]).sum(dim=-1).mean()
                scaled_aux = eff_lambda * ods_term + eff_dep_lambda * dep_term
                loss = loss + scaled_aux.to(loss.dtype)
                ep_aux_t = ep_aux_t + ods_term.detach()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            ep_rec_t = ep_rec_t + rec_vec.detach().mean().to(torch.float64)
            ep_res_mean_t = torch.maximum(ep_res_mean_t, diag["residual_mean_max_tensor"])
            n_batches += 1
            step += 1

        ep_rec = float(ep_rec_t.item())
        ep_aux = float(ep_aux_t.item())
        ep_res_mean = float(ep_res_mean_t.item())
        if not math.isfinite(ep_rec) or not math.isfinite(ep_aux):
            raise RuntimeError(f"Non-finite loss at epoch {ep + 1}: rec={ep_rec}, aux={ep_aux}")

        if arm == "S_count" and lagged_rows_dev is not None and dep_table_dev is not None:
            _, dep_f1_surr = adjoint_from_single_row(dep_table_dev.mean(dim=0), mean_G_scalar)
            mean_c = lagged_rows_dev.mean(dim=0)
            K, U, V = mean_c.unbind(dim=1)
            denom = U * mean_G_scalar + V * K
            valid = (mean_G_scalar > 0.0) & (denom > 0.0)
            scores = torch.where(valid, 2.0 * U * V / torch.where(valid, denom, torch.ones_like(denom)), torch.zeros_like(denom))
            peak_ep, sel_idx_ep = scores.max(dim=0)
            sel_t_ep = int(sel_idx_ep.item())
            ods_surr_ep = float(peak_ep.item())
        else:
            dep_f1_surr = 0.0
            sel_t_ep = int(cutoff)
            ods_surr_ep = 0.0
        history.append(
            dict(
                epoch=ep + 1,
                recon_loss=ep_rec / max(1, n_batches),
                aux_loss=ep_aux / max(1, n_batches),
                max_abs_residual_mean=ep_res_mean,
                selected_threshold=sel_t_ep,
                pooled_ods_surrogate=ods_surr_ep,
                pooled_deploy_surrogate=float(dep_f1_surr),
            )
        )

    if arm == "S_count" and lagged_rows_dev is not None and dep_table_dev is not None and mean_G_scalar > 1e-15:
        c_src = common_gpu if src_on_gpu and common_gpu is not None else common_train
        d_src = donor_gpu if src_on_gpu and donor_gpu is not None else donor_u8_train
        t_src = target_gpu if src_on_gpu and target_gpu is not None else target_u8_train
        _refresh_tables_batched_gpu(
            model,
            c_src,
            d_src,
            t_src,
            src_on_gpu,
            cutoff,
            lagged_rows_dev,
            dep_table_dev,
            dev,
        )
        final_comp, _, _ = _train_count_composite_gpu(lagged_rows_dev, mean_G_scalar, cutoff)
        if final_comp >= best_train_score:
            best_train_score = final_comp
            best_epoch = epochs
        elif best_state is not None:
            model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    del common_gpu, donor_gpu, target_gpu
    model.eval()
    meta = dict(
        schema=UNIVERSAL_SCHEMA,
        arm=arm,
        seed=int(seed),
        cutoff=int(cutoff),
        epochs=int(epochs),
        selected_train_epoch=int(best_epoch if arm == "S_count" else epochs),
        selected_train_composite=float(best_train_score if arm == "S_count" else 0.0),
        batch_size=int(batch_size),
        lr=float(lr),
        scales=scales,
        base_lambda=float(base_lambda),
        effective_lambda=float(eff_lambda),
        effective_deploy_lambda=float(eff_dep_lambda),
        parameter_counts=model.parameter_counts(),
        history=history,
    )
    return model, meta


@torch.no_grad()
def predict_universal_marker(
    model: core.RelationModel,
    common_eval: Tensor,
    donor_u8_eval: Tensor,
    device: torch.device | str,
    batch_size: int = 8,
) -> tuple[np.ndarray, dict]:
    """Predict float64-solved gauge-fixed Screened Residual output in uint8 [0, 255]."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    n = common_eval.shape[0]
    preds_u8 = np.empty_like(donor_u8_eval.cpu().numpy(), dtype=np.uint8)
    max_rel_res = 0.0
    max_zero_mean = 0.0
    hw_max = max(int(common_eval.shape[-2]), int(common_eval.shape[-1]))
    eff_bs = min(int(batch_size), 4 if hw_max >= 800 else (8 if hw_max >= 512 else 32))

    for start in range(0, n, eff_bs):
        stop = min(n, start + eff_bs)
        c_b = common_eval[start:stop].to(dev, non_blocking=True)
        d_f = donor_u8_eval[start:stop, None].to(device=dev, dtype=torch.float64) / 255.0
        residual, diag = fast_sc_forward(model, c_b, inference=True)
        max_rel_res = max(max_rel_res, float(diag["relative_residual_max"]))
        max_zero_mean = max(max_zero_mean, float(diag["residual_mean_max"]))
        pred01 = (d_f + residual.to(torch.float64)).clamp(0.0, 1.0)
        pred_byte = torch.round(pred01[:, 0] * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        preds_u8[start:stop] = pred_byte.cpu().numpy()

    return preds_u8, dict(
        inference_solver_dtype="float64",
        relative_residual_max=max_rel_res,
        residual_mean_max=max_zero_mean,
    )
