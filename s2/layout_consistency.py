"""Registered FM/direct-label occupancy losses in DGR's common target frame.

The extra loss never updates registration, labels or the frozen teacher.
Existing DGR objectives and their registration gradients remain unchanged.
"""
import math

import torch
import torch.nn.functional as F


THRESHOLDS = (22.0, 30.0, 18.0)
BIN_SIZE = 16
TEMPERATURE = 3.0


def _finite_tensor(value, name):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or not torch.isfinite(value).all()):
        raise ValueError(name + ' must be a finite float32 tensor')


def warp_detached_flow(src, flow):
    """DGR grid+flow sampling, with y/x pixel flow detached from the extra loss."""
    _finite_tensor(src, 'source')
    _finite_tensor(flow, 'flow')
    if src.ndim != 4 or src.shape[0] < 1 or min(src.shape[-2:]) < 2:
        raise ValueError('source must have nonempty [B,C,H,W] shape with H,W >= 2')
    batch, _, height, width = src.shape
    if flow.shape != (batch, 2, height, width) or flow.device != src.device:
        raise ValueError('flow shape or device differs from source')
    axes = [torch.arange(size, device=src.device, dtype=torch.float32)
            for size in (height, width)]
    grid = torch.stack(torch.meshgrid(*axes, indexing='ij')).unsqueeze(0)
    locations = grid + flow.detach()
    y = 2 * (locations[:, 0] / (height - 1) - .5)
    x = 2 * (locations[:, 1] / (width - 1) - .5)
    sample_grid = torch.stack((x, y), dim=-1)
    warped = F.grid_sample(src, sample_grid, mode='bilinear',
                           align_corners=True, padding_mode='border')
    valid = ((x >= -1) & (x <= 1) & (y >= -1) & (y <= 1)).unsqueeze(1)
    return warped, valid


def aligned_occupancy_terms(rec, gt, flow, layout, target):
    """Return masked 16-pixel occupancies and the common pixel validity mask."""
    if target not in ('fm', 'direct'):
        raise ValueError('target must be fm or direct')
    _finite_tensor(rec, 'prediction')
    _finite_tensor(gt, 'label')
    if (rec.ndim != 4 or rec.shape[1] != 3 or gt.shape != rec.shape
            or rec.device != gt.device or any(size % BIN_SIZE for size in rec.shape[-2:])):
        raise ValueError('prediction/label shape must agree as [B,3,H,W] with 16-pixel bins')
    if rec.min() < -1 or rec.max() > 1 or gt.min() < -1 or gt.max() > 1:
        raise ValueError('prediction and label must be in [-1,1]')
    warped, valid = warp_detached_flow(rec, flow)
    thresholds = rec.new_tensor(THRESHOLDS).view(1, 3, 1, 1)
    positive = torch.sigmoid(((warped + 1) * 127.5 - thresholds) / TEMPERATURE)
    with torch.no_grad():
        if target == 'fm':
            _finite_tensor(layout, 'layout')
            expected = (rec.shape[0], 3, rec.shape[2] // BIN_SIZE, rec.shape[3] // BIN_SIZE)
            if (layout.shape != expected or layout.device != rec.device
                    or layout.min() < 0 or layout.max() > 1):
                raise ValueError('layout shape/device or [0,1] range differs from prediction')
            expanded = F.interpolate(layout.detach(), size=rec.shape[-2:], mode='nearest')
            target_map, _ = warp_detached_flow(expanded, flow)
        else:
            target_map = (((gt.detach() + 1) * 127.5).clamp(0, 255).round() > thresholds).float()
        target_bins = F.avg_pool2d(target_map * valid, BIN_SIZE, stride=BIN_SIZE)
    prediction_bins = F.avg_pool2d(positive * valid, BIN_SIZE, stride=BIN_SIZE)
    return prediction_bins, target_bins, valid


def aligned_occupancy_loss(rec, gt, flow, layout, target, weight=1.0):
    """C9 normalized absolute counts, equally weighted across all three markers."""
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError('weight must be finite and nonnegative')
    predicted, expected, _ = aligned_occupancy_terms(rec, gt, flow, layout, target)
    axes = (0, 2, 3)
    per_marker = (predicted - expected).abs().sum(axes) / ((predicted + expected).sum(axes) + 1)
    return weight * per_marker.mean()
