"""Fixed endpoint-normalized pooled-count objective for FIT1192 development.

Counts are equal-image densities in K/U/V order.  Threshold chunks are
checkpointed, so differentiation never retains 255 full spatial fields.
The detached lagged table is deliberately independent of model/data loading.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import math

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .query_relation_scoring import tolerance_kernel


THRESHOLDS = 255
COUNT_FIELDS = ("K", "U", "V")
TABLE_SCHEMA = "query-relation-lagged-counts-v1"


@lru_cache(maxsize=1)
def _offsets():
    kernel = tolerance_kernel()
    cy, cx = (length // 2 for length in kernel.shape)
    return tuple((int(y)-cy, int(x)-cx) for y, x in zip(*kernel.nonzero()))


def _slices(height, width, dy, dx):
    y0, y1 = max(0, -dy), min(height, height-dy)
    x0, x1 = max(0, -dx), min(width, width-dx)
    if y0 >= y1 or x0 >= x1:
        return None
    return ((..., slice(y0, y1), slice(x0, x1)),
            (..., slice(y0+dy, y1+dy), slice(x0+dx, x1+dx)))


class _FootprintMaximum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        # Omitting exterior entries is exactly a -inf exterior, including
        # when all legal entries are zero.  Only two spatial fields are saved.
        maximum = torch.full_like(value, -math.inf)
        height, width = value.shape[-2:]
        for dy, dx in _offsets():
            slices = _slices(height, width, dy, dx)
            if slices is not None:
                output, source = slices
                maximum[output] = torch.maximum(maximum[output], value[source])
        ctx.save_for_backward(value, maximum)
        return maximum

    @staticmethod
    def backward(ctx, gradient):
        value, maximum = ctx.saved_tensors
        height, width = value.shape[-2:]
        ties = torch.zeros_like(value)
        for dy, dx in _offsets():
            slices = _slices(height, width, dy, dx)
            if slices is not None:
                output, source = slices
                ties[output] += value[source] == maximum[output]
        shared = gradient / ties
        result = torch.zeros_like(value)
        for dy, dx in _offsets():
            slices = _slices(height, width, dy, dx)
            if slices is not None:
                output, source = slices
                result[source] += shared[output] * (value[source] == maximum[output])
        return result


def prepare(z: Tensor, target_u8: Tensor) -> dict[str, Tensor]:
    """Prepare byte-unit prediction, native spatial max and target masks.

    The 255 byte conversion remains in the graph.  Ground truth is the fixed
    strict panCK endpoint ``target_u8 > 22``; exterior pixels never participate.
    """
    if (not isinstance(z, Tensor) or z.ndim != 4 or z.shape[1] != 1
            or min(z.shape) < 1 or z.dtype not in (torch.float32, torch.float64)):
        raise ValueError("z must be nonempty N1HW float32 or float64")
    if not bool(torch.isfinite(z).all()):
        raise ValueError("z must be finite")
    if (not isinstance(target_u8, Tensor) or target_u8.shape != z.shape
            or target_u8.dtype != torch.uint8 or target_u8.device != z.device):
        raise ValueError("target must be matching uint8 on the prediction device")
    a = 255 * z.clamp(0, 1)
    g = target_u8 > 22
    near = _FootprintMaximum.apply(g.to(z.dtype)) > 0
    return dict(a=a, maximum=_FootprintMaximum.apply(a), g=g, near=near,
                G=g.to(torch.float64).mean(dim=(1, 2, 3)))


def _thresholds(values, device):
    if values is None:
        return torch.arange(THRESHOLDS, device=device)
    values = torch.as_tensor(values, device=device)
    if (values.ndim != 1 or not values.numel() or values.dtype == torch.bool
            or values.is_complex() or not bool(torch.isfinite(values).all())
            or bool(((values < 0) | (values > 255)).any())
            or bool((values != values.to(torch.int64)).any())):
        raise ValueError("thresholds must be a nonempty integer vector in [0,255]")
    return values.to(torch.int64)


def _q(value, thresholds):
    value = value.to(torch.float64)
    thresholds = thresholds.to(torch.float64)[None, :, None, None]
    low = torch.sigmoid(-thresholds-0.5)
    span = torch.sigmoid(255-thresholds-0.5)-low
    q = (torch.sigmoid(value-thresholds-0.5)-low) / span
    return torch.where(thresholds < 255, q, torch.zeros_like(q))


def _chunk(a, maximum, g, near, thresholds):
    q, qm = _q(a, thresholds), _q(maximum, thresholds)
    return torch.stack((q.mean(dim=(-2, -1)),
                        (near*q).mean(dim=(-2, -1)),
                        (g*qm).mean(dim=(-2, -1))), dim=-1)


def densities(prepared: Mapping[str, Tensor], thresholds=None,
              chunk_size: int = 16) -> Tensor:
    """Return N x T x 3 double densities without retaining full sweep maps.

    The default sweep is 0..254.  Explicit threshold 255 is supported as the
    fixed all-zero endpoint.  Custom threshold order is preserved.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    a, maximum, g, near = (prepared[key] for key in ("a", "maximum", "g", "near"))
    values = _thresholds(thresholds, a.device)
    rows = []
    for part in values.split(chunk_size):
        if torch.is_grad_enabled() and (a.requires_grad or maximum.requires_grad):
            row = checkpoint(_chunk, a, maximum, g, near, part, use_reentrant=False)
        else:
            row = _chunk(a, maximum, g, near, part)
        rows.append(row)
    return torch.cat(rows, dim=1)


def _counts(value, shape, name):
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    value = value.to(dtype=torch.float64)
    if (not bool(torch.isfinite(value).all())
            or bool(((value < 0) | (value > 1)).any())):
        raise ValueError(f"{name} must contain finite densities in [0,1]")
    return value


def pool(mean_counts: Tensor, mean_G: Tensor | float) -> dict:
    """Pool equal-image means and select the smallest maximizing threshold.

    Phi includes the frozen factor 1/3 exactly once.  The returned adjoint is
    dPhi/d(K,U,V) at the selected row; the caller detaches it for a second pass.
    Neither the adjoint nor a single-image surrogate includes a factor 1/N.
    """
    counts = _counts(mean_counts, (THRESHOLDS, 3), "mean_counts")
    G = torch.as_tensor(mean_G, dtype=torch.float64, device=counts.device)
    if G.ndim or not bool(torch.isfinite(G)) or not 0 <= float(G.detach()) <= 1:
        raise ValueError("mean_G must be a finite scalar density in [0,1]")
    K, U, V = counts.unbind(dim=1)
    if bool((U > K+1e-12).any()) or bool((V > G+1e-12).any()):
        raise ValueError("counts violate U<=K or V<=G")
    denominator = U*G + V*K
    valid = (G > 0) & (denominator > 0)
    safe = torch.where(valid, denominator, torch.ones_like(denominator))
    scores = torch.where(valid, 2*U*V/safe, torch.zeros_like(denominator))
    peak, index = scores.max(dim=0)
    threshold = int(index.detach())
    k, u, v = counts[threshold]
    d = safe[threshold]
    adjoint = torch.stack((2*u*v*v, -2*v*v*k, -2*u*u*G)) / (3*d*d)
    adjoint = torch.where(valid[threshold], adjoint, torch.zeros_like(adjoint))
    phi = (1-peak)/3
    finite = bool(torch.isfinite(scores).all() & torch.isfinite(phi)
                  & torch.isfinite(adjoint).all())
    if not finite:
        raise ValueError("pooled objective or adjoint is nonfinite")
    diagnostics = dict(finite=finite, zero_denominator_count=int((denominator == 0).sum()),
                       mean_G=float(G.detach()), maximum_score=float(peak.detach()))
    return dict(phi=phi, scores=scores, threshold=threshold, adjoint=adjoint,
                finite=finite, diagnostics=diagnostics)


class LaggedCounts:
    """CPU float64 current row table; FIT1192 rows occupy 7.30 MB."""

    def __init__(self, n_images: int):
        if type(n_images) is not int or n_images < 1:
            raise ValueError("n_images must be a positive integer")
        self.n_images = n_images
        self.rows = torch.zeros(n_images, THRESHOLDS, 3, dtype=torch.float64)
        self.G = torch.zeros(n_images, dtype=torch.float64)
        self.initialized = False

    def refresh(self, rows: Tensor, G: Tensor) -> None:
        rows = _counts(rows, (self.n_images, THRESHOLDS, 3), "rows")
        G = _counts(G, (self.n_images,), "G")
        if bool((rows[..., 1] > rows[..., 0]+1e-12).any()
                | (rows[..., 2] > G[:, None]+1e-12).any()):
            raise ValueError("rows violate U<=K or V<=G")
        self.rows = rows.detach().to(device="cpu").clone()
        self.G = G.detach().to(device="cpu").clone()
        self.initialized = True

    def replace(self, index: int, row: Tensor) -> None:
        if not self.initialized:
            raise ValueError("refresh is required before row replacement")
        if type(index) is not int or not 0 <= index < self.n_images:
            raise ValueError("index is outside the lagged table")
        row = _counts(row, (THRESHOLDS, 3), "row").detach().to(device="cpu")
        if bool((row[:, 1] > row[:, 0]+1e-12).any()
                | (row[:, 2] > self.G[index]+1e-12).any()):
            raise ValueError("row violates U<=K or V<=G")
        self.rows[index].copy_(row)

    def statistics(self) -> dict:
        if not self.initialized:
            raise ValueError("refresh is required before pooling the lagged table")
        return pool(self.rows.mean(dim=0), self.G.mean())

    def state_dict(self) -> dict:
        if not self.initialized:
            raise ValueError("refresh is required before saving the lagged table")
        return dict(schema=TABLE_SCHEMA, n_images=self.n_images,
                    rows=self.rows.clone(), G=self.G.clone())

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "LaggedCounts":
        if (not isinstance(state, Mapping)
                or set(state) != {"schema", "n_images", "rows", "G"}
                or state["schema"] != TABLE_SCHEMA):
            raise ValueError("malformed lagged-count state/schema")
        result = cls(state["n_images"])
        result.refresh(state["rows"], state["G"])
        return result


def calibration_lambda(reconstruction_sq, auxiliary_sq) -> float:
    """Fixed sqrt(sum reconstruction-gradient squared / sum auxiliary squared)."""
    values = []
    for value in (reconstruction_sq, auxiliary_sq):
        value = torch.as_tensor(value, dtype=torch.float64)
        if value.ndim or not bool(torch.isfinite(value)) or float(value) <= 0:
            raise ValueError("calibration requires positive finite squared gradient norms")
        values.append(float(value))
    # Separate square roots avoid overflowing an otherwise finite norm ratio.
    result = math.sqrt(values[0]) / math.sqrt(values[1])
    if not math.isfinite(result) or result <= 0:
        raise ValueError("calibration lambda must be positive and finite")
    return result
