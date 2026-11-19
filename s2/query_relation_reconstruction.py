"""Finite query-coordinate residual models for the frozen 2026-10-27 plan.

The graph contains legal right/down edges in offset-major order.  Public graph
operators use B=D/4 by default; ``scale=1`` exposes the unscaled incidence D.
No source files, targets, augmentations or output-byte conversions are read here.
In particular, E_shift receives tokens already shifted *before* the runner's
joint D4 transform.  Shifting inside this module would reverse that order.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


OFFSETS = (1, 16)
ARMS = ("P_F", "M_F", "S_F", "E_F", "P_C", "S_C", "E_C", "E_shift")
SCALE_FLOOR = 1e-8
_CHUNK_PIXELS = 32768


def _offsets(offsets: Sequence[int]) -> tuple[int, ...]:
    result = tuple(offsets)
    if (not result or any(not isinstance(s, int) or isinstance(s, bool) or s < 1
                          for s in result) or len(set(result)) != len(result)):
        raise ValueError("offsets must be distinct positive integers")
    return result


def _field(x: Tensor, name: str) -> None:
    if not isinstance(x, Tensor) or x.ndim < 2 or min(x.shape[-2:]) < 1:
        raise ValueError(f"{name} must have two nonempty spatial dimensions")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")


def graph_gradient(x: Tensor, offsets: Sequence[int] = OFFSETS,
                   scale: float = 0.25) -> tuple[Tensor, ...]:
    """Return Bx as (right, down) legal-edge tensors for each offset.

    Leading dimensions are preserved.  An offset exceeding a spatial dimension
    yields the corresponding empty legal-edge tensor, without periodic wrap.
    """
    _field(x, "x")
    height, width = x.shape[-2:]
    fields = []
    for step in _offsets(offsets):
        w, h = max(0, width - step), max(0, height - step)
        fields.append((x[..., :height, step:step+w] - x[..., :height, :w]) * scale)
        fields.append((x[..., step:step+h, :width] - x[..., :h, :width]) * scale)
    return tuple(fields)


def graph_adjoint(fields: Sequence[Tensor], shape: Sequence[int],
                  offsets: Sequence[int] = OFFSETS,
                  scale: float = 0.25) -> Tensor:
    """Apply B transpose with the same signs and legal edges as graph_gradient.

    ``shape`` may be the complete output shape or just its (height, width).
    The latter infers all leading dimensions from the first edge field.
    """
    offsets, fields, shape = _offsets(offsets), tuple(fields), tuple(shape)
    if len(fields) != 2 * len(offsets):
        raise ValueError("one right/down field pair required per offset")
    if len(shape) < 2 or min(shape[-2:]) < 1:
        raise ValueError("shape must have two nonempty spatial dimensions")
    if len(shape) == 2:
        shape = tuple(fields[0].shape[:-2]) + shape
    result = fields[0].new_zeros(shape)
    height, width = shape[-2:]
    for i, step in enumerate(offsets):
        w, h = max(0, width - step), max(0, height - step)
        right, down = fields[2*i:2*i+2]
        for value, spatial in ((right, (height, w)), (down, (h, width))):
            _field(value, "edge field") if min(spatial) else None
            if (value.shape != (*shape[:-2], *spatial)
                    or value.dtype != result.dtype or value.device != result.device):
                raise ValueError("edge field shape, dtype or device differs from graph")
        if w:
            result[..., :height, :w] -= right * scale
            result[..., :height, step:step+w] += right * scale
        if h:
            result[..., :h, :width] -= down * scale
            result[..., step:step+h, :width] += down * scale
    return result


def laplacian(x: Tensor, offsets: Sequence[int] = OFFSETS) -> Tensor:
    """Matrix-free Lx = B transpose Bx, with B=D/4."""
    return graph_adjoint(graph_gradient(x, offsets), x.shape, offsets)


def _spatial_norm(x: Tensor) -> Tensor:
    return x.square().sum(dim=(-2, -1), keepdim=True).sqrt()


def solve_diagnostics(solution: Tensor, rhs: Tensor,
                      offsets: Sequence[int] = OFFSETS) -> dict:
    """Recompute the actual equation residual, never the CG recurrence proxy."""
    if solution.shape != rhs.shape:
        raise ValueError("solution and rhs shapes must match")
    with torch.no_grad():
        residual = solution + laplacian(solution, offsets) - rhs
        absolute, norm = _spatial_norm(residual), _spatial_norm(rhs)
        relative = torch.where(norm > 0, absolute / norm.clamp_min(torch.finfo(norm.dtype).tiny),
                               torch.zeros_like(norm))
        return {"relative_residual_max": float(relative.max()),
                "absolute_residual_max": float(absolute.max()),
                "zero_rhs_count": int((norm == 0).sum()),
                "zero_rhs_absolute_residual_max": float(
                    torch.where(norm == 0, absolute, torch.zeros_like(absolute)).max()),
                "solver_dtype": str(solution.dtype).removeprefix("torch.")}


def _solve_no_grad(rhs: Tensor, inference: bool,
                   offsets: tuple[int, ...]) -> Tensor:
    """Batched CG; each leading-index image gets its own convergence criterion."""
    dtype, tolerance = (torch.float64, 1e-10) if inference else (torch.float32, 1e-6)
    rhs = rhs.to(dtype=dtype)
    if not bool(torch.isfinite(rhs).all()):
        raise ValueError("screened solve requires a finite rhs")
    solution = torch.zeros_like(rhs)
    rhs_norm = _spatial_norm(rhs)
    if not bool((rhs_norm != 0).any()):
        return solution
    residual = rhs.clone()
    direction = residual.clone()
    rr = residual.square().sum(dim=(-2, -1), keepdim=True)
    tiny = torch.finfo(dtype).tiny
    # Kappa <= 2 for the frozen graph.  Sixty-four iterations also allow a
    # true-residual restart if recurrence roundoff is larger than its proxy.
    for _ in range(64):
        active = _spatial_norm(residual) > tolerance * rhs_norm
        if not bool(active.any()):
            # Match solve_diagnostics' parenthesization exactly.  In float32,
            # rhs - solution - Lx can pass while (solution + Lx) - rhs fails
            # at the fixed tolerance boundary.  Restart using that same true
            # equation residual rather than returning a rejected solution.
            residual = rhs - (solution + laplacian(solution, offsets))
            active = _spatial_norm(residual) > tolerance * rhs_norm
            if not bool(active.any()):
                break
            direction, rr = residual.clone(), residual.square().sum(
                dim=(-2, -1), keepdim=True)
        operated = direction + laplacian(direction, offsets)
        denominator = (direction * operated).sum(dim=(-2, -1), keepdim=True)
        alpha = torch.where(active, rr / denominator.clamp_min(tiny), torch.zeros_like(rr))
        solution = solution + alpha * direction
        residual = residual - alpha * operated
        new_rr = residual.square().sum(dim=(-2, -1), keepdim=True)
        beta = torch.where(active, new_rr / rr.clamp_min(tiny), torch.zeros_like(rr))
        direction, rr = residual + beta * direction, new_rr
    diagnostics = solve_diagnostics(solution, rhs, offsets)
    if (not bool(torch.isfinite(solution).all())
            or diagnostics["relative_residual_max"] > tolerance
            or diagnostics["zero_rhs_absolute_residual_max"] != 0):
        raise RuntimeError(f"screened solve failed equation tolerance: {diagnostics}")
    return solution


class _ScreenedSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs: Tensor, inference: bool, offsets: tuple[int, ...]) -> Tensor:
        ctx.inference, ctx.offsets, ctx.input_dtype = inference, offsets, rhs.dtype
        return _solve_no_grad(rhs, inference, offsets)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        # A is fixed and symmetric.  This implicit backward stores no CG tape,
        # and remains valid when the forward RHS (and solution) is exactly zero.
        result = _ScreenedSolve.apply(gradient, ctx.inference, ctx.offsets)
        return result.to(ctx.input_dtype), None, None


def screened_solve(rhs: Tensor, inference: bool = False,
                   offsets: Sequence[int] = OFFSETS) -> Tensor:
    """Solve (I+L)x=rhs with implicit autograd and a checked true residual.

    Training uses float32 and relative L2 tolerance 1e-6.  Inference/checks use
    float64 and tolerance 1e-10.  Exactly zero RHS always yields exact zero.
    """
    _field(rhs, "rhs")
    return _ScreenedSolve.apply(rhs, bool(inference), _offsets(offsets))


def _panck_pair(x: Tensor, y: Tensor) -> None:
    if (not isinstance(x, Tensor) or not isinstance(y, Tensor) or x.ndim != 4
            or x.shape[1] != 1 or x.shape != y.shape or x.device != y.device
            or min(x.shape[-2:]) <= 10):
        raise ValueError("matching N1HW panCK fields with dimensions >10 required")
    if not x.is_floating_point() or not y.is_floating_point():
        raise ValueError("panCK fields must be floating point in [0,1]")
    if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
        raise ValueError("panCK fields must be finite")
    if bool((x < 0).any() | (x > 1).any() | (y < 0).any() | (y > 1).any()):
        raise ValueError("panCK fields must be in [0,1]")


def panck_ssim(pred: Tensor, target: Tensor) -> Tensor:
    """Native-compatible per-image SSIM, differentiable in normalized panCK.

    This is the canonical 11x11 Gaussian (sigma 1.5), valid-center formula,
    including population second moments.  Float64 statistics prevent unstable
    cancellation in bright flat fields.  No padding or quantization is applied.
    """
    _panck_pair(pred, target)
    x, y = pred.to(torch.float64), target.to(torch.float64)
    coordinate = torch.arange(-5, 6, dtype=torch.float64, device=x.device)
    gaussian = torch.exp(-coordinate.square() / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :])[None, None]
    mu_x, mu_y = F.conv2d(x, kernel), F.conv2d(y, kernel)
    xx, yy, xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    var_x, var_y = F.conv2d(x.square(), kernel) - xx, F.conv2d(y.square(), kernel) - yy
    covariance = F.conv2d(x*y, kernel) - xy
    value = ((2*xy + 0.01**2) * (2*covariance + 0.03**2)
             / ((xx + yy + 0.01**2) * (var_x + var_y + 0.03**2)))
    return value.mean(dim=(1, 2, 3))


def loss_scales(donor: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Raw per-image reference terms; mean over training images BEFORE floor.

    Callers compute each training-subset equal-image mean and clamp it to 1e-8.
    Returning un-floored rows avoids changing this order of operations.
    """
    _panck_pair(donor, target)
    result = {"mse": (donor-target).square().mean(dim=(1, 2, 3)),
              "ssim": 1-panck_ssim(donor, target)}
    for step in OFFSETS:
        differences = graph_gradient(donor-target, (step,))
        counts = sum(value[0].numel() for value in differences)
        result[f"edge{step}"] = (sum(value.square().flatten(1).sum(1)
                                         for value in differences) / counts
                                    if counts else donor.flatten(1).sum(1) * 0)
    return result


def normalize_loss(pred: Tensor, target: Tensor, donor: Tensor,
                   scales: Mapping[str, Tensor | float]) -> dict[str, Tensor]:
    """Return per-image normalized image/edge terms and their fixed total.

    ``scales`` must come from the current training subset, never the current
    held image.  The donor argument binds shape/units and is not re-estimated.
    """
    _panck_pair(donor, target)
    raw = loss_scales(pred, target)
    normalized = {}
    for key, value in raw.items():
        if key not in scales:
            raise ValueError(f"missing training loss scale {key}")
        scale = torch.as_tensor(scales[key], device=value.device, dtype=value.dtype)
        if (scale.ndim > 1 or (scale.ndim == 1 and scale.shape != value.shape)
                or not bool(torch.isfinite(scale).all()) or bool((scale < 0).any())):
            raise ValueError("loss scales must be finite nonnegative scalars or N rows")
        normalized[key] = value / scale.clamp_min(SCALE_FLOOR)
    normalized["image"] = normalized["mse"] + normalized["ssim"]
    normalized["edge"] = 0.5 * (normalized["edge1"] + normalized["edge16"])
    normalized["total"] = normalized["image"] + normalized["edge"]
    return normalized


def _neighbor_context(h: Tensor, start: int, stop: int) -> Tensor:
    """296-channel unary context for a row chunk, including eight flags."""
    n, channels, height, width = h.shape
    neighbors, flags = [], []
    for step in OFFSETS:
        for dy, dx in ((0, step), (0, -step), (step, 0), (-step, 0)):
            value = h.new_zeros((n, channels, stop-start, width))
            valid = h.new_zeros((n, 1, stop-start, width))
            y0, y1 = max(start, -dy), min(stop, height-dy)
            x0, x1 = max(0, -dx), min(width, width-dx)
            if y1 > y0 and x1 > x0:
                value[..., y0-start:y1-start, x0:x1] = h[..., y0+dy:y1+dy, x0+dx:x1+dx]
                valid[..., y0-start:y1-start, x0:x1] = 1
            neighbors.append(value)
            flags.append(valid)
    return torch.cat([h[..., start:stop, :], *neighbors, *flags], dim=1)


def unary_context(h: Tensor) -> Tensor:
    """Expose the complete eight-neighbor context for small diagnostic cases."""
    if h.ndim != 4 or h.shape[1] != 32:
        raise ValueError("unary context requires N32HW backbone features")
    return _neighbor_context(h, 0, h.shape[-2])


class RelationModel(nn.Module):
    """P/M/S/E arms with a paired common projection/backbone initialization.

    Reset torch's seed before constructing each arm.  Every common module is
    constructed before the arm-specific head, preserving paired initialization.
    Token statistics are immutable saved buffers from the fitting subset.
    """
    def __init__(self, arm: str, token_mean: Tensor, token_std: Tensor):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown relation arm {arm!r}")
        mean, std = torch.as_tensor(token_mean), torch.as_tensor(token_std)
        if (mean.shape != (1536,) or std.shape != (1536,)
                or not bool(torch.isfinite(mean).all() & torch.isfinite(std).all())
                or bool((std < 0).any())):
            raise ValueError("finite token mean/std vectors of length1536 required")
        self.arm = arm
        self.register_buffer("token_mean", mean.to(torch.float32).reshape(1, 1536, 1, 1).clone())
        self.register_buffer("token_std", std.to(torch.float32).reshape(1, 1536, 1, 1).clone())
        self.token_projection = nn.Conv2d(1536, 16, 1)
        layers, incoming = [], 28
        for dilation in (1, 2, 4, 8, 16):
            layers.extend([nn.Conv2d(incoming, 32, 3, padding=dilation,
                                     dilation=dilation, padding_mode="reflect"), nn.SiLU()])
            incoming = 32
        self.backbone = nn.Sequential(*layers)
        self.unary_head = None if self.is_edge else nn.Sequential(
            nn.Linear(296, 64), nn.SiLU(), nn.Linear(64, 1))
        self.edge_head = nn.Sequential(nn.Linear(66, 64), nn.SiLU(), nn.Linear(64, 1)) if self.is_edge else None
        head = self.edge_head if self.is_edge else self.unary_head
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)

    @property
    def is_edge(self) -> bool:
        return self.arm.startswith("E_")

    @property
    def uses_direct_tokens(self) -> bool:
        return not self.arm.endswith("_C")

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        unused = 0 if self.uses_direct_tokens else sum(p.numel() for p in self.token_projection.parameters())
        return {"active": total-unused, "total": total}

    def project_tokens(self, tokens: Tensor, size: tuple[int, int]) -> Tensor:
        if not self.uses_direct_tokens:
            return self.token_projection.weight.new_zeros((tokens.shape[0], 16, *size))
        safe_std = self.token_std.masked_fill(self.token_std == 0, 1)
        standardized = ((tokens-self.token_mean) / safe_std).masked_fill(self.token_std == 0, 0)
        projected = self.token_projection(standardized)
        return F.interpolate(projected, size=size, mode="bilinear", align_corners=False)

    def edge_drive(self, hp: Tensor, hq: Tensor, displacement: Tensor) -> Tensor:
        """Antisymmetric drive; displacement is signed (dy,dx)/16."""
        if self.edge_head is None:
            raise ValueError("edge_drive is available only for an E arm")
        displacement = torch.as_tensor(displacement, device=hp.device, dtype=hp.dtype)
        displacement = displacement.expand(*hp.shape[:-1], 2)
        return 0.5 * (self.edge_head(torch.cat((hp, hq, displacement), dim=-1))
                      - self.edge_head(torch.cat((hq, hp, -displacement), dim=-1)))

    def _edge_chunk(self, h: Tensor, orientation: int, step: int,
                    start: int, stop: int) -> Tensor:
        n, _, _, width = h.shape
        if orientation == 0:
            hp, hq = h[..., start:stop, :width-step], h[..., start:stop, step:]
            displacement = h.new_tensor((0.0, step/16.0))
        else:
            hp, hq = h[..., start:stop, :], h[..., start+step:stop+step, :]
            displacement = h.new_tensor((step/16.0, 0.0))
        shape = hp.shape
        result = self.edge_drive(hp.permute(0, 2, 3, 1).reshape(-1, 32),
                                 hq.permute(0, 2, 3, 1).reshape(-1, 32), displacement)
        return result.reshape(n, shape[-2], shape[-1], 1).permute(0, 3, 1, 2)

    def _checkpoint(self, function, h: Tensor) -> Tensor:
        if self.training and torch.is_grad_enabled() and h.requires_grad:
            # Save only the common h reference.  Endpoint/context construction
            # is recomputed by each bounded chunk during backward.
            return checkpoint(function, h, use_reentrant=False)
        return function(h)

    def edge_fields(self, h: Tensor) -> tuple[Tensor, ...]:
        fields = []
        n, _, height, width = h.shape
        for step in OFFSETS:
            for orientation in (0, 1):
                rows, columns = ((height, max(0, width-step)) if orientation == 0
                                  else (max(0, height-step), width))
                if not rows or not columns:
                    fields.append(h.new_empty((n, 1, rows, columns)))
                    continue
                stride = max(1, _CHUNK_PIXELS // (n*columns))
                chunks = []
                for start in range(0, rows, stride):
                    stop = min(rows, start+stride)
                    function = lambda value, o=orientation, s=step, a=start, b=stop: self._edge_chunk(value, o, s, a, b)
                    chunks.append(self._checkpoint(function, h))
                fields.append(torch.cat(chunks, dim=-2))
        return tuple(fields)

    def unary_potential(self, h: Tensor) -> Tensor:
        if self.unary_head is None:
            raise ValueError("unary_potential is available only for P/M/S arms")
        n, _, height, width = h.shape
        stride, chunks = max(1, _CHUNK_PIXELS // (n*width)), []
        for start in range(0, height, stride):
            stop = min(height, start+stride)
            def function(value, a=start, b=stop):
                context = _neighbor_context(value, a, b).permute(0, 2, 3, 1)
                return self.unary_head(context).permute(0, 3, 1, 2)
            chunks.append(self._checkpoint(function, h))
        return torch.cat(chunks, dim=-2)

    def _forward(self, common: Tensor, tokens: Tensor, inference: bool):
        if (common.ndim != 4 or common.shape[1] != 12 or min(common.shape[-2:]) <= 16
                or tokens.shape != (common.shape[0], 1536, 32, 32)
                or common.device != tokens.device):
            raise ValueError("common N12HW (H,W>16) and tokens N1536x32x32 required")
        if (not common.is_floating_point() or not tokens.is_floating_point()
                or not bool(torch.isfinite(common).all() & torch.isfinite(tokens).all())):
            raise ValueError("common and tokens must be finite floating point")
        dtype = self.token_projection.weight.dtype
        common, tokens = common.to(dtype=dtype), tokens.to(dtype=dtype)
        h = self.backbone(torch.cat((common, self.project_tokens(tokens, common.shape[-2:])), dim=1))
        rhs = None
        if self.is_edge:
            fields = self.edge_fields(h)
            if inference:
                fields = tuple(value.to(torch.float64) for value in fields)
            rhs = graph_adjoint(fields, (common.shape[0], 1, *common.shape[-2:]))
        else:
            potential = self.unary_potential(h)
            if inference:
                potential = potential.to(torch.float64)
            if self.arm.startswith("S_"):
                rhs = laplacian(potential)
            elif self.arm.startswith("M_"):
                residual = potential-potential.mean(dim=(-2, -1), keepdim=True)
            else:
                residual = potential
        if rhs is not None:
            rhs = rhs.to(torch.float64 if inference else torch.float32)
            residual = screened_solve(rhs, inference=inference)
            diagnostics = solve_diagnostics(residual.detach(), rhs.detach())
        else:
            diagnostics = {"relative_residual_max": 0.0, "absolute_residual_max": 0.0,
                           "zero_rhs_count": 0, "zero_rhs_absolute_residual_max": 0.0,
                           "solver_dtype": None}
        if not bool(torch.isfinite(residual).all()):
            raise RuntimeError("relation model produced a nonfinite residual")
        diagnostics.update({"rhs": None if rhs is None else rhs.detach(),
                            "residual_mean_max": float(residual.detach().mean(dim=(-2, -1)).abs().max()),
                            "arm": self.arm})
        return residual, diagnostics

    def forward(self, common: Tensor, tokens: Tensor, inference: bool = False):
        if inference:
            with torch.no_grad():
                return self._forward(common, tokens, True)
        return self._forward(common, tokens, False)


def paired_models(arms: Sequence[str], token_mean: Tensor, token_std: Tensor,
                  seed: int) -> dict[str, RelationModel]:
    """Construct seed-paired arms without changing the caller's RNG state."""
    result = {}
    with torch.random.fork_rng(devices=[]):
        for arm in arms:
            torch.random.default_generator.manual_seed(seed)
            result[arm] = RelationModel(arm, token_mean, token_std)
    return result
