"""S2 components (development/contribution-design.md, slate 1).

`build(name, weight, device)` returns None for the control arm ('none'), or an object that is
callable with keyword tensors (x, rec, gt, ...) and returns a scalar loss added to the generator
objective. An object may also define `wrap(netG)`, which returns a module used in place of the
generator for training and evaluation.

Tensors are in DTR's range [-1, 1], shape (B, 3, H, W), channel order R = panCK, G = CD3, B = DAPI
(development/erratum-channel-order.md; the deployed scoring code keeps legacy key names).
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA_LR = 1e-3
PROJ_LR = 1e-4


class QuantileW1:
    """C1: per-image, per-marker 1-D Wasserstein distance between sorted output and target values,
    taken at `nq` quantiles. `swap=True` is the matched control: each image's target quantiles are
    taken from the other image in the batch (roll by one), which keeps the loss form and magnitude
    and destroys the image-specific correspondence."""

    def __init__(self, weight, nq=16, swap=False, device="cpu"):
        self.weight = weight
        self.swap = swap
        self.q = torch.linspace(0.02, 0.98, nq, device=device)

    def __call__(self, rec, gt, **_):
        r = rec.flatten(2)
        g = gt.flatten(2)
        if self.swap:
            g = g.roll(1, dims=0)
        qr = torch.quantile(r, self.q, dim=2)
        qg = torch.quantile(g, self.q, dim=2).detach()
        return self.weight * (qr - qg).abs().mean()


class LocalQuantileW1(QuantileW1):
    """C1 rung 2: the same sorted-quantile W1, per non-overlapping k x k window instead of per image.
    Windows are chosen larger than the training misregistration (noise 2 shifts the input tile by up
    to about 40 px, plus rotation and scale), so a window's intensity distribution survives the shift
    while panCK-dense and panCK-empty regions of one image are matched separately. `swap=True` takes
    each window's target from the same window of the other image in the batch."""

    def __init__(self, weight, k, nq=16, swap=False, device="cpu"):
        super().__init__(weight, nq=nq, swap=swap, device=device)
        self.k = k

    def __call__(self, rec, gt, **_):
        B, C, H, W = rec.shape
        k = self.k
        r = rec.unfold(2, k, k).unfold(3, k, k).reshape(B, C, -1, k * k)
        g = gt.unfold(2, k, k).unfold(3, k, k).reshape(B, C, -1, k * k)
        if self.swap:
            g = g.roll(1, dims=0)
        qr = torch.quantile(r, self.q, dim=-1)
        qg = torch.quantile(g, self.q, dim=-1).detach()
        return self.weight * (qr - qg).abs().mean()


class TolerantPositivityF1:
    """C6: soft, registration-tolerant F1 of marker-positive regions, computed on DGR's registered
    output. Pixel losses under misregistration are minimised by hedging: dimming a marker below its
    detection threshold costs little L1 when the true positives may be anywhere nearby (measured:
    12 fine-tuning steps with the reconstruction terms drop pooled F1 0.598 -> 0.383 without GAN,
    and only to 0.566 with GAN alone). This term scores positivity the way the evaluation does:
    soft positives p = sigmoid((I - thr) / s) on the uint8 scale, hard label positives g, a predicted
    positive is matched if a label positive lies within r pixels (max-pool dilation) and conversely.
    Counts are pooled over the batch per marker, and the loss is 1 - (tp_p + tp_g + 1) / (n_p + n_g + 1),
    the symmetric tolerant Dice with +1 smoothing, so a batch with no label positives still pushes
    false positives down. `swap=True` (matched control) takes the label map from the other image in
    the batch; `r=0` is the no-tolerance ablation."""

    THR = (22.0, 30.0, 18.0)  # panCK, CD3, DAPI on channels R, G, B; the charter's fixed thresholds

    def __init__(self, weight, r=5, s=3.0, swap=False, r_prec=None, levels=(1.0,), s_rel=None, channels=None):
        self.weight, self.r, self.s, self.swap = weight, r, s, swap
        self.channels = channels  # C6 scoped to markers: kept terms summed and divided by 3, so each keeps C6's gradient
        self.r_prec = r if r_prec is None else r_prec  # C6a: strict precision radius, tolerant recall radius
        self.levels = levels  # C6m: thresholds at these multiples of the charter thresholds, losses averaged
        self.s_rel = s_rel    # if set, temperature = s_rel * level threshold instead of the absolute s

    @staticmethod
    def _dil(t, r):
        return t if r == 0 else F.max_pool2d(t, 2 * r + 1, stride=1, padding=r)

    def __call__(self, rec_after_reg, gt, mask=None, **_):
        u = (rec_after_reg + 1) * 127.5
        v = (gt + 1) * 127.5
        m = None
        if mask is not None:
            m = mask[:, :1] if mask.shape[1] != 3 else mask
        losses = []
        for lv in self.levels:
            thr = torch.tensor(self.THR, device=u.device).view(1, 3, 1, 1) * lv
            temp = self.s if self.s_rel is None else self.s_rel * thr
            p = torch.sigmoid((u - thr) / temp)
            g = (v > thr).float()
            if self.swap:
                g = g.roll(1, dims=0)
            if m is not None:
                p, g = p * m, g * m
            tp_p = (p * self._dil(g, self.r_prec)).sum(dim=(0, 2, 3))
            tp_g = (g * self._dil(p, self.r)).sum(dim=(0, 2, 3))
            n = p.sum(dim=(0, 2, 3)) + g.sum(dim=(0, 2, 3))
            per = 1 - (tp_p + tp_g + 1) / (n + 1)
            losses.append(per.mean() if self.channels is None else per[list(self.channels)].sum() / 3)
        return self.weight * torch.stack(losses).mean()


def _marker_mask(mask):
    if mask is None:
        return None
    return mask[:, :1] if mask.shape[1] != 3 else mask


class ForegroundBCE:
    """Prior-art comparator for C6 (Ben-David et al. 2024, Eq. 1: semantic pixel-wise binary cross
    entropy on stain masks). The soft positivity is C6's, sigmoid((I - thr) / s), scored against the
    hard label mask by BCE averaged over valid pixels per marker. No spatial tolerance. Its gradient in
    I is (p - g) / s, which does not vanish far from the threshold, unlike C6's ratio of soft counts."""

    def __init__(self, weight, s=3.0):
        self.weight, self.s = weight, s

    def __call__(self, rec_after_reg, gt, mask=None, **_):
        u = (rec_after_reg + 1) * 127.5
        v = (gt + 1) * 127.5
        thr = torch.tensor(TolerantPositivityF1.THR, device=u.device).view(1, 3, 1, 1)
        g = (v > thr).float()
        bce = F.binary_cross_entropy_with_logits((u - thr) / self.s, g, reduction="none")
        m = _marker_mask(mask)
        if m is None:
            m = torch.ones_like(g[:, :1])
        m = m.expand_as(bce)
        per_marker = (bce * m).sum(dim=(0, 2, 3)) / m.sum(dim=(0, 2, 3)).clamp_min(1.0)
        return self.weight * per_marker.mean()


class BoundaryF1:
    """Prior-art comparator for C6 (Bokhovkin and Burnaev 2019, Section 3.2, Eqs. 1-4): soft
    boundaries y_b = pool(1 - y, theta0) - (1 - y), extended boundaries pool(y_b, theta), directed
    precision and recall, BF1 = 2PR / (P + R), loss 1 - BF1. Applied to C6's soft positivity and label
    masks with C6's tolerance (theta = 2r + 1) and C6's pooling over the batch per marker with +1
    smoothing, so it differs from C6 in scoring boundaries rather than whole regions and in the
    harmonic aggregation. theta0 = 3 as in the paper."""

    def __init__(self, weight, r=5, theta0=3, s=3.0):
        self.weight, self.r, self.theta0, self.s = weight, r, theta0, s

    @staticmethod
    def _pool(t, k):
        return t if k <= 1 else F.max_pool2d(t, k, stride=1, padding=k // 2)

    def __call__(self, rec_after_reg, gt, mask=None, **_):
        u = (rec_after_reg + 1) * 127.5
        v = (gt + 1) * 127.5
        thr = torch.tensor(TolerantPositivityF1.THR, device=u.device).view(1, 3, 1, 1)
        p = torch.sigmoid((u - thr) / self.s)
        g = (v > thr).float()
        bp = self._pool(1 - p, self.theta0) - (1 - p)
        bg = self._pool(1 - g, self.theta0) - (1 - g)
        m = _marker_mask(mask)
        if m is not None:  # boundaries first, then the mask, so the mask edge is not a boundary
            bp, bg = bp * m, bg * m
        k = 2 * self.r + 1
        prec = ((bp * self._pool(bg, k)).sum(dim=(0, 2, 3)) + 1) / (bp.sum(dim=(0, 2, 3)) + 1)
        rec = ((bg * self._pool(bp, k)).sum(dim=(0, 2, 3)) + 1) / (bg.sum(dim=(0, 2, 3)) + 1)
        return self.weight * (1 - 2 * prec * rec / (prec + rec)).mean()


class SpotlightLoss:
    """Prior-art comparator for C6 (Kalinin et al. 2025, Spotlight, Section 2 and Appendix A.4):
    lam * foreground-masked MSE + (1 - lam) * Dice between sigma_k(Y_hat) and the foreground mask M,
    with sigma_k(x) = (x - kx) / (k - 2k|x| + 1), k = -0.95, lam = 0.5, intensities in [0, 1]. M is the
    label above the charter thresholds, which are Otsu thresholds on train labels as in the paper.
    Sums are pooled over the batch per marker like C6; the Dice epsilon is 1."""

    def __init__(self, weight, k=-0.95, lam=0.5):
        self.weight, self.k, self.lam = weight, k, lam

    def sigma(self, x):
        k = self.k
        return (x - k * x) / (k - 2 * k * x.abs() + 1)

    def __call__(self, rec_after_reg, gt, mask=None, **_):
        yh = ((rec_after_reg + 1) / 2).clamp(0, 1)
        y = (gt + 1) / 2
        thr = torch.tensor(TolerantPositivityF1.THR, device=y.device).view(1, 3, 1, 1) / 255.0
        M = (y > thr).float()
        sk = self.sigma(yh)
        m = _marker_mask(mask)
        if m is not None:
            M, sk = M * m, sk * m
        nM = M.sum(dim=(0, 2, 3))
        mmse = (M * (y - yh) ** 2).sum(dim=(0, 2, 3)) / nM.clamp_min(1.0)
        dice = 1 - 2 * (sk * M).sum(dim=(0, 2, 3)) / (sk.sum(dim=(0, 2, 3)) + nM + 1.0)
        return self.weight * (self.lam * mmse + (1 - self.lam) * dice).mean()


class StableRecon:
    """C8: DGR's reconstruction term (SmoothL1, beta 1, on the registered output inside the warp mask,
    mean over all elements, rec_weight 10) kept only where the label's marker positivity is constant
    within r pixels, i.e. background and region interiors, and dropped in the band of width 2r + 1
    around positivity boundaries. There residual misregistration makes the target uncertain, and on
    [-1, 1] images SmoothL1 is about half the squared error, whose minimiser under positional
    uncertainty is a dim average: the hedging the 12-step diagnostic measured. Run with --rec_weight 0
    so this term replaces DGR's. r < 0 keeps every pixel, which is DGR's term exactly."""

    def __init__(self, weight, r=5):
        self.weight, self.r = weight, r

    def stable(self, gt):
        v = (gt + 1) * 127.5
        g = (v > torch.tensor(TolerantPositivityF1.THR, device=gt.device).view(1, 3, 1, 1)).float()
        if self.r < 0:
            return torch.ones_like(g)
        k = 2 * self.r + 1
        dil = F.max_pool2d(g, k, stride=1, padding=self.r)
        ero = 1 - F.max_pool2d(1 - g, k, stride=1, padding=self.r)
        return 1 - (dil - ero)

    def __call__(self, rec_after_reg, gt, mask, **_):
        per = F.smooth_l1_loss(rec_after_reg * mask, gt * mask, reduction="none")
        return self.weight * (per * self.stable(gt)).mean()


class WindowedPositivity:
    """C9: soft positive counts matched per window. C6's soft positivity p and the label mask g are
    averaged over k x k windows (stride k/2, default k = 16 > 2r + 1), and per marker the loss is
    sum|a_p - a_g| / (sum(a_p + a_g) + 1), pooled over the batch. A shift of a few pixels barely moves a
    window count, so misregistration is tolerated; a rim or a sparse lattice inside a region lowers the
    count (C6 and the 5-px tolerant F1 score those the same as a filled region; fill_diag_v1: C6 cut
    DAPI exact recall 0.72 -> 0.51 at unchanged tolerant recall), and spreading raises it."""

    def __init__(self, weight, k=16, s=3.0, channels=None):
        self.weight, self.k, self.s = weight, k, s
        self.channels = channels  # as in C6: kept terms summed and divided by 3

    def __call__(self, rec_after_reg, gt, mask=None, **_):
        u = (rec_after_reg + 1) * 127.5
        v = (gt + 1) * 127.5
        thr = torch.tensor(TolerantPositivityF1.THR, device=u.device).view(1, 3, 1, 1)
        p = torch.sigmoid((u - thr) / self.s)
        g = (v > thr).float()
        m = _marker_mask(mask)
        if m is not None:
            p, g = p * m, g * m
        ap = F.avg_pool2d(p, self.k, stride=self.k // 2)
        ag = F.avg_pool2d(g, self.k, stride=self.k // 2)
        per = (ap - ag).abs().sum(dim=(0, 2, 3)) / ((ap + ag).sum(dim=(0, 2, 3)) + 1)
        return self.weight * (per.mean() if self.channels is None else per[list(self.channels)].sum() / 3)


class BackgroundGlow:
    """C6G's added term: a hinge on sub-threshold intensity above the label, only at pixels with no
    label positive within r pixels (max-pool dilation, the tolerance C6 uses). No residual shift below
    r can bring a positive there, so the term cannot reward misplacing one. Per marker
    mean(min(relu(I - L), thr) / thr) over those pixels, on the uint8 scale; kept markers summed and
    divided by 3. Under C6 a sub-threshold output earns no F1 and costs little SmoothL1, which is
    quadratic for a glow of a few levels (ssim_decomp_v1: two thirds of C6's SSIM loss is panCK and
    CD3 background luminance); the hinge prices it at a constant 1 / thr per level."""

    def __init__(self, weight, r=5, channels=(0, 1)):
        self.weight, self.r, self.channels = weight, r, channels

    def certain_background(self, gt):
        thr = torch.tensor(TolerantPositivityF1.THR, device=gt.device).view(1, 3, 1, 1)
        g = ((gt + 1) * 127.5 > thr).float()
        return 1 - F.max_pool2d(g, 2 * self.r + 1, stride=1, padding=self.r)

    def __call__(self, rec_after_reg, gt, mask=None, **_):
        u = (rec_after_reg + 1) * 127.5
        v = (gt + 1) * 127.5
        thr = torch.tensor(TolerantPositivityF1.THR, device=u.device).view(1, 3, 1, 1)
        bg = self.certain_background(gt)
        m = _marker_mask(mask)
        if m is not None:
            bg = bg * m
        ex = torch.minimum(F.relu(u - v), thr) / thr
        per = (ex * bg).sum(dim=(0, 2, 3)) / (bg.sum(dim=(0, 2, 3)) + 1)
        return self.weight * per[list(self.channels)].sum() / 3


class Sum:
    """Several loss components added together (each keeps its own weight)."""

    def __init__(self, *parts):
        self.parts = parts

    def __call__(self, **kw):
        return sum(c(**kw) for c in self.parts)


def _colour_map_coeffs():
    with open(os.path.join(_HERE, "s2_colour_map.json")) as f:
        return json.load(f)["ridge_W"]


class ColourPriorWrap(nn.Module):
    """C2: out = netG(x) + alpha * (prior(x) - netG(x)), with per-marker alpha initialised at 0, so
    training starts exactly at the released generator. prior(x) is the frozen per-pixel quadratic
    colour map fitted on train pixels (s2/step0_atlas.py). `shuffle_blocks=True` is the matched
    control: the prior is permuted in 32x32 blocks, which keeps its value distribution and destroys
    per-pixel correspondence with the input."""

    def __init__(self, inner, W, shuffle_blocks=False, block=32):
        super().__init__()
        self.inner = inner
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))  # (10, 3), output in 0..255
        self.alpha = nn.Parameter(torch.zeros(3))
        self.shuffle_blocks = shuffle_blocks
        self.block = block

    def prior(self, x):
        f = (x + 1) / 2
        r, g, b = f[:, 0], f[:, 1], f[:, 2]
        feats = torch.stack([r, g, b, r * r, g * g, b * b, r * g, r * b, g * b, torch.ones_like(r)], -1)
        p = (feats @ self.W).clamp(0, 255) / 127.5 - 1  # (B, H, W, 3) in [-1, 1]
        p = p.permute(0, 3, 1, 2)
        if self.shuffle_blocks:
            B, C, H, Wd = p.shape
            k = self.block
            blocks = p.unfold(2, k, k).unfold(3, k, k)  # (B, C, H/k, W/k, k, k)
            nh, nw = blocks.shape[2], blocks.shape[3]
            gen = torch.Generator(device="cpu").manual_seed(0)
            perm = torch.randperm(nh * nw, generator=gen).to(p.device)
            blocks = blocks.reshape(B, C, nh * nw, k, k)[:, :, perm]
            p = blocks.reshape(B, C, nh, nw, k, k).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, Wd)
        return p

    def forward(self, x):
        out = self.inner(x)
        return out + self.alpha.view(1, 3, 1, 1) * (self.prior(x) - out)


class _WrapOnly:
    def __init__(self, shuffle_blocks):
        self.shuffle_blocks = shuffle_blocks

    def wrap(self, netG):
        return ColourPriorWrap(netG, _colour_map_coeffs(), shuffle_blocks=self.shuffle_blocks)

    def param_groups(self, netG, lr):
        # alpha gets its own group: at the generator's lr (1e-5) AdamW moves it by at most ~0.04 in
        # 3,718 steps, so the blend could never open. No weight decay, which would pull it to 0.
        inner = [p for n, p in netG.named_parameters() if n != "alpha"]
        return [{"params": inner}, {"params": [netG.alpha], "lr": ALPHA_LR, "weight_decay": 0.0}]

    def __call__(self, **_):
        return 0.0


def _cos_loss(h, tok):
    return (1 - F.cosine_similarity(h, tok, dim=1)).mean()


def _uni2h(pretrained):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import timm
    from timm.layers import SwiGLUPacked
    # the UNI2-h model card's timm_kwargs, as in the hest project's encode_extra.py
    return timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=pretrained,
                             img_size=224, patch_size=14, depth=24, num_heads=24,
                             init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
                             num_classes=0, no_embed_class=True, mlp_layer=SwiGLUPacked,
                             act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)


class FMAlign:
    """C4: REPA-style alignment of the generator bottleneck (Conv5, 1024 x H/16 x W/16) to a frozen
    pathology foundation model's patch tokens on the same H&E input. A 512 crop is resized to 448,
    which gives UNI2-h's 14-pixel patches a 32 x 32 grid, the bottleneck's grid. A 3-layer 1x1 MLP
    projects the bottleneck to the FM width, and the loss is 1 - cosine per position. The FM sees the
    generator's own (noise-shifted) input, so the target is spatially aligned with the features
    regardless of the input-label misregistration. `random_init=True` is the matched control: the
    same architecture with random weights (fixed seed), so compute, loss and projector are equal and
    the target carries no pathology knowledge. Construction runs inside fork_rng so it consumes no
    global RNG and the data stream stays paired with the control arm."""

    def __init__(self, weight, device, random_init=False, fm=None, grid=32, in_ch=1024):
        self.weight = weight
        self.grid = grid
        dev = torch.device(device)
        devs = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(1234)
            self.fm = (fm if fm is not None else _uni2h(pretrained=not random_init)).to(dev).eval()
            for q in self.fm.parameters():
                q.requires_grad_(False)
            width = self.fm.num_features
            self.proj = nn.Sequential(nn.Conv2d(in_ch, 2048, 1), nn.SiLU(), nn.Conv2d(2048, 2048, 1), nn.SiLU(),
                                      nn.Conv2d(2048, width, 1)).to(dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        self.feat = None
        self.patch = 14

    def _hook(self, module, inp, out):
        if torch.is_grad_enabled():
            self.feat = out

    def wrap(self, netG):
        netG.Conv5.register_forward_hook(self._hook)
        return netG

    def param_groups(self, netG, lr):
        return [{"params": list(netG.parameters())}, {"params": list(self.proj.parameters()), "lr": PROJ_LR}]

    def targets(self, x):
        B = x.shape[0]
        with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            z = F.interpolate((x + 1) / 2, size=self.grid * self.patch, mode="bilinear", antialias=True,
                              align_corners=False)
            tok = self.fm.forward_features((z - self.mean) / self.std)[:, self.fm.num_prefix_tokens:]
        assert tok.shape[1] == self.grid * self.grid, tok.shape
        return tok.float().transpose(1, 2).reshape(B, -1, self.grid, self.grid)

    def __call__(self, x, **_):
        assert self.feat is not None and self.feat.shape[-1] == self.grid, None if self.feat is None else self.feat.shape
        loss = self.weight * _cos_loss(self.proj(self.feat), self.targets(x))
        self.feat = None
        return loss


FML_THR = (22.0, 30.0, 18.0)  # panCK, CD3, DAPI on the uint8 scale (metric_panel.FIXED_THR)
HEAD_LR = 1e-3
_FM_CACHE = {}


def _fm(device, random_init):
    """One frozen UNI2-h per (device, init) per process, kept outside every module so that the EMA and
    SWA deep copies of the generator and its state dict never carry the 681M FM parameters."""
    key = (str(torch.device(device)), bool(random_init))
    if key not in _FM_CACHE:
        dev = torch.device(device)
        devs = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(1234)
            fm = _uni2h(pretrained=not random_init).to(dev).eval()
        for q in fm.parameters():
            q.requires_grad_(False)
        _FM_CACHE[key] = fm
    return _FM_CACHE[key]


def _zero_conv(cin, cout):
    c = nn.Conv2d(cin, cout, 1)
    nn.init.zeros_(c.weight)
    nn.init.zeros_(c.bias)
    return c


class FMLGen(nn.Module):
    """FML: the DGR generator with a frozen pathology FM's marker layout injected at the bottleneck.

    UNI2-h tokens of the input (512 crop resized to 448, 32 x 32 grid of 14 px patches, each covering a
    16 px cell of the crop) go through a linear head to per-cell, per-marker logits of the fraction of
    marker-positive pixels (`last_logit`, supervised by `FMLayout`). The sigmoid of the layout, detached
    so that only the layout loss shapes it, is added to Conv5's output through a zero-initialised 1x1
    convolution (`inj5`); with `multiscale` the upsampled layout is also added after Up5..Up2. With
    `tokens` the raw tokens are added through their own zero-initialised 1x1 convolution (`tok5`), the
    UNIStainNet-style injection without layout supervision. Every injection starts at zero, so the
    wrapped generator equals `inner` exactly at construction. The configuration is a buffer
    (`fml_cfg`: layout, multiscale, tokens, random_init) so a checkpoint rebuilds itself."""

    def __init__(self, inner, device, layout=True, multiscale=False, tokens=False, random_init=False, width=1536):
        super().__init__()
        self.inner = inner
        self.register_buffer("fml_cfg", torch.tensor([layout, multiscale, tokens, random_init], dtype=torch.int64))
        self.fm_key = (str(torch.device(device)), bool(random_init))
        self.layout, self.multiscale, self.tokens = layout, multiscale, tokens
        if layout:
            self.head = nn.Conv2d(width, 3, 1)
            self.inj5 = _zero_conv(3, 1024)
            if multiscale:
                self.inj4, self.inj3, self.inj2, self.inj1 = (_zero_conv(3, c) for c in (512, 256, 128, 64))
        if tokens:
            self.tok5 = _zero_conv(width, 1024)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.last_logit = None
        self.last_layout = None

    def install_readout(self, readout):
        """Install the selected clipped affine readout before optimizer/EMA creation."""
        if not self.layout or self.fm_key[1]:
            raise ValueError("frozen readout requires the pretrained-FM layout path")
        if hasattr(self, "fml_readout_mode"):
            raise ValueError("frozen readout is already installed")
        from frozen_layout_readout import FrozenLayoutReadout
        if not isinstance(readout, FrozenLayoutReadout):
            raise TypeError("expected the validated frozen affine readout")
        if readout.weight.shape != self.head.weight.shape:
            raise ValueError("readout width differs from the layout head")
        self.head = readout.to(self.head.weight.device)
        self.register_buffer("fml_readout_mode", torch.tensor(1, dtype=torch.int64,
                                                             device=self.head.weight.device))

    @classmethod
    def from_state(cls, inner, sd, device):
        lay, ms, tok, rnd = (bool(v) for v in sd["fml_cfg"].tolist())
        readout = None
        if "fml_readout_mode" in sd:
            mode = sd["fml_readout_mode"]
            if mode.ndim != 0 or mode.dtype != torch.int64 or mode.item() != 1:
                raise ValueError("unknown frozen-layout checkpoint mode")
            from frozen_layout_readout import FrozenLayoutReadout
            readout = FrozenLayoutReadout(sd["head.weight"], sd["head.bias"])
        _fm(device, rnd)
        m = cls(inner, device, layout=lay, multiscale=ms, tokens=tok, random_init=rnd).to(device)
        if readout is not None:
            m.install_readout(readout)
        m.load_state_dict(sd, strict=True)
        return m

    def tokens_of(self, x):
        fm = _FM_CACHE[self.fm_key]
        with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            z = F.interpolate((x + 1) / 2, size=448, mode="bilinear", antialias=True, align_corners=False)
            t = fm.forward_features((z - self.mean) / self.std)[:, fm.num_prefix_tokens:]
        assert t.shape[1] == 1024, t.shape
        return t.float().transpose(1, 2).reshape(x.shape[0], -1, 32, 32)

    def forward(self, x):
        g = self.inner
        x1 = g.Conv1(x)
        x2 = g.Conv2(g.Pooling(x1))
        x3 = g.Conv3(g.Pooling(x2))
        x4 = g.Conv4(g.Pooling(x3))
        x5 = g.Conv5(g.Pooling(x4))
        assert x5.shape[-1] == 32, "FML assumes 512 crops (32 x 32 bottleneck)"
        tok = self.tokens_of(x)
        lay = None
        if self.layout:
            if hasattr(self, "fml_readout_mode"):
                self.last_logit = None
                lay = self.head(tok).detach()
            else:
                self.last_logit = self.head(tok)
                lay = torch.sigmoid(self.last_logit).detach()
            self.last_layout = lay
            x5 = x5 + self.inj5(lay)
        if self.tokens:
            x5 = x5 + self.tok5(tok)
        up = (lambda d, inj: d + inj(F.interpolate(lay, size=d.shape[-2:], mode="bilinear", align_corners=False))) \
            if (self.layout and self.multiscale) else (lambda d, inj: d)
        d4 = up(g.Up5(x5), getattr(self, "inj4", None))
        cat4 = torch.cat((g.Att5(g=d4, x=x4), d4), dim=1)
        d3 = up(g.Up4(cat4), getattr(self, "inj3", None))
        cat3 = torch.cat((g.Att4(g=d3, x=x3), d3), dim=1)
        d2 = up(g.Up3(cat3), getattr(self, "inj2", None))
        cat2 = torch.cat((g.Att3(g=d2, x=x2), d2), dim=1)
        d1 = up(g.Up2(cat2), getattr(self, "inj1", None))
        cat1 = torch.cat((g.Att2(g=d1, x=x1), d1), dim=1)
        return g.last_activation(g.last_conv2(g.last_conv1(cat1)))


def cell_fraction(gt, k=16):
    """gt in [-1, 1], (B, 3, H, W) -> per-cell fraction of pixels above each marker's threshold, (B, 3, H/k, W/k)."""
    u8 = ((gt + 1) / 2).clamp(0, 1) * 255
    thr = torch.tensor(FML_THR, device=gt.device).view(1, 3, 1, 1)
    return F.avg_pool2d((u8.round() > thr).float(), k)


class FMLayout:
    """FML component: wraps the generator in `FMLGen` and adds the layout loss, BCE between the head's
    per-cell logits and the label's per-cell positive fraction (16 px cells, so a few pixels of
    residual misregistration move a target by a fraction of a cell rather than flipping it).
    `random_init=True` is the matched control: the same FM architecture at a fixed random init, so the
    head, loss, injection and compute are identical and the features carry no pathology knowledge."""

    def __init__(self, weight, device, layout=True, multiscale=False, tokens=False, random_init=False,
                 readout=None):
        self.weight = weight
        self.device = device
        self.kw = dict(layout=layout, multiscale=multiscale, tokens=tokens, random_init=random_init)
        self.net = None
        self.readout = readout

    def wrap(self, netG):
        dev = torch.device(self.device)
        devs = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
        _fm(self.device, self.kw["random_init"])
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(4321)
            self.net = FMLGen(netG, self.device, **self.kw)
            if self.readout is not None:
                self.net.install_readout(self.readout)
        return self.net

    def param_groups(self, netG, lr):
        head = [p for n, p in netG.named_parameters() if n.startswith("head.")]
        inj = [p for n, p in netG.named_parameters() if n.startswith(("inj", "tok5."))]
        inner = [p for n, p in netG.named_parameters() if n.startswith("inner.")]
        assert len(head) + len(inj) + len(inner) == len(list(netG.parameters()))
        groups = [{"params": inner}, {"params": inj, "lr": PROJ_LR}]
        if head:
            groups.append({"params": head, "lr": HEAD_LR})
        return groups

    def __call__(self, gt, **_):
        if not self.kw["layout"] or self.readout is not None:
            return 0.0
        logit = self.net.last_logit
        assert logit is not None and logit.shape[-1] == gt.shape[-1] // 16
        loss = self.weight * F.binary_cross_entropy_with_logits(logit, cell_fraction(gt))
        self.net.last_logit = None
        return loss


class FMLayoutConsistency(FMLayout):
    """Same frozen-readout renderer; only the auxiliary occupancy target differs."""

    def __init__(self, weight, device, readout, target):
        super().__init__(weight, device, layout=True, tokens=True, readout=readout)
        self.target = target

    def __call__(self, rec, gt, mesh, **_):
        from layout_consistency import aligned_occupancy_loss
        if self.net is None or self.net.last_layout is None:
            raise RuntimeError("consistency requires the current generator's frozen layout")
        return aligned_occupancy_loss(rec, gt, mesh, self.net.last_layout,
                                      self.target, self.weight)


def build(name, weight, device, *, layout_readout=None):
    readout_variants = ("FMLridge", "FMLconsFM", "FMLconsGT")
    if layout_readout is not None and name not in readout_variants:
        raise ValueError("layout_readout is only supported by frozen-readout FML variants")
    if name in readout_variants:
        if layout_readout is None:
            raise ValueError(name + " requires a selected train-fitted readout")
        from frozen_layout_readout import load_readout
        if name != "FMLridge":
            return FMLayoutConsistency(weight, device, load_readout(layout_readout),
                                       "fm" if name == "FMLconsFM" else "direct")
        return FMLayout(weight, device, layout=True, tokens=True,
                        readout=load_readout(layout_readout))
    if name == "none":
        return None
    if name.startswith("FML"):  # FML, FMLms, FMLtok, FMLtokonly and their _ctrl (random-init FM)
        base, ctrl = name.split("_")[0], name.endswith("_ctrl")
        cfg = {"FML": dict(layout=True), "FMLms": dict(layout=True, multiscale=True),
               "FMLtok": dict(layout=True, tokens=True), "FMLtokonly": dict(layout=False, tokens=True)}[base]
        return FMLayout(weight, device, random_init=ctrl, **cfg)
    if name == "C1":
        return QuantileW1(weight, device=device)
    if name == "C1_ctrl":
        return QuantileW1(weight, swap=True, device=device)
    if name == "C1_rung0":
        return QuantileW1(weight, nq=1, device=device)
    if name == "C6":
        return TolerantPositivityF1(weight)
    if name == "C6_ctrl":
        return TolerantPositivityF1(weight, swap=True)
    if name == "C6r0":
        return TolerantPositivityF1(weight, r=0)
    if name == "C6a":  # recall tolerant to 5 px, precision strict to 1 px: no credit for spreading positives
        return TolerantPositivityF1(weight, r=5, r_prec=1)
    if name == "C6m":  # tolerant F1 at 0.5x, 1x, 1.5x, 2x the thresholds, temperature 0.15 x level
        return TolerantPositivityF1(weight, levels=(0.5, 1.0, 1.5, 2.0), s_rel=0.15)
    if name == "C6pc":  # C6 on panCK and CD3 only (channels R, G); DAPI term removed
        return TolerantPositivityF1(weight, channels=(0, 1))
    if name == "C6cd3":  # C6 on CD3 only (channel G), the atlas bottleneck marker
        return TolerantPositivityF1(weight, channels=(1,))
    if name == "C6C1":  # C6 at `weight`, C1 at its fixed slate-1 weight of 10
        return Sum(TolerantPositivityF1(weight), QuantileW1(10.0, device=device))
    if name == "FBCE":  # prior-art comparators for C6 (development/prior-art-intake-2026-10-04.md)
        return ForegroundBCE(weight)
    if name == "BF1":
        return BoundaryF1(weight)
    if name == "SPOT":
        return SpotlightLoss(weight)
    if name == "C9":  # windowed positivity counts, k = 16
        return WindowedPositivity(weight)
    if name == "C6C9":  # C6 at `weight`, C9 at weight 1
        return Sum(TolerantPositivityF1(weight), WindowedPositivity(1.0))
    if name == "C6D9":  # C6 at `weight`, C9 on DAPI only at weight 1 (C6C9's DAPI gradient, no panCK/CD3 count terms)
        return Sum(TolerantPositivityF1(weight), WindowedPositivity(1.0, channels=(2,)))
    if name == "C6G":  # C6 at `weight`, background-glow hinge on panCK and CD3 at weight 1
        return Sum(TolerantPositivityF1(weight), BackgroundGlow(1.0))
    if name == "C6GD9":  # C6D9 and C6G's hinge together (conditional rung, contribution-design.md slate 4)
        return Sum(TolerantPositivityF1(weight), WindowedPositivity(1.0, channels=(2,)), BackgroundGlow(1.0))
    if name == "C8":  # run with --rec_weight 0 and comp_weight 10 (DGR's rec_weight)
        return StableRecon(weight)
    if name == "C8full":  # C8 with r < 0: DGR's reconstruction term reproduced through the component path
        return StableRecon(weight, r=-1)
    if name == "C8C6":  # C8 at `weight`, C6 at its slate-2 weight of 1
        return Sum(StableRecon(weight), TolerantPositivityF1(1.0))
    if name.startswith("C1L"):  # C1L<k> and C1L<k>_ctrl
        k = int(name[3:].split("_")[0])
        return LocalQuantileW1(weight, k, swap=name.endswith("_ctrl"), device=device)
    if name == "C2":
        return _WrapOnly(shuffle_blocks=False)
    if name == "C2_ctrl":
        return _WrapOnly(shuffle_blocks=True)
    if name == "C4":
        return FMAlign(weight, device)
    if name == "C4_ctrl":
        return FMAlign(weight, device, random_init=True)
    raise ValueError(f"unknown component {name!r}")


def self_test():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    c1 = QuantileW1(1.0)
    assert float(c1(rec=x, gt=x)) < 1e-6, "C1 must be 0 on identical inputs"
    assert float(c1(rec=x, gt=x.flip(-1))) < 1e-6, "C1 must ignore spatial rearrangement"
    assert float(c1(rec=x, gt=x * 0.5)) > 1e-3
    halves = torch.cat([x[..., 32:], x[..., :32]], -1)  # left and right halves swapped
    assert float(c1(rec=x, gt=halves)) < 1e-6, "global C1 cannot see a rearrangement"
    c1l = build("C1L16", 1.0, "cpu")
    assert float(c1l(rec=x, gt=x)) < 1e-6 and float(c1l(rec=x, gt=x.flip(-1).flip(-2).flip(-1).flip(-2))) < 1e-6
    xs = x.clone(); xs[..., :, :32] *= 0.2  # left half dimmed: a local deficit the global loss underweights
    assert float(c1l(rec=xs, gt=x)) > float(c1(rec=xs, gt=x)) * 0.99
    assert float(c1l(rec=x, gt=halves)) > 1e-3, "local C1 must see a rearrangement across windows"
    assert build("C1L16_ctrl", 1.0, "cpu").swap and build("C1L256", 1.0, "cpu").k == 256
    W = [[0.0] * 3 for _ in range(10)]
    W[9] = [127.5, 127.5, 127.5]  # constant prior = 0 in [-1, 1]
    inner = nn.Conv2d(3, 3, 1)
    wrap = ColourPriorWrap(inner, W)
    assert torch.allclose(wrap(x), inner(x)), "alpha=0 must equal the inner generator"
    wrap.alpha.data[:] = 1.0
    assert torch.allclose(wrap(x), torch.zeros_like(x), atol=1e-6), "alpha=1 must equal the prior"
    Wr = (torch.rand(10, 3) * 100).tolist()
    plain, shuf = ColourPriorWrap(inner, Wr), ColourPriorWrap(inner, Wr, shuffle_blocks=True)
    p, ps = plain.prior(x), shuf.prior(x)
    assert not torch.allclose(p, ps), "shuffle must move pixels"
    assert torch.allclose(p.flatten(2).sort(-1).values, ps.flatten(2).sort(-1).values), "shuffle keeps marginals"
    blk = p[0, :, :32, :32]
    assert any(torch.equal(blk, ps[0, :, i:i + 32, j:j + 32]) for i in range(0, 64, 32) for j in range(0, 64, 32)), \
        "shuffle moves whole 32x32 blocks"
    groups = _WrapOnly(False).param_groups(plain, 1e-5)
    assert sum(len(g["params"]) for g in groups) == len(list(plain.parameters()))
    # C6: blobs well clear of the thresholds, so the soft positives are near-binary
    gt6 = torch.full((2, 3, 64, 64), -1.0)
    for b in range(2):
        for (yy, xx) in ((10 + 20 * b, 12), (40, 30 + 10 * b), (20, 50)):
            gt6[b, :, yy - 3:yy + 3, xx - 3:xx + 3] = 1.0
    c6, c6r0 = build("C6", 1.0, "cpu"), build("C6r0", 1.0, "cpu")
    l_same = float(c6(rec_after_reg=gt6, gt=gt6))
    l_shift = float(c6(rec_after_reg=gt6.roll(3, dims=-1), gt=gt6))
    l_shift_r0 = float(c6r0(rec_after_reg=gt6.roll(3, dims=-1), gt=gt6))
    l_blank = float(c6(rec_after_reg=torch.full_like(gt6, -1.0), gt=gt6))
    assert l_same < 0.03 and abs(l_shift - l_same) < 0.01, (l_same, l_shift)
    assert l_shift_r0 > 0.3, l_shift_r0
    assert l_blank > 0.95, l_blank
    dim = gt6.clone(); dim[dim > 0] = -0.97  # positives dimmed to about 4/255, well below every threshold
    assert float(c6(rec_after_reg=dim, gt=gt6)) > 0.95, "hedging below threshold must cost the full loss"
    empty = torch.full_like(gt6, -1.0)
    fp = empty.clone(); fp[:, :, 5:9, 5:9] = 1.0
    assert float(c6(rec_after_reg=fp, gt=empty)) > float(c6(rec_after_reg=empty, gt=empty)), "false positives cost"
    rq = gt6.clone().requires_grad_(True)
    c6(rec_after_reg=rq * 0.5, gt=gt6).backward()
    assert rq.grad.abs().sum() > 0
    assert build("C6_ctrl", 1.0, "cpu").swap
    fat = F.max_pool2d(gt6, 7, stride=1, padding=3)  # every positive spread by 3 px
    l_fat, l_fat_a = float(c6(rec_after_reg=fat, gt=gt6)), float(build("C6a", 1.0, "cpu")(rec_after_reg=fat, gt=gt6))
    assert l_fat < 0.05 and l_fat_a > l_fat + 0.1, (l_fat, l_fat_a)  # symmetric C6 ignores spreading, C6a charges it
    l_shift_a = float(build("C6a", 1.0, "cpu")(rec_after_reg=gt6.roll(3, dims=-1), gt=gt6))
    # a single wide sigmoid (s = 10) was rejected: a perfect copy scores 0.51 because zero-intensity
    # background carries soft positive mass; C6m keeps the temperature relative to each level instead
    cv = build("C6m", 1.0, "cpu")
    a0, a3 = float(cv(rec_after_reg=gt6, gt=gt6)), float(cv(rec_after_reg=gt6.roll(3, dims=-1), gt=gt6))
    ab = float(cv(rec_after_reg=torch.full_like(gt6, -1.0), gt=gt6))
    assert a0 < 0.05 and abs(a3 - a0) < 0.02 and ab > 0.9, (a0, a3, ab)
    assert float(build("C6", 1.0, "cpu")(rec_after_reg=gt6, gt=gt6)) == l_same, "default path unchanged"
    combo = build("C6C1", 1.0, "cpu")
    x6 = gt6.clone() * 0.5
    lc = float(combo(rec=x6, rec_after_reg=x6, gt=gt6))
    assert abs(lc - (float(c6(rec_after_reg=x6, gt=gt6)) + float(QuantileW1(10.0)(rec=x6, gt=gt6)))) < 1e-5
    # prior-art comparators: perfect copy cheap, blank and dimmed expensive, no tolerance where none is claimed
    shifted, blank = gt6.roll(3, dims=-1), torch.full_like(gt6, -1.0)
    for nm in ("FBCE", "BF1", "SPOT"):
        c = build(nm, 1.0, "cpu")
        ls, lb, ld = (float(c(rec_after_reg=t, gt=gt6)) for t in (gt6, blank, dim))
        assert ls < 0.03 and lb > ls + 0.15 and ld > ls + 0.15, (nm, ls, lb, ld)  # BCE is a pixel mean: blobs are 2.6% of pixels
        rq = (gt6 * 0.5).requires_grad_(True)
        c(rec_after_reg=rq, gt=gt6, mask=torch.ones(2, 1, 64, 64)).backward()
        assert torch.isfinite(rq.grad).all() and rq.grad.abs().sum() > 0, nm
    assert float(build("FBCE", 1.0, "cpu")(rec_after_reg=shifted, gt=gt6)) > 0.1, "BCE has no tolerance"
    assert float(build("SPOT", 1.0, "cpu")(rec_after_reg=shifted, gt=gt6)) > 0.1, "Spotlight has no tolerance"
    bf, bf0 = build("BF1", 1.0, "cpu"), BoundaryF1(1.0, r=0)
    l_bf = float(bf(rec_after_reg=gt6, gt=gt6))
    assert abs(float(bf(rec_after_reg=shifted, gt=gt6)) - l_bf) < 0.02, "BF1 tolerates a 3 px shift at r = 5"
    assert float(bf0(rec_after_reg=shifted, gt=gt6)) > 0.3
    # discriminating: a hollow outline of a filled 40 px square has the same boundary (BF1 ~ 0) but
    # misses most of the region (C6 charges it), so the two losses do not compute the same thing
    sq = torch.full((1, 3, 96, 96), -1.0); sq[..., 28:68, 28:68] = 1.0
    ring = sq.clone(); ring[..., 29:67, 29:67] = -1.0
    assert float(bf(rec_after_reg=ring, gt=sq)) < 0.05 and float(c6(rec_after_reg=ring, gt=sq)) > 0.3
    # gradient shape: at a label-positive pixel far below threshold (I = 0) against one 3 levels below,
    # BCE keeps pulling (ratio ~ 1) while C6's sigmoid gradient has vanished (ratio ~ 0.003)
    g1 = torch.full((1, 3, 8, 8), -1.0); g1[..., 2:6, 2:6] = 1.0
    r1 = g1.clone(); r1[..., 2, 2] = -1.0; r1[..., 5, 5] = (torch.tensor(TolerantPositivityF1.THR) - 3).view(3) / 127.5 - 1
    ratios = {}
    for nm, c in (("FBCE", build("FBCE", 1.0, "cpu")), ("C6r0", c6r0)):
        rq = r1.clone().requires_grad_(True)
        c(rec_after_reg=rq, gt=g1).backward()
        ratios[nm] = float(rq.grad[0, :, 2, 2].abs().mean() / rq.grad[0, :, 5, 5].abs().mean())
    assert ratios["FBCE"] > 0.5 and ratios["C6r0"] < 0.01, ratios
    # C8: r < 0 reproduces DGR's term exactly; the stable mask is 0 on boundary bands and 1 elsewhere
    a8, b8 = torch.rand(2, 3, 32, 32) * 2 - 1, torch.rand(2, 3, 32, 32) * 2 - 1
    m8 = (torch.rand(2, 1, 32, 32) > 0.1).float()
    dgr = nn.SmoothL1Loss()(a8 * m8, b8 * m8) * 10
    assert torch.allclose(build("C8full", 10.0, "cpu")(rec_after_reg=a8, gt=b8, mask=m8), dgr, rtol=0, atol=1e-7)
    blob = torch.full((1, 3, 40, 40), -1.0); blob[..., 8:20, 8:20] = 1.0  # 12 px square
    st = StableRecon(1.0).stable(blob)[0, 0]
    assert st[0, 0] == 1 and st[13, 13] == 1 and st[8, 13] == 0 and st[3, 13] == 0 and st[2, 13] == 1, st[:, 13]
    # discriminating: label shifted 3 px from where the generator is right. DGR's term prefers a dim
    # average over both positions (hedging); C8 prefers the correct, shifted blob
    one = torch.ones(1, 1, 40, 40)
    shifted8 = blob.roll(3, dims=-1)
    hedge = torch.full_like(blob, -1.0); hedge[..., 8:20, 8:23] = 0.0
    full, c8 = build("C8full", 1.0, "cpu"), build("C8", 1.0, "cpu")
    lf = [float(full(rec_after_reg=t, gt=blob, mask=one)) for t in (shifted8, hedge)]
    l8 = [float(c8(rec_after_reg=t, gt=blob, mask=one)) for t in (shifted8, hedge)]
    assert lf[1] < lf[0] and l8[0] < l8[1], (lf, l8)
    assert float(c8(rec_after_reg=blob, gt=blob, mask=one)) == 0.0
    # C6pc / C6cd3: each kept marker gets exactly C6's gradient, removed markers get none
    x6s = (gt6 * 0.3).requires_grad_(True)
    c6(rec_after_reg=x6s, gt=gt6).backward()
    g_full = x6s.grad.clone()
    for nm, keep in (("C6pc", (0, 1)), ("C6cd3", (1,))):
        xq = (gt6 * 0.3).requires_grad_(True)
        build(nm, 1.0, "cpu")(rec_after_reg=xq, gt=gt6).backward()
        for ch in range(3):
            if ch in keep:
                assert torch.allclose(xq.grad[:, ch], g_full[:, ch], atol=1e-9), (nm, ch)
            else:
                assert float(xq.grad[:, ch].abs().sum()) == 0.0, (nm, ch)
    # C9: tolerant of a small shift, charges a sparse lattice, spreading and blanks; C6 does not see the lattice
    sq9 = torch.full((1, 3, 64, 64), -1.0); sq9[..., 16:48, 16:48] = 1.0
    lat9 = torch.full_like(sq9, -1.0); lat9[..., 16:48:6, 16:48:6] = 1.0
    fat9 = F.max_pool2d(sq9, 9, stride=1, padding=4)
    c9 = build("C9", 1.0, "cpu")
    v9 = {k: float(c9(rec_after_reg=t, gt=sq9)) for k, t in (("same", sq9), ("shift2", sq9.roll(2, dims=-1)),
          ("lattice", lat9), ("fat", fat9), ("blank", torch.full_like(sq9, -1.0)))}
    assert v9["same"] < 0.01 and v9["shift2"] < v9["fat"] / 2 and v9["lattice"] > 0.8 and v9["fat"] > 0.1 and v9["blank"] > 0.9, v9
    assert float(c6(rec_after_reg=lat9, gt=sq9)) < 0.25, "C6 scores the lattice nearly as a filled square"
    # C6D9: C6's gradient on panCK and CD3, C6C9's gradient on DAPI
    grads = {}
    thr_img = (torch.tensor(TolerantPositivityF1.THR) / 127.5 - 1).view(1, 3, 1, 1)
    x_thr = thr_img + (torch.rand_like(gt6) - 0.5) * 8 / 127.5  # within 4 levels of each threshold: unsaturated sigmoids
    for nm in ("C6", "C6C9", "C6D9"):
        xq = x_thr.clone().requires_grad_(True)
        build(nm, 1.0, "cpu")(rec_after_reg=xq, gt=gt6).backward()
        grads[nm] = xq.grad.clone()
    assert torch.allclose(grads["C6D9"][:, :2], grads["C6"][:, :2], atol=1e-9)
    assert torch.allclose(grads["C6D9"][:, 2], grads["C6C9"][:, 2], atol=1e-9)
    assert not torch.allclose(grads["C6D9"][:, 2], grads["C6"][:, 2]), "the DAPI count term must act"
    # C6G's hinge: zero for an exact copy and for a 3-px shift (moved positives stay within r of the
    # label's), positive for a 3-level glow far from positives, zero for the same glow within r, no DAPI gradient
    bgl, c6g = BackgroundGlow(1.0), build("C6G", 1.0, "cpu")
    assert float(bgl(rec_after_reg=gt6, gt=gt6)) == 0.0 and float(bgl(rec_after_reg=gt6.roll(3, dims=-1), gt=gt6)) == 0.0
    cb = bgl.certain_background(gt6)
    assert cb[0, 1, 10, 12] == 0 and cb[0, 1, 10, 19] == 0 and cb[0, 1, 10, 20] == 1, cb[0, 1, 10, 6:22]  # blob cols 9-14, r 5
    far, near = gt6.clone(), gt6.clone()
    far[0, 1, 54:60, 2:8] += 3 / 127.5     # CD3 glow of 3 levels, >= 5 px from every positive
    near[0, 1, 4:7, 10:16] += 3 / 127.5    # same glow, within r of the blob at rows 7-12
    l_far = float(bgl(rec_after_reg=far, gt=gt6))
    assert l_far > 0 and float(bgl(rec_after_reg=near, gt=gt6)) == 0.0, l_far
    exp_far = (36 * 3 / 30) / (float(cb[:, 1].sum()) + 1) / 3
    assert abs(l_far - exp_far) < 1e-4, (l_far, exp_far)  # known answer: 36 px at 3/30 over the certain background
    dapi_glow = gt6.clone(); dapi_glow[0, 2, 54:60, 2:8] += 3 / 127.5
    assert float(bgl(rec_after_reg=dapi_glow, gt=gt6)) == 0.0, "DAPI is not in the hinge"
    rq = far.clone().requires_grad_(True)
    bgl(rec_after_reg=rq, gt=gt6).backward()
    assert float(rq.grad[:, 2].abs().sum()) == 0.0 and float(rq.grad[0, 1, 54:60, 2:8].min()) > 0
    xg = gt6 * 0.3
    assert abs(float(c6g(rec_after_reg=xg, gt=gt6)) - float(c6(rec_after_reg=xg, gt=gt6)) - float(bgl(rec_after_reg=xg, gt=gt6))) < 1e-6
    l_gd9 = float(build("C6GD9", 1.0, "cpu")(rec_after_reg=x_thr, gt=gt6))
    l_parts = float(build("C6D9", 1.0, "cpu")(rec_after_reg=x_thr, gt=gt6)) + float(bgl(rec_after_reg=x_thr, gt=gt6))
    assert abs(l_gd9 - l_parts) < 1e-6 and float(bgl(rec_after_reg=x_thr, gt=gt6)) > 0, (l_gd9, l_parts)
    h = torch.randn(2, 8, 4, 4)
    assert float(_cos_loss(h, h * 3.0)) < 1e-6 and abs(float(_cos_loss(h, -h)) - 2) < 1e-5

    class _FakeFM(nn.Module):  # 2 prefix tokens + a 4x4 grid of width-6 tokens from 14-px patches
        num_features, num_prefix_tokens = 6, 2

        def __init__(self):
            super().__init__()
            self.pe = nn.Conv2d(3, 6, 14, 14)

        def forward_features(self, z):
            t = self.pe(z).flatten(2).transpose(1, 2)
            return torch.cat([t[:, :2] * 0, t], 1)

    class _Gen(nn.Module):
        def __init__(self):
            super().__init__()
            self.Conv5 = nn.Conv2d(3, 5, 16, 16)

        def forward(self, x):
            return self.Conv5(x).mean() + x

    fake = _FakeFM()
    st = torch.get_rng_state()
    c4 = FMAlign(1.0, "cpu", fm=fake, grid=4, in_ch=5)
    assert torch.equal(st, torch.get_rng_state()), "C4 construction must not consume global RNG"
    g = c4.wrap(_Gen())
    xx = torch.rand(2, 3, 64, 64) * 2 - 1
    g(xx)
    loss = c4(x=xx)
    loss.backward()
    assert c4.feat is None and 0 < float(loss) <= 2
    assert g.Conv5.weight.grad is not None and c4.proj[0].weight.grad is not None
    assert all(not q.requires_grad for q in c4.fm.parameters())
    with torch.no_grad():
        g(xx)
    assert c4.feat is None, "no-grad forwards (val scoring) must not be captured"
    return True


if __name__ == "__main__":
    print("self_test", self_test())
