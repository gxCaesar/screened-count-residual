"""Unified multi-backbone virtual-staining predictor zoo for HEMIT and Orion-CRC.

Wraps frozen virtual-staining generators across distinct architectural and objective
families into a deterministic ``predict_batch_u8(he_rgb_u8_nhwc) -> uint8 (N, C, H, W)``
contract without importing PyVIPS, PyTorch Lightning, W&B, or SlideVIPS:
  * Foundation ViT + Attention Matting Decoder : ``miphei_vit``, ``miphei_vit_hemit_native``
  * Foundation ViT + Multi-Scale LoRA UNETR    : ``unetr_hoptimus``
  * Modern Hierarchical ConvNet Attention U-Net: ``miphei_convnext``
  * Dual-Branch Swin-ResNet Hybrid             : ``hemit_orion``, ``miphei_hemit``, ``hemit_v1_native``
  * Conditional Adversarial cGAN               : ``pix2pix_orion``, ``dgr_released``
  * Conditional One-Step Latent Diffusion      : ``diffusion_ft``
  * Patch Sliding-Window ConvNeXt Regressor    : ``rosie_orion``
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ORION_16_MARKERS: tuple[str, ...] = (
    "Hoechst",
    "CD31",
    "CD45",
    "CD68",
    "CD4",
    "FOXP3",
    "CD8a",
    "CD45RO",
    "CD20",
    "PD-L1",
    "CD3e",
    "CD163",
    "E-cadherin",
    "Ki67",
    "Pan-CK",
    "SMA",
)
# Exact 16-channel slice into 17-band Orion TIFF (band 13 is PD-1, excluded by Sanofi)
ORION_TIFF_16_INDICES: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16,
)
ORION_NUCLEI_15_CLASSES: tuple[str, ...] = (
    "CD31_pos",
    "CD45_pos",
    "CD68_pos",
    "CD4_pos",
    "FOXP3_pos",
    "CD8a_pos",
    "CD45RO_pos",
    "CD20_pos",
    "PD-L1_pos",
    "CD3e_pos",
    "CD163_pos",
    "E-cadherin_pos",
    "Ki67_pos",
    "Pan-CK_pos",
    "SMA_pos",
)
# Channel indices within ORION_16_MARKERS corresponding to ORION_NUCLEI_15_CLASSES
ORION_CLASS_TO_CH16: dict[str, int] = {
    f"{m}_pos": idx for idx, m in enumerate(ORION_16_MARKERS) if m != "Hoechst"
}
# Projection of 16-ch Orion outputs onto HEMIT 3-ch order (panCK, CD3, DAPI/Hoechst)
ORION_TO_HEMIT_CH16: tuple[int, int, int] = (14, 10, 0)
HEMIT_3_MARKERS: tuple[str, str, str] = ("panCK", "CD3", "DAPI")


def _ensure_mipheivit_src(mipheivit_src_dir: str | Path) -> Path:
    src_path = Path(mipheivit_src_dir).resolve()
    if not src_path.is_dir():
        raise FileNotFoundError(f"mipheivit_src directory not found: {src_path}")
    s_str = str(src_path)
    if s_str not in sys.path:
        sys.path.insert(0, s_str)
    return src_path


def _load_benchmark_submodule(mipheivit_src_dir: str | Path, stem: str):
    """Load ``benchmark/models/<stem>.py`` directly without executing ``benchmark/models/__init__.py``."""
    src_path = _ensure_mipheivit_src(mipheivit_src_dir)
    mod_name = f"_sanofi_bench_{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    file_path = src_path / "benchmark" / "models" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _if_norm_to_u8(pred_nchw: torch.Tensor) -> np.ndarray:
    """Map Sanofi [-0.9, 0.9] IF space to uint8 [0, 255] matching NormalizationLayer.unormalize."""
    unorm = ((pred_nchw.to(torch.float32) + 0.9) * (255.0 / 1.8)).clamp(0.0, 255.0)
    return torch.round(unorm).to(torch.uint8).cpu().numpy()


def _unit_tanh_to_u8(pred_nchw: torch.Tensor) -> np.ndarray:
    """Map [-1, 1] Tanh space (0.9 * clamp(-1,1) in [-0.9, 0.9]) to uint8 [0, 255]."""
    unorm = ((pred_nchw.to(torch.float32).clamp(-1.0, 1.0) + 1.0) * 127.5).clamp(0.0, 255.0)
    return torch.round(unorm).to(torch.uint8).cpu().numpy()


def _adapt_hemit_statedict(state_dict: dict, model: nn.Module) -> dict:
    """Self-contained checkpoint key & Swin relative-bias adapter matching Sanofi utils.py."""
    from timm.layers import resample_patch_embed, resize_rel_pos_bias_table

    renamed = {}
    for k, v in state_dict.items():
        if ".downsample.norm" in k or "downsample.reduction" in k:
            parts = k.split(".")
            parts[2] = str(int(parts[2]) + 1)
            new_k = ".".join(parts)
        elif "relative_position_index" in k or "attn_mask" in k:
            continue
        else:
            new_k = k
        renamed[new_k] = v

    adapted = {}
    for k, v in renamed.items():
        if any(tok in k for tok in ("relative_position_index", "attn_mask")):
            continue
        if "swinT.patch_embed.proj.weight" in k and hasattr(model, "swinT"):
            _, _, h_t, w_t = model.swinT.patch_embed.proj.weight.shape
            if v.shape[-2] != h_t or v.shape[-1] != w_t:
                v = resample_patch_embed(v, (h_t, w_t), interpolation="bicubic", antialias=True)
        if k.endswith("relative_position_bias_table"):
            sub = model.get_submodule(k[:-29])
            if v.shape != sub.relative_position_bias_table.shape or sub.window_size[0] != sub.window_size[1]:
                v = resize_rel_pos_bias_table(
                    v,
                    new_window_size=sub.window_size,
                    new_bias_shape=sub.relative_position_bias_table.shape,
                )
        adapted[k] = v
    return adapted


class BackbonePredictor:
    """Uniform frozen virtual-staining backbone predictor returning uint8 (N, C, H, W)."""

    def __init__(
        self,
        name: str,
        family: str,
        marker_names: Sequence[str],
        device: torch.device | str,
        channel_slice: Sequence[int] | None = None,
    ) -> None:
        self.name = name
        self.family = family
        self.device = torch.device(device)
        self.channel_slice = tuple(channel_slice) if channel_slice is not None else None
        self.marker_names = tuple(marker_names)

    def _forward_full_u8(self, he_u8_nhwc: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @torch.no_grad()
    def predict_batch_u8(self, he_u8_nhwc: np.ndarray) -> np.ndarray:
        if he_u8_nhwc.ndim != 4 or he_u8_nhwc.shape[-1] != 3 or he_u8_nhwc.dtype != np.uint8:
            raise ValueError(f"Expected uint8 (N, H, W, 3), got {he_u8_nhwc.shape} {he_u8_nhwc.dtype}")
        out = self._forward_full_u8(he_u8_nhwc)
        if self.channel_slice is not None:
            out = out[:, list(self.channel_slice), :, :]
        return np.ascontiguousarray(out, dtype=np.uint8)


class SyntheticBackbonePredictor(BackbonePredictor):
    """Deterministic offline backbone predictor for CPU unit tests."""

    def __init__(
        self,
        name: str,
        family: str,
        marker_names: Sequence[str],
        attenuation: float = 0.62,
        bias_shift: float = -0.05,
        seed: int = 0,
    ) -> None:
        super().__init__(name=name, family=family, marker_names=marker_names, device="cpu")
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(0.2, 0.9, size=(len(marker_names), 3)).astype(np.float32)
        self.weights /= np.sum(self.weights, axis=1, keepdims=True)
        self.attenuation = float(attenuation)
        self.bias_shift = float(bias_shift)

    def _forward_full_u8(self, he_u8_nhwc: np.ndarray) -> np.ndarray:
        x = 1.0 - (he_u8_nhwc.astype(np.float32) / 255.0)
        # (N, H, W, 3) @ (3, C) -> (N, H, W, C) -> (N, C, H, W)
        raw = np.einsum("nhwc,mc->nmhw", x, self.weights)
        att = np.clip(self.attenuation * raw + self.bias_shift, 0.0, 1.0)
        return np.round(att * 255.0).astype(np.uint8)


class SanofiMIPHEIPredictor(BackbonePredictor):
    """Wraps Sanofi MIPHEI-vit, UNETR-hoptimus, MIPHEI-convnext, and MIPHEI_HEMIT."""

    def __init__(
        self,
        *,
        name: str,
        family: str,
        checkpoint_dir: str | Path,
        mipheivit_src_dir: str | Path,
        device: torch.device | str,
        img_size: int = 512,
        hoptimus0_ckpt_path: str | Path | None = None,
        channel_slice: Sequence[int] | None = None,
        marker_names: Sequence[str] = ORION_16_MARKERS,
    ) -> None:
        super().__init__(name, family, marker_names, device, channel_slice=channel_slice)
        _ensure_mipheivit_src(mipheivit_src_dir)
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from src.generators import foundation_models, get_generator
        from src.generators.hemit_models import resize_embed_hemit_statedict

        ckpt_dir = Path(checkpoint_dir).resolve()
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        cfg.train.foreground_head = bool(cfg.model.get("foreground_head", False))

        # Allow local H-optimus-0 weights override without touching HuggingFace hub
        if hoptimus0_ckpt_path is not None and cfg.model.encoder.get("encoder_name") == "hoptimus0":
            h_path = str(Path(hoptimus0_ckpt_path).resolve())
            if cfg.model.model_name.startswith("mipheivit"):
                cfg.model.encoder.encoder_weights = h_path
            else:
                orig_fn = foundation_models.FOUNDATION_MODEL_REGISTRY["hoptimus0"]
                foundation_models.FOUNDATION_MODEL_REGISTRY["hoptimus0"] = (
                    lambda sz, pretrained=True, ckpt_path=None, drop_path_rate=0.0, global_pool="":
                    orig_fn(sz, pretrained=True, ckpt_path=h_path, drop_path_rate=drop_path_rate, global_pool=global_pool)
                )
        # Avoid downloading ImageNet weights for smp_unet since model.safetensors overrides all params
        if cfg.model.model_name.startswith("smp_unet"):
            cfg.model.encoder.encoder_weights = None

        nc_out = len(cfg.data.targ_channel_names)
        model = get_generator(cfg.model.model_name, img_size, 3, nc_out, cfg)
        st_path = ckpt_dir / "model.safetensors"
        if st_path.exists():
            state_dict = load_file(str(st_path), device="cpu")
            strict_load = False
        else:
            ckpt_path = ckpt_dir / "model.weights.ckpt"
            raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            raw_sd = raw_ckpt["state_dict"] if "state_dict" in raw_ckpt else raw_ckpt
            state_dict = {
                k[len("netG.") :] if k.startswith("netG.") else k: v
                for k, v in raw_sd.items()
                if not k.startswith(("netD.", "criterion", "loss"))
            }
            strict_load = True
        if hasattr(model, "swinT"):
            state_dict = resize_embed_hemit_statedict(state_dict, model)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in {ckpt_dir}: {unexpected[:5]}")
        self.has_swint_topk = hasattr(model, "swinT")
        self.model = model.to(self.device).eval().half()
        self.mean = torch.tensor(list(cfg.data.normalization.mean), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(list(cfg.data.normalization.std), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _forward_full_u8(self, he_u8_nhwc: np.ndarray) -> np.ndarray:
        if self.has_swint_topk and he_u8_nhwc.shape[0] > 1:
            return np.concatenate([self._forward_full_u8(he_u8_nhwc[i : i + 1]) for i in range(he_u8_nhwc.shape[0])], axis=0)
        x = torch.from_numpy(he_u8_nhwc).to(device=self.device, dtype=torch.float32).permute(0, 3, 1, 2)
        x = ((x - self.mean) / self.std).half()
        out = self.model(x).float()
        return _if_norm_to_u8(out)


class SanofiPix2PixOrHEMITPredictor(BackbonePredictor):
    """Wraps Sanofi Pix2Pix (UnetGenerator) and HEMIT (ResnetGeneratorSwinT) checkpoints."""

    def __init__(
        self,
        *,
        name: str,
        family: str,
        kind: str,
        checkpoint_dir: str | Path,
        mipheivit_src_dir: str | Path,
        device: torch.device | str,
        img_size: int = 512,
        channel_slice: Sequence[int] | None = None,
        marker_names: Sequence[str] = ORION_16_MARKERS,
    ) -> None:
        super().__init__(name, family, marker_names, device, channel_slice=channel_slice)
        _ensure_mipheivit_src(mipheivit_src_dir)
        from omegaconf import OmegaConf
        UnetGenerator = _load_benchmark_submodule(mipheivit_src_dir, "pix2pix").UnetGenerator
        from src.generators.hemit_models import ResnetGeneratorSwinT, resize_embed_hemit_statedict

        ckpt_dir = Path(checkpoint_dir).resolve()
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        nc_out = len(cfg.data.targ_channel_names)
        if kind == "pix2pix":
            model = UnetGenerator(3, nc_out, 8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
            sd = torch.load(ckpt_dir / "best.pth", map_location="cpu", weights_only=True)
            model.load_state_dict(sd, strict=True)
        elif kind == "hemit":
            model = ResnetGeneratorSwinT(
                input_nc=3,
                output_nc=nc_out,
                img_size=[img_size, img_size],
                patch_size=4,
                window_size=8,
                depths=[2, 2, 6, 2],
                embed_dim=96,
            )
            sd = torch.load(ckpt_dir / "best.pth", map_location="cpu", weights_only=True)
            sd = _adapt_hemit_statedict(sd, model)
            sd = resize_embed_hemit_statedict(sd, model)
            model.load_state_dict(sd, strict=True)
        else:
            raise ValueError(f"Unknown kind: {kind!r}")

        self.has_swint_topk = hasattr(model, "swinT")
        self.model = model.to(self.device).eval()
        self.mean = torch.tensor(list(cfg.data.normalization.mean), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(list(cfg.data.normalization.std), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _forward_full_u8(self, he_u8_nhwc: np.ndarray) -> np.ndarray:
        if self.has_swint_topk and he_u8_nhwc.shape[0] > 1:
            return np.concatenate([self._forward_full_u8(he_u8_nhwc[i : i + 1]) for i in range(he_u8_nhwc.shape[0])], axis=0)
        x = torch.from_numpy(he_u8_nhwc).to(device=self.device, dtype=torch.float32).permute(0, 3, 1, 2)
        x = (x - self.mean) / self.std
        out = self.model(x)
        return _unit_tanh_to_u8(out)


class SanofiDiffusionFTPredictor(BackbonePredictor):
    """Wraps Sanofi DiffusionFT One-Step Conditional Latent Diffusion (SD2.1) pipeline."""

    def __init__(
        self,
        *,
        name: str,
        family: str,
        checkpoint_dir: str | Path,
        mipheivit_src_dir: str | Path,
        device: torch.device | str,
        channel_slice: Sequence[int] | None = None,
        marker_names: Sequence[str] = ORION_16_MARKERS,
    ) -> None:
        super().__init__(name, family, marker_names, device, channel_slice=channel_slice)
        _ensure_mipheivit_src(mipheivit_src_dir)
        from omegaconf import OmegaConf
        load_diffusion_pipeline = _load_benchmark_submodule(mipheivit_src_dir, "diffusionft").load_diffusion_pipeline

        ckpt_dir = Path(checkpoint_dir).resolve()
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        self.pipe = load_diffusion_pipeline(str(ckpt_dir), self.device)
        self.marker_ids = list(channel_slice) if channel_slice is not None else None
        # Note: if marker_ids are passed directly to pipe, channel_slice is already applied!
        self._sliced_in_pipe = self.marker_ids is not None
        if self._sliced_in_pipe:
            self.channel_slice = None
        self.mean = torch.tensor(list(cfg.data.normalization.mean), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(list(cfg.data.normalization.std), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _forward_full_u8(self, he_u8_nhwc: np.ndarray) -> np.ndarray:
        # Process 1..2 tiles per forward pass to keep peak VRAM small when decoding 16 markers
        outs_list = []
        for i in range(he_u8_nhwc.shape[0]):
            x = torch.from_numpy(he_u8_nhwc[i : i + 1]).to(device=self.device, dtype=torch.float32).permute(0, 3, 1, 2)
            x = (x - self.mean) / self.std
            # HE2MIFOneStepPipeline(..., return_torch=True) -> (B, M, H, W) in [0, 1]
            pred01 = self.pipe(x, marker_ids=self.marker_ids, return_torch=True)
            if pred01.ndim == 5:
                pred01 = pred01[:, :, 0, :, :]
            u8 = torch.round((pred01.clamp(0.0, 1.0) * 255.0)).to(torch.uint8).cpu().numpy()
            outs_list.append(u8)
        return np.concatenate(outs_list, axis=0)


class SanofiRosiePredictor(BackbonePredictor):
    """Fast GPU-batched ConvNeXt-Small sliding-window predictor matching Rosie-ORION."""

    def __init__(
        self,
        *,
        name: str,
        family: str,
        checkpoint_dir: str | Path,
        mipheivit_src_dir: str | Path,
        device: torch.device | str,
        crop_chunk: int = 512,
        channel_slice: Sequence[int] | None = None,
        marker_names: Sequence[str] = ORION_16_MARKERS,
    ) -> None:
        super().__init__(name, family, marker_names, device, channel_slice=channel_slice)
        _ensure_mipheivit_src(mipheivit_src_dir)
        from omegaconf import OmegaConf
        get_rosie_model = _load_benchmark_submodule(mipheivit_src_dir, "rosie").get_model

        ckpt_dir = Path(checkpoint_dir).resolve()
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        nc_out = len(cfg.data.targ_channel_names)
        model = get_rosie_model(num_outputs=nc_out)
        sd = torch.load(ckpt_dir / "rosie.pth", map_location="cpu", weights_only=True)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        self.model = model.to(self.device).eval().half()
        self.nc_out = nc_out
        self.crop_chunk = int(crop_chunk)
        self.mean = torch.tensor(list(cfg.data.normalization.mean), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(list(cfg.data.normalization.std), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _forward_full_u8(self, he_u8_nhwc: np.ndarray) -> np.ndarray:
        # Mirror centered 128x128 patch context at stride 8 across full HxW via reflect-64 padding
        # so every 8x8 superpixel has exact centered 128x128 context without zero-border cutoff!
        p_sz, stride = 128, 8
        pad = p_sz // 2
        n, h, w, _ = he_u8_nhwc.shape
        x_full = torch.from_numpy(he_u8_nhwc).to(device=self.device, dtype=torch.float32).permute(0, 3, 1, 2)
        x_norm = ((x_full - self.mean) / self.std).half()
        x_pad = F.pad(x_norm, (pad, pad - stride, pad, pad - stride), mode="reflect")
        # Unfold into (N, 3*128*128, L) where L = (H//8) * (W//8)
        gh, gw = h // stride, w // stride
        outs = np.empty((n, self.nc_out, h, w), dtype=np.uint8)

        coords = [(r * stride, c * stride) for r in range(gh) for c in range(gw)]
        for bi in range(n):
            xb = x_pad[bi : bi + 1]
            preds_grid = []
            for start in range(0, len(coords), self.crop_chunk):
                sub = coords[start : start + self.crop_chunk]
                crops = torch.cat([xb[:, :, ry : ry + p_sz, cx : cx + p_sz] for ry, cx in sub], dim=0)
                crops_224 = F.interpolate(crops, size=(224, 224), mode="bilinear", align_corners=False)
                out_c = self.model(crops_224).float().clamp(0.0, 1.0)
                preds_grid.append(out_c)
            grid = torch.cat(preds_grid, dim=0).transpose(0, 1).contiguous().view(1, self.nc_out, gh, gw)
            up = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False)[0]
            outs[bi] = torch.round((up * 255.0).clamp(0.0, 255.0)).to(torch.uint8).cpu().numpy()
        return outs
