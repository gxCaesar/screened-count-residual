"""Multi-backbone plug-and-play rescue + single-cell nuclear benchmark on HEMIT.

Evaluates frozen virtual-staining generators across distinct architectural and
objective families on HEMIT (3-plex mIF: panCK, CD3, DAPI at 512x512 center crop)
paired with Cellpose single-cell nuclear instance masks and positivity labels:
  * Upstream frozen backbone output
  * +S_recon (seed-paired normalized reconstruction + edge residual control)
  * +S_count (gauge-fixed Screened Count Residual with detached-table pooled
              spatial-tolerance calibration and automatic gradient-norm balance)

Strict split containment:
  * ``s5_val``: fits residual heads strictly on ``FIT1192`` (train), calibrates
    thresholds strictly on ``FIT1192``, saves frozen head weights, and evaluates
    ``Val (630)``. Hard-blocks any read of ``test``.
  * ``s6_locked_test``: requires ``protocol_status == 'frozen'`` in
    ``benchmark-protocol.json``, runs ``new_fits == 0``, loads frozen ``FIT1192``
    head weights and thresholds, and evaluates ``Test (945)`` across all 15
    coordinate-prefix regions plus Cellpose single-cell phenotyping & spatial
    epithelial-immune infiltration concordance.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import time
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
try:
    import tifffile as _tifffile
except ModuleNotFoundError:
    _tifffile = None


def _imread_tiff(path):
    if _tifffile is not None:
        try:
            return _tifffile.imread(path)
        except ValueError:
            pass
    with Image.open(path) as im:
        n_frames = getattr(im, "n_frames", 1)
        if n_frames <= 1:
            return np.asarray(im)
        frames = []
        for idx in range(n_frames):
            im.seek(idx)
            frames.append(np.asarray(im))
        return np.stack(frames, axis=-1)
import torch

from . import backbone_zoo as zoo
from . import clinical_cell_survival as ccs
from . import journal_cohort_eval as jce
from . import screened_universal_head as suh

HEMIT_MARKERS = ("panCK", "CD3", "DAPI")
HEMIT_CUTOFFS = (22, 30, 18)
HEMIT_CELL_COL_MAP = {
    "panCK": ("Pan-CK", "Pan-CK_pos", 0),
    "CD3": ("CD3", "CD3_pos", 1),
}


def prefix_of(name: str) -> str:
    if "_patch_" in name:
        return name.split("_patch_")[0]
    return name.rsplit("_", 1)[0]


def load_hemit_split_arrays(
    hemit_root: Path,
    nuclei_root: Path | None,
    split_dir_name: str,
    tile_names: Sequence[str],
    *,
    forbid_test: bool,
) -> dict:
    """Load centered 512x512 H&E, mIF target, and optional Cellpose nuclei masks/labels."""
    if forbid_test and split_dir_name == "test":
        raise RuntimeError("Holdout containment violation: 'test' split access forbidden in S5 mode")

    n = len(tile_names)
    he_list = []
    targ_list = []
    clusters = [prefix_of(nm) for nm in tile_names]

    in_dir = hemit_root / split_dir_name / "input"
    lb_dir = hemit_root / split_dir_name / "label"

    for i, nm in enumerate(tile_names):
        p_in = in_dir / nm
        p_lb = lb_dir / nm
        if forbid_test and ("/test/" in str(p_in) or "/test/" in str(p_lb)):
            raise RuntimeError(f"Forbidden test path: {p_in}")
        with Image.open(p_in) as im_in:
            arr_in = np.asarray(im_in.convert("RGB"), dtype=np.uint8)
        with Image.open(p_lb) as im_lb:
            arr_lb = np.asarray(im_lb.convert("RGB"), dtype=np.uint8)
        if arr_in.shape == (1024, 1024, 3):
            arr_in = arr_in[256:768, 256:768]
            arr_lb = arr_lb[256:768, 256:768]
        elif arr_in.ndim != 3 or arr_in.shape[2] != 3 or arr_in.shape != arr_lb.shape:
            raise ValueError(f"Unexpected HEMIT image shape {arr_in.shape} vs {arr_lb.shape} for {nm}")
        he_list.append(arr_in)
        targ_list.append(np.transpose(arr_lb, (2, 0, 1)))
    he_nhwc = np.stack(he_list, axis=0)
    targ_nchw = np.stack(targ_list, axis=0)
    exp_hw = he_nhwc.shape[1:3]

    nuclei_records = []
    if nuclei_root is not None and (nuclei_root / split_dir_name / "mask").is_dir():
        mask_dir = nuclei_root / split_dir_name / "mask"
        csv_dir = nuclei_root / split_dir_name / "csv"
        for i, nm in enumerate(tile_names):
            stem = Path(nm).stem
            m_path = mask_dir / nm
            if not m_path.exists():
                m_path = mask_dir / f"{stem}.tif"
            c_path = csv_dir / f"{stem}.csv"
            if not m_path.exists():
                nuclei_records.append(None)
                continue
            mask_full = _imread_tiff(m_path)
            if mask_full.shape == (1024, 1024) and exp_hw == (512, 512):
                mask_crop = mask_full[256:768, 256:768].astype(np.int32, copy=False)
            elif mask_full.shape == exp_hw:
                mask_crop = mask_full.astype(np.int32, copy=False)
            else:
                raise ValueError(f"Unexpected HEMIT mask shape {mask_full.shape} for {nm}")

            gt_nuc = ccs.extract_tile_nuclei(targ_nchw[i], mask_crop, min_area=10)
            if len(gt_nuc["cell_ids"]) == 0 or not c_path.exists():
                nuclei_records.append(None)
                continue
            df_c = pd.read_csv(c_path)
            if len(df_c) == 0 or "label" not in df_c.columns:
                nuclei_records.append(None)
                continue
            df_idx = df_c.drop_duplicates("label").set_index("label")
            valid_mask = np.isin(gt_nuc["cell_ids"], df_idx.index.to_numpy())
            if not np.any(valid_mask):
                nuclei_records.append(None)
                continue
            kept_ids = gt_nuc["cell_ids"][valid_mask]
            sub_df = df_idx.loc[kept_ids]
            idx_spec = ccs.prepare_tile_nuclei_index(mask_crop, kept_ids)
            nuclei_records.append(
                dict(
                    tile_idx=i,
                    tile_name=nm,
                    cluster=clusters[i],
                    mask_hw=mask_crop,
                    idx_spec=idx_spec,
                    cell_ids=kept_ids,
                    centroids_yx=gt_nuc["centroids_yx"][valid_mask],
                    gt_means=gt_nuc["means"][valid_mask],
                    panck_pos=sub_df["Pan-CK_pos"].astype(np.int32).to_numpy(),
                    cd3_pos=sub_df["CD3_pos"].astype(np.int32).to_numpy(),
                )
            )
    else:
        nuclei_records = [None] * n

    return dict(
        names=list(tile_names),
        clusters=clusters,
        he_nhwc=he_nhwc,
        targ_nchw=targ_nchw,
        nuclei_records=nuclei_records,
    )


def extract_pred_cell_means_for_records(
    pred_nchw: np.ndarray,
    nuclei_records: Sequence[dict | None],
) -> dict[str, np.ndarray]:
    """Extract aligned per-nucleus mean intensities from predicted (N, 3, 512, 512) mIF."""
    tile_idxs = []
    centroids_list = []
    gt_means_list = []
    pred_means_list = []
    panck_pos_list = []
    cd3_pos_list = []

    for rec in nuclei_records:
        if rec is None:
            continue
        ti = int(rec["tile_idx"])
        if "idx_spec" in rec:
            p_means = ccs.extract_means_with_index(pred_nchw[ti], rec["idx_spec"])
        else:
            nuc = ccs.extract_tile_nuclei(pred_nchw[ti], rec["mask_hw"], min_area=10)
            id_to_pos = {int(cid): idx for idx, cid in enumerate(nuc["cell_ids"].tolist())}
            rows = [id_to_pos[int(cid)] for cid in rec["cell_ids"].tolist()]
            p_means = nuc["means"][rows]
        k = len(rec["cell_ids"])
        tile_idxs.append(np.full(k, ti, dtype=np.int64))
        centroids_list.append(rec["centroids_yx"])
        gt_means_list.append(rec["gt_means"])
        pred_means_list.append(p_means)
        panck_pos_list.append(rec["panck_pos"])
        cd3_pos_list.append(rec["cd3_pos"])

    if not tile_idxs:
        return dict(
            tile_ids=np.zeros((0,), dtype=np.int64),
            centroids_yx=np.zeros((0, 2), dtype=np.float64),
            gt_means=np.zeros((0, 3), dtype=np.float64),
            pred_means=np.zeros((0, 3), dtype=np.float64),
            panck_pos=np.zeros((0,), dtype=np.int32),
            cd3_pos=np.zeros((0,), dtype=np.int32),
        )

    return dict(
        tile_ids=np.concatenate(tile_idxs, axis=0),
        centroids_yx=np.concatenate(centroids_list, axis=0),
        gt_means=np.concatenate(gt_means_list, axis=0),
        pred_means=np.concatenate(pred_means_list, axis=0),
        panck_pos=np.concatenate(panck_pos_list, axis=0),
        cd3_pos=np.concatenate(cd3_pos_list, axis=0),
    )


def evaluate_hemit_arm_cells_and_infiltration(
    train_cells: dict[str, np.ndarray],
    eval_cells: dict[str, np.ndarray],
    eval_nuclei_records: Sequence[dict | None],
    eval_pred_nchw: np.ndarray,
) -> dict:
    """Compute single-cell F1/AUPRC/count concordance + Pan-CK+->CD3+ infiltration concordance."""
    if len(eval_cells["tile_ids"]) == 0 or len(train_cells["tile_ids"]) == 0:
        return dict(available=False)

    out_markers = {}
    tau_phys_map = {}
    for m_name, col_idx, pos_key in (("panCK", 0, "panck_pos"), ("CD3", 1, "cd3_pos")):
        res_m = ccs.evaluate_single_cell_marker(
            train_gt_scores=train_cells["gt_means"][:, col_idx],
            train_pred_scores=train_cells["pred_means"][:, col_idx],
            train_labels=train_cells[pos_key],
            eval_pred_scores=eval_cells["pred_means"][:, col_idx],
            eval_labels=eval_cells[pos_key],
            eval_tile_ids=eval_cells["tile_ids"],
        )
        out_markers[m_name] = res_m
        tau_phys_map[m_name] = float(res_m["tau_physical"])

    # Spatial Pan-CK+ -> CD3+ nearest-neighbor infiltration concordance across co-positive tiles
    gt_nn_list, pred_nn_list = [], []
    gt_w25_list, pred_w25_list = [], []
    for rec in eval_nuclei_records:
        if rec is None:
            continue
        ti = int(rec["tile_idx"])
        if "idx_spec" in rec:
            pm = ccs.extract_means_with_index(eval_pred_nchw[ti], rec["idx_spec"])
        else:
            nuc = ccs.extract_tile_nuclei(eval_pred_nchw[ti], rec["mask_hw"], min_area=10)
            id_to_pos = {int(cid): idx for idx, cid in enumerate(nuc["cell_ids"].tolist())}
            rows = [id_to_pos[int(cid)] for cid in rec["cell_ids"].tolist()]
            pm = nuc["means"][rows]
        pred_pck = (pm[:, 0] > tau_phys_map["panCK"]).astype(np.int32)
        pred_cd3 = (pm[:, 1] > tau_phys_map["CD3"]).astype(np.int32)

        gt_inf = ccs.compute_tile_infiltration_metrics(
            rec["centroids_yx"], rec["panck_pos"], rec["cd3_pos"], pixel_size_um=0.46
        )
        pr_inf = ccs.compute_tile_infiltration_metrics(
            rec["centroids_yx"], pred_pck, pred_cd3, pixel_size_um=0.46
        )
        if np.isfinite(gt_inf["median_nn_um"]):
            gt_w25_list.append(gt_inf["within_25um_frac"])
            pred_w25_list.append(pr_inf["within_25um_frac"])
            if np.isfinite(pr_inf["median_nn_um"]):
                gt_nn_list.append(gt_inf["median_nn_um"])
                pred_nn_list.append(pr_inf["median_nn_um"])

    w25_r = (
        float(stats.pearsonr(gt_w25_list, pred_w25_list)[0])
        if len(gt_w25_list) >= 3 and np.std(gt_w25_list) > 1e-12 and np.std(pred_w25_list) > 1e-12
        else 0.0
    )
    w25_mae = (
        float(np.mean(np.abs(np.asarray(pred_w25_list) - np.asarray(gt_w25_list))))
        if gt_w25_list
        else 0.0
    )
    nn_mae = (
        float(np.mean(np.abs(np.asarray(pred_nn_list) - np.asarray(gt_nn_list))))
        if gt_nn_list
        else float("nan")
    )

    macro_deploy_f1 = 0.5 * (
        out_markers["panCK"]["deployable_cell_f1"] + out_markers["CD3"]["deployable_cell_f1"]
    )
    macro_recal_f1 = 0.5 * (
        out_markers["panCK"]["recalibrated_cell_f1"] + out_markers["CD3"]["recalibrated_cell_f1"]
    )
    macro_auprc = 0.5 * (out_markers["panCK"]["cell_auprc"] + out_markers["CD3"]["cell_auprc"])

    return dict(
        available=True,
        macro_deployable_cell_f1=macro_deploy_f1,
        macro_recalibrated_cell_f1=macro_recal_f1,
        macro_cell_auprc=macro_auprc,
        cell_mismatch_gap=macro_recal_f1 - macro_deploy_f1,
        per_marker=out_markers,
        spatial_infiltration=dict(
            n_copositive_tiles=len(gt_w25_list),
            n_detected_copositive_tiles=len(gt_nn_list),
            within_25um_pearson_r=w25_r,
            within_25um_mae=w25_mae,
            median_nn_um_mae=nn_mae if np.isfinite(nn_mae) else -1.0,
        ),
    )


def _atomic_torch_save(obj: object, path: Path) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(text)
    os.replace(tmp, path)


def build_common12_for_3ch(
    he_nhwc: np.ndarray,
    donor_nchw: np.ndarray,
    ch: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Construct (N, 12, H, W) float16 coordinate features in chunks to halve peak CPU RAM."""
    dev = torch.device(device)
    n, _, h, w = donor_nchw.shape
    out = torch.empty((n, 12, h, w), dtype=torch.float16)
    chunk = 64
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            he_t = torch.from_numpy(he_nhwc[s:e]).to(dev, non_blocking=True).permute(0, 3, 1, 2).to(torch.float32) / 255.0
            d_all = torch.from_numpy(donor_nchw[s:e]).to(dev, non_blocking=True).to(torch.float32) / 255.0
            bc = d_all[:, ch : ch + 1]
            bs = d_all[:, 2:3]  # DAPI structural reference
            bm = d_all.mean(dim=1, keepdim=True)
            out[s:e] = suh.build_universal_common12(he_t, bc, bs, bm).to(torch.float16).cpu()
    return out


class NativeHEMITBank:
    """Sequential low-VRAM loader for native HEMIT-trained generators (DGR, UniStainNet, HEMIT-FML, Hybrid Donor)."""

    def __init__(self, native_cfg_path: Path, device: torch.device) -> None:
        from . import pixel_comparator as pc
        from . import query_relation_scaled_data as qdata
        from . import run_pixel_comparator as rpc

        cfg = json.loads(native_cfg_path.read_text())
        self.cfg = cfg
        self.pc = pc
        self.qdata = qdata
        self.rpc = rpc
        self.device = device
        self.rt = rpc.runtime(cfg["native"])

    def _infer_single_model_f32(self, name: str, model: torch.nn.Module, he_nhwc: np.ndarray, batch_size: int) -> np.ndarray:
        n, h, w, _ = he_nhwc.shape
        out_f32 = np.empty((n, 3, h, w), dtype=np.float32)
        dev = self.device
        try:
            model = model.to(dev).eval()
        except torch.OutOfMemoryError:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            dev = torch.device("cpu")
            model = model.to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        s = 0
        cur_bs = batch_size
        with torch.inference_mode():
            while s < n:
                e = min(n, s + cur_bs)
                he_01 = he_nhwc[s:e].astype(np.float32) / 255.0
                he_bchw = np.transpose((he_01 - 0.5) / 0.5, (0, 3, 1, 2)).astype(np.float32)
                try:
                    he_t = torch.from_numpy(he_bchw).to(dev)
                    preds = self.rpc.predictions_of({name: model}, he_t)
                except torch.OutOfMemoryError:
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    if cur_bs > 1:
                        cur_bs = max(1, cur_bs // 2)
                        continue
                    dev = torch.device("cpu")
                    model = model.to(dev).eval()
                    he_t = torch.from_numpy(he_bchw).to(dev)
                    preds = self.rpc.predictions_of({name: model}, he_t)
                out_f32[s:e] = np.transpose(preds[name], (0, 3, 1, 2))
                s = e
        return out_f32

    @torch.no_grad()
    def predict_all_native(
        self,
        he_nhwc: np.ndarray,
        batch_size: int = 16,
        only_keys: set[str] | None = None,
        save_callback=None,
    ) -> dict[str, np.ndarray]:
        n, h, w, _ = he_nhwc.shape
        wanted = set(only_keys) if only_keys is not None else {
            "native_dgr", "native_unistainnet_uni2h", "native_hemit_fml", "native_hybrid_donor"
        }
        out: dict[str, np.ndarray] = {}

        # 1. DGR
        if "native_dgr" in wanted:
            m_dgr = self.rt["mp"].load_gen(self.cfg["dgr_checkpoint"], self.device).eval()
            dgr_f32 = self._infer_single_model_f32("DGR", m_dgr, he_nhwc, batch_size=batch_size)
            del m_dgr
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            arr_dgr = np.empty((n, 3, h, w), dtype=np.uint8)
            for i in range(n):
                arr_dgr[i] = self.pc.quantize(dgr_f32[i])
            del dgr_f32
            out["native_dgr"] = arr_dgr
            if save_callback is not None:
                save_callback("native_dgr", arr_dgr)

        # 2. F0 (HEMIT-FML) -- needed for native_hemit_fml OR native_hybrid_donor
        need_f0 = ("native_hemit_fml" in wanted) or ("native_hybrid_donor" in wanted)
        f0_f32 = None
        if need_f0:
            m_f0 = self.rt["mp"].load_gen(self.cfg["native"]["f0_checkpoint"], self.device).eval()
            f0_f32 = self._infer_single_model_f32("F0", m_f0, he_nhwc, batch_size=batch_size)
            del m_f0
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if "native_hemit_fml" in wanted:
                arr_f0 = np.empty((n, 3, h, w), dtype=np.uint8)
                for i in range(n):
                    arr_f0[i] = self.pc.quantize(f0_f32[i])
                out["native_hemit_fml"] = arr_f0
                if save_callback is not None:
                    save_callback("native_hemit_fml", arr_f0)

        # 3. UNI2IF (UNI2-h) + exact unrounded blend for Hybrid Donor
        need_uni = ("native_unistainnet_uni2h" in wanted) or ("native_hybrid_donor" in wanted)
        if need_uni:
            uni_cfg = dict(self.cfg["native"], seeds={"0": {"UNI2IF": self.cfg["native"]["uni_checkpoint"]}})
            m_uni = self.rpc.load_models(uni_cfg, self.rt, "0", self.device)["UNI2IF"]
            u_f32 = self._infer_single_model_f32("UNI2IF", m_uni, he_nhwc, batch_size=min(8, batch_size))
            del m_uni
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if "native_unistainnet_uni2h" in wanted:
                arr_uni = np.empty((n, 3, h, w), dtype=np.uint8)
                for i in range(n):
                    arr_uni[i] = self.pc.quantize(u_f32[i])
                out["native_unistainnet_uni2h"] = arr_uni
                if save_callback is not None:
                    save_callback("native_unistainnet_uni2h", arr_uni)
            if "native_hybrid_donor" in wanted and f0_f32 is not None:
                arr_hyb = np.empty((n, 3, h, w), dtype=np.uint8)
                for i in range(n):
                    arr_hyb[i] = self.pc.quantize(
                        np.float32(0.75) * u_f32[i] + np.float32(0.25) * f0_f32[i]
                    )
                out["native_hybrid_donor"] = arr_hyb
                if save_callback is not None:
                    save_callback("native_hybrid_donor", arr_hyb)
            del u_f32
        del f0_f32
        gc.collect()
        return out


HEMIT_SANOFI_FAMILIES: dict[str, str] = {
    "synthetic_vit": "Foundation ViT",
    "synthetic_gan": "Adversarial cGAN",
    "sanofi_miphei_vit": "Foundation ViT + Attention Matting",
    "sanofi_unetr_hoptimus": "Foundation ViT + LoRA UNETR",
    "sanofi_miphei_convnext": "ConvNeXt-Large + Attention U-Net",
    "sanofi_miphei_hemit": "Weighted-MSE Swin-ResNet",
    "sanofi_hemit_swin": "Adversarial Swin-ResNet",
    "sanofi_pix2pix": "Conditional Adversarial U-Net cGAN",
    "sanofi_diffusion_ft": "One-Step Latent Diffusion (SD2.1)",
    "sanofi_rosie": "Sliding-Window ConvNeXt Regressor",
}


def instantiate_sanofi_or_synthetic(
    key: str,
    sanofi_ckpt_root: Path | None,
    mipheivit_src: Path | None,
    hoptimus0_ckpt: Path | None,
    device: torch.device,
) -> zoo.BackbonePredictor:
    sl = zoo.ORION_TO_HEMIT_CH16
    mk = HEMIT_MARKERS
    if key == "synthetic_vit":
        return zoo.SyntheticBackbonePredictor("synthetic_vit", "Foundation ViT", mk, attenuation=0.58, seed=11)
    if key == "synthetic_gan":
        return zoo.SyntheticBackbonePredictor("synthetic_gan", "Adversarial cGAN", mk, attenuation=0.64, seed=29)
    if sanofi_ckpt_root is None or mipheivit_src is None:
        raise ValueError(f"sanofi_ckpt_root and mipheivit_src required for {key}")

    mapping = {
        "sanofi_miphei_vit": ("miphei", "Foundation ViT + Attention Matting", "MIPHEI-vit"),
        "sanofi_unetr_hoptimus": ("miphei", "Foundation ViT + LoRA UNETR", "UNETR-hoptimus"),
        "sanofi_miphei_convnext": ("miphei", "ConvNeXt-Large + Attention U-Net", "MIPHEI-convnext"),
        "sanofi_miphei_hemit": ("miphei", "Weighted-MSE Swin-ResNet", "MIPHEI_HEMIT"),
        "sanofi_hemit_swin": ("hemit", "Adversarial Swin-ResNet", "HEMIT"),
        "sanofi_pix2pix": ("pix2pix", "Conditional Adversarial U-Net cGAN", "Pix2Pix"),
        "sanofi_diffusion_ft": ("diffusion", "One-Step Latent Diffusion (SD2.1)", "DiffusionFT"),
        "sanofi_rosie": ("rosie", "Sliding-Window ConvNeXt Regressor", "Rosie-ORION"),
    }
    if key not in mapping:
        raise ValueError(f"Unknown backbone key: {key!r}")
    kind, fam, sub = mapping[key]
    cdir = sanofi_ckpt_root / sub
    if kind == "miphei":
        return zoo.SanofiMIPHEIPredictor(
            name=key,
            family=fam,
            checkpoint_dir=cdir,
            mipheivit_src_dir=mipheivit_src,
            device=device,
            img_size=512,
            hoptimus0_ckpt_path=hoptimus0_ckpt,
            channel_slice=sl,
            marker_names=mk,
        )
    if kind in ("hemit", "pix2pix"):
        return zoo.SanofiPix2PixOrHEMITPredictor(
            name=key,
            family=fam,
            kind=kind,
            checkpoint_dir=cdir,
            mipheivit_src_dir=mipheivit_src,
            device=device,
            img_size=512,
            channel_slice=sl,
            marker_names=mk,
        )
    if kind == "diffusion":
        return zoo.SanofiDiffusionFTPredictor(
            name=key,
            family=fam,
            checkpoint_dir=cdir,
            mipheivit_src_dir=mipheivit_src,
            device=device,
            channel_slice=sl,
            marker_names=mk,
        )
    return zoo.SanofiRosiePredictor(
        name=key,
        family=fam,
        checkpoint_dir=cdir,
        mipheivit_src_dir=mipheivit_src,
        device=device,
        channel_slice=sl,
        marker_names=mk,
    )


def predict_in_batches(predictor: zoo.BackbonePredictor, he_nhwc: np.ndarray, batch_size: int = 8) -> np.ndarray:
    n, h, w, _ = he_nhwc.shape
    c = len(predictor.marker_names)
    out = np.empty((n, c, h, w), dtype=np.uint8)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        out[s:e] = predictor.predict_batch_u8(he_nhwc[s:e])
    return out


def run_hemit_campaign(args: argparse.Namespace) -> dict:
    t0 = time.monotonic()
    device = torch.device(args.device)
    split_doc = json.loads(Path(args.split_manifest).read_text())
    parts = split_doc["partitions"]
    fit_names_full = list(parts["fit1192"])
    fit_names = fit_names_full[:: int(args.fit_stride)]

    if args.mode == "s5_val":
        forbid_test = True
        eval_split_dir = "val"
        eval_names = list(parts["val"])
    elif args.mode == "s6_locked_test":
        if not args.benchmark_protocol:
            raise ValueError("--benchmark-protocol required for s6_locked_test")
        proto = json.loads(Path(args.benchmark_protocol).read_text())
        if proto.get("protocol_status") != "frozen":
            raise RuntimeError("Refusing s6_locked_test: benchmark-protocol is not frozen")
        forbid_test = False
        eval_split_dir = "test"
        eval_names = list(parts["test"])
    else:
        raise ValueError(f"Unsupported mode: {args.mode!r}")

    if args.eval_limit and int(args.eval_limit) > 0:
        eval_names = eval_names[: int(args.eval_limit)]
        fit_names = fit_names[: max(4, min(len(fit_names), int(args.eval_limit)))]

    hemit_root = Path(args.hemit_root).resolve()
    nuclei_root = Path(args.nuclei_root).resolve() if args.nuclei_root else None
    out_dir = Path(args.out_dir).resolve()
    state_dir = Path(args.state_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            dict(
                event="loading_hemit_splits",
                mode=args.mode,
                n_fit_tiles=len(fit_names),
                n_eval_tiles=len(eval_names),
                eval_split=eval_split_dir,
            )
        ),
        flush=True,
    )
    def _load_or_cache_hemit(split_tag: str, names_list: list[str], *, forbid: bool) -> dict:
        if getattr(args, "bank_cache_dir", ""):
            bc_dir = Path(args.bank_cache_dir).resolve()
            bc_dir.mkdir(parents=True, exist_ok=True)
            bc_file = bc_dir / f"hemit_bank_{split_tag}_{len(names_list)}.pkl"
            if bc_file.exists():
                with open(bc_file, "rb") as fh:
                    cand = pickle.load(fh)
                if cand["names"] == list(names_list):
                    return cand
            loaded = load_hemit_split_arrays(hemit_root, nuclei_root, split_tag, names_list, forbid_test=forbid)
            tmp_bc = bc_file.with_name(f".tmp_{os.getpid()}_{bc_file.name}")
            with open(tmp_bc, "wb") as fh:
                pickle.dump(loaded, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_bc, bc_file)
            return loaded
        return load_hemit_split_arrays(hemit_root, nuclei_root, split_tag, names_list, forbid_test=forbid)

    train_data = _load_or_cache_hemit("train", fit_names, forbid=forbid_test)
    eval_data = _load_or_cache_hemit(eval_split_dir, eval_names, forbid=forbid_test)
    for _ds in (train_data, eval_data):
        for _rec in _ds["nuclei_records"]:
            if _rec is not None and "idx_spec" in _rec:
                _rec.pop("mask_hw", None)
    gc.collect()

    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    native_keys = {"native_dgr", "native_unistainnet_uni2h", "native_hemit_fml", "native_hybrid_donor"}
    native_preds_train: dict[str, np.ndarray] = {}
    native_preds_eval: dict[str, np.ndarray] = {}
    native_families = {
        "native_dgr": "Adversarial cGAN + Classifier (In-Domain)",
        "native_unistainnet_uni2h": "Foundation ViT UNI2-h (In-Domain)",
        "native_hemit_fml": "Dual-Branch Swin-ResNet (In-Domain)",
        "native_hybrid_donor": "Multi-Scale Hybrid Donor (In-Domain)",
    }
    b_cache_dir = state_dir / "upstream_cache"
    b_cache_dir.mkdir(parents=True, exist_ok=True)
    req_native = [b for b in backbones if b in native_keys]
    missing_native = []
    for nk in req_native:
        tr_c = b_cache_dir / f"{nk}_train_{len(fit_names)}.npz"
        ev_c = b_cache_dir / f"{nk}_{eval_split_dir}_{len(eval_names)}.npz"
        if tr_c.exists() and ev_c.exists():
            with np.load(tr_c) as z_tr:
                native_preds_train[nk] = z_tr["pred_u8"]
            with np.load(ev_c) as z_ev:
                native_preds_eval[nk] = z_ev["pred_u8"]
        else:
            missing_native.append(nk)
    if missing_native:
        if not args.native_config:
            raise ValueError("--native-config required when running native_* HEMIT backbones")
        nbank = NativeHEMITBank(Path(args.native_config).resolve(), device)
        want_set = set(missing_native)

        def _save_tr(nk: str, arr: np.ndarray) -> None:
            tr_c = b_cache_dir / f"{nk}_train_{len(fit_names)}.npz"
            if not tr_c.exists():
                tmp_tr = tr_c.with_name(f".tmp_{os.getpid()}_{tr_c.stem}.npz")
                np.savez_compressed(tmp_tr, pred_u8=arr)
                os.replace(tmp_tr, tr_c)

        def _save_ev(nk: str, arr: np.ndarray) -> None:
            ev_c = b_cache_dir / f"{nk}_{eval_split_dir}_{len(eval_names)}.npz"
            if not ev_c.exists():
                tmp_ev = ev_c.with_name(f".tmp_{os.getpid()}_{ev_c.stem}.npz")
                np.savez_compressed(tmp_ev, pred_u8=arr)
                os.replace(tmp_ev, ev_c)

        gen_tr = nbank.predict_all_native(train_data["he_nhwc"], only_keys=want_set, save_callback=_save_tr)
        gen_ev = nbank.predict_all_native(eval_data["he_nhwc"], only_keys=want_set, save_callback=_save_ev)
        for nk, arr_tr in gen_tr.items():
            native_preds_train[nk] = arr_tr
            native_preds_eval[nk] = gen_ev[nk]
        del nbank
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    summary_by_backbone = {}

    for b_key in backbones:
        b_t0 = time.monotonic()
        b_cache_dir = state_dir / "upstream_cache"
        b_cache_dir.mkdir(parents=True, exist_ok=True)
        tr_cache = b_cache_dir / f"{b_key}_train_{len(fit_names)}.npz"
        ev_cache = b_cache_dir / f"{b_key}_{eval_split_dir}_{len(eval_names)}.npz"

        if b_key in native_keys:
            b_family = native_families[b_key]
            up_train = native_preds_train[b_key]
            up_eval = native_preds_eval[b_key]
        elif tr_cache.exists() and ev_cache.exists():
            b_family = HEMIT_SANOFI_FAMILIES.get(b_key, b_key)
            with np.load(tr_cache) as z_tr:
                up_train = z_tr["pred_u8"]
            with np.load(ev_cache) as z_ev:
                up_eval = z_ev["pred_u8"]
        else:
            pred_obj = instantiate_sanofi_or_synthetic(
                b_key,
                Path(args.sanofi_ckpt_root) if args.sanofi_ckpt_root else None,
                Path(args.mipheivit_src) if args.mipheivit_src else None,
                Path(args.hoptimus0_ckpt) if args.hoptimus0_ckpt else None,
                device,
            )
            b_family = pred_obj.family
            bs_inf = 2 if "diffusion" in b_key or "rosie" in b_key else 8
            if tr_cache.exists():
                with np.load(tr_cache) as z_tr:
                    up_train = z_tr["pred_u8"]
            else:
                up_train = predict_in_batches(pred_obj, train_data["he_nhwc"], batch_size=bs_inf)
                tmp_tr = tr_cache.with_name(f".tmp_{os.getpid()}_{tr_cache.stem}.npz")
                np.savez_compressed(tmp_tr, pred_u8=up_train)
                os.replace(tmp_tr, tr_cache)
            if ev_cache.exists():
                with np.load(ev_cache) as z_ev:
                    up_eval = z_ev["pred_u8"]
            else:
                up_eval = predict_in_batches(pred_obj, eval_data["he_nhwc"], batch_size=bs_inf)
                tmp_ev = ev_cache.with_name(f".tmp_{os.getpid()}_{ev_cache.stem}.npz")
                np.savez_compressed(tmp_ev, pred_u8=up_eval)
                os.replace(tmp_ev, ev_cache)
            del pred_obj
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(
            json.dumps(dict(event="upstream_predicted", backbone=b_key, family=b_family)),
            flush=True,
        )

        b_state_dir = state_dir / b_key
        b_state_dir.mkdir(parents=True, exist_ok=True)

        # Arm predictions (N, 3, H, W)
        arm_train_preds: dict[str, np.ndarray] = {"upstream": up_train}
        arm_eval_preds: dict[str, np.ndarray] = {"upstream": up_eval}
        for arm in suh.ARMS:
            for sd in seeds:
                arm_train_preds[f"{arm}:{sd}"] = np.empty_like(up_train)
                arm_eval_preds[f"{arm}:{sd}"] = np.empty_like(up_eval)
            arm_train_preds[f"{arm}:ens2"] = np.empty_like(up_train)
            arm_eval_preds[f"{arm}:ens2"] = np.empty_like(up_eval)

        fit_meta_records = {}
        solver_diags = {}

        ch_list = list(enumerate(zip(HEMIT_MARKERS, HEMIT_CUTOFFS)))
        if getattr(args, "reverse_channels", False):
            ch_list = list(reversed(ch_list))

        for ch, (m_name, cutoff) in ch_list:
            c12_tr = build_common12_for_3ch(train_data["he_nhwc"], up_train, ch, device=device)
            d_tr_t = torch.from_numpy(up_train[:, ch])
            y_tr_t = torch.from_numpy(train_data["targ_nchw"][:, ch])
            d_ev_t = torch.from_numpy(up_eval[:, ch])

            fitted_models: dict[tuple[str, int], object] = {}
            for arm in suh.ARMS:
                tr_seed_list = []
                for sd in seeds:
                    ckpt_file = b_state_dir / f"{m_name}_{arm}_{sd}.pt"
                    meta_file = b_state_dir / f"{m_name}_{arm}_{sd}_meta.json"
                    if args.mode == "s5_val":
                        if ckpt_file.exists() and not args.overwrite_fits:
                            model = suh.build_coordinate_screened_model(sd, device)
                            model.load_state_dict(torch.load(ckpt_file, map_location=device, weights_only=True))
                            meta = json.loads(meta_file.read_text())
                        else:
                            model, meta = suh.fit_universal_marker(
                                arm=arm,
                                seed=sd,
                                common_train=c12_tr,
                                donor_u8_train=d_tr_t,
                                target_u8_train=y_tr_t,
                                cutoff=cutoff,
                                device=device,
                                epochs=int(args.epochs),
                                batch_size=int(args.batch_size),
                                lr=float(getattr(args, "lr", 1.2e-3)),
                                deploy_anchor_weight=float(getattr(args, "deploy_anchor_weight", 1.25)),
                            )
                            _atomic_torch_save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt_file)
                            _atomic_write_text(meta_file, json.dumps(meta, indent=2) + "\n")
                    else:
                        if not ckpt_file.exists():
                            raise FileNotFoundError(f"Missing frozen S5 head checkpoint: {ckpt_file}")
                        model = suh.build_coordinate_screened_model(sd, device)
                        model.load_state_dict(torch.load(ckpt_file, map_location=device, weights_only=True))
                        meta = json.loads(meta_file.read_text())

                    fit_meta_records[f"{m_name}:{arm}:{sd}"] = dict(
                        effective_lambda=meta["effective_lambda"],
                        base_lambda=meta["base_lambda"],
                        final_recon_loss=meta["history"][-1]["recon_loss"],
                    )
                    pr_tr, _ = suh.predict_universal_marker(model, c12_tr, d_tr_t, device=device, batch_size=16)
                    arm_train_preds[f"{arm}:{sd}"][:, ch] = pr_tr
                    tr_seed_list.append(pr_tr.astype(np.float32))
                    fitted_models[(arm, sd)] = model.cpu()

                arm_train_preds[f"{arm}:ens2"][:, ch] = np.round(np.mean(tr_seed_list, axis=0)).astype(np.uint8)

            del c12_tr
            gc.collect()

            c12_ev = build_common12_for_3ch(eval_data["he_nhwc"], up_eval, ch, device=device)
            for arm in suh.ARMS:
                ev_seed_list = []
                for sd in seeds:
                    model = fitted_models.pop((arm, sd)).to(device)  # type: ignore[attr-defined]
                    pr_ev, diag_ev = suh.predict_universal_marker(model, c12_ev, d_ev_t, device=device, batch_size=16)
                    solver_diags[f"{m_name}:{arm}:{sd}"] = diag_ev
                    arm_eval_preds[f"{arm}:{sd}"][:, ch] = pr_ev
                    ev_seed_list.append(pr_ev.astype(np.float32))
                    del model
                arm_eval_preds[f"{arm}:ens2"][:, ch] = np.round(np.mean(ev_seed_list, axis=0)).astype(np.uint8)

            del c12_ev, fitted_models
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Evaluate all arms on pixel + single-cell + spatial infiltration
        arms_eval_out = {}
        arm_cluster_macro: dict[str, dict[str, dict[str, float]]] = {}

        for arm_key, ev_pred in arm_eval_preds.items():
            tr_pred = arm_train_preds[arm_key]
            per_marker_pix = {}
            cluster_acc: dict[str, dict[str, list[float]]] = {}
            for ch, (m_name, cutoff) in enumerate(zip(HEMIT_MARKERS, HEMIT_CUTOFFS)):
                ch_res = jce.evaluate_channel_predictions(
                    train_pred_u8=tr_pred[:, ch],
                    train_targ_u8=train_data["targ_nchw"][:, ch],
                    eval_pred_u8=ev_pred[:, ch],
                    eval_targ_u8=eval_data["targ_nchw"][:, ch],
                    eval_clusters=eval_data["clusters"],
                    fixed_cutoff=cutoff,
                    device=device,
                )
                per_marker_pix[m_name] = ch_res["pooled"]
                for cl, cl_m in ch_res["per_cluster"].items():
                    d_cl = cluster_acc.setdefault(cl, {})
                    for mk in ("ODS_F", "AP", "Fixed_tol_F1", "Cal_F1", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed"):
                        d_cl.setdefault(mk, []).append(float(cl_m[mk]))

            macro_pix = {
                mk: float(np.mean([per_marker_pix[m][mk] for m in HEMIT_MARKERS]))
                for mk in ("ODS_F", "AP", "Fixed_tol_F1", "Cal_F1", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed")
            }
            arm_cluster_macro[arm_key] = {
                cl: {mk: float(np.mean(vals)) for mk, vals in mdict.items()}
                for cl, mdict in cluster_acc.items()
            }

            tr_cells = extract_pred_cell_means_for_records(tr_pred, train_data["nuclei_records"])
            ev_cells = extract_pred_cell_means_for_records(ev_pred, eval_data["nuclei_records"])
            cell_res = evaluate_hemit_arm_cells_and_infiltration(
                tr_cells, ev_cells, eval_data["nuclei_records"], ev_pred
            )
            arms_eval_out[arm_key] = dict(
                macro_pixel=macro_pix,
                per_marker_pixel=per_marker_pix,
                single_cell=cell_res,
            )

        # Paired 15-region cluster bootstrap comparisons
        sc_ens = "S_count:ens2"
        sr_ens = "S_recon:ens2"
        paired_vs_upstream = {
            mk: jce.paired_cluster_bootstrap(arm_cluster_macro[sc_ens], arm_cluster_macro["upstream"], mk)
            for mk in ("Fixed_tol_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB")
        }
        paired_vs_srecon = {
            mk: jce.paired_cluster_bootstrap(arm_cluster_macro[sc_ens], arm_cluster_macro[sr_ens], mk)
            for mk in ("Fixed_tol_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB")
        }

        # Export 2 compact visual example tiles in s6_locked_test for journal figures
        if args.mode == "s6_locked_test" and len(eval_names) >= 2:
            vis_dir = out_dir / "visual_crops"
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_indices = [0, len(eval_names) // 2]
            np.savez_compressed(
                vis_dir / f"{b_key}_examples.npz",
                names=np.array([eval_names[i] for i in vis_indices]),
                he_nhwc=eval_data["he_nhwc"][vis_indices],
                targ_nchw=eval_data["targ_nchw"][vis_indices],
                upstream_nchw=arm_eval_preds["upstream"][vis_indices],
                srecon_nchw=arm_eval_preds[sr_ens][vis_indices],
                scount_nchw=arm_eval_preds[sc_ens][vis_indices],
            )

        b_rec = dict(
            backbone=b_key,
            family=b_family,
            wall_seconds=time.monotonic() - b_t0,
            fit_meta=fit_meta_records,
            solver_diagnostics=solver_diags,
            arms=arms_eval_out,
            paired_cluster_bootstrap=dict(
                scount_ens2_vs_upstream=paired_vs_upstream,
                scount_ens2_vs_srecon_ens2=paired_vs_srecon,
            ),
            per_cluster_macro=arm_cluster_macro,
        )
        summary_by_backbone[b_key] = b_rec
        (out_dir / f"backbone_{b_key}_{args.mode}.json").write_text(json.dumps(b_rec, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    event="backbone_complete",
                    backbone=b_key,
                    mode=args.mode,
                    up_Fixed_F1=round(arms_eval_out["upstream"]["macro_pixel"]["Fixed_tol_F1"], 4),
                    sr_Fixed_F1=round(arms_eval_out[sr_ens]["macro_pixel"]["Fixed_tol_F1"], 4),
                    sc_Fixed_F1=round(arms_eval_out[sc_ens]["macro_pixel"]["Fixed_tol_F1"], 4),
                    up_ODS_F=round(arms_eval_out["upstream"]["macro_pixel"]["ODS_F"], 4),
                    sc_ODS_F=round(arms_eval_out[sc_ens]["macro_pixel"]["ODS_F"], 4),
                    up_Cell_F1=round(arms_eval_out["upstream"]["single_cell"].get("macro_deployable_cell_f1", 0.0), 4),
                    sc_Cell_F1=round(arms_eval_out[sc_ens]["single_cell"].get("macro_deployable_cell_f1", 0.0), 4),
                    sec=round(b_rec["wall_seconds"], 1),
                )
            ),
            flush=True,
        )

    final_doc = dict(
        schema="hemit-multibackbone-cell-v1",
        mode=args.mode,
        eval_split=eval_split_dir,
        n_fit_tiles=len(fit_names),
        n_eval_tiles=len(eval_names),
        seeds=list(seeds),
        epochs=int(args.epochs),
        total_wall_seconds=time.monotonic() - t0,
        backbones=summary_by_backbone,
    )
    out_file = out_dir / f"hemit_multibackbone_{args.mode}_summary.json"
    out_file.write_text(json.dumps(final_doc, indent=2) + "\n")
    return final_doc


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("s5_val", "s6_locked_test"), required=True)
    p.add_argument("--hemit-root", required=True)
    p.add_argument("--nuclei-root", default="")
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--benchmark-protocol", default="")
    p.add_argument("--sanofi-ckpt-root", default="")
    p.add_argument("--mipheivit-src", default="")
    p.add_argument("--hoptimus0-ckpt", default="")
    p.add_argument("--native-config", default="")
    p.add_argument("--backbones", required=True)
    p.add_argument("--state-dir", required=True)
    p.add_argument("--bank-cache-dir", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", default="202610270,202610271")
    p.add_argument("--fit-stride", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument("--deploy-anchor-weight", type=float, default=1.25)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--overwrite-fits", action="store_true")
    p.add_argument("--reverse-channels", action="store_true")
    return p


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    run_hemit_campaign(build_arg_parser().parse_args())
