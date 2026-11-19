"""Multi-backbone 16-marker mIF, 15-class single-cell & 5-fold PFS survival on Orion-CRC.

Implements Cohort 2 of the Nature Methods Route 1 study on Harvard Orion CRC
(Lin et al., Nature Cancer 2023; 40 patients C01..C40, 41 slides, 25 PFS events,
16 predicted mIF channels, 15 ground-truth nuclear positivity classes):
  * ``s5_fold0_val``: operates strictly on ``tiles_dev/`` (32 patients; 0 reads of
    ``tiles_sealed_fold0_test/``). Fits seed-paired ``(S_recon, S_count)`` heads on
    Fold 0 ``inner_train`` (26 patients) across all 16 mIF channels using physical
    fluorescence cutoffs derived strictly from Fold 0 ``inner_train`` ground-truth
    cells, saves frozen Fold 0 checkpoints, and evaluates Fold 0 ``inner_val``
    (6 patients including official Sanofi C19, C30).
  * ``s6_confirmatory``: requires frozen ``benchmark-protocol.json``.
      - Part A: evaluates frozen Fold 0 checkpoints (``new_fits == 0``) on sealed
        Fold 0 ``outer_test`` (8 patients including C02, C11; 684 tiles) across all
        16 mIF channels, 15 single-cell nuclear classes, and Pan-CK+ -> CD8a+/CD3e+
        spatial infiltration.
      - Part B: executes leak-free 5-fold patient-disjoint cross-validation across
        all 40 patients (Folds 0..4 from ``orion-patient-cv-manifest.json``) on the
        prognostic compartment triad (``Pan-CK``, ``CD3e``, ``CD8a``), computing
        out-of-patient ``R1/R2`` ``ImmunoScore`` and clinical ``PFS`` survival
        (Harrell's C-index + 10,000-bootstrap 95% CIs, KM curves, Mantel-Haenszel
        log-rank p, univariable HR, and Stage + MMR multivariable Cox PH).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
try:
    from skimage.filters import threshold_otsu
except ModuleNotFoundError:
    def threshold_otsu(arr):
        return float(np.quantile(arr, 0.85))

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

ORION_MARKERS = zoo.ORION_16_MARKERS
ORION_TIFF_IDXS = list(zoo.ORION_TIFF_16_INDICES)
ORION_CLASSES_15 = zoo.ORION_NUCLEI_15_CLASSES
PROGNOSTIC_TRIAD = ("Pan-CK", "CD3e", "CD8a")
PROGNOSTIC_CH16 = {m: ORION_MARKERS.index(m) for m in PROGNOSTIC_TRIAD}


def orion_slide_to_patient(orion_slide_id: str) -> str:
    """Map 'CRC01' / 'CRC33_01' / 'CRC33_02' -> 'C01' / 'C33'."""
    core = str(orion_slide_id).replace("CRC", "").split("_")[0]
    return f"C{int(core):02d}"


def center_crop_hw(arr: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == crop_size and w == crop_size:
        return arr
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot center-crop {crop_size} from shape {arr.shape}")
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return arr[y0 : y0 + crop_size, x0 : x0 + crop_size]


def _read_nuclei_label_table(path_parquet_or_csv: Path) -> pd.DataFrame:
    if path_parquet_or_csv.suffix == ".parquet":
        csv_alt = path_parquet_or_csv.with_suffix(".csv.gz")
        try:
            return pd.read_parquet(path_parquet_or_csv)
        except Exception:
            if csv_alt.exists():
                return pd.read_csv(csv_alt)
            raise
    return pd.read_csv(path_parquet_or_csv)


def load_orion_tile_bank(
    orion_root: Path,
    selected_df: pd.DataFrame,
    *,
    include_sealed_fold0_test: bool,
    crop_size: int = 256,
) -> dict:
    """Load centered 256x256 H&E, 16-ch mIF, nuclei masks, and ground-truth cell labels."""
    if not include_sealed_fold0_test:
        df = selected_df[~selected_df["is_fold0_sealed_test"].astype(bool)].reset_index(drop=True)
    else:
        df = selected_df.reset_index(drop=True)

    dev_dir = orion_root / "tiles_dev"
    sealed_dir = orion_root / "tiles_sealed_fold0_test"

    dev_lbl_df = _read_nuclei_label_table(dev_dir / "dev_nuclei_labels.parquet")
    lbl_dfs = [dev_lbl_df]
    if include_sealed_fold0_test:
        sealed_lbl_df = _read_nuclei_label_table(sealed_dir / "sealed_fold0_test_nuclei_labels.parquet")
        lbl_dfs.append(sealed_lbl_df)
    full_lbl = pd.concat(lbl_dfs, ignore_index=True)

    # Index ground-truth labels by slide_name -> DataFrame indexed by integer label
    slide_label_lookup: dict[str, pd.DataFrame] = {}
    for sl_name, grp in full_lbl.groupby("slide_name", sort=False):
        slide_label_lookup[str(sl_name)] = grp.drop_duplicates("label").set_index("label")

    n = len(df)
    he_nhwc = np.empty((n, crop_size, crop_size, 3), dtype=np.uint8)
    targ_nchw = np.empty((n, 16, crop_size, crop_size), dtype=np.uint8)
    patients = []
    slides = []
    nuclei_records = []

    for i, row in df.iterrows():
        is_sealed = bool(row["is_fold0_sealed_test"])
        if is_sealed and not include_sealed_fold0_test:
            raise RuntimeError("Holdout containment violation: attempted to read sealed Fold 0 test tile in S5")
        base = sealed_dir if is_sealed else dev_dir
        in_sl = str(row["in_slide_name"])
        pat = orion_slide_to_patient(str(row["orion_slide_id"]))
        patients.append(pat)
        slides.append(in_sl)

        with Image.open(base / str(row["image_path"])) as im_he:
            he_raw = np.asarray(im_he.convert("RGB"), dtype=np.uint8)
        if_raw = _imread_tiff(base / str(row["target_path"]))
        nuc_raw = _imread_tiff(base / str(row["nuclei_path"]))

        he_c = center_crop_hw(he_raw, crop_size)
        if_c = center_crop_hw(if_raw, crop_size)[:, :, ORION_TIFF_IDXS]
        nuc_c = center_crop_hw(nuc_raw, crop_size).astype(np.int32, copy=False)

        he_nhwc[i] = he_c
        targ_nchw[i] = np.transpose(if_c, (2, 0, 1))

        gt_nuc = ccs.extract_tile_nuclei(targ_nchw[i], nuc_c, min_area=10)
        sl_tbl = slide_label_lookup.get(in_sl)
        if len(gt_nuc["cell_ids"]) == 0 or sl_tbl is None:
            nuclei_records.append(None)
            continue
        valid = np.isin(gt_nuc["cell_ids"], sl_tbl.index.to_numpy())
        if not np.any(valid):
            nuclei_records.append(None)
            continue
        kept_ids = gt_nuc["cell_ids"][valid]
        sub_lbl = sl_tbl.loc[kept_ids, list(ORION_CLASSES_15)].astype(np.int32).to_numpy()
        idx_spec = ccs.prepare_tile_nuclei_index(nuc_c, kept_ids)
        nuclei_records.append(
            dict(
                tile_idx=i,
                patient_id=pat,
                slide_name=in_sl,
                idx_spec=idx_spec,
                cell_ids=kept_ids,
                centroids_yx=gt_nuc["centroids_yx"][valid],
                gt_means16=gt_nuc["means"][valid],
                labels15=sub_lbl,
            )
        )

    return dict(
        df=df,
        patients=np.asarray(patients),
        slides=np.asarray(slides),
        he_nhwc=he_nhwc,
        targ_nchw=targ_nchw,
        nuclei_records=nuclei_records,
    )


def gather_subset_cells(
    pred_sub_nchw: np.ndarray,
    sub_indices: Sequence[int],
    all_nuclei_records: Sequence[dict | None],
    channel_map_in_pred: Mapping[int, int] | None = None,
) -> dict:
    """Extract per-nucleus predicted & GT intensities for a tile subset ``sub_indices``.

    ``channel_map_in_pred``: optional mapping ``ch16_idx -> col_in_pred_sub_nchw`` when
    ``pred_sub_nchw`` only holds a subset of channels (e.g., 3 prognostic channels).
    """
    c_pred = pred_sub_nchw.shape[1]
    tile_local_ids = []
    patient_ids = []
    gt_means_list = []
    pred_means_list = []
    labels_list = []

    for local_i, global_i in enumerate(sub_indices):
        rec = all_nuclei_records[int(global_i)]
        if rec is None:
            continue
        if "idx_spec" in rec:
            pm = ccs.extract_means_with_index(pred_sub_nchw[local_i], rec["idx_spec"])
        else:
            nuc = ccs.extract_tile_nuclei(pred_sub_nchw[local_i], rec["mask_hw"], min_area=10)
            id_to_pos = {int(cid): idx for idx, cid in enumerate(nuc["cell_ids"].tolist())}
            rows = [id_to_pos[int(cid)] for cid in rec["cell_ids"].tolist()]
            pm = nuc["means"][rows]
        k = len(rec["cell_ids"])
        tile_local_ids.append(np.full(k, local_i, dtype=np.int64))
        patient_ids.extend([rec["patient_id"]] * k)
        gt_means_list.append(rec["gt_means16"])
        pred_means_list.append(pm)
        labels_list.append(rec["labels15"])

    if not tile_local_ids:
        return dict(
            tile_ids=np.zeros((0,), dtype=np.int64),
            patients=np.asarray([], dtype=str),
            gt_means16=np.zeros((0, 16), dtype=np.float64),
            pred_means=np.zeros((0, c_pred), dtype=np.float64),
            labels15=np.zeros((0, 15), dtype=np.int32),
            channel_map=dict(channel_map_in_pred) if channel_map_in_pred else {i: i for i in range(c_pred)},
        )

    return dict(
        tile_ids=np.concatenate(tile_local_ids, axis=0),
        patients=np.asarray(patient_ids),
        gt_means16=np.concatenate(gt_means_list, axis=0),
        pred_means=np.concatenate(pred_means_list, axis=0),
        labels15=np.concatenate(labels_list, axis=0),
        channel_map=dict(channel_map_in_pred) if channel_map_in_pred else {i: i for i in range(c_pred)},
    )


def calibrate_orion_physical_cutoffs_from_train(
    train_targ_nchw: np.ndarray,
    train_indices: Sequence[int],
    all_nuclei_records: Sequence[dict | None],
) -> tuple[list[int], dict[str, float]]:
    """Derive exact physical fluorescence cutoffs strictly on training-fold targets & cells."""
    # Channel 0: Hoechst nuclear Otsu threshold
    hoechst_sample = train_targ_nchw[:: max(1, len(train_targ_nchw) // 128), 0].ravel()
    try:
        h_otsu = float(threshold_otsu(hoechst_sample))
    except Exception:
        h_otsu = 24.0
    cutoffs_u8 = [int(np.clip(round(h_otsu), 12, 80))]
    tau_phys_float: dict[str, float] = {"Hoechst": float(cutoffs_u8[0])}

    gt_list, lbl_list = [], []
    for gi in train_indices:
        rec = all_nuclei_records[int(gi)]
        if rec is not None:
            gt_list.append(rec["gt_means16"])
            lbl_list.append(rec["labels15"])

    if gt_list:
        gt_all = np.concatenate(gt_list, axis=0)
        lbl_all = np.concatenate(lbl_list, axis=0)
    else:
        gt_all = np.zeros((0, 16), dtype=np.float64)
        lbl_all = np.zeros((0, 15), dtype=np.int32)

    for ch16 in range(1, 16):
        m_name = ORION_MARKERS[ch16]
        cls_idx = ch16 - 1
        pos_cnt = int(lbl_all[:, cls_idx].sum()) if len(lbl_all) else 0
        if pos_cnt >= 5 and pos_cnt < len(lbl_all):
            tau_f = float(ccs.fit_best_f1_threshold(gt_all[:, ch16], lbl_all[:, cls_idx]))
        else:
            q96 = float(np.quantile(train_targ_nchw[:, ch16], 0.96))
            tau_f = max(8.0, q96)
        tau_phys_float[m_name] = tau_f

        sample_ch = train_targ_nchw[:: max(1, len(train_targ_nchw) // 256), ch16].ravel()
        pos_px = sample_ch[sample_ch > 0]
        if pos_px.size >= 64 and int(pos_px.max()) > int(pos_px.min()):
            try:
                px_otsu = float(threshold_otsu(pos_px))
            except Exception:
                px_otsu = float(tau_f)
            cutoffs_u8.append(int(np.clip(round(px_otsu), 16, 80)))
        else:
            cutoffs_u8.append(int(np.clip(round(tau_f), 16, 80)))

    return cutoffs_u8, tau_phys_float


def _atomic_torch_save(obj: object, path: Path) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(text)
    os.replace(tmp, path)


def build_common12_orion(
    he_nhwc: np.ndarray,
    up_nchw16: np.ndarray,
    ch16: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    dev = torch.device(device)
    n, _, h, w = up_nchw16.shape
    out = torch.empty((n, 12, h, w), dtype=torch.float16)
    chunk = 128
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            he_t = torch.from_numpy(he_nhwc[s:e]).to(dev, non_blocking=True).permute(0, 3, 1, 2).to(torch.float32) / 255.0
            d_all = torch.from_numpy(up_nchw16[s:e]).to(dev, non_blocking=True).to(torch.float32) / 255.0
            bc = d_all[:, ch16 : ch16 + 1]
            bs = d_all[:, 0:1]  # Hoechst structural nuclear reference
            bm = d_all.mean(dim=1, keepdim=True)
            out[s:e] = suh.build_universal_common12(he_t, bc, bs, bm).to(torch.float16).cpu()
    return out


ORION_BACKBONE_FAMILIES: dict[str, str] = {
    "synthetic_orion_vit": "Foundation ViT",
    "synthetic_orion_gan": "Adversarial cGAN",
    "miphei_vit": "Foundation ViT + Attention Matting",
    "unetr_hoptimus": "Foundation ViT + LoRA UNETR",
    "miphei_convnext": "ConvNeXt-Large + Attention U-Net",
    "miphei_hemit_arch": "Weighted-MSE Swin-ResNet",
    "hemit_orion": "Adversarial Swin-ResNet",
    "pix2pix_orion": "Conditional Adversarial U-Net cGAN",
    "diffusion_ft": "One-Step Latent Diffusion (SD2.1)",
    "rosie_orion": "Sliding-Window ConvNeXt Regressor",
}


def instantiate_orion_backbone(
    key: str,
    sanofi_ckpt_root: Path | None,
    mipheivit_src: Path | None,
    hoptimus0_ckpt: Path | None,
    crop_size: int,
    device: torch.device,
) -> zoo.BackbonePredictor:
    mk = ORION_MARKERS
    if key == "synthetic_orion_vit":
        return zoo.SyntheticBackbonePredictor("synthetic_orion_vit", "Foundation ViT", mk, attenuation=0.58, seed=7)
    if key == "synthetic_orion_gan":
        return zoo.SyntheticBackbonePredictor("synthetic_orion_gan", "Adversarial cGAN", mk, attenuation=0.63, seed=19)
    if sanofi_ckpt_root is None or mipheivit_src is None:
        raise ValueError(f"sanofi_ckpt_root and mipheivit_src required for {key}")

    mapping = {
        "miphei_vit": ("miphei", "Foundation ViT + Attention Matting", "MIPHEI-vit"),
        "unetr_hoptimus": ("miphei", "Foundation ViT + LoRA UNETR", "UNETR-hoptimus"),
        "miphei_convnext": ("miphei", "ConvNeXt-Large + Attention U-Net", "MIPHEI-convnext"),
        "miphei_hemit_arch": ("miphei", "Weighted-MSE Swin-ResNet", "MIPHEI_HEMIT"),
        "hemit_orion": ("hemit", "Adversarial Swin-ResNet", "HEMIT"),
        "pix2pix_orion": ("pix2pix", "Conditional Adversarial U-Net cGAN", "Pix2Pix"),
        "diffusion_ft": ("diffusion", "One-Step Latent Diffusion (SD2.1)", "DiffusionFT"),
        "rosie_orion": ("rosie", "Sliding-Window ConvNeXt Regressor", "Rosie-ORION"),
    }
    if key not in mapping:
        raise ValueError(f"Unknown Orion backbone key: {key!r}")
    kind, fam, sub = mapping[key]
    cdir = sanofi_ckpt_root / sub
    if kind == "miphei":
        return zoo.SanofiMIPHEIPredictor(
            name=key,
            family=fam,
            checkpoint_dir=cdir,
            mipheivit_src_dir=mipheivit_src,
            device=device,
            img_size=crop_size,
            hoptimus0_ckpt_path=hoptimus0_ckpt,
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
            img_size=crop_size,
            marker_names=mk,
        )
    if kind == "diffusion":
        return zoo.SanofiDiffusionFTPredictor(
            name=key,
            family=fam,
            checkpoint_dir=cdir,
            mipheivit_src_dir=mipheivit_src,
            device=device,
            marker_names=mk,
        )
    return zoo.SanofiRosiePredictor(
        name=key,
        family=fam,
        checkpoint_dir=cdir,
        mipheivit_src_dir=mipheivit_src,
        device=device,
        marker_names=mk,
    )


def predict_orion_upstream(
    predictor: zoo.BackbonePredictor,
    he_nhwc: np.ndarray,
    cache_path: Path | None = None,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as npz:
            arr = npz["pred_u8"]
            if arr.shape[0] == he_nhwc.shape[0] and arr.shape[2:] == he_nhwc.shape[1:3]:
                return arr
    bs = 2 if ("diffusion" in predictor.name or "rosie" in predictor.name) else 16
    n, h, w, _ = he_nhwc.shape
    out = np.empty((n, 16, h, w), dtype=np.uint8)
    for s in range(0, n, bs):
        e = min(n, s + bs)
        out[s:e] = predictor.predict_batch_u8(he_nhwc[s:e])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_npz = cache_path.with_name(f".tmp_{os.getpid()}_{cache_path.stem}.npz")
        np.savez_compressed(tmp_npz, pred_u8=out)
        os.replace(tmp_npz, cache_path)
    return out


def compute_patient_compartments_for_subset(
    sub_indices: Sequence[int],
    sub_patients: Sequence[str],
    all_nuclei_records: Sequence[dict | None],
    pred_triad_nchw: np.ndarray | None,
    tau_triad: Mapping[str, float],
    *,
    use_ground_truth_labels: bool = False,
) -> dict[str, dict[str, float]]:
    """Aggregate per-patient R1/R2 CD3e & CD8a densities across tiles of each patient."""
    by_pat: dict[str, list[tuple[int, int, int, int]]] = {}
    for local_i, (gi, pat) in enumerate(zip(sub_indices, sub_patients)):
        rec = all_nuclei_records[int(gi)]
        if rec is None:
            by_pat.setdefault(str(pat), []).append((0, 0, 0, 0))
            continue
        k = len(rec["cell_ids"])
        if use_ground_truth_labels:
            # Pan-CK_pos = idx 13 in ORION_CLASSES_15; CD3e_pos = idx 9; CD8a_pos = idx 5
            pck = int(np.sum(rec["labels15"][:, 13] > 0))
            cd3 = int(np.sum(rec["labels15"][:, 9] > 0))
            cd8 = int(np.sum(rec["labels15"][:, 5] > 0))
        else:
            assert pred_triad_nchw is not None
            if "idx_spec" in rec:
                pm = ccs.extract_means_with_index(pred_triad_nchw[local_i], rec["idx_spec"])
            else:
                nuc = ccs.extract_tile_nuclei(pred_triad_nchw[local_i], rec["mask_hw"], min_area=10)
                id_to_pos = {int(cid): idx for idx, cid in enumerate(nuc["cell_ids"].tolist())}
                rows = [id_to_pos[int(cid)] for cid in rec["cell_ids"].tolist()]
                pm = nuc["means"][rows]  # cols 0=Pan-CK, 1=CD3e, 2=CD8a
            pck = int(np.sum(pm[:, 0] > float(tau_triad["Pan-CK"])))
            cd3 = int(np.sum(pm[:, 1] > float(tau_triad["CD3e"])))
            cd8 = int(np.sum(pm[:, 2] > float(tau_triad["CD8a"])))
        by_pat.setdefault(str(pat), []).append((k, pck, cd3, cd8))

    out = {}
    for pat, rows_t in sorted(by_pat.items()):
        arr = np.asarray(rows_t, dtype=np.int64)
        out[pat] = ccs.compute_patient_compartment_densities(
            tile_total_cells=arr[:, 0],
            tile_panck_pos=arr[:, 1],
            tile_cd3_pos=arr[:, 2],
            tile_cd8_pos=arr[:, 3],
        )
    return out


def evaluate_16ch_split(
    *,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    bank: dict,
    arm_train_preds: Mapping[str, np.ndarray],
    arm_eval_preds: Mapping[str, np.ndarray],
    cutoffs_u8: Sequence[int],
    tau_phys_float: Mapping[str, float],
    device: torch.device | str | None = None,
) -> tuple[dict, dict]:
    """Evaluate all 16 mIF channels + 15 single-cell classes + spatial infiltration."""
    eval_patients = [str(bank["patients"][i]) for i in eval_indices]
    train_targ = bank["targ_nchw"][train_indices]
    eval_targ = bank["targ_nchw"][eval_indices]

    arms_out = {}
    arm_patient_macro: dict[str, dict[str, dict[str, float]]] = {}

    for arm_key, ev_pred in arm_eval_preds.items():
        tr_pred = arm_train_preds[arm_key]
        per_marker_pix = {}
        pat_acc: dict[str, dict[str, list[float]]] = {}

        for ch16, m_name in enumerate(ORION_MARKERS):
            ch_res = jce.evaluate_channel_predictions(
                train_pred_u8=tr_pred[:, ch16],
                train_targ_u8=train_targ[:, ch16],
                eval_pred_u8=ev_pred[:, ch16],
                eval_targ_u8=eval_targ[:, ch16],
                eval_clusters=eval_patients,
                fixed_cutoff=int(cutoffs_u8[ch16]),
                device=device,
            )
            per_marker_pix[m_name] = ch_res["pooled"]
            for pat, pm in ch_res["per_cluster"].items():
                d_p = pat_acc.setdefault(pat, {})
                for mk in ("ODS_F", "AP", "Fixed_tol_F1", "Cal_F1", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed"):
                    d_p.setdefault(mk, []).append(float(pm[mk]))

        macro_pix = {
            mk: float(np.mean([per_marker_pix[m][mk] for m in ORION_MARKERS]))
            for mk in ("ODS_F", "AP", "Fixed_tol_F1", "Cal_F1", "SSIM", "PSNR_dB", "mismatch_gap_ods_minus_fixed")
        }
        arm_patient_macro[arm_key] = {
            pat: {mk: float(np.mean(vals)) for mk, vals in mdict.items()}
            for pat, mdict in pat_acc.items()
        }

        # 15-class single-cell phenotyping
        tr_cells = gather_subset_cells(tr_pred, train_indices, bank["nuclei_records"])
        ev_cells = gather_subset_cells(ev_pred, eval_indices, bank["nuclei_records"])
        per_class_cell = {}
        for ch16 in range(1, 16):
            m_name = ORION_MARKERS[ch16]
            cls_name = f"{m_name}_pos"
            cls_idx = ch16 - 1
            per_class_cell[cls_name] = ccs.evaluate_single_cell_marker(
                train_gt_scores=tr_cells["gt_means16"][:, ch16],
                train_pred_scores=tr_cells["pred_means"][:, ch16],
                train_labels=tr_cells["labels15"][:, cls_idx],
                eval_pred_scores=ev_cells["pred_means"][:, ch16],
                eval_labels=ev_cells["labels15"][:, cls_idx],
                eval_tile_ids=ev_cells["tile_ids"],
                physical_cutoff=float(tau_phys_float[m_name]),
            )

        valid_auprcs = [
            v["cell_auprc"] for v in per_class_cell.values() if np.isfinite(v["cell_auprc"])
        ]
        macro_dep_cell = float(np.mean([v["deployable_cell_f1"] for v in per_class_cell.values()]))
        macro_rec_cell = float(np.mean([v["recalibrated_cell_f1"] for v in per_class_cell.values()]))
        macro_auprc_cell = float(np.mean(valid_auprcs)) if valid_auprcs else 0.0

        # Pan-CK+ (ch 14) -> CD8a+ (ch 6) spatial infiltration concordance
        gt_w25, pr_w25 = [], []
        for local_i, gi in enumerate(eval_indices):
            rec = bank["nuclei_records"][int(gi)]
            if rec is None:
                continue
            if "idx_spec" in rec:
                pm = ccs.extract_means_with_index(ev_pred[local_i], rec["idx_spec"])
            else:
                nuc = ccs.extract_tile_nuclei(ev_pred[local_i], rec["mask_hw"], min_area=10)
                id_to_pos = {int(cid): idx for idx, cid in enumerate(nuc["cell_ids"].tolist())}
                rows = [id_to_pos[int(cid)] for cid in rec["cell_ids"].tolist()]
                pm = nuc["means"][rows]
            pr_pck = (pm[:, 14] > float(tau_phys_float["Pan-CK"])).astype(np.int32)
            pr_cd8 = (pm[:, 6] > float(tau_phys_float["CD8a"])).astype(np.int32)
            gt_inf = ccs.compute_tile_infiltration_metrics(
                rec["centroids_yx"], rec["labels15"][:, 13], rec["labels15"][:, 5], pixel_size_um=0.50
            )
            pr_inf = ccs.compute_tile_infiltration_metrics(
                rec["centroids_yx"], pr_pck, pr_cd8, pixel_size_um=0.50
            )
            if np.isfinite(gt_inf["median_nn_um"]):
                gt_w25.append(gt_inf["within_25um_frac"])
                pr_w25.append(pr_inf["within_25um_frac"])

        w25_r = (
            float(stats.pearsonr(gt_w25, pr_w25)[0])
            if len(gt_w25) >= 3 and np.std(gt_w25) > 1e-12 and np.std(pr_w25) > 1e-12
            else 0.0
        )
        w25_mae = float(np.mean(np.abs(np.asarray(pr_w25) - np.asarray(gt_w25)))) if gt_w25 else 0.0

        arms_out[arm_key] = dict(
            macro_pixel_16ch=macro_pix,
            per_marker_pixel=per_marker_pix,
            single_cell_15class=dict(
                macro_deployable_cell_f1=macro_dep_cell,
                macro_recalibrated_cell_f1=macro_rec_cell,
                macro_cell_auprc=macro_auprc_cell,
                cell_mismatch_gap=macro_rec_cell - macro_dep_cell,
                per_class=per_class_cell,
            ),
            spatial_panck_cd8_infiltration=dict(
                n_copositive_tiles=len(gt_w25),
                within_25um_pearson_r=w25_r,
                within_25um_mae=w25_mae,
            ),
        )

    return arms_out, arm_patient_macro


def _fold_patients(fd: Mapping[str, object], role: str) -> list[str]:
    if role in fd:
        return list(fd[role])  # type: ignore[arg-type]
    alt = f"{role}_slides"
    if alt in fd:
        return list(fd[alt])  # type: ignore[arg-type]
    raise KeyError(f"Fold entry missing {role!r} / {alt!r}: {list(fd.keys())}")


def run_orion_campaign(args: argparse.Namespace) -> dict:
    t0 = time.monotonic()
    device = torch.device(args.device)
    orion_root = Path(args.orion_root).resolve()
    cv_manifest = json.loads(Path(args.cv_manifest).read_text())
    clin_df = pd.read_csv(args.clinical_csv)
    if "PatientID" not in clin_df.columns and "slideName" in clin_df.columns:
        clin_df = clin_df.rename(columns={"slideName": "PatientID"})
    clin_df = clin_df.sort_values("PatientID").reset_index(drop=True)
    selected_df = pd.read_csv(orion_root / "selected_40patient_tiles.csv")

    if args.tile_limit_per_slide and int(args.tile_limit_per_slide) > 0:
        selected_df = (
            selected_df.groupby("in_slide_name", sort=False)
            .head(int(args.tile_limit_per_slide))
            .reset_index(drop=True)
        )

    if args.mode == "s5_fold0_val":
        include_sealed = False
    elif args.mode == "s6_confirmatory":
        if not args.benchmark_protocol:
            raise ValueError("--benchmark-protocol required for s6_confirmatory")
        proto = json.loads(Path(args.benchmark_protocol).read_text())
        if proto.get("protocol_status") != "frozen":
            raise RuntimeError("Refusing s6_confirmatory: benchmark-protocol is not frozen")
        include_sealed = True
    else:
        raise ValueError(f"Unsupported mode: {args.mode!r}")

    out_dir = Path(args.out_dir).resolve()
    state_dir = Path(args.state_dir).resolve()
    cache_dir = Path(args.pred_cache_dir).resolve() if args.pred_cache_dir else (state_dir / "upstream_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    bank = None
    bank_cache_file = None
    if getattr(args, "bank_cache_dir", ""):
        bc_dir = Path(args.bank_cache_dir).resolve()
        bc_dir.mkdir(parents=True, exist_ok=True)
        bc_tag = "all40" if include_sealed else "dev32"
        bank_cache_file = bc_dir / f"orion_bank_{bc_tag}_{len(selected_df)}_crop{int(args.crop_size)}.pkl"
        if bank_cache_file.exists():
            with open(bank_cache_file, "rb") as fh:
                cand_bank = pickle.load(fh)
            if len(cand_bank["df"]) == (len(selected_df) if include_sealed else int((~selected_df["is_fold0_sealed_test"].astype(bool)).sum())):
                bank = cand_bank
    if bank is None:
        bank = load_orion_tile_bank(
            orion_root,
            selected_df,
            include_sealed_fold0_test=include_sealed,
            crop_size=int(args.crop_size),
        )
        if bank_cache_file is not None:
            tmp_bc = bank_cache_file.with_name(f".tmp_{os.getpid()}_{bank_cache_file.name}")
            with open(tmp_bc, "wb") as fh:
                pickle.dump(bank, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_bc, bank_cache_file)
    print(
        json.dumps(
            dict(
                event="orion_bank_loaded",
                mode=args.mode,
                n_tiles=len(bank["df"]),
                n_patients=len(set(bank["patients"].tolist())),
                crop_size=int(args.crop_size),
            )
        ),
        flush=True,
    )

    fold0 = cv_manifest["folds"][0]
    f0_train_pats = set(_fold_patients(fold0, "inner_train"))
    f0_eval_pats = set(_fold_patients(fold0, "inner_val" if args.mode == "s5_fold0_val" else "outer_test"))
    f0_train_idx_all = np.nonzero(np.isin(bank["patients"], list(f0_train_pats)))[0]
    f0_train_idx = f0_train_idx_all[:: int(args.fit_stride)]
    f0_eval_idx = np.nonzero(np.isin(bank["patients"], list(f0_eval_pats)))[0]

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    summary_by_backbone = {}

    # Precompute clinical survival oracle reference in s6_confirmatory
    oracle_survival = None
    if args.mode == "s6_confirmatory":
        pfs_days = clin_df["PFSDays"].to_numpy(dtype=np.float64)
        rec_ev = clin_df["Recurrence"].to_numpy(dtype=np.int32)
        stage_num = clin_df["Stage"].map({"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}).to_numpy(dtype=np.float64)
        mmr_def = (clin_df["MMRIHC"].fillna("Intact").str.lower() != "intact").astype(np.float64).to_numpy()
        # Out-of-fold ground-truth mIF ImmunoScore on the exact same tiles & folds
        oof_gt_cont = np.zeros(len(clin_df), dtype=np.float64)
        oof_gt_disc = np.zeros(len(clin_df), dtype=np.float64)
        pat_to_row = {str(p): idx for idx, p in enumerate(clin_df["PatientID"].tolist())}
        for fd in cv_manifest["folds"]:
            tr_p = _fold_patients(fd, "inner_train")
            te_p = _fold_patients(fd, "outer_test")
            tr_i = np.nonzero(np.isin(bank["patients"], tr_p))[0]
            te_i = np.nonzero(np.isin(bank["patients"], te_p))[0]
            tr_comp = compute_patient_compartments_for_subset(
                tr_i, bank["patients"][tr_i], bank["nuclei_records"], None, {}, use_ground_truth_labels=True
            )
            te_comp = compute_patient_compartments_for_subset(
                te_i, bank["patients"][te_i], bank["nuclei_records"], None, {}, use_ground_truth_labels=True
            )
            tr_vecs = {k: np.array([tr_comp[p][k] for p in tr_p if p in tr_comp]) for k in ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2")}
            te_present = [p for p in te_p if p in te_comp]
            te_vecs = {k: np.array([te_comp[p][k] for p in te_present]) for k in ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2")}
            sc_gt = ccs.score_immunoscore_from_training_ref(tr_vecs, te_vecs)
            for pi, pat_id in enumerate(te_present):
                r_idx = pat_to_row[pat_id]
                oof_gt_cont[r_idx] = sc_gt["immunoscore_continuous"][pi]
                oof_gt_disc[r_idx] = sc_gt["immunoscore_discrete"][pi]

        oracle_survival = dict(
            published_wsi_mif_oracle=ccs.evaluate_pfs_survival(
                pfs_days=pfs_days,
                recurrence_events=rec_ev,
                protective_score=clin_df["ImmunoScore"].to_numpy(dtype=np.float64),
                dichotomy_cutoff=2.0,
                stage_numeric=stage_num,
                mmr_deficient=mmr_def,
                n_bootstrap=10000,
            ),
            oof_tile_subset_mif_oracle=ccs.evaluate_pfs_survival(
                pfs_days=pfs_days,
                recurrence_events=rec_ev,
                protective_score=oof_gt_cont,
                dichotomy_cutoff=2.0,
                stage_numeric=stage_num,
                mmr_deficient=mmr_def,
                n_bootstrap=10000,
            ),
        )

    for b_key in backbones:
        b_t0 = time.monotonic()
        cache_tag = "all40" if include_sealed else "dev32"
        c_path = cache_dir / f"{b_key}_{cache_tag}_{len(bank['df'])}.npz"
        up_all = None
        if c_path.exists():
            with np.load(c_path) as npz:
                cand = npz["pred_u8"]
                if cand.shape[0] == len(bank["df"]) and cand.shape[2:] == bank["he_nhwc"].shape[1:3]:
                    up_all = cand
        if up_all is not None:
            b_family = ORION_BACKBONE_FAMILIES.get(b_key, b_key)
        else:
            pred_obj = instantiate_orion_backbone(
                b_key,
                Path(args.sanofi_ckpt_root) if args.sanofi_ckpt_root else None,
                Path(args.mipheivit_src) if args.mipheivit_src else None,
                Path(args.hoptimus0_ckpt) if args.hoptimus0_ckpt else None,
                int(args.crop_size),
                device,
            )
            b_family = pred_obj.family
            up_all = predict_orion_upstream(
                pred_obj,
                bank["he_nhwc"],
                cache_path=c_path,
            )
            del pred_obj
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(json.dumps(dict(event="orion_upstream_ready", backbone=b_key, shape=list(up_all.shape))), flush=True)

        b_state_f0 = state_dir / b_key / "fold0"
        b_state_f0.mkdir(parents=True, exist_ok=True)
        cutoff_json = b_state_f0 / "fold0_physical_cutoffs.json"

        if args.mode == "s5_fold0_val":
            cutoffs_u8, tau_phys_float = calibrate_orion_physical_cutoffs_from_train(
                bank["targ_nchw"][f0_train_idx_all], f0_train_idx_all, bank["nuclei_records"]
            )
            cutoff_json.write_text(
                json.dumps(dict(cutoffs_u8=cutoffs_u8, tau_phys_float=tau_phys_float), indent=2) + "\n"
            )
        else:
            if not cutoff_json.exists():
                raise FileNotFoundError(f"Missing frozen Fold 0 cutoff file: {cutoff_json}")
            c_doc = json.loads(cutoff_json.read_text())
            cutoffs_u8 = list(c_doc["cutoffs_u8"])
            tau_phys_float = dict(c_doc["tau_phys_float"])

        # --- Part A: All 16 channels on Fold 0 (s5_val on inner_val OR s6 on sealed outer_test) ---
        up_f0_tr = up_all[f0_train_idx]
        up_f0_ev = up_all[f0_eval_idx]
        arm_f0_tr: dict[str, np.ndarray] = {"upstream": up_f0_tr}
        arm_f0_ev: dict[str, np.ndarray] = {"upstream": up_f0_ev}
        for arm in suh.ARMS:
            for sd in seeds:
                arm_f0_tr[f"{arm}:{sd}"] = np.empty_like(up_f0_tr)
                arm_f0_ev[f"{arm}:{sd}"] = np.empty_like(up_f0_ev)
            arm_f0_tr[f"{arm}:ens2"] = np.empty_like(up_f0_tr)
            arm_f0_ev[f"{arm}:ens2"] = np.empty_like(up_f0_ev)

        ch_iter = list(enumerate(ORION_MARKERS))
        if getattr(args, "reverse_channels", False):
            ch_iter = list(reversed(ch_iter))

        for ch16, m_name in ch_iter:
            c12_tr = build_common12_orion(bank["he_nhwc"][f0_train_idx], up_f0_tr, ch16, device=device)
            d_tr_t = torch.from_numpy(up_f0_tr[:, ch16])
            y_tr_t = torch.from_numpy(bank["targ_nchw"][f0_train_idx, ch16])
            d_ev_t = torch.from_numpy(up_f0_ev[:, ch16])

            fitted_models: dict[tuple[str, int], object] = {}
            for arm in suh.ARMS:
                tr_s = []
                for sd in seeds:
                    ck_p = b_state_f0 / f"{m_name}_{arm}_{sd}.pt"
                    mt_p = b_state_f0 / f"{m_name}_{arm}_{sd}_meta.json"
                    if args.mode == "s5_fold0_val":
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
                                cutoff=int(cutoffs_u8[ch16]),
                                device=device,
                                epochs=int(args.epochs),
                                batch_size=int(args.batch_size),
                                lr=float(getattr(args, "lr", 1.5e-3)),
                                deploy_anchor_weight=float(getattr(args, "deploy_anchor_weight", 1.25)),
                            )
                            _atomic_torch_save({k: v.cpu() for k, v in model.state_dict().items()}, ck_p)
                            _atomic_write_text(mt_p, json.dumps(meta, indent=2) + "\n")
                    else:
                        if not ck_p.exists():
                            raise FileNotFoundError(f"Missing frozen Fold 0 head: {ck_p}")
                        model = suh.build_coordinate_screened_model(sd, device)
                        model.load_state_dict(torch.load(ck_p, map_location=device, weights_only=True))

                    p_tr, _ = suh.predict_universal_marker(model, c12_tr, d_tr_t, device=device, batch_size=32)
                    arm_f0_tr[f"{arm}:{sd}"][:, ch16] = p_tr
                    tr_s.append(p_tr.astype(np.float32))
                    fitted_models[(arm, sd)] = model.cpu()
                arm_f0_tr[f"{arm}:ens2"][:, ch16] = np.round(np.mean(tr_s, axis=0)).astype(np.uint8)

            del c12_tr
            gc.collect()

            c12_ev = build_common12_orion(bank["he_nhwc"][f0_eval_idx], up_f0_ev, ch16, device=device)
            for arm in suh.ARMS:
                ev_s = []
                for sd in seeds:
                    model = fitted_models.pop((arm, sd)).to(device)  # type: ignore[attr-defined]
                    p_ev, _ = suh.predict_universal_marker(model, c12_ev, d_ev_t, device=device, batch_size=32)
                    arm_f0_ev[f"{arm}:{sd}"][:, ch16] = p_ev
                    ev_s.append(p_ev.astype(np.float32))
                    del model
                arm_f0_ev[f"{arm}:ens2"][:, ch16] = np.round(np.mean(ev_s, axis=0)).astype(np.uint8)

            del c12_ev, fitted_models
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        arms_16ch, pat_macro_16ch = evaluate_16ch_split(
            train_indices=f0_train_idx,
            eval_indices=f0_eval_idx,
            bank=bank,
            arm_train_preds=arm_f0_tr,
            arm_eval_preds=arm_f0_ev,
            cutoffs_u8=cutoffs_u8,
            tau_phys_float=tau_phys_float,
            device=device,
        )
        sc_ens = "S_count:ens2"
        sr_ens = "S_recon:ens2"
        paired_pat_boot = dict(
            scount_ens2_vs_upstream={
                mk: jce.paired_cluster_bootstrap(pat_macro_16ch[sc_ens], pat_macro_16ch["upstream"], mk)
                for mk in ("Fixed_tol_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB")
            },
            scount_ens2_vs_srecon_ens2={
                mk: jce.paired_cluster_bootstrap(pat_macro_16ch[sc_ens], pat_macro_16ch[sr_ens], mk)
                for mk in ("Fixed_tol_F1", "Cal_F1", "ODS_F", "AP", "SSIM", "PSNR_dB")
            },
        )

        # --- Part B: 5-Fold Patient-Disjoint Clinical PFS Survival (N=40) in s6_confirmatory ---
        survival_5fold_out = None
        if args.mode == "s6_confirmatory":
            vis_dir = out_dir / "visual_crops"
            vis_dir.mkdir(parents=True, exist_ok=True)
            if len(f0_eval_idx) >= 2:
                vi = [0, len(f0_eval_idx) // 2]
                np.savez_compressed(
                    vis_dir / f"orion_{b_key}_examples.npz",
                    patients=bank["patients"][f0_eval_idx[vi]],
                    he_nhwc=bank["he_nhwc"][f0_eval_idx[vi]],
                    targ_nchw=bank["targ_nchw"][f0_eval_idx[vi]],
                    upstream_nchw=arm_f0_ev["upstream"][vi],
                    srecon_nchw=arm_f0_ev[sr_ens][vi],
                    scount_nchw=arm_f0_ev[sc_ens][vi],
                )

            pat_to_row = {str(p): idx for idx, p in enumerate(clin_df["PatientID"].tolist())}
            oof_scores: dict[str, np.ndarray] = {
                k: np.zeros(len(clin_df), dtype=np.float64)
                for k in ("upstream", sr_ens, sc_ens)
            }
            oof_disc: dict[str, np.ndarray] = {
                k: np.zeros(len(clin_df), dtype=np.float64)
                for k in ("upstream", sr_ens, sc_ens)
            }

            for fd in cv_manifest["folds"]:
                f_id = int(fd["fold"])
                tr_p = _fold_patients(fd, "inner_train")
                te_p = _fold_patients(fd, "outer_test")
                tr_idx_full = np.nonzero(np.isin(bank["patients"], tr_p))[0]
                tr_idx = tr_idx_full[:: int(args.fit_stride)]
                te_idx = np.nonzero(np.isin(bank["patients"], te_p))[0]

                f_cuts_u8, f_tau_float = calibrate_orion_physical_cutoffs_from_train(
                    bank["targ_nchw"][tr_idx_full], tr_idx_full, bank["nuclei_records"]
                )
                tau_triad = {m: f_tau_float[m] for m in PROGNOSTIC_TRIAD}

                # Triad predictions (N_sub, 3, H, W) in order PROGNOSTIC_TRIAD = (Pan-CK, CD3e, CD8a)
                triad_tr = {
                    "upstream": np.stack([up_all[tr_idx, PROGNOSTIC_CH16[m]] for m in PROGNOSTIC_TRIAD], axis=1),
                    sr_ens: np.empty((len(tr_idx), 3, int(args.crop_size), int(args.crop_size)), dtype=np.uint8),
                    sc_ens: np.empty((len(tr_idx), 3, int(args.crop_size), int(args.crop_size)), dtype=np.uint8),
                }
                triad_te = {
                    "upstream": np.stack([up_all[te_idx, PROGNOSTIC_CH16[m]] for m in PROGNOSTIC_TRIAD], axis=1),
                    sr_ens: np.empty((len(te_idx), 3, int(args.crop_size), int(args.crop_size)), dtype=np.uint8),
                    sc_ens: np.empty((len(te_idx), 3, int(args.crop_size), int(args.crop_size)), dtype=np.uint8),
                }

                f_state_dir = state_dir / b_key / f"fold{f_id}"
                f_state_dir.mkdir(parents=True, exist_ok=True)

                for ti_m, m_name in enumerate(PROGNOSTIC_TRIAD):
                    ch16 = PROGNOSTIC_CH16[m_name]
                    c12_tr = build_common12_orion(bank["he_nhwc"][tr_idx], up_all[tr_idx], ch16, device=device)
                    c12_te = build_common12_orion(bank["he_nhwc"][te_idx], up_all[te_idx], ch16, device=device)
                    d_tr_t = torch.from_numpy(up_all[tr_idx, ch16])
                    y_tr_t = torch.from_numpy(bank["targ_nchw"][tr_idx, ch16])
                    d_te_t = torch.from_numpy(up_all[te_idx, ch16])

                    for arm, arm_ens_key in (("S_recon", sr_ens), ("S_count", sc_ens)):
                        tr_s, te_s = [], []
                        for sd in seeds:
                            ck_p = f_state_dir / f"{m_name}_{arm}_{sd}.pt"
                            if ck_p.exists():
                                model = suh.build_coordinate_screened_model(sd, device)
                                model.load_state_dict(torch.load(ck_p, map_location=device, weights_only=True))
                            else:
                                model, _ = suh.fit_universal_marker(
                                    arm=arm,
                                    seed=sd,
                                    common_train=c12_tr,
                                    donor_u8_train=d_tr_t,
                                    target_u8_train=y_tr_t,
                                    cutoff=int(f_cuts_u8[ch16]),
                                    device=device,
                                    epochs=int(args.epochs),
                                    batch_size=int(args.batch_size),
                                    lr=float(getattr(args, "lr", 1.5e-3)),
                                    deploy_anchor_weight=float(getattr(args, "deploy_anchor_weight", 1.25)),
                                )
                                _atomic_torch_save({k: v.cpu() for k, v in model.state_dict().items()}, ck_p)
                            p_tr, _ = suh.predict_universal_marker(model, c12_tr, d_tr_t, device=device, batch_size=32)
                            p_te, _ = suh.predict_universal_marker(model, c12_te, d_te_t, device=device, batch_size=32)
                            tr_s.append(p_tr.astype(np.float32))
                            te_s.append(p_te.astype(np.float32))
                            del model
                        triad_tr[arm_ens_key][:, ti_m] = np.round(np.mean(tr_s, axis=0)).astype(np.uint8)
                        triad_te[arm_ens_key][:, ti_m] = np.round(np.mean(te_s, axis=0)).astype(np.uint8)
                    del c12_tr, c12_te

                # Score out-of-fold ImmunoScore strictly from inner_train reference percentiles
                for arm_k in ("upstream", sr_ens, sc_ens):
                    tr_comp = compute_patient_compartments_for_subset(
                        tr_idx, bank["patients"][tr_idx], bank["nuclei_records"], triad_tr[arm_k], tau_triad
                    )
                    te_comp = compute_patient_compartments_for_subset(
                        te_idx, bank["patients"][te_idx], bank["nuclei_records"], triad_te[arm_k], tau_triad
                    )
                    tr_present = [p for p in tr_p if p in tr_comp]
                    te_present = [p for p in te_p if p in te_comp]
                    tr_vecs = {k: np.array([tr_comp[p][k] for p in tr_present]) for k in ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2")}
                    te_vecs = {k: np.array([te_comp[p][k] for p in te_present]) for k in ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2")}
                    sc_res = ccs.score_immunoscore_from_training_ref(tr_vecs, te_vecs)
                    for pi, pat_id in enumerate(te_present):
                        r_i = pat_to_row[pat_id]
                        oof_scores[arm_k][r_i] = sc_res["immunoscore_continuous"][pi]
                        oof_disc[arm_k][r_i] = sc_res["immunoscore_discrete"][pi]

            pfs_days = clin_df["PFSDays"].to_numpy(dtype=np.float64)
            rec_ev = clin_df["Recurrence"].to_numpy(dtype=np.int32)
            stage_num = clin_df["Stage"].map({"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}).to_numpy(dtype=np.float64)
            mmr_def = (clin_df["MMRIHC"].fillna("Intact").str.lower() != "intact").astype(np.float64).to_numpy()

            surv_up = ccs.evaluate_pfs_survival(
                pfs_days=pfs_days,
                recurrence_events=rec_ev,
                protective_score=oof_scores["upstream"],
                dichotomy_cutoff=2.0,
                stage_numeric=stage_num,
                mmr_deficient=mmr_def,
                n_bootstrap=10000,
            )
            surv_sr = ccs.evaluate_pfs_survival(
                pfs_days=pfs_days,
                recurrence_events=rec_ev,
                protective_score=oof_scores[sr_ens],
                dichotomy_cutoff=2.0,
                stage_numeric=stage_num,
                mmr_deficient=mmr_def,
                reference_protective_score=oof_scores["upstream"],
                n_bootstrap=10000,
            )
            surv_sc = ccs.evaluate_pfs_survival(
                pfs_days=pfs_days,
                recurrence_events=rec_ev,
                protective_score=oof_scores[sc_ens],
                dichotomy_cutoff=2.0,
                stage_numeric=stage_num,
                mmr_deficient=mmr_def,
                reference_protective_score=oof_scores["upstream"],
                n_bootstrap=10000,
            )
            surv_sc_vs_sr = ccs.evaluate_pfs_survival(
                pfs_days=pfs_days,
                recurrence_events=rec_ev,
                protective_score=oof_scores[sc_ens],
                dichotomy_cutoff=2.0,
                reference_protective_score=oof_scores[sr_ens],
                n_bootstrap=10000,
            )
            surv_sc["delta_c_index_vs_srecon"] = surv_sc_vs_sr["delta_c_index_vs_ref"]
            surv_sc["delta_c_index_vs_srecon_ci95"] = surv_sc_vs_sr["delta_c_index_ci95"]
            surv_sc["delta_c_index_vs_srecon_boot_p"] = surv_sc_vs_sr["delta_c_index_boot_p"]

            survival_5fold_out = dict(
                patient_ids=clin_df["PatientID"].tolist(),
                oof_immunoscore_continuous={k: v.tolist() for k, v in oof_scores.items()},
                oof_immunoscore_discrete={k: v.tolist() for k, v in oof_disc.items()},
                upstream=surv_up,
                srecon_ens2=surv_sr,
                scount_ens2=surv_sc,
            )

        b_rec = dict(
            backbone=b_key,
            family=b_family,
            wall_seconds=time.monotonic() - b_t0,
            fold0_physical_cutoffs_u8=dict(zip(ORION_MARKERS, cutoffs_u8)),
            fold0_tau_phys_float=dict(tau_phys_float),
            fold0_16ch_arms=arms_16ch,
            fold0_paired_patient_bootstrap=paired_pat_boot,
            fold0_per_patient_macro=pat_macro_16ch,
            survival_5fold_n40=survival_5fold_out,
        )
        summary_by_backbone[b_key] = b_rec
        (out_dir / f"orion_backbone_{b_key}_{args.mode}.json").write_text(json.dumps(b_rec, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    event="orion_backbone_complete",
                    backbone=b_key,
                    mode=args.mode,
                    up_Fixed_16ch=round(arms_16ch["upstream"]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                    sr_Fixed_16ch=round(arms_16ch[sr_ens]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                    sc_Fixed_16ch=round(arms_16ch[sc_ens]["macro_pixel_16ch"]["Fixed_tol_F1"], 4),
                    up_Cell_15cl=round(arms_16ch["upstream"]["single_cell_15class"]["macro_deployable_cell_f1"], 4),
                    sc_Cell_15cl=round(arms_16ch[sc_ens]["single_cell_15class"]["macro_deployable_cell_f1"], 4),
                    c_idx_up=(round(survival_5fold_out["upstream"]["c_index"], 4) if survival_5fold_out else None),
                    c_idx_sc=(round(survival_5fold_out["scount_ens2"]["c_index"], 4) if survival_5fold_out else None),
                    sec=round(b_rec["wall_seconds"], 1),
                )
            ),
            flush=True,
        )

    final_doc = dict(
        schema="orion-multibackbone-survival-v1",
        mode=args.mode,
        crop_size=int(args.crop_size),
        n_tiles_loaded=len(bank["df"]),
        seeds=list(seeds),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(getattr(args, "lr", 1.5e-3)),
        oracle_survival_n40=oracle_survival,
        total_wall_seconds=time.monotonic() - t0,
        backbones=summary_by_backbone,
    )
    (out_dir / f"orion_multibackbone_{args.mode}_summary.json").write_text(json.dumps(final_doc, indent=2) + "\n")
    return final_doc


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("s5_fold0_val", "s6_confirmatory"), required=True)
    p.add_argument("--orion-root", required=True)
    p.add_argument("--cv-manifest", required=True)
    p.add_argument("--clinical-csv", required=True)
    p.add_argument("--benchmark-protocol", default="")
    p.add_argument("--sanofi-ckpt-root", default="")
    p.add_argument("--mipheivit-src", default="")
    p.add_argument("--hoptimus0-ckpt", default="")
    p.add_argument("--backbones", required=True)
    p.add_argument("--state-dir", required=True)
    p.add_argument("--pred-cache-dir", default="")
    p.add_argument("--bank-cache-dir", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--seeds", default="202610270,202610271")
    p.add_argument("--fit-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--deploy-anchor-weight", type=float, default=1.25)
    p.add_argument("--tile-limit-per-slide", type=int, default=0)
    p.add_argument("--overwrite-fits", action="store_true")
    p.add_argument("--reverse-channels", action="store_true")
    return p


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    run_orion_campaign(build_arg_parser().parse_args())
