"""Single-cell nuclear phenotyping, spatial infiltration, and clinical PFS survival.

Provides deterministic, leak-free evaluation across:
  1. Instance-mask nuclear quantification (fast vectorized bincount over Cellpose /
     Orion instance masks with ``min_area`` filtering).
  2. Single-cell classification under BOTH uncalibrated physical cutoffs
     (``deployable_f1``) and training-fold recalibrated cutoffs (``recalibrated_f1``,
     ``auprc``, ``roc_auc``), plus per-tile cell-count concordance (Pearson ``r``,
     Spearman ``rho``, ``mae``, Bland-Altman bias & 95% LoA).
  3. Epithelial-immune spatial microenvironment: nearest-neighbor ``Pan-CK+ -> CD3+/CD8a+``
     infiltration distances, ``R1`` (tumor core) vs ``R2`` (invasive margin) compartment
     densities, and Galon/Orion 4-compartment ``ImmunoScore`` using training-fold
     empirical percentiles only.
  4. Patient-level progression-free survival (``PFSDays``, ``Recurrence``): exact
     Harrell's C-index, 10,000-resample paired patient bootstrap 95% CIs, Kaplan-Meier
     curves, Mantel-Haenszel log-rank test, and Breslow-tied Newton-Raphson Cox
     proportional hazards (univariable + Stage/MMR multivariable adjusted).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy import spatial, stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def extract_tile_nuclei(
    intensity_chw: np.ndarray,
    mask_hw: np.ndarray,
    min_area: int = 10,
) -> dict[str, np.ndarray]:
    """Vectorized extraction of per-nucleus mean intensities, centroids, and areas.

    Args:
        intensity_chw: (C, H, W) float or uint8 marker image.
        mask_hw: (H, W) integer nuclear instance segmentation mask (0 = background).
        min_area: minimum pixel area per nucleus (default 10, matching Sanofi Orion/HEMIT).

    Returns:
        dict with ``cell_ids`` (K,), ``areas`` (K,), ``centroids_yx`` (K, 2),
        and ``means`` (K, C) in float64.
    """
    if mask_hw.ndim != 2 or intensity_chw.ndim != 3 or intensity_chw.shape[1:] != mask_hw.shape:
        raise ValueError(
            f"Shape mismatch: intensity {intensity_chw.shape} vs mask {mask_hw.shape}"
        )
    c_dim, h, w = intensity_chw.shape
    flat_mask = mask_hw.reshape(-1).astype(np.int64, copy=False)
    fg = flat_mask > 0
    if not np.any(fg):
        return dict(
            cell_ids=np.zeros((0,), dtype=np.int64),
            areas=np.zeros((0,), dtype=np.int64),
            centroids_yx=np.zeros((0, 2), dtype=np.float64),
            means=np.zeros((0, c_dim), dtype=np.float64),
        )

    fg_ids = flat_mask[fg]
    uniq_ids, inv = np.unique(fg_ids, return_inverse=True)
    areas = np.bincount(inv)
    keep = areas >= int(min_area)
    if not np.any(keep):
        return dict(
            cell_ids=np.zeros((0,), dtype=np.int64),
            areas=np.zeros((0,), dtype=np.int64),
            centroids_yx=np.zeros((0, 2), dtype=np.float64),
            means=np.zeros((0, c_dim), dtype=np.float64),
        )

    ys, xs = np.nonzero(mask_hw > 0)
    cy = np.bincount(inv, weights=ys.astype(np.float64)) / areas
    cx = np.bincount(inv, weights=xs.astype(np.float64)) / areas

    flat_img = intensity_chw.reshape(c_dim, -1)[:, fg].astype(np.float64, copy=False)
    means = np.empty((len(uniq_ids), c_dim), dtype=np.float64)
    for ch in range(c_dim):
        means[:, ch] = np.bincount(inv, weights=flat_img[ch]) / areas

    return dict(
        cell_ids=uniq_ids[keep],
        areas=areas[keep],
        centroids_yx=np.stack([cy[keep], cx[keep]], axis=1),
        means=means[keep],
    )


def prepare_tile_nuclei_index(
    mask_hw: np.ndarray,
    selected_cell_ids: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray | int]:
    """Precompute flat foreground pixel indices and row mapping for O(pixels) multi-arm cell extraction."""
    flat_mask = mask_hw.reshape(-1).astype(np.int64, copy=False)
    fg_pos = np.nonzero(flat_mask > 0)[0]
    fg_ids = flat_mask[fg_pos]
    uniq_ids, inv = np.unique(fg_ids, return_inverse=True)
    areas = np.bincount(inv).astype(np.float64)
    id_to_u = {int(cid): idx for idx, cid in enumerate(uniq_ids.tolist())}
    rows = np.asarray([id_to_u[int(cid)] for cid in selected_cell_ids], dtype=np.int64)
    return dict(
        fg_pos=fg_pos,
        inv=inv,
        areas=areas,
        n_uniq=int(len(uniq_ids)),
        rows=rows,
    )


def extract_means_with_index(
    intensity_chw: np.ndarray,
    idx_spec: Mapping[str, np.ndarray | int],
) -> np.ndarray:
    """Extract (K_selected, C) per-nucleus mean intensities using ``prepare_tile_nuclei_index``."""
    c_dim = intensity_chw.shape[0]
    fg_pos = idx_spec["fg_pos"]  # type: ignore[assignment]
    inv = idx_spec["inv"]  # type: ignore[assignment]
    areas = idx_spec["areas"]  # type: ignore[assignment]
    n_uniq = int(idx_spec["n_uniq"])
    rows = idx_spec["rows"]  # type: ignore[assignment]
    flat_img = intensity_chw.reshape(c_dim, -1)[:, fg_pos].astype(np.float64, copy=False)
    means = np.empty((n_uniq, c_dim), dtype=np.float64)
    for ch in range(c_dim):
        means[:, ch] = np.bincount(inv, weights=flat_img[ch], minlength=n_uniq) / areas
    return means[rows]


def fit_best_f1_threshold(scores: np.ndarray, labels: np.ndarray, n_grid: int = 256) -> float:
    """Select threshold on training-set cell scores maximizing binary F1."""
    y = (labels > 0).astype(np.int32)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float(np.median(scores)) if len(scores) else 0.5
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi <= lo + 1e-12:
        return lo
    grid = np.unique(
        np.concatenate([
            np.linspace(lo, hi, n_grid),
            np.quantile(scores, np.linspace(0.01, 0.99, 99)),
        ])
    )
    best_thr, best_f1 = float(grid[0]), -1.0
    pos_total = int(y.sum())
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted)
    pred_cnt = np.arange(1, len(y) + 1)
    f1_vec = (2.0 * tp_cum) / np.maximum(pred_cnt + pos_total, 1)
    # Evaluate only at distinct score boundaries
    distinct = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    idx_best = int(distinct[int(np.argmax(f1_vec[distinct]))])
    best_f1 = float(f1_vec[idx_best])
    if idx_best + 1 < len(s_sorted):
        best_thr = 0.5 * float(s_sorted[idx_best] + s_sorted[idx_best + 1])
    else:
        best_thr = float(s_sorted[idx_best]) - 1e-9
    return best_thr


def evaluate_single_cell_marker(
    *,
    train_gt_scores: np.ndarray,
    train_pred_scores: np.ndarray,
    train_labels: np.ndarray,
    eval_pred_scores: np.ndarray,
    eval_labels: np.ndarray,
    eval_tile_ids: np.ndarray,
    physical_cutoff: float | None = None,
) -> dict[str, float]:
    """Evaluate single-cell classification and tile-level cell-count concordance.

    ``physical_cutoff``: if None, learned on ``train_gt_scores`` vs ``train_labels``
    (representing the fixed physical fluorescence positivity threshold tau_phys).
    ``recal_cutoff``: learned on ``train_pred_scores`` vs ``train_labels`` strictly
    inside the training split.
    """
    y_tr = (train_labels > 0).astype(np.int32)
    y_ev = (eval_labels > 0).astype(np.int32)

    tau_phys = (
        float(physical_cutoff)
        if physical_cutoff is not None
        else fit_best_f1_threshold(train_gt_scores, y_tr)
    )
    tau_recal = fit_best_f1_threshold(train_pred_scores, y_tr)

    pred_deploy = (eval_pred_scores > tau_phys).astype(np.int32)
    pred_recal = (eval_pred_scores > tau_recal).astype(np.int32)

    deploy_f1 = float(f1_score(y_ev, pred_deploy, zero_division=0))
    recal_f1 = float(f1_score(y_ev, pred_recal, zero_division=0))

    if 0 < int(y_ev.sum()) < len(y_ev):
        auprc = float(average_precision_score(y_ev, eval_pred_scores))
        roc_auc = float(roc_auc_score(y_ev, eval_pred_scores))
    else:
        auprc = float("nan")
        roc_auc = float("nan")

    # Tile-level positive cell counts
    uniq_tiles, inv = np.unique(eval_tile_ids, return_inverse=True)
    gt_counts = np.bincount(inv, weights=y_ev.astype(np.float64))
    deploy_counts = np.bincount(inv, weights=pred_deploy.astype(np.float64))
    recal_counts = np.bincount(inv, weights=pred_recal.astype(np.float64))

    def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0, 0.0
        r_val = float(stats.pearsonr(a, b)[0])
        rho_val = float(stats.spearmanr(a, b)[0])
        return r_val, rho_val

    r_dep, rho_dep = _corr(gt_counts, deploy_counts)
    r_rec, rho_rec = _corr(gt_counts, recal_counts)
    diff_dep = deploy_counts - gt_counts
    mae_dep = float(np.mean(np.abs(diff_dep))) if len(diff_dep) else 0.0
    ba_bias = float(np.mean(diff_dep)) if len(diff_dep) else 0.0
    ba_sd = float(np.std(diff_dep, ddof=1)) if len(diff_dep) > 1 else 0.0

    return dict(
        n_eval_cells=int(len(y_ev)),
        n_pos_cells=int(y_ev.sum()),
        n_eval_tiles=int(len(uniq_tiles)),
        tau_physical=tau_phys,
        tau_recalibrated=tau_recal,
        deployable_cell_f1=deploy_f1,
        recalibrated_cell_f1=recal_f1,
        cell_auprc=auprc,
        cell_roc_auc=roc_auc,
        tile_count_pearson_r_deploy=r_dep,
        tile_count_spearman_rho_deploy=rho_dep,
        tile_count_pearson_r_recal=r_rec,
        tile_count_spearman_rho_recal=rho_rec,
        tile_count_mae_deploy=mae_dep,
        bland_altman_bias=ba_bias,
        bland_altman_loa_low=ba_bias - 1.96 * ba_sd,
        bland_altman_loa_high=ba_bias + 1.96 * ba_sd,
    )


def compute_tile_infiltration_metrics(
    centroids_yx: np.ndarray,
    tumor_pos: np.ndarray,
    immune_pos: np.ndarray,
    pixel_size_um: float = 0.325,
    close_radius_um: float = 25.0,
) -> dict[str, float]:
    """Compute tumor-to-nearest-T-cell distance and proximity fraction on one tile/ROI."""
    t_idx = np.nonzero(tumor_pos > 0)[0]
    i_idx = np.nonzero(immune_pos > 0)[0]
    n_cells = len(centroids_yx)
    tumor_frac = float(len(t_idx) / max(n_cells, 1))
    immune_frac = float(len(i_idx) / max(n_cells, 1))
    if len(t_idx) == 0 or len(i_idx) == 0:
        return dict(
            tumor_frac=tumor_frac,
            immune_frac=immune_frac,
            median_nn_um=float("nan"),
            within_25um_frac=0.0,
        )
    tree = spatial.cKDTree(centroids_yx[i_idx] * float(pixel_size_um))
    dists, _ = tree.query(centroids_yx[t_idx] * float(pixel_size_um), k=1)
    return dict(
        tumor_frac=tumor_frac,
        immune_frac=immune_frac,
        median_nn_um=float(np.median(dists)),
        within_25um_frac=float(np.mean(dists <= float(close_radius_um))),
    )


def compute_patient_compartment_densities(
    tile_total_cells: np.ndarray,
    tile_panck_pos: np.ndarray,
    tile_cd3_pos: np.ndarray,
    tile_cd8_pos: np.ndarray,
    r1_panck_min: float = 0.30,
    r2_panck_min: float = 0.03,
) -> dict[str, float]:
    """Aggregate patient R1 (tumor center) and R2 (invasive margin) normalized CD3/CD8 densities.

    Mirrors Lin et al. (Nature Cancer 2023) Orion CRC compartment architecture:
      * ``R1`` (tumor core): epithelial-rich tiles (Pan-CK+ fraction >= ``r1_panck_min``)
      * ``R2`` (invasive margin / tumor-stroma interface): ``r2_panck_min <= Pan-CK+ < r1_panck_min``
    Falls back smoothly to overall positive density if a slide biopsy lacks one compartment.
    """
    tot = np.maximum(tile_total_cells.astype(np.float64), 1.0)
    pck_frac = tile_panck_pos.astype(np.float64) / tot
    r1_mask = pck_frac >= r1_panck_min
    r2_mask = (pck_frac >= r2_panck_min) & (pck_frac < r1_panck_min)

    overall_cd3 = float(np.sum(tile_cd3_pos) / np.maximum(np.sum(tile_total_cells), 1.0))
    overall_cd8 = float(np.sum(tile_cd8_pos) / np.maximum(np.sum(tile_total_cells), 1.0))

    if np.any(r1_mask):
        cd3_r1 = float(np.sum(tile_cd3_pos[r1_mask]) / np.maximum(np.sum(tile_total_cells[r1_mask]), 1.0))
        cd8_r1 = float(np.sum(tile_cd8_pos[r1_mask]) / np.maximum(np.sum(tile_total_cells[r1_mask]), 1.0))
    else:
        cd3_r1, cd8_r1 = overall_cd3, overall_cd8

    if np.any(r2_mask):
        cd3_r2 = float(np.sum(tile_cd3_pos[r2_mask]) / np.maximum(np.sum(tile_total_cells[r2_mask]), 1.0))
        cd8_r2 = float(np.sum(tile_cd8_pos[r2_mask]) / np.maximum(np.sum(tile_total_cells[r2_mask]), 1.0))
    else:
        cd3_r2, cd8_r2 = overall_cd3, overall_cd8

    return dict(
        norm_CD3R1=cd3_r1,
        norm_CD3R2=cd3_r2,
        norm_CD8R1=cd8_r1,
        norm_CD8R2=cd8_r2,
        mean_CD3ep=overall_cd3,
        mean_CD8ap=overall_cd8,
        n_tiles_r1=int(np.sum(r1_mask)),
        n_tiles_r2=int(np.sum(r2_mask)),
    )


def score_immunoscore_from_training_ref(
    train_compartments: Mapping[str, np.ndarray],
    eval_compartments: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute continuous [0, 4] and discrete {0..4} Galon ImmunoScore from training percentiles."""
    keys = ("norm_CD3R1", "norm_CD3R2", "norm_CD8R1", "norm_CD8R2")
    n_ev = len(eval_compartments[keys[0]])
    pct_sum = np.zeros(n_ev, dtype=np.float64)
    bin_sum = np.zeros(n_ev, dtype=np.float64)

    for k in keys:
        ref = np.sort(np.asarray(train_compartments[k], dtype=np.float64))
        ev = np.asarray(eval_compartments[k], dtype=np.float64)
        # Empirical CDF rank in [0, 1] anchored strictly to training patients
        ranks = np.searchsorted(ref, ev, side="right") / max(len(ref), 1)
        med = float(np.median(ref)) if len(ref) else 0.0
        pct_sum += ranks
        bin_sum += (ev > med).astype(np.float64)

    return dict(
        immunoscore_continuous=pct_sum,  # in [0, 4]
        immunoscore_discrete=bin_sum,    # integer 0..4 matching Lin et al. ImmunoScore
    )


def harrell_c_index(
    times: np.ndarray,
    events: np.ndarray,
    protective_scores: np.ndarray,
    unit_ids: np.ndarray | None = None,
) -> float:
    """Exact Harrell's concordance index where higher ``protective_scores`` = longer survival.

    When ``unit_ids`` is supplied (e.g. original patient indices during bootstrap resampling),
    pairs formed by two copies of the same underlying patient are excluded from permissible pairs.
    """
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=np.int32)
    s = np.asarray(protective_scores, dtype=np.float64)
    u = np.asarray(unit_ids) if unit_ids is not None else None
    n = len(t)
    concordant = 0.0
    permissible = 0.0
    for i in range(n):
        if e[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if u is not None and u[i] == u[j]:
                continue
            if t[i] < t[j]:
                permissible += 1.0
                if abs(s[i] - s[j]) <= 1e-12:
                    concordant += 0.5
                elif s[i] < s[j]:
                    concordant += 1.0
            elif t[i] == t[j] and e[j] == 0:
                permissible += 1.0
                if abs(s[i] - s[j]) <= 1e-12:
                    concordant += 0.5
                elif s[i] < s[j]:
                    concordant += 1.0
            elif t[i] == t[j] and e[j] == 1 and i < j:
                permissible += 1.0
                if abs(s[i] - s[j]) <= 1e-12:
                    concordant += 1.0
                else:
                    concordant += 0.5
    return float(concordant / permissible) if permissible > 0 else 0.5


def kaplan_meier_curve(times: np.ndarray, events: np.ndarray) -> dict[str, np.ndarray]:
    """Compute exact product-limit Kaplan-Meier survival step function with Greenwood SE."""
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=np.int32)
    uniq_t = np.unique(np.sort(t))
    surv = [1.0]
    se_var = [0.0]
    t_out = [0.0]
    at_risk_out = [int(len(t))]
    cur_s = 1.0
    cur_v = 0.0
    for ut in uniq_t:
        n_i = int(np.sum(t >= ut))
        d_i = int(np.sum((t == ut) & (e == 1)))
        if d_i > 0 and n_i > 0:
            cur_s *= (1.0 - d_i / n_i)
            if n_i > d_i:
                cur_v += d_i / (n_i * (n_i - d_i))
        t_out.append(float(ut))
        surv.append(float(cur_s))
        se_var.append(float((cur_s ** 2) * cur_v))
        at_risk_out.append(n_i)
    se = np.sqrt(np.asarray(se_var, dtype=np.float64))
    s_arr = np.asarray(surv, dtype=np.float64)
    return dict(
        time=np.asarray(t_out, dtype=np.float64),
        survival=s_arr,
        se=se,
        ci_low=np.clip(s_arr - 1.96 * se, 0.0, 1.0),
        ci_high=np.clip(s_arr + 1.96 * se, 0.0, 1.0),
        at_risk=np.asarray(at_risk_out, dtype=np.int64),
    )


def logrank_two_group(
    times: np.ndarray,
    events: np.ndarray,
    high_risk_group: np.ndarray,
) -> dict[str, float]:
    """Exact Mantel-Haenszel two-group log-rank test (high_risk_group == 1 vs 0)."""
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=np.int32)
    g = (np.asarray(high_risk_group) > 0).astype(np.int32)

    event_times = np.unique(t[e == 1])
    obs_1, exp_1, var_1 = 0.0, 0.0, 0.0
    obs_0, exp_0 = 0.0, 0.0

    for ut in event_times:
        at_risk = t >= ut
        n = int(np.sum(at_risk))
        if n <= 1:
            continue
        n1 = int(np.sum(at_risk & (g == 1)))
        d = int(np.sum((t == ut) & (e == 1)))
        d1 = int(np.sum((t == ut) & (e == 1) & (g == 1)))
        e1 = d * (n1 / n)
        v1 = (n1 * (n - n1) * d * (n - d)) / ((n ** 2) * (n - 1))
        obs_1 += d1
        exp_1 += e1
        obs_0 += (d - d1)
        exp_0 += (d - e1)
        var_1 += v1

    if var_1 <= 1e-15:
        return dict(chisq=0.0, p_value=1.0, hr_mantel_haenszel=1.0)
    chisq = ((obs_1 - exp_1) ** 2) / var_1
    p_val = float(stats.chi2.sf(chisq, df=1))
    hr_mh = ((obs_1 / max(exp_1, 1e-9)) / max(obs_0 / max(exp_0, 1e-9), 1e-9))
    return dict(chisq=float(chisq), p_value=p_val, hr_mantel_haenszel=float(hr_mh))


def fit_cox_ph_breslow(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    l2_pen: float = 1e-4,
    max_iter: int = 60,
) -> dict[str, np.ndarray]:
    """Fit Cox proportional hazards via Newton-Raphson on Breslow partial log-likelihood."""
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=np.int32)
    X = np.asarray(covariates, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)
    uniq_ev_t = np.unique(t[e == 1])

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        exp_eta = np.exp(eta)
        grad = -l2_pen * beta
        hess = -l2_pen * np.eye(p, dtype=np.float64)

        for ut in uniq_ev_t:
            ev_mask = (t == ut) & (e == 1)
            d = int(np.sum(ev_mask))
            if d == 0:
                continue
            risk_mask = t >= ut
            w = exp_eta[risk_mask]
            s0 = float(np.sum(w))
            if s0 <= 1e-15:
                continue
            Xr = X[risk_mask]
            s1 = (w[:, None] * Xr).sum(axis=0)
            mean_x = s1 / s0
            grad += X[ev_mask].sum(axis=0) - d * mean_x
            diff = Xr - mean_x[None, :]
            cov_x = ((w[:, None] * diff).T @ diff) / s0
            hess -= d * cov_x

        try:
            step = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(-hess) @ grad
        beta_new = beta + step
        if float(np.max(np.abs(step))) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    try:
        cov_beta = np.linalg.inv(-hess)
    except np.linalg.LinAlgError:
        cov_beta = np.linalg.pinv(-hess)
    se = np.sqrt(np.maximum(np.diag(cov_beta), 1e-15))
    z_scores = beta / se
    p_vals = 2.0 * stats.norm.sf(np.abs(z_scores))
    hr = np.exp(beta)
    hr_low = np.exp(beta - 1.96 * se)
    hr_high = np.exp(beta + 1.96 * se)
    return dict(
        beta=beta,
        se=se,
        z=z_scores,
        p_value=p_vals,
        hr=hr,
        hr_ci_low=hr_low,
        hr_ci_high=hr_high,
    )


def evaluate_pfs_survival(
    *,
    pfs_days: np.ndarray,
    recurrence_events: np.ndarray,
    protective_score: np.ndarray,
    dichotomy_cutoff: float,
    stage_numeric: np.ndarray | None = None,
    mmr_deficient: np.ndarray | None = None,
    reference_protective_score: np.ndarray | None = None,
    n_bootstrap: int = 10000,
    seed: int = 20261115,
) -> dict:
    """Complete patient-level survival evaluation with 10,000-bootstrap CIs and Cox PH."""
    t = np.asarray(pfs_days, dtype=np.float64)
    e = np.asarray(recurrence_events, dtype=np.int32)
    s = np.asarray(protective_score, dtype=np.float64)
    n = len(t)

    c_idx = harrell_c_index(t, e, s)
    rng = np.random.default_rng(seed)
    boot_c = np.empty(n_bootstrap, dtype=np.float64)
    boot_delta = np.empty(n_bootstrap, dtype=np.float64) if reference_protective_score is not None else None
    ref_s = np.asarray(reference_protective_score, dtype=np.float64) if reference_protective_score is not None else None

    valid_b = 0
    attempts = 0
    while valid_b < n_bootstrap and attempts < n_bootstrap * 3:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        if np.sum(e[idx] == 1) < 2:
            continue
        c_b = harrell_c_index(t[idx], e[idx], s[idx], unit_ids=idx)
        boot_c[valid_b] = c_b
        if ref_s is not None and boot_delta is not None:
            boot_delta[valid_b] = c_b - harrell_c_index(t[idx], e[idx], ref_s[idx], unit_ids=idx)
        valid_b += 1

    boot_c = boot_c[:valid_b]
    c_low, c_high = (float(np.quantile(boot_c, 0.025)), float(np.quantile(boot_c, 0.975))) if valid_b else (c_idx, c_idx)

    # Low ImmunoScore (<= dichotomy_cutoff) = High Risk (group 1)
    high_risk = (s <= float(dichotomy_cutoff)).astype(np.int32)
    lr_res = logrank_two_group(t, e, high_risk)
    km_high_risk = kaplan_meier_curve(t[high_risk == 1], e[high_risk == 1])
    km_low_risk = kaplan_meier_curve(t[high_risk == 0], e[high_risk == 0])

    # Univariable Cox PH for High-Risk indicator AND continuous standardized score (-z)
    cox_group = fit_cox_ph_breslow(t, e, high_risk.astype(np.float64))
    s_std = (s - np.mean(s)) / max(float(np.std(s)), 1e-8)
    cox_cont = fit_cox_ph_breslow(t, e, -s_std)

    out = dict(
        n_patients=int(n),
        n_events=int(np.sum(e == 1)),
        n_high_risk=int(np.sum(high_risk == 1)),
        n_low_risk=int(np.sum(high_risk == 0)),
        c_index=c_idx,
        c_index_ci95=[c_low, c_high],
        logrank_chisq=lr_res["chisq"],
        logrank_p=lr_res["p_value"],
        hr_high_vs_low_risk=float(cox_group["hr"][0]),
        hr_high_vs_low_ci95=[float(cox_group["hr_ci_low"][0]), float(cox_group["hr_ci_high"][0])],
        hr_high_vs_low_wald_p=float(cox_group["p_value"][0]),
        hr_per_sd_deficit=float(cox_cont["hr"][0]),
        hr_per_sd_ci95=[float(cox_cont["hr_ci_low"][0]), float(cox_cont["hr_ci_high"][0])],
        hr_per_sd_wald_p=float(cox_cont["p_value"][0]),
        km_high_risk={k: v.tolist() for k, v in km_high_risk.items()},
        km_low_risk={k: v.tolist() for k, v in km_low_risk.items()},
    )

    if boot_delta is not None and valid_b > 0:
        bd = boot_delta[:valid_b]
        delta_obs = c_idx - harrell_c_index(t, e, ref_s)
        p_two = float(2.0 * min(np.mean(bd <= 0.0), np.mean(bd >= 0.0)))
        out["delta_c_index_vs_ref"] = float(delta_obs)
        out["delta_c_index_ci95"] = [float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975))]
        out["delta_c_index_boot_p"] = p_two

    if stage_numeric is not None and mmr_deficient is not None:
        st_z = (np.asarray(stage_numeric, dtype=np.float64) - np.mean(stage_numeric)) / max(float(np.std(stage_numeric)), 1e-8)
        mmr_v = np.asarray(mmr_deficient, dtype=np.float64)
        X_multi = np.column_stack([-s_std, st_z, mmr_v])
        cox_multi = fit_cox_ph_breslow(t, e, X_multi)
        out["multivariable_cox"] = dict(
            covariates=["immunoscore_deficit_per_sd", "stage_per_sd", "mmr_deficient"],
            hr=cox_multi["hr"].tolist(),
            hr_ci_low=cox_multi["hr_ci_low"].tolist(),
            hr_ci_high=cox_multi["hr_ci_high"].tolist(),
            p_value=cox_multi["p_value"].tolist(),
        )
    return out
