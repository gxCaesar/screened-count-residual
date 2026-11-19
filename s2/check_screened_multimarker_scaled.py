"""Independent scalar-only verifier and statistical analyzer for multi-marker FIT1192.

Never imports ``run_screened_multimarker_scaled`` or opens pixels or ``.pt.gz``
checkpoints (all source images and state blobs are stat-only). Reconstructs
pooled count curves, ODS-F, AP, fixed-cutoff and training-threshold F1, native
SSIM/PSNR means, per-marker breakdowns, paired bootstrap CIs, and parity against
the accepted single-marker FIT1192-v2 summary.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sys

from .check_query_relation_ensemble import (
    Sources, close, integer, metadata, number, pooled, quality, read_json, require,
    check_quality_source,
)
from .check_query_relation_scaled import (
    CUTOFFS, EXCLUDED_INDICES, MARKERS, NEW_METRICS, PIXELS, SEEDS, TRAINING_INDICES,
    check_membership, check_membership_sources, check_source_coverage, counts, endpoints,
    expected_schedule, iter_rows, learning_rate, thresholds,
)

SCHEMA = 'screened-multimarker-fit1192-v1'
TRAINED_MARKERS = ('CD3', 'DAPI')
MARKER_TO_CHANNEL = {'panCK': 0, 'CD3': 1, 'DAPI': 2}
MARKER_TO_CUTOFF = {'panCK': 22, 'CD3': 30, 'DAPI': 18}
BASE_ARMS = ('S_count', 'S_recon')
EVAL_ARMS = (
    'S_count_3ch',
    'S_recon_3ch',
    'S_count_panck_cd3',
    'S_count_panck_dapi',
    'S_count_panck_only',
    'S_recon_panck_only',
)
COMPARISONS = (
    ('S_count_3ch', 'S_recon_3ch'),
    ('S_count_3ch', 'S_count_panck_only'),
    ('S_count_panck_cd3', 'S_count_panck_only'),
    ('S_count_panck_dapi', 'S_count_panck_only'),
    ('S_recon_3ch', 'S_recon_panck_only'),
)
CHARTER_FLOORS = dict(
    val_ods_f_sota_floor=0.812652,
    val_donor_ods_f=0.807586,
    val_released_dgr_ods_f=0.713968,
    val_ssim_floor=0.710816,
    val_psnr_floor=26.011020,
)


@dataclass(frozen=True)
class SyntheticMultiMarkerCoverage:
    training_indices: tuple = (1, 2)
    excluded_count: int = 8
    hold: int = 2
    val: int = 1
    epochs: int = 2


def _coverage(coverage: SyntheticMultiMarkerCoverage | None) -> dict:
    require(coverage is None or type(coverage) is SyntheticMultiMarkerCoverage,
            'explicit SyntheticMultiMarkerCoverage required')
    if coverage is None:
        return dict(training_indices=TRAINING_INDICES, excluded_count=8,
                    hold=300, val=630, epochs=32)
    return dict(
        training_indices=coverage.training_indices,
        excluded_count=coverage.excluded_count,
        hold=coverage.hold,
        val=coverage.val,
        epochs=coverage.epochs,
    )


def _check_terminal(path: str | Path, manifest: dict, sources: Sources, expected_reads: int) -> None:
    parent = Path(path).resolve().parent
    exit_path = sources.add(metadata(parent / 'exit-code.txt'))
    require(exit_path.read_text().strip() == '0', 'actual terminal process exit0 required')
    status = read_json(metadata(parent / 'status.json'), sources)
    require(status.get('schema') == SCHEMA
            and status.get('run_id') == manifest['run_id']
            and status.get('phase') == manifest['phase']
            and status.get('status') == 'complete'
            and status.get('actual_target_reads') == expected_reads,
            'matching complete terminal phase/read receipt required')
    for field in ('source_metadata', 'input_metadata', 'data_source_metadata'):
        for row in status.get(field, []):
            sources.add(row)


def _check_calibration(cal: dict, marker: str, cv: dict) -> dict:
    ch = MARKER_TO_CHANNEL[marker]
    cutoff = MARKER_TO_CUTOFF[marker]
    require(cal.get('schema') == SCHEMA and cal.get('marker') == marker
            and cal.get('channel') == ch and cal.get('cutoff') == cutoff
            and cal.get('training_only') is True
            and cal.get('training_members') == len(cv['training_indices'])
            and cal.get('training_indices') == list(cv['training_indices']),
            f'{marker} calibration identity/training-only check failed')
    scales = cal.get('loss_scales')
    require(isinstance(scales, dict) and set(scales) == {'mse', 'ssim', 'edge1', 'edge16'}
            and all(number(v, 'scale') > 0 for v in scales.values()),
            f'{marker} positive loss_scales required')
    rsq = number(cal.get('reconstruction_gradient_sq'), 'rec_sq')
    asq = number(cal.get('auxiliary_gradient_sq'), 'aux_sq')
    lam = number(cal.get('lambda_count'), 'lambda_count')
    require(rsq > 0 and asq > 0 and lam > 0, f'{marker} positive gradient norms required')
    close(lam, math.sqrt(rsq) / math.sqrt(asq), f'{marker} lambda_count ratio')
    donor_c = counts(cal.get('donor_train_counts'), f'{marker}.donor_train_counts',
                     pixels=len(cv['training_indices']) * PIXELS)
    expected_thr = endpoints(donor_c, cutoff, 0)['ODS_threshold']
    require(cal.get('donor_train_threshold') == expected_thr,
            f'{marker} donor_train_threshold mismatch')
    return cal


def _check_milestones(fit: dict, marker: str, members: dict, cv: dict,
                      sources: Sources, calibration: dict) -> int:
    identities = {r['index']: r['name'] for r in members['training']}
    n = len(identities)
    epochs = cv['epochs']
    total = n * epochs
    schedule_map = {
        idx: item
        for idx, item in enumerate(expected_schedule(fit['seed'], cv['training_indices'], epochs), 1)
    }
    seen_indices = []
    for row in iter_rows(fit['updates_milestones'], sources):
        u_idx = integer(row.get('update_index'), 'milestone.update_index')
        require(1 <= u_idx <= total, 'milestone update_index out of range')
        seen_indices.append(u_idx)
        ep, orig_idx, code = schedule_map[u_idx]
        require(row.get('marker') == marker
                and row.get('channel') == MARKER_TO_CHANNEL[marker]
                and row.get('cutoff') == MARKER_TO_CUTOFF[marker]
                and row.get('arm') == fit['arm']
                and row.get('seed') == fit['seed']
                and row.get('epoch') == ep
                and row.get('original_index') == orig_idx
                and row.get('name') == identities[orig_idx]
                and row.get('D4_code') == code,
                'milestone schedule/identity mismatch')
        close(row.get('learning_rate'), learning_rate(u_idx, n, epochs), 'milestone lr')
        for f_name in ('train_total', 'base', 'aux', 'grad_norm'):
            number(row.get(f_name), 'milestone.' + f_name)
        if fit['arm'].endswith('_count'):
            require(0 <= number(row.get('soft_phi'), 'soft_phi') <= 1 / 3 + 1e-12
                    and type(row.get('threshold')) is int and 0 <= row['threshold'] < 255,
                    'count milestone soft_phi/threshold invalid')
            close(row['train_total'], row['base'] + calibration['lambda_count'] * row['aux'],
                  'count surrogate total')
        else:
            require(row.get('soft_phi') is None and row.get('threshold') is None and row['aux'] == 0,
                    'recon milestone must have zero aux')
    require(seen_indices[0] == 1 and seen_indices[-1] == total
            and seen_indices == sorted(set(seen_indices)),
            'ordered milestones including first and final update required')
    summary = fit.get('updates_summary', {})
    require(summary.get('total_updates') == total and summary.get('all_finite') is True,
            'updates_summary total/finite mismatch')
    return len(seen_indices)


def _check_trajectory(fit: dict, marker: str, cv: dict, sources: Sources) -> list[dict]:
    n = len(cv['training_indices'])
    epochs = cv['epochs']
    rows = list(iter_rows(fit['trajectory'], sources))
    require(len(rows) == epochs, f'{marker} trajectory must have {epochs} epoch rows')
    for ep, row in enumerate(rows, 1):
        require(row.get('marker') == marker
                and row.get('arm') == fit['arm']
                and row.get('seed') == fit['seed']
                and row.get('epoch') == ep
                and row.get('parameter_updates') == ep * n,
                'trajectory epoch/update mismatch')
        phi = number(row.get('soft_phi'), 'traj.soft_phi')
        ods = number(row.get('soft_ODS'), 'traj.soft_ODS')
        require(0 <= phi <= 1 / 3 + 1e-12 and 0 <= ods <= 1 + 1e-12, 'traj soft bounds')
        close(phi, (1.0 - ods) / 3.0, 'traj soft_phi vs soft_ODS')
        max_age = integer(row.get('table_max_age_updates'), 'traj.max_age')
        mean_age = number(row.get('table_mean_age_updates'), 'traj.mean_age')
        disc = number(row.get('table_max_abs_discrepancy'), 'traj.disc')
        require(0 <= mean_age <= max_age <= n and 0 <= disc <= 1.0,
                'lagged count-table age/discrepancy range mismatch')
        if fit['arm'].endswith('_count'):
            require(max_age == n, 'complete once-per-epoch count-table update age required')
            close(mean_age, (n + 1) / 2, 'complete count-table mean update age')
        else:
            require(max_age == mean_age == disc == 0, 'reconstruction arm has no lagged count table')
        if ep == epochs:
            counts(row.get('byte_counts'), 'traj.final_byte_counts', pixels=n * PIXELS)
            q = row.get('quality')
            require(isinstance(q, dict) and -1 <= number(q.get('ssim'), 'traj.ssim') <= 1,
                    'traj final quality invalid')
            number(q.get('psnr'), 'traj.psnr')
    return rows


def check_train_marker_manifest(path: str | Path, *, coverage: SyntheticMultiMarkerCoverage | None = None) -> dict:
    cv = _coverage(coverage)
    synthetic = coverage is not None
    sources = Sources()
    manifest = read_json(metadata(Path(path).resolve()), sources)
    marker = manifest.get('marker')
    require(marker in ( MARKERS if synthetic else TRAINED_MARKERS ), 'unexpected marker in train manifest')
    require(manifest.get('schema') == SCHEMA
            and manifest.get('phase') == f'train_{marker}'
            and manifest.get('status') == 'complete'
            and manifest.get('checkpoint_selection') is False
            and manifest.get('method_claim') is False
            and manifest.get('actual_held_target_reads') == 0,
            'train_marker manifest header invalid')
    _check_terminal(path, manifest, sources, 0)
    members = check_membership(manifest.get('membership'), cv)
    check_membership_sources(manifest, sources, cv, synthetic)
    cal = _check_calibration(read_json(manifest['calibration'], sources), marker, cv)

    fits = manifest.get('fits')
    expected_groups = [(a, s) for a in BASE_ARMS for s in SEEDS]
    require(isinstance(fits, list) and [(f.get('arm'), f.get('seed')) for f in fits] == expected_groups,
            'exact 4 marker fits required')
    n = len(cv['training_indices'])
    steps = n * cv['epochs']
    fit_summaries = []
    for fit in fits:
        state_path = sources.add(fit['state'])
        require(state_path.name == 'final.pt.gz', 'final.pt.gz state required')
        require(fit.get('epochs') == cv['epochs'] and fit.get('parameter_updates') == steps,
                'fit epoch/step count mismatch')
        milestones_checked = _check_milestones(fit, marker, members, cv, sources, cal)
        traj = _check_trajectory(fit, marker, cv, sources)
        final_c = counts(fit.get('final_train_counts'), 'final_train_counts', pixels=n * PIXELS)
        expected_thr = endpoints(final_c, MARKER_TO_CUTOFF[marker], 0)['ODS_threshold']
        require(fit.get('train_threshold') == expected_thr, 'fit train_threshold mismatch')
        fit_summaries.append(dict(
            arm=fit['arm'],
            seed=fit['seed'],
            parameter_updates=steps,
            milestones_verified=milestones_checked,
            train_threshold=fit['train_threshold'],
            final_soft_ODS=traj[-1]['soft_ODS'],
            final_train_ODS_F=endpoints(final_c, MARKER_TO_CUTOFF[marker], expected_thr)['ODS_F'],
            wall_seconds=number(fit.get('wall_seconds'), 'wall_seconds'),
        ))
    sources.finish()
    return dict(
        schema=SCHEMA,
        checked_kind='train_marker',
        marker=marker,
        channel=MARKER_TO_CHANNEL[marker],
        cutoff=MARKER_TO_CUTOFF[marker],
        passed=True,
        checked_at=datetime.now(timezone.utc).isoformat(),
        lambda_count=cal['lambda_count'],
        donor_train_threshold=cal['donor_train_threshold'],
        fits=fit_summaries,
        stat_only_files=len(sources.rows),
    )


def _summarize_population(rows: list[dict], train_thresholds: list[int]) -> dict:
    n_img = len(rows)
    marker_Origin = {m: None for m in MARKERS}
    for row in rows:
        sc = row['scores']
        for m in MARKERS:
            c = counts(sc['counts'][m], f'row.{m}')
            if marker_Origin[m] is None:
                marker_Origin[m] = {k: (int(v) if k == 'G' else list(v)) for k, v in c.items()}
            else:
                marker_Origin[m]['G'] += c['G']
                for k in ('K', 'U', 'V', 'T'):
                    marker_Origin[m][k] = [a + b for a, b in zip(marker_Origin[m][k], c[k])]
    per_marker = {}
    for ch, (m, cutoff, thr) in enumerate(zip(MARKERS, CUTOFFS, train_thresholds)):
        ep = endpoints(marker_Origin[m], cutoff, thr)
        ep_clean = {k: ep[k] for k in (
            'ODS_F', 'ODS_threshold', 'AP', 'cutoff_tolerant_F', 'cutoff_exact_F1',
            'train_threshold', 'train_tolerant_F', 'train_exact_F1',
        )}
        ep_clean['ssim'] = math.fsum(r['scores']['quality']['ssim_channels'][ch] for r in rows) / n_img
        ep_clean['psnr'] = math.fsum(r['scores']['quality']['psnr_channels'][ch] for r in rows) / n_img
        ep_clean['counts'] = marker_Origin[m]
        per_marker[m] = ep_clean

    agg = {}
    for metric in ('ODS_F', 'AP', 'cutoff_tolerant_F', 'cutoff_exact_F1', 'train_tolerant_F', 'train_exact_F1'):
        agg[metric] = sum(per_marker[m][metric] for m in MARKERS) / 3.0
    agg['avg_ssim'] = math.fsum(r['scores']['quality']['avg_ssim'] for r in rows) / n_img
    agg['avg_psnr'] = math.fsum(r['scores']['quality']['avg_psnr'] for r in rows) / n_img
    agg['rgb_psnr'] = math.fsum(r['scores']['quality']['rgb_psnr'] for r in rows) / n_img
    return dict(metrics=agg, per_marker=per_marker, n_images=n_img)


def _paired_bootstrap_quality(rows_a_by_seed: dict, rows_b_by_seed: dict,
                              metric_fn, n_boot: int = 2000, seed: int = 20261111) -> dict:
    import numpy as np
    n_img = len(next(iter(rows_a_by_seed.values())))
    deltas = []
    for idx in range(n_img):
        va = sum(metric_fn(rows_a_by_seed[s][idx]) for s in SEEDS) / len(SEEDS)
        vb = sum(metric_fn(rows_b_by_seed[s][idx]) for s in SEEDS) / len(SEEDS)
        deltas.append(va - vb)
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_img, size=(n_boot, n_img))
    means = arr[boot_idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(
        mean_delta=float(arr.mean()),
        ci95=[float(lo), float(hi)],
        positive_image_fraction=float((arr > 0).mean()),
    )


def _paired_bootstrap_pooled_ods(rows_a_by_seed: dict, rows_b_by_seed: dict,
                                 n_boot: int = 500, seed: int = 20261111) -> dict:
    """Image-cluster bootstrap for pooled 3-marker seed-mean ODS-F difference."""
    import numpy as np
    n_img = len(next(iter(rows_a_by_seed.values())))

    def pack_curves(rows_by_seed):
        packed = {}
        for s, rlist in rows_by_seed.items():
            g_arr = np.zeros((n_img, 3), dtype=np.float64)
            kuv_arr = np.zeros((n_img, 3, 256, 3), dtype=np.float64)
            for i, r in enumerate(rlist):
                for ch, m in enumerate(MARKERS):
                    c = r['scores']['counts'][m]
                    g_arr[i, ch] = c['G']
                    kuv_arr[i, ch, :, 0] = c['K']
                    kuv_arr[i, ch, :, 1] = c['U']
                    kuv_arr[i, ch, :, 2] = c['V']
            packed[s] = (kuv_arr, g_arr)
        return packed

    pa = pack_curves(rows_a_by_seed)
    pb = pack_curves(rows_b_by_seed)

    def pooled_ods(packed, indices):
        vals = []
        for s in SEEDS:
            kuv, g = packed[s]
            ksum = kuv[indices].sum(axis=0)  # (3, 256, 3)
            gsum = g[indices].sum(axis=0)    # (3,)
            K, U, V = ksum[:, :, 0], ksum[:, :, 1], ksum[:, :, 2]
            G = gsum[:, None]
            den = U * G + V * K
            f = np.where(den > 0, 2.0 * U * V / np.where(den > 0, den, 1.0), 0.0)
            vals.append(float(f.max(axis=1).mean()))
        return sum(vals) / len(vals)

    full_idx = np.arange(n_img)
    point = pooled_ods(pa, full_idx) - pooled_ods(pb, full_idx)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        b_idx = rng.integers(0, n_img, size=n_img)
        boots.append(pooled_ods(pa, b_idx) - pooled_ods(pb, b_idx))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(delta=float(point), ci95=[float(lo), float(hi)], resamples=n_boot)


def _resolve_evaluate_membership_sources(manifest: dict, members: dict, sources: Sources,
                                         cv: dict, synthetic: bool) -> None:
    if synthetic:
        return
    if isinstance(manifest.get('membership_sources'), dict):
        check_membership_sources(manifest, sources, cv, synthetic)
        return
    t_refs = manifest.get('training_manifests')
    require(isinstance(t_refs, dict) and {'panCK', 'CD3', 'DAPI', 'cache'} <= set(t_refs),
            'evaluate training_manifests dict required')
    panck_m = read_json(t_refs['panCK'], sources)
    cd3_m = read_json(t_refs['CD3'], sources)
    dapi_m = read_json(t_refs['DAPI'], sources)
    require(panck_m.get('membership') == members
            and cd3_m.get('membership') == members
            and dapi_m.get('membership') == members,
            'evaluate and training_manifests membership mismatch')
    require(isinstance(cd3_m.get('membership_sources'), dict)
            and cd3_m.get('membership_sources') == dapi_m.get('membership_sources'),
            'matching CD3/DAPI role-bound membership_sources required')
    check_membership_sources(
        dict(membership=members, membership_sources=cd3_m['membership_sources']),
        sources, cv, synthetic,
    )


def check_evaluate_manifest(path: str | Path, *,
                            coverage: SyntheticMultiMarkerCoverage | None = None,
                            paired_rows_out: list[dict] | None = None) -> dict:
    cv = _coverage(coverage)
    synthetic = coverage is not None
    sources = Sources()
    manifest = read_json(metadata(Path(path).resolve()), sources)
    require(manifest.get('schema') == SCHEMA
            and manifest.get('phase') == 'evaluate'
            and manifest.get('status') == 'complete'
            and manifest.get('locked_test_access') is False
            and manifest.get('method_claim') is False,
            'evaluate manifest header invalid')
    expected_total = cv['hold'] + cv['val']
    n_composites = len(EVAL_ARMS) * len(SEEDS)
    _check_terminal(path, manifest, sources, expected_total)
    members = check_membership(manifest.get('membership'), cv)
    check_source_coverage(manifest, sources, members, 'evaluate', synthetic)
    _resolve_evaluate_membership_sources(manifest, members, sources, cv, synthetic)
    check_quality_source(manifest.get('quality_metadata'), sources)

    for single in manifest.get('single_channel_models', []):
        sources.add(single['state'])

    model_specs = manifest.get('models', [])
    expected_pairs = [(a, s) for a in EVAL_ARMS for s in SEEDS]
    require([(m['arm'], m['seed']) for m in model_specs] == expected_pairs,
            'exact 12 composite model specs required')
    thr_map = {(m['arm'], m['seed']): thresholds(m['train_thresholds'], m['arm']) for m in model_specs}

    # Verify strict generated -> target_read alternation in access.jsonl.gz
    access_rows = list(iter_rows(manifest['access'], sources))
    require(len(access_rows) == 2 * expected_total, 'exact 2*N access events required')
    expected_identities = (
        [('hold', name) for name in members['hold']]
        + [('val', name) for name in members['val']]
    )
    for i, (sp, nm) in enumerate(expected_identities):
        gen_ev, read_ev = access_rows[2 * i], access_rows[2 * i + 1]
        require(gen_ev.get('event') == 'generated'
                and gen_ev.get('outputs') == n_composites
                and gen_ev.get('index') == i
                and gen_ev.get('split') == sp
                and gen_ev.get('name') == nm,
                'access generated event mismatch')
        require(read_ev.get('event') == 'target_read'
                and read_ev.get('index') == i
                and read_ev.get('split') == sp
                and read_ev.get('name') == nm,
                'access target_read event mismatch')

    donor_rows = list(iter_rows(manifest['donors'], sources))
    require(len(donor_rows) == expected_total, 'donor row count mismatch')
    for r in donor_rows:
        q = quality(r['scores']['quality'], 'donor.quality')
        number(r['scores']['quality'].get('rgb_psnr'), 'donor.rgb_psnr')

    pred_rows = list(iter_rows(manifest['predictions'], sources))
    require(len(pred_rows) == expected_total * n_composites, 'prediction row count mismatch')

    by_split_arm_seed = {sp: {(a, s): [] for a, s in expected_pairs} for sp in ('hold', 'val')}
    donor_by_split = {'hold': [], 'val': []}
    for d_row in donor_rows:
        donor_by_split[d_row['split']].append(d_row)
    for p_row in pred_rows:
        quality(p_row['scores']['quality'], 'pred.quality')
        number(p_row['scores']['quality'].get('rgb_psnr'), 'pred.rgb_psnr')
        by_split_arm_seed[p_row['split']][(p_row['arm'], p_row['seed'])].append(p_row)

    donor_thresholds = manifest['models'][expected_pairs.index(('S_count_panck_only', SEEDS[0]))]['train_thresholds']
    # Donor thresholds for panCK come from cache manifest:
    cache_manifest = read_json(manifest['training_manifests']['cache'], sources)
    donor_thr_vec = thresholds(cache_manifest['donor_train_thresholds'], 'donor_thresholds')

    populations = {}
    for sp in ('hold', 'val'):
        d_sum = _summarize_population(donor_by_split[sp], donor_thr_vec)
        per_seed = {}
        seed_means = {}
        seed_per_marker = {}
        for arm in EVAL_ARMS:
            s_dicts = {}
            for s in SEEDS:
                s_dicts[str(s)] = _summarize_population(by_split_arm_seed[sp][(arm, s)], thr_map[(arm, s)])
            per_seed[arm] = s_dicts
            seed_means[arm] = {
                m_key: sum(s_dicts[str(s)]['metrics'][m_key] for s in SEEDS) / len(SEEDS)
                for m_key in NEW_METRICS
            }
            seed_per_marker[arm] = {
                mk: {
                    k: sum(s_dicts[str(s)]['per_marker'][mk][k] for s in SEEDS) / len(SEEDS)
                    for k in ('ODS_F', 'AP', 'cutoff_tolerant_F', 'cutoff_exact_F1',
                              'train_tolerant_F', 'train_exact_F1', 'ssim', 'psnr')
                }
                for mk in MARKERS
            }
        comps = {}
        for a, b in COMPARISONS:
            key = f'{a}_minus_{b}'
            ra = {s: by_split_arm_seed[sp][(a, s)] for s in SEEDS}
            rb = {s: by_split_arm_seed[sp][(b, s)] for s in SEEDS}
            comps[key] = dict(
                point_deltas={
                    m_key: seed_means[a][m_key] - seed_means[b][m_key]
                    for m_key in NEW_METRICS
                },
                pooled_ODS_F_bootstrap=_paired_bootstrap_pooled_ods(ra, rb),
                avg_ssim_bootstrap=_paired_bootstrap_quality(
                    ra, rb, lambda r: r['scores']['quality']['avg_ssim']),
                avg_psnr_bootstrap=_paired_bootstrap_quality(
                    ra, rb, lambda r: r['scores']['quality']['avg_psnr']),
            )
        # Also compare S_count_3ch against donor D
        ra_3ch = {s: by_split_arm_seed[sp][('S_count_3ch', s)] for s in SEEDS}
        rb_donor = {s: donor_by_split[sp] for s in SEEDS}
        comps['S_count_3ch_minus_donor'] = dict(
            point_deltas={
                m_key: seed_means['S_count_3ch'][m_key] - d_sum['metrics'][m_key]
                for m_key in NEW_METRICS
            },
            pooled_ODS_F_bootstrap=_paired_bootstrap_pooled_ods(ra_3ch, rb_donor),
            avg_ssim_bootstrap=_paired_bootstrap_quality(
                ra_3ch, rb_donor, lambda r: r['scores']['quality']['avg_ssim']),
            avg_psnr_bootstrap=_paired_bootstrap_quality(
                ra_3ch, rb_donor, lambda r: r['scores']['quality']['avg_psnr']),
        )
        populations[sp] = dict(
            n_images=len(donor_by_split[sp]),
            donor=d_sum,
            per_seed=per_seed,
            seed_means=seed_means,
            seed_per_marker=seed_per_marker,
            comparisons=comps,
        )

    parity_check = None
    if manifest.get('panck_fit1192_summary') is not None:
        old_sum = read_json(manifest['panck_fit1192_summary'], sources)
        parity_check = {}
        group_lookup = {}
        if isinstance(old_sum.get('groups'), list):
            for g in old_sum['groups']:
                group_lookup[(g.get('split'), g.get('arm'), g.get('seed'))] = g
        for sp in ('hold', 'val'):
            parity_check[sp] = {}
            for single_arm, ref_arm in (('S_count_panck_only', 'S_count'),
                                        ('S_recon_panck_only', 'S_recon')):
                actual = populations[sp]['seed_means'][single_arm]
                if group_lookup:
                    seed_groups = [group_lookup[(sp, ref_arm, s)] for s in SEEDS]
                    expected = {
                        m_key: sum(float(g[m_key]) for g in seed_groups) / len(SEEDS)
                        for m_key in NEW_METRICS
                    }
                else:
                    expected = old_sum['populations'][sp]['seed_means'][ref_arm]
                for m_key in NEW_METRICS:
                    tol = 2e-5 if m_key == 'rgb_psnr' else 1e-12
                    require(math.isclose(actual[m_key], expected[m_key], rel_tol=0.0, abs_tol=tol),
                            f'FIT1192-v2 parity failed for {sp} {single_arm} {m_key}: {actual[m_key]} vs {expected[m_key]}')
                parity_check[sp][single_arm] = 'exact_match_verified'

    if paired_rows_out is not None:
        for sp in ('hold', 'val'):
            for i, d_row in enumerate(donor_by_split[sp]):
                dq = d_row['scores']['quality']
                arms_summary = {}
                for arm in EVAL_ARMS:
                    seed_entries = {}
                    for s in SEEDS:
                        pr = by_split_arm_seed[sp][(arm, s)][i]
                        require(pr['name'] == d_row['name'] and pr['index'] == d_row['index'],
                                'per-image arm/donor order mismatch')
                        pq = pr['scores']['quality']
                        seed_entries[str(s)] = dict(
                            avg_ssim=pq['avg_ssim'],
                            avg_psnr=pq['avg_psnr'],
                            ssim_channels=pq['ssim_channels'],
                            psnr_channels=pq['psnr_channels'],
                            changed_pixels=pr['scores']['changed_pixels'],
                            changed_values_by_marker=pr['scores']['changed_values_by_marker'],
                        )
                    arms_summary[arm] = seed_entries
                paired_rows_out.append(dict(
                    index=d_row['index'],
                    split=sp,
                    name=d_row['name'],
                    pixels=d_row['scores']['pixels'],
                    donor_quality=dict(
                        avg_ssim=dq['avg_ssim'],
                        avg_psnr=dq['avg_psnr'],
                        rgb_psnr=dq['rgb_psnr'],
                        ssim_channels=dq['ssim_channels'],
                        psnr_channels=dq['psnr_channels'],
                    ),
                    arms=arms_summary,
                ))

    val_3ch = populations['val']['seed_means']['S_count_3ch']
    hold_3ch = populations['hold']['seed_means']['S_count_3ch']
    val_donor = populations['val']['donor']['metrics']
    hold_donor = populations['hold']['donor']['metrics']
    sota_gate = dict(
        val_ODS_F=val_3ch['ODS_F'],
        clears_val_sota_floor_0812652=bool(val_3ch['ODS_F'] > CHARTER_FLOORS['val_ods_f_sota_floor']),
        clears_val_ssim_floor_0710816=bool(val_3ch['avg_ssim'] >= CHARTER_FLOORS['val_ssim_floor']),
        clears_val_psnr_floor_26011020=bool(val_3ch['avg_psnr'] >= CHARTER_FLOORS['val_psnr_floor']),
        beats_val_3ch_recon=bool(val_3ch['ODS_F'] > populations['val']['seed_means']['S_recon_3ch']['ODS_F']),
        beats_hold_donor=bool(hold_3ch['ODS_F'] > hold_donor['ODS_F']),
        beats_val_donor=bool(val_3ch['ODS_F'] > val_donor['ODS_F']),
        charter_floors=CHARTER_FLOORS,
    )
    sources.finish()
    return dict(
        schema=SCHEMA,
        checked_kind='evaluate',
        passed=True,
        checked_at=datetime.now(timezone.utc).isoformat(),
        run_id=manifest['run_id'],
        actual_target_reads=expected_total,
        predictions=expected_total * n_composites,
        donors=expected_total,
        groups=n_composites,
        locked_test_access=False,
        fit1192_v2_panck_parity=parity_check,
        sota_gate_summary=sota_gate,
        populations=populations,
        stat_only_files=len(sources.rows),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('train_marker', 'evaluate'), required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--accepted-output', default=None)
    parser.add_argument('--paired-output', default=None)
    args = parser.parse_args(argv)
    out = Path(args.output).resolve()
    require(not out.exists(), 'preserve existing checker output: ' + str(out))
    acc_out = Path(args.accepted_output).resolve() if args.accepted_output else None
    pair_out = Path(args.paired_output).resolve() if args.paired_output else None
    if acc_out is not None:
        require(not acc_out.exists(), 'preserve existing accepted output: ' + str(acc_out))
    if pair_out is not None:
        require(not pair_out.exists(), 'preserve existing paired output: ' + str(pair_out))

    paired_rows = [] if (args.kind == 'evaluate' and pair_out is not None) else None
    result = (check_train_marker_manifest(args.manifest)
              if args.kind == 'train_marker'
              else check_evaluate_manifest(args.manifest, paired_rows_out=paired_rows))

    torch_imported = int('torch' in sys.modules)
    require(torch_imported == 0 and 'cv2' not in sys.modules and 'PIL' not in sys.modules,
            'pure scalar consumer must never import torch/cv2/PIL')

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with out.open('xb') as handle:
        handle.write(payload)

    if pair_out is not None and paired_rows is not None:
        pair_out.parent.mkdir(parents=True, exist_ok=True)
        with pair_out.open('xb') as raw_fh:
            with gzip.GzipFile(fileobj=raw_fh, mode='wb', mtime=0) as gz_fh:
                for row in paired_rows:
                    gz_fh.write((json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode())

    if acc_out is not None and args.kind == 'evaluate':
        acc_out.parent.mkdir(parents=True, exist_ok=True)
        accepted_payload = dict(
            schema=SCHEMA,
            status='passed_saved_scalar_consistency',
            checked_at=result['checked_at'],
            run_id=result['run_id'],
            predictions=result['predictions'],
            donors=result['donors'],
            groups=result['groups'],
            target_reads=result['actual_target_reads'],
            locked_test_access=result['locked_test_access'],
            fit1192_v2_panck_parity=result['fit1192_v2_panck_parity'],
            sota_gate_summary=result['sota_gate_summary'],
            source_files_stat_checked=result['stat_only_files'],
            consumer=dict(
                torch_imports=0,
                array_loads=0,
                image_reads=0,
                target_reads=0,
                RGB_rescores=0,
                state_loads=0,
                new_fits=0,
            ),
        )
        with acc_out.open('xb') as handle:
            handle.write((json.dumps(accepted_payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())

    print(json.dumps(dict(event='independent_check_passed', kind=args.kind, output=str(out))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
