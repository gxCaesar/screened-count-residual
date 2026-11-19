"""Fixed three-fold residual ensemble and compact, integer-first reductions.

No model fitting, source discovery or file access lives in this module.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from . import query_relation_scoring as scoring
from . import query_relation_data as data


SCHEMA = 'query-relation-ensemble930-v1'
ARMS = ('E_C', 'S_C', 'P_C')
SEEDS = (202610270, 202610271)
FOLDS = (1, 2, 3)
GROUPS = tuple((a, s) for a in ARMS for s in SEEDS)
MARKERS = scoring.MARKERS
METRICS = ('ODS_F', 'FITtau_tolerant_F', 'cutoff_tolerant_F',
           'FITtau_exact_F1', 'cutoff_exact_F1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_models(manifest):
    """Select exactly the18 repaired inner states, with their own statistics."""
    require(manifest.get('schema') == 'query-relation-optimization-v1'
            and manifest.get('status') == 'complete_training_only_32_epoch_diagnostic'
            and manifest.get('objective_version') == 'preclip_mse_v1',
            'accepted repaired training manifest required')
    expected = {(a, s, f) for a, s in GROUPS for f in FOLDS}
    selected = {}
    for row in manifest.get('fits', []):
        key = row.get('arm'), row.get('seed'), row.get('fold')
        if key not in expected:
            continue
        a, s, f = key
        require(key not in selected, 'duplicate selected model identity')
        training = [i for i in data.TRAINING_INDICES if i % 4 != f]
        held = [i for i in data.TRAINING_INDICES if i % 4 == f]
        require(row.get('split') == f'inner{f}' and row.get('epochs') == 32
                and row.get('training_indices') == training and row.get('held_indices') == held
                and row.get('objective_version') == 'preclip_mse_v1'
                and row.get('training_mse_uses_clamp') is False,
                'wrong fold, membership, epoch or repaired objective')
        require(row.get('state') == row.get('state_metadata') == row.get('epoch032')
                and isinstance(row.get('original_statistics'), dict),
                'explicit epoch032 state/statistics binding required')
        selected[key] = dict(arm=a, seed=s, fold=f, epoch=32,
            training_indices=training, held_indices=held,
            state=copy.deepcopy(row['state']), statistics=copy.deepcopy(row['original_statistics']))
    require(set(selected) == expected, 'all18 distinct inner states required')
    require(len({r['state']['path'] for r in selected.values()}) == 18,
            'all18 distinct physical state paths required')
    return [selected[a, s, f] for a, s in GROUPS for f in FOLDS]


def validate_sources(sources, full_manifest, original_fit_names):
    """Membership validation is metadata-only and prohibits every FIT32 name."""
    order = full_manifest.get('ordered_items')
    populations = full_manifest.get('metadata_contract', {}).get('populations', {})
    require(isinstance(order, list) and len(order) == 930
            and set(populations) == {'hold', 'val'}
            and len(populations['hold']) == 300 and len(populations['val']) == 630,
            'bound HOLD300/val630 full manifest required')
    require(isinstance(original_fit_names, list) and len(original_fit_names) == 32
            and len(set(original_fit_names)) == 32
            and original_fit_names == full_manifest['metadata_contract'].get('accepted_FIT32'),
            'all32 original FIT names required')
    identities = [(r.get('split'), r.get('name')) for r in sources]
    require(identities == [tuple(x) for x in order] and len(set(identities)) == 930
            and len({n for _, n in identities}) == 930,
            'exact ordered930 distinct source identities required')
    require(not set(original_fit_names) & {n for _, n in identities},
            'forbidden original FIT32/outer8 membership')
    require({s: [n for t, n in identities if t == s] for s in ('hold', 'val')} == populations,
            'full population membership differs')
    for row in sources:
        require(set(row) == {'split', 'name', 'frame', 'target', 'crop'}
                and row['split'] in ('hold', 'val')
                and row['crop'] == [256, 256, 768, 768],
                'native source metadata and center512 crop required')
    require(len({r['frame']['path'] for r in sources}) == 930
            and len({r['target']['path'] for r in sources}) == 930
            and not {r['frame']['path'] for r in sources} & {r['target']['path'] for r in sources},
            'distinct target-free frame and target paths required')
    return identities


def mean_residuals(residuals):
    require(len(residuals) == 3, 'exactly three fold residuals required')
    arrays = [np.asarray(value) for value in residuals]
    require(all(a.shape == (512, 512) and a.dtype == np.float64
                and np.isfinite(a).all() for a in arrays),
            'native512 finite float64 fold residuals required')
    # Never clip/round individual folds. This order is the fixed inference rule.
    return (arrays[0] + arrays[1] + arrays[2]) / np.float64(3)


def final_bytes(donor, residuals):
    require(isinstance(donor, np.ndarray) and donor.dtype == np.uint8
            and donor.shape == (512, 512, 3), 'native RGB donor required')
    residual = mean_residuals(residuals)
    z = donor[..., 0].astype(np.float64)/255 + residual
    clipped = np.clip(z, 0., 1.)
    image = donor.copy()
    image[..., 0] = np.rint(255*clipped).astype(np.uint8)
    require(np.array_equal(image[..., 1:], donor[..., 1:]), 'CD3/DAPI channel order changed')
    return image, dict(pixels=512*512, below_zero=int((z < 0).sum()),
        above_one=int((z > 1).sum()), preclip_mean=float(z.mean()),
        postclip_mean=float(clipped.mean()),
        rounding_mean_change=float(image[..., 0].mean()/255-clipped.mean()))


def marker_counts(image, target, channel):
    raw = scoring.reference.counts(image[..., channel],
        target[..., channel] > scoring.CUTOFFS[channel], scoring.tolerance_kernel())
    value = {k: int(v) if k == 'G' else v.tolist() for k, v in raw.items()}
    return scoring.validate_counts(value, MARKERS[channel], pixels=512*512)


def identity(split, name):
    return dict(schema=SCHEMA, split=split, name=name, marker_order=list(MARKERS),
                shape=[512, 512, 3], pixels=512*512)


def score_donor(donor, target, quality, *, split, name):
    for value in (donor, target):
        require(value.shape == (512, 512, 3) and value.dtype == np.uint8,
                'strict native512 RGB bytes required')
    measured = quality(donor, target)
    q = scoring._quality(measured, 'donor quality')
    if 'rgb_psnr' in measured:
        q['rgb_psnr'] = scoring._number(measured['rgb_psnr'], 'native RGB PSNR')
    return dict(identity(split, name), counts={m: marker_counts(donor, target, c)
        for c, m in enumerate(MARKERS)}, quality=q,
        output_sum=donor.astype(np.int64).sum(axis=(0, 1)).tolist(),
        output_target_SSE=np.square(donor.astype(np.int64)-target).sum(axis=(0, 1)).tolist())


def score_prediction(image, target, donor, quality, *, split, name, arm, seed, clipping, solver):
    require((arm, seed) in GROUPS and np.array_equal(image[..., 1:], donor[..., 1:]),
            'fixed arm/seed and unchanged canonical CD3/DAPI required')
    require(image.shape == target.shape == donor.shape == (512, 512, 3)
            and all(v.dtype == np.uint8 for v in (image, target, donor)), 'strict RGB bytes required')
    x, y, d = (a.astype(np.int64) for a in (image, target, donor))
    edit, error = x-d, x-y
    moments = dict(output_target_SSE=np.square(error).sum(axis=(0, 1)).tolist(),
        donor_target_SSE=np.square(d-y).sum(axis=(0, 1)).tolist(),
        edit_SSE=np.square(edit).sum(axis=(0, 1)).tolist(),
        edit_donor_error_cross=(edit*(d-y)).sum(axis=(0, 1)).tolist())
    measured = quality(image, target)
    q = scoring._quality(measured, 'prediction quality')
    if 'rgb_psnr' in measured:
        q['rgb_psnr'] = scoring._number(measured['rgb_psnr'], 'native RGB PSNR')
    return dict(identity(split, name), arm=arm, seed=seed, counts=marker_counts(image, target, 0),
        quality=q, moments=moments,
        output_sum=x.sum(axis=(0, 1)).tolist(), panCK_output_sum=int(x[..., 0].sum()),
        D=moments['output_target_SSE'][0]-moments['donor_target_SSE'][0], C=moments['edit_SSE'][0],
        changed_pixels=int(np.count_nonzero(np.any(edit, axis=2))),
        changed_values_by_marker=np.count_nonzero(edit, axis=(0, 1)).tolist(),
        clipping=clipping, solver=solver)


class Pool:
    """Hold only pooled counts and scalar sums, never per-image arrays/rows."""
    def __init__(self):
        self.groups = {}

    def add(self, row, donor=None):
        key = row['split'], row.get('arm', 'donor'), row.get('seed')
        if key not in self.groups:
            self.groups[key] = dict(images=0, counts={m: dict(G=0, **{k: np.zeros(256, np.int64)
                for k in scoring.COUNT_FIELDS}) for m in MARKERS},
                quality={k: np.zeros(3, np.float64) for k in ('ssim_channels', 'psnr_channels')},
                D=0, C=0, changed_pixels=0)
        group = self.groups[key]
        group['images'] += 1
        for c, m in enumerate(MARKERS):
            counts = row['counts'][m] if donor is None else (row['counts'] if c == 0 else donor['counts'][m])
            scoring.validate_counts(counts, pixels=row['pixels'])
            group['counts'][m]['G'] += counts['G']
            for k in scoring.COUNT_FIELDS:
                group['counts'][m][k] += counts[k]
        for k in group['quality']:
            group['quality'][k] += row['quality'][k]
        if donor is not None:
            for k in ('D', 'C', 'changed_pixels'):
                group[k] += row[k]

    def report(self):
        rows = []
        for (split, arm, seed), g in self.groups.items():
            n = g['images']
            marker_rows = []
            for m, cutoff in zip(MARKERS, scoring.CUTOFFS):
                counts = {k: int(v) if k == 'G' else v.tolist() for k, v in g['counts'][m].items()}
                ep = scoring.endpoints(counts, cutoff)
                marker_rows.append(dict(marker=m, **{k: ep[k] for k in (*METRICS, 'ODS_threshold')}))
            q = {k: (v/n).tolist() for k, v in g['quality'].items()}
            row = dict(split=split, arm=arm, seed=seed, images=n, marker_rows=marker_rows,
                **{k: sum(m[k] for m in marker_rows)/3 for k in METRICS},
                **q, avg_ssim=sum(q['ssim_channels'])/3, avg_psnr=sum(q['psnr_channels'])/3)
            if arm != 'donor':
                row.update({k: g[k] for k in ('D', 'C', 'changed_pixels')})
            require(all(math.isfinite(row[k]) for k in (*METRICS, 'avg_ssim', 'avg_psnr')),
                    'nonfinite pooled scalar')
            rows.append(row)
        return dict(groups=[r for r in rows if r['arm'] != 'donor'],
                    donors=[r for r in rows if r['arm'] == 'donor'])
