"""Train-only, streaming pixel comparator; no model/data loading or validation selection.

Callback contract: stream_factory() yields (name, {method: HWC float32 intensities},
HWC uint8 label), in any order, exactly once per selected train name each pass.
Prediction intensities come from normalized output via ``intensities``; do not
quantize individual models before blending. Two passes are required. A remote
runner may stream existing prediction files or rerun frozen GPU prediction.
``scorer`` returns counts (3,4) in panCK/CD3/DAPI order, columns n_p,tp_p,n_g,tp_g,
and avg_ssim/avg_psnr using the unchanged common evaluator. Callbacks must not
load validation/test images. The selected recipe can subsequently be applied
once to validation with the existing full-curve scorer, outside this selector.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

MARKERS = ('panCK', 'CD3', 'DAPI')
METHOD_ORDER = ('direct', 'FM', 'UNI2IF')
ALPHAS = (0., .25, .5, .75, 1.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def intensities(normalized_hwc):
    """Float32 intensity conversion; defer clipping until after blend/affine."""
    x = np.asarray(normalized_hwc, dtype=np.float32)
    return ((x + np.float32(1)) / np.float32(2)) * np.float32(255)


def quantize(x):
    return np.rint(np.clip(np.asarray(x, dtype=np.float32), 0, 255)).astype(np.uint8)


@dataclass(frozen=True)
class Family:
    methods: tuple
    alpha: float = 1.

    def pixels(self, predictions):
        x = np.asarray(predictions[self.methods[0]], dtype=np.float32)
        if len(self.methods) == 1:
            return x
        return (np.float32(self.alpha) * x + np.float32(1 - self.alpha)
                * np.asarray(predictions[self.methods[1]], dtype=np.float32))


def families(methods):
    methods = tuple(methods)
    require(methods == tuple(m for m in METHOD_ORDER if m in methods),
            'methods must be unique in direct, FM, UNI2IF order')
    require('direct' in methods and 'FM' in methods, 'direct and FM are required')
    # Alpha endpoints are exactly the existing single-model families, not duplicates.
    return ([Family((m,)) for m in methods] +
            [Family(pair, a) for pair in combinations(methods, 2) for a in ALPHAS[1:-1]])


class Moments:
    """Channelwise sufficient statistics; constant memory, all pixels equally weighted."""
    def __init__(self):
        self.n = 0
        self.sums = np.zeros((4, 3), dtype=np.float64)

    def add(self, x, y):
        x, y = np.asarray(x, np.float64).reshape(-1, 3), np.asarray(y, np.float64).reshape(-1, 3)
        require(x.shape == y.shape and len(x) > 0, 'invalid moment shapes')
        self.n += len(x)
        self.sums += np.stack((x.sum(0), y.sum(0), (x*x).sum(0), (x*y).sum(0)))

    def affine(self):
        require(self.n > 0, 'empty affine fit')
        sx, sy, sxx, sxy = self.sums / self.n
        variance = np.maximum(0., sxx - sx*sx)
        slope = np.ones(3)
        np.divide(sxy - sx*sy, variance, out=slope, where=variance > 0)
        slope = np.clip(slope, .5, 2.)
        intercept = np.clip(sy - slope*sx, -32., 32.)
        return slope, intercept

    def plus(self, other):
        result = Moments()
        result.n, result.sums = self.n + other.n, self.sums + other.sums
        return result


def apply(predictions, family, affine=None):
    x = family.pixels(predictions)
    if affine is not None:
        slope, intercept = affine
        x = x * np.asarray(slope, np.float32) + np.asarray(intercept, np.float32)
    return quantize(x)


def population(train_names, selected_names, forbidden_names=(), expected_n=300):
    require(type(expected_n) is int and expected_n >= 2, 'need at least two training images')
    train, selected, forbidden = list(train_names), list(selected_names), set(forbidden_names)
    require(train == sorted(set(train)), 'train population must be sorted and unique')
    require(not set(train) & forbidden, 'train overlaps validation/test membership')
    expected = train[::max(1, len(train)//expected_n)][:expected_n]
    require(len(selected) == expected_n and selected == expected,
            'selected population missing, duplicate, reordered or not selected_train')
    return {name: i % 2 for i, name in enumerate(selected)}


def checked_stream(factory, folds, methods):
    seen = set()
    for name, predictions, target in factory():
        require(name in folds, 'non-training or unexpected image in stream')
        require(name not in seen, 'duplicate image in stream')
        seen.add(name)
        target = np.asarray(target)
        require(target.dtype == np.uint8 and target.ndim == 3 and target.shape[-1] == 3,
                'label must be HWC uint8 with three unchanged channels')
        require(set(predictions) == set(methods), 'missing or extra prediction method')
        for x in predictions.values():
            x = np.asarray(x)
            require(x.shape == target.shape and x.size > 0 and np.isfinite(x).all(),
                    'invalid prediction pixels')
        yield name, predictions, target
    require(seen == set(folds), 'missing image population in stream')


def pooled_fixed_f1(counts):
    counts = np.asarray(counts, dtype=np.float64)
    require(counts.shape == (3, 4) and np.isfinite(counts).all() and (counts >= 0).all()
            and (counts == np.floor(counts)).all(), 'invalid tolerant counts')
    n_p, tp_p, n_g, tp_g = counts.T
    require((tp_p <= n_p).all() and (tp_g <= n_g).all(), 'invalid matched counts')
    precision = np.divide(tp_p, n_p, out=np.zeros(3), where=n_p > 0)
    recall = np.divide(tp_g, n_g, out=np.zeros(3), where=n_g > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=precision+recall > 0)
    f1[(n_p == 0) & (n_g == 0)] = 1.
    return float(f1.mean())


class NoEligibleCandidate(ValueError):
    """The completed OOF comparison has no guard-passing candidate."""


def choose(rows):
    eligible = [r for r in rows if r['avg_ssim'] >= .710816 and r['avg_psnr'] >= 26.011020]
    if not eligible:
        raise NoEligibleCandidate('no OOF candidate passes both image guards')
    best = max(r['fixed_macro_f1'] for r in eligible)
    tied = [r for r in eligible if best - r['fixed_macro_f1'] <= 1e-12]
    return min(tied, key=lambda r: (len(r['family'].methods), r['transformed'],
                                   ALPHAS.index(r['family'].alpha), r['family_index']))


def fit_select(stream_factory, scorer, train_names, selected_names, methods,
               forbidden_names=(), expected_n=300):
    """Fit on opposite folds, select on pooled OOF fixed F1, refit winner on all train.

    expected_n is adjustable only for synthetic tests; production uses 300.
    Missing UNI2IF is supported and recorded; it does not imply a paired UNI result.
    No validation predictions/scores are accepted by this API.
    """
    folds = population(train_names, selected_names, forbidden_names, expected_n)
    fs = families(methods)
    stats = [[Moments(), Moments()] for _ in fs]
    for name, predictions, gt in checked_stream(stream_factory, folds, methods):
        for f, moments in zip(fs, stats):
            moments[folds[name]].add(f.pixels(predictions), gt)
    affines = [[m[1].affine(), m[0].affine()] for m in stats]
    rows = [dict(family=f, family_index=i, transformed=t, counts=np.zeros((3, 4), np.int64),
                 ssim_sum=0., psnr_sum=0.) for i, f in enumerate(fs) for t in (False, True)]
    for name, predictions, gt in checked_stream(stream_factory, folds, methods):
        for row in rows:
            affine = affines[row['family_index']][folds[name]] if row['transformed'] else None
            score = scorer(apply(predictions, row['family'], affine), gt)
            counts = np.asarray(score['counts'])
            pooled_fixed_f1(counts)  # validate integer counts before accumulation
            require(np.isfinite(score['avg_ssim']) and np.isfinite(score['avg_psnr']), 'nonfinite image guards')
            row['counts'] += counts.astype(np.int64)
            row['ssim_sum'] += float(score['avg_ssim'])
            row['psnr_sum'] += float(score['avg_psnr'])
    for row in rows:
        row['fixed_macro_f1'] = pooled_fixed_f1(row['counts'])
        row['avg_ssim'] = row.pop('ssim_sum') / len(folds)
        row['avg_psnr'] = row.pop('psnr_sum') / len(folds)
    try:
        winner = choose(rows)
    except NoEligibleCandidate:
        winner = None
    affine = None
    if winner is not None and winner['transformed']:
        moments = stats[winner['family_index']]
        affine = moments[0].plus(moments[1]).affine()
    return dict(status='selected' if winner is not None else 'no_eligible_candidate',
                selected=winner, affine=affine,
                oof_rows=rows, fold_affines=affines, selected_names=list(selected_names),
                methods=list(methods), uni2if_available='UNI2IF' in methods,
                paired_uni2if_claim=False, channel_order=MARKERS,
                constant_predictor_slope=1., selection_split='train_oof')


def common_scorer(metric_panel):
    """Inject the deployed metric_panel module (and thus its existing SSIM/backend)."""
    require(tuple(metric_panel.TRUE) == MARKERS, 'common marker order differs')
    require(metric_panel.FIXED_THR == dict(panCK=22., CD3=30., DAPI=18.)
            and metric_panel.TOL == 5, 'common fixed protocol differs')
    def score(pred, gt):
        row = metric_panel.score(pred, gt, metric_panel.FIXED_THR)
        return dict(counts=np.asarray([[row[f'{m}_{k}'] for k in ('np','tpp','ng','tpg')]
                                       for m in MARKERS], dtype=np.int64),
                    avg_ssim=float(np.mean([row[f'{m}_ssim'] for m in MARKERS])),
                    avg_psnr=float(np.mean([row[f'{m}_psnr'] for m in MARKERS])))
    return score
