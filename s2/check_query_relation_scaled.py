"""Independently accept FIT1192 saved scalar evidence without opening pixels/states.

This consumer reconstructs threshold arithmetic and comparisons; it never imports
or calls the training runner. Source files, images and opaque states are stat-only.
A successful check establishes saved-scalar consistency, not a scientific win or
an independent measurement of truth.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import zlib

from .check_query_relation_ensemble import (
    Sources, close, counts as _validated_counts, decode, integer, metadata, number, pooled,
    quality, read_json, require, check_quality_source,
)

SCHEMA = 'query-relation-fit1192-v1'
ARMS = ('E_recon', 'S_recon', 'E_count', 'S_count')
SEEDS = (202610270, 202610271)
MARKERS = ('panCK', 'CD3', 'DAPI')
CUTOFFS = (22, 30, 18)
EXCLUDED_INDICES = tuple((k * 1199) // 31 for k in range(0, 32, 4))
TRAINING_INDICES = tuple(i for i in range(1200) if i not in EXCLUDED_INDICES)
PIXELS = 512 * 512
COMMON = ('ODS_F', 'cutoff_tolerant_F', 'cutoff_exact_F1', 'avg_ssim', 'avg_psnr')
NEW_METRICS = (*COMMON, 'AP', 'train_tolerant_F', 'train_exact_F1', 'rgb_psnr')
PAIRS = (('E_count', 'S_count'), ('E_recon', 'S_recon'),
         ('E_count', 'E_recon'), ('S_count', 'S_recon'),
         ('E_count', 'S_recon'), ('E_recon', 'S_count'))


@dataclass(frozen=True)
class SyntheticCoverage:
    """Explicit reduced fixtures; never selected by a manifest or the CLI."""
    training_indices: tuple = (1, 2)
    excluded_count: int = 8
    hold: int = 2
    val: int = 1
    epochs: int = 2
    historical_arm_rows: int = 1
    historical_controls: int = 1


def coverage_values(coverage):
    require(coverage is None or type(coverage) is SyntheticCoverage,
            'explicit SyntheticCoverage fixture required')
    if coverage is None:
        return dict(training_indices=TRAINING_INDICES, excluded_count=8,
                    hold=300, val=630, epochs=32,
                    historical_arm_rows=96, historical_controls=32)
    value = vars(coverage).copy()
    require(value['training_indices'] and len(set(value['training_indices'])) == len(value['training_indices'])
            and all(type(i) is int and i in TRAINING_INDICES for i in value['training_indices'])
            and tuple(sorted(value['training_indices'])) == value['training_indices'],
            'synthetic original training identities required')
    for field in value.keys() - {'training_indices'}:
        require(type(value[field]) is int and value[field] > 0, 'positive synthetic coverage: '+field)
    return value


def iter_rows(row, sources):
    path = sources.add(row)
    require(path.name.endswith('.jsonl.gz'), 'gzip JSONL scalar stream required')
    found = False
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                require(bool(line.strip()), 'empty scalar stream row')
                value = decode(line)
                require(isinstance(value, dict), 'scalar row object required')
                found = True
                yield value
    except (EOFError, OSError, zlib.error, UnicodeError) as error:
        raise ValueError('incomplete gzip/EOF: '+str(path)) from error
    require(found, 'nonempty scalar stream required')


def counts(value, label, pixels=PIXELS):
    value = _validated_counts(value, label, pixels=pixels)
    # Each of the actual81 footprint positions can match at most one pixel.
    # These are necessary scalar constraints, including globally empty GT or
    # predictions; they cannot establish actual spatial matching from counts.
    require(all(u <= 81*value['G'] and v <= 81*k for k, u, v in
                zip(value['K'], value['U'], value['V'])),
            label+': native81 footprint/empty-population count definition mismatch')
    return value


def check_source_coverage(manifest, sources, members, phase, synthetic):
    rows = manifest.get('source_metadata')
    require(isinstance(rows, list) and rows, 'nonempty source identity coverage required')
    paths = [sources.add(r) for r in rows]
    if synthetic:
        return
    suffixes = [p.parts[-3:] for p in paths]
    expected = []
    if phase == 'train':
        expected = [('train', role, r['name']) for r in members['training'] for role in ('input', 'label')]
    else:
        expected = [('train' if split == 'hold' else 'val', role, name)
                    for split in ('hold', 'val') for name in members[split] for role in ('input', 'label')]
    require(all(suffixes.count(identity) == 1 for identity in expected),
            'complete unique native input/label source stat coverage required')
    require(not any(p.name in members['excluded_outer8'] for p in paths)
            and not any(p.parts[-3:-1] == ('test', role) for p in paths for role in ('input', 'label')),
            'old outer8/locked test source forbidden')


def thresholds(value, label):
    require(isinstance(value, list) and len(value) == 3
            and all(type(v) is int and 0 <= v <= 255 for v in value),
            'three explicit marker training thresholds required: '+label)
    return value


def endpoints(value, cutoff, train_threshold):
    """Native PR step integral and first-maximum ODS after integer pooling."""
    tolerant, exact, precision, recall = [], [], [], []
    for k, u, v, t in zip(*(value[f] for f in ('K', 'U', 'V', 'T'))):
        den = u * value['G'] + v * k
        tolerant.append(2*u*v/den if den else 0.)
        exact.append(2*t/(k+value['G']) if k+value['G'] else 1.)
        precision.append(u/k if k else 0.)
        recall.append(v/value['G'] if value['G'] else 0.)
    previous, terms = 0., []
    for t in range(255, -1, -1):
        if value['K'][t]:
            terms.append((recall[t]-previous)*precision[t])
            previous = recall[t]
    peak = max(tolerant)
    return dict(ODS_F=peak, ODS_threshold=tolerant.index(peak), AP=math.fsum(terms),
                cutoff_tolerant_F=tolerant[cutoff], cutoff_exact_F1=exact[cutoff],
                train_threshold=train_threshold, train_tolerant_F=tolerant[train_threshold],
                train_exact_F1=exact[train_threshold], tolerant_F_curve=tolerant,
                exact_F1_curve=exact, precision_curve=precision, recall_curve=recall)


def _names(value, n, label):
    require(isinstance(value, list) and len(value) == n
            and all(isinstance(v, str) and v not in ('', '.', '..')
                    and Path(v).name == v and '\\' not in v for v in value)
            and len(set(value)) == n, 'exact unique plain-name membership required: '+label)
    return value


def check_membership(value, cv):
    require(isinstance(value, dict) and set(value) == {'training', 'excluded_outer8', 'hold', 'val'},
            'exact training/excluded_outer8/hold/val membership required')
    training = value['training']
    require(isinstance(training, list) and len(training) == len(cv['training_indices'])
            and all(isinstance(v, dict) and set(v) == {'index', 'name'} for v in training),
            'original indexed training membership required')
    require(all(type(v['index']) is int for v in training)
            and [v['index'] for v in training] == list(cv['training_indices']),
            'exact original FIT1192 indices/exclusion required')
    names = _names([v['name'] for v in training], len(training), 'training')
    groups = [names, _names(value['excluded_outer8'], cv['excluded_count'], 'old outer8'),
              _names(value['hold'], cv['hold'], 'hold'), _names(value['val'], cv['val'], 'val')]
    require(sum(map(len, groups)) == len(set().union(*(set(g) for g in groups))),
            'training/old outer8/hold/val populations must be disjoint')
    return value


def check_membership_sources(manifest, sources, cv, synthetic):
    if synthetic:
        return
    refs = manifest.get('membership_sources')
    require(isinstance(refs, dict) and set(refs) == {'fit_metadata', 'pilot_manifest', 'data_manifest'},
            'role-bound original membership scalar sources required')
    fit, pilot, data = (read_json(refs[k], sources) for k in
                        ('fit_metadata', 'pilot_manifest', 'data_manifest'))
    original_train = _names(data.get('train'), 3717, 'native train3717')
    val = _names(data.get('val'), 630, 'native val630')
    require(original_train == sorted(original_train) and val == sorted(val),
            'native sorted train/val identities required')
    selected = original_train[::2][:1500]
    fit_names = _names(fit.get('fit_names'), 1200, 'original FIT1200')
    hold = _names(fit.get('hold_names'), 300, 'original HOLD300')
    require(fit_names == selected[:1200] and hold == selected[1200:],
            'original stride2 FIT1200/HOLD300 binding mismatch')
    fit32 = pilot.get('populations', {}).get('fit', {})
    indices = [k*1199//31 for k in range(32)]
    require(fit32.get('source_size') == 1200 and fit32.get('indices') == indices
            and fit32.get('names') == [fit_names[i] for i in indices],
            'original FIT32 scalar binding mismatch')
    expected = dict(training=[dict(index=i, name=fit_names[i]) for i in TRAINING_INDICES],
                    excluded_outer8=[fit_names[i] for i in EXCLUDED_INDICES], hold=hold, val=val)
    require(manifest['membership'] == expected, 'exact original FIT1192/excluded8/HOLD/val binding mismatch')


def check_terminal(path, manifest, sources, expected_reads):
    parent = Path(path).resolve().parent
    exit_path = sources.add(metadata(parent/'exit-code.txt'))
    require(exit_path.read_text().strip() == '0', 'actual terminal process exit0 required')
    status = read_json(metadata(parent/'status.json'), sources)
    require(status.get('schema') == SCHEMA and status.get('run_id') == manifest['run_id']
            and status.get('phase') == manifest['phase'] and status.get('status') == 'complete'
            and status.get('actual_target_reads') == expected_reads,
            'matching complete terminal phase/read receipt required')
    for field in ('source_metadata', 'input_metadata', 'data_source_metadata'):
        for row in status.get(field, []):
            sources.add(row)


def learning_rate(update, n, epochs):
    if update <= n:
        return 1e-4 + 9e-4*(update-1)/(n-1) if n > 1 else 1e-4
    return 1e-5 + .5*(1e-3-1e-5)*(1+math.cos(math.pi*(update-n)/((epochs-1)*n)))


def expected_schedule(seed, indices, epochs):
    # NumPy is used only for the specified SeedSequence/PCG64 RNG, never for
    # loading arrays. No producer order/D4/LR implementation is imported.
    import numpy as np
    for epoch in range(1, epochs+1):
        order = np.random.default_rng(np.random.SeedSequence([seed, 0, epoch])).permutation(indices)
        for original_index in order:
            i = int(original_index)
            code = int(np.random.default_rng(np.random.SeedSequence([seed, epoch, i])).integers(8))
            yield epoch, i, code


def check_updates(fit, members, cv, sources, calibration):
    identities = {r['index']: r['name'] for r in members['training']}
    n, epochs = len(identities), cv['epochs']
    expected = iter(expected_schedule(fit['seed'], cv['training_indices'], epochs))
    found = 0
    for found, row in enumerate(iter_rows(fit['updates'], sources), 1):
        require(found <= n*epochs, 'too many parameter updates')
        epoch, original_index, code = next(expected)
        require(row.get('arm') == fit['arm'] and type(row.get('seed')) is int and row['seed'] == fit['seed'],
                'update arm/seed binding mismatch')
        require(type(row.get('epoch')) is int and row['epoch'] == epoch
                and type(row.get('update_index')) is int and row['update_index'] == found,
                'complete ordered epoch/update coverage required')
        require(type(row.get('original_index')) is int and row['original_index'] == original_index
                and row.get('name') == identities[original_index],
                'paired original FIT1200 order/name mismatch')
        require(type(row.get('D4_code')) is int and row['D4_code'] == code,
                'paired independently seeded D4 mismatch')
        close(row.get('learning_rate'), learning_rate(found, n, epochs), 'complete32 learning rate')
        for field in ('train_total', 'base', 'aux', 'grad_norm'):
            number(row.get(field), 'update.'+field)
        require(row['base'] >= 0 and row['grad_norm'] >= 0, 'training loss/gradient range mismatch')
        if fit['arm'].endswith('_count'):
            require(0 <= number(row.get('soft_phi'), 'update.soft_phi') <= 1/3+1e-12
                    and type(row.get('threshold')) is int and 0 <= row['threshold'] < 255,
                    'finite count-arm soft-Phi and selected threshold0..254 required')
            close(row['train_total'], row['base']+calibration['lambda_count']*row['aux'],
                  'population-gradient surrogate objective')
        else:
            require(row.get('soft_phi') is None and row.get('threshold') is None and row['aux'] == 0,
                    'reconstruction arm must not use count objective')
            close(row['train_total'], row['base'], 'reconstruction objective')
        require(row.get('parameter_finite') is True, 'finite parameter update receipt required')
    require(found == n*epochs, 'missing parameter updates/full epochs')
    return found


def check_trajectory(fit, cv, sources):
    n = len(cv['training_indices'])
    found, last = 0, None
    for found, row in enumerate(iter_rows(fit['trajectory'], sources), 1):
        require(found <= cv['epochs'] and row.get('epoch') == found
                and type(row.get('epoch')) is int and row.get('parameter_updates') == found*n,
                'complete final-weight epoch trajectory required')
        phi = number(row.get('soft_phi'), 'trajectory.soft_phi')
        require(0 <= phi <= 1/3+1e-12, 'trajectory soft-Phi range mismatch')
        counts(row.get('byte_counts'), 'trajectory pooled panCK', pixels=n*PIXELS)
        q = row.get('quality')
        require(isinstance(q, dict), 'trajectory panCK quality required')
        require(-1 <= number(q.get('ssim'), 'trajectory panCK SSIM') <= 1,
                'trajectory panCK SSIM outside range')
        number(q.get('psnr'), 'trajectory panCK PSNR')
        maximum = integer(row.get('table_max_age_updates'), 'table maximum update age')
        average = number(row.get('table_mean_age_updates'), 'table average update age')
        discrepancy = number(row.get('table_max_abs_discrepancy'), 'table discrepancy')
        require(0 <= average <= maximum <= n and 0 <= discrepancy <= 1,
                'lagged count-table age/discrepancy range mismatch')
        if fit['arm'].endswith('_count'):
            require(maximum == n, 'complete once-per-epoch count-table update age required')
            close(average, (n+1)/2, 'complete count-table mean update age')
        else:
            require(maximum == average == discrepancy == 0, 'reconstruction arm has no lagged count table')

        last = row
    require(found == cv['epochs'], 'missing epoch trajectory')
    return last


def _calibration(value, members):
    require(isinstance(value, dict) and value.get('schema') == SCHEMA
            and value.get('training_only') is True, 'training-only calibration scalar object required')
    require(value.get('training_indices') == [r['index'] for r in members['training']]
            and value.get('training_members') == len(members['training']),
            'train-only calibration membership required')
    scales = value.get('loss_scales')
    require(isinstance(scales, dict) and set(scales) == {'mse', 'ssim', 'edge1', 'edge16'},
            'four donor normalization scales required')
    for field, scalar in scales.items():
        require(number(scalar, 'normalization scale.'+field) >= 1e-8,
                'floored training normalization scale required')
    base = number(value.get('reconstruction_gradient_sq'), 'initial base output-gradient norm squared')
    aux = number(value.get('auxiliary_gradient_sq'), 'initial auxiliary output-gradient norm squared')
    require(base > 0 and aux > 0, 'positive finite initial output-gradient norms required')
    close(value.get('lambda_count'), math.sqrt(base/aux), 'train-only output-gradient-balanced lambda')
    require(value.get('output_gradient_coordinate') == 'normalized_z'
            and value.get('auxiliary_byte_chain') == 255
            and value.get('additional_N_factor') is False
            and value.get('additional_marker_factor') is False,
            'normalized-z gradient/255/N/marker calibration convention mismatch')
    require(0 <= number(value.get('donor_soft_phi'), 'donor soft Phi') <= 1/3+1e-12
            and type(value.get('donor_threshold')) is int and 0 <= value['donor_threshold'] < 255,
            'valid donor pooled soft objective/threshold required')
    require(isinstance(value.get('donor_adjoint'), list) and len(value['donor_adjoint']) == 3,
            'three donor K/U/V adjoint entries required')
    adjoint = [number(v, 'donor adjoint') for v in value['donor_adjoint']]
    require(adjoint[0] >= 0 and adjoint[1] <= 0 and adjoint[2] <= 0,
            'pooled-Phi K/U/V derivative signs mismatch')
    donor_counts = value.get('donor_train_counts')
    require(isinstance(donor_counts, dict) and set(donor_counts) == set(MARKERS),
            'all-marker training donor counts required')
    selected = thresholds(value.get('donor_train_thresholds'), 'donor calibration')
    for j, marker in enumerate(MARKERS):
        counts(donor_counts[marker], 'training donor '+marker, pixels=len(members['training'])*PIXELS)
        require(endpoints(donor_counts[marker], CUTOFFS[j], selected[j])['ODS_threshold'] == selected[j],
                'training donor first-maximum threshold mismatch: '+marker)
    return value


def _check_train(path, sources, coverage):
    cv = coverage_values(coverage)
    manifest = read_json(metadata(Path(path).resolve()), sources)
    require(manifest.get('schema') == SCHEMA and manifest.get('phase') == 'train'
            and manifest.get('status') == 'complete', 'terminal complete FIT1192 train manifest required')
    require(isinstance(manifest.get('run_id'), str) and manifest['run_id'], 'nonempty training run ID required')
    if coverage is not None:
        require(manifest.get('synthetic') is True, 'explicit synthetic manifest required for reduced coverage')
    else:
        require(manifest.get('synthetic') is not True, 'synthetic evidence cannot satisfy production acceptance')
    members = check_membership(manifest.get('membership'), cv)
    check_membership_sources(manifest, sources, cv, coverage is not None)
    check_terminal(path, manifest, sources, len(cv['training_indices']))
    require(manifest.get('actual_training_target_reads') == len(cv['training_indices'])
            and manifest.get('actual_held_target_reads') == 0
            and manifest.get('checkpoint_selection') is False,
            'training-only once-target-read and fixed final-checkpoint receipts required')
    expected_access = iter(members['training'])
    accesses = 0
    for accesses, row in enumerate(iter_rows(manifest['training_access'], sources), 1):
        require(accesses <= len(members['training']), 'too many training target accesses')
        member = next(expected_access)
        require(row.get('event') == 'target_read' and row.get('row') == accesses-1
                and row.get('original_index') == member['index'] and row.get('name') == member['name'],
                'complete once-only original training target access order required')
    require(accesses == len(members['training']), 'missing training target access rows')
    calibration = _calibration(read_json(manifest['calibration'], sources), members)
    check_source_coverage(manifest, sources, members, 'train', coverage is not None)
    fits = manifest.get('fits')
    require(isinstance(fits, list), 'eight training fit receipts required')
    keys = [(f.get('arm'), f.get('seed')) for f in fits]
    expected = {(a, seed) for a in ARMS for seed in SEEDS}
    require(len(keys) == 8 and set(keys) == expected
            and all(type(f['seed']) is int for f in fits), 'exact eight unique arm/seed fits required')
    updates, paths = 0, set()
    for fit in fits:
        require(fit.get('epochs') == cv['epochs'] and type(fit.get('epochs')) is int
                and fit.get('parameter_updates') == len(cv['training_indices'])*cv['epochs']
                and fit.get('training_indices') == list(cv['training_indices']),
                'complete32 schedule and exact FIT1192 fit membership required')
        state = sources.add(fit['state'])
        require(str(state) not in paths, 'distinct final state for every fresh fit required')
        paths.add(str(state))
        updates += check_updates(fit, members, cv, sources, calibration)
        trajectory = check_trajectory(fit, cv, sources)
        selected = thresholds(fit.get('train_thresholds'), 'final fit')
        train_counts = fit.get('final_train_counts')
        final = fit.get('training_final')
        require(isinstance(train_counts, dict) and isinstance(final, dict)
                and final.get('byte_counts') == train_counts.get('panCK'),
                'unaugmented final training panCK count binding required')
        require(isinstance(final.get('quality'), dict)
                and -1 <= number(final['quality'].get('ssim'), 'final train panCK SSIM') <= 1,
                'finite final training panCK quality required')
        number(final['quality'].get('psnr'), 'final train panCK PSNR')
        require(0 <= number(final.get('soft_phi'), 'final train soft Phi') <= 1/3+1e-12,
                'final training soft-Phi range mismatch')
        require(isinstance(train_counts, dict) and set(train_counts) == set(MARKERS),
                'final native three-marker training counts required')
        for j, marker in enumerate(MARKERS):
            counts(train_counts[marker], 'final training '+marker, pixels=len(members['training'])*PIXELS)
            require(endpoints(train_counts[marker], CUTOFFS[j], selected[j])['ODS_threshold'] == selected[j],
                    'final train first-maximum threshold mismatch: '+marker)
            require(train_counts[marker]['G'] == calibration['donor_train_counts'][marker]['G'],
                    'fixed training target G changed across fits/calibration')
            if j == 0:
                require(trajectory['byte_counts']['G'] == train_counts[marker]['G'],
                        'trajectory training target G changed')
            if j:
                require(train_counts[marker] == calibration['donor_train_counts'][marker],
                        'unchanged marker training donor calibration changed')
    require(updates == 8*len(cv['training_indices'])*cv['epochs']
            and manifest.get('parameter_updates') == updates,
            'total eight-fit parameter update count mismatch')
    for row in manifest.get('source_metadata', []):
        sources.add(row)
    return manifest, calibration, dict(status='passed_training_saved_scalars', schema=SCHEMA,
        run_id=manifest['run_id'], fits=8, parameter_updates=updates,
        train_images=len(cv['training_indices']), epochs=cv['epochs'], synthetic=coverage is not None,
        trajectory_epochs=8*cv['epochs'], source_files_stat_checked=len(sources.rows),
        opaque_states_opened=0, scientific_promotion=False)


def check_train(train_manifest_path, *, coverage=None):
    sources = Sources()
    _, _, accepted = _check_train(train_manifest_path, sources, coverage)
    sources.finish()
    return accepted


def tail(histogram):
    total, result = sum(histogram), []
    for value in histogram:
        total -= value
        result.append(total)
    return result


def check_scores(row, donor=None):
    """Validate full score_image scalar output against independent identities."""
    require(isinstance(row, dict) and row.get('schema_version') == 1
            and row.get('marker_order') == list(MARKERS), 'canonical saved score schema/marker order required')
    require(row.get('shape') == [512, 512, 3] and row.get('pixels') == PIXELS,
            'native512 saved geometry required')
    quality(row.get('quality'), 'native quality')
    require(set(row.get('counts', {})) == set(MARKERS) and set(row.get('donor_counts', {})) == set(MARKERS),
            'all-marker output and donor counts required')
    moments = row.get('moments')
    require(isinstance(moments, dict), 'native byte SSE moments required')
    for field in ('output_target_SSE', 'donor_target_SSE', 'edit_SSE', 'edit_donor_error_cross'):
        require(isinstance(moments.get(field), list) and len(moments[field]) == 3,
                'three SSE moment channels required: '+field)
        for value in moments[field]:
            integer(value, field, field.endswith('cross'))
    changed = row.get('changed_values_by_marker')
    require(isinstance(changed, list) and len(changed) == 3, 'three changed marker counts required')
    for j, (marker, cutoff) in enumerate(zip(MARKERS, CUTOFFS)):
        c = counts(row['counts'][marker], marker)
        before = counts(row['donor_counts'][marker], 'donor '+marker)
        hist = row['histograms'][marker]
        for side in ('output', 'donor', 'target'):
            values = hist[side]
            require(isinstance(values, list) and len(values) == 256
                    and all(type(v) is int and v >= 0 for v in values) and sum(values) == PIXELS,
                    'native histogram count/pixel mismatch: '+marker+'.'+side)
        require(c['K'] == tail(hist['output']) and before['K'] == tail(hist['donor'])
                and c['G'] == before['G'] == tail(hist['target'])[cutoff],
                'strict-threshold K or target G/histogram mismatch: '+marker)
        require(hist.get('shift') == [a-b for a, b in zip(hist['output'], hist['donor'])],
                'signed histogram shift mismatch: '+marker)
        for side in ('output', 'donor'):
            close(row['means'][marker][side], sum(t*v for t, v in enumerate(hist[side]))/PIXELS,
                  marker+'.'+side+' mean')
        close(row['means'][marker]['shift'], row['means'][marker]['output']-row['means'][marker]['donor'],
              marker+' mean shift')
        # score_image's legacy per-image FITtau is explicitly21. It is checked
        # here only as the native saved field, never used as train calibration.
        ep = endpoints(c, cutoff, 21)
        old = row['endpoints'][marker]
        expected_old = {k: ep[k] for k in ('ODS_F', 'ODS_threshold', 'cutoff_tolerant_F',
                                         'cutoff_exact_F1', 'tolerant_F_curve', 'exact_F1_curve')}
        expected_old.update(FITtau_tolerant_F=ep['train_tolerant_F'], FITtau_exact_F1=ep['train_exact_F1'])
        require(old == expected_old, 'saved native endpoint/curve mismatch: '+marker)
        output, base, cost, cross = (moments[f][j] for f in
            ('output_target_SSE', 'donor_target_SSE', 'edit_SSE', 'edit_donor_error_cross'))
        support = integer(changed[j], marker+' changed pixels')
        require(support <= PIXELS and support <= cost <= 255**2*support
                and max(output, base) <= 255**2*PIXELS, 'byte SSE/support denominator mismatch: '+marker)
        require(output-base == cost+2*cross and cross**2 <= cost*base,
                'signed SSE/Cauchy identity mismatch: '+marker)
        close(row['quality']['psnr_channels'][j],
              20*math.log10(255/(math.sqrt(output/PIXELS)+1e-8)), marker+' PSNR/SSE')
        if donor is not None:
            require(before == donor['counts'][marker]
                    and base == donor['moments']['output_target_SSE'][j]
                    and hist['donor'] == donor['histograms'][marker]['output']
                    and hist['target'] == donor['histograms'][marker]['target'],
                    'paired donor/target count-histogram-SSE anchor mismatch: '+marker)
        if j:
            require(support == cost == cross == 0 and output == base and c == before
                    and hist['output'] == hist['donor'], 'unchanged CD3/DAPI scalar mismatch')
            if donor is not None:
                for field in ('ssim_channels', 'psnr_channels'):
                    close(row['quality'][field][j], donor['quality'][field][j], 'unchanged donor quality')
    d = moments['output_target_SSE'][0]-moments['donor_target_SSE'][0]
    cost = moments['edit_SSE'][0]
    require(type(row.get('D')) is int and row['D'] == d and type(row.get('C')) is int and row['C'] == cost,
            'signed panCK D/C identity mismatch')
    require(type(row.get('changed_pixels')) is int and row['changed_pixels'] == changed[0],
            'changed pixel count mismatch')
    close(row['changed_pixel_fraction'], changed[0]/PIXELS, 'changed pixel fraction')
    require(row.get('no_op') is (cost == 0) and row.get('D_over_C') == (d/cost if cost else None),
            'no-op or D/C mismatch')
    close(row['macro_ODS_F'], math.fsum(row['endpoints'][m]['ODS_F'] for m in MARKERS)/3,
          'native per-image macro ODS')
    clipping = row.get('clipping')
    require(isinstance(clipping, dict) and clipping.get('pixels') == PIXELS, 'clipping receipt required')
    if clipping.get('status') == 'measured':
        low = integer(clipping.get('below_zero'), 'below-zero clipping')
        high = integer(clipping.get('above_one'), 'above-one clipping')
        require(low+high <= PIXELS and clipping.get('clipped_pixels') == low+high, 'clipping support mismatch')
        close(clipping['clipping_fraction'], (low+high)/PIXELS, 'clipping fraction')
        number(clipping['preclip_mean'], 'preclip mean')
        require(0 <= number(clipping['postclip_mean'], 'postclip mean') <= 1, 'postclip mean range')
        close(clipping['rounding_mean_change'], row['means']['panCK']['output']/255-clipping['postclip_mean'],
              'clipping rounding mean')
    else:
        require(clipping.get('status') == 'not_run', 'known clipping measurement status required')
    rgb = 20*math.log10(255/(math.sqrt(sum(moments['output_target_SSE'])/(3*PIXELS))+1e-8))
    require('rgb_psnr' in row['quality'], 'separate coherent RGB PSNR diagnostic required')
    if 'rgb_psnr' in row['quality']:
        # A saved native torch.float32 diagnostic has its own rounding error.
        require(math.isclose(number(row['quality']['rgb_psnr'], 'native coherent RGB PSNR'), rgb,
                             rel_tol=0., abs_tol=2e-5), 'coherent RGB PSNR/SSE mismatch')
    if donor is not None and cost == 0:
        require(row['counts'] == donor['counts'] and row['quality'] == donor['quality'],
                'no-op score differs from native donor anchor')
    return row['quality'].get('rgb_psnr', rgb)


def check_historical_donor(row, scores, identity):
    require(isinstance(row, dict) and row.get('schema') == 'query-relation-ensemble930-v1'
            and (row.get('split'), row.get('name')) == identity,
            'exact ordered historical donor name/split membership required')
    require(row.get('marker_order') == list(MARKERS) and row.get('shape') == [512, 512, 3]
            and row.get('pixels') == PIXELS, 'native historical donor marker/geometry required')
    require(isinstance(row.get('counts'), dict) and set(row['counts']) == set(MARKERS),
            'all3 historical donor native count curves required')
    for marker in MARKERS:
        counts(row['counts'][marker], 'historical donor '+marker)
    quality(row.get('quality'), 'historical native donor quality')
    number(row['quality'].get('rgb_psnr'), 'historical native donor RGB PSNR')
    require(row['counts'] == scores['counts'], 'per-image historical donor all256 count parity mismatch')
    require(set(row['quality']) == set(scores['quality'])
            and {k: v for k, v in row['quality'].items() if k != 'rgb_psnr'}
            == {k: v for k, v in scores['quality'].items() if k != 'rgb_psnr'}
            and math.isclose(number(row['quality']['rgb_psnr'], 'historical native donor RGB PSNR'),
                             number(scores['quality']['rgb_psnr'], 'native coherent RGB PSNR'),
                             rel_tol=0., abs_tol=2e-5),
            'per-image historical donor exact native quality parity mismatch')
    if 'output_target_SSE' in row:
        require(row['output_target_SSE'] == scores['moments']['output_target_SSE'],
                'per-image historical donor SSE parity mismatch')
    if 'output_sum' in row:
        require(row['output_sum'] == [sum(scores['counts'][marker]['K']) for marker in MARKERS],
                'per-image historical donor byte sum parity mismatch')


def check_solver(value):
    require(isinstance(value, dict) and value.get('inference') is True
            and value.get('solver_dtype') in ('float64', 'torch.float64'),
            'float64 inference solver receipt required')
    require(0 <= number(value.get('relative_residual_max'), 'inference relative residual') <= 1e-10
            and value.get('zero_rhs_absolute_residual_max') == 0
            and 0 <= number(value.get('residual_mean_max'), 'inference residual mean') <= 1e-12,
            'inference screened residual tolerance failed')


def image_metrics(scores, selected):
    markers = [endpoints(scores['counts'][m], cutoff, selected[j])
               for j, (m, cutoff) in enumerate(zip(MARKERS, CUTOFFS))]
    result = {field: math.fsum(m[field] for m in markers)/3 for field in
              ('ODS_F', 'AP', 'cutoff_tolerant_F', 'cutoff_exact_F1', 'train_tolerant_F', 'train_exact_F1')}
    result.update({field: scores['quality'][field] for field in ('avg_ssim', 'avg_psnr')})
    result['rgb_psnr'] = scores['quality']['rgb_psnr']
    result['panCK_ODS_F'] = markers[0]['ODS_F']
    return result


def accumulator():
    return dict(counts={m: dict(G=0, **{f: [0]*256 for f in ('K', 'U', 'V', 'T')}) for m in MARKERS},
                quality={f: [] for f in ('avg_ssim', 'avg_psnr', 'rgb_psnr')},
                channels={f: [[], [], []] for f in ('ssim_channels', 'psnr_channels')},
                images=0, D=0, C=0, changed_pixels=0)


def add_scores(total, scores, rgb):
    total['images'] += 1
    for marker in MARKERS:
        total['counts'][marker] = pooled([total['counts'][marker], scores['counts'][marker]])
    for field in ('avg_ssim', 'avg_psnr'):
        total['quality'][field].append(scores['quality'][field])
    total['quality']['rgb_psnr'].append(rgb)
    for field in total['channels']:
        for j in range(3):
            total['channels'][field][j].append(scores['quality'][field][j])
    for field in ('D', 'C', 'changed_pixels'):
        total[field] += scores[field]


def finish_group(total, split, arm, seed, selected):
    n = total['images']
    require(n > 0, 'nonempty complete metric group required')
    marker_rows = [dict(marker=m, counts=total['counts'][m],
                        **endpoints(total['counts'][m], cutoff, selected[j]))
                   for j, (m, cutoff) in enumerate(zip(MARKERS, CUTOFFS))]
    result = dict(split=split, arm=arm, seed=seed, images=n, marker_rows=marker_rows,
                  train_thresholds=selected, D=total['D'], C=total['C'], changed_pixels=total['changed_pixels'])
    for field in ('ODS_F', 'AP', 'cutoff_tolerant_F', 'cutoff_exact_F1', 'train_tolerant_F', 'train_exact_F1'):
        result[field] = math.fsum(r[field] for r in marker_rows)/3
    result['panCK_ODS_F'] = marker_rows[0]['ODS_F']
    for field, values in total['quality'].items():
        result[field] = math.fsum(values)/n
    for field, channels in total['channels'].items():
        result[field] = [math.fsum(v)/n for v in channels]
    return result


def historical_analysis(baseline, groups, donors, cv):
    require(isinstance(baseline, dict) and isinstance(baseline.get('populations'), dict)
            and set(baseline['populations']) == {'hold', 'val'}, 'both historical populations required')
    result = {}
    for split in ('hold', 'val'):
        old = baseline['populations'][split]
        refs = []
        for field, n in (('arm_rows', cv['historical_arm_rows']), ('original_controls', cv['historical_controls'])):
            require(isinstance(old.get(field), list) and len(old[field]) == n,
                    'all128 unchanged historical rows required: '+split+'.'+field)
            for i, row in enumerate(old[field]):
                require(isinstance(row.get('arm'), str) and row['arm'], 'historical arm identity required')
                for metric in COMMON:
                    number(row.get(metric), 'historical '+metric)
                refs.append((field, i, row))
        b0 = [row for row in old['original_controls'] if row['arm'] == 'B0']
        require(len(b0) == 1, 'unique unchanged historical B0 donor reference required')
        reconstructed = next(g for g in donors if g['split'] == split)
        for field in COMMON:
            close(reconstructed[field], b0[0][field], 'historical B0 reconstructed donor '+field)
        strongest = max(row['ODS_F'] for _, _, row in refs)
        new = {(g['arm'], g['seed']): g for g in groups if g['split'] == split}
        pairs = [dict(candidate=[a, seed], reference=[b, seed],
                      delta={f: new[a, seed][f]-new[b, seed][f] for f in NEW_METRICS})
                 for seed in SEEDS for a, b in PAIRS]
        comparisons = []
        for (arm, seed), group in new.items():
            for source, i, row in refs:
                fields = (*COMMON, 'AP') if 'AP' in row else COMMON
                comparisons.append(dict(candidate=[arm, seed], reference=dict(source=source, index=i, arm=row['arm']),
                    delta={f: group[f]-number(row[f], 'historical '+f) for f in fields}))
        result[split] = dict(imported_historical_rows=old,
            reconstructed_historical_donor=dict(arm='B0', exact_common_metric_parity=True,
                values={field: reconstructed[field] for field in COMMON}),
            strongest_known_ODS=[dict(source=s, index=i, row=r) for s, i, r in refs if r['ODS_F'] == strongest],
            paired_deltas=pairs, all_new_to_all_historical_deltas=comparisons,
            adverse_deltas=[r for r in pairs+comparisons if any(v < -1e-12 for v in r['delta'].values())])
    return result


def check(train_manifest_path, eval_manifest_path, output=None, *, coverage=None, max_output_bytes=16_000_000):
    sources = Sources()
    training, calibration, train_acceptance = _check_train(train_manifest_path, sources, coverage)
    path = Path(eval_manifest_path).resolve()
    manifest = read_json(metadata(path), sources)
    require(manifest.get('schema') == SCHEMA and manifest.get('phase') == 'evaluate'
            and manifest.get('status') == 'complete', 'terminal complete FIT1192 evaluation manifest required')
    require(manifest.get('run_id') == training['run_id'], 'train/evaluation run ID mismatch')
    require(manifest.get('training_manifest') == metadata(Path(train_manifest_path).resolve()),
            'exact frozen training manifest binding required')
    if coverage is not None:
        require(manifest.get('synthetic') is True, 'explicit synthetic evaluation fixture required')
    else:
        require(manifest.get('synthetic') is not True, 'synthetic evaluation cannot satisfy production acceptance')
    cv = coverage_values(coverage)
    check_terminal(path, manifest, sources, cv['hold']+cv['val'])
    check_quality_source(manifest.get('quality_metadata', {}), sources)
    require(manifest.get('membership') == training['membership'], 'evaluation membership changed after training')
    members = training['membership']
    ordered = [(s, name) for s in ('hold', 'val') for name in members[s]]
    fits = {(r['arm'], r['seed']): r for r in training['fits']}
    models = manifest.get('models')
    require(isinstance(models, list) and len(models) == 8
            and {(r.get('arm'), r.get('seed')) for r in models} == set(fits), 'eight unique frozen final models required')
    for model in models:
        fit = fits[model['arm'], model['seed']]
        require(model.get('state') == fit['state'] and model.get('train_thresholds') == fit['train_thresholds'],
                'evaluation final state/training calibration changed')
        sources.add(model['state'])
    check_source_coverage(manifest, sources, members, 'evaluate', coverage is not None)
    anchors, totals = {}, {}
    donor_totals = {s: accumulator() for s in ('hold', 'val')}
    historical_donors = iter(iter_rows(manifest['historical_donors'], sources))
    for index, row in enumerate(iter_rows(manifest['donors'], sources)):
        require(index < len(ordered) and type(row.get('index')) is int and row['index'] == index
                and (row.get('split'), row.get('name')) == ordered[index], 'complete ordered unique donor membership required')
        scores = row.get('scores')
        rgb = check_scores(scores)
        historic = next(historical_donors, None)
        require(historic is not None, 'missing historical donor rows')
        check_historical_donor(historic, scores, ordered[index])
        require(scores['C'] == scores['D'] == scores['changed_pixels'] == 0
                and scores['counts'] == scores['donor_counts'], 'donor must be native no-op scalar anchor')
        anchors[index] = scores
        add_scores(donor_totals[row['split']], scores, rgb)
    require(len(anchors) == len(ordered), 'missing donor rows')
    require(next(historical_donors, None) is None, 'extra historical donor rows')
    paired, found, image_rows = [], 0, {}
    expected_outputs = [(a, seed) for a in ARMS for seed in SEEDS]
    for found, row in enumerate(iter_rows(manifest['predictions'], sources), 1):
        index, position = divmod(found-1, 8)
        require(index < len(ordered), 'too many prediction rows')
        split, name = ordered[index]
        arm, seed = expected_outputs[position]
        require(type(row.get('index')) is int and row['index'] == index
                and (row.get('split'), row.get('name'), row.get('arm'), row.get('seed')) == (split, name, arm, seed)
                and type(row.get('seed')) is int, 'complete ordered unique eight-per-image prediction membership required')
        scores = row.get('scores')
        rgb = check_scores(scores, anchors[index])
        check_solver(row.get('solver'))
        key = split, arm, seed
        add_scores(totals.setdefault(key, accumulator()), scores, rgb)
        image_rows[arm, seed] = image_metrics(scores, fits[arm, seed]['train_thresholds'])
        if position == 7:
            for paired_seed in SEEDS:
                for a, b in PAIRS:
                    delta = {f: image_rows[a, paired_seed][f]-image_rows[b, paired_seed][f]
                             for f in (*NEW_METRICS, 'panCK_ODS_F')}
                    paired.append(dict(index=index, split=split, name=name, candidate=[a, paired_seed],
                        reference=[b, paired_seed], delta=delta,
                        adverse=any(v < -1e-12 for v in delta.values())))
            image_rows.clear()
    require(found == len(ordered)*8, 'missing prediction rows')
    seen = 0
    for seen, row in enumerate(iter_rows(manifest['access'], sources), 1):
        index, position = divmod(seen-1, 2)
        require(index < len(ordered), 'too many target access rows')
        split, name = ordered[index]
        require(type(row.get('index')) is int and row['index'] == index
                and (row.get('split'), row.get('name')) == (split, name)
                and row.get('event') == ('generated' if position == 0 else 'target_read'),
                'all8-generation-before-single-target-read chronology required')
        if position == 0:
            require(type(row.get('outputs')) is int and row['outputs'] == 8, 'all8 generated outputs required')
    require(seen == 2*len(ordered), 'missing target access rows')
    for field, expected in (('actual_target_reads', len(ordered)), ('generated_outputs', len(ordered)*8),
                            ('RGB_calls', len(ordered)*9)):
        require(type(manifest.get(field)) is int and manifest[field] == expected, 'exact evaluation accounting mismatch: '+field)
    groups = [finish_group(totals[s, a, seed], s, a, seed, fits[a, seed]['train_thresholds'])
              for s in ('hold', 'val') for a in ARMS for seed in SEEDS]
    donors = [finish_group(donor_totals[s], s, 'donor', None, calibration['donor_train_thresholds']) for s in ('hold', 'val')]
    baseline = read_json(manifest['baseline_analysis'], sources)
    comparisons = historical_analysis(baseline, groups, donors, cv)
    sources.finish()
    accepted = dict(status='passed_saved_scalar_consistency', schema=SCHEMA, run_id=training['run_id'],
        observed_at=datetime.now(timezone.utc).isoformat(), synthetic=coverage is not None,
        training=train_acceptance, predictions=found, donors=len(anchors), groups=len(groups),
        historical_rows_per_population=cv['historical_arm_rows']+cv['historical_controls'],
        target_reads=manifest['actual_target_reads'], source_files_stat_checked=len(sources.rows),
        gzip_EOF_streams=21,
        historical_donor_parity=dict(images=len(ordered), markers=3, thresholds=256,
            count_fields=['K', 'U', 'V', 'T', 'G'], exact_count_parity=True,
            exact_native_quality_parity=True, rgb_psnr_float32_abs_tol=2e-5, exact_order=True),
        consumer=dict(torch_imports=0, array_loads=0, image_reads=0,
            target_reads=0, RGB_rescores=0, state_loads=0, new_fits=0),
        scientific_promotion=False, method_success_claim=False,
        limit='Saved-scalar consistency is not independent truth or evidence of patient independence.')
    analysis = dict(schema=SCHEMA, groups=groups, donors=donors, populations=comparisons,
        paired_per_image=dict(file='paired-per-image.jsonl.gz', rows=len(paired),
                              adverse_rows=sum(r['adverse'] for r in paired),
                              scope='All six same-seed arm comparisons for every image; adverse rows retained.'),
        calibration_scope=dict(new='Per-final-fit, per-marker pooled training first-maximum thresholds.',
            historical='Literal original FITtau calibrations retained only within historical rows.',
            legacy_score_image='Per-image FITtau21 fields are consistency-checked, not used for new train calibration.',
            historical_delta_fields=list(COMMON), excluded_historical_delta_fields=['FITtau_tolerant_F', 'FITtau_exact_F1']),
        quality_scope='Image mean of native marker/channel means; coherent RGB PSNR kept separately.',
        primary_population='val', secondary_population='hold', scientific_promotion=False,
        independent_confirmation='not_run')
    if output is not None:
        destination = Path(output).resolve()
        require(not destination.exists(), 'new acceptance output directory required')
        reports = {name: (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
                   for name, value in {'accepted.json': accepted, 'analysis.json': analysis}.items()}
        reports['paired-per-image.jsonl.gz'] = gzip.compress(
            ''.join(json.dumps(r, allow_nan=False)+'\n' for r in paired).encode(), mtime=0)
        total = sum(map(len, reports.values()))
        require(type(max_output_bytes) is int and max_output_bytes > 0 and total <= max_output_bytes,
                'checker output byte bound exceeded before writing')
        destination.mkdir(parents=True)
        for name, raw in reports.items():
            with (destination/name).open('xb') as stream:
                stream.write(raw)
    return accepted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--eval-manifest')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.eval_manifest:
        require(bool(args.output), 'new --output directory required for final acceptance')
        result = check(args.train_manifest, args.eval_manifest, args.output)
    else:
        result = check_train(args.train_manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
