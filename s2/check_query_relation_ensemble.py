"""Accept ensemble930 saved scalars, without Torch, images, targets or rescoring.

Count arithmetic and historical comparisons are implemented here independently
of the producer. This is saved-output consistency, not a second truth measure.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path

SCHEMA = 'query-relation-ensemble930-v1'
ARMS = ('E_C', 'S_C', 'P_C')
SEEDS = (202610270, 202610271)
FOLDS = (1, 2, 3)
TRAINING = tuple(i for i in range(32) if i % 4)
POPULATIONS = {'hold': 300, 'val': 630}
MARKERS = ('panCK', 'CD3', 'DAPI')
CUTOFFS = (22, 30, 18)
PIXELS = 512*512
HISTORICAL_ROWS = {'arm_rows': 96, 'original_controls': 32}
METRICS = ('ODS_F', 'FITtau_tolerant_F', 'cutoff_tolerant_F',
           'FITtau_exact_F1', 'cutoff_exact_F1', 'avg_ssim', 'avg_psnr')
HISTORICAL_COMMON_METRICS = ('ODS_F', 'cutoff_tolerant_F', 'cutoff_exact_F1', 'avg_ssim', 'avg_psnr')


def require(value, message):
    if not value:
        raise ValueError(message)


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'finite scalar required: '+label)
    return float(value)


def integer(value, label, signed=False):
    require(type(value) is int and (signed or value >= 0), 'integer count required: '+label)
    return value


def close(actual, expected, label):
    require(math.isclose(number(actual, label), expected, rel_tol=1e-12, abs_tol=1e-12),
            'scalar identity mismatch: '+label)


def metadata(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.is_file()
            and not path.is_symlink() and path.resolve() == path,
            'canonical regular file required: '+str(path))
    stat = path.stat()
    require(stat.st_size > 0, 'nonempty file required: '+str(path))
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


class Sources:
    def __init__(self):
        self.rows = {}

    def add(self, row):
        require(isinstance(row, dict) and set(row) == {'path', 'size_bytes', 'mtime_ns'},
                'exact path/size/mtime source metadata required')
        require(type(row['size_bytes']) is int and type(row['mtime_ns']) is int,
                'integer source size/mtime required')
        require(metadata(row['path']) == row, 'source metadata changed: '+str(row['path']))
        require(self.rows.get(row['path'], row) == row, 'conflicting source metadata')
        self.rows[row['path']] = row
        return Path(row['path'])

    def finish(self):
        for row in list(self.rows.values()):
            self.add(row)


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        require(math.isfinite(value), 'nonfinite saved scalar')


def decode(text):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    value = json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)
    finite_tree(value)
    return value


def read_json(row, sources):
    path = sources.add(row)
    return decode(path.read_text())


def iter_rows(row, sources):
    path = sources.add(row)
    require(path.name.endswith(('.jsonl', '.jsonl.gz')), 'JSONL scalar stream required')
    found = False
    try:
        with (gzip.open if path.suffix == '.gz' else open)(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                require(line.strip(), 'empty scalar stream row')
                value = decode(line)
                require(isinstance(value, dict), 'scalar row object required')
                found = True
                yield value
    except (EOFError, OSError) as error:
        raise ValueError('incomplete gzip/EOF: '+str(path)) from error
    require(found, 'nonempty scalar stream required')


def read_rows(row, sources):
    return list(iter_rows(row, sources))


def counts(value, label, pixels=PIXELS):
    require(isinstance(value, dict) and set(value) == {'K', 'U', 'V', 'T', 'G'},
            label+': exact K/U/V/T/G counts required')
    g = integer(value['G'], label+'.G')
    require(g <= pixels, label+': G exceeds pixel population')
    for field in ('K', 'U', 'V', 'T'):
        curve = value[field]
        require(isinstance(curve, list) and len(curve) == 256,
                label+': 256 threshold counts required')
        for v in curve:
            integer(v, label+'.'+field)
        require(curve[-1] == 0 and all(a >= b for a, b in zip(curve, curve[1:])),
                label+': strict-threshold monotone tail required')
    require(value['K'][0] <= pixels, label+': K exceeds pixel population')
    require(all(t <= u <= k and t <= v <= g for k, u, v, t in
                zip(*(value[f] for f in ('K', 'U', 'V', 'T')))),
            label+': impossible overlap/denominator')
    return value


def endpoints(value, cutoff, fit_threshold=21):
    require(type(fit_threshold) is int and 0 <= fit_threshold < 256, 'explicit valid FIT threshold required')
    tolerant, exact = [], []
    for k, u, v, t in zip(*(value[f] for f in ('K', 'U', 'V', 'T'))):
        denominator = u*value['G']+v*k
        tolerant.append(2*u*v/denominator if denominator else 0.)
        exact.append(2*t/(k+value['G']) if k+value['G'] else 1.)
    peak = max(tolerant)
    return dict(ODS_F=peak, ODS_threshold=tolerant.index(peak),
                FITtau_tolerant_F=tolerant[fit_threshold], cutoff_tolerant_F=tolerant[cutoff],
                FITtau_exact_F1=exact[fit_threshold], cutoff_exact_F1=exact[cutoff])


def quality(value, label):
    require(isinstance(value, dict), label+': canonical native quality required')
    for field, average in (('ssim_channels', 'avg_ssim'), ('psnr_channels', 'avg_psnr')):
        require(isinstance(value.get(field), list) and len(value[field]) == 3,
                label+': canonical three quality channels required')
        channels = [number(v, label+'.'+field) for v in value[field]]
        if field == 'ssim_channels':
            require(all(-1 <= v <= 1 for v in channels), label+': SSIM outside range')
        close(value[average], math.fsum(channels)/3, label+'.'+average)
    return value


def geometry(row):
    require(row.get('schema') == SCHEMA, 'saved row ensemble schema required')
    require(row.get('marker_order') == list(MARKERS), 'canonical marker order/channel rotation mismatch')
    require(row.get('shape') == [512, 512, 3] and row.get('pixels') == PIXELS,
            'native512 geometry/pixel denominator mismatch')
    quality(row.get('quality'), 'quality')


def scalar_identities(row, donor):
    changed = row.get('changed_values_by_marker')
    require(isinstance(changed, list) and len(changed) == 3,
            'three changed-value counts required')
    for value in changed:
        integer(value, 'changed_values_by_marker')
    require(changed[1:] == [0, 0] and changed[0] <= PIXELS
            and type(row.get('changed_pixels')) is int and row['changed_pixels'] == changed[0],
            'unchanged CD3/DAPI or changed pixel count mismatch')
    d, c = integer(row.get('D'), 'D', True), integer(row.get('C'), 'C')
    require(changed[0] <= c <= 255**2*changed[0], 'edit SSE/changed support mismatch')
    moments = row['moments']
    for field in ('output_target_SSE', 'donor_target_SSE', 'edit_SSE', 'edit_donor_error_cross'):
        require(isinstance(moments.get(field), list) and len(moments[field]) == 3,
                'three moment channels required: '+field)
        for value in moments[field]:
            integer(value, field, field.endswith('cross'))
    for channel in range(3):
        output, before, cost, cross = (moments[f][channel] for f in
            ('output_target_SSE', 'donor_target_SSE', 'edit_SSE', 'edit_donor_error_cross'))
        require(max(output, before) <= 255**2*PIXELS, 'target SSE exceeds byte denominator')
        require(output-before == cost+2*cross, 'signed D moment identity mismatch')
        require(cross**2 <= cost*before, 'cross moment violates Cauchy bound')
        if channel:
            require(cost == cross == 0 and output == before, 'unchanged marker moment mismatch')
        close(row['quality']['psnr_channels'][channel],
              20*math.log10(255/(math.sqrt(output/PIXELS)+1e-8)), 'canonical PSNR/SSE')
        close(donor['quality']['psnr_channels'][channel],
              20*math.log10(255/(math.sqrt(before/PIXELS)+1e-8)), 'canonical donor PSNR/SSE')
        require(before == donor['output_target_SSE'][channel], 'paired donor SSE anchor changed')
    require(moments['edit_SSE'][0] == c and
            moments['output_target_SSE'][0]-moments['donor_target_SSE'][0] == d,
            'signed byte D/C mismatch')
    clip = row['clipping']
    require(clip.get('pixels') == PIXELS, 'clipping pixel denominator mismatch')
    low, high = (integer(clip.get(f), 'clipping.'+f) for f in ('below_zero', 'above_one'))
    require(low+high <= PIXELS, 'clipping counts exceed pixel denominator')
    for field in ('preclip_mean', 'postclip_mean', 'rounding_mean_change'):
        number(clip.get(field), 'clipping.'+field)
    require(0 <= clip['postclip_mean'] <= 1, 'postclip mean outside range')
    output_sum = integer(row.get('panCK_output_sum'), 'panCK_output_sum')
    require(output_sum <= 255*PIXELS, 'output byte sum exceeds denominator')
    require(output_sum == sum(row['counts']['K']), 'panCK byte sum/strict-threshold K identity mismatch')
    require(isinstance(row.get('output_sum'), list) and len(row['output_sum']) == 3
            and row['output_sum'][0] == output_sum
            and row['output_sum'][1:] == donor['output_sum'][1:], 'output byte/channel sum mismatch')
    close(clip['rounding_mean_change'], output_sum/(255*PIXELS)-clip['postclip_mean'],
          'rounding mean change')
    for field in ('ssim_channels', 'psnr_channels'):
        for channel in (1, 2):
            close(row['quality'][field][channel], donor['quality'][field][channel],
                  'unchanged marker quality')
    if c == 0:
        require(row['counts'] == donor['counts']['panCK'] and row['quality'] == donor['quality'],
                'no-op donor scalar mismatch')
    solvers = row.get('solver')
    require(isinstance(solvers, list) and len(solvers) == 3
            and {s.get('fold') for s in solvers} == set(FOLDS), 'exact three original inference folds required')
    for solver in solvers:
        require(solver.get('inference') is True, 'inference solver receipt required')
        if row['arm'] != 'P_C':
            require(solver.get('solver_dtype') in ('float64', 'torch.float64'), 'float64 screened inference required')
            require(0 <= number(solver['relative_residual_max'], 'solver relative residual') <= 1e-10
                    and solver['zero_rhs_absolute_residual_max'] == 0,
                    'screened inference residual exceeds accepted tolerance')
            require(0 <= number(solver['residual_mean_max'], 'inferred residual mean') <= 1e-12,
                    'screened fold residual constant mode changed')


def pooled(values):
    return dict(G=sum(v['G'] for v in values), **{
        field: [sum(v[field][t] for v in values) for t in range(256)]
        for field in ('K', 'U', 'V', 'T')})


def aggregate_group(rows, donors, split, arm, seed):
    selected = [r for r in rows if r['split'] == split and r['arm'] == arm and r['seed'] == seed]
    members = [d for d in donors if d['split'] == split]
    n = len(members)
    require(n > 0, 'nonempty group membership required')
    markers = []
    for marker, cutoff in zip(MARKERS, CUTOFFS):
        values = ([r['counts'] for r in selected] if marker == 'panCK' and arm != 'donor'
                  else [d['counts'][marker] for d in members])
        markers.append(dict(marker=marker, **endpoints(pooled(values), cutoff)))
    values = members if arm == 'donor' else selected
    require(len(values) == n, 'complete group rows required')
    result = dict(split=split, arm=arm, seed=seed, images=n, marker_rows=markers,
                  **{f: math.fsum(m[f] for m in markers)/3 for f in METRICS[:5]})
    for field in ('avg_ssim', 'avg_psnr'):
        result[field] = math.fsum(r['quality'][field] for r in values)/n
    for field in ('ssim_channels', 'psnr_channels'):
        result[field] = [math.fsum(r['quality'][field][i] for r in values)/n for i in range(3)]
    if arm != 'donor':
        for field in ('D', 'C', 'changed_pixels'):
            result[field] = sum(r[field] for r in values)
    return result


def aggregate(rows, donors):
    groups = [aggregate_group(rows, donors, split, arm, seed)
              for split in POPULATIONS for arm in ARMS for seed in SEEDS]
    before = [aggregate_group(rows, donors, split, 'donor', None) for split in POPULATIONS]
    return groups, before


def compare_tree(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'summary fields mismatch: '+label)
        for key in expected:
            compare_tree(actual[key], expected[key], label+'.'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'summary length mismatch: '+label)
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare_tree(a, e, label+'.'+str(i))
    elif type(expected) is float:
        close(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, 'summary value mismatch: '+label)


def validate_rows(rows, donors, members):
    ordered_members = [(s, n) for s, names in members.items() for n in names]
    expected_members = set(ordered_members)
    donor_keys = [(d.get('split'), d.get('name')) for d in donors]
    require(len(donor_keys) == len(expected_members) and set(donor_keys) == expected_members,
            'complete unique donor membership required')
    require(donor_keys == ordered_members, 'canonical donor stream order required')
    expected = {(s, n, a, seed) for s, n in expected_members for a in ARMS for seed in SEEDS}
    keys = [(r.get('split'), r.get('name'), r.get('arm'), r.get('seed')) for r in rows]
    require(len(keys) == len(expected) and set(keys) == expected,
            'complete unique prediction membership required')
    require(keys == [(s, n, a, seed) for s, n in ordered_members for a in ARMS for seed in SEEDS],
            'canonical six-per-image prediction stream order required')
    anchors = {(d['split'], d['name']): d for d in donors}
    for donor in donors:
        geometry(donor)
        require(set(donor['counts']) == set(MARKERS), 'donor three marker counts required')
        require(isinstance(donor.get('output_sum'), list) and len(donor['output_sum']) == 3
                and isinstance(donor.get('output_target_SSE'), list) and len(donor['output_target_SSE']) == 3,
                'donor three channel byte sums/SSE required')
        for channel, marker in enumerate(MARKERS):
            counts(donor['counts'][marker], 'donor '+marker)
            require(integer(donor['output_sum'][channel], 'donor output sum') == sum(donor['counts'][marker]['K']),
                    'donor byte sum/strict-threshold K identity mismatch')
            sse = integer(donor['output_target_SSE'][channel], 'donor SSE')
            require(sse <= 255**2*PIXELS, 'donor SSE exceeds byte denominator')
            close(donor['quality']['psnr_channels'][channel],
                  20*math.log10(255/(math.sqrt(sse/PIXELS)+1e-8)), 'canonical donor PSNR/SSE')
    for row in rows:
        geometry(row)
        donor = anchors[row['split'], row['name']]
        counts(row['counts'], 'prediction panCK')
        require(row['counts']['G'] == donor['counts']['panCK']['G'], 'panCK target G changed across paired rows')
        scalar_identities(row, donor)


def analyze(groups, donor_groups, baseline, rows=None):
    """Preserve literal old rows; choose a whole-row ODS reference, never a composite."""
    result = {}
    for split in POPULATIONS:
        old = baseline['populations'][split]
        require(all(len(old[field]) == n for field, n in HISTORICAL_ROWS.items()),
                'complete unchanged historical baseline row pool required')
        imported = dict(arm_rows=old['arm_rows'], original_controls=old['original_controls'])
        references = [(('arm_rows', i), row) for i, row in enumerate(old['arm_rows'])]
        references += [(('original_controls', i), row) for i, row in enumerate(old['original_controls'])]
        require(references, 'complete historical strong-baseline pool required')
        for _, row in references:
            for field in METRICS:
                number(row[field], 'historical '+field)
        strongest_value = max(r['ODS_F'] for _, r in references)
        strongest = [dict(source=where[0], index=where[1], row=row) for where, row in references
                     if row['ODS_F'] == strongest_value]
        new = [g for g in groups if g['split'] == split]
        pairs, all_old = [], []
        for seed in SEEDS:
            candidate = next(g for g in new if g['arm'] == 'E_C' and g['seed'] == seed)
            for arm in ('S_C', 'P_C'):
                control = next(g for g in new if g['arm'] == arm and g['seed'] == seed)
                pairs.append(dict(candidate=['E_C', seed], reference=[arm, seed],
                                  delta={f: candidate[f]-control[f] for f in METRICS}))
            for where, row in references:
                all_old.append(dict(candidate=['E_C', seed], reference=dict(source=where[0], index=where[1],
                    arm=row['arm'], priority=row.get('priority')),
                    delta={f: candidate[f]-row[f] for f in HISTORICAL_COMMON_METRICS}))
        result[split] = dict(groups=new, donor=next(g for g in donor_groups if g['split'] == split),
            imported_historical_rows=imported, strongest_known_ODS=strongest,
            paired_E_C_deltas=pairs, E_C_to_all_historical_deltas=all_old,
            adverse_deltas=[r for r in pairs+all_old if any(v < -1e-12 for v in r['delta'].values())])
        if rows is not None:
            by_key = {(r['name'], r['arm'], r['seed']): r for r in rows if r['split'] == split}
            paired = []
            for seed in SEEDS:
                for arm in ('S_C', 'P_C'):
                    statistics = {}
                    for metric in ('avg_ssim', 'avg_psnr'):
                        values = [by_key[n, 'E_C', seed]['quality'][metric]-by_key[n, arm, seed]['quality'][metric]
                                  for n, a, s in by_key if a == 'E_C' and s == seed]
                        statistics[metric] = dict(mean=math.fsum(values)/len(values), minimum=min(values), maximum=max(values),
                            better=sum(v > 1e-12 for v in values), equal=sum(abs(v) <= 1e-12 for v in values),
                            worse=sum(v < -1e-12 for v in values))
                    paired.append(dict(candidate=['E_C', seed], reference=[arm, seed],
                                       images=len(values), statistics=statistics))
            result[split]['paired_per_image_quality_deltas'] = paired
    return dict(status='accepted_saved_scalar_development_analysis', populations=result,
        scientific_promotion=False, method_success_claim=False, independent_confirmation='not_run',
        ODS_definition='Per-marker maximum over 256 pooled integer-derived tolerant-F values, then mean of three markers.',
        quality_scope='Canonical native per-image three-marker channel means, then arithmetic image mean.',
        auxiliary_FITtau_scope=dict(new_groups='All markers use the unchanged pilot threshold21.',
            historical_rows='Literal original per-arm/per-marker FIT calibrations retained.',
            cross_history_delta_fields=list(HISTORICAL_COMMON_METRICS),
            not_subtracted=['FITtau_tolerant_F', 'FITtau_exact_F1'],
            reason='Differently calibrated auxiliary FITtau fields are not the same estimand; fixed-cutoff F1 and ODS remain comparable.'),
        limits=['HOLD300 and exposed val630 are development populations.',
                'Saved-scalar algebra is not independent truth measurement or evidence of patient independence.',
                'Every seed and all compatible historical rows are retained; no metric-wise composite winner.'])


def check_quality_source(value, sources):
    require(value.get('marker_order') == list(MARKERS), 'canonical quality marker order required')
    paths = {Path(r['path']).name: sources.add(r) for r in value['sources']}
    require(set(paths) == {'metrics.py', 'metric_panel.py', 'canonical-source.json'},
            'three canonical quality sources required')
    metrics, panel = paths['metrics.py'].read_text(), paths['metric_panel.py'].read_text()
    captured = decode(paths['canonical-source.json'].read_text())['sources']['metrics.py']
    def ssim(text):
        nodes = [n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'SSIM']
        require(len(nodes) == 1, 'one canonical SSIM class required')
        return ast.get_source_segment(text, nodes[0])
    require(ssim(metrics) == ssim(captured), 'SSIM source differs from accepted capture')
    score = [n for n in ast.parse(panel).body if isinstance(n, ast.FunctionDef) and n.name == 'score']
    require(len(score) == 1, 'one canonical quality score function required')
    expressions = [n.value for n in ast.walk(score[0]) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.JoinedStr)
                and any(isinstance(v, ast.Constant) and v.value == '_psnr' for v in t.slice.values)
                for t in n.targets)]
    expected = ast.parse('float(20 * np.log10(255.0 / (np.sqrt(mse) + 1e-8)))', mode='eval').body
    require(len(expressions) == 1 and ast.dump(expressions[0], include_attributes=False) ==
            ast.dump(expected, include_attributes=False), 'canonical native PSNR source mismatch')


def check_freeze(freeze, sources):
    require(freeze.get('schema') == SCHEMA and freeze.get('marker_order') == list(MARKERS),
            'accepted ensemble schema/canonical marker order required')
    require(freeze.get('new_fits') == freeze.get('optimizer_steps') == 0,
            'ensemble must not introduce fitting or optimization')
    original = read_json(freeze['training_manifest'], sources)
    full = read_json(freeze['full_manifest'], sources)
    for field in ('config', 'spec'):
        sources.add(freeze[field])
    contract = full['metadata_contract']
    members = contract['populations']
    require(set(members) == set(POPULATIONS), 'both canonical development populations required')
    for split, n in POPULATIONS.items():
        require(isinstance(members[split], list) and len(members[split]) == n
                and len(set(members[split])) == n and all(isinstance(v, str) for v in members[split]),
                'exact canonical population size/unique names required: '+split)
    ordered = [(s, name) for s in POPULATIONS for name in members[s]]
    require(full['ordered_items'] == [list(v) for v in ordered], 'full930 canonical item order changed')
    require(not set(members['hold']) & set(members['val']), 'HOLD/validation population overlap')
    names = freeze['original_fit_names']
    require(names == contract['accepted_FIT32'] and len(names) == 32 and len(set(names)) == 32,
            'actual accepted FIT32 name binding required')
    require(not set(names) & set(members['hold']), 'FIT32/HOLD membership overlap')
    require(not set(names) & set(members['val']), 'FIT32/validation membership overlap')
    model_keys = [(m['fold'], m['arm'], m['seed']) for m in freeze['models']]
    expected = {(f, a, s) for f in FOLDS for a in ARMS for s in SEEDS}
    require(len(model_keys) == len(expected) and set(model_keys) == expected,
            'exact18 frozen original-fold models required')
    require(original.get('status') == 'complete_training_only_32_epoch_diagnostic'
            and original.get('schema') == 'query-relation-optimization-v1'
            and original.get('objective_version') == 'preclip_mse_v1',
            'accepted preclip32 training manifest required')
    for model in freeze['models']:
        fold, arm, seed = model['fold'], model['arm'], model['seed']
        train = [i for i in TRAINING if i % 4 != fold]
        held = [i for i in TRAINING if i % 4 == fold]
        require(model['epoch'] == 32 and model['training_indices'] == train and model['held_indices'] == held,
                'wrong frozen fold/epoch/train membership')
        fits = [f for f in original['fits'] if (f['fold'], f['arm'], f['seed']) == (fold, arm, seed)]
        require(len(fits) == 1, 'unique accepted original model source required')
        fit = fits[0]
        require(fit['training_indices'] == train and fit['held_indices'] == held,
                'source fit fold membership mismatch')
        require(fit['epochs'] == 32 and fit['split'] == 'inner'+str(fold)
                and fit['training_mse_uses_clamp'] is False
                and fit['objective_version'] == 'preclip_mse_v1', 'source fit objective/epoch mismatch')
        state = fit.get('epoch032', fit.get('state'))
        require(model['state'] == state == fit['state'] == fit['state_metadata']
                and model['statistics'] == fit['original_statistics'],
                'frozen state/statistics differ from accepted source fit')
        sources.add(model['state'])
        stat = read_json(model['statistics'], sources)
        require(stat['fold'] == fold and stat['training_indices'] == train
                and stat['outer_and_inner_held_used'] is False
                and stat['token_statistics']['metadata']['training_indices'] == train,
                'train-only original statistics membership mismatch')
    require(len({m['state']['path'] for m in freeze['models']}) == len(expected),
            'distinct18 frozen model paths required')
    frozen = freeze['sources']
    require([(v['split'], v['name']) for v in frozen] == ordered,
            'frozen source order/membership differs from exact full930 manifest')
    native = {r['path']: r for r in full['source_metadata']}
    for row in frozen:
        require(row['crop'] == [256, 256, 768, 768], 'canonical target crop changed')
        require(all(native.get(row[k]['path']) == row[k] for k in ('frame', 'target')),
                'frozen input differs from accepted native full930 source metadata')
        sources.add(row['frame']); sources.add(row['target'])
    require(len({r['frame']['path'] for r in frozen}) == len(ordered)
            and len({r['target']['path'] for r in frozen}) == len(ordered)
            and not {r['frame']['path'] for r in frozen} & {r['target']['path'] for r in frozen},
            'distinct target-free frame and target paths required')
    for row in freeze['source_metadata']:
        sources.add(row)
    check_quality_source(freeze['quality_metadata'], sources)
    return members, ordered


def check_history(freeze, donors, donor_groups, baseline, sources):
    anchors = {(d['split'], d['name']): d for d in donors}
    arms = {s: {r['arm'] for r in baseline['populations'][s]['original_controls']} for s in POPULATIONS}
    seen = set()
    for row in iter_rows(freeze['historical_counts'], sources):
        key = (row['split'], row['name'], row['arm'], row['marker'])
        require(key[:2] in anchors and row['arm'] in arms[row['split']]
                and row['marker'] in MARKERS and key not in seen,
                'historical count duplicate/membership mismatch')
        seen.add(key)
        if row['arm'] == 'B0':
            value = counts(row['counts'], 'historical B0 '+row['marker'])
            require(value == anchors[key[:2]]['counts'][row['marker']],
                    'historical donor count parity mismatch')
    require(len(seen) == sum(len(arms[s])*POPULATIONS[s]*3 for s in POPULATIONS),
            'complete historical count coverage required')
    seen = set()
    for row in iter_rows(freeze['historical_quality'], sources):
        key = (row['split'], row['name'], row['arm'])
        require(key[:2] in anchors and row['arm'] in arms[row['split']] and key not in seen,
                'historical quality duplicate/membership mismatch')
        seen.add(key)
        if row['arm'] == 'B0':
            for field in ('avg_ssim', 'avg_psnr'):
                close(row[field], anchors[key[:2]]['quality'][field], 'historical donor '+field)
    require(len(seen) == sum(len(arms[s])*POPULATIONS[s] for s in POPULATIONS),
            'complete historical quality coverage required')
    full = read_json(freeze['full_manifest'], sources)
    calibration = full['metadata_contract']['original_calibration']['B0']
    require(set(calibration) == set(MARKERS) and all(type(v) is int and 0 <= v < 256
            for v in calibration.values()), 'original per-marker B0 FIT calibration required')
    restored = []
    for group in donor_groups:
        old = [r for r in baseline['populations'][group['split']]['original_controls'] if r['arm'] == 'B0']
        require(len(old) == 1, 'one literal historical B0 aggregate required')
        original_endpoints = [endpoints(pooled([d['counts'][marker] for d in donors
            if d['split'] == group['split']]), cutoff, calibration[marker])
            for marker, cutoff in zip(MARKERS, CUTOFFS)]
        reconstructed = dict(group)
        for field in ('FITtau_tolerant_F', 'FITtau_exact_F1'):
            reconstructed[field] = math.fsum(v[field] for v in original_endpoints)/3
        for field in METRICS:
            close(reconstructed[field], old[0][field], 'historical pooled donor '+field)
        restored.append(dict(split=group['split'], original_FIT_thresholds=calibration,
            values={f: reconstructed[f] for f in METRICS}, exact_historical_parity=True))
    return dict(status='passed', original_B0_FIT_thresholds=calibration,
                new_group_FIT_thresholds={m: 21 for m in MARKERS}, populations=restored)


def check_access(audit, commits, ordered, manifest, synthetic=False, frozen_sources=None):
    expected_outputs = {(a, s) for a in ARMS for s in SEEDS}
    expected_events = (('frame_open', 'all6_generated', 'synthetic_target_created', 'image_committed')
        if synthetic else ('frame_open', 'all6_generated', 'target_open', 'target_read', 'image_committed'))
    require(len(audit) == len(ordered)*len(expected_events), 'complete target-access chronology required')
    for index, (split, name) in enumerate(ordered):
        chunk = audit[index*len(expected_events):(index+1)*len(expected_events)]
        require([r.get('event') for r in chunk] == list(expected_events),
                'all six outputs must precede the single target open/read')
        require(all(r.get('index') == index and (r.get('split'), r.get('name')) == (split, name) for r in chunk),
                'target-access exact membership/order mismatch')
        if frozen_sources is not None:
            source = frozen_sources[index]
            require(chunk[0].get('file') == source['frame'], 'actual frame access/source binding mismatch')
            if synthetic:
                require(chunk[2].get('real_target_opened') is False, 'synthetic profile real-target denial required')
            else:
                require(chunk[2].get('file') == source['target'] and chunk[2].get('crop') == source['crop'],
                        'actual target access/source/crop binding mismatch')
        outputs = chunk[1]['outputs']
        require(len(outputs) == 6 and {(r['arm'], r['seed']) for r in outputs} == expected_outputs,
                'six fixed outputs required before target access')
    require(len(commits) == len(ordered), 'one atomic scalar commit per image required')
    previous = {'predictions': 0, 'donors': 0}
    for index, (row, item) in enumerate(zip(commits, ordered)):
        require(row['index'] == index and (row['split'], row['name']) == item
                and row['prediction_rows'] == 6 and row['donor_rows'] == 1,
                'atomic commit membership/count mismatch')
        for field in previous:
            start, end = row[field]['start'], row[field]['end']
            integer(start, 'commit start'); integer(end, 'commit end')
            require(start == previous[field] and end > start, 'atomic commit offset gap/overlap')
            previous[field] = end
    for field in previous:
        require(previous[field] == manifest[field]['size_bytes'], 'committed stream physical EOF mismatch')
    n = len(ordered)
    expected = dict(frame_reads=n, target_reads=0 if synthetic else n, generated_outputs=6*n, RGB_calls=7*n,
                    prediction_rows=6*n, donor_rows=n, new_fits=0, optimizer_steps=0,
                    FIT32_target_reads=0, outer8_target_reads=0, test_target_reads=0)
    if 'model_inference_calls' in manifest['counters']:
        expected['model_inference_calls'] = 18*n
    for field, value in expected.items():
        require(type(manifest['counters'].get(field)) is int and manifest['counters'][field] == value,
                'actual counter mismatch: '+field)


def write_reports(destination, reports, max_output_bytes):
    require(type(max_output_bytes) is int and max_output_bytes > 0, 'positive checker output byte bound required')
    encoded = {name: (json.dumps(value, indent=2, allow_nan=False)+'\n').encode() for name, value in reports.items()}
    total = sum(map(len, encoded.values()))
    require(total <= max_output_bytes,
            f'checker output byte bound exceeded before writing: {total} > {max_output_bytes}; inputs retained')
    require(all(not (destination/name).exists() for name in reports), 'acceptance output already exists')
    destination.mkdir(parents=True, exist_ok=True)
    for name, raw in encoded.items():
        with (destination/name).open('xb') as stream:
            stream.write(raw)


def check(manifest_path, output=None, max_output_bytes=8_000_000):
    sources = Sources()
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(metadata(manifest_path), sources)
    require(manifest.get('schema') == SCHEMA and manifest.get('status') == 'completed',
            'terminal completed ensemble scalar manifest required')
    require(manifest.get('phase') == 'evaluate', 'evaluate manifest required')
    exit_path = sources.add(metadata(manifest_path.parent/'exit-code.txt'))
    require(exit_path.read_text().strip() == '0', 'actual evaluate process exit0 required')
    status = read_json(metadata(manifest_path.parent/'status.json'), sources)
    require(status.get('status') == 'completed' and status.get('phase') == 'evaluate'
            and status['counters'] == manifest['counters'], 'actual evaluate terminal receipt required')
    freeze = read_json(manifest['freeze'], sources)
    require(manifest['run_id'] == freeze['run_id'], 'run/freeze binding mismatch')
    members, ordered = check_freeze(freeze, sources)
    for row in manifest['source_metadata']:
        sources.add(row)
    predictions = read_rows(manifest['predictions'], sources)
    donors = read_rows(manifest['donors'], sources)
    validate_rows(predictions, donors, members)
    groups, donor_groups = aggregate(predictions, donors)
    summary = read_json(manifest['summary'], sources)
    order = lambda g: (g['split'], g['arm'], g['seed'] or -1)
    compare_tree(sorted(summary['groups'], key=order), sorted(groups, key=order), 'groups')
    compare_tree(sorted(summary['donors'], key=order), sorted(donor_groups, key=order), 'donors')
    baseline = read_json(freeze['baseline_analysis'], sources)
    require(summary['baseline_import'] == baseline, 'literal historical baseline import changed')
    historical_parity = check_history(freeze, donors, donor_groups, baseline, sources)
    check_access(read_rows(manifest['access_audit'], sources), read_rows(manifest['commits'], sources), ordered, manifest,
                 frozen_sources=freeze['sources'])
    require(manifest.get('donor_parity', {}).get('status') == 'passed', 'producer historical donor parity required')
    analysis = analyze(groups, donor_groups, baseline, predictions)
    sources.finish()
    accepted = dict(status='passed', schema=SCHEMA, run_id=manifest['run_id'],
        observed_at=datetime.now(timezone.utc).isoformat(), manifest=metadata(manifest_path),
        predictions=len(predictions), donors=len(donors), groups=len(groups), frozen_models=len(freeze['models']),
        populations={s: len(v) for s, v in members.items()}, gzip_EOF_streams=4,
        source_files_stat_checked=len(sources.rows),
        historical_donor_parity=historical_parity,
        consumer=dict(torch_imports=0, array_loads=0, image_reads=0, target_reads=0, RGB_rescores=0, new_fits=0),
        scientific_promotion=False, method_success_claim=False,
        limit='Independent saved-scalar consistency and source stat checks; no independent truth measurement.')
    destination = Path(output).resolve() if output is not None else manifest_path.parent
    write_reports(destination, {'analysis.json': analysis, 'accepted.json': accepted}, max_output_bytes)
    return accepted


def check_profile(profile_path, output=None, max_output_bytes=8_000_000):
    sources = Sources()
    profile_path = Path(profile_path).resolve()
    profile_row = metadata(profile_path)
    profile = read_json(profile_row, sources)
    require(profile.get('schema') == SCHEMA and profile.get('status') == 'passed'
            and profile.get('actual_native_source_frame') is True
            and profile.get('native_shape') == [512, 512, 3]
            and profile.get('real_target_reads') == 0 and profile.get('synthetic_target') is True
            and profile.get('models') == 18 and profile.get('outputs') == 6
            and profile.get('all_outputs_finite') is True, 'complete native-source synthetic-target profile required')
    exit_path = sources.add(metadata(profile_path.parent/'exit-code.txt'))
    require(exit_path.read_text().strip() == '0', 'actual profile process exit0 required')
    status = read_json(metadata(profile_path.parent/'status.json'), sources)
    require(status.get('status') == 'completed' and status.get('phase') == 'profile'
            and status['counters'] == profile['counters'], 'actual profile terminal receipt required')
    freeze = read_json(profile['freeze'], sources)
    _, ordered = check_freeze(freeze, sources)
    paths = {k: metadata(profile_path.parent/name) for k, name in
        (('predictions', 'predictions.jsonl.gz'), ('donors', 'donors.jsonl.gz'),
         ('access_audit', 'access-audit.jsonl'), ('commits', 'commits.jsonl'))}
    rows, donors = read_rows(paths['predictions'], sources), read_rows(paths['donors'], sources)
    split, name = ordered[0]
    validate_rows(rows, donors, {split: [name]})
    check_access(read_rows(paths['access_audit'], sources), read_rows(paths['commits'], sources),
                 ordered[:1], dict(paths, counters=profile['counters']), synthetic=True, frozen_sources=freeze['sources'][:1])
    sources.finish()
    result = dict(status='passed', phase='profile', schema=SCHEMA, run_id=freeze['run_id'],
        profile=profile_row, freeze=profile['freeze'], frozen_models=freeze['models'],
        source_metadata=freeze['source_metadata'], quality_metadata=freeze['quality_metadata'],
        outputs=6, donors=1, models=18, real_target_reads=0, native_shape=[512, 512, 3],
        source_files_stat_checked=len(sources.rows),
        consumer=dict(torch_imports=0, array_loads=0, image_reads=0, target_reads=0, RGB_rescores=0, new_fits=0),
        scientific_promotion=False, method_success_claim=False,
        limit='Native-source engineering and synthetic-target saved-scalar consistency only.')
    destination = Path(output).resolve() if output is not None else profile_path.parent
    write_reports(destination, {'accepted.json': result}, max_output_bytes)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('profile', 'evaluate'), default='evaluate')
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-output-bytes', type=int, default=8_000_000)
    args = parser.parse_args()
    print(json.dumps((check_profile if args.kind == 'profile' else check)(args.manifest, args.output, args.max_output_bytes)))


if __name__ == '__main__':
    main()
