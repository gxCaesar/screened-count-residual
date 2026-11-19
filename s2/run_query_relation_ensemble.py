"""One-shot CPU4 profile/evaluation of the frozen18-state full930 ensemble.

Inputs are explicit metadata bindings. Each image is streamed, all six outputs
are generated before its one target open, and only scalar/count rows persist.
A failed phase preserves its partial files and cannot resume or overwrite.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import gzip
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from PIL import Image
import torch

from . import query_relation_ensemble as ensemble
from . import query_relation_data as data
from . import query_relation_reconstruction as core
from . import query_relation_scoring as scoring
from . import run_query_relation_reconstruction as original


require = ensemble.require
SCHEMA = ensemble.SCHEMA


def parse_json(text):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read_json(path):
    return parse_json(Path(path).read_text())


def metadata(path):
    return data.file_row(str(Path(path).absolute()))


def stable(row):
    require(metadata(row['path']) == row, 'source metadata changed: '+row['path'])


def code_metadata():
    return [metadata(path) for path in (Path(__file__).resolve(), Path(ensemble.__file__).resolve(),
        Path(data.__file__).resolve(), Path(core.__file__).resolve(), Path(scoring.__file__).resolve(),
        Path(original.__file__).resolve(), Path(scoring.reference.__file__).resolve())]


def validate_spec(spec):
    expected = dict(schema=SCHEMA, arms=list(ensemble.ARMS), seeds=list(ensemble.SEEDS),
        folds=list(ensemble.FOLDS), epoch=32, primary_population='val',
        populations=dict(hold=300, val=630), aggregation='mean_float_residuals_before_clip_quantize',
        new_fits=0, optimizer_steps=0)
    require(all(spec.get(k) == v for k, v in expected.items()), 'fixed accepted ensemble protocol required')


def quality_for(cfg):
    quality = scoring.load_quality(cfg.get('quality_directory'))
    # Preserve the actual native RGB PSNR class separately from the charter's
    # arithmetic mean of per-marker PSNR. Neither is substituted for the other.
    source = Path(quality.metadata['sources'][0]['path'])
    text = source.read_text()
    node = scoring._node(text, ast.ClassDef, 'PSNR')
    namespace = dict(torch=torch, np=np, cv2=cv2)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    rgb = namespace['PSNR']()
    def measured(image, target):
        result = quality(image, target)
        result['rgb_psnr'] = float(rgb(torch.from_numpy(np.array(image, copy=True)),
            torch.from_numpy(np.array(target, copy=True))))
        return result
    measured.metadata = quality.metadata
    measured.metadata['RGB_PSNR_class_source'] = ast.get_source_segment(text, node)
    measured.metadata['RGB_PSNR_role'] = 'separate full-RGB diagnostic; charter guards use marker means'
    return measured


def freeze_inputs(cfg, spec, config_path, spec_path, quality):
    train_row, full_row = metadata(cfg['training_manifest']), metadata(cfg['full_manifest'])
    train, full = read_json(train_row['path']), read_json(full_row['path'])
    models = ensemble.select_models(train)
    sources = cfg['sources']
    ensemble.validate_sources(sources, full, cfg['original_fit_names'])
    native = {r['path']: r for r in full['source_metadata']}
    for row in sources:
        for field in ('frame', 'target'):
            stable(row[field])
            require(native.get(row[field]['path']) == row[field],
                    'source not bound to accepted native full930 manifest')
    for model in models:
        for field in ('state', 'statistics'):
            stable(model[field])
    refs = {k: metadata(cfg[k]) for k in
            ('baseline_analysis', 'historical_counts', 'historical_quality')}
    baseline = read_json(refs['baseline_analysis']['path'])
    require(set(baseline.get('populations', {})) == {'hold', 'val'}, 'full baseline populations required')
    for split in ('hold', 'val'):
        require(len(baseline['populations'][split]['arm_rows']) == 96
                and len(baseline['populations'][split]['original_controls']) == 32,
                'all128 original baseline rows per population required')
    return dict(schema=SCHEMA, run_id=cfg['run_id'], models=models, sources=sources,
        training_manifest=train_row, full_manifest=full_row, **refs,
        config=metadata(config_path), spec=metadata(spec_path),
        quality_metadata=quality.metadata, source_metadata=code_metadata(),
        original_fit_names=cfg['original_fit_names'], marker_order=list(ensemble.MARKERS),
        new_fits=0, optimizer_steps=0), baseline, full['metadata_contract']['original_controls']


def check_frozen(freeze):
    rows = [freeze[k] for k in ('training_manifest', 'full_manifest', 'baseline_analysis',
                               'historical_counts', 'historical_quality', 'config', 'spec')]
    rows += freeze['source_metadata'] + freeze['quality_metadata']['sources']
    rows += [r[k] for r in freeze['sources'] for k in ('frame', 'target')]
    rows += [r[k] for r in freeze['models'] for k in ('state', 'statistics')]
    if 'profile_acceptance' in freeze:
        rows.append(freeze['profile_acceptance'])
    for row in rows:
        stable(row)


def load_models(rows):
    models = {}
    for row in rows:
        stable(row['state']); stable(row['statistics'])
        saved = torch.load(row['state']['path'], map_location='cpu', weights_only=True)
        require(saved.get('schema') == 'query-relation-optimization-state-v1'
                and saved.get('objective_version') == 'preclip_mse_v1'
                and saved.get('original_plan_id') == original.PLAN_ID
                and saved.get('split') == 'inner'+str(row['fold'])
                and saved.get('token_statistics_file') == row['statistics']
                and all(saved.get(k) == row[k] for k in
                    ('arm', 'seed', 'fold', 'epoch', 'training_indices', 'held_indices')),
                'saved state fold/statistics/protocol identity differs')
        stats_file = read_json(row['statistics']['path'])
        require(stats_file.get('training_indices') == row['training_indices']
                and stats_file.get('fold') == row['fold']
                and stats_file.get('outer_and_inner_held_used') is False,
                'fold statistics training membership differs')
        stats = stats_file['token_statistics']
        require(stats['metadata']['training_indices'] == row['training_indices'],
                'own-fold token statistics metadata differs')
        model = original.model_for(row['arm'], row['seed'], stats, torch.device('cpu'))
        model.load_state_dict(saved['state_dict'], strict=True)
        for name, expected in (('token_mean', stats['mean']), ('token_std', stats['std'])):
            require(torch.equal(getattr(model, name).flatten(), torch.tensor(expected, dtype=torch.float32)),
                    'saved token statistics buffer differs from own fold')
        require(all(torch.isfinite(t).all().item() for t in model.state_dict().values()),
                'nonfinite saved model tensor')
        model.eval().requires_grad_(False)
        models[row['arm'], row['seed'], row['fold']] = model
        stable(row['state']); stable(row['statistics'])
    require(len(models) == 18, 'all18 fixed models required')
    return models


def read_frame(row):
    stable(row)
    with zipfile.ZipFile(row['path'], 'r') as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and all(n.endswith('.npy') for n in names),
                'duplicate or non-array native frame member')
        require(not {Path(n).stem.lower() for n in names} & {'target', 't', 'r', 'label'},
                'target-bearing frame forbidden')
        require(set(k+'.npy' for k in data.INPUT_MEMBERS) <= set(names), 'four native source members required')
        frame = {}
        for key in data.INPUT_MEMBERS:
            with archive.open(key+'.npy') as stream:
                frame[key] = data._validate_member(key, np.lib.format.read_array(stream, allow_pickle=False))
    stable(row)
    frame['donor'] = data.original_donor(frame['u'], frame['fs'])
    frame['common12'] = data.common_inputs(frame['he'], frame['u'], frame['fs'], frame['donor'])
    return frame


def generate_all(models, frame, counters=None):
    common, tokens, _, _ = original.tensors(frame, torch.device('cpu'))
    outputs = []
    for arm, seed in ensemble.GROUPS:
        residuals, solver = [], []
        for fold in ensemble.FOLDS:
            if counters is not None:
                counters['model_inference_calls'] += 1
            with torch.inference_mode():
                residual, diagnostic = models[arm, seed, fold](common, tokens, inference=True)
            array = np.array(residual.cpu().numpy()[0, 0], copy=True)
            require(array.dtype == np.float64, 'float64 inference residual required')
            residuals.append(array)
            receipt = dict(fold=fold, inference=True, **original.diagnostic_scalars(diagnostic))
            require(arm == 'P_C' or receipt['solver_dtype'] == 'float64', 'fp64 solve required')
            solver.append(receipt)
        image, clipping = ensemble.final_bytes(frame['donor'], residuals)
        image.setflags(write=False)
        outputs.append(dict(arm=arm, seed=seed, image=image, clipping=clipping, solver=solver))
        if counters is not None:
            counters['generated_outputs'] += 1
    require([(r['arm'], r['seed']) for r in outputs] == list(ensemble.GROUPS),
            'all six outputs required before target access')
    return outputs


class Phase:
    def __init__(self, output, phase, limit):
        self.root = Path(output).absolute()/phase
        require(not self.root.exists() and not self.root.is_symlink(), 'fresh exclusive phase required')
        require(type(limit) is int and limit >= 1_000_000, 'output byte bound required')
        self.root.mkdir(parents=True)
        self.limit, self.started = limit, time.monotonic()
        self.counters = dict(frame_reads=0, target_reads=0, generated_outputs=0, RGB_calls=0,
            prediction_rows=0, donor_rows=0, new_fits=0, optimizer_steps=0,
            FIT32_target_reads=0, outer8_target_reads=0, test_target_reads=0, model_inference_calls=0)
        self.write('status.json', dict(status='in_progress', phase=phase))

    def used(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def admit(self, count, *, reserve=None):
        if reserve is None:
            reserve = min(1_500_000, self.limit//4)
        require(self.used()+count+reserve <= self.limit, 'output byte bound would be exceeded; partials retained')

    def write(self, filename, value):
        raw = (json.dumps(original.plain(value), allow_nan=False, sort_keys=True, separators=(',', ':'))+'\n').encode()
        self.admit(len(raw), reserve=0 if filename in ('failure.json', 'status.json') else None)
        with (self.root/filename).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        return metadata(self.root/filename)

    def append(self, filename, value):
        raw = (json.dumps(original.plain(value), allow_nan=False, sort_keys=True, separators=(',', ':'))+'\n').encode()
        self.admit(len(raw))
        with (self.root/filename).open('ab') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())

    def event(self, event, index, source, **fields):
        self.append('access-audit.jsonl', dict(event=event, index=index,
            split=source['split'], name=source['name'], **fields))

    def commit(self, index, source, predictions, donor):
        require(len(predictions) == 6, 'all6 prediction rows required')
        compressed = {}
        for key, rows in (('predictions', predictions), ('donors', [donor])):
            raw = b''.join((json.dumps(r, allow_nan=False, separators=(',', ':'))+'\n').encode() for r in rows)
            compressed[key] = gzip.compress(raw, mtime=0)
        self.admit(sum(map(len, compressed.values()))+4096)
        receipt = dict(index=index, split=source['split'], name=source['name'], prediction_rows=6, donor_rows=1)
        for key, raw in compressed.items():
            path = self.root/(key+'.jsonl.gz')
            with path.open('ab') as stream:
                start = stream.tell(); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                receipt[key] = dict(start=start, end=stream.tell())
        self.append('commits.jsonl', receipt)
        self.counters['prediction_rows'] += 6; self.counters['donor_rows'] += 1
        self.event('image_committed', index, source)

    def finish(self, status, **fields):
        receipt = dict(status=status, counters=self.counters,
            elapsed_seconds=time.monotonic()-self.started, output_limit_bytes=self.limit,
            retained_bytes=self.used(), peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*
                (1 if sys.platform == 'darwin' else 1024), automatic_retry=False, **fields)
        temporary = self.write('terminal-status.json', receipt)
        Path(temporary['path']).replace(self.root/'status.json')
        with (self.root/'exit-code.txt').open('x') as stream:
            stream.write('0\n' if status == 'completed' else '1\n')
            stream.flush(); os.fsync(stream.fileno())
        return receipt


class HistoricalDonor:
    """Read the old literal B0 scalars once, in complete original image order."""
    def __init__(self, counts, quality, controls):
        self.controls, self.images = controls, 0
        self.stack = ExitStack()
        self.counts = self.stack.enter_context(gzip.open(counts, 'rt'))
        self.quality = self.stack.enter_context(gzip.open(quality, 'rt'))

    def check(self, donor):
        split, name = donor['split'], donor['name']
        found = {}
        for arm in self.controls:
            for marker in ensemble.MARKERS:
                row = parse_json(next(self.counts))
                require((row['split'], row['name'], row['arm'], row['marker']) == (split, name, arm, marker),
                        'historical donor count stream identity/order differs')
                if arm == 'B0': found[marker] = row['counts']
        require(found == donor['counts'], 'native donor all256 counts differ from historical B0')
        for arm in self.controls:
            row = parse_json(next(self.quality))
            require((row['split'], row['name'], row['arm']) == (split, name, arm),
                    'historical donor quality stream identity/order differs')
            if arm == 'B0':
                require(all(abs(row[k]-donor['quality'][k]) <= 1e-12 for k in ('avg_ssim', 'avg_psnr')),
                        'native donor quality differs from historical B0')
        self.images += 1

    def finish(self):
        require(next(self.counts, None) is None and next(self.quality, None) is None and self.images == 930,
                'complete historical930 donor parity required')
        return dict(status='passed', images=self.images, reference_arm='B0',
                    channels=3, thresholds=256, quality_absolute_tolerance=1e-12)

    def close(self):
        self.stack.close()


def score_one(phase, index, source, models, quality, *, synthetic):
    phase.event('frame_open', index, source, file=source['frame'])
    frame = read_frame(source['frame']); phase.counters['frame_reads'] += 1
    outputs = generate_all(models, frame, phase.counters)
    phase.event('all6_generated', index, source,
                outputs=[dict(arm=o['arm'], seed=o['seed']) for o in outputs])
    if synthetic:
        target = np.random.default_rng(20261103).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        phase.event('synthetic_target_created', index, source, real_target_opened=False)
    else:
        stable(source['target'])
        phase.event('target_open', index, source, file=source['target'], crop=source['crop'])
        phase.counters['target_reads'] += 1
        with Image.open(source['target']['path']) as image:
            require(image.size == (1024, 1024), 'native1024 target required; no resize')
            target = np.array(image.convert('RGB').crop(tuple(source['crop'])), dtype=np.uint8)
        stable(source['target'])
        phase.event('target_read', index, source)
    target.setflags(write=False)
    donor = ensemble.score_donor(frame['donor'], target, quality, split=source['split'], name=source['name'])
    predictions = []
    for output in outputs:
        predictions.append(ensemble.score_prediction(output['image'], target, frame['donor'], quality,
            split=source['split'], name=source['name'], arm=output['arm'], seed=output['seed'],
            clipping=output['clipping'], solver=output['solver']))
    phase.counters['RGB_calls'] += 7
    return predictions, donor


def run(args):
    cfg, spec = read_json(args.config), read_json(args.spec)
    validate_spec(spec)
    require(args.cpu_threads == 4, 'fixed lab CPU4 required')
    original.configure_runtime(torch.device('cpu'), args.cpu_threads)
    cv2.setNumThreads(1)
    quality = quality_for(cfg)
    freeze, baseline, controls = freeze_inputs(cfg, spec, args.config, args.spec, quality)
    if args.phase == 'evaluate':
        require('profile_acceptance' in cfg, 'independent profile acceptance required before evaluate')
        freeze['profile_acceptance'] = metadata(cfg['profile_acceptance'])
    phase = Phase(args.output, args.phase, cfg.get('output_limit_bytes', 50_000_000))
    try:
        freeze_row = phase.write('freeze.json', freeze)
        models = load_models(freeze['models'])
        pool = ensemble.Pool()
        if args.phase == 'profile':
            started = time.monotonic()
            rows, donor = score_one(phase, 0, freeze['sources'][0], models, quality, synthetic=True)
            for row in rows: pool.add(row, donor)
            pool.add(donor)
            phase.commit(0, freeze['sources'][0], rows, donor)
            require(all(np.isfinite(row['quality']['ssim_channels']).all()
                        and np.isfinite(row['quality']['psnr_channels']).all() for row in rows),
                    'nonfinite actual-full512 profile outputs')
            phase.write('profile.json', dict(schema=SCHEMA, status='passed', freeze=freeze_row,
                actual_native_source_frame=True, native_shape=[512, 512, 3], real_target_reads=0,
                synthetic_target=True, models=18, outputs=6,
                all_outputs_finite=True, per_image_seconds=time.monotonic()-started,
                estimated930_seconds=(time.monotonic()-started)*930,
                environment=original.environment(torch.device('cpu')), counters=phase.counters))
        else:
            acceptance = read_json(cfg['profile_acceptance'])
            require(acceptance.get('status') == 'passed' and acceptance.get('phase') == 'profile',
                    'passed independent profile receipt required')
            for key in ('profile', 'freeze'):
                stable(acceptance[key])
            prior_freeze = read_json(acceptance['freeze']['path'])
            require(acceptance['run_id'] == prior_freeze['run_id'] == freeze['run_id']
                    and acceptance['frozen_models'] == freeze['models']
                    and all(prior_freeze[k] == freeze[k] for k in
                        ('models', 'sources', 'source_metadata', 'quality_metadata', 'training_manifest',
                         'full_manifest', 'baseline_analysis', 'historical_counts', 'historical_quality')),
                    'profile accepted different states/sources/scorer; no evaluate')
            historical = HistoricalDonor(freeze['historical_counts']['path'],
                freeze['historical_quality']['path'], controls)
            try:
                for index, source in enumerate(freeze['sources']):
                    rows, donor = score_one(phase, index, source, models, quality, synthetic=False)
                    historical.check(donor)  # A mismatch is terminal before any dependent image.
                    pool.add(donor)
                    for row in rows: pool.add(row, donor)
                    phase.commit(index, source, rows, donor)
                parity = historical.finish()
            finally:
                historical.close()
            summary = dict(schema=SCHEMA, **pool.report(), baseline_import=baseline, donor_parity=parity)
            summary_row = phase.write('summary.json', summary)
            manifest = dict(schema=SCHEMA, run_id=cfg['run_id'], phase='evaluate', status='completed',
                freeze=freeze_row, summary=summary_row, donor_parity=parity, counters=phase.counters,
                source_metadata=freeze['source_metadata'],
                **{k: metadata(phase.root/name) for k, name in
                   (('predictions', 'predictions.jsonl.gz'), ('donors', 'donors.jsonl.gz'),
                    ('access_audit', 'access-audit.jsonl'), ('commits', 'commits.jsonl'))})
            phase.write('manifest.json', manifest)
        check_frozen(freeze)
        return phase.finish('completed', phase=args.phase)
    except BaseException as error:
        try:
            phase.write('failure.json', dict(status='failed', error_type=type(error).__name__,
                error=str(error), counters=phase.counters, partials_retained=True, automatic_retry=False))
            phase.finish('failed', phase=args.phase, error_type=type(error).__name__, error=str(error))
        except BaseException:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('profile', 'evaluate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
