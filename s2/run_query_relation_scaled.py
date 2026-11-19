"""Prospective FIT1192 four-arm runner, isolated from the frozen FIT32 code.

The phases are profile, train and evaluate. Native source inference is streamed;
only HE/U/F0 and panCK training bytes are retained in RAM. Every held image has
all eight outputs frozen before its one target read. No locked test is exposed.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import io
import json
import math
import os
from pathlib import Path
import random
import resource
import time
import traceback

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import torch

from . import query_relation_scaled_data as data
from . import query_relation_count_objective as counts
from . import query_relation_reconstruction as core
from . import query_relation_scoring as scoring
from . import query_relation_data as old_data
from . import run_query_relation_reconstruction as original
from . import run_query_relation_optimization as repaired
from . import run_query_relation_ensemble as old_eval
from .run_query_relation_budgeted import RetainedBudget

SCHEMA = 'query-relation-fit1192-v1'
STATE_SCHEMA = 'query-relation-fit1192-state-v1'
ARMS = ('E_recon', 'S_recon', 'E_count', 'S_count')
SEEDS = (202610270, 202610271)
EPOCHS = 32
N = 1192
STEPS = N * EPOCHS
GROUPS = tuple((a, s) for a in ARMS for s in SEEDS)
require = data.require
metadata = old_eval.metadata
read_json = old_eval.read_json
stable = old_eval.stable


def learning_rate(update, n=N, epochs=EPOCHS):
    require(type(update) is int and 1 <= update <= n*epochs, 'update outside complete schedule')
    if update <= n:
        return 1e-4 + 9e-4*(update-1)/(n-1)
    return 1e-5 + .5*(1e-3-1e-5)*(1+math.cos(math.pi*(update-n)/((epochs-1)*n)))


def model_for(arm, seed, device):
    require(arm in ARMS and seed in SEEDS, 'fixed arm/seed required')
    # Keep the original constructor, token_projection, and RNG order. C arms
    # bypass token standardization completely, so buffers are valid zero/one.
    return original.model_for(arm[0]+'_C', seed,
        dict(mean=[0.]*1536, std=[1.]*1536), torch.device(device))


def finite_parameters(model):
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            'nonfinite parameter; affected execution stopped')


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def resources(device):
    sync(device)
    # Linux ru_maxrss is KiB; Darwin is bytes. Native runs use Linux.
    import sys
    row = dict(peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*
               (1 if sys.platform == 'darwin' else 1024), torch_threads=torch.get_num_threads(),
               interop_threads=torch.get_num_interop_threads())
    tasks = Path('/proc/self/task')
    row['threads'] = len(list(tasks.iterdir())) if tasks.exists() else None
    if torch.device(device).type == 'cuda':
        row.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                   cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                   gpu_name=torch.cuda.get_device_name(device))
    return row


class Writer:
    """Bound each encoded write; keep complete gzip members on interruption."""
    def __init__(self, root, allowance):
        self.root = Path(root)
        self.guard = RetainedBudget(root, allowance)
        self.pending_rows = {}

    def blob(self, path, payload, *, append=False, replace=False):
        path = Path(path)
        require(not (append and replace), 'ambiguous write mode')
        require(append or replace or not path.exists(), 'preserve existing output: '+str(path))
        self.guard.check(path, len(payload), append=append)
        path.parent.mkdir(parents=True, exist_ok=True)
        destination = path.with_name(path.name+'.pending') if replace else path
        if replace:
            # Atomic recovery requires room for the old and new state together.
            require(not destination.exists(), 'preserve interrupted pending state')
            self.guard.check(destination, len(payload))
        with destination.open('ab' if append else 'xb') as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        if replace:
            os.replace(destination, path)
        return metadata(path)

    def json(self, path, value, *, replace=False):
        payload = (json.dumps(original.plain(value), allow_nan=False, sort_keys=True,
                              separators=(',', ':'))+'\n').encode()
        return self.blob(path, payload, replace=replace)

    def row(self, path, value):
        return self.rows(path, [value])

    def rows(self, path, values):
        payload = ''.join(json.dumps(original.plain(value), allow_nan=False,
            sort_keys=True, separators=(',', ':'))+'\n' for value in values).encode()
        require(bool(payload), 'nonempty scalar batch required')
        return self.blob(path, gzip.compress(payload, mtime=0), append=True)

    def buffered_row(self, path, value):
        pending = self.pending_rows.setdefault(Path(path), [])
        pending.append(value)
        if len(pending) == 64:
            self.flush(path)

    def flush(self, path=None):
        paths = [Path(path)] if path is not None else list(self.pending_rows)
        for destination in paths:
            values = self.pending_rows.get(destination, [])
            if values:
                self.rows(destination, values)
                values.clear()

    def state(self, path, value, *, replace=False):
        require(Path(path).name.endswith('.pt.gz'), 'gzip state requires explicit .pt.gz suffix')
        stream = io.BytesIO(); torch.save(value, stream)
        return self.blob(path, gzip.compress(stream.getvalue(), mtime=0), replace=replace)


def read_state(row):
    """Validate the entire gzip member and binding before returning safe state."""
    path = Path(row['path'])
    require(path.name.endswith('.pt.gz'), 'gzip state requires explicit .pt.gz suffix')
    stable(row)
    with gzip.open(path, 'rb') as handle:
        payload = handle.read()  # Reach EOF to verify the trailer/CRC before torch.load.
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    stable(row)
    return saved


def validate_spec(spec):
    expected = dict(schema=SCHEMA, arms=list(ARMS), seeds=list(SEEDS), epochs=EPOCHS,
        training_members=N, parameter_updates_per_fit=STEPS, parameter_updates=8*STEPS,
        primary_population='val', populations=dict(hold=300, val=630),
        schedule='complete32_update_warmup_cosine_v1', locked_test_access=False,
        training_precision='float32', inference_solver_precision='float64',
        target_reads=930, prediction_rows=7440, donor_rows=930)
    require(all(spec.get(k) == v for k, v in expected.items()), 'fixed FIT1192 protocol required')


def code_metadata():
    modules = (__file__, data.__file__, counts.__file__, core.__file__, scoring.__file__,
        old_data.__file__, original.__file__, repaired.__file__, old_eval.__file__,
        scoring.reference.__file__)
    return [metadata(Path(p).resolve()) for p in modules]


def source_bindings(cfg, membership, splits):
    rows = [metadata(cfg[k]) for k in ('fit_metadata', 'pilot_manifest', 'data_manifest')]
    native = cfg['native']
    rows += [metadata(native[k]) for k in ('uni_checkpoint', 'f0_checkpoint', 'manifest')]
    for split in splits:
        names = ([r['name'] for r in membership['training']] if split == 'training' else membership[split])
        directory = 'val' if split == 'val' else 'train'
        for name in names:
            for role in ('input', 'label'):
                rows.append(metadata(Path(native['data_root'])/directory/role/name))
    rows += [metadata(p) for p in cfg.get('native_source_files', [])]
    return list({r['path']: r for r in rows}.values())


class Phase:
    def __init__(self, cfg, spec, root, name, device, config_path, spec_path):
        validate_spec(spec)
        self.cfg, self.spec, self.name = cfg, spec, name
        self.device, self.root = torch.device(device), Path(root).absolute()
        self.path = self.root/name
        require(not self.path.exists(), 'phase is one-shot; retain failed output')
        self.path.mkdir(parents=True)
        self.writer = Writer(self.root, cfg['retained_budget_bytes'])
        self.membership = data.build_membership(*(read_json(cfg[k]) for k in
            ('fit_metadata', 'pilot_manifest', 'data_manifest')))
        require(read_json(cfg['data_manifest']) == read_json(cfg['native']['manifest']),
                'bound original/native train-val manifest differs')
        self.code = list({row['path']: row for row in [*code_metadata(),
            *[metadata(p) for p in cfg.get('deployed_source_files', [])]]}.values())
        self.inputs = [metadata(config_path), metadata(spec_path)]
        self.data_sources = []
        self.target_reads = 0
        self.started = time.monotonic()

    def close(self, result=None, error=None):
        self.writer.flush()
        for row in [*self.code, *self.inputs, *self.data_sources]:
            stable(row)
        report = dict(schema=SCHEMA, run_id=self.cfg['run_id'], phase=self.name,
            status='failed' if error else 'complete', source_metadata=self.code,
            input_metadata=self.inputs, data_source_metadata=self.data_sources,
            actual_target_reads=self.target_reads, wall_seconds=time.monotonic()-self.started,
            resources=resources(self.device), result=result, failure=error)
        self.writer.json(self.path/'status.json', report)
        self.writer.blob(self.path/'exit-code.txt', b'1\n' if error else b'0\n')
        return report


def add_counts(total, value):
    if total is None:
        return {k: int(v) if k == 'G' else np.asarray(v, dtype=np.int64).copy() for k, v in value.items()}
    for key in total:
        total[key] += value[key] if key == 'G' else np.asarray(value[key], dtype=np.int64)
    return total


def byte_counts(image, target, cutoff=22):
    raw = scoring.reference.counts(image, target > cutoff, scoring.tolerance_kernel())
    return {k: int(v) if k == 'G' else np.asarray(v, dtype=np.int64) for k, v in raw.items()}


def byte_panck_quality(pred, target, quality):
    # Repeating one channel invokes exactly the bound native formula and does
    # not turn the other two cached training markers into synthetic evidence.
    value = quality(np.repeat(pred[..., None], 3, 2), np.repeat(target[..., None], 3, 2))
    return dict(ssim=value['ssim_channels'][0], psnr=value['psnr_channels'][0])


def preload(phase):
    phase.data_sources = source_bindings(phase.cfg, phase.membership, ('training',))
    n, shape = N, (N, 3, 512, 512)
    arrays = [np.empty(shape, dtype=np.float32) for _ in range(3)]
    labels = np.empty((n, 512, 512), dtype=np.uint8)
    side_counts = [None, None]
    sources = data.NativeSources(phase.cfg['native'], phase.device, phase.membership)
    try:
        for row, (name, frame, loader) in enumerate(sources.items('training')):
            require(name == phase.membership['training'][row]['name'], 'training source order changed')
            for array, key in zip(arrays, ('he', 'u', 'f0')):
                array[row] = frame[key]
            target = loader(); phase.target_reads += 1
            require(target.shape == (512, 512, 3) and target.dtype == np.uint8, 'native RGB target required')
            labels[row] = target[..., 0]
            for channel in (1, 2):
                side_counts[channel-1] = add_counts(side_counts[channel-1], byte_counts(
                    frame['donor'][..., channel], target[..., channel], scoring.CUTOFFS[channel]))
            phase.writer.row(phase.path/'training-access.jsonl.gz', dict(event='target_read',
                row=row, original_index=phase.membership['training'][row]['index'], name=name))
            if row == 0 or (row+1) % 100 == 0:
                print(json.dumps(dict(event='training_source_loaded', images=row+1)), flush=True)
    finally:
        release = sources.close()
        require(release['owned_cache_entries_remaining'] == 0,
                'owned source caches remain before head training')
        phase.writer.json(phase.path/'source-release.json', release)
    del sources, frame, target
    gc.collect()
    if phase.device.type == 'cuda':
        torch.cuda.empty_cache()  # Releases only this process's unused allocator blocks.
    require(phase.target_reads == N, 'training target access count differs')
    store = data.ArrayDataset(*arrays, labels, source_indices=data.TRAINING_INDICES)
    require(store.array_bytes == data.BASE_ARRAY_BYTES, 'shared RAM byte budget differs')
    return store, [original.plain(c) for c in side_counts]


def calibrate(store, device, side_counts=None):
    """Training-only donor scales, pooled adjoint, and output-gradient lambda."""
    raw, density_rows, G, donor_panck = [], [], [], None
    for row in range(len(store)):
        item = store.tensors(row, device)
        with torch.no_grad():
            raw.append({k: float(v) for k, v in core.loss_scales(item['donor'], item['target']).items()})
            prepared = counts.prepare(item['donor'], item['target_u8'])
            density_rows.append(counts.densities(prepared)[0].cpu())
            G.append(prepared['G'][0].cpu())
            donor_panck = add_counts(donor_panck, byte_counts(item['donor_rgb'][..., 0],
                item['target_u8'].cpu().numpy()[0, 0]))
    scales = {k: max(math.fsum(v[k] for v in raw)/len(raw), core.SCALE_FLOOR) for k in raw[0]}
    table = counts.LaggedCounts(len(store)); table.refresh(torch.stack(density_rows), torch.stack(G))
    initial = table.statistics()
    A, threshold = initial['adjoint'].detach().to(device), initial['threshold']
    rec_sq, aux_sq = [], []
    for row in range(len(store)):
        item = store.tensors(row, device)
        z = item['donor'].detach().clone().requires_grad_(True)
        base = repaired.objective(z, item['target'], item['donor'], scales)['train_total'].mean()
        auxiliary = (counts.densities(counts.prepare(z, item['target_u8']), [threshold])[0, 0]*A).sum()
        bg, = torch.autograd.grad(base, z, retain_graph=True)
        ag, = torch.autograd.grad(auxiliary, z)
        rec_sq.append(float(bg.to(torch.float64).square().sum()))
        aux_sq.append(float(ag.to(torch.float64).square().sum()))
    sums = math.fsum(rec_sq), math.fsum(aux_sq)
    lam = counts.calibration_lambda(*sums)
    result = dict(schema=SCHEMA, training_only=True, training_members=len(store),
        training_indices=list(store.source_indices), loss_scales=scales, lambda_count=lam,
        reconstruction_gradient_sq=sums[0], auxiliary_gradient_sq=sums[1],
        donor_soft_phi=float(initial['phi']), donor_threshold=threshold,
        donor_adjoint=initial['adjoint'].tolist(), output_gradient_coordinate='normalized_z',
        auxiliary_byte_chain=255, additional_N_factor=False, additional_marker_factor=False)
    if side_counts is not None:
        native = [original.plain(donor_panck), *side_counts]
        result['donor_train_counts'] = dict(zip(scoring.MARKERS, native))
        result['donor_train_thresholds'] = [scoring.endpoints(c, cutoff)['ODS_threshold']
            for c, cutoff in zip(native, scoring.CUTOFFS)]
    return result


def final_bytes(donor, residual):
    require(residual.dtype == np.float64 and residual.shape == donor.shape[:2],
            'float64 inference residual required')
    z = donor[..., 0].astype(np.float64)/255 + residual
    clipped = np.clip(z, 0, 1)
    image = donor.copy(); image[..., 0] = np.rint(255*clipped).astype(np.uint8)
    return image, dict(pixels=int(z.size), below_zero=int((z < 0).sum()), above_one=int((z > 1).sum()),
        preclip_mean=float(z.mean()), postclip_mean=float(clipped.mean()),
        rounding_mean_change=float(image[..., 0].mean()/255-clipped.mean()))


def population_pass(model, store, seed, epoch, device, quality=None, *, inference=False):
    """Fixed-weight, same-D4 population read for refresh and true objective."""
    was_training = model.training; model.eval()
    rows, gs, native, qualities = [], [], None, []
    with torch.no_grad():
        for row, source_index in enumerate(store.source_indices):
            code = data.seeded_d4(seed, epoch, source_index) if epoch else 0
            item = store.tensors(row, device, code)
            residual, _ = model(item['common'], item['tokens'], inference=inference)
            z = item['donor'].to(residual.dtype)+residual
            prepared = counts.prepare(z, item['target_u8'])
            rows.append(counts.densities(prepared)[0].cpu()); gs.append(prepared['G'][0].cpu())
            if quality is not None:
                prediction = torch.round(z.clamp(0, 1)*255).to(torch.uint8).cpu().numpy()[0, 0]
                target = item['target_u8'].cpu().numpy()[0, 0]
                native = add_counts(native, byte_counts(prediction, target))
                qualities.append(byte_panck_quality(prediction, target, quality))
    model.train(was_training)
    table = counts.LaggedCounts(len(store)); table.refresh(torch.stack(rows), torch.stack(gs))
    pooled = table.statistics()
    result = dict(soft_phi=float(pooled['phi']), soft_threshold=pooled['threshold'],
                  soft_ODS=float(pooled['scores'].max()))
    if quality is not None:
        result.update(byte_counts=original.plain(native), quality={k: math.fsum(r[k] for r in qualities)/len(qualities)
            for k in ('ssim', 'psnr')})
    return table, result


def capture_state(model, optimizer, arm, seed, epoch, position, updates, calibration,
                  table, age, device):
    return dict(schema=STATE_SCHEMA, arm=arm, seed=seed, epoch=epoch, position=position,
        parameter_updates=updates, schedule_total_epochs=EPOCHS,
        training_indices=list(data.TRAINING_INDICES), calibration=calibration,
        model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        optimizer=optimizer.state_dict(), lagged_counts=None if table is None else table.state_dict(),
        table_last_update=None if age is None else age.tolist(), **repaired.rng_state(torch.device(device)))


def restore_state(saved, model, optimizer, arm, seed, calibration, device):
    require(saved.get('schema') == STATE_SCHEMA and saved.get('arm') == arm and saved.get('seed') == seed
        and saved.get('training_indices') == list(data.TRAINING_INDICES)
        and saved.get('calibration') == calibration and saved.get('schedule_total_epochs') == EPOCHS,
        'recovery identity/protocol/calibration changed')
    epoch, position, updates = (saved[k] for k in ('epoch', 'position', 'parameter_updates'))
    require(type(epoch) is int and 1 <= epoch <= EPOCHS and type(position) is int and 0 <= position <= N
            and updates == (epoch-1)*N+position, 'recovery position inconsistent')
    model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng_state'])
    state = saved['numpy_rng_state']
    np.random.set_state((state['bit_generator'], np.asarray(state['keys'], np.uint32),
                        state['position'], state['has_gauss'], state['cached_gaussian']))
    random.setstate(saved['python_rng_state'])
    if torch.device(device).type == 'cuda':
        torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
    table = None if saved['lagged_counts'] is None else counts.LaggedCounts.from_state_dict(saved['lagged_counts'])
    age = None if saved['table_last_update'] is None else np.asarray(saved['table_last_update'], dtype=np.int64)
    require(arm.endswith('_count') == (table is not None), 'recovery count table missing/unexpected')
    if age is not None:
        require(age.shape == (N,) and bool(((age >= (epoch-1)*N) & (age <= updates)).all()), 'recovery table age invalid')
    finite_parameters(model)
    return table, age


def fit_one(phase, store, calibration, side_counts, quality, arm, seed):
    path = phase.path/'fits'/arm/str(seed); path.mkdir(parents=True)
    model = model_for(arm, seed, phase.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .999),
                                 weight_decay=1e-4, eps=1e-8)
    scales, lam = calibration['loss_scales'], calibration['lambda_count']
    is_count = arm.endswith('_count'); steps = 0
    source_to_row = {s: i for i, s in enumerate(store.source_indices)}
    started = time.monotonic()
    for epoch in range(1, EPOCHS+1):
        table, _ = population_pass(model, store, seed, epoch, phase.device) if is_count else (None, None)
        age = np.full(len(store), steps, dtype=np.int64) if is_count else None
        order = data.paired_order(store.source_indices, seed, epoch)
        for position, source_index in enumerate(order, 1):
            row = source_to_row[source_index]; code = data.seeded_d4(seed, epoch, source_index)
            item = store.tensors(row, phase.device, code)
            optimizer.zero_grad(set_to_none=True)
            rate = learning_rate(steps+1)
            for group in optimizer.param_groups: group['lr'] = rate
            residual, diagnostic = model(item['common'], item['tokens'])
            z = item['donor']+residual
            base = repaired.objective(z, item['target'], item['donor'], scales)['train_total'].mean()
            auxiliary = z.new_zeros((), dtype=torch.float64); phi, threshold = None, None
            if is_count:
                with torch.no_grad():
                    current = counts.densities(counts.prepare(z.detach(), item['target_u8']))[0].cpu()
                    table.replace(row, current); age[row] = steps
                    pooled = table.statistics(); threshold = pooled['threshold']; phi = float(pooled['phi'])
                    A = pooled['adjoint'].detach().to(phase.device)
                selected = counts.densities(counts.prepare(z, item['target_u8']), [threshold])[0, 0]
                auxiliary = (A*selected).sum()
            total = base + (lam*auxiliary if is_count else 0)
            require(bool(torch.isfinite(total)), 'nonfinite training loss; stop affected execution')
            total.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients), 'nonfinite/absent parameter gradient')
            norm = math.sqrt(math.fsum(float(g.detach().to(torch.float64).square().sum()) for g in gradients))
            optimizer.step(); finite_parameters(model); steps += 1
            row_value = dict(arm=arm, seed=seed, epoch=epoch, update_index=steps,
                original_index=source_index, name=phase.membership['training'][row]['name'],
                D4_code=code, learning_rate=rate,
                train_total=float(total.detach()), base=float(base.detach()), aux=float(auxiliary.detach()),
                soft_phi=phi, threshold=threshold, grad_norm=norm, parameter_finite=True,
                solver=original.diagnostic_scalars(diagnostic))
            phase.writer.buffered_row(path/'updates.jsonl.gz', row_value)
            if steps == 1:
                phase.writer.flush(path/'updates.jsonl.gz')
                print(json.dumps(dict(event='first_finite_update', **row_value)), flush=True)
            if position % 64 == 0:
                phase.writer.flush(path/'updates.jsonl.gz')
                phase.writer.state(path/'recovery.pt.gz', capture_state(model, optimizer, arm, seed,
                    epoch, position, steps, calibration, table, age, phase.device), replace=True)
        phase.writer.flush(path/'updates.jsonl.gz')
        exact_table, end = population_pass(model, store, seed, epoch, phase.device, quality)
        end.update(epoch=epoch, parameter_updates=steps, arm=arm, seed=seed,
            table_max_age_updates=int((steps-age).max()) if age is not None else 0,
            table_mean_age_updates=float((steps-age).mean()) if age is not None else 0.,
            table_max_abs_discrepancy=float((exact_table.rows-table.rows).abs().max()) if table is not None else 0.)
        phase.writer.row(path/'trajectory.jsonl.gz', end)
        phase.writer.state(path/'recovery.pt.gz', capture_state(model, optimizer, arm, seed,
            epoch, len(store), steps, calibration, table, age, phase.device), replace=True)
        print(json.dumps(dict(event='epoch_complete', arm=arm, seed=seed, epoch=epoch,
                              parameter_updates=steps, soft_phi=end['soft_phi'])), flush=True)
    # Training thresholds use unaugmented final inference, after the final update.
    _, final = population_pass(model, store, seed, 0, phase.device, quality, inference=True)
    native = [final['byte_counts'], *side_counts]
    thresholds = [scoring.endpoints(c, cutoff)['ODS_threshold'] for c, cutoff in zip(native, scoring.CUTOFFS)]
    final_state = capture_state(model, optimizer, arm, seed, EPOCHS, N, steps,
                                calibration, table, age, phase.device)
    final_state['train_thresholds'] = thresholds
    state_row = phase.writer.state(path/'final.pt.gz', final_state)
    fit = dict(arm=arm, seed=seed, epochs=EPOCHS, parameter_updates=steps,
        training_indices=list(data.TRAINING_INDICES), state=state_row,
        updates=metadata(path/'updates.jsonl.gz'), trajectory=metadata(path/'trajectory.jsonl.gz'),
        train_thresholds=thresholds, training_final=final, side_marker_train_counts=side_counts,
        final_train_counts={m: c for m, c in zip(scoring.MARKERS, native)},
        schedule_total_epochs=EPOCHS, parameter_counts=model.parameter_counts(),
        optimizer_state_retained=True, RNG_state_retained=True, wall_seconds=time.monotonic()-started)
    phase.writer.json(path/'fit.json', fit)
    del optimizer, model, final_state, table, exact_table
    gc.collect()
    return fit


def train(phase):
    profile_row = metadata(phase.cfg['profile_acceptance'])
    acceptance = read_json(profile_row['path'])
    require(acceptance.get('passed') is True and acceptance.get('schema') == SCHEMA,
            'accepted native full512 profile required')
    report_row = acceptance['profile_report']; stable(report_row)
    profile_report = read_json(report_row['path'])
    require(profile_report.get('status') == 'complete'
            and profile_report.get('actual_target_reads') == 0
            and profile_report.get('native_shape') == [512, 512]
            and profile_report.get('all_four_paths') is True, 'wrong native profile evidence')
    for row in profile_report['source_metadata']: stable(row)
    profiled = {r['path']: r for r in profile_report['source_metadata']}
    require(all(profiled.get(row['path']) == row for row in phase.code),
            'training code differs from accepted full512 profile')
    store, side_counts = preload(phase)
    calibration = calibrate(store, phase.device, side_counts)
    cal_row = phase.writer.json(phase.path/'calibration.json', calibration)
    quality = old_eval.quality_for(phase.cfg)
    fits = [fit_one(phase, store, calibration, side_counts, quality, arm, seed) for arm, seed in GROUPS]
    result = dict(schema=SCHEMA, phase='train', status='complete', run_id=phase.cfg['run_id'],
        membership=phase.membership, calibration=cal_row, fits=fits, parameter_updates=8*STEPS,
        membership_sources={key: metadata(phase.cfg[key]) for key in
            ('fit_metadata', 'pilot_manifest', 'data_manifest')},
        actual_training_target_reads=phase.target_reads, actual_held_target_reads=0,
        source_metadata=phase.code+phase.data_sources, profile_acceptance=profile_row,
        source_release=metadata(phase.path/'source-release.json'),
        training_access=metadata(phase.path/'training-access.jsonl.gz'),
        checkpoint_selection=False, method_claim=False)
    stable(profile_row)
    return dict(manifest=phase.writer.json(phase.path/'manifest.json', result), fits=8, parameter_updates=8*STEPS)


def score_bytes(image, target, donor, quality, clipping=None):
    measured = {}
    def tracked(x, y):
        measured.update(quality(x, y)); return measured
    value = scoring.score_image(image, target, donor, tracked, clipping=clipping)
    value['quality']['rgb_psnr'] = measured['rgb_psnr']
    # Source identity lives once in the manifest, not in each per-image row.
    value.pop('quality_metadata', None)
    return value


def generate_outputs(models, frame, device):
    require([(fit['arm'], fit['seed']) for fit, _ in models] == list(GROUPS),
            'exact eight model identities required before target access')
    common = torch.from_numpy(frame['common12']).unsqueeze(0).to(device)
    tokens = torch.zeros((1, 1536, 32, 32), device=device)
    generated = []
    for fit, model in models:
        with torch.inference_mode():
            residual, diagnostic = model(common, tokens, inference=True)
        image, clip = final_bytes(frame['donor'], residual.cpu().numpy()[0, 0])
        image.setflags(write=False)
        generated.append((fit, image, clip, dict(inference=True,
            **original.diagnostic_scalars(diagnostic))))
    return generated


def evaluate(phase):
    train_row = metadata(phase.cfg['training_manifest']); training = read_json(train_row['path'])
    require(training.get('schema') == SCHEMA and training.get('status') == 'complete'
            and training.get('membership') == phase.membership and len(training.get('fits', [])) == 8
            and training.get('parameter_updates') == 8*STEPS, 'complete8 fixed fits required')
    require([(r['arm'], r['seed']) for r in training['fits']] == list(GROUPS), 'fixed eight models required')
    cal_row = training['calibration']; stable(cal_row); calibration = read_json(cal_row['path'])
    models = []
    for fit in training['fits']:
        saved = read_state(fit['state'])
        require(saved.get('schema') == STATE_SCHEMA and saved.get('calibration') == calibration
            and saved.get('epoch') == EPOCHS and saved.get('parameter_updates') == STEPS
            and saved.get('position') == N and saved.get('arm') == fit['arm']
            and saved.get('seed') == fit['seed']
            and saved.get('training_indices') == list(data.TRAINING_INDICES)
            and saved.get('train_thresholds') == fit['train_thresholds'], 'frozen final state differs')
        model = model_for(fit['arm'], fit['seed'], phase.device)
        model.load_state_dict(saved['model'], strict=True); finite_parameters(model)
        model.eval().requires_grad_(False)
        models.append((fit, model))
    del saved
    frozen_models = [dict(arm=f['arm'], seed=f['seed'], state=f['state'], train_thresholds=f['train_thresholds'])
                     for f, _ in models]
    phase.writer.json(phase.path/'frozen-models.json', dict(models=frozen_models,
        calibration=cal_row, training_manifest=train_row, target_reads=0))
    phase.data_sources = source_bindings(phase.cfg, phase.membership, ('hold', 'val'))
    quality = old_eval.quality_for(phase.cfg)
    sources = data.NativeSources(phase.cfg['native'], phase.device, phase.membership)
    index = 0
    try:
        for split in ('hold', 'val'):
            for name, frame, loader in sources.items(split):
                generated = generate_outputs(models, frame, phase.device)
                require(len(generated) == 8, 'all eight outputs before target required')
                identity = dict(index=index, split=split, name=name)
                phase.writer.row(phase.path/'access.jsonl.gz', dict(event='generated', outputs=8, **identity))
                target = loader(); phase.target_reads += 1
                phase.writer.row(phase.path/'access.jsonl.gz', dict(event='target_read', **identity))
                donor_score = score_bytes(frame['donor'], target, frame['donor'], quality)
                phase.writer.row(phase.path/'donors.jsonl.gz', dict(**identity, scores=donor_score))
                scored_predictions = []
                for fit, image, clip, solver in generated:
                    value = score_bytes(image, target, frame['donor'], quality, clip)
                    scored_predictions.append(dict(**identity, arm=fit['arm'], seed=fit['seed'],
                                                   scores=value, solver=solver))
                phase.writer.rows(phase.path/'predictions.jsonl.gz', scored_predictions)
                index += 1
                if index == 1 or index % 50 == 0:
                    print(json.dumps(dict(event='held_scored', images=index, predictions=index*8)), flush=True)
    finally:
        sources.close()
    require(index == phase.target_reads == 930, 'exact930 once target reads required')
    for fit, _ in models: stable(fit['state'])
    stable(train_row); stable(cal_row)
    result = dict(schema=SCHEMA, phase='evaluate', status='complete', run_id=phase.cfg['run_id'],
        training_manifest=train_row, membership=phase.membership, models=frozen_models,
        predictions=metadata(phase.path/'predictions.jsonl.gz'), donors=metadata(phase.path/'donors.jsonl.gz'),
        access=metadata(phase.path/'access.jsonl.gz'), source_metadata=phase.code+phase.data_sources,
        actual_target_reads=930, generated_outputs=7440, RGB_calls=8370,
        baseline_analysis=metadata(phase.cfg['baseline_analysis']), quality_metadata=quality.metadata,
        historical_donors=metadata(phase.cfg['historical_donors']),
        method_claim=False, locked_test_access=False)
    return dict(manifest=phase.writer.json(phase.path/'manifest.json', result), predictions=7440, target_reads=930)


def profile(phase):
    """Native sources; unopened labels; full512 nonzero heads and synthetic y."""
    parity = phase.cfg['profile_parity']
    require(len(parity) >= 1 and all(r['name'] in {x['name'] for x in phase.membership['training']} for r in parity),
            'profile uses admissible old FIT32 training members only')
    sync(phase.device); source_started = time.monotonic()
    sources = data.NativeSources(phase.cfg['native'], phase.device, phase.membership)
    phase.data_sources = [metadata(phase.cfg['native'][k]) for k in ('uni_checkpoint', 'f0_checkpoint')]
    phase.data_sources += [metadata(p) for p in phase.cfg.get('native_source_files', [])]
    source_rows, frames = [], []
    try:
        for binding, (name, frame, _unopened) in zip(parity, sources.items('training', [r['name'] for r in parity])):
            reference_row = metadata(binding['frame']); old = old_eval.read_frame(reference_row)
            phase.data_sources += [reference_row, metadata(Path(phase.cfg['native']['data_root'])/
                'train'/'input'/name)]
            require(name == binding['name'], 'profile source order differs')
            checks = {k: bool(np.array_equal(frame[k], old[k])) for k in ('he', 'u', 'donor', 'common12')}
            checks['f0'] = bool(np.array_equal(frame['f0'], old['fs'][0]))
            require(all(checks.values()), 'native source/donor/common parity failed')
            source_rows.append(dict(name=name, reference=reference_row, checks=checks))
            frames.append((frame, old['tokens']))
    finally:
        release = sources.close()
        require(release['owned_cache_entries_remaining'] == 0,
                'owned source cache references remain after close')
    source_resources = resources(phase.device)
    source_seconds = time.monotonic()-source_started
    del sources; gc.collect()
    if phase.device.type == 'cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(phase.device)
    frame, real_tokens = frames[0]
    y = np.random.default_rng(193).integers(0, 256, (1, 512, 512), dtype=np.uint8)
    store = data.ArrayDataset(*(np.ascontiguousarray(frame[k][None]) for k in ('he', 'u', 'f0')),
                             y, source_indices=[1], synthetic=True)
    item = store.tensors(0, phase.device)
    real = torch.from_numpy(np.ascontiguousarray(old_data.token_grid(real_tokens)))[None].to(phase.device, torch.float32)
    scale = {k: max(float(v), core.SCALE_FLOOR) for k, v in core.loss_scales(item['donor'], item['target']).items()}
    rows = []
    for arm in ARMS:
        model = model_for(arm, SEEDS[0], phase.device).train()
        # Profile the nonzero path, including the screened solver and its VJP.
        head = model.edge_head if model.is_edge else model.unary_head
        with torch.no_grad(): head[-1].weight.normal_(0, .001); head[-1].bias.fill_(.001)
        with torch.no_grad():
            first, _ = model(item['common'], item['tokens'])
            second, _ = model(item['common'], real)
        require(torch.equal(first, second), 'C arm real/zero token equivalence failed')
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), weight_decay=1e-4, eps=1e-8)
        sync(phase.device); started = time.monotonic()
        residual, diagnostic = model(item['common'], item['tokens']); z = item['donor']+residual
        base = repaired.objective(z, item['target'], item['donor'], scale)['train_total'].mean()
        sweep_started = time.monotonic()
        with torch.no_grad():
            prepared = counts.prepare(z.detach(), item['target_u8'])
            density = counts.densities(prepared)
            pooled = counts.pool(density[0], prepared['G'][0])
        sync(phase.device); sweep_seconds = time.monotonic()-sweep_started
        aux = (counts.densities(counts.prepare(z, item['target_u8']), [pooled['threshold']])[0, 0]
               *pooled['adjoint'].detach()).sum()
        total = base+(aux if arm.endswith('_count') else 0)
        total.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        require(all(bool(torch.isfinite(g).all()) for g in gradients), 'nonfinite native profile gradient')
        before = head[-1].weight.detach().clone(); optimizer.step(); finite_parameters(model)
        require(not torch.equal(before, head[-1].weight), 'profile update made no head change')
        sync(phase.device); update_seconds = time.monotonic()-started
        started = time.monotonic()
        with torch.inference_mode(): infer, infer_diag = model(item['common'], item['tokens'], inference=True)
        sync(phase.device); inference_seconds = time.monotonic()-started
        require(infer.dtype == torch.float64 and infer_diag['solver_dtype'] == 'float64', 'native inference must use fp64 solver')
        rows.append(dict(arm=arm, real_zero_tokens_equal=True, finite_loss=float(total.detach()),
            finite_gradients=True, parameters_changed=True, count_dtype=str(density.dtype),
            solver=original.diagnostic_scalars(diagnostic), inference=original.diagnostic_scalars(infer_diag),
            update_seconds=update_seconds, count_sweep_seconds=sweep_seconds,
            inference_seconds=inference_seconds, resources=resources(phase.device)))
        del optimizer, model, total, base, aux, z, residual, infer, gradients
    result = dict(schema=SCHEMA, phase='profile', status='complete', run_id=phase.cfg['run_id'],
        source_parity=source_rows, source_resources=source_resources,
        source_load_and_inference_seconds=source_seconds, paths=rows,
        source_release=release, head_resources=resources(phase.device),
        native_shape=[512, 512], synthetic_targets=True, actual_target_reads=0,
        source_metadata=phase.code+phase.data_sources, all_four_paths=True)
    return dict(report=phase.writer.json(phase.path/'profile.json', result), actual_target_reads=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('profile', 'train', 'evaluate'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    cfg, spec = read_json(args.config), read_json(args.spec)
    require(cfg['cpu_threads'] in (1, 2, 4), 'bounded explicit CPU threads required')
    torch.set_num_threads(cfg['cpu_threads']); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    phase = Phase(cfg, spec, args.output, args.phase, args.device, args.config, args.spec)
    try:
        result = globals()[args.phase](phase)
    except Exception:
        failure = traceback.format_exc()
        phase.close(error=failure); print(failure, flush=True); return 1
    phase.close(result=result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
