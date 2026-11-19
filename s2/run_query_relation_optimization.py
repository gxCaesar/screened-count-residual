"""Training-only hard-clamp diagnosis and fixed32-epoch objective intervention.

This executor imports the original frozen model, input transforms, initializers,
data order, schedule and clipped validation loss. It cannot generate RGB outputs
or read any original outer frame/target. Each phase is one-shot. Diagnosis must
pass on actual saved weights before any fit can start. No automatic retry,
selection, scoring or continuation is performed here.

CLI: python -m s2.run_query_relation_optimization --config CONFIG.json
     --output NEW_ROOT --phase diagnose|train|self-test [--device cpu|cuda]

Native config fields are run_id, data (the original data object),
training_manifest (the complete original manifest path), frozen_metadata (exact
regular-file metadata rows), retained_budget_bytes=120000000 and cpu_threads=4.
The self-test phase needs only run_id and the retained budget. The train phase
uses OUTPUT/diagnose/diagnosis.json, or an explicit diagnosis metadata row.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import io
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from . import query_relation_data as data
from . import query_relation_reconstruction as core
from . import run_query_relation_reconstruction as original
from . import run_query_relation_budgeted as budgeted
from .run_query_relation_budgeted import RetainedBudget


OBJECTIVE_VERSION = 'preclip_mse_v1'
SCHEMA = 'query-relation-optimization-v1'
STATE_SCHEMA = 'query-relation-optimization-state-v1'
EPOCHS = 32
VALIDATION_EPOCHS = (0, 1, 2, 4, 8, 16, 32)
RETAINED_BUDGET_BYTES = 120000000
ARMS, SEEDS, TRAINING = original.ARMS, original.SEEDS, original.TRAINING
require = original.require


def code_metadata():
    return [*original.source_metadata(), original.file_metadata(__file__),
            original.file_metadata(budgeted.__file__)]


class Writer:
    """Check the entire retained tree before each actual encoded write."""
    def __init__(self, root, allowance):
        require(allowance == RETAINED_BUDGET_BYTES, 'fixed120MB retained allowance required')
        self.guard = RetainedBudget(root, allowance)

    def blob(self, path, payload, *, append=False):
        path = Path(path)
        require(append or not path.exists(), 'one-shot output already exists: '+str(path))
        self.guard.check(path, len(payload), append=append)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('ab' if append else 'xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return original.file_metadata(path)

    def json(self, path, value, *, compressed=False):
        blob = (json.dumps(original.plain(value), sort_keys=True, allow_nan=False,
                           separators=(',', ':'))+'\n').encode()
        return self.blob(path, gzip.compress(blob, mtime=0) if compressed else blob)

    def row(self, path, value):
        # Concatenated gzip members are standard gzip. Each append is bounded
        # before touching disk, including compression headers/trailers.
        blob = (json.dumps(original.plain(value), sort_keys=True, allow_nan=False,
                           separators=(',', ':'))+'\n').encode()
        return self.blob(path, gzip.compress(blob, mtime=0), append=True)

    def state(self, path, value):
        stream = io.BytesIO()
        torch.save(value, stream)
        return self.blob(path, stream.getvalue())


class TrainingOnlyDataset(data.QueryRelationDataset):
    """Refuse outer INPUTS as well as targets before the audited physical open."""
    def _read(self, index, member, purpose, freeze_receipt=None):
        require(type(index) is int and index in TRAINING,
                'optimization reader refuses all outer frames/tokens/targets')
        require(freeze_receipt is None and purpose in ('training', 'inference', 'token_statistics'),
                'optimization reader forbids scoring/held-data purpose')
        return super()._read(index, member, purpose, freeze_receipt)


def clipping_receipt(z):
    x = z.detach()
    require(bool(torch.isfinite(x).all()), 'nonfinite preclip prediction')
    below, above = x < 0, x > 1
    inside = ~(below | above)
    clipped = torch.clamp(x, 0, 1)
    zero = clipped == 0
    quantized_zero = torch.round(clipped*255) == 0
    pixels = x.numel()
    return dict(pixels=pixels, below_zero=int(below.sum()), above_one=int(above.sum()),
                inside=int(inside.sum()), active=int(inside.sum()),
                below_zero_fraction=float(below.sum())/pixels,
                above_one_fraction=float(above.sum())/pixels,
                active_fraction=float(inside.sum())/pixels,
                fully_saturated=not bool(inside.any()), zero_pixels=int(zero.sum()),
                all_zero_output=bool(zero.all()), quantized_zero_pixels=int(quantized_zero.sum()),
                all_zero_bytes=bool(quantized_zero.all()), z_min=float(x.min()), z_max=float(x.max()),
                z_mean=float(x.to(torch.float64).mean()))


def gradient_receipt(value):
    value = value.detach()
    finite = bool(torch.isfinite(value).all())
    require(finite, 'nonfinite gradient; affected execution stopped')
    return dict(l2=float(value.to(torch.float64).square().sum().sqrt()),
                max_abs=float(value.abs().max()) if value.numel() else 0.,
                nonzero=int(torch.count_nonzero(value)), finite=finite, elements=value.numel())


def objective(z, target, donor, scales):
    """The sole intervention: substitute preclip MSE for clipped MSE.

    Algebraically Lnew=Lold(clamp(z))-MSE(clamp(z),y)/s0+MSE(z,y)/s0.
    Summing the retained SSIM/edge terms directly avoids subtractive cancellation.
    The original helper still defines every clipped component and denominator.
    """
    require(z.shape == target.shape and bool(torch.isfinite(z).all()),
            'finite matching preclip prediction required')
    losses = core.normalize_loss(torch.clamp(z, 0, 1), target, donor, scales)
    scale = torch.as_tensor(scales['mse'], device=z.device, dtype=z.dtype).clamp_min(core.SCALE_FLOOR)
    preclip_mse = (z-target).square().mean(dim=(1, 2, 3))/scale
    train_total = preclip_mse + losses['ssim'] + losses['edge']
    return dict(losses=losses, preclip_mse=preclip_mse, train_total=train_total,
                clipped_image_loss=losses['image'])


def mse_gradient(z, target, scales):
    scale = torch.as_tensor(scales['mse'], device=z.device, dtype=z.dtype).clamp_min(core.SCALE_FLOOR)
    return 2*(z.detach()-target.detach())/(z.numel()*scale)


def saturated_identity(z, target, scales, actual_gradient):
    """Check the derivative of the ACTUAL optimized scalar outside the clamp."""
    mask = (z.detach() < 0) | (z.detach() > 1)
    expected = mse_gradient(z, target, scales)
    error = (actual_gradient.detach()-expected).masked_select(mask)
    expected_masked = expected.masked_select(mask)
    actual_masked = actual_gradient.detach().masked_select(mask)
    maximum = float(error.abs().max()) if error.numel() else 0.
    largest = float(expected_masked.abs().max()) if error.numel() else 0.
    tolerance = 1e-10 + 2e-5*largest
    finite = bool(torch.isfinite(actual_gradient).all())
    return dict(pixels=int(mask.sum()), max_abs_error=maximum, tolerance=tolerance,
                expected_nonzero=int(torch.count_nonzero(expected_masked)),
                actual_nonzero=int(torch.count_nonzero(actual_masked)),
                expected_l2=float(expected_masked.to(torch.float64).norm()),
                actual_l2=float(actual_masked.to(torch.float64).norm()),
                passed=finite and maximum <= tolerance, finite=finite)


def expected_parameter_status(model, name):
    if not model.uses_direct_tokens and name.startswith('token_projection.'):
        return 'unused_C_token_projection'
    if model.is_edge and name == 'edge_head.2.bias':
        return 'cancelled_antisymmetric_bias'
    if model.arm.startswith(('S_', 'M_')) and name == 'unary_head.2.bias':
        return 'cancelled_centered_or_laplacian_bias'
    return 'active'


def parameter_group_receipts(model, gradients=None):
    named = list(model.named_parameters())
    gradients = [p.grad for _, p in named] if gradients is None else gradients
    require(len(named) == len(gradients), 'complete parameter-gradient list required')
    groups = {}
    for (name, param), gradient in zip(named, gradients):
        key = name.split('.')[0]
        group = groups.setdefault(key, dict(parameters=0, elements=0, none_parameters=[],
                 zero_parameters=[], expected_unused_parameters=[], cancelled_biases=[],
                 finite=True, squared_l2=0., max_abs=0.))
        group['parameters'] += 1
        group['elements'] += param.numel()
        status = expected_parameter_status(model, name)
        if status == 'unused_C_token_projection':
            group['expected_unused_parameters'].append(name)
        elif status != 'active':
            group['cancelled_biases'].append(dict(name=name, reason=status))
        if gradient is None:
            group['none_parameters'].append(name)
            require(status == 'unused_C_token_projection', 'unexpected missing gradient: '+name)
            continue
        item = gradient_receipt(gradient)
        group['squared_l2'] += item['l2']**2
        group['max_abs'] = max(group['max_abs'], item['max_abs'])
        if item['nonzero'] == 0:
            group['zero_parameters'].append(name)
    for group in groups.values():
        group['l2'] = math.sqrt(group.pop('squared_l2'))
    return groups


def diagnosis_row(model, frame, target, checkpoint, arm, seed, epoch, index, scales, device,
                  model_label, *, synthetic=False):
    # eval() prevents checkpoint recomputation during two backwards while using
    # exactly the same mathematical model as original inference/validation.
    model.eval()
    common, tokens, donor, y = original.tensors(frame, device, target, arm=arm)
    residual, diagnostic = model(common, tokens)
    z = donor+residual
    terms = objective(z, y, donor, scales)
    old = terms['losses']['total'].mean()
    new = terms['train_total'].mean()
    old_z, = torch.autograd.grad(old, z, retain_graph=True)
    new_z, = torch.autograd.grad(new, z, retain_graph=True)
    component = dict(clipped_mse=terms['losses']['mse'], raw_mse=terms['preclip_mse'],
                     ssim=terms['losses']['ssim'], edge1=terms['losses']['edge1'],
                     edge16=terms['losses']['edge16'], edge=terms['losses']['edge'])
    term_receipts = {}
    for key, value in component.items():
        gradient, = torch.autograd.grad(value.mean(), z, retain_graph=True)
        term_receipts[key] = gradient_receipt(gradient)
    parameters = tuple(model.parameters())
    old_parameters = torch.autograd.grad(old, parameters, allow_unused=True, retain_graph=True)
    old_groups = parameter_group_receipts(model, old_parameters)
    del old_parameters
    new_parameters = torch.autograd.grad(new, parameters, allow_unused=True)
    new_groups = parameter_group_receipts(model, new_parameters)
    identity = saturated_identity(z, y, scales, new_z)
    require(identity['passed'], 'actual optimized loss violates saturated-gradient identity')
    return dict(arm=arm, seed=seed, epoch=epoch, original_index=index, model_label=model_label,
                checkpoint=checkpoint, objective_version=OBJECTIVE_VERSION,
                native_saved_weights=not synthetic, synthetic=synthetic, optimizer_steps=0,
                losses=original.scalar_loss(terms['losses']),
                train_total=float(new.detach()), preclip_mse=float(terms['preclip_mse'].mean().detach()),
                clipped_image_loss=float(terms['clipped_image_loss'].mean().detach()),
                training_mse_uses_clamp=False, clipping=clipping_receipt(z),
                old_grad_z=gradient_receipt(old_z), new_grad_z=gradient_receipt(new_z),
                term_grad_z=term_receipts, old_parameter_groups=old_groups,
                new_parameter_groups=new_groups, saturated_identity=identity,
                diagnostics=original.diagnostic_scalars(diagnostic))


def load_original(cfg):
    """Resolve immutable sources/statistics, without opening array members."""
    frozen = cfg.get('frozen_metadata')
    require(isinstance(frozen, list) and frozen, 'nonempty frozen metadata required')
    for row in frozen:
        original.check_metadata(row)
    manifest_row = original.file_metadata(cfg['training_manifest'])
    manifest = original.read_json(manifest_row['path'])
    require(manifest.get('plan_id') == original.PLAN_ID and manifest.get('fits_completed') == 64
            and len(manifest.get('fits', [])) == 64
            and len(manifest.get('selected_weights', [])) == 16,
            'complete original64-fit/16-selected manifest required')
    require({(f['arm'], f['seed'], f['fold']) for f in manifest['fits']}
            == {(a, s, fold) for a in ARMS for s in SEEDS for fold in (0, 1, 2, 3)},
            'original fit identities differ')
    statistics = {}
    require(len(manifest.get('statistics', [])) == 4, 'four original statistics refs required')
    for row in manifest['statistics']:
        fold = row['fold']
        require(fold in (0, 1, 2, 3) and fold not in statistics, 'unique original statistics folds required')
        path = original.check_metadata(row['file'])
        saved = original.read_json(path)
        indices = TRAINING if fold == 0 else tuple(i for i in TRAINING if i % 4 != fold)
        require(saved['fold'] == fold and saved['training_indices'] == list(indices)
                and saved['outer_and_inner_held_used'] is False,
                'original training-only statistics population differs')
        stats, scales = saved['token_statistics'], saved['loss_scales']
        require(np.asarray(stats['mean']).shape == (1536,) and np.asarray(stats['std']).shape == (1536,)
                and np.isfinite(stats['mean']).all() and np.isfinite(stats['std']).all()
                and (np.asarray(stats['std']) >= 0).all()
                and set(scales) == set(original.LOSS_FIELDS)
                and all(math.isfinite(v) and v >= 1e-8 for v in scales.values()),
                'finite original token statistics and loss scales required')
        statistics[fold] = dict(file=row['file'], saved=saved)
    return manifest_row, manifest, statistics


def diagnostic_weights(manifest):
    selected = manifest['selected_weights']
    require({(r['arm'], r['seed']) for r in selected} == {(a, s) for a in ARMS for s in SEEDS},
            'all16 original selected identities required')
    rows = [dict(arm=r['arm'], seed=r['seed'], epoch=r['selected_epoch'], file=r['file'],
                 model_label='selected') for r in selected]
    for seed in SEEDS:
        fit = next(f for f in manifest['fits'] if (f['arm'], f['seed'], f['fold']) == ('P_C', seed, 0))
        snapshot = next(s for s in fit['snapshots'] if s['epoch'] == 32)
        rows.append(dict(arm='P_C', seed=seed, epoch=32, file=snapshot['file'], model_label='P_C_epoch32'))
    require(len(rows) == len({r['file']['path'] for r in rows}) == 18,
            'diagnosis requires18 distinct native selected16-plus-P_C32 weights')
    require(any((r['arm'], r['seed'], r['epoch']) == ('P_C', SEEDS[0], 128) for r in rows),
            'native P_C seed202610270 epoch128 must be an original selected weight')
    for row in rows:
        original.check_metadata(row['file'])
    return rows


class Phase:
    def __init__(self, root, name, cfg, config_path, device):
        self.root, self.name = Path(root).resolve(), name
        self.path, self.device = self.root/name, device
        self.path.mkdir(parents=True, exist_ok=False)
        self.writer = Writer(self.root, cfg['retained_budget_bytes'])
        self.started = time.monotonic()
        self.cfg, self.dataset = cfg, None
        self.code, self.config_metadata = code_metadata(), original.file_metadata(config_path)
        self.writer.json(self.path/'started.json', dict(schema=SCHEMA, run_id=cfg['run_id'],
            objective_version=OBJECTIVE_VERSION, phase=name, status='in_progress', automatic_retry=False,
            started_at=datetime.now(timezone.utc).isoformat(), source_metadata=self.code,
            config_metadata=self.config_metadata, environment=original.environment(device)))
        self.writer.json(self.path/'config.json', cfg)

    def preload(self):
        require(self.dataset is None, 'exactly one native dataset/cache per phase required')
        # The original durable audit writer is bounded in advance for its fixed
        # 120 physical members. It has no per-update or validation writes.
        self.writer.guard.check(self.path/'access-audit.jsonl', 2000000)
        self.dataset = TrainingOnlyDataset(self.cfg['data'], self.path/'access-audit.jsonl')
        frames, targets = {}, {}
        for index in TRAINING:
            frames[index] = self.dataset.read_frame(index)
            targets[index] = self.dataset.read_target(index, purpose='training')
            self.writer.guard.check(self.path/'budget-next-write', 0)
        audit = self.dataset.audit_summary()
        require(audit['training_target_reads'] == 24 and audit['outer_target_reads'] == 0
                and audit['physical_member_opens'] == 120, 'exact training24 cache reads required')
        require(all(row['request']['original_index'] in TRAINING for row in audit['member_accesses']),
                'outer input member was opened')
        return frames, targets

    def finish(self, result, error=None):
        if error is None:
            require(code_metadata() == self.code, 'implementation changed during phase')
            original.check_metadata(self.config_metadata)
            for row in self.cfg.get('frozen_metadata', []):
                original.check_metadata(row)
        audit = None
        if self.dataset is not None:
            data.stable(self.dataset.source_metadata())
            audit = self.dataset.audit_summary()
            self.dataset.close()
            self.writer.json(self.path/'access-summary.json', audit)
        payload = dict(schema=SCHEMA, objective_version=OBJECTIVE_VERSION,
            run_id=self.cfg['run_id'], phase=self.name, status='failed' if error else 'completed',
            source_metadata=self.code, config_metadata=self.config_metadata, audit=audit,
            wall_seconds=time.monotonic()-self.started,
            resources=original.resource_receipt(self.root, self.device), **result)
        if error is not None:
            payload['failure'] = dict(type=type(error).__name__, message=str(error),
                traceback=traceback.format_exc(), automatic_retry=False, dependent_work_stopped=True)
        self.writer.json(self.path/'status.json', payload)
        self.writer.blob(self.path/'exit-code.txt', b'1\n' if error else b'0\n')
        return payload


def native_trap_gate(rows):
    require(all(row.get('native_saved_weights') is True and row.get('synthetic') is False for row in rows),
            'synthetic fixtures cannot satisfy the native saved-weight trap gate')
    cases = []
    for epoch in (32, 128):
        group = [r for r in rows if (r['arm'], r['seed'], r['epoch']) == ('P_C', SEEDS[0], epoch)]
        require(len(group) == 24, 'native trap gate needs complete24 training records per checkpoint')
        saturated = [r for r in group if r['clipping']['fully_saturated']]
        witnesses = [r for r in saturated if r['old_grad_z']['nonzero'] == 0
                     and r['new_grad_z']['nonzero'] > 0 and r['saturated_identity']['passed']]
        finite = all(g['finite'] for r in group for kind in ('old_parameter_groups', 'new_parameter_groups')
                     for g in r[kind].values())
        cases.append(dict(arm='P_C', seed=SEEDS[0], epoch=epoch,
            fully_saturated_images=len(saturated), zero_old_nonzero_new_images=len(witnesses),
            finite_parameter_gradients=finite, passed=bool(witnesses) and finite))
    return dict(passed=all(r['passed'] for r in cases), cases=cases,
                actual_native_weights=True, synthetic_evidence_accepted=False)


def diagnose(cfg, phase, device):
    original_manifest, manifest, statistics = load_original(cfg)
    weights = diagnostic_weights(manifest)
    frames, targets = phase.preload()
    rows = []
    for entry in weights:
        model, saved = original.load_checkpoint(entry['file'], entry['arm'], entry['seed'], device)
        require(saved['split'] == 'final24' and saved['epoch'] == entry['epoch']
                and saved['training_indices'] == list(TRAINING)
                and saved['loss_scales'] == statistics[0]['saved']['loss_scales'],
                'diagnostic checkpoint must use original final24 statistics/recipe')
        stats = statistics[0]['saved']['token_statistics']
        require(torch.equal(model.token_mean.flatten().cpu(), torch.tensor(stats['mean'], dtype=torch.float32))
                and torch.equal(model.token_std.flatten().cpu(), torch.tensor(stats['std'], dtype=torch.float32)),
                'saved checkpoint token buffers differ from exact original statistics')
        for index in TRAINING:
            row = diagnosis_row(model, frames[index], targets[index], entry['file'],
                entry['arm'], entry['seed'], entry['epoch'], index,
                statistics[0]['saved']['loss_scales'], device, entry['model_label'])
            phase.writer.row(phase.path/'diagnosis-rows.jsonl.gz', row)
            rows.append(row)
        del model, saved
    require(len(rows) == 432, 'complete18-times24 diagnosis required')
    gate = native_trap_gate(rows)
    for row in [original_manifest, *[v['file'] for v in statistics.values()], *[r['file'] for r in weights]]:
        original.check_metadata(row)
    result = dict(schema=SCHEMA, objective_version=OBJECTIVE_VERSION, run_id=cfg['run_id'],
        status='passed' if gate['passed'] else 'failed', records=432, weights=18,
        diagnostic_weights=weights, native_trap_gate=gate,
        rows=original.file_metadata(phase.path/'diagnosis-rows.jsonl.gz'),
        original_training_manifest=original_manifest, statistics=manifest['statistics'],
        frozen_metadata=cfg['frozen_metadata'], source_metadata=phase.code,
        data_source_metadata=phase.dataset.source_metadata(), audit=phase.dataset.audit_summary(),
        actual_training_target_reads=24, actual_outer_frame_reads=0, actual_outer_target_reads=0,
        optimizer_steps=0, new_fits=0, RGB_calls=0, synthetic=False,
        old_outputs_and_statistics_immutable=True)
    receipt = phase.writer.json(phase.path/'diagnosis.json', result)
    require(gate['passed'], 'native hard-clamp trap gate failed; no training permitted')
    return dict(diagnosis=receipt, records=432, native_trap_gate=gate, optimizer_steps=0, new_fits=0)


def accepted_diagnosis(cfg, phase, manifest_row, statistics):
    row = cfg.get('diagnosis') or original.file_metadata(phase.root/'diagnose/diagnosis.json')
    path = original.check_metadata(row)
    report = original.read_json(path)
    require(report.get('schema') == SCHEMA and report.get('objective_version') == OBJECTIVE_VERSION
            and report.get('run_id') == cfg['run_id'] and report.get('status') == 'passed'
            and report.get('native_trap_gate', {}).get('passed') is True
            and report.get('records') == 432 and report.get('weights') == 18
            and report.get('synthetic') is False and report.get('optimizer_steps') == 0
            and report.get('original_training_manifest') == manifest_row
            and report.get('frozen_metadata') == cfg['frozen_metadata']
            and report.get('source_metadata') == phase.code,
            'matching complete native diagnosis must pass before any fit')
    require({r['fold']: r['file'] for r in report['statistics']}
            == {fold: value['file'] for fold, value in statistics.items()},
            'diagnosis/statistics physical reuse binding differs')
    original.check_metadata(report['rows'])
    for entry in report['diagnostic_weights']:
        original.check_metadata(entry['file'])
    return row


def rng_state(device):
    numpy = np.random.get_state()
    return dict(torch_rng_state=torch.get_rng_state(),
        numpy_rng_state=dict(bit_generator=numpy[0], keys=numpy[1].tolist(),
                             position=numpy[2], has_gauss=numpy[3], cached_gaussian=numpy[4]),
        python_rng_state=random.getstate(),
        cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])


def continuation_state(model, optimizer, *, arm, seed, fold, split, epoch, steps,
                       indices, held, scales, statistics_file, device):
    """Retain optimizer moments and actual RNG states, not weights-only recovery."""
    return dict(schema=STATE_SCHEMA, original_plan_id=original.PLAN_ID,
        objective_version=OBJECTIVE_VERSION, arm=arm, seed=seed, fold=fold, split=split,
        epoch=epoch, parameter_updates=steps, training_indices=list(indices), held_indices=list(held),
        loss_scales=scales, token_statistics_file=statistics_file,
        state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        optimizer_state_dict=optimizer.state_dict(), optimizer_state_retained=True,
        RNG_state_retained=True, schedule_total_epochs=128, next_epoch=epoch+1, **rng_state(device))


def evaluate(model, frames, targets, indices, scales, arm, seed, fold, epoch, device, population):
    # The original loss/tensors/model are called in the same no-grad forward.
    # Add clipping receipts here because the old scalar-only validation helper
    # does not expose z and cannot distinguish an all-zero collapse.
    rows = []
    training = TRAINING if fold == 0 else tuple(i for i in TRAINING if i % 4 != fold)
    model.eval()
    with torch.no_grad():
        for index in indices:
            common, tokens, donor, target = original.tensors(frames[index], device, targets[index], arm=arm)
            residual, diagnostic = model(common, tokens)
            z = donor+residual
            losses = original.scalar_loss(core.normalize_loss(torch.clamp(z, 0, 1), target, donor, scales))
            rows.append(dict(arm=arm, seed=seed, fold=fold, epoch=epoch, original_index=index,
                losses=losses, selection_criterion=losses['image'], scale_training_indices=list(training),
                diagnostics=original.diagnostic_scalars(diagnostic), clipping=clipping_receipt(z),
                objective_version=OBJECTIVE_VERSION, evaluation_population=population,
                clipped_image_loss=losses['image']))
    model.train()
    return rows


def fit_one(phase, device, arm, seed, fold, indices, held, statistics, frames, targets):
    split = f'inner{fold}' if fold else 'final24'
    path = phase.path/'fits'/split/arm/f'seed-{seed}'
    path.mkdir(parents=True, exist_ok=False)
    original_stats = statistics['saved']
    scales, stats = original_stats['loss_scales'], original_stats['token_statistics']
    require(original_stats['training_indices'] == list(indices), 'fit original statistics membership differs')
    model = original.model_for(arm, seed, stats, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, betas=(.9, .999))
    validation, steps = [], 0
    population, evaluated_indices = ('inner_held', held) if fold else ('training24', TRAINING)
    validation.extend(evaluate(model, frames, targets, evaluated_indices, scales,
                               arm, seed, fold, 0, device, population))
    started = time.monotonic()
    for epoch in range(1, EPOCHS+1):
        rate = original.learning_rate(epoch)  #128-epoch prefix, never rescaled.
        for group in optimizer.param_groups:
            group['lr'] = rate
        order = original.paired_order(indices, seed, fold, epoch)
        codes = [data.seeded_d4(seed, epoch, i) for i in order]
        sums = dict(train_total=[], clipped_image_loss=[], preclip_mse=[])
        old_sums = {key: [] for key in (*original.LOSS_FIELDS, 'image', 'edge', 'total')}
        max_residual = 0.
        for index, code in zip(order, codes):
            common, tokens, donor, target = original.tensors(frames[index], device, targets[index],
                                                            code=code, arm=arm)
            optimizer.zero_grad(set_to_none=True)
            residual, diagnostic = model(common, tokens)
            z = donor+residual
            terms = objective(z, target, donor, scales)
            losses = original.scalar_loss(terms['losses'])
            scalars = {key: float(terms[key].detach().mean()) for key in sums}
            require(all(math.isfinite(value) for value in scalars.values()), 'nonfinite training objective')
            gradient, = torch.autograd.grad(terms['train_total'].mean(), z, retain_graph=True)
            identity = saturated_identity(z, target, scales, gradient)
            require(identity['passed'], 'training loss accidentally re-clamps MSE; affected fit stopped')
            z_receipt = gradient_receipt(gradient)
            z_receipt.update(actual_total_l2=z_receipt['l2'],
                             preclip_mse_l2=gradient_receipt(mse_gradient(z, target, scales))['l2'])
            clip = clipping_receipt(z)
            terms['train_total'].mean().backward()
            groups = parameter_group_receipts(model)
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                    'nonfinite updated parameter; affected fit stopped')
            steps += 1
            update = dict(arm=arm, seed=seed, fold=fold, split=split, epoch=epoch,
                original_index=index, update_index=steps, D4_code=code, learning_rate=rate,
                objective_version=OBJECTIVE_VERSION, training_mse_uses_clamp=False,
                losses=losses, **scalars, clipping=clip, z_gradient=z_receipt,
                raw_mse_saturated_identity=identity, parameter_groups=groups,
                parameters_finite_after_update=True,
                diagnostics=original.diagnostic_scalars(diagnostic))
            phase.writer.row(path/'updates.jsonl.gz', update)
            for key in sums:
                sums[key].append(scalars[key])
            for key in old_sums:
                old_sums[key].append(losses[key])
            max_residual = max(max_residual, float(diagnostic.get('relative_residual_max', 0.)))
        phase.writer.row(path/'trajectory.jsonl.gz', dict(epoch=epoch, learning_rate=rate,
            order=order, D4_codes=codes, images=len(indices), parameter_updates=steps,
            objective_version=OBJECTIVE_VERSION, training_mse_uses_clamp=False,
            losses={key: math.fsum(v)/len(v) for key, v in old_sums.items()},
            **{key: math.fsum(v)/len(v) for key, v in sums.items()},
            maximum_training_solve_relative_residual=max_residual))
        if epoch in VALIDATION_EPOCHS:
            validation.extend(evaluate(model, frames, targets, evaluated_indices, scales,
                                       arm, seed, fold, epoch, device, population))
    optimizer.zero_grad(set_to_none=True)
    state = continuation_state(model, optimizer, arm=arm, seed=seed, fold=fold, split=split,
        epoch=32, steps=steps, indices=indices, held=held, scales=scales,
        statistics_file=statistics['file'], device=device)
    saved_state = phase.writer.state(path/'epoch-032.pt', state)
    result = dict(arm=arm, seed=seed, fold=fold, split=split, epochs=32,
        parameter_updates=steps, training_indices=list(indices), held_indices=list(held),
        loss_scales=scales, original_statistics=statistics['file'],
        objective_version=OBJECTIVE_VERSION, schedule_total_epochs=128,
        state_schema=STATE_SCHEMA,
        validation_epochs=list(VALIDATION_EPOCHS), evaluation_population=population,
        state=saved_state, state_metadata=saved_state, epoch032=saved_state,
        optimizer_state_retained=True, RNG_state_retained=True,
        updates=original.file_metadata(path/'updates.jsonl.gz'),
        trajectory=original.file_metadata(path/'trajectory.jsonl.gz'),
        parameter_counts=model.parameter_counts(), wall_seconds=time.monotonic()-started,
        initialization_and_data_streams_paired=True, training_mse_uses_clamp=False)
    result['file'] = phase.writer.json(path/'fit.json', result)
    del optimizer, model, state
    return result, validation


def train(cfg, phase, device):
    original_manifest, manifest, statistics = load_original(cfg)
    diagnosis = accepted_diagnosis(cfg, phase, original_manifest, statistics)
    frames, targets = phase.preload()
    report = original.read_json(original.check_metadata(diagnosis))
    require(report['data_source_metadata'] == phase.dataset.source_metadata(),
            'training source data differ from native diagnosis')
    fits, validation, final_evaluation = [], [], []
    for fold, indices, held in original.subset_definitions()[:3]:
        for arm in ARMS:
            for seed in SEEDS:
                fit, rows = fit_one(phase, device, arm, seed, fold, indices, held,
                                    statistics[fold], frames, targets)
                fits.append(fit)
                validation.extend(rows)
    for seed in SEEDS:
        fit, rows = fit_one(phase, device, 'P_C', seed, 0, TRAINING, (), statistics[0], frames, targets)
        fits.append(fit)
        final_evaluation.extend(rows)
    require(len(fits) == 50 and len(validation) == 2688 and len(final_evaluation) == 336
            and sum(row['parameter_updates'] for row in fits) == 26112,
            'fixed50-fit32-epoch/update/validation coverage differs')
    validation_file = phase.writer.json(phase.path/'validation.json.gz', validation, compressed=True)
    final_file = phase.writer.json(phase.path/'final24-evaluation.json', final_evaluation)
    for row in [original_manifest, diagnosis, *[v['file'] for v in statistics.values()]]:
        original.check_metadata(row)
    result = dict(schema=SCHEMA, objective_version=OBJECTIVE_VERSION, run_id=cfg['run_id'],
        status='complete_training_only_32_epoch_diagnostic', fits_planned=50, fits_completed=50,
        fits=fits, parameter_updates=26112, inner_parameter_updates=24576, final24_parameter_updates=1536,
        inner_fits=48, final24_fits=2, epochs=32, schedule_total_epochs=128,
        validation_epochs=list(VALIDATION_EPOCHS), validation_rows=2688, final24_evaluation_rows=336,
        validation=validation_file, final24_evaluation=final_file, diagnosis=diagnosis,
        original_training_manifest=original_manifest, statistics=manifest['statistics'],
        frozen_metadata=cfg['frozen_metadata'], source_metadata=phase.code,
        state_metadata=[fit['state'] for fit in fits], data_source_metadata=phase.dataset.source_metadata(),
        audit=phase.dataset.audit_summary(), actual_training_target_reads=24,
        actual_outer_target_reads=0, actual_outer_frame_reads=0, RGB_calls=0,
        training_mse_uses_clamp=False, original_statistics_physically_reused=True,
        recomputed_statistics=0, optimizer_state_retained=True, RNG_state_retained=True,
        initialization_and_data_streams_paired=True, epoch_selection_performed=False,
        canonical_RGB_scoring_performed=False, method_promotion=False, automatic_retry=False)
    receipt = phase.writer.json(phase.path/'manifest.json', result)
    return dict(manifest=receipt, fits_completed=50, parameter_updates=26112,
                validation_rows=2688, final24_evaluation_rows=336, RGB_calls=0)


def synthetic_objective_check():
    """Finite differences in and outside [0,1], plus a known re-clamp defect."""
    y = torch.linspace(.1, .9, 32*32, dtype=torch.float64).reshape(1, 1, 32, 32)
    donor = torch.full_like(y, .4)
    scales = dict(mse=.1, ssim=.2, edge1=.01, edge16=.02)
    z = torch.linspace(-.4, 1.4, y.numel(), dtype=torch.float64).reshape_as(y).requires_grad_()
    terms = objective(z, y, donor, scales)
    grad, = torch.autograd.grad(terms['train_total'].mean(), z)
    finite_difference = []
    eps = 1e-5
    for position in (0, 512, 1023):
        plus, minus = z.detach().clone(), z.detach().clone()
        plus.flatten()[position] += eps
        minus.flatten()[position] -= eps
        fd = (objective(plus, y, donor, scales)['train_total'].mean()
              - objective(minus, y, donor, scales)['train_total'].mean())/(2*eps)
        error = abs(float(fd)-float(grad.flatten()[position]))
        require(error < 1e-7, 'synthetic finite difference failed')
        finite_difference.append(dict(position=position, analytic=float(grad.flatten()[position]),
                                      finite_difference=float(fd), absolute_error=error))
    identity = saturated_identity(z, y, scales, grad)
    require(identity['passed'], 'valid synthetic saturated derivative failed')
    defective = core.normalize_loss(torch.clamp(z, 0, 1), y, donor, scales)['total'].mean()
    bad, = torch.autograd.grad(defective, z)
    bad_receipt = saturated_identity(z, y, scales, bad)
    require(not bad_receipt['passed'], 'known accidental re-clamp defect was not rejected')
    return dict(finite_differences=finite_difference, valid_saturated_identity=identity,
                known_reclamp_defect=bad_receipt, known_defect_rejected=True)


def synthetic_model_check(arm, size=512):
    require(arm in ('P_C', 'E_F'), 'representative P_C and E_F self-tests only')
    torch.manual_seed(719)
    stats = dict(mean=np.zeros(1536), std=np.ones(1536))
    model = original.model_for(arm, 719, stats, torch.device('cpu')).train()
    with torch.no_grad():
        head = model.edge_head if model.is_edge else model.unary_head
        head[-1].weight.normal_(0, .01)
    common = torch.rand(1, 12, size, size)
    tokens = torch.rand(1, 1536, 32, 32)
    donor = torch.full((1, 1, size, size), .4)
    target = torch.linspace(.1, .9, size*size).reshape_as(donor)
    residual, diagnostic = model(common, tokens)
    z = donor+residual
    terms = objective(z, target, donor, dict(mse=.1, ssim=.2, edge1=.01, edge16=.02))
    terms['train_total'].mean().backward()
    groups = parameter_group_receipts(model)
    return dict(arm=arm, shape=[size, size], source='synthetic_model_and_arrays_only',
                native_targets=0, optimizer_steps=0, parameter_groups=groups,
                diagnostics=original.diagnostic_scalars(diagnostic), finite=True)


def self_test(cfg, phase, device):
    require(device.type == 'cpu', 'CPU synthetic self-test required')
    checks = dict(objective=synthetic_objective_check(),
                  models=[synthetic_model_check(arm) for arm in ('P_C', 'E_F')])
    report = dict(schema=SCHEMA, status='passed', objective_version=OBJECTIVE_VERSION,
        synthetic=True, native_diagnosis_pass=False, actual_target_reads=0,
        optimizer_steps=0, new_fits=0, checks=checks)
    return dict(self_test=phase.writer.json(phase.path/'self-test.json', report),
                actual_target_reads=0, optimizer_steps=0, new_fits=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('diagnose', 'train', 'self-test'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    cfg = original.read_json(args.config)
    require(isinstance(cfg.get('run_id'), str) and cfg['run_id'].strip(), 'nonempty run_id required')
    require(args.cpu_threads == cfg.get('cpu_threads', 4) == 4, 'fixed CPU4 executor required')
    require(cfg.get('retained_budget_bytes') == RETAINED_BUDGET_BYTES, 'fixed120MB retained budget required')
    require(cfg.get('objective_version', OBJECTIVE_VERSION) == OBJECTIVE_VERSION,
            'config objective_version differs from fixed intervention')
    require(not set(cfg) & {'epochs', 'seeds', 'arms', 'learning_rate', 'training_indices',
                            'outer_indices', 'screening', 'snapshots', 'amplitude'},
            'scientific overrides forbidden in fixed diagnostic')
    if args.phase != 'self-test':
        require(isinstance(cfg.get('data'), dict) and 'training_manifest' in cfg,
                'explicit original data and training manifest required')
    device = torch.device(args.device)
    original.configure_runtime(device, 4)
    phase = Phase(args.output, args.phase, cfg, args.config, device)
    try:
        action = {'diagnose': diagnose, 'train': train, 'self-test': self_test}[args.phase]
        result = phase.finish(action(cfg, phase, device))
    except BaseException as error:
        try:
            phase.finish({}, error)
        except BaseException as receipt_error:
            print('failure receipt error: '+repr(receipt_error), file=__import__('sys').stderr)
        raise
    print(json.dumps(dict(phase=args.phase, status=result['status'],
                          status_file=str(phase.path/'status.json')), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
