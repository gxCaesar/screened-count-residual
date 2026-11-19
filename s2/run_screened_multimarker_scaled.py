"""Multi-marker (CD3 + DAPI + frozen panCK) FIT1192 screened-residual runner.

Extends the accepted FIT1192 single-marker (panCK, channel 0, cutoff 22)
online detached-table pooled-count screened residual (``S_count``) and matched
reconstruction control (``S_recon``) to the remaining HEMIT markers:
  * ``CD3``  (channel 1, reference cutoff 30)
  * ``DAPI`` (channel 2, reference cutoff 18)

All frozen FIT32 and single-marker FIT1192 modules remain untouched.
Phases:
  * ``prepare_cache``: stream native sources once over the 1,192 FIT1192
    training patches, verify parity on (1, 5), and materialize read-only
    precomputed ``common12``, ``donor_rgb``, and ``target_rgb`` arrays in
    ephemeral local scratch while recording source metadata on persistent disk.
  * ``train_marker``: calibrate per-channel loss scales and output-gradient
    norm balance ``lambda_count`` on the 1,192 training patches and fit
    ``(S_count, S_recon) x (202610270, 202610271)`` for 32 epochs (38,144
    updates/fit) using exact D4 augmentation and GPU-resident arrays.
  * ``evaluate``: bind frozen ``panCK`` FIT1192-v2 states together with the
    newly trained ``CD3`` and ``DAPI`` states before any held target access,
    then score 3-marker ``S_count_3ch``, matched ``S_recon_3ch``, and channel
    ablations across HOLD (300) and Val (630) in a single target-read pass.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import torch
from torch import Tensor

from . import query_relation_count_objective as counts
from . import query_relation_data as old_data
from . import query_relation_reconstruction as core
from . import query_relation_scaled_data as data
from . import query_relation_scoring as scoring
from . import run_query_relation_ensemble as old_eval
from . import run_query_relation_optimization as repaired
from . import run_query_relation_reconstruction as original
from . import run_query_relation_scaled as scaled

SCHEMA = 'screened-multimarker-fit1192-v1'
STATE_SCHEMA = 'screened-multimarker-fit1192-state-v1'
CACHE_SCHEMA = 'screened-multimarker-fit1192-cache-v1'
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
EVAL_RECIPES = {
    'S_count_3ch': ('S_count', 'S_count', 'S_count'),
    'S_recon_3ch': ('S_recon', 'S_recon', 'S_recon'),
    'S_count_panck_cd3': ('S_count', 'S_count', 'donor'),
    'S_count_panck_dapi': ('S_count', 'donor', 'S_count'),
    'S_count_panck_only': ('S_count', 'donor', 'donor'),
    'S_recon_panck_only': ('S_recon', 'donor', 'donor'),
}
SEEDS = scaled.SEEDS
EPOCHS = scaled.EPOCHS
N = scaled.N
STEPS = scaled.STEPS
require = data.require
metadata = old_eval.metadata
read_json = old_eval.read_json
stable = old_eval.stable


def apply_d4_torch(value: Tensor, code: int, dims: tuple[int, int] = (-2, -1)) -> Tensor:
    """Bit-exact PyTorch D4 dihedral transform matching ``old_data.apply_d4``."""
    require(type(code) is int and 0 <= code < 8, 'D4 code0..7 required')
    result = torch.rot90(value, code % 4, dims=dims)
    if code >= 4:
        result = torch.flip(result, dims=(dims[1],))
    return result.contiguous()


def prepare_channel(z: Tensor, target_u8: Tensor, cutoff: int) -> dict[str, Tensor]:
    """Channel-generalized counterpart of ``counts.prepare`` with strict ``target_u8 > cutoff``."""
    require(type(cutoff) is int and 0 <= cutoff < 255, 'strict byte cutoff 0..254 required')
    if (not isinstance(z, Tensor) or z.ndim != 4 or z.shape[1] != 1
            or min(z.shape) < 1 or z.dtype not in (torch.float32, torch.float64)):
        raise ValueError('z must be nonempty N1HW float32 or float64')
    if not bool(torch.isfinite(z).all()):
        raise ValueError('z must be finite')
    if (not isinstance(target_u8, Tensor) or target_u8.shape != z.shape
            or target_u8.dtype != torch.uint8 or target_u8.device != z.device):
        raise ValueError('target must be matching uint8 on the prediction device')
    a = 255 * z.clamp(0, 1)
    g = target_u8 > cutoff
    near = counts._FootprintMaximum.apply(g.to(z.dtype)) > 0
    return dict(a=a, maximum=counts._FootprintMaximum.apply(a), g=g, near=near,
                G=g.to(torch.float64).mean(dim=(1, 2, 3)))


def model_for(arm: str, seed: int, device) -> torch.nn.Module:
    require(arm in BASE_ARMS and seed in SEEDS, 'fixed screened arm/seed required')
    return original.model_for('S_C', seed, dict(mean=[0.] * 1536, std=[1.] * 1536),
                              torch.device(device))


class ChannelPrecomputedStore:
    """Fast precomputed FIT1192 store with optional GPU residency and bit-exact D4."""

    def __init__(self, common12: np.ndarray, donor_channel: np.ndarray,
                 target_channel: np.ndarray, *, channel: int, cutoff: int,
                 source_indices=None, device='cpu', preload_to_device: bool = False,
                 synthetic: bool = False):
        require(type(synthetic) is bool, 'explicit synthetic flag required')
        require(channel in (0, 1, 2) and cutoff == scoring.CUTOFFS[channel],
                'channel/cutoff must match canonical HEMIT marker contract')
        n = len(common12)
        require(synthetic or n == N, 'production store requires FIT1192')
        height, width = common12.shape[-2:]
        require(synthetic or (height, width) == (512, 512), 'production store requires 512x512')
        require(common12.shape == (n, 12, height, width) and common12.dtype == np.float32,
                'common12 must be N x 12 x H x W float32')
        require(donor_channel.shape == (n, height, width) and donor_channel.dtype == np.uint8,
                'donor_channel must be N x H x W uint8')
        require(target_channel.shape == (n, height, width) and target_channel.dtype == np.uint8,
                'target_channel must be N x H x W uint8')
        if source_indices is None:
            require(synthetic, 'production source indices required')
            source_indices = tuple(range(n))
        self.source_indices = tuple(source_indices)
        require(len(self.source_indices) == n and len(set(self.source_indices)) == n,
                'unique source indices required')
        require(synthetic or self.source_indices == data.TRAINING_INDICES,
                'production store must match exact FIT1192 indices')
        self.channel = channel
        self.cutoff = cutoff
        self.synthetic = synthetic
        self.store_device = torch.device(device) if preload_to_device else torch.device('cpu')
        self.common = torch.from_numpy(np.array(common12, dtype=np.float32, copy=True)).to(self.store_device)
        self.donor_u8 = torch.from_numpy(np.array(donor_channel, dtype=np.uint8, copy=True)).to(self.store_device)
        self.target_u8 = torch.from_numpy(np.array(target_channel, dtype=np.uint8, copy=True)).to(self.store_device)
        self._zero_tokens = {}

    def __len__(self) -> int:
        return len(self.source_indices)

    def tensors(self, row: int, device, code: int = 0) -> dict:
        require(type(row) is int and 0 <= row < len(self), 'dataset row out of range')
        dev = torch.device(device)
        key = str(dev)
        if key not in self._zero_tokens:
            self._zero_tokens[key] = torch.zeros((1, 1536, 32, 32), device=dev, dtype=torch.float32)
        common_t = apply_d4_torch(self.common[row], code).unsqueeze(0).to(dev)
        donor_u8 = apply_d4_torch(self.donor_u8[row], code)
        target_u8 = apply_d4_torch(self.target_u8[row], code).to(dev)[None, None]
        donor_dev = donor_u8.to(dev)
        return dict(
            common=common_t,
            tokens=self._zero_tokens[key],
            donor=donor_dev.to(torch.float32)[None, None] / 255.0,
            target=target_u8.to(torch.float32) / 255.0,
            target_u8=target_u8,
            donor_u8_cpu=donor_u8.cpu().numpy(),
        )


def validate_spec(spec: dict, *, synthetic: bool = False) -> None:
    expected_common = dict(
        schema=SCHEMA,
        trained_markers=list(TRAINED_MARKERS),
        frozen_panck_arms=list(BASE_ARMS),
        trained_arms=list(BASE_ARMS),
        eval_arms=list(EVAL_ARMS),
        seeds=list(SEEDS),
        primary_population='val',
        schedule='complete32_update_warmup_cosine_v1',
        locked_test_access=False,
        training_precision='float32',
        inference_solver_precision='float64',
    )
    require(all(spec.get(k) == v for k, v in expected_common.items()),
            'fixed multi-marker FIT1192 protocol spec mismatch')
    if not synthetic:
        expected_prod = dict(
            epochs=EPOCHS,
            training_members=N,
            parameter_updates_per_fit=STEPS,
            parameter_updates_per_marker=len(BASE_ARMS) * len(SEEDS) * STEPS,
            populations=dict(hold=300, val=630),
            target_reads=930,
            prediction_rows=930 * len(EVAL_ARMS) * len(SEEDS),
            donor_rows=930,
        )
        require(all(spec.get(k) == v for k, v in expected_prod.items()),
                'production multi-marker FIT1192 counts mismatch')


def code_metadata() -> list[dict]:
    modules = (
        __file__,
        scaled.__file__,
        data.__file__,
        counts.__file__,
        core.__file__,
        scoring.__file__,
        old_data.__file__,
        original.__file__,
        repaired.__file__,
        old_eval.__file__,
        scoring.reference.__file__,
    )
    return [metadata(Path(p).resolve()) for p in modules]


class MultiMarkerPhase:
    def __init__(self, cfg: dict, spec: dict, root: str | Path, name: str,
                 device: str, config_path: str | Path, spec_path: str | Path):
        self.synthetic = bool(cfg.get('synthetic_fixture', False))
        validate_spec(spec, synthetic=self.synthetic)
        self.cfg, self.spec, self.name = cfg, spec, name
        self.device, self.root = torch.device(device), Path(root).resolve()
        self.path = self.root / name
        require(not self.path.exists(), 'phase is one-shot; retain failed output: ' + str(self.path))
        self.path.mkdir(parents=True)
        self.writer = scaled.Writer(self.root, cfg['retained_budget_bytes'])
        if self.synthetic:
            self.membership = cfg['synthetic_membership']
        else:
            self.membership = data.build_membership(*(read_json(cfg[k]) for k in
                ('fit_metadata', 'pilot_manifest', 'data_manifest')))
            require(read_json(cfg['data_manifest']) == read_json(cfg['native']['manifest']),
                    'bound original/native train-val manifest differs')
        self.code = list({row['path']: row for row in [
            *code_metadata(),
            *[metadata(p) for p in cfg.get('deployed_source_files', [])],
        ]}.values())
        self.inputs = [metadata(config_path), metadata(spec_path)]
        self.data_sources = []
        self.target_reads = 0
        self.started = time.monotonic()

    def close(self, result=None, error=None):
        self.writer.flush()
        for row in [*self.code, *self.inputs, *self.data_sources]:
            stable(row)
        report = dict(
            schema=SCHEMA,
            run_id=self.cfg['run_id'],
            phase=self.name,
            status='failed' if error else 'complete',
            source_metadata=self.code,
            input_metadata=self.inputs,
            data_source_metadata=self.data_sources,
            actual_target_reads=self.target_reads,
            wall_seconds=time.monotonic() - self.started,
            resources=scaled.resources(self.device),
            result=result,
            failure=error,
        )
        self.writer.json(self.path / 'status.json', report)
        self.writer.blob(self.path / 'exit-code.txt', b'1\n' if error else b'0\n')
        return report


def calibrate_channel(store: ChannelPrecomputedStore, marker: str, device) -> dict:
    """Per-marker training-only donor loss scales, pooled adjoint, and gradient-balanced lambda."""
    channel = MARKER_TO_CHANNEL[marker]
    cutoff = MARKER_TO_CUTOFF[marker]
    require(store.channel == channel and store.cutoff == cutoff, 'store marker/cutoff mismatch')
    raw, density_rows, G, donor_counts = [], [], [], None
    for row in range(len(store)):
        item = store.tensors(row, device, 0)
        with torch.no_grad():
            raw.append({k: float(v) for k, v in core.loss_scales(item['donor'], item['target']).items()})
            prepared = prepare_channel(item['donor'], item['target_u8'], cutoff)
            density_rows.append(counts.densities(prepared)[0].cpu())
            G.append(prepared['G'][0].cpu())
            donor_counts = scaled.add_counts(
                donor_counts,
                scaled.byte_counts(item['donor_u8_cpu'], item['target_u8'].cpu().numpy()[0, 0], cutoff),
            )
    scales = {k: max(math.fsum(v[k] for v in raw) / len(raw), core.SCALE_FLOOR) for k in raw[0]}
    table = counts.LaggedCounts(len(store))
    table.refresh(torch.stack(density_rows), torch.stack(G))
    initial = table.statistics()
    A, threshold = initial['adjoint'].detach().to(device), initial['threshold']
    rec_sq, aux_sq = [], []
    for row in range(len(store)):
        item = store.tensors(row, device, 0)
        z = item['donor'].detach().clone().requires_grad_(True)
        base = repaired.objective(z, item['target'], item['donor'], scales)['train_total'].mean()
        auxiliary = (counts.densities(prepare_channel(z, item['target_u8'], cutoff), [threshold])[0, 0] * A).sum()
        bg, = torch.autograd.grad(base, z, retain_graph=True)
        ag, = torch.autograd.grad(auxiliary, z)
        rec_sq.append(float(bg.to(torch.float64).square().sum()))
        aux_sq.append(float(ag.to(torch.float64).square().sum()))
    sums = math.fsum(rec_sq), math.fsum(aux_sq)
    lam = counts.calibration_lambda(*sums)
    plain_donor = original.plain(donor_counts)
    return dict(
        schema=SCHEMA,
        marker=marker,
        channel=channel,
        cutoff=cutoff,
        training_only=True,
        training_members=len(store),
        training_indices=list(store.source_indices),
        loss_scales=scales,
        lambda_count=lam,
        reconstruction_gradient_sq=sums[0],
        auxiliary_gradient_sq=sums[1],
        donor_soft_phi=float(initial['phi']),
        donor_threshold=threshold,
        donor_adjoint=initial['adjoint'].tolist(),
        donor_train_counts=plain_donor,
        donor_train_threshold=scoring.endpoints(plain_donor, cutoff)['ODS_threshold'],
        output_gradient_coordinate='normalized_z',
        auxiliary_byte_chain=255,
        additional_N_factor=False,
        additional_marker_factor=False,
    )


def population_pass_channel(model, store: ChannelPrecomputedStore, seed: int, epoch: int,
                            device, quality=None, *, inference: bool = False):
    """Fixed-weight, same-D4 population pass for LaggedCounts refresh and final byte threshold."""
    was_training = model.training
    model.eval()
    rows, gs, native, qualities = [], [], None, []
    with torch.no_grad():
        for row, source_index in enumerate(store.source_indices):
            code = data.seeded_d4(seed, epoch, source_index) if epoch else 0
            item = store.tensors(row, device, code)
            residual, _ = model(item['common'], item['tokens'], inference=inference)
            z = item['donor'].to(residual.dtype) + residual
            prepared = prepare_channel(z, item['target_u8'], store.cutoff)
            rows.append(counts.densities(prepared)[0].cpu())
            gs.append(prepared['G'][0].cpu())
            if quality is not None:
                prediction = torch.round(z.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()[0, 0]
                target = item['target_u8'].cpu().numpy()[0, 0]
                native = scaled.add_counts(native, scaled.byte_counts(prediction, target, store.cutoff))
                qualities.append(scaled.byte_panck_quality(prediction, target, quality))
    model.train(was_training)
    table = counts.LaggedCounts(len(store))
    table.refresh(torch.stack(rows), torch.stack(gs))
    pooled = table.statistics()
    result = dict(
        soft_phi=float(pooled['phi']),
        soft_threshold=pooled['threshold'],
        soft_ODS=float(pooled['scores'].max()),
    )
    if quality is not None:
        result.update(
            byte_counts=original.plain(native),
            quality={k: math.fsum(r[k] for r in qualities) / len(qualities) for k in ('ssim', 'psnr')},
        )
    return table, result


def _write_scratch_recovery(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    buf = io.BytesIO()
    torch.save(payload, buf)
    compressed = gzip.compress(buf.getvalue(), mtime=0)
    with pending.open('wb') as handle:
        handle.write(compressed)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def fit_marker_one(phase: MultiMarkerPhase, store: ChannelPrecomputedStore,
                   calibration: dict, quality, marker: str, arm: str, seed: int,
                   *, epochs: int, scratch_dir: Path) -> dict:
    path = phase.path / 'fits' / arm / str(seed)
    path.mkdir(parents=True)
    scratch_fit = scratch_dir / 'recovery' / marker / arm / str(seed)
    model = model_for(arm, seed, phase.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .999),
                                 weight_decay=1e-4, eps=1e-8)
    scales, lam = calibration['loss_scales'], calibration['lambda_count']
    cutoff = store.cutoff
    is_count = arm.endswith('_count')
    n_members = len(store)
    total_steps = n_members * epochs
    milestone_stride = max(1, n_members // 4)
    steps = 0
    source_to_row = {s: i for i, s in enumerate(store.source_indices)}
    started = time.monotonic()
    update_hasher = hashlib.content_fingerprint()
    max_grad_norm = 0.0
    min_total, max_total = float('inf'), float('-inf')

    for epoch in range(1, epochs + 1):
        table, _ = (population_pass_channel(model, store, seed, epoch, phase.device)
                    if is_count else (None, None))
        age = np.full(n_members, steps, dtype=np.int64) if is_count else None
        order = data.paired_order(store.source_indices, seed, epoch)
        for position, source_index in enumerate(order, 1):
            row = source_to_row[source_index]
            code = data.seeded_d4(seed, epoch, source_index)
            item = store.tensors(row, phase.device, code)
            optimizer.zero_grad(set_to_none=True)
            rate = scaled.learning_rate(steps + 1, n=n_members, epochs=epochs)
            for group in optimizer.param_groups:
                group['lr'] = rate
            residual, diagnostic = model(item['common'], item['tokens'])
            z = item['donor'] + residual
            base = repaired.objective(z, item['target'], item['donor'], scales)['train_total'].mean()
            auxiliary = z.new_zeros((), dtype=torch.float64)
            phi, threshold = None, None
            if is_count:
                with torch.no_grad():
                    current = counts.densities(prepare_channel(z.detach(), item['target_u8'], cutoff))[0].cpu()
                    table.replace(row, current)
                    age[row] = steps
                    pooled = table.statistics()
                    threshold = pooled['threshold']
                    phi = float(pooled['phi'])
                    A = pooled['adjoint'].detach().to(phase.device)
                selected = counts.densities(prepare_channel(z, item['target_u8'], cutoff), [threshold])[0, 0]
                auxiliary = (A * selected).sum()
            total = base + (lam * auxiliary if is_count else 0)
            require(bool(torch.isfinite(total)), 'nonfinite training loss; stop affected execution')
            total.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients),
                    'nonfinite/absent parameter gradient')
            norm = math.sqrt(math.fsum(float(g.detach().to(torch.float64).square().sum()) for g in gradients))
            optimizer.step()
            scaled.finite_parameters(model)
            steps += 1
            total_val = float(total.detach())
            base_val = float(base.detach())
            aux_val = float(auxiliary.detach())
            max_grad_norm = max(max_grad_norm, norm)
            min_total = min(min_total, total_val)
            max_total = max(max_total, total_val)
            row_value = dict(
                marker=marker,
                channel=store.channel,
                cutoff=cutoff,
                arm=arm,
                seed=seed,
                epoch=epoch,
                update_index=steps,
                original_index=source_index,
                name=phase.membership['training'][row]['name'],
                D4_code=code,
                learning_rate=rate,
                train_total=total_val,
                base=base_val,
                aux=aux_val,
                soft_phi=phi,
                threshold=threshold,
                grad_norm=norm,
                parameter_finite=True,
                solver=original.diagnostic_scalars(diagnostic),
            )
            update_hasher.update(
                f'{steps}:{epoch}:{source_index}:{code}:{rate:.12e}:{total_val:.12e}:{base_val:.12e}:{aux_val:.12e}\n'.encode()
            )
            if steps == 1 or steps == total_steps or position % milestone_stride == 0:
                phase.writer.buffered_row(path / 'updates-milestones.jsonl.gz', row_value)
            if steps == 1:
                phase.writer.flush(path / 'updates-milestones.jsonl.gz')
                print(json.dumps(dict(event='first_finite_update', **row_value)), flush=True)
        phase.writer.flush(path / 'updates-milestones.jsonl.gz')
        eval_quality = quality if epoch == epochs else None
        exact_table, end = population_pass_channel(model, store, seed, epoch, phase.device, eval_quality)
        end.update(
            marker=marker,
            channel=store.channel,
            cutoff=cutoff,
            epoch=epoch,
            parameter_updates=steps,
            arm=arm,
            seed=seed,
            table_max_age_updates=int((steps - age).max()) if age is not None else 0,
            table_mean_age_updates=float((steps - age).mean()) if age is not None else 0.0,
            table_max_abs_discrepancy=float((exact_table.rows - table.rows).abs().max()) if table is not None else 0.0,
        )
        phase.writer.row(path / 'trajectory.jsonl.gz', end)
        _write_scratch_recovery(
            scratch_fit / 'recovery.pt.gz',
            dict(
                schema=STATE_SCHEMA,
                marker=marker,
                channel=store.channel,
                cutoff=cutoff,
                arm=arm,
                seed=seed,
                epoch=epoch,
                position=n_members,
                parameter_updates=steps,
                schedule_total_epochs=epochs,
                training_indices=list(store.source_indices),
                calibration=calibration,
                model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),
                lagged_counts=None if table is None else table.state_dict(),
                table_last_update=None if age is None else age.tolist(),
                **repaired.rng_state(phase.device),
            ),
        )
        print(json.dumps(dict(
            event='epoch_complete',
            marker=marker,
            arm=arm,
            seed=seed,
            epoch=epoch,
            parameter_updates=steps,
            soft_phi=end['soft_phi'],
            soft_ODS=end['soft_ODS'],
            elapsed_seconds=round(time.monotonic() - started, 2),
        )), flush=True)

    _, final = population_pass_channel(model, store, seed, 0, phase.device, quality, inference=True)
    train_threshold = scoring.endpoints(final['byte_counts'], cutoff)['ODS_threshold']
    compact_final_state = dict(
        schema=STATE_SCHEMA,
        marker=marker,
        channel=store.channel,
        cutoff=cutoff,
        arm=arm,
        seed=seed,
        epoch=epochs,
        position=n_members,
        parameter_updates=steps,
        schedule_total_epochs=epochs,
        training_indices=list(store.source_indices),
        calibration=calibration,
        train_threshold=train_threshold,
        final_train_counts=final['byte_counts'],
        model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    )
    state_row = phase.writer.state(path / 'final.pt.gz', compact_final_state)
    if scratch_fit.exists():
        shutil.rmtree(scratch_fit, ignore_errors=True)
    fit = dict(
        marker=marker,
        channel=store.channel,
        cutoff=cutoff,
        arm=arm,
        seed=seed,
        epochs=epochs,
        parameter_updates=steps,
        training_indices=list(store.source_indices),
        state=state_row,
        updates_milestones=metadata(path / 'updates-milestones.jsonl.gz'),
        updates_summary=dict(
            total_updates=steps,
            milestone_stride=milestone_stride,
            stream_digest_16=update_hasher.hexdigest()[:16],
            max_grad_norm=max_grad_norm,
            min_train_total=min_total,
            max_train_total=max_total,
            all_finite=True,
        ),
        trajectory=metadata(path / 'trajectory.jsonl.gz'),
        train_threshold=train_threshold,
        training_final=final,
        final_train_counts=final['byte_counts'],
        schedule_total_epochs=epochs,
        parameter_counts=model.parameter_counts(),
        compact_inference_state=True,
        wall_seconds=time.monotonic() - started,
    )
    phase.writer.json(path / 'fit.json', fit)
    del optimizer, model, compact_final_state, table, exact_table
    gc.collect()
    return fit


def prepare_cache(phase: MultiMarkerPhase) -> dict:
    """Extract FIT1192 common12, donor_rgb, and target_rgb once into local scratch."""
    cache_dir = Path(phase.cfg['scratch_cache_dir']).expanduser()
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir.parent.resolve() / cache_dir.name
    require(not cache_dir.exists(), 'scratch_cache_dir must be fresh: ' + str(cache_dir))
    cache_dir.mkdir(parents=True)
    n_members = len(phase.membership['training'])
    h = int(phase.cfg.get('synthetic_hw', 512))
    common12 = np.lib.format.open_memmap(
        cache_dir / 'common12.npy', mode='w+', dtype=np.float32, shape=(n_members, 12, h, h))
    donor_rgb = np.lib.format.open_memmap(
        cache_dir / 'donor_rgb.npy', mode='w+', dtype=np.uint8, shape=(n_members, h, h, 3))
    target_rgb = np.lib.format.open_memmap(
        cache_dir / 'target_rgb.npy', mode='w+', dtype=np.uint8, shape=(n_members, h, h, 3))
    donor_counts = [None, None, None]

    if phase.synthetic:
        frames = phase.cfg['synthetic_training_frames']
        for row, entry in enumerate(frames):
            frame = data.frame_from_arrays(entry['he'], entry['u'], entry['f0'], synthetic=True)
            common12[row] = frame['common12']
            donor_rgb[row] = frame['donor']
            target = entry['target']
            phase.target_reads += 1
            target_rgb[row] = target
            for ch in range(3):
                donor_counts[ch] = scaled.add_counts(
                    donor_counts[ch],
                    scaled.byte_counts(frame['donor'][..., ch], target[..., ch], scoring.CUTOFFS[ch]),
                )
            phase.writer.row(phase.path / 'training-access.jsonl.gz', dict(
                event='target_read',
                row=row,
                original_index=phase.membership['training'][row]['index'],
                name=phase.membership['training'][row]['name'],
            ))
        release = dict(synthetic=True, owned_cache_entries_remaining=0)
        parity_rows = []
    else:
        phase.data_sources = scaled.source_bindings(phase.cfg, phase.membership, ('training',))
        parity_by_name = {r['name']: r for r in phase.cfg.get('profile_parity', [])}
        parity_rows = []
        sources = data.NativeSources(phase.cfg['native'], phase.device, phase.membership)
        try:
            for row, (name, frame, loader) in enumerate(sources.items('training')):
                require(name == phase.membership['training'][row]['name'], 'training source order changed')
                if name in parity_by_name:
                    ref_row = metadata(parity_by_name[name]['frame'])
                    old = old_eval.read_frame(ref_row)
                    phase.data_sources.append(ref_row)
                    checks = {k: bool(np.array_equal(frame[k], old[k])) for k in ('he', 'u', 'donor', 'common12')}
                    checks['f0'] = bool(np.array_equal(frame['f0'], old['fs'][0]))
                    require(all(checks.values()), 'native source parity failed on ' + name)
                    parity_rows.append(dict(name=name, reference=ref_row, checks=checks))
                derived = data.frame_from_arrays(frame['he'], frame['u'], frame['f0'])
                require(np.array_equal(derived['common12'], frame['common12'])
                        and np.array_equal(derived['donor'], frame['donor']),
                        'frame_from_arrays parity mismatch')
                common12[row] = derived['common12']
                donor_rgb[row] = derived['donor']
                target = loader()
                phase.target_reads += 1
                require(target.shape == (512, 512, 3) and target.dtype == np.uint8, 'native RGB target required')
                target_rgb[row] = target
                for ch in range(3):
                    donor_counts[ch] = scaled.add_counts(
                        donor_counts[ch],
                        scaled.byte_counts(derived['donor'][..., ch], target[..., ch], scoring.CUTOFFS[ch]),
                    )
                phase.writer.row(phase.path / 'training-access.jsonl.gz', dict(
                    event='target_read',
                    row=row,
                    original_index=phase.membership['training'][row]['index'],
                    name=name,
                ))
                if row == 0 or (row + 1) % 100 == 0 or (row + 1) == n_members:
                    print(json.dumps(dict(event='cache_source_loaded', images=row + 1, total=n_members)), flush=True)
        finally:
            release = sources.close()
            require(release['owned_cache_entries_remaining'] == 0,
                    'owned source caches remain after prepare_cache')
        del sources
        gc.collect()
        if phase.device.type == 'cuda':
            torch.cuda.empty_cache()

    common12.flush()
    donor_rgb.flush()
    target_rgb.flush()
    del common12, donor_rgb, target_rgb
    phase.writer.json(phase.path / 'source-release.json', release)
    plain_donor_counts = {m: original.plain(c) for m, c in zip(scoring.MARKERS, donor_counts)}
    donor_thresholds = [
        scoring.endpoints(plain_donor_counts[m], cutoff)['ODS_threshold']
        for m, cutoff in zip(scoring.MARKERS, scoring.CUTOFFS)
    ]
    array_files = {
        name: metadata(cache_dir / f'{name}.npy')
        for name in ('common12', 'donor_rgb', 'target_rgb')
    }
    manifest = dict(
        schema=CACHE_SCHEMA,
        run_id=phase.cfg['run_id'],
        status='complete',
        membership=phase.membership,
        training_members=n_members,
        actual_training_target_reads=phase.target_reads,
        actual_held_target_reads=0,
        parity_checks=parity_rows,
        donor_train_counts=plain_donor_counts,
        donor_train_thresholds=donor_thresholds,
        scratch_arrays=array_files,
        source_release=metadata(phase.path / 'source-release.json'),
        training_access=metadata(phase.path / 'training-access.jsonl.gz'),
        source_metadata=phase.code + phase.data_sources,
    )
    return dict(
        manifest=phase.writer.json(phase.path / 'cache-manifest.json', manifest),
        training_members=n_members,
        actual_target_reads=phase.target_reads,
    )


def train_marker(phase: MultiMarkerPhase, marker: str) -> dict:
    require(marker in (scoring.MARKERS if phase.synthetic else TRAINED_MARKERS),
            'marker must be CD3 or DAPI')
    cache_row = metadata(phase.cfg['cache_manifest'])
    cache_manifest = read_json(cache_row['path'])
    require(cache_manifest.get('schema') == CACHE_SCHEMA
            and cache_manifest.get('status') == 'complete'
            and cache_manifest.get('membership') == phase.membership,
            'verified prepare_cache manifest required')
    for arr_row in cache_manifest['scratch_arrays'].values():
        stable(arr_row)
    common12 = np.load(cache_manifest['scratch_arrays']['common12']['path'], mmap_mode='r')
    donor_rgb = np.load(cache_manifest['scratch_arrays']['donor_rgb']['path'], mmap_mode='r')
    target_rgb = np.load(cache_manifest['scratch_arrays']['target_rgb']['path'], mmap_mode='r')
    channel = MARKER_TO_CHANNEL[marker]
    cutoff = MARKER_TO_CUTOFF[marker]
    indices = [r['index'] for r in phase.membership['training']]
    preload_gpu = bool(phase.cfg.get('preload_to_gpu', phase.device.type == 'cuda'))
    store = ChannelPrecomputedStore(
        common12,
        np.ascontiguousarray(donor_rgb[..., channel]),
        np.ascontiguousarray(target_rgb[..., channel]),
        channel=channel,
        cutoff=cutoff,
        source_indices=indices,
        device=phase.device,
        preload_to_device=preload_gpu,
        synthetic=phase.synthetic,
    )
    del common12, donor_rgb, target_rgb
    gc.collect()

    calibration = calibrate_channel(store, marker, phase.device)
    require(calibration['donor_train_counts'] == cache_manifest['donor_train_counts'][marker]
            and calibration['donor_train_threshold'] == cache_manifest['donor_train_thresholds'][channel],
            'marker donor count/threshold disagrees with prepare_cache')
    cal_row = phase.writer.json(phase.path / 'calibration.json', calibration)
    print(json.dumps(dict(
        event='marker_calibrated',
        marker=marker,
        channel=channel,
        cutoff=cutoff,
        lambda_count=calibration['lambda_count'],
        donor_soft_phi=calibration['donor_soft_phi'],
        donor_threshold=calibration['donor_threshold'],
    )), flush=True)

    quality = old_eval.quality_for(phase.cfg)
    epochs = int(phase.spec['epochs'])
    scratch_dir = Path(phase.cfg['scratch_work_dir']).absolute()
    groups = [(arm, seed) for arm in BASE_ARMS for seed in SEEDS]
    fits = [
        fit_marker_one(phase, store, calibration, quality, marker, arm, seed,
                       epochs=epochs, scratch_dir=scratch_dir)
        for arm, seed in groups
    ]
    steps_per_fit = len(store) * epochs
    manifest = dict(
        schema=SCHEMA,
        phase=f'train_{marker}',
        marker=marker,
        channel=channel,
        cutoff=cutoff,
        status='complete',
        run_id=phase.cfg['run_id'],
        cache_manifest=cache_row,
        membership=phase.membership,
        membership_sources=({} if phase.synthetic else {
            key: metadata(phase.cfg[key]) for key in ('fit_metadata', 'pilot_manifest', 'data_manifest')
        }),
        calibration=cal_row,
        fits=fits,
        parameter_updates=len(fits) * steps_per_fit,
        actual_training_target_reads=0,
        actual_held_target_reads=0,
        source_metadata=phase.code,
        checkpoint_selection=False,
        method_claim=False,
    )
    stable(cache_row)
    return dict(
        manifest=phase.writer.json(phase.path / 'manifest.json', manifest),
        marker=marker,
        fits=len(fits),
        parameter_updates=len(fits) * steps_per_fit,
    )


def apply_channel_residual(image: np.ndarray, donor: np.ndarray, channel: int,
                           residual: np.ndarray) -> dict:
    require(residual.dtype == np.float64 and residual.shape == donor.shape[:2],
            'float64 HxW inference residual required')
    z = donor[..., channel].astype(np.float64) / 255.0 + residual
    clipped = np.clip(z, 0.0, 1.0)
    image[..., channel] = np.rint(255.0 * clipped).astype(np.uint8)
    return dict(
        pixels=int(z.size),
        below_zero=int((z < 0).sum()),
        above_one=int((z > 1).sum()),
        preclip_mean=float(z.mean()),
        postclip_mean=float(clipped.mean()),
        rounding_mean_change=float(image[..., channel].mean() / 255.0 - clipped.mean()),
    )


def _zero_clip(pixels: int, donor_channel: np.ndarray) -> dict:
    mean_val = float(donor_channel.astype(np.float64).mean() / 255.0)
    return dict(
        pixels=pixels,
        below_zero=0,
        above_one=0,
        preclip_mean=mean_val,
        postclip_mean=mean_val,
        rounding_mean_change=0.0,
    )


def score_multimarker_bytes(image: np.ndarray, target: np.ndarray, donor: np.ndarray,
                            quality, channel_clips: list[dict] | None = None,
                            precomputed_donor_counts: dict | None = None) -> dict:
    """Canonical 3-marker byte scoring across panCK (22), CD3 (30), and DAPI (18)."""
    image = scoring._byte_rgb(image, 'image')
    target = scoring._byte_rgb(target, 'target')
    donor = scoring._byte_rgb(donor, 'donor')
    require(image.shape == target.shape == donor.shape, 'image/target/donor shape mismatch')
    measured = quality(image, target)
    q = scoring._quality(measured, 'quality')
    if 'rgb_psnr' in measured:
        q['rgb_psnr'] = float(measured['rgb_psnr'])
    else:
        mse_rgb = float(np.square(image.astype(np.float64) - target.astype(np.float64)).mean())
        q['rgb_psnr'] = float(20.0 * np.log10(255.0 / (np.sqrt(mse_rgb) + 1e-8)))

    kernel = scoring.tolerance_kernel()
    pixels = int(image.shape[0] * image.shape[1])
    out_counts = {}
    donor_counts = {}
    means = {}
    clipping_by_marker = {}

    for ch, (marker, cutoff) in enumerate(zip(scoring.MARKERS, scoring.CUTOFFS)):
        labels = target[..., ch] > cutoff

        def _cnt(arr):
            raw = scoring.reference.counts(arr, labels, kernel)
            return {k: int(v) if k == 'G' else v.tolist() for k, v in raw.items()}

        if precomputed_donor_counts is not None:
            d_cnt = precomputed_donor_counts[marker]
        else:
            d_cnt = _cnt(donor[..., ch])
        donor_counts[marker] = d_cnt
        if np.array_equal(image[..., ch], donor[..., ch]):
            out_counts[marker] = d_cnt
        else:
            out_counts[marker] = _cnt(image[..., ch])

        out_mean = float(image[..., ch].astype(np.float64).sum() / pixels)
        don_mean = float(donor[..., ch].astype(np.float64).sum() / pixels)
        means[marker] = dict(output=out_mean, donor=don_mean, shift=out_mean - don_mean)
        c_rec = None if channel_clips is None else channel_clips[ch]
        clipping_by_marker[marker] = scoring._clipping(c_rec, pixels, out_mean)

    edit = image.astype(np.int64) - donor.astype(np.int64)
    changed_by_marker = np.count_nonzero(edit, axis=(0, 1)).tolist()
    return dict(
        schema_version=2,
        shape=list(image.shape),
        pixels=pixels,
        marker_order=list(scoring.MARKERS),
        counts=out_counts,
        quality=q,
        means=means,
        clipping_by_marker=clipping_by_marker,
        changed_pixels=int(np.count_nonzero(np.any(edit != 0, axis=2))),
        changed_values_by_marker=changed_by_marker,
    )


def _load_panck_models(panck_manifest_path: str | Path, membership: dict, device) -> tuple[dict, dict]:
    panck_row = metadata(panck_manifest_path)
    panck_manifest = read_json(panck_row['path'])
    require(panck_manifest.get('schema') == scaled.SCHEMA
            and panck_manifest.get('status') == 'complete'
            and panck_manifest.get('membership') == membership,
            'accepted FIT1192 panCK training manifest required')
    cal_row = panck_manifest['calibration']
    stable(cal_row)
    calibration = read_json(cal_row['path'])
    loaded = {}
    by_key = {(f['arm'], f['seed']): f for f in panck_manifest['fits']}
    for arm in BASE_ARMS:
        for seed in SEEDS:
            fit = by_key[(arm, seed)]
            saved = scaled.read_state(fit['state'])
            require(saved.get('schema') == scaled.STATE_SCHEMA
                    and saved.get('arm') == arm
                    and saved.get('seed') == seed
                    and saved.get('calibration') == calibration
                    and saved.get('train_thresholds') == fit['train_thresholds'],
                    'frozen panCK state verification failed')
            model = model_for(arm, seed, device)
            model.load_state_dict(saved['model'], strict=True)
            scaled.finite_parameters(model)
            model.eval().requires_grad_(False)
            loaded[(arm, seed)] = dict(
                marker='panCK',
                channel=0,
                cutoff=22,
                arm=arm,
                seed=seed,
                state=fit['state'],
                train_threshold=fit['train_thresholds'][0],
                donor_thresholds=fit['train_thresholds'],
                model=model,
            )
    return panck_row, loaded


def _load_trained_marker_models(manifest_path: str | Path, marker: str,
                                membership: dict, epochs: int, device) -> tuple[dict, dict]:
    m_row = metadata(manifest_path)
    m_manifest = read_json(m_row['path'])
    channel = MARKER_TO_CHANNEL[marker]
    cutoff = MARKER_TO_CUTOFF[marker]
    require(m_manifest.get('schema') == SCHEMA
            and m_manifest.get('status') == 'complete'
            and m_manifest.get('marker') == marker
            and m_manifest.get('channel') == channel
            and m_manifest.get('cutoff') == cutoff
            and m_manifest.get('membership') == membership,
            f'complete {marker} training manifest required')
    cal_row = m_manifest['calibration']
    stable(cal_row)
    calibration = read_json(cal_row['path'])
    by_key = {(f['arm'], f['seed']): f for f in m_manifest['fits']}
    loaded = {}
    n_members = len(membership['training'])
    for arm in BASE_ARMS:
        for seed in SEEDS:
            fit = by_key[(arm, seed)]
            saved = scaled.read_state(fit['state'])
            require(saved.get('schema') == STATE_SCHEMA
                    and saved.get('marker') == marker
                    and saved.get('channel') == channel
                    and saved.get('cutoff') == cutoff
                    and saved.get('arm') == arm
                    and saved.get('seed') == seed
                    and saved.get('epoch') == epochs
                    and saved.get('parameter_updates') == n_members * epochs
                    and saved.get('calibration') == calibration
                    and saved.get('train_threshold') == fit['train_threshold'],
                    f'frozen {marker} state verification failed')
            model = model_for(arm, seed, device)
            model.load_state_dict(saved['model'], strict=True)
            scaled.finite_parameters(model)
            model.eval().requires_grad_(False)
            loaded[(arm, seed)] = dict(
                marker=marker,
                channel=channel,
                cutoff=cutoff,
                arm=arm,
                seed=seed,
                state=fit['state'],
                train_threshold=fit['train_threshold'],
                model=model,
            )
    return m_row, loaded


def evaluate(phase: MultiMarkerPhase) -> dict:
    cache_row = metadata(phase.cfg['cache_manifest'])
    cache_manifest = read_json(cache_row['path'])
    require(cache_manifest.get('schema') == CACHE_SCHEMA
            and cache_manifest.get('status') == 'complete'
            and cache_manifest.get('membership') == phase.membership,
            'complete prepare_cache manifest required')
    donor_thresholds = cache_manifest['donor_train_thresholds']
    epochs = int(phase.spec['epochs'])

    if phase.synthetic:
        panck_row, panck_models = _load_trained_marker_models(
            phase.cfg['panck_training_manifest'], 'panCK', phase.membership, epochs, phase.device)
    else:
        panck_row, panck_models = _load_panck_models(
            phase.cfg['panck_training_manifest'], phase.membership, phase.device)
        require(panck_models[('S_count', SEEDS[0])]['donor_thresholds'][1:] == donor_thresholds[1:],
                'CD3/DAPI donor thresholds differ from FIT1192-v2')

    cd3_row, cd3_models = _load_trained_marker_models(
        phase.cfg['cd3_training_manifest'], 'CD3', phase.membership, epochs, phase.device)
    dapi_row, dapi_models = _load_trained_marker_models(
        phase.cfg['dapi_training_manifest'], 'DAPI', phase.membership, epochs, phase.device)

    marker_banks = {'panCK': panck_models, 'CD3': cd3_models, 'DAPI': dapi_models}
    frozen_single_models = []
    for marker in scoring.MARKERS:
        for arm in BASE_ARMS:
            for seed in SEEDS:
                entry = marker_banks[marker][(arm, seed)]
                frozen_single_models.append(dict(
                    marker=marker,
                    channel=entry['channel'],
                    cutoff=entry['cutoff'],
                    arm=arm,
                    seed=seed,
                    state=entry['state'],
                    train_threshold=entry['train_threshold'],
                ))

    composite_specs = []
    for eval_arm in EVAL_ARMS:
        recipe = EVAL_RECIPES[eval_arm]
        for seed in SEEDS:
            thresholds = []
            states = {}
            for ch, (marker, source_kind) in enumerate(zip(scoring.MARKERS, recipe)):
                if source_kind == 'donor':
                    thresholds.append(donor_thresholds[ch])
                    states[marker] = None
                else:
                    entry = marker_banks[marker][(source_kind, seed)]
                    thresholds.append(entry['train_threshold'])
                    states[marker] = entry['state']
            composite_specs.append(dict(
                arm=eval_arm,
                recipe=list(recipe),
                seed=seed,
                train_thresholds=thresholds,
                channel_states=states,
            ))

    phase.writer.json(phase.path / 'frozen-models.json', dict(
        single_channel_models=frozen_single_models,
        composite_arms=composite_specs,
        donor_train_thresholds=donor_thresholds,
        training_manifests=dict(panCK=panck_row, CD3=cd3_row, DAPI=dapi_row, cache=cache_row),
        target_reads=0,
    ))

    quality = old_eval.quality_for(phase.cfg)
    expected_total = len(phase.membership['hold']) + len(phase.membership['val'])
    n_composites = len(composite_specs)

    def score_split_items(split_name: str, iterator, start_index: int) -> int:
        idx = start_index
        for name, frame, loader in iterator:
            h, w = frame['donor'].shape[:2]
            common = torch.from_numpy(frame['common12']).unsqueeze(0).to(phase.device)
            tokens = torch.zeros((1, 1536, 32, 32), device=phase.device)
            single_residuals = {}
            single_solvers = {}
            for marker in scoring.MARKERS:
                for base_arm in BASE_ARMS:
                    for seed in SEEDS:
                        model = marker_banks[marker][(base_arm, seed)]['model']
                        with torch.inference_mode():
                            res, diag = model(common, tokens, inference=True)
                        single_residuals[(marker, base_arm, seed)] = res.cpu().numpy()[0, 0]
                        single_solvers[(marker, base_arm, seed)] = original.diagnostic_scalars(diag)

            generated = []
            pixels = h * w
            for comp in composite_specs:
                seed = comp['seed']
                img = frame['donor'].copy()
                clips = []
                ch_solvers = {}
                for ch, (marker, source_kind) in enumerate(zip(scoring.MARKERS, comp['recipe'])):
                    if source_kind == 'donor':
                        clips.append(_zero_clip(pixels, frame['donor'][..., ch]))
                    else:
                        clips.append(apply_channel_residual(
                            img, frame['donor'], ch, single_residuals[(marker, source_kind, seed)]))
                        ch_solvers[marker] = single_solvers[(marker, source_kind, seed)]
                img.setflags(write=False)
                generated.append((comp, img, clips, dict(inference=True, channels=ch_solvers)))

            require(len(generated) == n_composites, 'all composite outputs must be frozen before target read')
            identity = dict(index=idx, split=split_name, name=name)
            phase.writer.row(phase.path / 'access.jsonl.gz',
                             dict(event='generated', outputs=n_composites, **identity))
            target = loader()
            phase.target_reads += 1
            phase.writer.row(phase.path / 'access.jsonl.gz', dict(event='target_read', **identity))
            donor_score = score_multimarker_bytes(frame['donor'], target, frame['donor'], quality)
            phase.writer.row(phase.path / 'donors.jsonl.gz', dict(**identity, scores=donor_score))
            scored_rows = []
            for comp, img, ch_clips, solver_info in generated:
                val = score_multimarker_bytes(
                    img, target, frame['donor'], quality, ch_clips,
                    precomputed_donor_counts=donor_score['counts'],
                )
                scored_rows.append(dict(
                    **identity,
                    arm=comp['arm'],
                    seed=comp['seed'],
                    scores=val,
                    solver=solver_info,
                ))
            phase.writer.rows(phase.path / 'predictions.jsonl.gz', scored_rows)
            idx += 1
            if idx == 1 or idx % 50 == 0 or idx == expected_total:
                print(json.dumps(dict(
                    event='held_scored',
                    images=idx,
                    total=expected_total,
                    predictions=idx * n_composites,
                )), flush=True)
        return idx

    index = 0
    if phase.synthetic:
        for split in ('hold', 'val'):
            def _syn_iter(sp):
                for entry in phase.cfg['synthetic_held_frames'][sp]:
                    fr = data.frame_from_arrays(entry['he'], entry['u'], entry['f0'], synthetic=True)
                    yield entry['name'], fr, (lambda t=entry['target']: t)
            index = score_split_items(split, _syn_iter(split), index)
    else:
        phase.data_sources = scaled.source_bindings(phase.cfg, phase.membership, ('hold', 'val'))
        sources = data.NativeSources(phase.cfg['native'], phase.device, phase.membership)
        try:
            for split in ('hold', 'val'):
                index = score_split_items(split, sources.items(split), index)
        finally:
            sources.close()

    require(index == phase.target_reads == expected_total, 'exact held target read count required')
    for entry in frozen_single_models:
        stable(entry['state'])
    for row in (panck_row, cd3_row, dapi_row, cache_row):
        stable(row)

    manifest = dict(
        schema=SCHEMA,
        phase='evaluate',
        status='complete',
        run_id=phase.cfg['run_id'],
        training_manifests=dict(panCK=panck_row, CD3=cd3_row, DAPI=dapi_row, cache=cache_row),
        membership=phase.membership,
        membership_sources=({} if phase.synthetic else {
            key: metadata(phase.cfg[key]) for key in ('fit_metadata', 'pilot_manifest', 'data_manifest')
        }),
        single_channel_models=frozen_single_models,
        models=composite_specs,
        predictions=metadata(phase.path / 'predictions.jsonl.gz'),
        donors=metadata(phase.path / 'donors.jsonl.gz'),
        access=metadata(phase.path / 'access.jsonl.gz'),
        source_metadata=phase.code + phase.data_sources,
        actual_target_reads=expected_total,
        generated_outputs=expected_total * n_composites,
        RGB_calls=expected_total * (n_composites + 1),
        baseline_analysis=(None if phase.synthetic else metadata(phase.cfg['baseline_analysis'])),
        panck_fit1192_summary=(None if phase.synthetic else metadata(phase.cfg['panck_fit1192_summary'])),
        quality_metadata=quality.metadata,
        locked_test_access=False,
        method_claim=False,
    )
    return dict(
        manifest=phase.writer.json(phase.path / 'manifest.json', manifest),
        predictions=expected_total * n_composites,
        target_reads=expected_total,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('prepare_cache', 'train_marker', 'evaluate'), required=True)
    parser.add_argument('--marker', choices=scoring.MARKERS, default=None)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    cfg, spec = read_json(args.config), read_json(args.spec)
    require(cfg['cpu_threads'] in (1, 2, 4), 'bounded explicit CPU threads required')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    phase_name = f'train_{args.marker}' if args.phase == 'train_marker' else args.phase
    if args.phase == 'train_marker':
        allowed_markers = scoring.MARKERS if cfg.get('synthetic_fixture') else TRAINED_MARKERS
        require(args.marker in allowed_markers, '--marker CD3 or DAPI required for train_marker')
    phase = MultiMarkerPhase(cfg, spec, args.output, phase_name, args.device, args.config, args.spec)
    try:
        if args.phase == 'prepare_cache':
            result = prepare_cache(phase)
        elif args.phase == 'train_marker':
            result = train_marker(phase, args.marker)
        else:
            result = evaluate(phase)
    except Exception:
        failure = traceback.format_exc()
        phase.close(error=failure)
        print(failure, flush=True)
        return 1
    phase.close(result=result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
