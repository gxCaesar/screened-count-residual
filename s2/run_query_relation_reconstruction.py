"""One-shot native executor for the prospective query-relation pilot.

Each CLI phase owns a new ``OUTPUT/phase`` directory. A started phase cannot be
overwritten, resumed, or automatically retried. Resource admission and remote
launching belong to the caller. This module neither discovers data nor opens a
target outside the audited original-FIT reader.

Runner config: ``run_id`` and ``data`` (the query-relation-data-v1 object) are
required. Optional paths are ``quality_directory``, ``profile_acceptance``,
``training_manifest``, ``generation_manifest``, ``generation_acceptance`` and
``old229``. The four phases use the same output root by default. Scientific
settings come exclusively from the fixed spec, rather than runner overrides.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import resource
import stat
import sys
import time
import traceback

import numpy as np
import torch

from . import query_relation_data as data
from . import query_relation_reconstruction as core
from . import query_relation_scoring as scoring


PLAN_ID = 'query-relation-reconstruction-20261027'
ARMS = ('P_F', 'M_F', 'S_F', 'E_F', 'P_C', 'S_C', 'E_C', 'E_shift')
SEEDS = (202610270, 202610271)
OUTER = tuple(range(0, 32, 4))
TRAINING = tuple(i for i in range(32) if i not in OUTER)
EPOCHS = 128
SNAPSHOTS = (32, 64, 96, 128)
LOSS_FIELDS = ('mse', 'ssim', 'edge1', 'edge16')
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_metadata(path):
    path = Path(path).resolve(strict=True)
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_size > 0,
            'nonempty regular file required: '+str(path))
    return dict(path=str(path), size_bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def check_metadata(row):
    require(isinstance(row, dict) and set(row) == {'path', 'size_bytes', 'mtime_ns'},
            'exact regular-file metadata required')
    require(file_metadata(row['path']) == row, 'file changed: '+str(row['path']))
    return Path(row['path'])


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=invalid)


def plain(value):
    if isinstance(value, torch.Tensor):
        require(value.numel() == 1, 'non-scalar tensor cannot enter a JSON receipt')
        return value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(plain(value), handle, allow_nan=False, sort_keys=True, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return file_metadata(path)


def replace_json(path, value):
    """Only the explicitly mutable progress receipt uses atomic replacement."""
    path = Path(path)
    temporary = path.with_name(path.name+'.pending')
    write_json(temporary, value)
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(plain(value), allow_nan=False, sort_keys=True)+'\n')
        handle.flush()
        os.fsync(handle.fileno())


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return file_metadata(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def source_metadata():
    return [file_metadata(path) for path in
            (Path(__file__), Path(data.__file__), Path(core.__file__),
             Path(scoring.__file__), ROOT/'s2/_local_exchange_reference.py')]


def validate_spec(spec):
    """Reject a changed scientific protocol instead of partially honoring it."""
    require(spec.get('plan_id') == PLAN_ID, 'fixed pilot plan_id required')
    require(spec.get('population', {}).get('outer_indices') == list(OUTER)
            and spec['population'].get('training_indices') == list(TRAINING),
            'exact original FIT8/training24 membership required')
    require(tuple(row.get('id') for row in spec.get('arms', [])) == ARMS,
            'all eight fixed arms, in protocol order, required')
    tr = spec.get('training', {})
    expected = dict(seeds=list(SEEDS), fits_planned=64, batch_size=1,
                    full_frame=True, epochs=EPOCHS, snapshot_epochs=list(SNAPSHOTS),
                    optimizer='AdamW', learning_rate=.001, weight_decay=.0001,
                    betas=[.9, .999], warmup_epochs=5,
                    warmup_initial_learning_rate=.0001, scale_floor=1e-8,
                    mixed_precision=False, TF32=False, outer_oracle=False)
    require(all(tr.get(key) == value for key, value in expected.items()),
            'runner implements only the fixed64-fit/128-epoch protocol')
    graph = spec.get('graph', {})
    require(graph.get('shape') == [512, 512] and graph.get('offsets') == [1, 16]
            and graph.get('incidence_scale') == .25 and graph.get('screening') == 1.,
            'fixed full512 graph and screening required')
    require(spec.get('freeze_and_score', {}).get('new_outputs') == 128
            and spec['freeze_and_score'].get('new_outer_target_reads_planned') == 8
            and spec['freeze_and_score'].get('new_RGB_calls_planned') == 128,
            'fixed128-output/eight-target-read scoring required')


def learning_rate(epoch):
    require(type(epoch) is int and 1 <= epoch <= EPOCHS, 'epoch1..128 required')
    if epoch <= 5:
        return .0001 + (.001-.0001)*(epoch-1)/4
    return .00001 + .5*(.001-.00001)*(1+math.cos(math.pi*(epoch-5)/(EPOCHS-5)))


def paired_order(indices, seed, split, epoch):
    """The stream contains no arm ID; all arms receive paired data orders."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, split, epoch]))
    return [int(index) for index in rng.permutation(indices)]


def configure_runtime(device, cpu_threads):
    require(type(cpu_threads) is int and 1 <= cpu_threads <= 8, 'CPU threads1..8 required')
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'requested CUDA is unavailable')
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    # Reflection-padding CUDA backward does not supply strict deterministic
    # algorithms on every supported PyTorch. Record actual settings; the
    # protocol requires paired initialization and data streams, not a false
    # claim of bitwise deterministic CUDA training.


def environment(device):
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(), device=str(device),
                  torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  TF32_cudnn=torch.backends.cudnn.allow_tf32, mixed_precision=False,
                  backend_thread_environment={key: os.environ.get(key) for key in
                      ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                       'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
    if device.type == 'cuda':
        prop = torch.cuda.get_device_properties(device)
        result['GPU'] = dict(name=prop.name, total_memory_bytes=prop.total_memory,
                             capability=list(torch.cuda.get_device_capability(device)))
    return result


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def resource_receipt(path, device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(peak_process_RSS_bytes=int(rss if sys.platform == 'darwin' else rss*1024),
                  retained_bytes=sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file()),
                  retained_files=sum(1 for p in Path(path).rglob('*') if p.is_file()))
    if device.type == 'cuda':
        result.update(peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


class Phase:
    def __init__(self, root, phase, cfg, spec, config_path, spec_path, device):
        self.root, self.phase, self.device = Path(root).resolve(), phase, device
        self.path = self.root/phase
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.mkdir(exist_ok=False)
        self.started = time.monotonic()
        self.code = source_metadata()
        self.inputs = [file_metadata(config_path), file_metadata(spec_path)]
        self.run_id = cfg['run_id']
        self.dataset = None
        write_json(self.path/'started.json', dict(plan_id=PLAN_ID, run_id=self.run_id,
                   phase=phase, status='in_progress', retry=False,
                   started_at=datetime.now(timezone.utc).isoformat(),
                   source_metadata=self.code, control_metadata=self.inputs,
                   environment=environment(device)))
        write_json(self.path/'config.json', cfg)
        write_json(self.path/'spec.json', spec)

    def open_data(self, cfg):
        require(self.dataset is None, 'one audited dataset per phase required')
        self.dataset = data.QueryRelationDataset(cfg['data'], self.path/'access-audit.jsonl')
        return self.dataset

    def progress(self, value):
        replace_json(self.path/'progress.json', dict(plan_id=PLAN_ID, run_id=self.run_id,
                     phase=self.phase, status='in_progress', **value))

    def finish(self, payload, *, failure=None):
        current_code = source_metadata()
        if failure is None:
            require(current_code == self.code, 'scientific implementation changed during phase')
            for row in self.inputs:
                check_metadata(row)
            if self.dataset is not None:
                data.stable(self.dataset.source_metadata())
        if self.dataset is not None:
            self.dataset.close()
            audit = self.dataset.audit_summary()
            write_json(self.path/'access-summary.json', audit)
        else:
            audit = None
        result = dict(plan_id=PLAN_ID, run_id=self.run_id, phase=self.phase,
                      status='failed' if failure else 'completed',
                      wall_seconds=time.monotonic()-self.started,
                      source_metadata=self.code, current_source_metadata=current_code,
                      control_metadata=self.inputs, audit=audit,
                      resources=resource_receipt(self.path, self.device), **payload)
        if failure is not None:
            result['failure'] = failure
        write_json(self.path/'status.json', result)
        with (self.path/'exit-code.txt').open('x') as handle:
            handle.write('1\n' if failure else '0\n')
        return result


def model_for(arm, seed, stats, device):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    return core.RelationModel(arm, torch.as_tensor(stats['mean'], dtype=torch.float32),
                              torch.as_tensor(stats['std'], dtype=torch.float32)).to(device)


def tensors(frame, device, target=None, *, code=0, arm='P_F'):
    transformed, labels = data.transform_d4(frame, target, code=code, shift_tokens=arm == 'E_shift')
    common = torch.from_numpy(np.ascontiguousarray(transformed['common12'])).unsqueeze(0).to(device)
    tokens = torch.from_numpy(np.ascontiguousarray(data.token_grid(transformed['tokens']))).unsqueeze(0)
    tokens = tokens.to(device=device, dtype=torch.float32)
    donor = torch.from_numpy(np.ascontiguousarray(transformed['donor'][..., 0])).to(device)
    donor = donor.to(torch.float32).unsqueeze(0).unsqueeze(0)/255
    y = None
    if labels is not None:
        y = torch.from_numpy(np.ascontiguousarray(labels[..., 0])).to(device)
        y = y.to(torch.float32).unsqueeze(0).unsqueeze(0)/255
    return common, tokens, donor, y


def scalar_loss(loss):
    result = {key: float(value.detach().mean().cpu()) for key, value in loss.items()}
    require(all(math.isfinite(value) for value in result.values()), 'nonfinite objective; no automatic retry')
    return result


def diagnostic_scalars(diagnostics):
    return {key: plain(value) for key, value in diagnostics.items() if key != 'rhs'}


def final_bytes(donor, residual):
    """Native-byte output; the other two canonical RGB markers are untouched."""
    require(donor.dtype == np.uint8 and donor.shape == (512, 512, 3), 'native donor RGB required')
    residual = np.asarray(residual)
    require(residual.shape == (512, 512) and residual.dtype == np.float64
            and np.isfinite(residual).all(), 'full512 finite float64 residual required')
    before = donor[..., 0].astype(np.float64)/255 + residual
    clipped = np.clip(before, 0., 1.)
    image = donor.copy()
    image[..., 0] = np.rint(255*clipped).astype(np.uint8)
    receipt = dict(pixels=512*512, below_zero=int(np.count_nonzero(before < 0)),
                   above_one=int(np.count_nonzero(before > 1)), preclip_mean=float(before.mean()),
                   postclip_mean=float(clipped.mean()),
                   rounding_mean_change=float(image[..., 0].mean()/255-clipped.mean()))
    require(np.array_equal(image[..., 1:], donor[..., 1:]), 'CD3/DAPI byte preservation failed')
    return image, receipt


def checkpoint(path, model, *, arm, seed, split, epoch, train_indices, scales,
               synthetic_parameter_fixture=False):
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = dict(schema='query-relation-weights-v1', plan_id=PLAN_ID, arm=arm, seed=seed,
                   split=split, epoch=epoch, training_indices=list(train_indices), loss_scales=scales,
                   state_dict=state, synthetic_parameter_fixture=synthetic_parameter_fixture,
                   optimizer_state_retained=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return file_metadata(path)


def load_checkpoint(row, arm, seed, device, *, synthetic=False):
    path = check_metadata(row)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    require(saved.get('schema') == 'query-relation-weights-v1' and saved.get('plan_id') == PLAN_ID
            and saved.get('arm') == arm and saved.get('seed') == seed
            and saved.get('synthetic_parameter_fixture') is synthetic,
            'fixed-arm checkpoint binding differs')
    # Buffer names are deliberately obtained from the model schema, while the
    # exact stored statistics are subsequently restored by strict state load.
    stats = dict(mean=np.zeros(1536), std=np.ones(1536))
    model = model_for(arm, seed, stats, device)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    return model, saved


def inference(model, frame, arm, device):
    common, tokens, _, _ = tensors(frame, device, arm=arm)
    with torch.no_grad():
        residual, diagnostic = model(common, tokens, inference=True)
    residual = np.ascontiguousarray(residual.detach().cpu().numpy()[0, 0], dtype=np.float64)
    rhs = diagnostic.get('rhs')
    rhs = np.empty(0, np.float64) if rhs is None else np.ascontiguousarray(
        rhs.detach().cpu().numpy()[0, 0], dtype=np.float64)
    require(rhs.shape in ((0,), (512, 512)) and np.isfinite(rhs).all(), 'invalid native RHS')
    image, clipping = final_bytes(frame['donor'], residual)
    return dict(image=image, residual=residual, rhs=rhs), clipping, diagnostic_scalars(diagnostic)


def require_no_outer_targets(dataset):
    audit = dataset.audit_summary()
    require(audit.get('outer_target_reads') == 0, 'outer target opened before independent freeze acceptance')
    return audit


def quality_for(cfg):
    return scoring.load_quality(cfg.get('quality_directory'))


def profile(cfg, spec, phase, device):
    dataset = phase.open_data(cfg)
    frame = dataset.read_frame(1)
    require(frame['he'].shape == (3, 512, 512), 'native training-index1 full512 required')
    stats = dict(mean=np.zeros(1536), std=np.ones(1536))
    quality = quality_for(cfg)
    rng = np.random.default_rng(SEEDS[0])
    a = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    native = quality(a, b)['ssim_channels'][0]
    ta = torch.tensor(a[..., 0].copy(), device=device, dtype=torch.float64)[None, None]/255
    tb = torch.tensor(b[..., 0].copy(), device=device, dtype=torch.float64)[None, None]/255
    differentiable = float(core.panck_ssim(ta, tb).detach().cpu()[0])
    require(abs(differentiable-native) <= 1e-8, 'differentiable/native canonical SSIM parity failed')
    synthetic_target = torch.linspace(.1, .9, 512*512, device=device).reshape(1, 1, 512, 512)
    rows = []
    for arm in ARMS:
        sync(device); started = time.monotonic()
        model = model_for(arm, SEEDS[0], stats, device)
        common, tokens, donor, _ = tensors(frame, device, arm=arm)
        initial, diag = model(common, tokens)
        require(torch.count_nonzero(initial).item() == 0, 'all final heads must initialize to exact donor no-op')
        initial_residual = np.ascontiguousarray(initial.detach().cpu().numpy()[0, 0], dtype=np.float64)
        initial_rhs = diag.get('rhs')
        initial_rhs = np.empty(0, np.float64) if initial_rhs is None else np.ascontiguousarray(
            initial_rhs.detach().cpu().numpy()[0, 0], dtype=np.float64)
        initial_image, initial_clipping = final_bytes(frame['donor'], initial_residual)
        require(np.array_equal(initial_image, frame['donor']), 'initial quantized output must be donor bytes')
        initial_file = save_npz(phase.path/'noops'/f'{arm}.npz', image=initial_image,
                                residual=initial_residual, rhs=initial_rhs)
        initial_diagnostics = diagnostic_scalars(diag)
        initial_loss = core.loss_scales(torch.clamp(donor+initial, 0, 1), synthetic_target)
        sum(value.mean() for value in initial_loss.values()).backward()
        require(all(param.grad is None or torch.isfinite(param.grad).all().item()
                    for param in model.parameters()), 'nonfinite initial synthetic backward')
        model.zero_grad(set_to_none=True)
        # This deterministic fixture is never optimized or counted as a fit.
        generator = torch.Generator(device='cpu').manual_seed(SEEDS[0])
        with torch.no_grad():
            for param in model.parameters():
                param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype).to(device)*.01)
        nonzero, diag = model(common, tokens)
        require(torch.count_nonzero(nonzero).item() > 0, 'synthetic fixture failed to exercise a nonzero graph/output')
        fixture_loss = core.loss_scales(torch.clamp(donor+nonzero, 0, 1), synthetic_target)
        sum(value.mean() for value in fixture_loss.values()).backward()
        require(all(param.grad is None or torch.isfinite(param.grad).all().item()
                    for param in model.parameters()), 'nonfinite nonzero synthetic backward')
        saved_weights = checkpoint(phase.path/'weights'/f'{arm}.pt', model, arm=arm,
            seed=SEEDS[0], split='engineering-synthetic-only', epoch=0, train_indices=[], scales=None,
            synthetic_parameter_fixture=True)
        arrays, clipping, diagnostics = inference(model.eval(), frame, arm, device)
        output = save_npz(phase.path/'outputs'/f'{arm}.npz', **arrays)
        reloaded = load_npz(output['path'])
        require(set(reloaded) == {'image', 'residual', 'rhs'}
                and all(np.array_equal(reloaded[key], value) for key, value in arrays.items()),
                'full512 native output save/reload differs')
        restored, _ = load_checkpoint(saved_weights, arm, SEEDS[0], device, synthetic=True)
        repeated, _, _ = inference(restored, frame, arm, device)
        require(all(np.array_equal(repeated[key], arrays[key]) for key in arrays),
                'native parameter save/reload changes inference')
        sync(device)
        rows.append(dict(arm=arm, seed=SEEDS[0], original_index=1, file=output,
                         weights=saved_weights, clipping=clipping, diagnostics=diagnostics,
                         initial_noop_file=initial_file, initial_noop_clipping=initial_clipping,
                         initial_noop_diagnostics=initial_diagnostics,
                         parameter_counts=model.parameter_counts(), zero_initial_noop=True,
                         synthetic_nonzero=True, output_save_reload_exact=True,
                         weights_save_reload_exact=True, wall_seconds=time.monotonic()-started))
        phase.progress(dict(completed_arms=len(rows), arms=8, new_fits=0, optimizer_steps=0))
        del model, restored, common, tokens, donor, initial, nonzero
    audit = require_no_outer_targets(dataset)
    require(audit.get('training_target_reads') == 0 and audit.get('validation_target_reads') == 0,
            'engineering profile must not open any actual target')
    manifest = dict(plan_id=PLAN_ID, status='complete_native_full512_engineering_profile',
                    phase='profile', original_index=1, outputs=rows,
                    source_metadata=dataset.source_metadata(), code_metadata=phase.code, audit=audit,
                    new_fits=0, optimizer_steps=0, actual_target_reads=0,
                    profile_RGB_calls=1, synthetic_targets_only=True,
                    canonical_ssim_parity=dict(native=native, differentiable=differentiable,
                                               absolute_error=abs(native-differentiable), tolerance=1e-8))
    receipt = write_json(phase.path/'profile-manifest.json', manifest)
    return dict(manifest=receipt, completed_arms=8, generated_outputs=8,
                new_fits=0, optimizer_steps=0, actual_target_reads=0,
                efficacy_inference=False, profile_RGB_calls=1)


def check_profile_acceptance(cfg, phase):
    path = cfg.get('profile_acceptance')
    require(path, 'independent full512 profile_acceptance path required before the64 fits')
    accepted = read_json(check_metadata(file_metadata(path)))
    require(accepted.get('status') == 'passed' and accepted.get('phase') == 'profile'
            and accepted.get('independent_consumer') is True,
            'independent native full512 profile acceptance required')
    bound = accepted.get('profile_manifest')
    require(isinstance(bound, dict), 'profile acceptance must bind profile_manifest metadata')
    manifest = read_json(check_metadata(bound))
    require(manifest.get('plan_id') == PLAN_ID and manifest.get('phase') == 'profile'
            and manifest.get('new_fits') == 0 and manifest.get('optimizer_steps') == 0
            and manifest.get('actual_target_reads') == 0 and len(manifest.get('outputs', [])) == 8,
            'accepted profile is incomplete or counted actual training')
    require(manifest.get('source_metadata') == phase.dataset.source_metadata(),
            'profile/source binding differs')
    require(manifest.get('code_metadata') == phase.code, 'accepted engineering profile implementation differs')
    for row in manifest['outputs']:
        check_metadata(row['file']); check_metadata(row['weights']); check_metadata(row['initial_noop_file'])
    return file_metadata(path)


def subset_definitions():
    return [(fold, tuple(i for i in TRAINING if i % 4 != fold),
             tuple(i for i in TRAINING if i % 4 == fold)) for fold in (1, 2, 3)] + [(0, TRAINING, ())]


def mean_scales(raw, indices):
    require(indices and set(indices) <= set(TRAINING), 'train-only scale population required')
    return {key: max(1e-8, math.fsum(raw[index][key] for index in indices)/len(indices))
            for key in LOSS_FIELDS}


def validate_fit(model, frames, targets, indices, scales, arm, seed, fold, epoch, device):
    rows = []
    model.eval()
    with torch.no_grad():
        for index in indices:
            common, tokens, donor, target = tensors(frames[index], device, targets[index], arm=arm)
            residual, diagnostic = model(common, tokens)
            output = torch.clamp(donor+residual, 0, 1)
            loss = scalar_loss(core.normalize_loss(output, target, donor, scales))
            rows.append(dict(arm=arm, seed=seed, fold=fold, epoch=epoch, original_index=index,
                             losses=loss, selection_criterion=loss['image'],
                             scale_training_indices=[i for i in TRAINING if i % 4 != fold],
                             diagnostics=diagnostic_scalars(diagnostic)))
    model.train()
    return rows


def fit_one(arm, seed, fold, indices, held, stats, scales, frames, targets, phase, device):
    name = 'final24' if fold == 0 else f'inner{fold}'
    path = phase.path/'fits'/name/arm/f'seed-{seed}'
    path.mkdir(parents=True, exist_ok=False)
    model = model_for(arm, seed, stats, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, betas=(.9, .999))
    snapshots, validation = [], []
    steps = 0
    sync(device); fit_started = time.monotonic()
    for epoch in range(1, EPOCHS+1):
        sync(device); started = time.monotonic()
        rate = learning_rate(epoch)
        for group in optimizer.param_groups:
            group['lr'] = rate
        order = paired_order(indices, seed, fold, epoch)
        transforms = [data.seeded_d4(seed, epoch, index) for index in order]
        sums = None
        max_solver_residual = 0.
        for index, code in zip(order, transforms):
            common, tokens, donor, target = tensors(frames[index], device, targets[index], code=code, arm=arm)
            optimizer.zero_grad(set_to_none=True)
            residual, diagnostic = model(common, tokens)
            output = torch.clamp(donor+residual, 0, 1)
            loss = core.normalize_loss(output, target, donor, scales)
            values = scalar_loss(loss)
            loss['total'].mean().backward()
            require(all(param.grad is None or torch.isfinite(param.grad).all().item()
                        for param in model.parameters()),
                    f'nonfinite gradient: {name}/{arm}/{seed}/epoch{epoch}; dependent work stopped')
            optimizer.step()
            require(all(torch.isfinite(param).all().item() for param in model.parameters()),
                    f'nonfinite updated parameter: {name}/{arm}/{seed}/epoch{epoch}')
            steps += 1
            if sums is None:
                sums = {key: [] for key in values}
            for key, value in values.items():
                sums[key].append(value)
            max_solver_residual = max(max_solver_residual, float(diagnostic.get('relative_residual_max', 0.)))
        sync(device)
        trajectory = dict(epoch=epoch, learning_rate=rate, images=len(indices), parameter_updates=steps,
                          order=order, D4_codes=transforms,
                          losses={key: math.fsum(value)/len(value) for key, value in sums.items()},
                          maximum_training_solve_relative_residual=max_solver_residual,
                          wall_seconds=time.monotonic()-started)
        append_json(path/'trajectory.jsonl', trajectory)
        if epoch in SNAPSHOTS:
            saved = checkpoint(path/f'epoch-{epoch:03d}.pt', model, arm=arm, seed=seed,
                               split=name, epoch=epoch, train_indices=indices, scales=scales)
            snapshots.append(dict(epoch=epoch, file=saved))
            if held:
                validation.extend(validate_fit(model, frames, targets, held, scales,
                                               arm, seed, fold, epoch, device))
        phase.progress(dict(active_fit=dict(arm=arm, seed=seed, fold=fold, epoch=epoch),
                            active_fit_parameter_updates=steps))
    validation_file = write_json(path/'validation.json', validation)
    result = dict(arm=arm, seed=seed, fold=fold, split=name, training_indices=list(indices),
                  held_indices=list(held), epochs=EPOCHS, parameter_updates=steps,
                  parameter_counts=model.parameter_counts(), snapshot_epochs=list(SNAPSHOTS),
                  snapshots=snapshots, trajectory=file_metadata(path/'trajectory.jsonl'),
                  validation=validation_file, loss_scales=scales,
                  optimizer_recovery_retained=False, optimizer_state_retained=False,
                  wall_seconds=time.monotonic()-fit_started,
                  retained_parameter_snapshot_bytes=sum(row['file']['size_bytes'] for row in snapshots))
    write_json(path/'fit.json', result)
    del optimizer, model
    return result, validation


def select_epochs(validation):
    selections = []
    for arm in ARMS:
        candidates = []
        for epoch in SNAPSHOTS:
            rows = [row for row in validation if row['arm'] == arm and row['epoch'] == epoch]
            keys = {(row['seed'], row['original_index']) for row in rows}
            require(len(rows) == 48 and keys == {(s, i) for s in SEEDS for i in TRAINING},
                    'epoch selection requires all24 inner-held images and both seeds exactly once')
            value = math.fsum(row['selection_criterion'] for row in rows)/48
            candidates.append(dict(epoch=epoch, image_only_mean=value, observations=48))
        chosen = min(candidates, key=lambda row: (row['image_only_mean'], row['epoch']))
        selections.append(dict(arm=arm, epoch=chosen['epoch'], candidates=candidates,
                               seeds=list(SEEDS), criterion='equal-image two-term image loss only',
                               exact_tie='earliest epoch', outer_targets_used=False))
    return selections


def training_witness(cfg, phase, frames, targets):
    quality, rows = quality_for(cfg), []
    for index in TRAINING:
        donor = frames[index]['donor']
        error = (targets[index][..., 0].astype(np.float64)-donor[..., 0].astype(np.float64))/255
        residual = np.ascontiguousarray(error-error.mean())
        image, clipping = final_bytes(donor, residual)
        row = scoring.score_image(image, targets[index], donor, quality, clipping=clipping)
        row.update(original_index=index, privileged_training_only=True,
                   deployed_algorithm=False, ODS_upper_bound=False,
                   preclip_residual_mean=float(residual.mean()))
        write_json(phase.path/'training-witness'/f'{index:02d}.json', row)
        rows.append(row)
    pooled = scoring.aggregate_training_witness(rows)
    return write_json(phase.path/'training-witness.json', dict(
        status='complete_training_only_privileged_feasible_witness', image_rows=rows,
        pooled=pooled, training_RGB_calls=24, outer_RGB_calls=0, outer_targets_used=False,
        actual_target_reads_from_cache=0, actual_training_target_reads_owned_by_preload=24,
        feasible_example=True, deployed_algorithm=False, ODS_upper_bound=False))


def train(cfg, spec, phase, device):
    dataset = phase.open_data(cfg)
    accepted_profile = check_profile_acceptance(cfg, phase)
    frames, targets = {}, {}
    for index in TRAINING:
        frames[index] = dataset.read_frame(index)
        targets[index] = dataset.read_target(index, purpose='training')
    audit = require_no_outer_targets(dataset)
    require(audit.get('training_target_reads') == 24, 'exactly24 actual training target opens required')
    stats_by_fold, scales_by_fold, statistic_files = {}, {}, []
    raw = {}
    with torch.no_grad():
        for index in TRAINING:
            _, _, donor, target = tensors(frames[index], device, targets[index])
            raw[index] = scalar_loss(core.loss_scales(donor, target))
    for fold, indices, _ in subset_definitions():
        stats = data.fit_token_statistics(dataset, indices, cached_frames=frames)
        scales = mean_scales(raw, indices)
        stats_by_fold[fold], scales_by_fold[fold] = stats, scales
        receipt = write_json(phase.path/'statistics'/f'fold-{fold}.json',
            dict(fold=fold, training_indices=list(indices), token_statistics=stats,
                 donor_raw_per_image={str(i): raw[i] for i in indices},
                 loss_scales=scales, floor_after_equal_image_mean=1e-8,
                 outer_and_inner_held_used=False))
        statistic_files.append(dict(fold=fold, file=receipt))
    witness = training_witness(cfg, phase, frames, targets)
    fits, validation = [], []
    for fold, indices, held in subset_definitions()[:3]:
        for arm in ARMS:
            for seed in SEEDS:
                fit, held_rows = fit_one(arm, seed, fold, indices, held, stats_by_fold[fold],
                    scales_by_fold[fold], frames, targets, phase, device)
                fits.append(fit); validation.extend(held_rows)
                phase.progress(dict(fits_completed=len(fits), fits_planned=64))
    selections = select_epochs(validation)
    selection_file = write_json(phase.path/'epoch-selection.json', selections)
    for arm in ARMS:
        for seed in SEEDS:
            fit, _ = fit_one(arm, seed, 0, TRAINING, (), stats_by_fold[0], scales_by_fold[0],
                             frames, targets, phase, device)
            fits.append(fit)
            phase.progress(dict(fits_completed=len(fits), fits_planned=64))
    selected = []
    choices = {row['arm']: row['epoch'] for row in selections}
    for fit in fits:
        if fit['fold'] == 0:
            entry = next(row for row in fit['snapshots'] if row['epoch'] == choices[fit['arm']])
            selected.append(dict(arm=fit['arm'], seed=fit['seed'], selected_epoch=entry['epoch'],
                                 file=entry['file'], training_indices=list(TRAINING),
                                 loss_scales=fit['loss_scales']))
    require(len(fits) == 64 and len(selected) == 16, 'fixed64-fit/16-selected-model coverage failed')
    audit = require_no_outer_targets(dataset)
    manifest = dict(plan_id=PLAN_ID, status='all64_fits_completed_16_selected_weights_frozen',
                    fits_planned=64, fits_completed=64, fits=fits, selected_weights=selected,
                    epoch_selection=selection_file, statistics=statistic_files,
                    training_witness=witness, source_metadata=dataset.source_metadata(), audit=audit,
                    accepted_profile=accepted_profile, parameter_snapshots=64*4,
                    parameter_snapshot_bytes=sum(fit['retained_parameter_snapshot_bytes'] for fit in fits),
                    optimizer_recovery_retained=False, optimizer_state_retained=False,
                    actual_training_target_reads=24, actual_outer_target_reads=0,
                    privileged_training_RGB_calls=24, outer_RGB_calls=0,
                    initialization_and_data_streams_paired=True, outer_epoch_selection=False)
    receipt = write_json(phase.path/'training-manifest.json', manifest)
    return dict(manifest=receipt, fits_completed=64, selected_models=16,
                parameter_snapshots=256, optimizer_recovery_retained=False,
                parameter_updates=sum(fit['parameter_updates'] for fit in fits),
                actual_training_target_reads=24, actual_outer_target_reads=0,
                privileged_training_RGB_calls=24, outer_RGB_calls=0)


def check_selected_weights(rows):
    require(isinstance(rows, list) and len(rows) == 16
            and {(row.get('arm'), row.get('seed')) for row in rows} == {(a, s) for a in ARMS for s in SEEDS},
            'exact16 arm/seed selected weights required')
    for row in rows:
        check_metadata(row['file'])
        require(row.get('selected_epoch') in SNAPSHOTS and row.get('training_indices') == list(TRAINING),
                'selected weights must be a final24 fixed snapshot')


def generate(cfg, spec, phase, device):
    dataset = phase.open_data(cfg)
    training_file = file_metadata(cfg.get('training_manifest', phase.root/'train/training-manifest.json'))
    trained = read_json(training_file['path'])
    require(trained.get('plan_id') == PLAN_ID
            and trained.get('status') == 'all64_fits_completed_16_selected_weights_frozen'
            and trained.get('fits_completed') == 64 and trained.get('parameter_snapshots') == 256
            and trained.get('actual_outer_target_reads') == 0,
            'complete fixed64-fit training manifest required')
    require(trained.get('source_metadata') == dataset.source_metadata(), 'training/generation source binding differs')
    weights = trained.get('selected_weights')
    check_selected_weights(weights)
    frames = {index: dataset.read_frame(index) for index in OUTER}
    outputs = []
    for weight in weights:
        arm, seed = weight['arm'], weight['seed']
        model, saved = load_checkpoint(weight['file'], arm, seed, device)
        require(saved.get('split') == 'final24' and saved.get('epoch') == weight['selected_epoch']
                and saved.get('training_indices') == list(TRAINING), 'selected checkpoint contents differ')
        for index in OUTER:
            sync(device); started = time.monotonic()
            arrays, clipping, diagnostics = inference(model, frames[index], arm, device)
            path = phase.path/'outputs'/arm/f'seed-{seed}'/f'{index:02d}.npz'
            output_file = save_npz(path, **arrays)
            sync(device)
            outputs.append(dict(arm=arm, seed=seed, original_index=index, file=output_file,
                                selected_weight=weight['file'], selected_epoch=weight['selected_epoch'],
                                clipping=clipping, diagnostics=diagnostics,
                                residual_shape=[512, 512], residual_dtype='float64',
                                rhs_shape=list(arrays['rhs'].shape), rhs_dtype='float64',
                                image_shape=[512, 512, 3], image_dtype='uint8',
                                CD3_DAPI_exact=True,
                                no_op=bool(np.array_equal(arrays['image'], frames[index]['donor'])),
                                parameter_counts=model.parameter_counts(),
                                wall_seconds=time.monotonic()-started))
            phase.progress(dict(generated_outputs=len(outputs), outputs_planned=128))
        del model
    require(len(outputs) == 128 and len({(r['arm'], r['seed'], r['original_index']) for r in outputs}) == 128,
            'all128 exact output keys required')
    audit = require_no_outer_targets(dataset)
    require(audit.get('training_target_reads') == 0 and audit.get('validation_target_reads') == 0,
            'generation must not open actual targets')
    check_metadata(training_file)
    check_selected_weights(weights)
    manifest = dict(plan_id=PLAN_ID, status='all128_outputs_and_selected_weights_frozen', phase='generate',
                    outputs=outputs, selected_weights=weights, training_manifest=training_file,
                    source_metadata=dataset.source_metadata(), code_metadata=phase.code,
                    audit=audit, target_reads=0, new_fits=0, RGB_calls=0,
                    array_contract=dict(image='uint8[512,512,3] canonical panCK/CD3/DAPI',
                        residual='float64[512,512] before clipping',
                        rhs='float64[512,512] for S/E; empty float64[0] for P/M'),
                    scoring_authorized_only_after_independent_generation_acceptance=True)
    receipt = write_json(phase.path/'generation-manifest.json', manifest)
    return dict(manifest=receipt, generated_outputs=128, selected_models=16,
                target_reads=0, RGB_calls=0, new_fits=0,
                independent_generation_acceptance='pending')


def score(cfg, spec, phase, device):
    dataset = phase.open_data(cfg)
    manifest_file = file_metadata(cfg.get('generation_manifest', phase.root/'generate/generation-manifest.json'))
    acceptance_path = cfg.get('generation_acceptance')
    require(acceptance_path, 'independent generation_acceptance required before first outer target')
    acceptance_file = file_metadata(acceptance_path)
    freeze = dict(manifest=manifest_file, generation_acceptance=acceptance_file)
    # Validate the entire physical freeze, not merely a caller-supplied flag.
    data.validate_freeze_receipt(freeze, dataset.source_metadata())
    manifest = read_json(manifest_file['path'])
    check_selected_weights(manifest['selected_weights'])
    require(cfg.get('old229'), 'unchanged old229 scalar reference required before any outer target')
    old_file = file_metadata(cfg['old229'])
    old = read_json(old_file['path'])
    require(isinstance(old, dict) and len(old.get('all229_pooled_endpoints', [])) == 229
            and len(old.get('all1832_image_endpoints', [])) == 1832,
            'complete literal old229 scalar source required')
    quality = quality_for(cfg)
    rows, calls = [], 0
    for index in OUTER:
        frame = dataset.read_frame(index)
        target = dataset.read_target(index, purpose='scoring', freeze_receipt=freeze)
        entries = [row for row in manifest['outputs'] if row['original_index'] == index]
        require(len(entries) == 16, 'all16 frozen outputs/image required')
        for entry in entries:
            saved = load_npz(check_metadata(entry['file']))
            require(set(saved) == {'image', 'residual', 'rhs'}, 'generation archive members differ')
            row = scoring.score_image(saved['image'], target, frame['donor'], quality,
                                      clipping=entry['clipping'])
            calls += 1
            row.update(arm=entry['arm'], seed=entry['seed'], original_index=index,
                       frozen_output=entry['file'])
            write_json(phase.path/'image-rows'/entry['arm']/f"seed-{entry['seed']}"/f'{index:02d}.json', row)
            rows.append(row)
        phase.progress(dict(scored_images=len(rows)//16, image_rows=len(rows), RGB_calls=calls,
                            actual_outer_target_reads=dataset.audit_summary().get('outer_target_reads')))
    require(calls == 128, 'exact128 native RGB calls required')
    audit = dataset.audit_summary()
    require(audit.get('outer_target_reads') == 8 and audit.get('training_target_reads') == 0
            and audit.get('validation_target_reads') == 0, 'exactlyeight new outer target opens required')
    checked = scoring.check_saved_scalars(rows)
    analysis = scoring.analyze(rows, old)
    require(analysis.get('status') == 'complete_saved_scalar_analysis', 'complete clipping/scalar evidence required')
    check_metadata(manifest_file); check_metadata(acceptance_file); check_metadata(old_file)
    analysis.update(plan_id=PLAN_ID, frozen_generation=manifest_file,
                    independent_generation_acceptance=acceptance_file, unchanged_old229_source=old_file,
                    actual_new_outer_target_reads=8, prior_correspondence_target_reads=9,
                    correspondence_family_cumulative_target_reads=17,
                    actual_training_target_reads_this_phase=0, outer_RGB_calls=128,
                    training_privileged_RGB_calls_this_phase=0, new_fits_this_phase=0,
                    actual_scoring_audit=audit, canonical_quality_metadata=quality.metadata)
    result = write_json(phase.path/'analysis.json', analysis)
    scalar_check = write_json(phase.path/'saved-scalar-consistency.json',
                             {key: value for key, value in checked.items() if key != 'groups'})
    groups = write_json(phase.path/'groups.json', checked['groups'])
    return dict(analysis=result, groups=groups, saved_scalar_consistency=scalar_check,
                scored_outputs=128, pooled_groups=16, actual_new_outer_target_reads=8,
                prior_correspondence_target_reads=9, cumulative_target_reads=17,
                actual_training_target_reads=0, outer_RGB_calls=128, new_fits=0,
                scientific_decision=analysis['decision'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('profile', 'train', 'generate', 'score'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--spec', type=Path, default=ROOT/'development/query-relation-reconstruction/spec.json')
    args = parser.parse_args(argv)
    cfg, spec = read_json(args.config), read_json(args.spec)
    validate_spec(spec)
    require(isinstance(cfg, dict) and isinstance(cfg.get('run_id'), str) and cfg['run_id'].strip()
            and isinstance(cfg.get('data'), dict), 'runner config requires run_id and explicit data object')
    require(not set(cfg) & {'epochs', 'seeds', 'arms', 'learning_rate', 'training_indices',
                            'outer_indices', 'screening', 'snapshots', 'amplitude'},
            'scientific runner overrides are prohibited; use the accepted fixed spec')
    device = torch.device(args.device)
    configure_runtime(device, cfg.get('cpu_threads', 8))
    phase = Phase(args.output, args.phase, cfg, spec, args.config, args.spec, device)
    try:
        result = globals()[args.phase](cfg, spec, phase, device)
        final = phase.finish(result)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc(), automatic_retry=False,
                       dependent_work_stopped=True)
        try:
            phase.finish({}, failure=failure)
        except BaseException as receipt_error:
            print('failure receipt error: '+repr(receipt_error), file=sys.stderr)
        raise
    print(json.dumps(dict(phase=args.phase, status=final['status'],
                          status_file=str(phase.path/'status.json')), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
