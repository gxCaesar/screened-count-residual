"""Finite train-only pixel selection followed by frozen common-endpoint readout.

Run with --config CONFIG [--profile-only]. All source, cache, image and checkpoint
paths are supplied explicitly; this runner downloads nothing. The parent must
also enforce the explicitly authorized limit with its finite process wrapper.
"""
import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import torch

try:
    from . import pixel_comparator as pc
except ImportError:
    import pixel_comparator as pc


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value, exclusive=True, allow_nan=False):
    with Path(path).open('x' if exclusive else 'w') as handle:
        json.dump(json_value(value), handle, indent=2, allow_nan=allow_nan)
        handle.write('\n')


def regular(path, directory=False):
    path = Path(path).absolute()
    pc.require(path.exists() and not path.is_symlink() and path.resolve() == path,
               'path must exist, be canonical and not a symlink: '+str(path))
    pc.require(path.is_dir() if directory else path.is_file(), 'unexpected path kind: '+str(path))
    return path


def validate_config(cfg):
    for key in ('manifest','uni_checkpoint'):
        regular(cfg[key])
    for key in ('data_root','metrics_root','fml_source','uni_source','torch_home'):
        regular(cfg[key], directory=True)
    pc.require(cfg.get('max_seconds') in (1800, 2700), 'max_seconds requires a registered 1800 or 2700 second card')
    pc.require(set(cfg['seeds']) == {'0','1'}, 'both registered seeds are required')
    for seed, methods in cfg['seeds'].items():
        expected = set(pc.METHOD_ORDER if seed == '0' else pc.METHOD_ORDER[:2])
        pc.require(set(methods) == expected, 'seed method membership differs')
        for checkpoint in methods.values():
            regular(checkpoint)
    pc.require(Path(cfg['uni_checkpoint']).resolve() == Path(cfg['seeds']['0']['UNI2IF']).resolve(),
               'UNI2IF differs from frozen source-selected checkpoint')
    regular(Path(cfg['torch_home'])/'hub/checkpoints/alexnet-owt-7be5be79.pth')
    output = Path(cfg['output']).absolute()
    pc.require(not output.exists() and not output.is_symlink(), 'preserve existing output directory')
    pc.require(output.parent.resolve(strict=True) == output.parent, 'output parent must be canonical')
    for key in ('data_root','metrics_root','fml_source','uni_source','torch_home'):
        pc.require(not output.is_relative_to(Path(cfg[key])), 'output must be outside data/source/cache')
    return output


def offline():
    def reject(*args, **kwargs):
        raise RuntimeError('downloads forbidden; existing offline cache required')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
    torch.hub.download_url_to_file = reject
    # torchvision / LPIPS delegate network fetch to torch.hub; fail if cache absent.


def runtime(cfg):
    os.environ['TORCH_HOME'] = cfg['torch_home']
    offline()
    local = str(Path(__file__).resolve().parent)
    for path in (cfg['uni_source'], cfg['fml_source'], local, cfg['metrics_root']):
        sys.path.insert(0, path)
    from scoring_interface import validate_scoring_interface
    interface = validate_scoring_interface(cfg['metrics_root'])
    # Legacy scorers prepend cwd on import. Restore the explicit source order.
    sys.path[:0] = [cfg['metrics_root'],local,cfg['fml_source'],cfg['uni_source']]
    # Load the actual trained-source FML implementation explicitly, while leaving
    # the newly deployed evaluator and runner helpers ahead of older source copies.
    component_path = regular(Path(cfg['fml_source'])/'s2_components.py')
    component_spec = importlib.util.spec_from_file_location('s2_components',component_path)
    components = importlib.util.module_from_spec(component_spec)
    sys.modules['s2_components'] = components
    component_spec.loader.exec_module(components)
    import metric_panel as mp
    import sweep_panel as sp
    # Scorers may live in an older snapshot containing an evaluator of the same
    # name. Bind the evaluator to this deployed runner, independent of sys.path.
    evaluator_path = regular(Path(local)/'eval_unistainnet_if.py')
    evaluator_spec = importlib.util.spec_from_file_location('eval_unistainnet_if', evaluator_path)
    evaluator = importlib.util.module_from_spec(evaluator_spec)
    sys.modules['eval_unistainnet_if'] = evaluator
    evaluator_spec.loader.exec_module(evaluator)
    from eval_unistainnet_if import center_items, evaluate, HemitUNIStainNet
    from curve_evidence import selected_train, validate_manifest
    from unistainnet_if_uni2h import MatchedUNIStainNetIFTrainer
    import cv2
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    manifest = json.loads(Path(cfg['manifest']).read_text())
    validate_manifest(Path(cfg['data_root']), manifest)
    pc.require(len(manifest['val']) == 630, 'common validation population must contain 630 images')
    selected = selected_train(manifest['train'], 300)
    pc.population(manifest['train'], selected, manifest['val'])
    interface['fml_components'] = str(component_path)
    interface['evaluator'] = str(Path(sys.modules['eval_unistainnet_if'].__file__).resolve())
    pc.require(Path(interface['evaluator']).parent == Path(local), 'evaluator must be the current runner sibling')
    return dict(mp=mp, sp=sp, center_items=center_items, evaluate=evaluate, trainer=MatchedUNIStainNetIFTrainer,
                wrapper=HemitUNIStainNet, manifest=manifest, selected=selected, scoring_interface=interface)


def load_models(cfg, rt, seed, device):
    models = {}
    for name in pc.METHOD_ORDER:
        if name not in cfg['seeds'][seed]:
            continue
        path = cfg['seeds'][seed][name]
        if name == 'UNI2IF':
            trainer = rt['trainer'].load_from_checkpoint(path, map_location='cpu', strict=True)
            for key, value in (('uni_dim',1536), ('uni_spatial_size',32), ('uni_spatial_pool_size',32),
                               ('input_skip',True), ('edge_encoder','v2'), ('disable_uni',False)):
                pc.require(getattr(trainer.hparams,key) == value, 'UNI2IF recipe differs: '+key)
            model = rt['wrapper'](trainer, encoder='matched_uni2h').to(device)
        else:
            model = rt['mp'].load_gen(path, device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[name] = model
    return models


def predictions_of(models, he):
    """Match each baseline's native-dtype intensity arithmetic before blending.

    FML renders in float32 and only UNI's forward uses bf16. Crucially, native
    output dtype also computes ((prediction+1)/2).clamp(0,1)*255, just as the
    original scorer does. Cast these unrounded intensities to float32 only then;
    blending/affine precede the one final round. Casting bf16 output first changes
    singleton pixels and would silently alter the source baseline.
    """
    output = {}
    with torch.inference_mode():
        for name, model in models.items():
            enabled = name == 'UNI2IF' and he.device.type == 'cuda'
            with torch.autocast(device_type=he.device.type, dtype=torch.bfloat16, enabled=enabled):
                prediction = model(he.float())
            pc.require(prediction.shape == he.shape and bool(torch.isfinite(prediction).all()),
                       'invalid model prediction shape or values')
            intensity = ((prediction+1)/2).clamp(0,1)*255
            output[name] = intensity.float().permute(0,2,3,1).cpu().numpy()
    return output


QUANTIZATION_CONTRACT = ('Each model retains native output dtype through '
    '((prediction+1)/2).clamp(0,1)*255; then float32 unrounded intensity blend/affine; '
    'final clip and ties-to-even uint8 round; exact float32 uint8-to-normalized roundtrip.')


def u8_to_normalized(pixels, device):
    values = np.asarray(pixels)
    pc.require(values.dtype == np.uint8 and values.ndim == 4 and values.shape[-1] == 3,
               'expected BHWC uint8 output')
    return (torch.from_numpy(np.ascontiguousarray(values)).to(device=device,dtype=torch.float32)
            .permute(0,3,1,2)/255)*2-1


class SelectedPixels(torch.nn.Module):
    def __init__(self, models, family, affine):
        super().__init__()
        self.models = torch.nn.ModuleDict({name: models[name] for name in family.methods})
        self.family, self.affine = family, affine

    def forward(self, he):
        predictions = predictions_of(self.models, he)
        pixels = np.stack([pc.apply({name: x[i] for name,x in predictions.items()},
                                   self.family, self.affine) for i in range(len(he))])
        return u8_to_normalized(pixels, he.device)


def stream_factory(models, rt, root, device):
    def stream():
        for name, he, gt in rt['center_items'](root, 'train', rt['selected']):
            predictions = predictions_of(models, he[None].to(device))
            yield name, {m: x[0] for m,x in predictions.items()}, gt
    return stream


def profile_seed(models, rt, root, device):
    scorer = pc.common_scorer(rt['mp'])
    fs = pc.families(tuple(models))
    # Fixed representative two-model family; no label/score-dependent choice.
    # Include UNI when present, so its forward participates in full-readout timing.
    representative = pc.Family(('FM','UNI2IF') if 'UNI2IF' in models else ('direct','FM'), .5)
    wrapped = SelectedPixels(models,representative,(np.ones(3),np.zeros(3))).eval()
    timings = []
    for name, he, gt in rt['center_items'](root,'train',rt['selected'][:2]):
        start = time.monotonic()
        prediction = predictions_of(models, he[None].to(device))
        prediction = {m:x[0] for m,x in prediction.items()}
        predicted = time.monotonic()
        for family in fs:
            moments = pc.Moments(); moments.add(family.pixels(prediction),gt)
            for affine in (None,moments.affine()):
                scorer(pc.apply(prediction,family,affine),gt)
        end = time.monotonic()
        # The common full-curve path also computes panel guards as validation does.
        # Use these same two TRAIN images; discard all scores/counts after timing.
        with torch.inference_mode():
            rt['sp'].run({'profile-combination':wrapped},[(name,he,gt)],device,
                         per_image=[],panel_rows=[])
        swept = time.monotonic()
        timings.append(dict(name=name,prediction_seconds=predicted-start,
                            all_candidate_fixed_scoring_seconds=end-predicted,
                            representative_prediction_and_common_sweep_seconds=swept-end))
    pc.require(len(timings)==2, 'profile requires exactly two selected training images')
    return dict(status='profile_only', training_names=rt['selected'][:2], timings=timings,
                candidate_count=2*len(fs), acceptance_scores_written=False,
                representative_family=representative,
                quantization_contract=QUANTIZATION_CONTRACT,
                full_evaluation='not_run',
                limitation=('Two training images measure fixed-candidate and representative prediction/full-curve '
                            'timing only; scores are discarded. No OOF selection or validation access.'))


def process_seed(cfg, rt, seed, models, directory, device, profile_only=False):
    directory.mkdir(exist_ok=False)
    if profile_only:
        result = profile_seed(models, rt, Path(cfg['data_root']), device)
        write_json(directory/'profile.json',result)
        return dict(status='profile_only',full_evaluation='not_run')
    result = pc.fit_select(stream_factory(models,rt,Path(cfg['data_root']),device),
                           pc.common_scorer(rt['mp']),rt['manifest']['train'],rt['selected'],
                           tuple(models),forbidden_names=rt['manifest']['val'])
    result['quantization_contract'] = QUANTIZATION_CONTRACT
    # This complete train-only selection record is durable before the first val pixel.
    write_json(directory/'train-selection.json',result)
    write_json(directory/'seed-status.json',dict(status=result['status'],stage='train_selection_saved'))
    if result['selected'] is None:
        return dict(status='no_eligible_candidate',validation='not_run')
    selected = SelectedPixels(models,result['selected']['family'],result['affine']).eval()
    arm = f'pixel-comparator-s{seed}'
    with torch.inference_mode():
        panel,sweep,payload,counts = rt['evaluate'](selected,Path(cfg['data_root']),rt['manifest'],
                arm,str(directory/'train-selection.json'),device,300)
    # Match the existing evaluator's JSON convention for undefined per-image
    # Pearson values; these are not fabricated finite observations.
    write_json(directory/'panel.json',panel,allow_nan=True)
    write_json(directory/'sweep.json',sweep,allow_nan=True)
    with (directory/'per-image-curves.npz').open('xb') as handle:
        np.savez_compressed(handle,**payload)
    pc.require(counts == {'train':300,'val':630}, 'common endpoint population incomplete')
    receipt = dict(status='completed',images=counts,selection='train_only_oof_fixed_f1',
                   checkpoints=cfg['seeds'][seed],scoring_interface=rt['scoring_interface'],
                   precision='FML float32; UNI2IF-only bf16 forward; final float32 uint8 roundtrip',
                   quantization_contract=QUANTIZATION_CONTRACT,
                   endpoint='pooled tolerant ODS/AP/calibrated/fixed F1 and common image guards',
                   paired_uni2if_claim=False)
    write_json(directory/'evaluation.json',receipt)
    write_json(directory/'seed-status.json',dict(status='completed',stage='common_evaluation'),exclusive=False)
    return dict(status='completed',images=counts)


def run(cfg, profile_only=False):
    root = validate_config(cfg)
    root.mkdir(exist_ok=False)
    started, cpu_start = time.monotonic(), time.process_time()
    limit = cfg.get('max_seconds', 1800)
    state = dict(status='in_progress',started=datetime.now(timezone.utc).isoformat(),
                 profile_only=profile_only,seeds={},config=cfg,external_timeout_required_seconds=limit)
    write_json(root/'campaign-status.json',state)
    code, gpu = 1, False
    old_handler = signal.getsignal(signal.SIGALRM)
    def timeout(signum, frame):
        raise TimeoutError(f'{limit}-second finite execution limit reached; incomplete outputs are not results')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(limit)
    try:
        rt = runtime(cfg)
        pc.require(torch.cuda.is_available(), 'registered runner requires the externally assigned CUDA device')
        device = torch.device('cuda:0')
        # is_available() does not initialize the allocator in the deployed Torch.
        torch.cuda.set_device(device)
        gpu = True
        torch.cuda.reset_peak_memory_stats(device)
        for seed in ('0','1'):
            state['active_seed'] = seed
            write_json(root/'campaign-status.json',state,exclusive=False)
            models = load_models(cfg,rt,seed,device)
            outcome = process_seed(cfg,rt,seed,models,root/f'seed-{seed}',device,profile_only)
            state['seeds'][seed] = outcome
            write_json(root/'campaign-status.json',state,exclusive=False)
            del models
            gc.collect()
            torch.cuda.empty_cache()
        statuses = [v['status'] for v in state['seeds'].values()]
        state['status'] = ('profile_only' if profile_only else
                           'completed' if statuses == ['completed','completed'] else 'completed_with_no_eligible_candidate')
        state['full_evaluation'] = 'not_run' if profile_only else ('completed' if state['status']=='completed' else 'partial_or_none')
        code = 0
    except Exception as error:
        traceback.print_exc()
        state.update(status='failed',error=f'{type(error).__name__}: {error}',full_evaluation='incomplete_or_not_run')
        print(state['error'],flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,old_handler)
        state.update(finished=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
                     process_cpu_seconds=time.process_time()-cpu_start,
                     allocated_gpu_hours_upper_bound=(time.monotonic()-started)/3600 if gpu else 0.,
                     gpu_time_note='Conservative allocated wall time, not CUDA active-kernel time.')
        if gpu:
            state.update(peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/1e6,
                         peak_cuda_reserved_mb=torch.cuda.max_memory_reserved()/1e6)
        write_json(root/'campaign-status.json',state,exclusive=False)
        (root/'exit-code.txt').write_text(str(code)+'\n')
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--profile-only',action='store_true')
    args=parser.parse_args()
    return run(json.loads(regular(args.config).read_text()),args.profile_only)


if __name__ == '__main__':
    raise SystemExit(main())
