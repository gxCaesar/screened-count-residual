"""Evaluate an adapted UNIStainNet checkpoint with the existing HEMIT scorer.

Uses centered train calibration and all manifest validation images. The pinned
UNIStainNet source and the deployed DTR metrics.py must be supplied explicitly.
Weights must already be available; this command does not authorize a run or a
download. Source LPIPS selection occurs before this evaluator is invoked.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from curve_evidence import per_image_payload, selected_train, validate_manifest, validate_per_image
from paired_manifest_dataset import _safe_path


def uni_crops(he):
    """Restore native HEMIT uint8 H&E and the loader's exact 4x4 UNI view."""
    if (he.ndim != 4 or tuple(he.shape[1:]) != (3, 512, 512)
            or not he.is_floating_point() or not torch.isfinite(he).all()
            or he.numel() == 0 or he.min() < -1 or he.max() > 1):
        raise ValueError('expected finite H&E [B,3,512,512] in [-1,1]')
    images = (((he.detach().float().cpu() + 1) / 2) * 255).round().byte()
    result = []
    for tensor in images:
        image = Image.fromarray(tensor.permute(1, 2, 0).numpy())
        crops = []
        for row in range(4):
            for col in range(4):
                crop = TF.crop(image, row * 128, col * 128, 128, 128)
                crop = TF.resize(crop, [224, 224], interpolation=TF.InterpolationMode.BICUBIC)
                crops.append(TF.normalize(TF.to_tensor(crop),
                                          [.485, .456, .406], [.229, .224, .225]))
        result.append(torch.stack(crops))
    return torch.stack(result).to(he.device)


class HemitUNIStainNet(torch.nn.Module):
    """Adapt the upstream frozen UNI/EMA path to sweep_panel's image callable."""

    def __init__(self, trainer, encoder='uni_vitl16'):
        super().__init__()
        if encoder not in ('uni_vitl16', 'matched_uni2h'):
            raise ValueError('unknown UNIStainNet encoder recipe')
        self.trainer = trainer
        self.encoder = encoder

    @torch.no_grad()
    def forward(self, he):
        if self.encoder == 'matched_uni2h':
            tokens = self.trainer.features_from_he(he)
            width = 1536
        else:
            crops = uni_crops(he)
            tokens = self.trainer._extract_uni_from_sub_crops(crops)
            width = 1024
        if (tuple(tokens.shape) != (len(he), 1024, width)
                or not torch.isfinite(tokens).all()):
            raise ValueError(f'{self.encoder} must return finite [B,1024,{width}] tokens')
        labels = torch.full((len(he),), 4, dtype=torch.long, device=he.device)
        generated = self.trainer.generate(he, tokens, labels, guidance_scale=1.0)
        if generated.shape != he.shape or not torch.isfinite(generated).all():
            raise ValueError('generator must return finite [B,3,512,512] images')
        return generated


def center_items(root, split, names):
    if split not in ('train', 'val'):
        raise ValueError('only train/val center crops are allowed')
    root = _safe_path(root)
    for name in names:
        if Path(name).name != name or not name or name in ('.', '..'):
            raise ValueError('image name must be a plain basename')
        paths = [_safe_path(root / split / part / name) for part in ('input', 'label')]
        images = []
        for path in paths:
            with Image.open(path) as image:
                if image.size != (1024, 1024):
                    raise ValueError('HEMIT evaluation requires native 1024 images')
                images.append(image.convert('RGB').crop((256, 256, 768, 768)))
        yield name, (TF.to_tensor(images[0]) - .5) * 2, np.array(images[1])


def evaluate(model, root, manifest, arm, checkpoint, device, n_train=300, *, item_provider=center_items):
    """One prediction pass per image; common metrics, curves and image counts."""
    import metric_panel as MP
    import sweep_panel as SP
    validate_manifest(root, manifest)
    if not arm or '__' in arm:
        raise ValueError('arm must be nonempty and cannot contain __')
    models = {arm: model.eval()}
    per_train, per_val, panel_rows = [], [], []
    tr, _ = SP.run(models, item_provider(root, 'train', selected_train(manifest['train'], n_train)),
                   device, per_image=per_train)
    va, fixed = SP.run(models, item_provider(root, 'val', manifest['val']), device,
                       per_image=per_val, panel_rows=panel_rows)
    curves = {split: {name: {marker: {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                    for key, value in counts.items()}
                            for marker, counts in markers.items()}
                      for name, markers in data.items()}
              for split, data in (('train', tr), ('val', va))}
    sweep = dict(thresholds=SP.THR, tol_px=SP.TOL, n_train=n_train,
                 ckpts=[f'{arm}={checkpoint}'], summary=SP.summarise_curves(tr, va),
                 curves=curves, fixed_rows=fixed)
    panel = dict(thresholds=MP.FIXED_THR, tol_px=MP.TOL,
                 summary=MP.summarise(panel_rows, [arm]), rows=panel_rows)
    payload = per_image_payload({'train': per_train, 'val': per_val},
                                [arm], SP.TRUE, manifest, SP.THR, SP.TOL)
    counts = validate_evidence(panel, sweep, payload, manifest)
    if counts != {'train': n_train, 'val': len(manifest['val'])}:
        raise ValueError('evaluated image counts differ from requested membership')
    return panel, sweep, payload, counts


def validate_evidence(panel, sweep, payload, manifest):
    """Check generic baseline arm names against image and curve evidence."""
    import sweep_panel as SP
    from compare_panel import validate_panel
    names = validate_panel(panel)
    if (panel['thresholds'] != sweep['thresholds'] or panel['tol_px'] != sweep['tol_px']
            or set(panel['summary']) != set(sweep['summary'])
            or set(names) != set(manifest['val'])):
        raise ValueError('panel/sweep protocol, arm or image membership differs')
    counts = validate_per_image(payload, sweep, manifest)
    recomputed = SP.summarise_curves(sweep['curves']['train'], sweep['curves']['val'])
    if set(recomputed) != set(sweep['summary']):
        raise ValueError('summary arm membership differs from curves')
    for arm, summary in recomputed.items():
        stored = sweep['summary'][arm]
        if set(stored) != set(summary):
            raise ValueError('summary fields differ from common scorer')
        for key, value in summary.items():
            pairs = value.items() if isinstance(value, dict) else [(key, value)]
            actual = stored[key] if isinstance(value, dict) else stored
            if isinstance(value, dict) and set(actual) != set(value):
                raise ValueError('marker summary fields differ from common scorer')
            for field, expected in pairs:
                if (not isinstance(actual[field], (int, float)) or not math.isfinite(actual[field])
                        or abs(actual[field] - expected) > 1e-9):
                    raise ValueError('summary differs from pooled threshold curves')
    by_name = {row['name']: row for row in panel['rows']}
    for row in sweep['fixed_rows']:
        image_row = by_name[row['name']]
        for arm in panel['summary']:
            for marker in SP.TRUE:
                expected = [image_row[f'{arm}__{marker}_{key}'] for key in ('np', 'tpp', 'ng', 'tpg')]
                if row[f'{arm}__{marker}'] != expected:
                    raise ValueError('panel fixed counts differ from threshold sweep')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--metrics-root', required=True, type=Path)
    parser.add_argument('--torch-home', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--encoder', choices=('uni_vitl16', 'matched_uni2h'),
                        default='uni_vitl16')
    parser.add_argument('--selection', required=True, choices=('source_min_val_lpips_ema', 'final_ema'))
    parser.add_argument('--device', required=True, choices=('cpu', 'cuda'))
    parser.add_argument('--precision', required=True, choices=('32-true', 'bf16-mixed'))
    parser.add_argument('--n-train', type=int, default=300)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output directory already exists; refusing to overwrite evidence')
    if args.device == 'cpu' and args.precision != '32-true':
        parser.error('CPU evaluation requires 32-true precision')
    if args.manifest.suffix.lower() != '.json':
        parser.error('isolated JSON manifest required')
    manifest = json.loads(_safe_path(args.manifest).read_text())
    validate_manifest(_safe_path(args.data_root), manifest)
    selected_train(manifest['train'], args.n_train)
    checkpoint = _safe_path(args.checkpoint)
    if not checkpoint.is_file() or checkpoint.suffix != '.ckpt':
        parser.error('regular Lightning .ckpt file required')
    cache = args.torch_home / 'hub/checkpoints/alexnet-owt-7be5be79.pth'
    if not cache.is_file() or cache.is_symlink():
        parser.error('existing regular AlexNet cache required; downloads are not allowed')
    os.environ.update(TORCH_HOME=str(args.torch_home), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')

    def no_download(*unused_args, **unused_kwargs):
        raise RuntimeError('weights must already be available for offline evaluation')

    torch.hub.download_url_to_file = no_download
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(2)
    sys.path.insert(0, str(args.source_root.resolve(strict=True)))
    sys.path.insert(0, str(args.metrics_root.resolve(strict=True)))
    from scoring_interface import validate_scoring_interface
    scoring_interface = validate_scoring_interface(args.metrics_root)
    if args.encoder == 'matched_uni2h':
        from unistainnet_if_uni2h import MatchedUNIStainNetIFTrainer as TrainerClass, RECIPE_VERSION
        width = 1536
    else:
        from unistainnet_if import UNIStainNetIFTrainer as TrainerClass, RECIPE_VERSION
        width = 1024
    trainer = TrainerClass.load_from_checkpoint(str(checkpoint), map_location='cpu', strict=True)
    for name, expected in (('uni_dim', width), ('uni_spatial_size', 32),
                           ('uni_spatial_pool_size', 32), ('input_skip', True),
                           ('edge_encoder', 'v2'), ('disable_uni', False)):
        if getattr(trainer.hparams, name) != expected:
            raise ValueError(f'checkpoint {name} differs from the declared source-near recipe')
    model = HemitUNIStainNet(trainer, encoder=args.encoder).to(args.device).eval()
    with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=torch.bfloat16,
                                                enabled=args.precision == 'bf16-mixed'):
        panel, sweep, payload, counts = evaluate(model, args.data_root, manifest, args.arm,
                                               str(checkpoint), args.device, args.n_train)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, data in (('panel.json', panel), ('sweep.json', sweep)):
        with (args.out / name).open('x') as handle:
            json.dump(data, handle)
    with (args.out / 'per-image-curves.npz').open('xb') as handle:
        np.savez_compressed(handle, **payload)
    receipt = dict(status='completed', completed_at=datetime.now(timezone.utc).isoformat(),
                   recipe_version=RECIPE_VERSION, encoder=args.encoder,
                   selection=args.selection, arm=args.arm,
                   checkpoint=str(checkpoint), manifest=str(args.manifest),
                   source_root=str(args.source_root), metrics_root=str(args.metrics_root),
                   device=args.device, precision=args.precision, images=counts,
                   scoring_interface=scoring_interface,
                   endpoint='existing pooled tolerant ODS/AP/calibrated/fixed F1 and DTR image guards',
                   limitation='No patient-level uncertainty without verified independent-unit mapping')
    (args.out / 'evaluation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
