"""Train/val-only HER2match adapter for the UNIStainNet five-item interface.

The default PIL path preserves historical scaffolding. Production callers must
explicitly select resize_backend='tensor', resize_interpolation='bilinear' and
record preprocessing_version. No score metadata, grade conditioning,
augmentation or directory discovery. This is not an official reproduction.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    from .inventory_her2_trainval import IMAGE
except ImportError:
    from inventory_her2_trainval import IMAGE


def _safe_path(path):
    path = Path(path).absolute()
    if any(part.lower() == 'test' or 'score' in part.lower()
           for part in path.parts):
        raise ValueError('test and score paths are forbidden')
    if '..' in path.parts or path.resolve(strict=True) != path:
        raise ValueError('noncanonical or symlink path is forbidden')
    return path


def _tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0


class PairedManifestDataset(Dataset):
    """Return (HE, IHC, UNI sub-crops, constant null token 4, filename).

    HE/IHC are RGB float32 [3,512,512] in [-1,1]. UNI sub-crops are
    row-major [16,3,224,224], bicubic resized and ImageNet-normalized.
    Root comes only from the isolated inventory manifest. Only its declared
    split files are checked; neither test nor score files are enumerated.
    """

    def __init__(self, manifest, split, *, resize_interpolation, resize_backend='pil'):
        if split not in ('train', 'val'):
            raise ValueError('only train and val are allowed')
        if resize_backend not in ('pil', 'tensor'):
            raise ValueError('resize_backend must be pil or tensor')
        if resize_backend == 'tensor' and resize_interpolation != 'bilinear':
            raise ValueError('tensor backend requires bilinear interpolation')
        kernels = {'bilinear': Image.Resampling.BILINEAR,
                   'bicubic': Image.Resampling.BICUBIC,
                   'lanczos': Image.Resampling.LANCZOS}
        if resize_interpolation not in kernels:
            raise ValueError('explicit bilinear, bicubic or lanczos interpolation required')
        # Reject CSV and unsafe paths before opening even the manifest.
        if Path(manifest).suffix.lower() != '.json':
            raise ValueError('an isolated JSON manifest is required, never a score CSV')
        manifest = _safe_path(manifest)
        with manifest.open() as handle:
            data = json.load(handle)
        if data.get('scope') != 'train_val_only' or set(data.get('splits', {})) != {'train', 'val'}:
            raise ValueError('manifest must contain exactly train and val')
        name_sets, group_sets = [], []
        for key in ('train', 'val'):
            info = data['splits'][key]
            names = info.get('names')
            if (not isinstance(names, list) or not names or
                    any(not isinstance(n, str) or not IMAGE.fullmatch(n) for n in names)):
                raise ValueError(f'{key}: invalid image basenames')
            if len(set(names)) != len(names) or info.get('n_pairs') != len(names):
                raise ValueError(f'{key}: duplicate names or inconsistent pair count')
            groups = {n: IMAGE.fullmatch(n).group(1) for n in names}
            counts = {g: sum(value == g for value in groups.values()) for g in set(groups.values())}
            if (info.get('group_ids') != groups or info.get('wsi_counts') != counts or
                    info.get('n_filename_wsi_groups') != len(counts)):
                raise ValueError(f'{key}: inconsistent filename WSI groups')
            name_sets.append(set(names))
            group_sets.append(set(groups.values()))
        if name_sets[0] & name_sets[1] or group_sets[0] & group_sets[1]:
            raise ValueError('train/val names or source-WSI groups overlap')
        self.root = _safe_path(data['root'])
        self.split = split
        self.names = tuple(data['splits'][split]['names'])
        self.resize_interpolation = resize_interpolation
        self.resize_backend = resize_backend
        self.preprocessing_version = f'her2match-pil-{resize_interpolation}-512-v1'
        self._kernel = kernels[resize_interpolation]
        if resize_backend == 'tensor':
            # Required dependency only for the explicit production backend;
            # import failures must propagate rather than falling back to PIL.
            from torchvision.transforms import v2
            self._tensor_resize = v2.Compose([
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.CenterCrop(1024),
                v2.Resize(512, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            ])
            self.preprocessing_version = 'her2match-tensor-bilinear-aa-512-uni-round-pil-v1'
        # Preflight the entire selected membership before any pixel read.
        for name in self.names:
            self._pair_paths(name)

    def _pair_paths(self, name):
        paths = tuple(_safe_path(self.root / modality / self.split / name)
                      for modality in ('HE', 'IHC'))
        if any(not path.is_file() for path in paths):
            raise ValueError('paired image must be a regular file')
        return paths

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        # Recheck both paths before either read in case a symlink changed.
        paths = self._pair_paths(name)
        images = []
        for path in paths:
            with Image.open(path) as image:
                if image.size != (1024, 1024) or image.mode != 'RGB':
                    raise ValueError('HER2match input must be 1024x1024 RGB')
                if self.resize_backend == 'tensor':
                    images.append(self._tensor_resize(image))
                else:
                    images.append(image.resize((512, 512), resample=self._kernel))
        if self.resize_backend == 'tensor':
            he_float, ihc_float = images
            # Quantize only the FM view; renderer and target keep float pixels.
            uni_u8 = (he_float.clamp(0, 1) * 255).round().to(torch.uint8)
            he = Image.fromarray(uni_u8.permute(1, 2, 0).numpy())
            he_tensor = (he_float - 0.5) / 0.5
            ihc_tensor = (ihc_float - 0.5) / 0.5
        else:
            he, ihc = images
            he_tensor, ihc_tensor = _tensor(he) * 2 - 1, _tensor(ihc) * 2 - 1
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        crops = []
        for row in range(4):
            for col in range(4):
                crop = he.crop((col * 128, row * 128, (col + 1) * 128, (row + 1) * 128))
                crop = crop.resize((224, 224), resample=Image.Resampling.BICUBIC)
                crops.append((_tensor(crop) - mean) / std)
        return he_tensor, ihc_tensor, torch.stack(crops), 4, name
