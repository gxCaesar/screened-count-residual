"""Separate FIT1192 inputs for the scaled query-relation experiment.

The original FIT32 reader and its access checks are deliberately untouched.
This store owns shared, unrounded float32 H&E/U/F0 arrays and panCK bytes only.
It does not own target-read accounting or permission to open held targets.
Native inference dependencies are imported only when ``NativeSources`` is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np

from .pixel_comparator import quantize
from .query_relation_data import apply_d4


FIT_COUNT = 1200
TRAINING_COUNT = 1192
FIT32_INDICES = tuple(k * (FIT_COUNT - 1) // 31 for k in range(32))
EXCLUDED_INDICES = tuple(FIT32_INDICES[k] for k in range(0, 32, 4))
TRAINING_INDICES = tuple(i for i in range(FIT_COUNT) if i not in EXCLUDED_INDICES)
BASE_ARRAY_BYTES = TRAINING_COUNT * 512 * 512 * (3 * 3 * 4 + 1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _names(value, count, label):
    require(isinstance(value, list) and len(value) == count,
            f'{label}: exact{count} names required')
    require(all(isinstance(name, str) and name not in ('', '.', '..')
                and Path(name).name == name and '\\' not in name for name in value),
            label + ': plain basenames required')
    require(len(set(value)) == count, label + ': duplicate names forbidden')
    return value


def build_membership(fit_metadata, pilot_manifest, data_manifest):
    """Validate original manifests and return FIT1192 in original1200 order.

    FIT/hold retain the native stride2, first1500 train selection. Exclusions
    are FIT32 *positions* 0,4,...28, mapped to their original1200 indices.
    """
    require(all(isinstance(value, dict) for value in
                (fit_metadata, pilot_manifest, data_manifest)), 'metadata objects required')
    train = _names(data_manifest.get('train'), 3717, 'original train')
    val = _names(data_manifest.get('val'), 630, 'original val')
    fit = _names(fit_metadata.get('fit_names'), FIT_COUNT, 'original FIT')
    hold = _names(fit_metadata.get('hold_names'), 300, 'original HOLD')
    require(train == sorted(train) and val == sorted(val),
            'original train/val ordered membership required')
    require(not set(train) & set(val), 'train/val membership must be disjoint')
    selected = train[::2][:1500]
    require(fit == selected[:FIT_COUNT] and hold == selected[FIT_COUNT:],
            'FIT1200/HOLD300 must match native ordered stride2 train selection')
    populations = pilot_manifest.get('populations')
    require(isinstance(populations, dict) and isinstance(populations.get('fit'), dict),
            'original FIT32 population required')
    pilot = populations['fit']
    fit32 = _names(pilot.get('names'), 32, 'original FIT32')
    indices = pilot.get('indices')
    require(pilot.get('source_size') == FIT_COUNT and isinstance(indices, list)
            and all(type(i) is int for i in indices) and indices == list(FIT32_INDICES),
            'exact original FIT32 source indices required')
    require(fit32 == [fit[i] for i in FIT32_INDICES],
            'FIT32 must preserve its original FIT1200 names and order')
    excluded = [fit32[i] for i in range(0, 32, 4)]
    training = [dict(index=i, name=fit[i]) for i in TRAINING_INDICES]
    require(len(training) == TRAINING_COUNT and len(set(excluded)) == 8,
            'exact FIT1192 and excluded8 required')
    require(not {row['name'] for row in training} & (set(excluded) | set(hold) | set(val)),
            'training cannot contain old outer8, HOLD or val')
    return dict(training=training, excluded_outer8=excluded, hold=list(hold), val=list(val))


def _index(index):
    require(type(index) is int and 0 <= index < FIT_COUNT,
            'original FIT1200 index0..1199 required')
    return index


def _seed_epoch(seed, epoch):
    require(type(seed) is int and seed >= 0 and type(epoch) is int and epoch >= 0,
            'nonnegative integer seed/epoch required')


def seeded_d4(seed, epoch, index):
    _seed_epoch(seed, epoch)
    return int(np.random.default_rng(
        np.random.SeedSequence([seed, epoch, _index(index)])).integers(8))


def paired_order(indices, seed, epoch):
    """Permutation of original1200 identities, without an arm-dependent seed."""
    _seed_epoch(seed, epoch)
    indices = list(indices)
    require(bool(indices) and len(set(indices)) == len(indices),
            'nonempty unique original1200 indices required')
    for index in indices:
        _index(index)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, epoch]))
    return [int(i) for i in rng.permutation(indices)]


def _source_arrays(he, u, f0, *, synthetic=False):
    require(type(synthetic) is bool, 'explicit synthetic fixture flag required')
    shape = getattr(he, 'shape', None)
    require(isinstance(shape, tuple) and len(shape) == 3 and shape[0] == 3
            and shape[1] == shape[2] and shape[1] > 0
            and (synthetic or shape == (3, 512, 512)),
            'native CHW512 source shape required; small fixtures must be explicit')
    for label, value, lower, upper in (('he', he, -1, 1), ('u', u, 0, 255), ('f0', f0, 0, 255)):
        require(isinstance(value, np.ndarray) and value.shape == shape
                and value.dtype == np.float32 and np.isfinite(value).all()
                and np.all((value >= lower) & (value <= upper)),
                'native float32 shape/range/finite mismatch: ' + label)


def frame_from_arrays(he, u, f0, *, synthetic=False):
    """Retain source arrays; derive donor once and common HE/U/F0/donor channels.

    U/F0 must be unrounded float32 native-dtype-converted intensities. No F1 or
    direct token arrays are created or read by this adapter.
    """
    _source_arrays(he, u, f0, synthetic=synthetic)
    donor = quantize(np.float32(.75) * u + np.float32(.25) * f0).transpose(1, 2, 0).copy()
    common = np.concatenate(((he + np.float32(1)) / np.float32(2), u / np.float32(255),
                             f0 / np.float32(255),
                             donor.transpose(2, 0, 1).astype(np.float32) / np.float32(255)))
    return dict(he=he, u=u, f0=f0, donor=donor, common12=common)


class ArrayDataset:
    """One immutable array store shared by every arm, with no derived caches.

    Constructor takes ownership of supplied array objects and freezes them in
    place, without copying. ``index`` in ``tensors`` is a dataset row, whereas
    ``source_index(row)`` returns its original FIT1200 identity for D4 seeding.
    The caller fills arrays and accounts for each target access before this
    constructor. Synthetic fixtures have explicit, non-production coverage.
    """
    def __init__(self, he, u, f0, target_panck, *, source_indices=None, synthetic=False):
        require(type(synthetic) is bool, 'explicit synthetic fixture flag required')
        require(isinstance(he, np.ndarray) and he.ndim == 4 and len(he) > 0,
                'nonempty NCHW source arrays required')
        n = len(he)
        require(synthetic or n == TRAINING_COUNT, 'production store requires FIT1192')
        require(all(isinstance(value, np.ndarray) and value.shape == he.shape
                    and value.dtype == np.float32 and value.flags.c_contiguous
                    for value in (he, u, f0)), 'shared contiguous NCHW float32 arrays required')
        for row in range(n):
            _source_arrays(he[row], u[row], f0[row], synthetic=synthetic)
        require(isinstance(target_panck, np.ndarray) and target_panck.dtype == np.uint8
                and target_panck.shape == (n, *he.shape[-2:]) and target_panck.flags.c_contiguous,
                'shared contiguous NHW uint8 panCK target array required')
        if source_indices is None:
            require(synthetic, 'production original1200 source indices required')
            source_indices = range(n)
        self.source_indices = tuple(source_indices)
        require(len(self.source_indices) == n and len(set(self.source_indices)) == n,
                'one distinct source1200 index required per dataset row')
        for index in self.source_indices:
            _index(index)
        require(synthetic or self.source_indices == TRAINING_INDICES,
                'production source rows must be exact ordered FIT1192 without old outer8')
        self.he, self.u, self.f0, self.target_panck = he, u, f0, target_panck
        self.synthetic = synthetic
        for value in (he, u, f0, target_panck):
            value.flags.writeable = False
        self._zero_tokens = {}

    def __len__(self):
        return len(self.source_indices)

    @property
    def array_bytes(self):
        return sum(value.nbytes for value in (self.he, self.u, self.f0, self.target_panck))

    def source_index(self, row):
        require(type(row) is int and 0 <= row < len(self), 'dataset row out of range')
        return self.source_indices[row]

    def tensors(self, index, device, code=0):
        """Paired D4 of one dataset row, with cached shared zero direct tokens."""
        import torch
        self.source_index(index)
        frame = frame_from_arrays(self.he[index], self.u[index], self.f0[index],
                                  synthetic=self.synthetic)
        common = apply_d4(frame['common12'], code)
        donor_rgb = apply_d4(frame['donor'], code, axes=(0, 1))
        target = apply_d4(self.target_panck[index], code)
        device = torch.device(device)
        key = str(device)
        if key not in self._zero_tokens:
            self._zero_tokens[key] = torch.zeros((1, 1536, 32, 32), device=device, dtype=torch.float32)
        # Copies are limited to the current frame; no common12/donor dataset cache.
        common_tensor = torch.from_numpy(common).unsqueeze(0).to(device)
        donor_tensor = torch.from_numpy(donor_rgb[..., 0].copy()).to(device)
        target_tensor = torch.from_numpy(target.copy()).to(device).unsqueeze(0).unsqueeze(0)
        return dict(common=common_tensor, tokens=self._zero_tokens[key],
                    donor=donor_tensor.to(torch.float32)[None, None] / 255,
                    target=target_tensor.to(torch.float32) / 255,
                    target_u8=target_tensor, donor_rgb=donor_rgb)


class _NativeCacheOwner:
    """Identity-track entries created during synchronous native source calls."""
    def __init__(self):
        self.caches = {}

    @staticmethod
    def discover():
        # A cached matched-UNI module can still bind an older components module
        # after runtime replaces sys.modules['s2_components']. Follow that exact
        # binding too, without importing or modifying either source module.
        names = {'s2_components', 'unistainnet_if_uni2h'}
        modules = [module for name, module in list(sys.modules.items())
                   if name.rsplit('.', 1)[-1] in names and module is not None]
        modules.extend(vars(module)['components'] for module in list(modules)
                       if vars(module).get('components') is not None)
        caches = {}
        for module in modules:
            namespace = vars(module)
            for attribute in ('_FM_CACHE', '_TOKEN_VIEWS'):
                cache = namespace.get(attribute)
                if isinstance(cache, dict):
                    caches[id(cache)] = dict(module=namespace.get('__name__', '<native>'),
                                             module_path=namespace.get('__file__'),
                                             attribute=attribute, cache=cache)
        return caches

    @contextmanager
    def observe(self):
        before = self.discover()
        snapshots = {identity: dict(info['cache']) for identity, info in before.items()}
        for identity, info in before.items():
            self.caches.setdefault(identity, dict(info, owned={}))
        try:
            yield
        finally:
            for identity, info in self.discover().items():
                tracked = self.caches.setdefault(identity, dict(info, owned={}))
                baseline = snapshots.get(identity, {})
                for key, value in info['cache'].items():
                    if key not in baseline:
                        tracked['owned'][key] = value

    def release(self):
        rows = []
        missing = object()
        for info in self.caches.values():
            cache, owned = info['cache'], info['owned']
            removed, absent, replacements = [], [], []
            for key, value in owned.items():
                current = cache.get(key, missing)
                if current is value:
                    del cache[key]
                    removed.append(repr(key))
                elif current is missing:
                    absent.append(repr(key))
                else:
                    replacements.append(repr(key))
            rows.append(dict(module=info['module'], module_path=info['module_path'],
                             attribute=info['attribute'], removed_keys=removed,
                             already_absent_keys=absent, replacement_keys_preserved=replacements,
                             foreign_entries_preserved=len(cache),
                             owned_entries_remaining=sum(cache.get(key, missing) is value
                                                         for key, value in owned.items())))
        self.caches.clear()  # Drop tracker refs to both removed values and cache dictionaries.
        return dict(owned_cache_entries_removed=sum(len(row['removed_keys']) for row in rows),
                    owned_cache_entries_already_absent=sum(len(row['already_absent_keys']) for row in rows),
                    foreign_replacements_preserved=sum(len(row['replacement_keys_preserved']) for row in rows),
                    foreign_cache_entries_preserved=sum(row['foreign_entries_preserved'] for row in rows),
                    owned_cache_entries_remaining=sum(row['owned_entries_remaining'] for row in rows),
                    caches=rows)


class NativeSources:
    """Lazy native UNI2IF/F0 inference and unopened target loaders.

    ``items`` accepts only training/HOLD/val names bound by ``build_membership``.
    Held target authorization and access accounting remain with its caller.
    No teacher, F1 or raw token reader is instantiated.
    """
    def __init__(self, cfg, device, membership, *, runtime_factory=None,
                 model_loader=None, prediction_fn=None):
        import torch
        if runtime_factory is None or model_loader is None or prediction_fn is None:
            from . import run_pixel_comparator as runner
            runtime_factory = runtime_factory or runner.runtime
            model_loader = model_loader or runner.load_models
            prediction_fn = prediction_fn or runner.predictions_of
        require(isinstance(membership, dict) and set(membership) ==
                {'training', 'excluded_outer8', 'hold', 'val'}, 'validated FIT1192 membership required')
        self.allowed = dict(training=[row['name'] for row in membership['training']],
                            hold=list(membership['hold']), val=list(membership['val']))
        require([row['index'] for row in membership['training']] == list(TRAINING_INDICES),
                'exact original FIT1192 training identities required')
        for split, count in (('training', TRAINING_COUNT), ('hold', 300), ('val', 630)):
            _names(self.allowed[split], count, split)
        excluded = _names(membership['excluded_outer8'], 8, 'excluded outer8')
        require(not set(excluded) & set().union(*(set(names) for names in self.allowed.values()))
                and sum(len(names) for names in self.allowed.values()) ==
                len(set().union(*(set(names) for names in self.allowed.values()))),
                'training/HOLD/val/excluded memberships must be disjoint')
        self.cfg, self.device = dict(cfg), torch.device(device)
        self.models = {}
        self._cache_owner = _NativeCacheOwner()
        self._predict = prediction_fn
        self._closed = False
        self.release_receipt = None
        try:
            with self._cache_owner.observe():
                self.rt = runtime_factory(cfg)
                model_cfg = dict(cfg, seeds={'0': {'UNI2IF': cfg['uni_checkpoint']}})
                self.models = model_loader(model_cfg, self.rt, '0', self.device)
                require(set(self.models) == {'UNI2IF'}, 'load only native UNI2IF before F0')
                self.models['F0'] = self.rt['mp'].load_gen(cfg['f0_checkpoint'], self.device)
                for model in self.models.values():
                    model.eval()
                    for parameter in model.parameters():
                        parameter.requires_grad_(False)
        except BaseException:
            self.close()
            raise

    def items(self, split, names=None):
        from .target_lazy_center_items import target_lazy_center_items
        require(not self._closed and split in self.allowed, 'open training/hold/val source required')
        selected = self.allowed[split] if names is None else list(names)
        require(bool(selected) and len(set(selected)) == len(selected)
                and set(selected) <= set(self.allowed[split]),
                'requested names exceed declared split; old outer8 forbidden')
        data_split = 'val' if split == 'val' else 'train'
        for name, he, target_loader in target_lazy_center_items(
                self.rt['center_items'], Path(self.cfg['data_root']), data_split, selected):
            with self._cache_owner.observe():
                predictions = self._predict(self.models, he[None].to(self.device))
            require(set(predictions) == {'UNI2IF', 'F0'}, 'exact U/F0 predictions required')
            he_array = he.detach().cpu().numpy()
            u, f0 = (predictions[key][0].transpose(2, 0, 1) for key in ('UNI2IF', 'F0'))
            yield name, frame_from_arrays(he_array, u, f0), target_loader

    def close(self):
        """Release own models/cache identities; return an idempotent scalar receipt.

        Existing cache entries and later replacements are preserved. The receipt
        describes removed references, not an assertion about physical GPU usage;
        the native profile owns the post-release allocator/process measurement.
        """
        if self._closed:
            return self.release_receipt
        released = sorted(self.models)
        self.models.clear()
        self._predict = None
        cache_receipt = self._cache_owner.release()
        self._closed = True
        self.release_receipt = dict(status='closed', model_references_released=released,
                                    **cache_receipt)
        return self.release_receipt
