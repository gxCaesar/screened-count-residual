"""Audited original-FIT inputs for the query-relation development protocol.

Construction reads metadata only. ``read_frame`` opens he/u/fs/tokens, while
``read_target`` is a separate capability. Every physical ZIP-member attempt is
written to an fsync'ed JSONL audit, including denied and failed attempts.
No host path or discovery configuration is inferred by this module.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import weakref
import zipfile

import numpy as np

from .pixel_comparator import quantize


PLAN_ID = 'query-relation-reconstruction-20261027'
SCHEMA = 'query-relation-data-v1'
OUTER_INDICES = tuple(range(0, 32, 4))
TRAINING_INDICES = tuple(i for i in range(32) if i not in OUTER_INDICES)
ARMS = ('P_F', 'M_F', 'S_F', 'E_F', 'P_C', 'S_C', 'E_C', 'E_shift')
SEEDS = (202610270, 202610271)
INPUT_MEMBERS = ('he', 'u', 'fs', 'tokens')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_row(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.is_file()
            and not path.is_symlink() and path.resolve(strict=True) == path,
            'canonical regular source required: '+str(path))
    stat = path.stat()
    require(stat.st_size > 0, 'nonempty source required: '+str(path))
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _metadata(row):
    require(isinstance(row, dict) and set(row) == {'path', 'size_bytes', 'mtime_ns'}
            and isinstance(row['path'], str) and type(row['size_bytes']) is int
            and row['size_bytes'] > 0 and type(row['mtime_ns']) is int
            and row['mtime_ns'] >= 0, 'exact path/size/mtime metadata row required')
    return dict(row)


def stable(rows):
    for row in rows:
        require(file_row(row['path']) == _metadata(row),
                'source metadata changed: '+row['path'])


def _unique(rows):
    found = {}
    for row in rows:
        row = _metadata(row)
        require(found.get(row['path'], row) == row, 'conflicting source metadata')
        found[row['path']] = row
    return [found[key] for key in sorted(found)]


def _json(path):
    require(Path(path).suffix == '.json', 'JSON configuration/receipt path required')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def _index(index):
    require(type(index) is int and 0 <= index < 32, 'original FIT index0..31 required')
    return index


def training_subset(indices):
    """Return one exact inner16 or final24, preserving original FIT order."""
    selected = tuple(indices)
    require(all(type(i) is int for i in selected), 'integer training indices required')
    legal = [TRAINING_INDICES, *(tuple(i for i in TRAINING_INDICES if i % 4 != fold)
                                for fold in (1, 2, 3))]
    require(selected in legal, 'exact original inner16 or final24 training subset required')
    return selected


def _path_map(mapping, label):
    require(isinstance(mapping, dict), label+': index/path mapping required')
    result = {}
    for key, value in mapping.items():
        require(type(key) is int or isinstance(key, str) and key.isdigit(),
                label+': integer index key required')
        index = _index(int(key))
        require(index not in result, label+': duplicate logical index')
        row = _metadata(value) if isinstance(value, dict) else file_row(value)
        require(file_row(row['path']) == row, label+': source metadata changed')
        require(not {part.lower() for part in Path(row['path']).parts}
                & {'hold', 'val', 'validation', 'test'}, 'HOLD/val/test source path forbidden')
        result[index] = row
    require(set(result) == set(range(32)) and len({r['path'] for r in result.values()}) == 32,
            label+': all32 distinct original FIT paths required')
    return result


def validate_freeze_receipt(receipt, source_metadata):
    """Validate the physical freeze and independent generation acceptance.

    Receipt: ``{'manifest': metadata_row, 'generation_acceptance': metadata_row}``.
    Manifest fields are documented by the checks below. Output and weight rows
    contain ``arm``, ``seed``, ``file``; outputs additionally contain
    ``original_index``. Extra scientific/checker fields may be retained.
    """
    require(isinstance(receipt, dict)
            and set(receipt) == {'manifest', 'generation_acceptance'},
            'outer scoring requires frozen manifest and generation acceptance')
    manifest_row, acceptance_row = (_metadata(receipt[k])
                                     for k in ('manifest', 'generation_acceptance'))
    stable([manifest_row, acceptance_row])
    require(not {manifest_row['path'], acceptance_row['path']}
            & {r['path'] for r in source_metadata}, 'freeze receipts cannot be numeric source archives')
    manifest, accepted = _json(manifest_row['path']), _json(acceptance_row['path'])
    require(manifest.get('plan_id') == PLAN_ID
            and manifest.get('status') == 'all128_outputs_and_selected_weights_frozen',
            'complete original query-relation freeze required')
    outputs, weights = manifest.get('outputs'), manifest.get('selected_weights')
    require(isinstance(outputs, list) and len(outputs) == 128
            and isinstance(weights, list) and len(weights) == 16,
            'all128 outputs and16 selected weights required before outer targets')
    expected_outputs = {(a, s, i) for a in ARMS for s in SEEDS for i in OUTER_INDICES}
    expected_weights = {(a, s) for a in ARMS for s in SEEDS}
    require(all(isinstance(r, dict) and isinstance(r.get('arm'), str) and type(r.get('seed')) is int
                and type(r.get('original_index')) is int for r in outputs)
            and {(r.get('arm'), r['seed'], r['original_index']) for r in outputs} == expected_outputs,
            'exact128 arm/seed/outer output identities required')
    require(all(isinstance(r, dict) and isinstance(r.get('arm'), str)
                and type(r.get('seed')) is int for r in weights)
            and {(r.get('arm'), r['seed']) for r in weights} == expected_weights,
            'exact16 arm/seed selected-weight identities required')
    frozen = [_metadata(r['file']) for r in [*outputs, *weights]]
    require(len({r['path'] for r in frozen}) == 144,
            'distinct physical outputs and selected weights required')
    require(not {r['path'] for r in frozen} & {r['path'] for r in source_metadata},
            'generated outputs and weights cannot alias original sources')
    require(manifest.get('source_metadata') == list(source_metadata),
            'freeze must bind the current original source metadata')
    require(accepted.get('status') == 'passed' and accepted.get('phase') == 'generate'
            and accepted.get('generation_manifest') == manifest_row
            and type(accepted.get('generated_outputs')) is int
            and accepted['generated_outputs'] == 128
            and type(accepted.get('checked_selected_weights')) is int
            and accepted['checked_selected_weights'] == 16
            and type(accepted.get('target_reads')) is int and accepted['target_reads'] == 0
            and accepted.get('independent_consumer') is True,
            'complete independent target-free generation acceptance required')
    stable([*source_metadata, *frozen, manifest_row, acceptance_row])
    return dict(manifest=manifest_row, generation_acceptance=acceptance_row)


class MemberAudit:
    """Guard actual ZipFile.open calls, rather than trusting member declarations.

    ``active()`` can also surround a checker known-defect probe. ZIP reads
    outside the current exact source/member/index request are denied. Use one
    thread and DataLoader worker0 while a process-global ZIP guard is active.
    """
    _current = None

    def __init__(self, frames, targets, sources, audit_path, training_indices):
        self.frames, self.targets = frames, targets
        self.sources = list(sources)
        self.training_indices = tuple(training_indices)
        self.owners = {}
        for mapping in (frames, targets):
            for index, row in mapping.items():
                path = row['path']
                require(self.owners.get(path, index) == index, 'source path has conflicting index owners')
                self.owners[path] = index
        self.path = Path(audit_path)
        require(self.path.is_absolute() and '..' not in self.path.parts
                and self.path.parent.resolve(strict=True) == self.path.parent,
                'canonical absolute audit path required')
        require(str(self.path) not in {r['path'] for r in sources}, 'audit cannot overwrite source')
        with self.path.open('x'):
            pass
        self._identity = self.path.stat().st_ino
        self.events, self.opens = [], []
        self._request, self._active, self._closed = None, False, False
        self._outer_opened = set()
        self.record(event='audit_created', status='metadata_only', source_metadata=sources,
                    training_indices=list(training_indices), outer_target_reads=0)

    def record(self, **event):
        require(not self._closed and not self.path.is_symlink()
                and self.path.stat().st_ino == self._identity, 'durable audit file changed or closed')
        event = dict(sequence=len(self.events), **event)
        with self.path.open('a') as stream:
            stream.write(json.dumps(event, separators=(',', ':'), allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)
        return event

    @contextlib.contextmanager
    def active(self):
        require(not self._closed and not self._active and MemberAudit._current is None,
                'nested or closed member audit')
        stable(self.sources)
        original_open, original_init = zipfile.ZipFile.open, zipfile.ZipFile.__init__
        protected = {r['path'] for r in self.sources}
        owner = self
        def guarded_init(archive, file, mode='r', *args, **kwargs):
            path = str(file if isinstance(file, (str, Path)) else getattr(file, 'name', ''))
            canonical = str(Path(path).resolve()) if path else path
            if mode != 'r' and canonical in protected:
                archive.fp = None
                owner.record(event='archive_write', path=path, status='denied')
                raise ValueError('protected input ZIP write forbidden')
            return original_init(archive, file, mode, *args, **kwargs)
        def guarded_open(archive, name, mode='r', pwd=None, *, force_zip64=False):
            return owner._zip_open(original_open, archive, name, mode, pwd, force_zip64)
        self._active, MemberAudit._current = True, self
        zipfile.ZipFile.__init__, zipfile.ZipFile.open = guarded_init, guarded_open
        try:
            yield self
        finally:
            zipfile.ZipFile.__init__, zipfile.ZipFile.open = original_init, original_open
            self._active, MemberAudit._current = False, None
            stable(self.sources)

    @contextlib.contextmanager
    def reading(self, index, member, purpose, *, path=None, freeze_receipt=None):
        require(self._active and self._request is None, 'one active exact member request required')
        index = _index(index)
        mapping = self.targets if member == 'target' else self.frames
        self._request = dict(original_index=index, member=member, purpose=purpose,
                             path=str(path) if path is not None else mapping[index]['path'],
                             freeze_receipt=freeze_receipt)
        try:
            yield self
        finally:
            self._request = None

    def _authorize(self, path, member):
        request = self._request
        require(request is not None and path == request['path'],
                'ZIP read outside exact declared member request')
        require(path in self.owners and self.owners[path] == request['original_index'],
                'ZIP canonical path/index owner mismatch')
        require(member == request['member']+'.npy', 'forbidden physical ZIP member: '+str(member))
        index, field, purpose = (request[k] for k in ('original_index', 'member', 'purpose'))
        if field in INPUT_MEMBERS:
            require(path == self.frames[index]['path'] and purpose in ('inference', 'token_statistics'),
                    'input source path/purpose mismatch')
            require(purpose != 'token_statistics' or field == 'tokens' and index in self.training_indices,
                    'statistics may open declared training tokens only')
        else:
            require(field == 'target' and path == self.targets[index]['path'], 'unknown source member')
            if purpose == 'training':
                require(index in self.training_indices, 'training target outside declared training subset')
            elif purpose == 'validation':
                require(index in set(TRAINING_INDICES)-set(self.training_indices),
                        'validation target outside declared inner-held subset')
            else:
                require(purpose == 'scoring' and index in OUTER_INDICES,
                        'outer target allowed only for frozen scoring')
                require(index not in self._outer_opened, 'outer target already opened in this audit')
                validate_freeze_receipt(request['freeze_receipt'], self.sources)
        stable([r for r in self.sources if r['path'] == path])

    def _zip_open(self, original, archive, name, mode, pwd, force_zip64):
        path = str(getattr(archive, 'filename', ''))
        if mode == 'w':
            require(path not in {r['path'] for r in self.sources}, 'protected input member write forbidden')
            return original(archive, name, mode, pwd, force_zip64=force_zip64)
        member = name.filename if isinstance(name, zipfile.ZipInfo) else name
        request = {k: v for k, v in (self._request or {}).items() if k != 'freeze_receipt'}
        event = dict(event='member_open', attempt=len(self.opens), path=path, zip_member=str(member),
                     request=request, status='attempt')
        self.record(**event)
        try:
            require(mode == 'r' and self._active, 'only active input ZIP reads allowed')
            self._authorize(path, member)
            require(len(archive.namelist()) == len(set(archive.namelist())), 'duplicate physical ZIP member')
        except BaseException as error:
            event.update(status='denied', error=type(error).__name__+': '+str(error))
            self.record(**event)
            self.opens.append(event)
            raise
        try:
            result = original(archive, name, mode, pwd, force_zip64=force_zip64)
        except BaseException as error:
            event.update(status='failed', error=type(error).__name__+': '+str(error))
            self.record(**event)
            self.opens.append(event)
            raise
        event.update(status='opened')
        self.opens.append(event)
        if self._request['member'] == 'target' and self._request['original_index'] in OUTER_INDICES:
            self._outer_opened.add(self._request['original_index'])
        self.record(**event)
        return result

    def summary(self):
        opened = [r for r in self.opens if r['status'] == 'opened']
        targets = [r for r in opened if r['request'].get('member') == 'target']
        return dict(audit_path=str(self.path), physical_member_attempts=len(self.opens),
                    physical_member_opens=len(opened), denied_member_opens=sum(
                        r['status'] == 'denied' for r in self.opens),
                    failed_member_opens=sum(r['status'] == 'failed' for r in self.opens),
                    training_target_reads=sum(r['request']['purpose'] in ('training', 'validation')
                                              for r in targets),
                    validation_target_reads=sum(r['request']['purpose'] == 'validation' for r in targets),
                    outer_target_reads=sum(r['request']['purpose'] == 'scoring' for r in targets),
                    member_accesses=list(self.opens))

    def close(self):
        require(not self._active, 'cannot close active member audit')
        self._closed = True


def _validate_member(member, value):
    specs = dict(he=((3, 512, 512), np.float32), u=((3, 512, 512), np.float32),
                 fs=((2, 3, 512, 512), np.float32), tokens=((1024, 1536), np.float16),
                 target=((512, 512, 3), np.uint8))
    shape, dtype = specs[member]
    require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype
            and np.isfinite(value).all(), 'native shape/dtype/finite mismatch: '+member)
    if member == 'he':
        require(np.all((value >= -1) & (value <= 1)), 'native H&E range mismatch')
    if member in ('u', 'fs'):
        require(np.all((value >= 0) & (value <= 255)), 'native intensity range mismatch: '+member)
    return value


def original_donor(u, fs):
    """Original float32 donor0 quantizer; returns canonical HWC RGB bytes."""
    _validate_member('u', u)
    _validate_member('fs', fs)
    return quantize(np.float32(.75)*u+np.float32(.25)*fs[0]).transpose(1, 2, 0).copy()


def common_inputs(he, u, fs, donor):
    """Common twelve CHW float32 channels, in he/u/f0/donor order."""
    return np.concatenate(((he+np.float32(1))/np.float32(2), u/np.float32(255),
                           fs[0]/np.float32(255),
                           donor.transpose(2, 0, 1).astype(np.float32)/np.float32(255)))


class QueryRelationDataset:
    """Original-FIT member loader with no constructor target or array preload.

    Config is a dict or an explicit JSON path with ``schema``, ``frame_paths``
    (all32 index-to-path/metadata rows), optional ``target_paths`` (same archives
    by default), ``original_fit_names``, ``protected`` metadata and
    ``source_config_path``. The latter binds caller-discovered cluster-node config
    metadata; this loader never searches a host or guesses a cache location.
    """
    def __init__(self, config, audit_path, *, training_indices=TRAINING_INDICES):
        config_row = None
        if isinstance(config, (str, Path)):
            config_row = file_row(config)
            config = _json(config_row['path'])
        require(isinstance(config, dict) and config.get('schema') == SCHEMA,
                'explicit query-relation source configuration required')
        self.training_indices = training_subset(training_indices)
        self.frames = _path_map(config.get('frame_paths'), 'frame_paths')
        self.targets = (_path_map(config['target_paths'], 'target_paths')
                        if 'target_paths' in config else self.frames)
        names = config.get('original_fit_names', [f'{i:02d}' for i in range(32)])
        require(isinstance(names, list) and len(names) == len(set(names)) == 32
                and all(isinstance(n, str) and n not in ('', '.', '..') and Path(n).name == n
                        for n in names), 'original32 unique ordered basenames required')
        self.names = tuple(names)
        sources = [*self.frames.values(), *self.targets.values(), *config.get('protected', [])]
        if config_row is not None:
            sources.append(config_row)
        if config.get('source_config_path') is not None:
            sources.append(file_row(config['source_config_path']))
        self._sources = _unique(sources)
        stable(self._sources)
        self.audit = MemberAudit(self.frames, self.targets, self._sources, audit_path, self.training_indices)
        self._tokens_seen = {}

    def source_metadata(self):
        return [dict(row) for row in self._sources]

    def _read(self, index, member, purpose, freeze_receipt=None):
        index = _index(index)
        active = contextlib.nullcontext() if self.audit._active else self.audit.active()
        mapping = self.targets if member == 'target' else self.frames
        request = dict(original_index=index, member=member, purpose=purpose, path=mapping[index]['path'])
        self.audit.record(event='array_read', request=request, status='requested')
        path = request['path']
        try:
            with active, self.audit.reading(index, member, purpose, freeze_receipt=freeze_receipt):
                with zipfile.ZipFile(path, 'r') as archive:
                    with archive.open(member+'.npy', 'r') as stream:
                        value = np.lib.format.read_array(stream, allow_pickle=False)
                _validate_member(member, value)
                stable([r for r in self._sources if r['path'] == path])
        except BaseException as error:
            self.audit.record(event='array_read', request=request, status='failed',
                              error=type(error).__name__+': '+str(error))
            raise
        self.audit.record(event='array_read', request=request, status='read')
        if member == 'tokens':
            self._tokens_seen[index] = weakref.ref(value)
        return value

    def read_frame(self, index):
        """Read only four inference members, returning raw tokens plus derived inputs."""
        frame = {key: self._read(index, key, 'inference') for key in INPUT_MEMBERS}
        frame['donor'] = original_donor(frame['u'], frame['fs'])
        frame['common12'] = common_inputs(frame['he'], frame['u'], frame['fs'], frame['donor'])
        return frame

    def read_tokens(self, index):
        require(_index(index) in self.training_indices, 'statistics cannot read held or outer tokens')
        return self._read(index, 'tokens', 'token_statistics')

    def read_target(self, index, purpose='training', freeze_receipt=None):
        """Read canonical HWC RGB bytes; outer reads require accepted physical freeze."""
        return self._read(index, 'target', purpose, freeze_receipt)

    def audit_summary(self):
        return self.audit.summary()

    def close(self):
        self.audit.close()


def fit_token_statistics(dataset, indices=None, *, cached_frames=None):
    """All selected training tokens, population std, with no held normalization.

    Cached frames must contain the same raw token arrays previously opened by
    this dataset. Reuse adds no fictitious physical member-read counter.
    """
    selected = training_subset(dataset.training_indices if indices is None else indices)
    require(set(selected) <= set(dataset.training_indices), 'statistics exceed declared training subset')
    mean, m2, count = np.zeros(1536), np.zeros(1536), 0
    for index in selected:
        if cached_frames is None:
            tokens = dataset.read_tokens(index)
        else:
            require(index in cached_frames and isinstance(cached_frames[index], dict),
                    'complete cached training frame population required')
            tokens = cached_frames[index].get('tokens')
            observed = dataset._tokens_seen.get(index)
            require(observed is not None and observed() is tokens,
                    'cached statistics require original audited raw token arrays')
        x = _validate_member('tokens', tokens).astype(np.float64)
        n, local_mean = len(x), x.mean(axis=0)
        delta = local_mean-mean
        m2 += np.square(x-local_mean).sum(axis=0)+delta*delta*(count*n/(count+n))
        mean += delta*(n/(count+n))
        count += n
    std = np.sqrt(m2/count)
    rows = [dict(dataset.frames[i]) for i in selected]
    stable(rows)
    return dict(mean=mean, std=std, zero_variance=std == 0,
                metadata=dict(training_indices=list(selected), token_count=count,
                              variance='population', raw_token_order='row-major32x32',
                              source_metadata=rows, reused_audited_cache=cached_frames is not None))


def standardize_tokens(tokens, statistics):
    _validate_member('tokens', tokens)
    mean, std = np.asarray(statistics['mean']), np.asarray(statistics['std'])
    require(mean.shape == std.shape == (1536,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and np.all(std >= 0), 'finite1536 token statistics required')
    zero = std == 0
    result = (tokens.astype(np.float64)-mean)/np.where(zero, 1, std)
    result[:, zero] = 0
    return result.astype(np.float32)


def token_grid(tokens):
    """Raw row-major token matrix to CHW32x32; no interpolation or projection."""
    require(isinstance(tokens, np.ndarray) and tokens.shape == (1024, 1536)
            and tokens.dtype in (np.dtype('float16'), np.dtype('float32'))
            and np.isfinite(tokens).all(), 'finite raw/standardized token matrix required')
    return tokens.reshape(32, 32, 1536).transpose(2, 0, 1).copy()


def apply_d4(value, code, axes=(-2, -1)):
    require(type(code) is int and 0 <= code < 8, 'D4 code0..7 required')
    result = np.rot90(value, code % 4, axes=axes)
    if code >= 4:
        result = np.flip(result, axis=axes[1])
    return np.ascontiguousarray(result)


def seeded_d4(seed, epoch, index):
    require(type(seed) is int and seed >= 0 and type(epoch) is int and epoch >= 0,
            'nonnegative integer augmentation seed/epoch required')
    return int(np.random.default_rng(np.random.SeedSequence([seed, epoch, _index(index)])).integers(8))


def transform_d4(frame, target=None, code=0, shift_tokens=False):
    """Joint D4, with E_shift rolling the RAW token grid16 bins before D4.

    Returns ``(frame, target)`` with the same shapes as read_frame: tokens stay
    raw row-major [1024,1536], donor/target stay HWC and common12 stays CHW.
    Projection and bilinear interpolation belong to the trainable model.
    """
    require(type(shift_tokens) is bool and set(frame) == {*INPUT_MEMBERS, 'donor', 'common12'},
            'exact inference frame and explicit token shift required')
    result = {key: apply_d4(frame[key], code) for key in ('he', 'u', 'fs', 'common12')}
    result['donor'] = apply_d4(frame['donor'], code, axes=(0, 1))
    raw = _validate_member('tokens', frame['tokens']).reshape(32, 32, 1536)
    if shift_tokens:
        raw = np.roll(raw, (16, 16), axis=(0, 1))
    result['tokens'] = apply_d4(raw, code, axes=(0, 1)).reshape(1024, 1536)
    if target is not None:
        target = apply_d4(_validate_member('target', target), code, axes=(0, 1))
    return result, target
