"""Run the unchanged pilot with a single-writer bound on actual retained files.

The allowance must come from a fresh complete project budget. It is not a new
scientific quota. Reaching it preserves completed files and stops dependent work.
"""
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from . import run_query_relation_reconstruction as runner


class RetainedBudget:
    def __init__(self, root, allowance):
        self.root = Path(root).resolve()
        if type(allowance) is not int or allowance <= 0:
            raise ValueError('fresh positive retained_budget_bytes required')
        self.allowance = allowance

    def check(self, path, size, append=False):
        path = Path(path).resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError('writer destination outside accounted root')
        files = [p for p in self.root.rglob('*') if p.is_file()] if self.root.exists() else []
        # Include actual allocated blocks and directory overhead, conservatively.
        current = sum(max(p.stat().st_size, p.stat().st_blocks*512) for p in files)
        current += 4096*(1+sum(p.is_dir() for p in self.root.rglob('*'))) if self.root.exists() else 4096
        if path.exists() and not append:
            current -= max(path.stat().st_size, path.stat().st_blocks*512)
        addition = math.ceil(size/4096)*4096 + 32768
        if current+addition > self.allowance:
            raise RuntimeError('waiting_capacity: actual retained writer allowance exhausted; preserve files, no automatic retry')


def install(root, allowance):
    guard = RetainedBudget(root, allowance)
    original_json, original_replace, original_append = runner.write_json, runner.replace_json, runner.append_json
    original_sources = runner.source_metadata

    def write_json(path, value):
        guard.check(path, len(json.dumps(runner.plain(value), allow_nan=False, sort_keys=True, indent=2).encode())+1)
        return original_json(path, value)

    def replace_json(path, value):
        guard.check(path, len(json.dumps(runner.plain(value), allow_nan=False, sort_keys=True, indent=2).encode())+1)
        return original_replace(path, value)

    def append_json(path, value):
        guard.check(path, len(json.dumps(runner.plain(value), allow_nan=False).encode())+1, append=True)
        return original_append(path, value)

    def save_npz(path, **arrays):
        buffer = io.BytesIO();np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue();guard.check(path, len(payload))
        path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as output:output.write(payload)
        return runner.file_metadata(path)

    def checkpoint(path, model, **values):
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        payload = dict(schema='query-relation-weights-v1', plan_id=runner.PLAN_ID,
                       arm=values['arm'], seed=values['seed'], split=values['split'],
                       epoch=values['epoch'], training_indices=list(values['train_indices']),
                       loss_scales=values['scales'], state_dict=state,
                       synthetic_parameter_fixture=values.get('synthetic_parameter_fixture', False),
                       optimizer_state_retained=False)
        stream=io.BytesIO();torch.save(payload, stream);blob=stream.getvalue()
        guard.check(path, len(blob));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as output:output.write(blob)
        return runner.file_metadata(path)

    runner.write_json, runner.replace_json, runner.append_json = write_json, replace_json, append_json
    runner.save_npz, runner.checkpoint = save_npz, checkpoint
    runner.source_metadata = lambda: [*original_sources(), runner.file_metadata(Path(__file__))]
    return guard


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    config = Path(argv[argv.index('--config')+1])
    root = Path(argv[argv.index('--output')+1])
    cfg = json.loads(config.read_text())
    install(root, cfg['retained_budget_bytes'])
    return runner.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
