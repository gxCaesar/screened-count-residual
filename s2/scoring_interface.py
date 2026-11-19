"""Verify the deployed common-scoring interface before constructing a model."""
import importlib
import inspect
from pathlib import Path
import sys


def validate_scoring_interface(metrics_root):
    root = Path(metrics_root).resolve(strict=True)
    names = ('metrics', 's2_scoring', 'metric_panel', 'sweep_panel')
    for name in names:
        path = root / (name + '.py')
        if not path.is_file() or path.is_symlink():
            raise ValueError('missing regular scorer module: ' + str(path))
    sys.path.insert(0, str(root))
    modules = {}
    for name in names:
        module = importlib.import_module(name)
        actual = Path(module.__file__).resolve()
        if actual != root / (name + '.py'):
            raise ValueError('scorer module outside metrics_root: ' + str(actual))
        modules[name] = str(actual)
    signature = inspect.signature(sys.modules['sweep_panel'].run)
    try:
        signature.bind({}, [], 'cpu', per_image=[], panel_rows=[])
    except TypeError as error:
        raise ValueError('incompatible sweep_panel.run: panel_rows/per_image required') from error
    return {'modules': modules, 'sweep_run_signature': str(signature)}
