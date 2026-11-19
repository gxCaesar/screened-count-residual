"""Keep the native center-crop primitives while deferring label pixel reads."""
from pathlib import Path


def target_lazy_center_items(source_center_items, root, data_split, names):
    """Yield ``(name, normalized_HE, target_loader)`` using the bound source.

    ``source_center_items`` is the caller's accepted native callable, normally
    ``adapter.rt['center_items']``. Its globals supply every image/array/transform
    primitive; this module imports no scientific runtime. The caller owns name
    membership, source identity, scientific split labels and execution authority.
    ``data_split`` is only the original on-disk train/val location.

    Each zero-argument target loader binds its own image path and rechecks that
    path when invoked. Advancing the iterator never opens a label image.
    """
    if data_split not in ('train', 'val'):
        raise ValueError('only train/val center crops are allowed')
    namespace = getattr(source_center_items, '__globals__', {})
    try:
        image_open = namespace['Image'].open
        to_tensor = namespace['TF'].to_tensor
        as_array = namespace['np'].array
        safe_path = namespace['_safe_path']
    except (KeyError, AttributeError) as error:
        raise ValueError('bound native center_items primitives are required') from error
    if not all(callable(value) for value in (image_open, to_tensor, as_array, safe_path)):
        raise ValueError('bound native center_items primitives must be callable')
    root = safe_path(root)

    def center_crop(path):
        with image_open(safe_path(path)) as image:
            if image.size != (1024, 1024):
                raise ValueError('HEMIT evaluation requires native 1024 images')
            return image.convert('RGB').crop((256, 256, 768, 768))

    def make_target_loader(path):
        def target_loader():
            return as_array(center_crop(path))
        return target_loader

    for name in names:
        if not isinstance(name, str) or Path(name).name != name or not name or name in ('.', '..'):
            raise ValueError('image name must be a plain basename')
        he = (to_tensor(center_crop(root/data_split/'input'/name)) - .5) * 2
        yield name, he, make_target_loader(root/data_split/'label'/name)
