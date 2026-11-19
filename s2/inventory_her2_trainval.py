"""Inventory only HER2match HE/IHC train and val directories, never score CSV or test.

Filenames establish paired tiles and source-WSI labels. They do not establish
patient identity, image alignment, clinical labels or image quality.
"""
import argparse
import json
from pathlib import Path
import re


IMAGE = re.compile(r'^(wsi\d+)_id\d+_x\d+_y\d+\.jpg$')


def inventory(root):
    root = Path(root).resolve(strict=True)
    result = {'scope': 'train_val_only', 'root': str(root), 'splits': {},
              'group_type': 'filename_source_wsi', 'patient_identity': 'unresolved',
              'score_metadata_read': False, 'test_directory_read': False,
              'image_pixels_read': False}
    all_names, all_groups = {}, {}
    for split in ('train', 'val'):
        names, ignored = {}, {}
        for modality in ('HE', 'IHC'):
            directory = root/modality/split
            if directory.resolve(strict=True) != directory:
                raise ValueError(f'{modality}/{split}: split directory is a symlink')
            names[modality], ignored[modality] = [], []
            for path in directory.iterdir():
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f'{modality}/{split}: nonregular entry')
                if path.name == '.DS_Store':
                    ignored[modality].append(path.name)
                    continue
                if not IMAGE.fullmatch(path.name):
                    raise ValueError(f'{modality}/{split}: unrecognized image basename {path.name}')
                names[modality].append(path.name)
            names[modality].sort()
        if not names['HE'] or names['HE'] != names['IHC']:
            raise ValueError(f'{split}: absent or unmatched HE/IHC images')
        groups = {name: IMAGE.fullmatch(name).group(1) for name in names['HE']}
        counts = {group: sum(g == group for g in groups.values()) for group in sorted(set(groups.values()))}
        result['splits'][split] = {'n_pairs': len(groups), 'names': names['HE'],
                                  'group_ids': groups, 'wsi_counts': counts,
                                  'n_filename_wsi_groups': len(counts), 'ignored_metadata': ignored}
        all_names[split], all_groups[split] = set(groups), set(groups.values())
    if all_names['train'] & all_names['val'] or all_groups['train'] & all_groups['val']:
        raise ValueError('train/val names or source-WSI groups overlap')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    result = inventory(args.root)
    with open(args.out, 'x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({split: {k: v for k, v in info.items() if k not in ('names', 'group_ids')}
                      for split, info in result['splits'].items()}))
