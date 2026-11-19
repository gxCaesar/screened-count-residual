"""CPU-only membership and pooled-recomposition checks for optional sweep evidence."""

import json
from pathlib import Path

import numpy as np


COUNT_KEYS = ("n_p", "tp_p", "tp_g", "n_g")
CURVE_KEYS = ("n_p", "tp_p", "tp_g")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def selected_train(names, n):
    _require(type(n) is int and 0 < n <= len(names), "n_train must be within train membership")
    return names[::max(1, len(names) // n)][:n]


def validate_manifest(data_root, manifest):
    """Validate complete train/val input-label membership before any image is opened."""
    root = Path(data_root).resolve(strict=True)
    _require(isinstance(manifest, dict), "manifest must be an object")
    _require(set(manifest) <= {"train", "val", "group_ids"}, "unexpected manifest fields")
    result = {}
    for split in ("train", "val"):
        names = manifest.get(split)
        _require(isinstance(names, list) and bool(names), f"{split}: expected nonempty names")
        _require(all(isinstance(name, str) and name and name not in (".", "..")
                     and Path(name).name == name and not Path(name).is_absolute()
                     and "\\" not in name for name in names),
                 f"{split}: paths must be plain image basenames, never test/out-of-root paths")
        _require(names == sorted(set(names)), f"{split}: names must be sorted and unique")
        for kind in ("input", "label"):
            directory = root / split / kind
            _require(directory.resolve(strict=True) == directory,
                     f"{split}/{kind}: directory must remain inside the data root")
            actual = {p.name for p in directory.iterdir()}
            _require(actual == set(names),
                     f"{split}/{kind}: missing or unexpected image membership")
            for name in names:
                path = directory / name
                _require(path.is_file() and path.resolve(strict=True).parent == directory,
                         f"{split}/{kind}/{name}: file leaves its split directory")
        result[split] = names
    _require(not (set(result["train"]) & set(result["val"])),
             "train/val image membership overlaps")
    groups = manifest.get("group_ids", {})
    _require(isinstance(groups, dict) and set(groups) <= {"train", "val"},
             "group_ids must contain only train/val mappings")
    for split, mapping in groups.items():
        _require(isinstance(mapping, dict) and set(mapping) == set(result[split])
                 and all(isinstance(value, str) and value for value in mapping.values()),
                 f"group_ids/{split}: expected a nonempty ID for every image in the split")
    return result


def per_image_payload(rows_by_split, arms, markers, manifest, thresholds, tol):
    """Pack only threshold counts, image names and explicitly supplied group IDs."""
    payload = {"schema_version": np.array(1), "arms": np.asarray(arms),
               "markers": np.asarray(markers), "tol_px": np.array(tol),
               "thresholds_json": np.array(json.dumps(thresholds, sort_keys=True))}
    for split, rows in rows_by_split.items():
        names = [name for name, _ in rows]
        groups = manifest.get("group_ids", {}).get(split)
        payload[f"{split}_names"] = np.asarray(names)
        payload[f"{split}_group_ids"] = np.asarray(
            [groups[name] if groups is not None else "" for name in names], dtype=str)
        for qi, key in enumerate(CURVE_KEYS):
            payload[f"{split}_{key}"] = np.asarray(
                [[counts[(arm, marker)][qi] for marker in markers] for _, counts in rows
                 for arm in arms], dtype=np.int64).reshape(len(rows), len(arms), len(markers), 256)
        payload[f"{split}_n_g"] = np.asarray(
            [[counts[(arm, marker)][3] for marker in markers] for _, counts in rows
             for arm in arms], dtype=np.int64).reshape(len(rows), len(arms), len(markers))
    return payload


def validate_per_image(payload, sweep, manifest):
    """Check every image curve and its exact pooled reconstruction against sweep.json."""
    arms = list(sweep["summary"])
    markers = list(sweep["thresholds"])
    _require(list(payload["arms"]) == arms and list(payload["markers"]) == markers,
             "per-image arm or marker axes differ from sweep")
    _require(int(payload["schema_version"]) == 1, "unsupported per-image schema")
    _require(int(payload["tol_px"]) == sweep["tol_px"], "tolerance differs from sweep")
    _require(json.loads(str(payload["thresholds_json"])) == sweep["thresholds"],
             "fixed thresholds differ from sweep")
    result = {}
    for split in ("train", "val"):
        names = [str(x) for x in payload[f"{split}_names"]]
        expected = selected_train(manifest["train"], sweep["n_train"]) if split == "train" else manifest["val"]
        _require(len(names) == len(set(names)) and set(names) == set(expected)
                 and len(names) == len(expected), f"{split}: duplicate, missing or unexpected image membership")
        group_ids = [str(x) for x in payload[f"{split}_group_ids"]]
        _require(len(group_ids) == len(names), f"{split}: group ID count differs from images")
        supplied = manifest.get("group_ids", {}).get(split)
        _require(group_ids == ([supplied[n] for n in names] if supplied is not None else [""] * len(names)),
                 f"{split}: group IDs differ from supplied manifest")
        shape = (len(names), len(arms), len(markers), 256)
        arrays = {}
        for key in CURVE_KEYS:
            a = np.asarray(payload[f"{split}_{key}"])
            _require(a.shape == shape and a.dtype.kind in "iu" and np.all(a >= 0)
                     and np.all(a[..., :-1] >= a[..., 1:]),
                     f"{split}/{key}: invalid 256-threshold integer curve")
            arrays[key] = a
        ng = np.asarray(payload[f"{split}_n_g"])
        _require(ng.shape == shape[:3] and ng.dtype.kind in "iu" and np.all(ng >= 0),
                 f"{split}/n_g: invalid label counts")
        _require(np.all(arrays["tp_p"] <= arrays["n_p"])
                 and np.all(arrays["tp_g"] <= ng[..., None])
                 and np.all((ng[..., None] != 0) | (arrays["tp_p"] == 0))
                 and np.all((arrays["n_p"] != 0) | (arrays["tp_g"] == 0))
                 and np.all(arrays["n_p"][..., -1] == 0),
                 f"{split}: impossible per-image count bounds")
        _require(np.all(ng == ng[:, :1, :]), f"{split}: label counts differ across arms")
        for ai, arm in enumerate(arms):
            for mi, marker in enumerate(markers):
                curve = sweep["curves"][split][arm][marker]
                for key in CURVE_KEYS:
                    _require(np.array_equal(arrays[key][:, ai, mi].sum(axis=0), curve[key]),
                             f"{split}/{arm}/{marker}/{key}: pooled curve differs")
                _require(int(ng[:, ai, mi].sum()) == curve["n_g"],
                         f"{split}/{arm}/{marker}/n_g: pooled count differs")
                if split == "val":
                    by_name = {row["name"]: row for row in sweep["fixed_rows"]}
                    _require(set(by_name) == set(names) and len(by_name) == len(names),
                             "val: fixed_rows membership differs")
                    t = int(sweep["thresholds"][marker])
                    for i, name in enumerate(names):
                        fixed = [int(arrays["n_p"][i, ai, mi, t]),
                                 int(arrays["tp_p"][i, ai, mi, t]), int(ng[i, ai, mi]),
                                 int(arrays["tp_g"][i, ai, mi, t])]
                        _require(fixed == by_name[name][f"{arm}__{marker}"],
                                 f"val/{name}/{arm}/{marker}: fixed counts differ")
        result[split] = len(names)
    return result


def load_and_validate(path, sweep, manifest):
    with np.load(path, allow_pickle=False) as payload:
        return validate_per_image(payload, sweep, manifest)
