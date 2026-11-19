"""Final-byte scoring and scalar-only consistency analysis for the fixed pilot.

Only ``score_image`` accepts pixels. The saved-scalar consumer never reads a
file, reopens a target, calls RGB quality, fits a model, or establishes a second
truth measurement. RGB order here follows the accepted metric_panel contract
(panCK, CD3, DAPI), rather than the stale labels in s2_scoring.py.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import json
import math
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

from . import _local_exchange_reference as reference


MARKERS = ("panCK", "CD3", "DAPI")
CUTOFFS = (22, 30, 18)
FIT_THRESHOLD = 21
OUTER = tuple(range(0, 32, 4))
TRAINING = tuple(i for i in range(32) if i % 4)
ARMS = ("P_F", "M_F", "S_F", "E_F", "P_C", "S_C", "E_C", "E_shift")
SEEDS = (202610270, 202610271)
COUNT_FIELDS = ("K", "U", "V", "T")
ATOL = 1e-12
ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOT = ROOT / "experiments/20261013-s2-evidence-recovery/pixel-full-control/code"
QUALITY_CAPTURE = (ROOT / "experiments/20261021-s2-full-catalogue-count-fit/"
                   "correspondence-profile-control/canonical-source.json")
COMMON_METRICS = ("ODS_F", "FITtau_tolerant_F", "cutoff_tolerant_F",
                  "FITtau_exact_F1", "cutoff_exact_F1", "avg_ssim", "avg_psnr")
NEW_METRICS = (*COMMON_METRICS, "macro_ODS_F", "D", "C", "changed_pixels")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(value, name, *, signed=False):
    _require(type(value) is int and (signed or value >= 0), name + ": integer required")
    return value


def _number(value, name):
    _require(type(value) in (int, float) and math.isfinite(value), name + ": finite number required")
    return float(value)


def _same(actual, expected, name):
    _require(math.isclose(actual, expected, rel_tol=0., abs_tol=ATOL), name + ": scalar mismatch")


def _byte_rgb(value, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.uint8,
             name + ": uint8 ndarray required; no implicit casting or quantization")
    _require(value.ndim == 3 and value.shape[2] == 3 and min(value.shape[:2]) > 10,
             name + ": HWC RGB with dimensions >10 required")
    return value


def _node(text, kind, name):
    nodes = [n for n in ast.parse(text).body if isinstance(n, kind) and n.name == name]
    _require(len(nodes) == 1, "one native source definition required: " + name)
    return nodes[0]


def canonical_quality(metrics_path=None, panel_path=None, *, capture_path=None):
    """Bind the actual native SSIM class and accepted native PSNR expression.

    Missing or changed sources fail explicitly. No skimage/uniform-window or
    learned quality fallback is used. This factory reads code and metadata only.
    """
    metrics_path = Path(metrics_path) if metrics_path is not None else QUALITY_ROOT / "metrics.py"
    panel_path = Path(panel_path) if panel_path is not None else QUALITY_ROOT / "metric_panel.py"
    capture_path = Path(capture_path) if capture_path is not None else QUALITY_CAPTURE
    for path in (metrics_path, panel_path, capture_path):
        _require(path.is_file(), "canonical quality source unavailable: " + str(path))
    text, panel = metrics_path.read_text(), panel_path.read_text()
    node = _node(text, ast.ClassDef, "SSIM")
    captured = json.loads(capture_path.read_text())["sources"]["metrics.py"]
    _require(ast.get_source_segment(text, node) == ast.get_source_segment(
        captured, _node(captured, ast.ClassDef, "SSIM")), "native SSIM source differs from accepted capture")
    score_node = _node(panel, ast.FunctionDef, "score")
    expressions = [n.value for n in ast.walk(score_node) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.JoinedStr)
                           and any(isinstance(v, ast.Constant) and v.value == "_psnr"
                                   for v in t.slice.values) for t in n.targets)]
    expected = ast.parse("float(20 * np.log10(255.0 / (np.sqrt(mse) + 1e-8)))", mode="eval").body
    _require(len(expressions) == 1 and ast.dump(expressions[0], include_attributes=False) ==
             ast.dump(expected, include_attributes=False), "native per-channel PSNR expression differs")
    namespace = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(metrics_path), "exec"), namespace)
    ssim = namespace["SSIM"]._ssim
    psnr = compile(ast.Expression(body=expressions[0]), str(panel_path), "eval")

    def quality(image, target):
        _byte_rgb(image, "quality image"); _byte_rgb(target, "quality target")
        _require(image.shape == target.shape, "quality shape mismatch")
        mse = np.square(image.astype(np.float64) - target.astype(np.float64)).mean(axis=(0, 1))
        sc = [float(ssim(image[..., c], target[..., c])) for c in range(3)]
        pc = [float(eval(psnr, {"np": np, "float": float}, {"mse": float(v)})) for v in mse]
        return dict(avg_ssim=sum(sc)/3, avg_psnr=sum(pc)/3,
                    ssim_channels=sc, psnr_channels=pc)

    quality.metadata = dict(sources=[dict(path=str(p.resolve()), size_bytes=p.stat().st_size,
                                         mtime_ns=p.stat().st_mtime_ns)
                                     for p in (metrics_path, panel_path, capture_path)],
                            SSIM="native 11x11 Gaussian sigma1.5, valid-center",
                            PSNR="native per-channel byte MSE; arithmetic channel mean",
                            marker_order=list(MARKERS), target_reads=0)
    return quality


def load_quality(directory=None):
    """Load canonical source from an explicit/staged directory or the repo."""
    staged = ROOT / "quality"
    if directory is not None or staged.is_dir():
        directory = Path(directory) if directory is not None else staged
        return canonical_quality(directory/"metrics.py", directory/"metric_panel.py",
                                 capture_path=directory/"canonical-source.json")
    return canonical_quality()


@lru_cache(maxsize=1)
def _default_quality():
    return load_quality()


@lru_cache(maxsize=1)
def tolerance_kernel():
    """The native sweep_panel L2/5 chamfer footprint, with zero image exterior."""
    z = np.ones((17, 17), np.uint8); z[8, 8] = 0
    if cv2 is not None:
        kernel = cv2.distanceTransform(z, cv2.DIST_L2, 5) <= 5
    else:
        from scipy.ndimage import distance_transform_edt
        kernel = distance_transform_edt(z) <= 5.0
    _require(int(kernel.sum()) == 81, "native tolerance footprint changed")
    kernel.setflags(write=False)
    return kernel


def validate_counts(counts, name="counts", *, pixels=None):
    _require(isinstance(counts, dict) and set(counts) == {*COUNT_FIELDS, "G"}, name + ": K/U/V/T/G required")
    g = _integer(counts["G"], name + ".G")
    if pixels is not None:
        _require(g <= pixels, name + ": G exceeds pixel denominator")
    for field in COUNT_FIELDS:
        curve = counts[field]
        _require(isinstance(curve, list) and len(curve) == 256
                 and all(type(v) is int and v >= 0 for v in curve), name + "." + field + ": 256 integer counts required")
        _require(curve[-1] == 0 and all(a >= b for a, b in zip(curve, curve[1:])),
                 name + "." + field + ": strict-threshold monotone tail required")
    _require(all(t <= u <= k and t <= v <= g and t <= min(k, g)
                 for k, u, v, t in zip(*(counts[f] for f in COUNT_FIELDS))), name + ": impossible overlap/denominator")
    _require(pixels is None or counts["K"][0] <= pixels, name + ": K exceeds pixels")
    return counts


def endpoints(counts, cutoff=22):
    """ODS is max after image-count pooling; the empty tolerant curve is zero."""
    tolerant, exact = [], []
    for k, u, v, t in zip(*(counts[f] for f in COUNT_FIELDS)):
        den = u*counts["G"] + v*k
        tolerant.append(2*u*v/den if den else 0.)
        exact.append(2*t/(k+counts["G"]) if k+counts["G"] else 1.)
    peak = max(tolerant)
    return dict(ODS_F=peak, ODS_threshold=tolerant.index(peak),
                FITtau_tolerant_F=tolerant[FIT_THRESHOLD], cutoff_tolerant_F=tolerant[cutoff],
                FITtau_exact_F1=exact[FIT_THRESHOLD], cutoff_exact_F1=exact[cutoff],
                tolerant_F_curve=tolerant, exact_F1_curve=exact)


def _quality(value, name):
    _require(isinstance(value, dict), name + ": native quality dictionary required")
    result = {k: _number(value[k], name + "." + k) for k in ("avg_ssim", "avg_psnr")}
    for field, average in (("ssim_channels", "avg_ssim"), ("psnr_channels", "avg_psnr")):
        _require(isinstance(value.get(field), (list, tuple)) and len(value[field]) == 3,
                 name + ": all3 canonical quality channels required")
        result[field] = [_number(v, name + "." + field) for v in value[field]]
        _same(result[average], sum(result[field])/3, name + ".RGB channel mean")
    return result


def _histogram(value):
    return np.bincount(value.ravel(), minlength=256).astype(np.int64).tolist()


def _tail(histogram):
    total, result = sum(histogram), []
    for n in histogram:
        total -= n; result.append(total)
    return result


def _clipping(receipt, pixels, byte_mean):
    if receipt is None or isinstance(receipt, dict) and receipt.get("status") == "not_run":
        return dict(status="not_run", reason="preclipping values are not inferable from final bytes", pixels=pixels)
    _require(isinstance(receipt, dict), "clipping: scalar receipt required")
    _require(receipt.get("pixels") == pixels, "clipping: pixel denominator mismatch")
    low, high = (_integer(receipt.get(k), "clipping." + k) for k in ("below_zero", "above_one"))
    _require(low+high <= pixels, "clipping: counts exceed pixels")
    means = {k: _number(receipt.get(k), "clipping." + k)
             for k in ("preclip_mean", "postclip_mean", "rounding_mean_change")}
    _require(0 <= means["postclip_mean"] <= 1, "clipping: invalid postclip mean")
    _same(means["rounding_mean_change"], byte_mean/255-means["postclip_mean"], "clipping.rounding_mean_change")
    return dict(status="measured", pixels=pixels, below_zero=low, above_one=high,
                clipped_pixels=low+high, clipping_fraction=(low+high)/pixels, **means)


def score_image(image, target, donor, quality=None, *, clipping=None):
    """Score one frozen output, using final bytes and exactly one RGB call.

    ``clipping`` is the frozen generator's scalar receipt in normalized units,
    not a quantity recovered from clipped bytes. The caller attaches ``arm``,
    ``seed`` and ``original_index`` before grouping. No file or target is opened.
    """
    image, target, donor = (_byte_rgb(v, n) for v, n in
                            ((image, "image"), (target, "target"), (donor, "donor")))
    _require(image.shape == target.shape == donor.shape, "image/target/donor shape mismatch")
    _require(np.array_equal(image[..., 1:], donor[..., 1:]), "CD3/DAPI must be unchanged donor bytes")
    quality = _default_quality() if quality is None else quality
    _require(callable(quality), "canonical quality callable required")
    q = _quality(quality(image, target), "quality")
    kernel, counts, before, histograms = tolerance_kernel(), {}, {}, {}
    for channel, (marker, cutoff) in enumerate(zip(MARKERS, CUTOFFS)):
        labels = target[..., channel] > cutoff
        def count(values):
            raw = reference.counts(values, labels, kernel)
            return {k: int(v) if k == "G" else v.tolist() for k, v in raw.items()}
        before[marker] = count(donor[..., channel])
        counts[marker] = count(image[..., channel]) if channel == 0 else before[marker].copy()
        histograms[marker] = {k: _histogram(v[..., channel])
                              for k, v in (("output", image), ("donor", donor), ("target", target))}
        histograms[marker]["shift"] = [a-b for a, b in zip(histograms[marker]["output"], histograms[marker]["donor"])]
    x, y, d = (v.astype(np.int64) for v in (image, target, donor))
    error, base, edit = x-y, d-y, x-d
    moments = dict(output_target_SSE=np.square(error).sum(axis=(0, 1)).tolist(),
                   donor_target_SSE=np.square(base).sum(axis=(0, 1)).tolist(),
                   edit_SSE=np.square(edit).sum(axis=(0, 1)).tolist(),
                   edit_donor_error_cross=(edit*base).sum(axis=(0, 1)).tolist())
    D, C = moments["output_target_SSE"][0]-moments["donor_target_SSE"][0], moments["edit_SSE"][0]
    pixels = int(image.shape[0]*image.shape[1])
    changed = np.count_nonzero(edit, axis=(0, 1)).tolist()
    means = {m: dict(output=sum(t*n for t, n in enumerate(histograms[m]["output"]))/pixels,
                     donor=sum(t*n for t, n in enumerate(histograms[m]["donor"]))/pixels)
             for m in MARKERS}
    for value in means.values(): value["shift"] = value["output"]-value["donor"]
    row = dict(schema_version=1, shape=list(image.shape), pixels=pixels, marker_order=list(MARKERS),
               counts=counts, donor_counts=before, endpoints={m: endpoints(counts[m], t) for m, t in zip(MARKERS, CUTOFFS)},
               quality=q, quality_metadata=getattr(quality, "metadata", {"source_binding": "caller_supplied"}),
               moments=moments, D=D, C=C, D_over_C=D/C if C else None,
               changed_pixels=int(np.count_nonzero(np.any(edit != 0, axis=2))), changed_values_by_marker=changed,
               changed_pixel_fraction=changed[0]/pixels, no_op=C == 0,
               histograms=histograms, means=means, clipping=_clipping(clipping, pixels, means["panCK"]["output"]))
    row["macro_ODS_F"] = sum(row["endpoints"][m]["ODS_F"] for m in MARKERS)/3
    return row


def check_saved_scalar(row):
    """Pure Python algebra on JSON scalars; this is not independent truth."""
    _require(isinstance(row, dict) and row.get("marker_order") == list(MARKERS), "saved row: canonical marker order required")
    shape = row.get("shape")
    _require(isinstance(shape, list) and len(shape) == 3 and shape[2] == 3
             and all(type(v) is int and v > 10 for v in shape[:2]), "saved row: HWC geometry required")
    n = _integer(row.get("pixels"), "pixels")
    _require(n == shape[0]*shape[1], "saved row: pixel denominator mismatch")
    _quality(row.get("quality"), "quality")
    moments = row.get("moments")
    _require(isinstance(moments, dict), "saved row: SSE moments required")
    for key in ("output_target_SSE", "donor_target_SSE", "edit_SSE", "edit_donor_error_cross"):
        values = moments.get(key)
        _require(isinstance(values, list) and len(values) == 3, "moments." + key + ": all3 channels required")
        for value in values: _integer(value, "moments." + key, signed=key.endswith("cross"))
    changed = row.get("changed_values_by_marker")
    _require(isinstance(changed, list) and len(changed) == 3, "changed values: all3 channels required")
    for channel, (marker, cutoff) in enumerate(zip(MARKERS, CUTOFFS)):
        count = validate_counts(row["counts"][marker], marker + ".counts", pixels=n)
        before = validate_counts(row["donor_counts"][marker], marker + ".donor_counts", pixels=n)
        histogram = row["histograms"][marker]
        for side in ("output", "donor", "target"):
            h = histogram[side]
            _require(isinstance(h, list) and len(h) == 256 and all(type(v) is int and v >= 0 for v in h)
                     and sum(h) == n, marker + ": histogram/pixel denominator mismatch")
        _require(count["K"] == _tail(histogram["output"]) and before["K"] == _tail(histogram["donor"]), marker + ": K/histogram mismatch")
        _require(histogram.get("shift") == [a-b for a, b in zip(histogram["output"], histogram["donor"])],
                 marker + ": signed histogram shift mismatch")
        _require(count["G"] == before["G"] == _tail(histogram["target"])[cutoff], marker + ": target cutoff/G mismatch")
        expected = endpoints(count, cutoff)
        _require(row["endpoints"][marker] == expected, marker + ": saved endpoint/curve mismatch")
        for side in ("output", "donor"):
            _same(_number(row["means"][marker][side], marker + ".mean"),
                  sum(t*v for t, v in enumerate(histogram[side]))/n, marker + ".mean/histogram")
        _same(row["means"][marker]["shift"], row["means"][marker]["output"]-row["means"][marker]["donor"], marker + ".mean shift")
        c = _integer(changed[channel], marker + ".changed values")
        cost = moments["edit_SSE"][channel]
        _require(c <= n and c <= cost <= 255**2*c, marker + ": changed/SSE denominator mismatch")
        _require(max(moments["output_target_SSE"][channel], moments["donor_target_SSE"][channel]) <= 255**2*n,
                 marker + ": target SSE exceeds byte/pixel denominator")
        _require(moments["edit_donor_error_cross"][channel]**2 <= cost*moments["donor_target_SSE"][channel],
                 marker + ": saved cross moment violates Cauchy bound")
        delta = moments["output_target_SSE"][channel]-moments["donor_target_SSE"][channel]
        _require(delta == cost+2*moments["edit_donor_error_cross"][channel], marker + ": signed D moment identity mismatch")
        _require(abs(sum(t*(histogram["output"][t]-histogram["donor"][t]) for t in range(256))) <= 255*c,
                 marker + ": mean shift exceeds changed-value support")
        if channel:
            _require(c == cost == delta == 0 and count == before
                     and histogram["output"] == histogram["donor"], marker + ": unchanged donor channel defect")
    D = moments["output_target_SSE"][0]-moments["donor_target_SSE"][0]
    C = moments["edit_SSE"][0]
    _require(type(row.get("D")) is int and row["D"] == D and type(row.get("C")) is int and row["C"] == C,
             "saved row: signed byte D/C mismatch")
    _require(row.get("changed_pixels") == changed[0] and type(row.get("changed_pixels")) is int,
             "saved row: changed pixel denominator mismatch")
    _same(row["changed_pixel_fraction"], changed[0]/n, "changed pixel fraction")
    _require(type(row.get("no_op")) is bool and row["no_op"] == (C == 0), "saved row: no-op mismatch")
    _require(row.get("D_over_C") == (D/C if C else None), "saved row: D/C ratio mismatch")
    _same(row["macro_ODS_F"], sum(row["endpoints"][m]["ODS_F"] for m in MARKERS)/3, "macro ODS")
    receipt = row.get("clipping")
    _require(isinstance(receipt, dict) and receipt == _clipping(receipt, n, row["means"]["panCK"]["output"]),
             "saved row: clipping receipt mismatch")
    return dict(status="passed_saved_scalar_algebra", pixels_read=0, target_reads=0, RGB_rescores=0,
                new_fits=0, independent_second_truth_measurement=False)


def group_key(arm, seed):
    return f"{arm}/seed-{seed}"


def _pool_counts(rows, side, marker):
    return {"G": sum(r[side][marker]["G"] for r in rows),
            **{f: [sum(r[side][marker][f][t] for r in rows) for t in range(256)] for f in COUNT_FIELDS}}


def aggregate(rows):
    """Require exactly the eight original queries of one frozen arm/seed."""
    _require(isinstance(rows, list) and len(rows) == 8 and
             all(type(r.get("original_index")) is int for r in rows) and
             sorted(r.get("original_index", -1) for r in rows) == list(OUTER), "aggregate: all8 original queries required")
    arm, seed = rows[0].get("arm"), rows[0].get("seed")
    _require(arm in ARMS and type(seed) is int and seed in SEEDS
             and all(r.get("arm") == arm and r.get("seed") == seed for r in rows), "aggregate: one fixed arm/seed required")
    return _aggregate(rows, key=group_key(arm, seed), arm=arm, seed=seed,
                      population="outer_exposed_FIT8")


def aggregate_training_witness(rows):
    """Pool the training-only feasible witness, independently of outer128."""
    _require(isinstance(rows, list) and len(rows) == 24 and
             all(type(r.get("original_index")) is int for r in rows) and
             sorted(r.get("original_index", -1) for r in rows) == list(TRAINING),
             "training witness: exact24 training indices required; no outer queries")
    return _aggregate(rows, key="training24_privileged_feasible_witness",
                      population="training24_only_not_outer_not_upper_bound")


def _aggregate(rows, **identity):
    rows = sorted(rows, key=lambda r: r["original_index"])
    for row in rows: check_saved_scalar(row)
    _require(all(r["shape"] == rows[0]["shape"] for r in rows), "aggregate: native shape mismatch")
    images = len(rows)
    counts = {m: _pool_counts(rows, "counts", m) for m in MARKERS}
    q = {k: sum(r["quality"][k] for r in rows)/images for k in ("avg_ssim", "avg_psnr")}
    q.update({k: [sum(r["quality"][k][c] for r in rows)/images for c in range(3)]
              for k in ("ssim_channels", "psnr_channels")})
    ep = {m: endpoints(counts[m], t) for m, t in zip(MARKERS, CUTOFFS)}
    D, C = sum(r["D"] for r in rows), sum(r["C"] for r in rows)
    n, changed = sum(r["pixels"] for r in rows), sum(r["changed_pixels"] for r in rows)
    measured = all(r["clipping"]["status"] == "measured" for r in rows)
    clip = dict(status="measured" if measured else "not_run", pixels=n,
                measured_images=sum(r["clipping"]["status"] == "measured" for r in rows))
    if measured:
        clip.update({k: sum(r["clipping"][k] for r in rows) for k in ("below_zero", "above_one", "clipped_pixels")})
        clip["clipping_fraction"] = clip["clipped_pixels"]/n
        clip.update({k: sum(r["clipping"][k] for r in rows)/images
                     for k in ("preclip_mean", "postclip_mean", "rounding_mean_change")})
    histograms = {m: {side: [sum(r["histograms"][m][side][t] for r in rows) for t in range(256)]
                      for side in ("output", "donor", "target", "shift")} for m in MARKERS}
    means = {m: {side: sum(r["means"][m][side] for r in rows)/images
                 for side in ("output", "donor", "shift")} for m in MARKERS}
    mean_image = sum(r["endpoints"]["panCK"]["ODS_F"] for r in rows)/images
    return dict(**identity, images=images, original_indices=[r["original_index"] for r in rows],
                pixels=n, counts=counts, donor_counts={m: _pool_counts(rows, "donor_counts", m) for m in MARKERS},
                endpoints=ep, quality=q, **{k: ep["panCK"][k] for k in COMMON_METRICS[:5]},
                avg_ssim=q["avg_ssim"], avg_psnr=q["avg_psnr"], macro_ODS_F=sum(ep[m]["ODS_F"] for m in MARKERS)/3,
                mean_image_ODS_F=mean_image, pooled_ODS_minus_mean_image_ODS=ep["panCK"]["ODS_F"]-mean_image,
                D=D, C=C, D_over_C=D/C if C else None, changed_pixels=changed, changed_pixel_fraction=changed/n,
                changed_values_by_marker=[sum(r["changed_values_by_marker"][c] for r in rows) for c in range(3)],
                noops=sum(r["no_op"] for r in rows), clipping=clip, histograms=histograms, means=means)


def _image_endpoint(row):
    return dict(**{k: row["endpoints"]["panCK"][k] for k in COMMON_METRICS[:5]},
                avg_ssim=row["quality"]["avg_ssim"], avg_psnr=row["quality"]["avg_psnr"],
                **{k: row[k] for k in ("macro_ODS_F", "D", "C", "changed_pixels")})


def _delta(a, b, fields):
    return {k: a[k]-b[k] for k in fields}


def check_saved_scalars(rows, groups=None):
    """Check complete coverage and saved pooling with zero pixel access."""
    _require(isinstance(rows, list) and len(rows) == 128, "saved rows: all128 outputs required")
    _require(all(type(r.get("original_index")) is int and type(r.get("seed")) is int
                 and type(r.get("arm")) is str for r in rows), "saved rows: integer query/seed and string arm identities required")
    keys = [(r.get("arm"), r.get("seed"), r.get("original_index")) for r in rows]
    _require(len(set(keys)) == 128 and set(keys) == {(a, s, i) for a in ARMS for s in SEEDS for i in OUTER},
             "saved rows: exact128 arm/seed/query memberships required")
    result = [aggregate([r for r in rows if r["arm"] == a and r["seed"] == s]) for a in ARMS for s in SEEDS]
    by_image = {i: [r for r in rows if r["original_index"] == i] for i in OUTER}
    for index, image_rows in by_image.items():
        anchor = image_rows[0]
        for row in image_rows[1:]:
            _require(row["donor_counts"] == anchor["donor_counts"], f"image{index}: donor count anchor changed")
            for marker in MARKERS:
                _require(row["histograms"][marker]["target"] == anchor["histograms"][marker]["target"]
                         and row["histograms"][marker]["donor"] == anchor["histograms"][marker]["donor"],
                         f"image{index}: donor/target histogram anchor changed")
            _require(row["moments"]["donor_target_SSE"] == anchor["moments"]["donor_target_SSE"],
                     f"image{index}: donor target-SSE anchor changed")
            for c in (1, 2):
                for k in ("ssim_channels", "psnr_channels"):
                    _same(row["quality"][k][c], anchor["quality"][k][c], f"image{index}: unchanged marker quality")
    if groups is not None:
        _require(groups == result, "saved groups: pooled counts/ODS/quality or delivery mismatch")
    return dict(status="passed_all128_saved_scalar_consistency", checked_outputs=128, checked_groups=16,
                checked_marker_count_curves=128*3*2*4, pixels_read=0, target_reads=0, RGB_rescores=0, new_fits=0,
                independent_second_truth_measurement=False, groups=result)


def analyze(rows, old229):
    """Preserve all rows and old panCK references; apply the prospective gate.

    Patient independence, adequate optimization, novelty and the charter gate
    cannot be established by this scalar consumer and remain unassessed.
    """
    checked = check_saved_scalars(rows)
    groups = checked["groups"]; lookup = {g["key"]: g for g in groups}
    _require(isinstance(old229, dict), "old229: unchanged saved JSON object required")
    old = old229.get("all229_pooled_endpoints")
    _require(isinstance(old, list) and len(old) == 229 and len({r.get("key") for r in old}) == 229,
             "old229: exact229 unique pooled endpoints required")
    old_lookup = {r["key"]: r for r in old}
    required = ("donor_noop", "coupled/density-P-constant", "coupled/density-P-cheap", "coupled/density-P-FM")
    _require(all(k in old_lookup for k in required) and any("cheap" in k and k.endswith("/half") for k in old_lookup),
             "old229: donor/constant/cheap/FM/cheap-half named references required")
    for endpoint in old:
        _require(endpoint.get("images") == 8, "old229: unchanged FIT8 denominator required")
        for k in COMMON_METRICS: _number(endpoint[k], "old229." + k)
    old_images = old229.get("all1832_image_endpoints")
    _require(isinstance(old_images, list) and len(old_images) == 1832, "old229: all1832 image endpoints required")
    old_image_lookup = {(r.get("key"), r.get("original_index")): r for r in old_images}
    _require(set(old_image_lookup) == {(k, i) for k in old_lookup for i in OUTER}, "old229: image membership mismatch")
    for endpoint in old_images:
        for k in COMMON_METRICS: _number(endpoint[k], "old229 image." + k)
    new_image = {(group_key(r["arm"], r["seed"]), r["original_index"]): _image_endpoint(r) for r in rows}
    contrasts, old_comparisons = [], []
    for a in groups:
        for b in groups:
            if a["key"] == b["key"]: continue
            paired = [dict(original_index=i, delta=_delta(new_image[a["key"], i], new_image[b["key"], i], NEW_METRICS)) for i in OUTER]
            signs = {k: dict(positive=sum(p["delta"][k] > ATOL for p in paired),
                            negative=sum(p["delta"][k] < -ATOL for p in paired),
                            zero=sum(abs(p["delta"][k]) <= ATOL for p in paired)) for k in NEW_METRICS}
            contrasts.append(dict(policy=a["key"], reference=b["key"], pooled_delta=_delta(a, b, NEW_METRICS),
                                  all8_image_differences=paired, image_signs=signs))
        for b in old:
            old_comparisons.append(dict(policy=a["key"], reference=b["key"], endpoint_scope="panCK and saved coherent RGB only",
                pooled_delta=_delta(a, b, COMMON_METRICS), all8_image_differences=[
                    dict(original_index=i, delta=_delta(new_image[a["key"], i], old_image_lookup[b["key"], i], COMMON_METRICS)) for i in OUTER]))
    strongest_value = max(r["ODS_F"] for r in old)
    strongest_keys = [r["key"] for r in old if r["ODS_F"] == strongest_value]
    donor = old_lookup["donor_noop"]
    # Same-FIT donor counts and G are already checked across every new output;
    # the old donor quality remains a literal imported endpoint, never rescored.
    for g in groups:
        donor_panck = endpoints(g["donor_counts"]["panCK"])
        for k in COMMON_METRICS[:5]: _same(donor_panck[k], donor[k], "old229 donor." + k)
    ordinary = ("P_F", "M_F", "S_F", "P_C", "S_C")
    def joint(a, b):
        return a["ODS_F"]-b["ODS_F"] > ATOL and all(a[k]-b[k] >= -ATOL for k in ("avg_ssim", "avg_psnr"))
    per_seed = []
    for seed in SEEDS:
        e = lookup[group_key("E_F", seed)]; smooth = lookup[group_key("S_F", seed)]
        conditions = dict(beats_every_old229_ODS=e["ODS_F"]-strongest_value > ATOL,
                          beats_paired_ordinary_ODS=all(e["ODS_F"]-lookup[group_key(a, seed)]["ODS_F"] > ATOL for a in ordinary),
                          donor_quality_not_lost=all(e[k]-donor[k] >= -ATOL for k in ("avg_ssim", "avg_psnr")),
                          beats_S_F_without_quality_loss=joint(e, smooth))
        per_seed.append(dict(seed=seed, conditions=conditions, joint_GO=all(conditions.values()),
                             direct_token_comparisons={a: joint(e, lookup[group_key(a, seed)]) for a in ("E_C", "E_shift")}))
    go = all(r["joint_GO"] for r in per_seed)
    complete_clipping = all(r["clipping"]["status"] == "measured" for r in rows)
    ordinary_gains = [a for a in ordinary if all(lookup[group_key(a, s)]["ODS_F"]-strongest_value > ATOL
                       and all(lookup[group_key(a, s)][k]-donor[k] >= -ATOL for k in ("avg_ssim", "avg_psnr")) for s in SEEDS)]
    ordinary_explanation = any(not all(lookup[group_key("E_F", s)]["ODS_F"]-
                                      lookup[group_key(a, s)]["ODS_F"] > ATOL for s in SEEDS)
                               for a in ordinary_gains)
    named_keys = list(dict.fromkeys([*required, *strongest_keys,
                                    *(k for k in old_lookup if "cheap" in k and k.endswith("/half"))]))
    return dict(status="complete_saved_scalar_analysis" if complete_clipping else "incomplete_clipping_evidence",
                coverage=dict(image_rows=128, pooled_groups=16, images=8, markers=3, thresholds=256,
                              ordered_new_contrasts=len(contrasts), paired_new_image_differences=len(contrasts)*8,
                              old_reference_groups=229, old_reference_contrasts=len(old_comparisons),
                              old_paired_image_differences=len(old_comparisons)*8),
                saved_scalar_check={k: v for k, v in checked.items() if k != "groups"},
                all128_image_rows=rows, all16_pooled_groups=groups, all240_new_contrasts=contrasts,
                old229_reference=old229, all_old_reference_comparisons=old_comparisons,
                named_old_reference_comparisons=[r for r in old_comparisons if r["reference"] in named_keys],
                strongest_old_panCK_ODS=dict(value=strongest_value, keys_at_exact_max=strongest_keys),
                decision=dict(joint_GO=go and complete_clipping, scalar_joint_conditions_met=go,
                              conditional_direct_token_support=go and complete_clipping and all(all(r["direct_token_comparisons"].values()) for r in per_seed),
                              ordinary_explanation=ordinary_explanation, ordinary_joint_improvement_arms=ordinary_gains,
                              per_seed=per_seed, sufficient_optimization="not_assessed_from_scalars", method_claim=False,
                              final_charter_gate="not_evaluated", sign_tolerance=ATOL, sign_tolerance_is_scientific_margin=False),
                uncertainty="Eight exposed development patches and two new-head seeds; patient independence and confirmation are not established.",
                pixels_read=0, target_reads=0, RGB_rescores=0, new_fits=0, independent_second_truth_measurement=False)
