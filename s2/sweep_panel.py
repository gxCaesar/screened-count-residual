"""Threshold sweep of the tolerance F1: is a pooled-F1 difference structure or intensity calibration?

The primary metric (`metric_panel.f1_pool`) thresholds the prediction at the label's Otsu level
(panCK 22, CD3 30, DAPI 18). A generator whose intensities sit a few levels low loses recall with
the right structure. This script keeps the label positives fixed and sweeps the prediction
threshold t over every uint8 level, with the same 5 px tolerance matching:

  n_p(t)  predicted positives (p > t)
  tp_p(t) of them within 5 px of a label positive        (distance transform of the label mask)
  tp_g(t) label positives within 5 px of a predicted one  (grey dilation of p by the tolerance disk)

The disk is the set of offsets whose cv2 L2/5 chamfer distance is <= 5, so tp_g(t) equals
metric_panel.tol_counts at every t, not only at the fixed threshold; checked exactly in self_test
and against the stored counts of every val image at the fixed threshold.

Per arm and marker, from counts pooled over images:
  fixed   F1 at the label threshold (reproduces f1_pool)
  ods     best F1 over t on val (BSDS ODS convention; optimistic, identical for every arm)
  cal     F1 on val at the t that maximises pooled F1 on --n_train train centre crops (honest)
  ap      area under the pooled tolerant precision-recall curve over t

Run from a DTR checkout:
  python sweep_panel.py --data_root HEMIT --out OUT.json --ckpt released=W [--ckpt NAME=W ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.getcwd())
import metric_panel as MP  # noqa: E402
import s2_scoring  # noqa: E402
from curve_evidence import per_image_payload, selected_train, validate_manifest, validate_per_image  # noqa: E402

TRUE, TOL, THR = MP.TRUE, MP.TOL, MP.FIXED_THR
NL = 256


def tol_kernel(tol=TOL, r=8):
    z = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    z[r, r] = 0
    d = cv2.distanceTransform(z, cv2.DIST_L2, 5)
    return (d <= tol).astype(np.uint8)


KERNEL = tol_kernel()


def sweep_counts(p, gm):
    """p uint8 (H, W), gm bool (H, W) -> (n_p[t], tp_p[t], tp_g[t]) for t = 0..255 (positives p > t), n_g."""
    n_g = int(gm.sum())
    hist_all = np.bincount(p.ravel(), minlength=NL)
    if n_g:
        near = cv2.distanceTransform((~gm).astype(np.uint8), cv2.DIST_L2, 5) <= TOL
        hist_near = np.bincount(p[near], minlength=NL)
        hist_mg = np.bincount(cv2.dilate(p, KERNEL)[gm], minlength=NL)
    else:
        hist_near = np.zeros(NL, np.int64)
        hist_mg = np.zeros(NL, np.int64)
    # count of values strictly above t: reverse cumulative sum shifted by one
    above = lambda h: np.concatenate([np.cumsum(h[::-1])[::-1][1:], [0]])
    return above(hist_all), above(hist_near), above(hist_mg), n_g


def f1_curve(n_p, tp_p, n_g, tp_g):
    n_p, tp_p, tp_g = (np.asarray(x, np.float64) for x in (n_p, tp_p, tp_g))
    prec = np.divide(tp_p, n_p, out=np.zeros_like(n_p), where=n_p > 0)
    rec = tp_g / n_g if n_g else np.zeros_like(tp_g)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    return prec, rec, f1


def ap_from(prec, rec, n_p):
    """Area under the PR curve, thresholds from high to low; only thresholds with predictions count."""
    order = np.arange(NL - 1, -1, -1)
    r_prev, ap = 0.0, 0.0
    for t in order:
        if n_p[t] == 0:
            continue
        ap += (rec[t] - r_prev) * prec[t]
        r_prev = rec[t]
    return float(ap)


def self_test():
    rng = np.random.default_rng(0)
    for trial in range(20):
        h, w = 64 + trial, 80
        g = np.zeros((h, w), np.uint8)
        for _ in range(rng.integers(0, 8)):
            cv2.circle(g, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 6)), 200, -1)
        p = cv2.GaussianBlur(np.roll(g, int(rng.integers(-7, 8)), axis=1), (0, 0), 1.5)
        p = np.clip(p.astype(int) + rng.integers(0, 40, p.shape), 0, 255).astype(np.uint8)
        gm = g > 100
        n_p, tp_p, tp_g, n_g = sweep_counts(p, gm)
        for t in (0, 5, 22, 30, 57, 100, 180, 254, 255):
            ref = MP.tol_counts(p > t, gm)
            got = (int(n_p[t]), int(tp_p[t]) if n_g and n_p[t] else 0, n_g, int(tp_g[t]) if n_g and n_p[t] else 0)
            assert got == ref, (trial, t, got, ref)
    # a perfect copy has AP 1 and F1 1 at the label level; a dimmed copy has F1 0 at it and 1 at its own level
    g = np.zeros((64, 64), np.uint8)
    cv2.circle(g, (30, 30), 6, 200, -1)
    n_p, tp_p, tp_g, n_g = sweep_counts(g, g > 100)
    pr, rc, f1 = f1_curve(n_p, tp_p, n_g, tp_g)
    assert abs(f1[100] - 1) < 1e-12 and abs(ap_from(pr, rc, n_p) - 1) < 1e-12
    dim = (g // 4).astype(np.uint8)  # 50 inside the disk
    n_p, tp_p, tp_g, n_g = sweep_counts(dim, g > 100)
    pr, rc, f1 = f1_curve(n_p, tp_p, n_g, tp_g)
    assert f1[100] == 0 and abs(f1.max() - 1) < 1e-12
    return True


def run(models, items, dev, per_image=None, panel_rows=None):
    """items yields (name, input tensor (3,H,W) in [-1,1], label uint8 (H,W,3)); returns per-arm pooled curves
    and per-image counts at the label threshold for the known-answer check."""
    acc = {n: {m: {"n_p": np.zeros(NL, np.int64), "tp_p": np.zeros(NL, np.int64), "tp_g": np.zeros(NL, np.int64),
                   "n_g": 0} for m in TRUE} for n in models}
    fixed_rows = []
    with torch.no_grad():
        for i, (name, x, y) in enumerate(items):
            row = {"name": name}
            panel_row = {"name": name} if panel_rows is not None else None
            image_counts = {} if per_image is not None else None
            for n, mod in models.items():
                pr = s2_scoring.to_u8(mod(x[None].to(dev))[0])
                if panel_row is not None:
                    for key, value in MP.score(pr, y, THR).items():
                        panel_row[f"{n}__{key}"] = value
                for c, m in enumerate(TRUE):
                    n_p, tp_p, tp_g, n_g = sweep_counts(np.ascontiguousarray(pr[..., c]), y[..., c] > THR[m])
                    a = acc[n][m]
                    a["n_p"] += n_p
                    a["tp_p"] += tp_p
                    a["tp_g"] += tp_g
                    a["n_g"] += n_g
                    if image_counts is not None:
                        image_counts[(n, m)] = (n_p, tp_p, tp_g, n_g)
                    t = int(THR[m])
                    ok = n_g and n_p[t]
                    row[f"{n}__{m}"] = [int(n_p[t]), int(tp_p[t]) if ok else 0, n_g, int(tp_g[t]) if ok else 0]
            fixed_rows.append(row)
            if panel_rows is not None:
                panel_rows.append(panel_row)
            if per_image is not None:
                per_image.append((name, image_counts))
            if i % 100 == 0:
                print(i, flush=True)
    return acc, fixed_rows


def val_items(data_root, names=None):
    from batch_utils import UNI
    val = UNI("hemit", data_root, 512, "val", 0)
    if names is not None:
        pairs = val.dataset.image_names
        by_name = {Path(a).name: (a, b) for a, b in pairs}
        if len(by_name) != len(pairs) or set(by_name) != set(names):
            raise ValueError("val: dataset membership differs from manifest")
        val.dataset.image_names = [by_name[name] for name in names]
    for i in range(len(val)):
        it = val[i]
        yield it["name"], it["input"], s2_scoring.to_u8(it["gt"])


def train_items(data_root, n, names=None):
    tr_in = os.path.join(data_root, "train", "input")
    if names is None:
        names = sorted(os.listdir(tr_in))
        chosen = names[:: max(1, len(names) // n)][:n]
    else:
        chosen = selected_train(names, n)
    for fn in chosen:
        x = Image.open(os.path.join(tr_in, fn)).convert("RGB")
        y = np.asarray(Image.open(os.path.join(data_root, "train", "label", fn)).convert("RGB"))[256:768, 256:768]
        yield fn, ((TF.to_tensor(x) - 0.5) * 2)[:, 256:768, 256:768], np.ascontiguousarray(y)


def summarise_curves(tr, va):
    """Original pooled readout calculation, shared by DTR and adapted baselines."""
    summ = {}
    for n in va:
        s = {}
        for m in TRUE:
            v, t_ = va[n][m], tr[n][m]
            pv, rv, fv = f1_curve(v["n_p"], v["tp_p"], v["n_g"], v["tp_g"])
            _, _, ft = f1_curve(t_["n_p"], t_["tp_p"], t_["n_g"], t_["tp_g"])
            k = int(THR[m])
            t_cal = int(np.argmax(ft))
            t_ods = int(np.argmax(fv))
            s[m] = {"fixed": float(fv[k]), "fixed_P": float(pv[k]), "fixed_R": float(rv[k]),
                    "area_ratio_fixed": float(v["n_p"][k] / v["n_g"]),
                    "ods": float(fv[t_ods]), "t_ods": t_ods, "cal": float(fv[t_cal]), "t_cal": t_cal,
                    "cal_P": float(pv[t_cal]), "cal_R": float(rv[t_cal]), "ap": ap_from(pv, rv, v["n_p"]),
                    "train_fixed": float(ft[k]), "train_best": float(ft[t_cal])}
        for q in ("fixed", "ods", "cal", "ap"):
            s[f"avg_{q}"] = float(np.mean([s[m][q] for m in TRUE]))
        summ[n] = s
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--known", default=None, help="a metric/calib panel json whose 'released' arm holds stored val counts")
    ap.add_argument("--manifest", help="explicit train/val basename manifest, validated before images open")
    ap.add_argument("--per_image_out", help="optional .npz with per-image 256-threshold counts; requires --manifest")
    a = ap.parse_args()
    assert not os.path.exists(a.out)
    if a.per_image_out and not a.manifest:
        ap.error("--per_image_out requires --manifest")
    if a.per_image_out and (not a.per_image_out.endswith(".npz") or os.path.exists(a.per_image_out)):
        ap.error("--per_image_out must be a new .npz path")
    manifest = None
    if a.manifest:
        with open(a.manifest) as handle:
            manifest = json.load(handle)
        validate_manifest(a.data_root, manifest)
    assert self_test() and MP.self_test()
    cv2.setNumThreads(2)
    dev = torch.device("cuda")
    models = {n: MP.load_gen(p, dev) for n, p in (s.split("=", 1) for s in a.ckpt)}

    per_train = [] if a.per_image_out else None
    per_val = [] if a.per_image_out else None
    tr, _ = run(models, train_items(a.data_root, a.n_train,
                                   manifest["train"] if manifest else None), dev, per_train)
    print("train done", flush=True)
    va, rows = run(models, val_items(a.data_root, manifest["val"] if manifest else None), dev, per_val)

    if a.known:
        stored = {r["name"]: r for r in json.load(open(a.known))["rows"]}
        arm = "released"
        assert arm in models and len(stored) == len(rows) == 630
        for r in rows:
            s = stored[r["name"]]
            for m in TRUE:
                ref = [s[f"{arm}__{m}_{q}"] for q in ("np", "tpp", "ng", "tpg")]
                assert r[f"{arm}__{m}"] == ref, (r["name"], m, r[f"{arm}__{m}"], ref)
        print("known-answer: released counts match the stored panel on all 630 val images", flush=True)

    summ = summarise_curves(tr, va)
    curves = {split: {n: {m: {q: (d[n][m][q].tolist() if q != "n_g" else d[n][m][q]) for q in d[n][m]} for m in TRUE}
                      for n in models} for split, d in (("val", va), ("train", tr))}
    result = {"thresholds": THR, "tol_px": TOL, "n_train": a.n_train, "ckpts": a.ckpt, "summary": summ,
              "curves": curves, "fixed_rows": rows}
    if a.per_image_out:
        payload = per_image_payload({"train": per_train, "val": per_val},
                                    list(models), TRUE, manifest, THR, TOL)
        validate_per_image(payload, result, manifest)
        with open(a.per_image_out, "xb") as handle:
            np.savez_compressed(handle, **payload)
    json.dump(result, open(a.out, "w"))
    print(f"{'arm':22s} {'fixed':>7s} {'ods':>7s} {'cal':>7s} {'ap':>7s}   per marker fixed/cal (t_cal)")
    for n, s in summ.items():
        print(f"{n:22s} {s['avg_fixed']:7.4f} {s['avg_ods']:7.4f} {s['avg_cal']:7.4f} {s['avg_ap']:7.4f}   "
              + "  ".join(f"{m} {s[m]['fixed']:.3f}/{s[m]['cal']:.3f} ({s[m]['t_cal']})" for m in TRUE), flush=True)


if __name__ == "__main__":
    main()
