"""Compare arms scored by metric_panel.py under the charter's primary metric (pooled tolerance F1).

Reads one or more panel json files (arms merged by name; rows aligned by image name), prints per-arm
primary metric and guards, the seed-to-seed spread of the control arm, and paired differences
with a bootstrap over images. The bootstrap resamples images and recomputes the pooled counts, so it
measures image sampling only; the training-seed spread is the noise floor.

  python compare_panel.py --panel A.json [--panel B.json ...] [--ctrl none_s0_ema none_s1_ema ...]
                          [--pair ARM_A:ARM_B ...] [--out OUT.json]
"""
import argparse
import json

import numpy as np

TRUE = ["panCK", "CD3", "DAPI"]
GUARD_SSIM, GUARD_PSNR = 0.7108, 26.011  # released DGR on HEMIT val; replaced by the panel's own 'released' arm when present
FIXED_THR = {"panCK": 22.0, "CD3": 30.0, "DAPI": 18.0}
TOL = 5


def f1(n_p, tp_p, n_g, tp_g):
    n_p, tp_p, n_g, tp_g = (np.asarray(v, dtype=np.float64) for v in (n_p, tp_p, n_g, tp_g))
    prec = np.divide(tp_p, n_p, out=np.zeros_like(n_p), where=n_p > 0)
    rec = np.divide(tp_g, n_g, out=np.zeros_like(n_g), where=n_g > 0)
    out = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    return np.where((n_p == 0) & (n_g == 0), 1.0, out)


def validate_panel(d, source="panel"):
    """Validate stored evidence without changing its metric or allowed NaN correlations."""
    def fail(message):
        raise ValueError(f"{source}: {message}")

    if not isinstance(d, dict):
        fail("expected a JSON object")
    if d.get("thresholds") != FIXED_THR or d.get("tol_px") != TOL:
        fail("thresholds or tol_px differ from the registered metric")
    rows, summaries = d.get("rows"), d.get("summary")
    if not isinstance(rows, list) or not rows or not isinstance(summaries, dict) or not summaries:
        fail("rows and summary must be nonempty")
    names = [r.get("name") for r in rows if isinstance(r, dict)]
    if len(names) != len(rows) or any(not isinstance(n, str) or not n for n in names):
        fail("each row requires a nonempty image name")
    if len(set(names)) != len(names):
        fail("duplicate image names")
    for arm, summary in summaries.items():
        if not isinstance(arm, str) or not arm or not isinstance(summary, dict):
            fail("each arm requires a nonempty name and summary object")
        try:
            marker_f1 = []
            for m in TRUE:
                counts = np.asarray([[r[f"{arm}__{m}_{k}"] for k in ("np", "tpp", "ng", "tpg")]
                                     for r in rows], dtype=np.float64)
                if (not np.isfinite(counts).all() or (counts < 0).any()
                        or (counts != np.floor(counts)).any()
                        or (counts[:, 1] > counts[:, 0]).any()
                        or (counts[:, 3] > counts[:, 2]).any()):
                    fail(f"{arm}/{m}: invalid pooled counts")
                if ((counts[:, 2] == 0) & (counts[:, 1] != 0)).any() or ((counts[:, 0] == 0) & (counts[:, 3] != 0)).any():
                    fail(f"{arm}/{m}: matches without positives")
                value = float(f1(*counts.sum(axis=0)))
                marker_f1.append(value)
                if not np.isfinite(summary[f"{m}_f1_pool"]) or abs(value - summary[f"{m}_f1_pool"]) > 1e-9:
                    fail(f"{arm}/{m}: pooled summary disagrees with counts")
                per_f1 = f1(*counts.T)
                stored_f1 = np.asarray([r[f"{arm}__{m}_f1_tol"] for r in rows])
                if not np.isfinite(stored_f1).all() or not np.allclose(per_f1, stored_f1, atol=1e-9, rtol=0):
                    fail(f"{arm}/{m}: per-image F1 disagrees with counts")
                for metric in ("ssim", "psnr", "f1_tol"):
                    values = np.asarray([r[f"{arm}__{m}_{metric}"] for r in rows], dtype=np.float64)
                    target = summary[f"{m}_{metric}"]
                    if not np.isfinite(values).all() or not np.isfinite(target) or abs(float(values.mean()) - target) > 1e-9:
                        fail(f"{arm}/{m}: invalid or inconsistent {metric}")
            for metric in ("ssim", "psnr", "f1_tol"):
                value = float(np.mean([summary[f"{m}_{metric}"] for m in TRUE]))
                if not np.isfinite(summary[f"avg_{metric}"]) or abs(value - summary[f"avg_{metric}"]) > 1e-9:
                    fail(f"{arm}: inconsistent avg_{metric}")
            if not np.isfinite(summary["avg_f1_pool"]) or abs(float(np.mean(marker_f1)) - summary["avg_f1_pool"]) > 1e-9:
                fail(f"{arm}: pooled summary disagrees with counts")
        except (KeyError, TypeError, OverflowError) as exc:
            fail(f"{arm}: missing or malformed metric ({exc})")
    return names


def load(paths):
    """Each entry is PATH, or NAME=PATH for a run's own panel.json (arms 'raw'/'ema' become NAME_raw/NAME_ema)."""
    arms, names = {}, None
    for spec in paths:
        prefix, p = (spec.split("=", 1) if "=" in spec else ("", spec))
        d = json.load(open(p))
        validate_panel(d, p)
        rows = sorted(d["rows"], key=lambda r: r["name"])
        nm = [r["name"] for r in rows]
        if names is None:
            names = nm
        if nm != names:
            raise ValueError(f"{p}: image set differs")
        for arm in d["summary"]:
            out_name = f"{prefix}_{arm}" if prefix else arm
            c = {m: np.array([[r[f"{arm}__{m}_{k}"] for k in ("np", "tpp", "ng", "tpg")] for r in rows], dtype=np.int64)
                 for m in TRUE}
            per = {k: np.array([np.mean([r[f"{arm}__{m}_{k}"] for m in TRUE]) for r in rows])
                   for k in ("ssim", "psnr", "f1_tol")}
            if out_name in arms:
                raise ValueError(f"duplicate arm {out_name}")
            arms[out_name] = {"counts": c, "summary": d["summary"][arm], "per_image": per}
    return arms, names


def pooled(arm, idx=None):
    vals = []
    for m in TRUE:
        c = arm["counts"][m] if idx is None else arm["counts"][m][idx]
        s = c.sum(axis=-2)
        vals.append(f1(*np.moveaxis(s, -1, 0)))
    return np.mean(vals, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--ctrl", nargs="*", default=[])
    ap.add_argument("--pair", nargs="*", default=[])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out")
    a = ap.parse_args()
    arms, names = load(a.panel)

    # known answer: pooled F1 recomputed from the per-image counts equals the panel's own summary
    for n, arm in arms.items():
        assert abs(float(pooled(arm)) - arm["summary"]["avg_f1_pool"]) < 1e-9, n

    gs, gp = GUARD_SSIM, GUARD_PSNR
    if "released" in arms:
        gs, gp = arms["released"]["summary"]["avg_ssim"], arms["released"]["summary"]["avg_psnr"]
    print(f"guards: SSIM >= {gs:.4f}, PSNR >= {gp:.3f}")
    print(f"{'arm':22s}{'F1_pool':>10s}{'F1_img':>10s}{'SSIM':>9s}{'PSNR':>9s}{'R0':>8s}{'areaR':>8s}  guards  "
          + "  ".join(f"{m}_F1" for m in TRUE))
    rep = {"arms": {}}
    for n, arm in arms.items():
        s = arm["summary"]
        g = "ok" if s["avg_ssim"] >= gs and s["avg_psnr"] >= gp else (
            "SSIM" if s["avg_ssim"] < gs and s["avg_psnr"] >= gp else
            "PSNR" if s["avg_ssim"] >= gs else "both")
        print(f"{n:22s}{s['avg_f1_pool']:10.4f}{s['avg_f1_tol']:10.4f}{s['avg_ssim']:9.4f}{s['avg_psnr']:9.2f}"
              f"{s['avg_r0']:8.3f}{s['avg_frac_corr0']:8.3f}  {g:6s}  " + "  ".join(f"{s[m + '_f1_pool']:.3f}" for m in TRUE))
        rep["arms"][n] = {"f1_pool": s["avg_f1_pool"], "ssim": s["avg_ssim"], "psnr": s["avg_psnr"], "guards": g}

    if len(a.ctrl) >= 2:
        v = np.array([arms[c]["summary"]["avg_f1_pool"] for c in a.ctrl])
        floor = {"values": v.tolist(), "range": float(v.max() - v.min()), "sd": float(v.std(ddof=1))}
        rep["noise_floor"] = floor
        print(f"control seeds {a.ctrl}: F1_pool {np.round(v, 4).tolist()}  range {floor['range']:.4f}  sd {floor['sd']:.4f}")

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(names), (a.n_boot, len(names)))
    rep["pairs"] = {}
    for pr in a.pair:
        x, y = pr.split(":")
        d = float(pooled(arms[x]) - pooled(arms[y]))
        bs = pooled(arms[x], idx) - pooled(arms[y], idx)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ds = arms[x]["summary"]["avg_ssim"] - arms[y]["summary"]["avg_ssim"]
        rep["pairs"][pr] = {"d_f1_pool": d, "boot95": [float(lo), float(hi)], "d_ssim": ds}
        print(f"{x} - {y}: F1_pool {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  SSIM {ds:+.4f}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
